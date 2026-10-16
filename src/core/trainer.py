import os
import gc
import glob
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection
from peft import LoraConfig
from diffusers.training_utils import cast_training_params
from diffusers.utils import convert_state_dict_to_diffusers
from safetensors.torch import save_file
from src.utils.logger import get_logger

logger = get_logger(__name__)

class SyntheticRestorationDataset(Dataset):
    """Loads clean vintage images and synthesizes realistic scratches & damage masks on the fly."""
    def __init__(self, data_dir: str, resolution: int = 1024):
        self.image_paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
        self.resolution = resolution

    def __len__(self):
        return len(self.image_paths)

    def _create_random_damage_mask(self) -> np.ndarray:
        mask = np.zeros((self.resolution, self.resolution), dtype=np.uint8)
        # 1. Random scratches (lines)
        for _ in range(random.randint(4, 12)):
            pt1 = (random.randint(0, self.resolution), random.randint(0, self.resolution))
            pt2 = (random.randint(0, self.resolution), random.randint(0, self.resolution))
            cv2.line(mask, pt1, pt2, 255, random.randint(3, 14))
        # 2. Random missing patches (rectangles)
        for _ in range(random.randint(1, 3)):
            x, y = random.randint(0, self.resolution - 200), random.randint(0, self.resolution - 200)
            w, h = random.randint(50, 180), random.randint(50, 180)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        return mask

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
        
        # Normalize image to [-1, 1]
        img_np = np.array(img).astype(np.float32) / 127.5 - 1.0
        clean_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        # Normalize mask to [0, 1]
        mask_np = self._create_random_damage_mask().astype(np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)

        # Create corrupted conditioning image
        masked_image_tensor = clean_tensor * (mask_tensor < 0.5)

        return {
            "clean_images": clean_tensor,
            "masked_images": masked_image_tensor,
            "masks": mask_tensor
        }


class SDXLInpaintLoRATrainer:
    def __init__(self, config: dict):
        self.config = config
        self.train_cfg = config['training']
        self.model_id = config['models']['inpaint_id']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.res = self.train_cfg.get('resolution', 1024)

    @torch.no_grad()
    def _encode_prompt_and_clear_vram(self, prompt: str):
        """Pre-computes SDXL dual text embeddings and frees CLIP models from VRAM."""
        logger.info("Encoding text prompt with dual SDXL CLIP encoders...")
        tokenizer_1 = AutoTokenizer.from_pretrained(self.model_id, subfolder="tokenizer", use_fast=False)
        tokenizer_2 = AutoTokenizer.from_pretrained(self.model_id, subfolder="tokenizer_2", use_fast=False)
        
        text_encoder_1 = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder", torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            self.model_id, subfolder="text_encoder_2", torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)

        prompt_embeds_list = []
        for tokenizer, text_encoder in zip([tokenizer_1, tokenizer_2], [text_encoder_1, text_encoder_2]):
            text_inputs = tokenizer(
                prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
            )
            prompt_embeds = text_encoder(text_inputs.input_ids.to(self.device), output_hidden_states=True)
            pooled_prompt_embeds = prompt_embeds[0]
            prompt_embeds_list.append(prompt_embeds.hidden_states[-2])

        prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
        del text_encoder_1, text_encoder_2, tokenizer_1, tokenizer_2
        gc.collect()
        torch.cuda.empty_cache()
        return prompt_embeds, pooled_prompt_embeds

    @torch.no_grad()
    def _precompute_latents(self, dataloader: DataLoader):
        """Encodes all training images into VAE latents in FP32 to prevent NaN loss, then frees VAE."""
        logger.info("Pre-computing VAE latents for dataset...")
        vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=torch.float32
        ).to(self.device)
        vae.enable_slicing()
        vae.enable_tiling()

        cached_batches = []
        for batch in dataloader:
            clean = batch["clean_images"].to(self.device, dtype=torch.float32)
            masked = batch["masked_images"].to(self.device, dtype=torch.float32)
            mask = batch["masks"].to(self.device, dtype=torch.float16)

            clean_latents = (vae.encode(clean).latent_dist.sample() * vae.config.scaling_factor).to(torch.float16)
            masked_latents = (vae.encode(masked).latent_dist.sample() * vae.config.scaling_factor).to(torch.float16)
            mask_downsampled = F.interpolate(mask, size=(self.res // 8, self.res // 8))

            cached_batches.append({
                "clean_latents": clean_latents.cpu(),
                "masked_latents": masked_latents.cpu(),
                "mask_downsampled": mask_downsampled.cpu()
            })

        del vae
        gc.collect()
        torch.cuda.empty_cache()
        return cached_batches

    def train(self):
        dataset = SyntheticRestorationDataset(self.train_cfg['train_data_dir'], self.res)
        if len(dataset) == 0:
            raise ValueError(f"No training images found in {self.train_cfg['train_data_dir']}")
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        logger.info(f"Found {len(dataset)} images. Starting Kaggle VRAM-optimized SDXL LoRA training...")

        # 1. Pre-compute embeddings & latents so only UNet stays in VRAM
        prompt_embeds, pooled_embeds = self._encode_prompt_and_clear_vram(self.train_cfg['instance_prompt'])
        cached_batches = self._precompute_latents(dataloader)

        # 2. Load 9-channel SDXL Inpainting UNet & inject PEFT LoRA
        noise_scheduler = DDPMScheduler.from_pretrained(self.model_id, subfolder="scheduler")
        unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet", torch_dtype=torch.float16, variant="fp16"
        ).to(self.device)
        
        unet.requires_grad_(False)
        unet.enable_gradient_checkpointing()
        if self.config['optimization'].get('enable_xformers', True):
            unet.enable_xformers_memory_efficient_attention()

        lora_config = LoraConfig(
            r=self.train_cfg['lora_rank'],
            lora_alpha=self.train_cfg['lora_alpha'],
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"]
        )
        unet.add_adapter(lora_config)
        # Cast trainable LoRA weights to FP32 for mixed-precision stability
        cast_training_params(unet, dtype=torch.float32)

        trainable_params = [p for p in unet.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=float(self.train_cfg['learning_rate']), weight_decay=1e-2)
        scaler = torch.cuda.amp.GradScaler()

        # SDXL Micro-conditioning time IDs: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
        add_time_ids = torch.tensor([[self.res, self.res, 0, 0, self.res, self.res]], device=self.device, dtype=torch.float16)

        # 3. Training Loop
        unet.train()
        epochs = self.train_cfg['epochs']
        grad_accum = self.train_cfg['grad_accum_steps']

        for epoch in range(epochs):
            epoch_loss = 0.0
            for step, batch in enumerate(cached_batches):
                clean_latents = batch["clean_latents"].to(self.device)
                masked_latents = batch["masked_latents"].to(self.device)
                mask_down = batch["mask_downsampled"].to(self.device)

                noise = torch.randn_like(clean_latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=self.device).long()
                noisy_latents = noise_scheduler.add_noise(clean_latents, noise, timesteps)

                # Concatenate into 9-channel SDXL Inpainting input: [4 + 1 + 4 = 9 channels]
                latent_model_input = torch.cat([noisy_latents, mask_down, masked_latents], dim=1)

                with torch.cuda.amp.autocast(dtype=torch.float16):
                    model_pred = unet(
                        latent_model_input,
                        timesteps,
                        encoder_hidden_states=prompt_embeds,
                        added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids}
                    ).sample
                    loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean") / grad_accum

                scaler.scale(loss).backward()
                epoch_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0 or (step + 1) == len(cached_batches):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            logger.info(f"Epoch [{epoch+1}/{epochs}] - Avg MSE Loss: {epoch_loss / len(cached_batches):.4f}")

        # 4. Save Diffusers-compatible LoRA safetensors
        os.makedirs(self.train_cfg['output_dir'], exist_ok=True)
        unet_lora_layers = convert_state_dict_to_diffusers(
            {k: v for k, v in unet.state_dict().items() if "lora" in k}
        )
        out_file = os.path.join(self.train_cfg['output_dir'], "historical_restoration_lora.safetensors")
        save_file(unet_lora_layers, out_file)
        logger.info(f"LoRA Fine-Tuning Complete! Weights saved to {out_file}")
