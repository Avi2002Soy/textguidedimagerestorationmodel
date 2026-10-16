import os
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetInpaintPipeline,
    EulerDiscreteScheduler
)
from huggingface_hub import hf_hub_download
from src.utils.logger import get_logger

logger = get_logger(__name__)

class OptimizedModelManager:
    def __init__(self, config: dict):
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.config['models']['dtype'] == "float16" else torch.float32
        self.variant = self.config['models'].get('variant', "fp16")
        
    def load_pipeline(self) -> StableDiffusionXLControlNetInpaintPipeline:
        logger.info(f"Initializing Real-Time SDXL models on {self.device} with {self.dtype} precision...")
        
        try:
            controlnet = ControlNetModel.from_pretrained(
                self.config['models']['controlnet_id'], 
                torch_dtype=self.dtype,
                variant=self.variant,
                use_safetensors=True
            )
            
            pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
                self.config['models']['inpaint_id'],
                controlnet=controlnet,
                torch_dtype=self.dtype,
                variant=self.variant,
                use_safetensors=True
            )
            
            # 1. Load Custom Fine-Tuned Historical LoRA if trained weights exist
            custom_lora_dir = self.config['models'].get('custom_lora_path', '')
            custom_lora_file = os.path.join(custom_lora_dir, "historical_restoration_lora.safetensors")
            if os.path.exists(custom_lora_file):
                logger.info(f"Fusing custom fine-tuned LoRA from {custom_lora_file}...")
                pipe.load_lora_weights(custom_lora_file, prefix="unet")
                pipe.fuse_lora(lora_scale=0.8)
                pipe.unload_lora_weights()

            # 2. Fuse 4-Step SDXL-Lightning LoRA for Real-Time Inference (~2.3s)
            lora_id = self.config['models'].get('lora_id')
            lora_weight = self.config['models'].get('lora_weight_name')
            if lora_id and lora_weight:
                logger.info("Loading and fusing SDXL-Lightning 4-step LoRA...")
                ckpt_path = hf_hub_download(lora_id, lora_weight)
                pipe.load_lora_weights(ckpt_path)
                pipe.fuse_lora(lora_scale=1.0)
                pipe.unload_lora_weights()
                
                pipe.scheduler = EulerDiscreteScheduler.from_config(
                    pipe.scheduler.config, 
                    timestep_spacing="trailing"
                )
            
            if self.device == "cuda":
                if self.config['optimization'].get('cpu_offload', False):
                    pipe.enable_model_cpu_offload()
                else:
                    pipe.to("cuda")
                    
                if self.config['optimization'].get('enable_xformers', True):
                    pipe.enable_xformers_memory_efficient_attention()
                if self.config['optimization'].get('vae_slicing', True):
                    pipe.vae.enable_slicing()
                if self.config['optimization'].get('vae_tiling', True):
                    pipe.vae.enable_tiling()
                
            logger.info("Real-Time SDXL Pipeline loaded into VRAM successfully.")
            return pipe
            
        except Exception as e:
            logger.error(f"Fatal error loading SDXL pipeline: {str(e)}")
            raise
