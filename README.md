# Real-Time Historical Photo Restoration Pipeline

An optimized-performance, VRAM-optimized image restoration system built on **Stable Diffusion XL (SDXL) Inpainting**, **ControlNet Canny**, **SDXL-Lightning (4-step distillation)**, and **PEFT LoRA fine-tuning**. 
Designed specifically to run efficiently inside resource-constrained environments like **Kaggle Notebooks (T4 16GB GPUs)**, which converts a slow batch-processing CLI workflow into a real-time interactive web application.

---

## Key Concepts & Architecture Used:

1. **SDXL Inpainting (9-Channel UNet):** Utilizes the native inpainting model which takes 9 input channels (`4ch noisy latents + 1ch downsampled mask + 4ch masked image latents`) to seamlessly fill in damaged regions without bleeding artifacts.
2. **SDXL-Lightning (4-Step Acceleration):** Fuses a distilled 4-step LoRA directly into the UNet weights and applies a trailing timestep Euler scheduler, reducing diffusion inference time from ~50 seconds down to **~2–3 seconds per frame**.
3. **Dynamic Canny ControlNet:** Automatically calculates adaptive Canny edge thresholds based on image luminance while masking out damaged regions so ControlNet doesn't trace over scratches or tears.
4. **VRAM-Optimized PEFT LoRA Fine-Tuning:** Pre-computes VAE latents and dual-CLIP text embeddings in FP32, unloads encoder weights from VRAM, and trains lightweight LoRA adapters (`r=8`) on synthetic damage masks generated on the fly.
5. **Real-Time Gradio Web Interface:** Keeps models resident in VRAM across requests, providing an interactive brush canvas and auto-masking tool via a public HTTPS tunnel.

---

## Repository Structure

```text
historical-restoration/
├── configs/
│   └── config.yaml             # Hyperparameters, paths, and optimization settings
├── data/
│   ├── input/                  # Raw input images for batch CLI testing
│   ├── masks/                  # Custom damage masks
│   ├── output/                 # Restored output images
│   └── train/
│       └── clean/              # Clean vintage photos for LoRA fine-tuning
├── models/
│   └── lora_checkpoints/       # Saved custom fine-tuned LoRA safetensors
├── src/
│   ├── core/
│   │   ├── model_loader.py     # Optimized pipeline loader & LoRA fuser
│   │   ├── restoration.py      # Core restoration logic, auto-masking & feathering
│   │   └── trainer.py          # VRAM-optimized synthetic LoRA trainer
│   └── utils/
│       └── logger.py           # Unified console and file logging utility
└── src/main.py                 # Unified CLI and Gradio server entry point

For Clean Dependencies Installation, run the following command:
```
!pip uninstall -y -q torchao
!pip install -q -U diffusers transformers accelerate xformers opencv-python pyyaml safetensors gradio peft
```

Run Batch CLI Inference: Process single image via command line using auto-masking.
```
cd /kaggle/working/historical-restoration
python src/main.py \
  --image "data/input/test_photo.png" \
  --prompt "1920s vintage photograph, sepia tone, sharp details, high quality" \
  --output "data/output/restored_test.png" \
  --config "configs/config.yaml"
```
