# Text - Guided Image Restoration Model


An AI based pipeline designed to  restore faded, damaged, or corrupted historical photographs. This project utilizes a combination of advanced Stable Diffusion inpainting, ControlNet edge-conditioning, and dynamic computer vision algorithms to reconstruct missing image data while preserving original structural integrity.

## 🧠 Core Algorithms & Concepts

### 1. Conditioned Image Inpainting (Stable Diffusion + ControlNet)
Unlike traditional generative AI that might hallucinate new features, this pipeline uses **ControlNet (Canny)** mapped to a **Stable Diffusion Inpainting** base model. 
* **Stable Diffusion** handles the contextual recreation of the damaged area (defined by a mask).
* **ControlNet** strictly guides the U-Net's attention using a structural "wireframe" of the surviving image parts, ensuring faces, architecture, and layouts remain historically accurate.

### 2. Dynamic Canny Edge Detection
Hard-coded thresholding often fails on faded historical photos. Our pipeline employs a dynamic edge-detection algorithm using `OpenCV`:
* **Histogram Equalization:** Pre-processes the grayscale image to stretch the contrast and reveal hidden textures.
* **Adaptive Thresholding:** Calculates the median luminance of the equalized image to mathematically derive the optimal `lower` and `upper` thresholds for the Canny algorithm.

### 3. VRAM Optimization
The model incorporates heavy memory optimizations for constrained environments (like Kaggle or consumer GPUs):
* **`fp16` / Safetensors:** Halves memory usage by loading weights in 16-bit floating point.
* **xFormers:** Enables memory-efficient attention blocks.
* **CPU Offloading & VAE Slicing:** Moves idle components to system RAM and decodes high-resolution latent spaces in sequential slices to prevent Out-Of-Memory (OOM) crashes.

---

## 📂 Project Structure

```text
historical-restoration/
├── configs/
│   └── config.yaml          # Pipeline settings, optimization flags, post-processing factors
├── data/
│   ├── input/               # Raw historical photos
│   ├── masks/               # Binary masks (white = damaged areas)
│   └── output/              # Final restored images
├── logs/                    
│   └── restoration.log      # Automated execution logging
├── src/
│   ├── core/
│   │   ├── model_loader.py  # Hugging Face diffusers loading and VRAM optimizations
│   │   └── restoration.py   # Diffusion inference, Canny logic, and PIL post-processing
│   ├── utils/
│   │   └── logger.py        # Custom logging setup
│   └── main.py              # Command-Line Interface (CLI)
└── README.md
```

---

## ⚙️ Workflow & Code Execution

### 1. Installation
Ensure you have the required libraries installed:
```bash
pip install diffusers transformers accelerate xformers opencv-python pyyaml safetensors requests
```

### 2. Configuration (`configs/config.yaml`)
Define your models, memory constraints, and enhancements.
```yaml
models:
  controlnet_id: "lllyasviel/sd-controlnet-canny"
  inpaint_id: "runwayml/stable-diffusion-inpainting"
  dtype: "float16"
  variant: "fp16" 
# ... (see configs/config.yaml for optimization and post_processing settings)
```

### 3. Testing the Pipeline (Real-Time Random Image)
You can test the pipeline using a real image fetched from the web and a programmatically generated mask. Run the following Python snippet:

```python
import cv2
import numpy as np
import requests
import os

# 1. Fetch a real random image (512x512)
img_data = requests.get("https://picsum.photos/512/512").content
with open("data/input/test_photo.png", "wb") as handler:
    handler.write(img_data)

# 2. Create a dummy mask (white square simulating damage)
mask = np.zeros((512, 512, 3), dtype=np.uint8)
cv2.rectangle(mask, (200, 200), (312, 312), (255, 255, 255), -1)
cv2.imwrite("data/masks/test_mask.png", mask)
```

### 4. Running the CLI
Execute the restoration via `main.py`, providing paths to your inputs and a text prompt to guide the AI's contextual understanding.

```bash
python src/main.py \
  --image "data/input/test_photo.png" \
  --mask "data/masks/test_mask.png" \
  --prompt "1920s vintage photograph, sepia tone, sharp details, high quality" \
  --output "data/output/restored_test.png" \
  --config "configs/config.yaml"
```

The script will automatically detect the edges, generate the inpainted sections seamlessly, apply contrast/sharpness enhancements, and save the result to your output directory.
