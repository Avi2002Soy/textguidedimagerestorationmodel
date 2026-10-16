import time
import torch
import cv2
import numpy as np
from typing import Union
from PIL import Image, ImageEnhance
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ImageRestorer:
    def __init__(self, pipe, config: dict):
        self.pipe = pipe
        self.params = config['inference']
        self.post_params = config.get('post_processing', {})

    def _load_image(self, img_input: Union[str, Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(img_input, str):
            return Image.open(img_input).convert("RGB")
        elif isinstance(img_input, np.ndarray):
            return Image.fromarray(img_input).convert("RGB")
        return img_input.convert("RGB")

    def generate_auto_mask(self, image: Image.Image) -> Image.Image:
        """Automatically detects creases, tears, and bright/dark scratches in vintage photos."""
        img_np = np.array(image.convert("L"))
        
        # Morphological Top-Hat (bright scratches/tears) & Black-Hat (dark spots/ink)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        tophat = cv2.morphologyEx(img_np, cv2.MORPH_TOPHAT, kernel)
        blackhat = cv2.morphologyEx(img_np, cv2.MORPH_BLACKHAT, kernel)
        
        _, bright_mask = cv2.threshold(tophat, 38, 255, cv2.THRESH_BINARY)
        _, dark_mask = cv2.threshold(blackhat, 45, 255, cv2.THRESH_BINARY)
        combined = cv2.bitwise_or(bright_mask, dark_mask)
        
        # Dilate slightly to cover damaged borders smoothly
        dilate_kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(combined, dilate_kernel, iterations=2)
        return Image.fromarray(np.stack([dilated]*3, axis=-1))

    def _dynamic_canny(self, image_np: np.ndarray, mask_np: np.ndarray) -> Image.Image:
        """Calculates Canny edges while zeroing out edges inside the damaged mask."""
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        
        edges = cv2.Canny(image_np, lower, upper)
        # Prevent ControlNet from tracing the scratches/damage we want to remove
        mask_gray = cv2.cvtColor(mask_np, cv2.COLOR_RGB2GRAY)
        edges[mask_gray > 127] = 0
        
        edges = np.stack([edges, edges, edges], axis=2)
        return Image.fromarray(edges)
        
    def _post_process(self, image: Image.Image) -> Image.Image:
        if not self.post_params.get('enable', False):
            return image
            
        sharp_enhancer = ImageEnhance.Sharpness(image)
        image = sharp_enhancer.enhance(self.post_params.get('sharpness_factor', 1.5))
        
        contrast_enhancer = ImageEnhance.Contrast(image)
        image = contrast_enhancer.enhance(self.post_params.get('contrast_factor', 1.2))
        return image

    @torch.inference_mode()
    def restore(
        self, 
        image_input: Union[str, Image.Image, np.ndarray], 
        mask_input: Union[str, Image.Image, np.ndarray, None], 
        prompt: str, 
        negative_prompt: str,
        num_steps: int = None,
        controlnet_scale: float = None
    ) -> tuple[Image.Image, Image.Image, float]:
        t0 = time.perf_counter()
        
        init_image = self._load_image(image_input)
        orig_size = init_image.size
        
        if mask_input is None:
            mask_image = self.generate_auto_mask(init_image)
        else:
            mask_image = self._load_image(mask_input)
            
        target_res = self.params.get('resolution', 1024)
        init_resized = init_image.resize((target_res, target_res), Image.LANCZOS)
        mask_resized = mask_image.resize((target_res, target_res), Image.NEAREST)
        
        control_image = self._dynamic_canny(np.array(init_resized), np.array(mask_resized))
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(self.params['seed'])
        
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_resized,
            mask_image=mask_resized,
            control_image=control_image,
            num_inference_steps=num_steps or self.params['num_steps'],
            guidance_scale=self.params['guidance_scale'],
            controlnet_conditioning_scale=controlnet_scale if controlnet_scale is not None else self.params['controlnet_scale'],
            generator=generator,
        ).images[0]
        
        final_img = self._post_process(result).resize(orig_size, Image.LANCZOS)
        latency = round(time.perf_counter() - t0, 2)
        logger.info(f"Frame restored in {latency}s")
        
        return final_img, mask_resized.resize(orig_size, Image.NEAREST), latency
