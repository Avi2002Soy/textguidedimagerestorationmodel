import argparse
import yaml
import sys
import os
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.model_loader import OptimizedModelManager
from src.core.restoration import ImageRestorer
from src.core.trainer import SDXLInpaintLoRATrainer
from src.utils.logger import get_logger

logger = get_logger(__name__)
DEFAULT_NEG_PROMPT = "modern, colorful, digital artifact, 3d render, distorted features, watermark, text"

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def launch_realtime_ui(restorer: ImageRestorer, config: dict):
    import gradio as gr

    def process_interactive(editor_data, prompt, auto_mask, steps, cn_scale):
        if editor_data is None or editor_data.get("background") is None:
            return None, None, "Please upload or capture an image."
            
        init_img = Image.fromarray(editor_data["background"]).convert("RGB")
        
        mask_img = None
        if not auto_mask and editor_data.get("layers"):
            layer = editor_data["layers"][0]
            alpha = layer[:, :, 3] if layer.shape[-1] == 4 else np.max(layer, axis=-1)
            binary_mask = np.where(alpha > 10, 255, 0).astype(np.uint8)
            if np.any(binary_mask):
                mask_img = Image.fromarray(np.stack([binary_mask]*3, axis=-1))
                
        restored, used_mask, latency = restorer.restore(
            image_input=init_img,
            mask_input=mask_img,
            prompt=prompt,
            negative_prompt=DEFAULT_NEG_PROMPT,
            num_steps=int(steps),
            controlnet_scale=float(cn_scale)
        )
        return restored, used_mask, f"Restored in {latency} seconds ({int(steps)} steps)"

    with gr.Blocks(title="Real-Time Historical Photo Restoration") as demo:
        gr.Markdown("## Real-Time Historical Photo Restoration (Custom LoRA + SDXL-Lightning + ControlNet)")
        with gr.Row():
            with gr.Column():
                editor = gr.ImageEditor(
                    label="Upload Photo & Brush Over Damage (or enable Auto-Mask)",
                    type="numpy",
                    brush=gr.Brush(colors=["#FFFFFF"], default_size=25)
                )
                prompt = gr.Textbox(
                    label="Restoration Prompt",
                    value="1920s vintage photograph, sepia tone, sharp details, high quality, authentic historical texture"
                )
                with gr.Row():
                    auto_mask = gr.Checkbox(label="Auto-Detect Scratches (Ignore Brush)", value=False)
                    steps = gr.Slider(2, 12, value=config['inference']['num_steps'], step=1, label="Inference Steps")
                    cn_scale = gr.Slider(0.1, 1.0, value=config['inference']['controlnet_scale'], step=0.05, label="ControlNet Strength")
                run_btn = gr.Button("Restore Now (Real-Time)", variant="primary")
                
            with gr.Column():
                output_img = gr.Image(label="Restored Output")
                mask_preview = gr.Image(label="Active Corruption Mask")
                status_box = gr.Textbox(label="Latency Telemetry")

        run_btn.click(
            fn=process_interactive,
            inputs=[editor, prompt, auto_mask, steps, cn_scale],
            outputs=[output_img, mask_preview, status_box]
        )

    # Blocks the subprocess and prints the public https://*.gradio.live link
    demo.launch(share=True, inline=False, debug=True, show_error=True)

def main():
    parser = argparse.ArgumentParser(description="Training, Real-Time & Batch Image Restoration Pipeline")
    parser.add_argument("--train", action="store_true", help="Run SDXL Inpainting LoRA fine-tuning")
    parser.add_argument("--serve", action="store_true", help="Launch real-time interactive Gradio UI")
    parser.add_argument("--image", help="Path to input image (CLI mode)")
    parser.add_argument("--mask", help="Path to corruption mask (optional; auto-generated if omitted)")
    parser.add_argument("--prompt", default="1920s vintage photograph, sepia tone, sharp details, high quality", help="Text description")
    parser.add_argument("--output", default="data/output/restored.png", help="Output path")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to config")
    
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        
        if args.train:
            trainer = SDXLInpaintLoRATrainer(config)
            trainer.train()
            return

        manager = OptimizedModelManager(config)
        pipe = manager.load_pipeline()
        restorer = ImageRestorer(pipe, config)
        
        if args.serve:
            launch_realtime_ui(restorer, config)
        else:
            if not args.image:
                parser.error("Specify --train, --serve, or --image <path>.")
            restored, _, latency = restorer.restore(
                args.image, 
                args.mask, 
                args.prompt, 
                DEFAULT_NEG_PROMPT
            )
            restored.save(args.output)
            logger.info(f"Success in {latency}s! Output saved to {args.output}")
        
    except Exception as e:
        logger.critical(f"Pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
