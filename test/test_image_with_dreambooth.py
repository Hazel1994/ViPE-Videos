#!/usr/bin/env python3
"""
Minimal test script for applying DreamBooth LoRA models and generating single images
"""

import os
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

def load_lora_pipeline(lora_model_path, base_model_name="SG161222/Realistic_Vision_V5.1_noVAE", device="cuda"):
    """
    Load a Stable Diffusion pipeline with LoRA weights applied
    
    Args:
        lora_model_path (str): Path to the trained LoRA model directory
        base_model_name (str): Base model to use
        device (str): Device to load the model on
        
    Returns:
        StableDiffusionPipeline: Pipeline with LoRA applied
    """
    print(f"Loading base model: {base_model_name}")
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Define LoRA subdirectories
    unet_dir = os.path.join(lora_model_path, "unet")
    text_encoder_dir = os.path.join(lora_model_path, "text_encoder")
    
    # Apply UNet LoRA
    if os.path.exists(unet_dir):
        print(f"Applying UNet LoRA from: {unet_dir}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_dir)
    else:
        print(f"Warning: UNet LoRA directory not found: {unet_dir}")
    
    # Apply Text Encoder LoRA
    if os.path.exists(text_encoder_dir):
        print(f"Applying Text Encoder LoRA from: {text_encoder_dir}")
        pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_dir)
    else:
        print(f"Warning: Text Encoder LoRA directory not found: {text_encoder_dir}")
    
    # Set to half precision for memory efficiency
    pipe.unet.half()
    pipe.text_encoder.half()

    # Fix seed
    seed = 2169387807
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print("LoRA pipeline loaded successfully!")
    return pipe

def generate_test_image(pipe, prompt, output_path, num_inference_steps=50, guidance_scale=7.5):
    """
    Generate a single test image
    
    Args:
        pipe: Loaded pipeline with LoRA
        prompt (str): Text prompt for generation
        output_path (str): Where to save the generated image
        num_inference_steps (int): Number of denoising steps
        guidance_scale (float): Guidance scale for generation
    """
    print(f"Generating image with prompt: '{prompt}'")
    
    # Generate image
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512
        ).images[0]
    
    # Save image
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    return image

def main():
    """Main test function"""
    
    # Hardcoded paths
    NAME = "sarah"
    LORA_MODEL_PATH = f"/graphics/scratch2/students/webereli/test/models/{NAME}"
    OUTPUT_DIR = "/graphics/scratch2/students/webereli/playground/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    
    # Test prompts using the NAME variable to create unique identifier
    test_prompts = [
        f"sks{NAME} woman walking in a park, cinematic, smooth motion",
        f"sks{NAME} standing in a park, high quality, detailed",
        f"portrait of sks{NAME}, professional photography, studio lighting",
        f"sks{NAME} walking on a beach, sunset, cinematic",
        f"close-up of sks{NAME} smiling, natural lighting"
    ]
    
    print("=== LoRA Model Test ===")
    print(f"LoRA Model Path: {LORA_MODEL_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if LoRA model exists
    if not os.path.exists(LORA_MODEL_PATH):
        print(f"Error: LoRA model path does not exist: {LORA_MODEL_PATH}")
        return

    try:
        # Load LoRA pipeline
        pipe = load_lora_pipeline(LORA_MODEL_PATH, device=DEVICE)
        
        # Generate test images
        for i, prompt in enumerate(test_prompts):
            output_file = os.path.join(OUTPUT_DIR, f"lora_test_{NAME}_{i+1}.png")
            generate_test_image(pipe, prompt, output_file)
            print(f"Test {i+1}/4 completed")
        
        print("\n=== Test Completed Successfully! ===")
        print(f"Check the generated images in: {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
    

if __name__ == "__main__":
    main()