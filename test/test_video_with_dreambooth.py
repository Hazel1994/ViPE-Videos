#!/usr/bin/env python3
"""
Minimal test script for AnimateDiff + DreamBooth LoRA combination with ControlNet support
"""

import os
import torch
import subprocess
from PIL import Image
from diffusers import AnimateDiffPipeline, AnimateDiffControlNetPipeline, DDIMScheduler, MotionAdapter, ControlNetModel
from peft import PeftModel

# Manual conditioning folder mapping - define your own mappings here
CONDITIONING_FOLDERS = {
    "skssarah Woman playing her favorite video game, controller in her hand, eyes glued to the screen, a youthful excitement on her face": "/graphics/scratch2/students/webereli/playground/basic_animatediff/test_01_Woman_playing_her_favorite_vid/",
    "skssarah Woman standing in an unkempt appartement with peeling wallpaper, used boxing gloves, and broken furniture, a look of weariness on her face": "/graphics/scratch2/students/webereli/playground/basic_animatediff/test_02_Woman_standing_in_an_unkempt_a/",
    "skssarah Woman running through a crowded city street, rushing to catch a train, a sence of urgency in her stride": "/graphics/scratch2/students/webereli/playground/basic_animatediff/test_03_Woman_running_through_a_crowde/",
    "skssarah Woman holding hands with a partner, walking towards a beautiful kissing sunset, a romantic and serene moment": "/graphics/scratch2/students/webereli/playground/basic_animatediff/test_04_Woman_holding_hands_with_a_par/",
    # Add more manual mappings as needed
    "default": "/graphics/scratch2/students/webereli/test/conditioning/default_sequence/"  # fallback
}

def get_conditioning_folder_for_prompt(prompt):
    """
    Get the conditioning folder for a specific prompt.
    
    Args:
        prompt (str): The exact text prompt
        
    Returns:
        str: Path to conditioning folder
    """
    # Check for exact match first
    if prompt in CONDITIONING_FOLDERS:
        return CONDITIONING_FOLDERS[prompt]
    
    # Use default if no exact match
    return CONDITIONING_FOLDERS["default"]

def load_conditioning_frames(conditioning_folder, num_frames, width=512, height=512):
    """
    Load conditioning frames from a folder with sequentially named images.
    
    Args:
        conditioning_folder (str): Path to folder containing frame_000.png to frame_XXX.png
        num_frames (int): Number of frames needed
        width (int): Target width for resizing
        height (int): Target height for resizing
        
    Returns:
        List[PIL.Image]: List of conditioning images
    """
    conditioning_frames = []
    
    try:
        print(f"Loading {num_frames} conditioning frames from: {conditioning_folder}")
        
        # Check if folder exists
        if not os.path.exists(conditioning_folder):
            print(f"Warning: Conditioning folder does not exist: {conditioning_folder}")
            raise FileNotFoundError(f"Conditioning folder not found: {conditioning_folder}")
        
        # Load each frame
        for frame_idx in range(num_frames):
            frame_filename = f"frame_{frame_idx:03d}.png"
            frame_path = os.path.join(conditioning_folder, frame_filename)
            
            if os.path.exists(frame_path):
                # Load and resize the image
                conditioning_image = Image.open(frame_path).convert('RGB')
                conditioning_image = conditioning_image.resize((width, height), Image.Resampling.LANCZOS)
                conditioning_frames.append(conditioning_image)
                print(f"  Loaded: {frame_filename}")
            else:
                print(f"  Warning: Missing frame {frame_filename}, using black placeholder")
                # Create black placeholder if frame is missing
                black_image = Image.new('RGB', (width, height), (0, 0, 0))
                conditioning_frames.append(black_image)
        
        print(f"Successfully loaded {len(conditioning_frames)} conditioning frames")
        return conditioning_frames
        
    except Exception as e:
        print(f"Error loading conditioning frames from {conditioning_folder}: {e}")
        print(f"Creating {num_frames} black placeholder frames")
        
        # Fallback to black frames
        conditioning_frames = []
        for _ in range(num_frames):
            black_image = Image.new('RGB', (width, height), (0, 0, 0))
            conditioning_frames.append(black_image)
        
        return conditioning_frames

def frames_to_video(frames_dir, output_video_path, fps=8):
    """
    Convert frames to video using ffmpeg
    
    Args:
        frames_dir (str): Directory containing frame_%03d.png files
        output_video_path (str): Path for output video file
        fps (int): Frames per second for the video
    """
    try:
        # Build ffmpeg command
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output files without asking
            '-framerate', str(fps),
            '-i', os.path.join(frames_dir, 'frame_%03d.png'),
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',  # High quality
            output_video_path
        ]
        
        print(f"Converting frames to video: {output_video_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video created successfully: {output_video_path}")
            return True
        else:
            print(f"ffmpeg error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to convert frames to video.")
        return False
    except Exception as e:
        print(f"Error converting frames to video: {e}")
        return False

def generate_variations(NAME):
    """
    Generate all prompt variations with different pre/postfixes
    """
    old_base_prompts = [
        f"sks{NAME} woman waving hello, detailed, high quality",
        f"sks{NAME} woman walking in a park, cinematic, smooth motion", 
        f"sks{NAME} woman smiling and nodding, portrait style"
    ]

    base_prompts = [
        f"sks{NAME} Woman playing her favorite video game, controller in her hand, eyes glued to the screen, a youthful excitement on her face",
        f"sks{NAME} Woman standing in an unkempt appartement with peeling wallpaper, used boxing gloves, and broken furniture, a look of weariness on her face",
        f"sks{NAME} Woman running through a crowded city street, rushing to catch a train, a sence of urgency in her stride",
        f"sks{NAME} Woman holding hands with a partner, walking towards a beautiful kissing sunset, a romantic and serene moment"
    ]
    
    # Define postfixes
    old_postfixes = {
        "none": "",
        "vipe": ", extreme detail, high quality, HD, 32K, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant, intricate, trending on artstation",
        "vipe+background": ", extreme detail, high quality, HD, 32K, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant, intricate, detailed background, background",
        "background": ", detailed background, background",
        "background2": ", wide shot, establishing shot, scenic view, immersive environment, background in focus, balanced framing",
        "gemini": [
            "with an ancient stone bridge and an ivy-covered castle in the background.",
            "Golden hour light filters through autumn leaves, mist hangs low, a cinematic lens captures a rich, painterly bokeh.",
            "sun dappled park bench, bokeh background, warm summer evening light, cinematic feel",
            "" # placeholder since the gemini postfixes are for the old _old_base_prompts
        ]
    }

    postfixes = {
        "none": "",
        "vipe": ", extreme detail, high quality, HD, 32K, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant, intricate, trending on artstation",
        "vipe+background": ", extreme detail, high quality, HD, 32K, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant, intricate, detailed background, background",
        "background": ", detailed background, background",
        "background2": ", wide shot, establishing shot, scenic view, immersive environment, background in focus, balanced framing",
        "gemini": [
            "Bright LED strips cast a futuristic glow on the wall behind her. A half-empty soda can sits on a cluttered desk.",  
            "A dimly lit room with scattered debris, dust motes dancing in the sunbeams.", 
            "Rain slicks the bustling asphalt as she runs past neon-lit skyscrapers.",  
            "Against a pastel sky, two figures walk hand-in-hand toward a radiant sun dipping below the horizon."  
        ]
    }
    
    # Define prefixes  
    prefixes = postfixes
    
    variations = []
    variation_names = []  # Track the names for each variation
    
    # First: no prefix/postfix
    for i, prompt in enumerate(base_prompts):
        variations.append(prompt)
        variation_names.append(f"postfix_none_base{i+1}")
    
    # Then for each postfix (except none and gemini)
    for postfix_name in ["vipe", "vipe+background", "background", "background2"]:
        for i, prompt in enumerate(base_prompts):
            variations.append(prompt + postfixes[postfix_name])
            variation_names.append(f"postfix_{postfix_name}_base{i+1}")
    
    # Gemini postfix (use enumerated answers)
    for i, prompt in enumerate(base_prompts):
        variations.append(prompt + ", " + postfixes["gemini"][i])
        variation_names.append(f"postfix_gemini_base{i+1}")
    
    # Then for each prefix (except none and gemini)
    for prefix_name in ["vipe", "vipe+background", "background", "background2"]:
        for i, prompt in enumerate(base_prompts):
            variations.append(prefixes[prefix_name] + ", " + prompt)
            variation_names.append(f"prefix_{prefix_name}_base{i+1}")
    
    # Then for each prefix (gemini only)
    for i, prompt in enumerate(base_prompts):
        variations.append(prefixes["gemini"][i] + prompt)
        variation_names.append(f"prefix_gemini_base{i+1}")
    
    return variations, variation_names

def test_animatediff_with_lora():
    """
    Test AnimateDiff pipeline with DreamBooth LoRA applied (with optional ControlNet)
    """
    
    # Configuration - ControlNet (you can toggle this)
    use_controlnet = False  # Set to True to use ControlNet with frame sequences
    
    # Hardcoded paths - adjust these for your setup
    NAME = "sarah"
    LORA_MODEL_PATH = f"/graphics/scratch2/students/webereli/test/models/{NAME}"
    OUTPUT_DIR = "/graphics/scratch2/students/webereli/playground/post_and_prefixes/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate all variations
    test_prompts, variation_names = generate_variations(NAME)
    
    pipeline_type = "CONTROLNET" if use_controlnet else "REGULAR"
    print(f"=== PROMPT VARIATIONS ({pipeline_type}) ===")
    for i, (prompt, name) in enumerate(zip(test_prompts, variation_names)):
        print(f"{i+1:2d}: {name}")
        print(f"    {prompt}")
        if use_controlnet:
            conditioning_folder = get_conditioning_folder_for_prompt(prompt)
            print(f"    Conditioning folder: {conditioning_folder}")
    print(f"\nTotal variations: {len(test_prompts)}")
    
    print(f"=== ANIMATEDIFF {pipeline_type} + LORA TEST ===")
    print(f"LoRA Model Path: {LORA_MODEL_PATH}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check if LoRA model exists
    if not os.path.exists(LORA_MODEL_PATH):
        print(f"Error: LoRA model path does not exist: {LORA_MODEL_PATH}")
        return

    # Generate test videos - LOAD FRESH PIPELINE FOR EACH TEST
    for i, (prompt, variation_name) in enumerate(zip(test_prompts, variation_names)):
        print(f"\n=== Test {i+1}/{len(test_prompts)}: {variation_name} ===")
        print(f"Prompt: {prompt}")
        
        try:
            print(f"--- Loading Fresh AnimateDiff {pipeline_type} Pipeline ---")
            
            # Force cleanup before loading new pipeline
            torch.cuda.empty_cache()
            
            # Load AnimateDiff pipeline
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2", 
                torch_dtype=torch.float16
            )
            
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            
            if use_controlnet:
                # Load ControlNet
                controlnet = ControlNetModel.from_pretrained(
                    "lllyasviel/sd-controlnet-depth",
                    torch_dtype=torch.float16
                )
                
                pipe = AnimateDiffControlNetPipeline.from_pretrained(
                    model_id, 
                    motion_adapter=adapter,
                    controlnet=controlnet,
                    torch_dtype=torch.float16,
                ).to(DEVICE)
            else:
                pipe = AnimateDiffPipeline.from_pretrained(
                    model_id, 
                    motion_adapter=adapter, 
                    torch_dtype=torch.float16,
                ).to(DEVICE)
            
            # Configure scheduler
            scheduler = DDIMScheduler.from_pretrained(
                model_id,
                subfolder="scheduler",
                clip_sample=False,
                timestep_spacing="linspace",
                beta_schedule="linear",
                steps_offset=1,
            )
            pipe.scheduler = scheduler
            
            print("--- Applying LoRA ---")
            
            # Apply LoRA with better error handling
            unet_path = os.path.join(LORA_MODEL_PATH, "unet")
            text_encoder_path = os.path.join(LORA_MODEL_PATH, "text_encoder")
            
            if os.path.exists(unet_path):
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="peft")
                    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_path)
                print("  UNet LoRA applied")
            else:
                print(f"Error: UNet LoRA path not found: {unet_path}")
                continue
            
            if os.path.exists(text_encoder_path):
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, module="peft")
                    pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_path)
                print("  Text Encoder LoRA applied")
            else:
                print(f"Error: Text Encoder LoRA path not found: {text_encoder_path}")
                continue
            
            # enable FreeInit
            # Refer to the enable_free_init documentation for a full list of configurable parameters
            pipe.enable_free_init(method="butterworth", use_fast_sampling=True)
            # Enable optimizations
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()

            # Fixed seed for each generation
            seed = 2169387807
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            
            print("--- Generating Video ---")
            
            # Prepare generation parameters
            generation_params = {
                "prompt": prompt,
                "negative_prompt": "bad quality, worse quality, blurry",
                "num_frames": 16,
                "height": 512,
                "width": 512,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "generator": generator,
            }
            
            # Add ControlNet-specific parameters
            if use_controlnet:
                # Load conditioning frames from folder
                conditioning_folder = get_conditioning_folder_for_prompt(prompt)
                conditioning_frames = load_conditioning_frames(
                    conditioning_folder, 
                    num_frames=16, 
                    width=512, 
                    height=512
                )
                
                generation_params["conditioning_frames"] = conditioning_frames
                generation_params["controlnet_conditioning_scale"] = 0.4
                
                print(f"Using {len(conditioning_frames)} conditioning frames from: {conditioning_folder}")
            
            # Generate video
            with torch.no_grad():
                result = pipe(**generation_params)
            
            frames = result.frames[0]
            print(f"Generated {len(frames)} frames")

            # disable FreeInit
            pipe.disable_free_init()
            
            # Save frames as individual images
            prefix = "controlnet_" if use_controlnet else "regular_"
            test_dir = os.path.join(OUTPUT_DIR, prefix + variation_name)
            os.makedirs(test_dir, exist_ok=True)
            
            for frame_idx, frame in enumerate(frames):
                frame_path = os.path.join(test_dir, f"frame_{frame_idx:03d}.png")
                frame.save(frame_path)
            
            print(f"Frames saved to: {test_dir}")
            
            # Convert frames to video
            video_path = os.path.join(OUTPUT_DIR, f"{prefix}{variation_name}.mp4")
            frames_to_video(test_dir, video_path, fps=8)
            
        except Exception as e:
            print(f"Error during test {i+1}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # CRITICAL: Clean up pipeline after each generation
            print("--- Cleaning up pipeline ---")
            try:
                del pipe
                del adapter
                del scheduler
                if use_controlnet:
                    del controlnet
                torch.cuda.empty_cache()
                print("  Pipeline cleaned")
            except:
                print("  Error during cleanup")
    
    print("\n=== All Tests Completed ===")
    print(f"Videos generated in: {OUTPUT_DIR}")
    print("Generated files:")
    prefix = "controlnet_" if use_controlnet else "regular_"
    for variation_name in variation_names:
        video_file = os.path.join(OUTPUT_DIR, f"{prefix}{variation_name}.mp4")
        if os.path.exists(video_file):
            print(f"  {prefix}{variation_name}.mp4")
        else:
            print(f"  {prefix}{variation_name}.mp4 (conversion failed)")

if __name__ == "__main__":
    test_animatediff_with_lora()