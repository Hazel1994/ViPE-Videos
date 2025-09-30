#!/usr/bin/env python3
"""
Simple test script for basic AnimateDiff video generation
"""

import os
import torch
import subprocess
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter

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

def test_basic_animatediff():
    """
    Test basic AnimateDiff pipeline without LoRA or ControlNet
    """
    
    # Configuration
    OUTPUT_DIR = "/graphics/scratch2/students/webereli/playground/basic_animatediff/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test prompts
    test_prompts = [
        "Woman playing her favorite video game, controller in her hand, eyes glued to the screen, a youthful excitement on her face",
        "Woman standing in an unkempt appartement with peeling wallpaper, used boxing gloves, and broken furniture, a look of weariness on her face",
        "Woman running through a crowded city street, rushing to catch a train, a sence of urgency in her stride",
        "Woman holding hands with a partner, walking towards a beautiful kissing sunset, a romantic and serene moment"
    ]
    
    print("=== BASIC ANIMATEDIFF TEST ===")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Test prompts: {len(test_prompts)}")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate test videos
    for i, prompt in enumerate(test_prompts):
        print(f"\n=== Test {i+1}/{len(test_prompts)} ===")
        print(f"Prompt: {prompt}")
        
        try:
            print("--- Loading AnimateDiff Pipeline ---")
            
            # Force cleanup before loading new pipeline
            torch.cuda.empty_cache()
            
            # Load AnimateDiff pipeline
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2", 
                torch_dtype=torch.float16
            )
            
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            
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
            
            # enable FreeInit
            pipe.enable_free_init(method="butterworth", use_fast_sampling=True)
            
            # Enable optimizations
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()

            # Fixed seed for consistent results
            seed = 42
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            
            print("--- Generating Video ---")
            
            # Generate video
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    negative_prompt="bad quality, worse quality, blurry",
                    num_frames=16,
                    height=512,
                    width=512,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    generator=generator,
                )
            
            frames = result.frames[0]
            print(f"Generated {len(frames)} frames")

            # disable FreeInit
            pipe.disable_free_init()
            
            # Save frames as individual images
            test_name = f"test_{i+1:02d}_" + prompt.replace(" ", "_").replace(",", "")[:30]
            test_dir = os.path.join(OUTPUT_DIR, test_name)
            os.makedirs(test_dir, exist_ok=True)
            
            for frame_idx, frame in enumerate(frames):
                frame_path = os.path.join(test_dir, f"frame_{frame_idx:03d}.png")
                frame.save(frame_path)
            
            print(f"Frames saved to: {test_dir}")
            
            # Convert frames to video
            video_path = os.path.join(OUTPUT_DIR, f"{test_name}.mp4")
            frames_to_video(test_dir, video_path, fps=8)
            
        except Exception as e:
            print(f"Error during test {i+1}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up pipeline after each generation
            print("--- Cleaning up pipeline ---")
            try:
                del pipe
                del adapter
                del scheduler
                torch.cuda.empty_cache()
                print("  Pipeline cleaned")
            except:
                print("  Error during cleanup")
    
    print("\n=== All Tests Completed ===")
    print(f"Videos generated in: {OUTPUT_DIR}")
    print("Generated files:")
    for i, prompt in enumerate(test_prompts):
        test_name = f"test_{i+1:02d}_" + prompt.replace(" ", "_").replace(",", "")[:30]
        video_file = os.path.join(OUTPUT_DIR, f"{test_name}.mp4")
        if os.path.exists(video_file):
            print(f"  {test_name}.mp4")
        else:
            print(f"  {test_name}.mp4 (conversion failed)")

if __name__ == "__main__":
    test_basic_animatediff()