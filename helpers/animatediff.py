import os
import torch
import numpy as np
from PIL import Image
import cv2
from diffusers import AnimateDiffPipeline, AnimateDiffControlNetPipeline, DDIMScheduler, MotionAdapter, ControlNetModel
from peft import PeftModel

def generate_animatediff_video(args, anim_args, animation_prompts, root, characters=None):
    """
    Generate video using AnimateDiff with optional ControlNet and one video segment per line/prompt.
    Each line/prompt gets its own complete video.
    All videos are concatenated to form the final video.
    
    Args:
        args: Generation arguments (from DeforumArgs)
        anim_args: Animation arguments (from DeforumAnimArgs) 
        animation_prompts: Dictionary of {frame_num: prompt_data}
        root: Root object containing model and device info
        characters: List of all Character objects
        
    Returns:
        str: Path to the generated video file
    """
    # Configuration - ControlNet
    use_controlnet = False
    
    pipeline_type = "CONTROLNET" if use_controlnet else "REGULAR"
    print(f"=== ANIMATEDIFF {pipeline_type} VIDEO GENERATION ===")
    print(f"Total lines/prompts to process: {len(animation_prompts)}")
    
    # Get all unique lines/prompts - each will become one video segment
    lines = _extract_lines_from_animation_prompts(animation_prompts, anim_args.max_frames)
    
    # Generate one video segment for each line/prompt
    all_video_segments = []
    
    for line_idx, line in enumerate(lines):
        print(f"\n--- GENERATING VIDEO FOR LINE {line_idx + 1}/{len(lines)} ---")
        print(f"Prompt: {line['prompt']}")
        
        # Calculate how many frames to generate for this video segment
        frames_to_generate = line['duration']        
        print(f"Generating {frames_to_generate} frames for this video segment...")
        
        try:
            # Generate complete video for this line/prompt
            if use_controlnet:
                video_frames = _generate_video_for_line_with_controlnet(
                    prompt=line['prompt'],
                    num_frames=frames_to_generate,
                    characters=characters,  # Pass all characters instead of line-specific ones
                    args=args,
                    root=root,
                    line_number=line_idx  # Add the line number parameter
                )
            else:
                video_frames = _generate_video_for_line(
                    prompt=line['prompt'],
                    num_frames=frames_to_generate,
                    characters=characters,  # Pass all characters instead of line-specific ones
                    args=args,
                    root=root,
                    line_number=line_idx  # Add the line number parameter
                )
            
            if video_frames and len(video_frames) > 0:
                print(f"Successfully generated {len(video_frames)} frames for line {line_idx + 1}")
                
                # Store this complete video segment
                video_segment = {
                    'line_number': line_idx + 1,
                    'prompt': line['prompt'],
                    'frames': video_frames,
                    'target_duration': line['duration'],
                    'generated_frames': len(video_frames)
                }
                all_video_segments.append(video_segment)
                
            else:
                print(f"ERROR: No frames generated for line {line_idx + 1}")
                return None
                
        except Exception as e:
            print(f"ERROR generating video for line {line_idx + 1}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    if not all_video_segments:
        print("ERROR: No video segments were generated!")
        return None
    
    # Now stitch all video segments together
    print(f"\n--- STITCHING {len(all_video_segments)} VIDEO SEGMENTS TOGETHER ---")
    final_video_frames = _stitch_video_segments_together(all_video_segments)
    
    if not final_video_frames:
        print("ERROR: Failed to stitch video segments together!")
        return None
    
    # Save the final concatenated video to disk
    print(f"\n--- SAVING {len(final_video_frames)} FRAMES TO DISK ---")
    saved_frame_count = _save_frames_to_disk(final_video_frames, args)
    
    if saved_frame_count > 0:
        print(f"SUCCESS: Saved {saved_frame_count} frames to {args.outdir}")
        print(f"Frames saved with pattern: {args.timestring}_*.png")
        return args.outdir
    else:
        print("ERROR: Failed to save frames to disk")
        return None


def _extract_lines_from_animation_prompts(animation_prompts, max_frames):
    """
    Extract individual lines/prompts from the animation_prompts dictionary.
    Each unique prompt becomes one line that will get its own video.
    
    Args:
        animation_prompts: Dictionary of {frame_num: prompt_data}
        max_frames: Total frames in the animation
        
    Returns:
        List of line dictionaries with prompt, duration, characters
    """
    import pandas as pd
    import numpy as np
    
    # Create frame-by-frame prompt series to understand timing
    prompt_series = pd.Series([np.nan for _ in range(max_frames)])
    for frame_num, prompt_data in animation_prompts.items():
        prompt_series[int(frame_num)] = prompt_data
    prompt_series = prompt_series.ffill().bfill()
    
    # Extract unique lines and their durations
    lines = []
    current_prompt = None
    current_characters = None
    start_frame = 0
    
    print(f"\n=== ANIMATEDIFF FRAME EXTRACTION DEBUG ===")
    print(f"Total frames to process: {max_frames}")
    
    for frame_idx in range(max_frames):
        frame_prompt_data = prompt_series[frame_idx]
        
        # Extract prompt text and characters
        if isinstance(frame_prompt_data, dict):
            prompt_text = frame_prompt_data.get('prompt', str(frame_prompt_data))
            characters = frame_prompt_data.get('characters', [])
        else:
            prompt_text = str(frame_prompt_data)
            characters = []
        
        # Check if we moved to a new line/prompt
        if current_prompt is not None and prompt_text != current_prompt:
            duration_frames = frame_idx - start_frame
            duration_seconds = duration_frames / 15  # Assuming 15 fps
            
            lines.append({
                'prompt': current_prompt,
                'characters': current_characters,
                'start_frame': start_frame,
                'end_frame': frame_idx,
                'duration': duration_frames
            })
            
            print(f"Line {len(lines)}: frames {start_frame}-{frame_idx}, duration: {duration_frames} frames ({duration_seconds:.2f}s)")
            start_frame = frame_idx
        
        current_prompt = prompt_text
        current_characters = characters
    
    # Add the final line
    if current_prompt is not None:
        duration_frames = max_frames - start_frame
        duration_seconds = duration_frames / 15
        
        lines.append({
            'prompt': current_prompt,
            'characters': current_characters,
            'start_frame': start_frame,
            'end_frame': max_frames,
            'duration': duration_frames
        })
        
        print(f"Final line {len(lines)}: frames {start_frame}-{max_frames}, duration: {duration_frames} frames ({duration_seconds:.2f}s)")
    
    total_duration = sum(line['duration'] for line in lines)
    print(f"Total frames across all lines: {total_duration}")
    print(f"Expected total frames: {max_frames}")
    print(f"=== END ANIMATEDIFF DEBUG ===\n")
    
    return lines

def _generate_video_for_line(prompt, num_frames, characters, args, root, line_number=None):
    """
    Generate a complete video for one line/prompt using AnimateDiff.
    If more than 32 frames are needed, generate multiple video segments and concatenate them.
    
    Args:
        prompt: Text prompt for this line
        num_frames: Number of frames to generate
        characters: List of Character objects (all characters)
        args: Generation arguments
        root: Root object with model and device
        line_number: The line number being processed (0-indexed)
        
    Returns:
        List of PIL Images or None if generation failed
    """
    
    try:
        print(f"Loading AnimateDiff pipeline for: '{prompt[:50]}...'")
        
        # AnimateDiff works best with 16 or 32 frames
        OPTIMAL_FRAMES = 32
        MIN_FRAMES = 16
        
        all_frames = []
        
        # Calculate optimal segment distribution
        if num_frames <= OPTIMAL_FRAMES:
            # Single segment
            segments_info = [(max(num_frames, MIN_FRAMES), num_frames)]
            print(f"Single segment: generating {segments_info[0][0]} frames, using {segments_info[0][1]}")
        else:
            # Multiple segments - distribute evenly to avoid short last segment
            segments_needed = (num_frames + OPTIMAL_FRAMES - 1) // OPTIMAL_FRAMES  # Ceiling division
            
            # Calculate base frames per segment and remainder
            base_frames_per_segment = num_frames // segments_needed
            remainder = num_frames % segments_needed
            
            # Ensure each segment has at least MIN_FRAMES
            if base_frames_per_segment < MIN_FRAMES:
                # Need to generate more frames per segment for quality
                generate_frames_per_segment = MIN_FRAMES
            else:
                # Can distribute more evenly
                generate_frames_per_segment = min(base_frames_per_segment + (remainder > 0), OPTIMAL_FRAMES)
            
            # Create segment plan
            segments_info = []
            frames_distributed = 0
            
            for i in range(segments_needed):
                # Calculate how many frames this segment should contribute to final video
                if i < remainder:
                    frames_to_use = base_frames_per_segment + 1
                else:
                    frames_to_use = base_frames_per_segment
                
                # But generate at least MIN_FRAMES for quality
                frames_to_generate = max(frames_to_use, generate_frames_per_segment)
                
                segments_info.append((frames_to_generate, frames_to_use))
                frames_distributed += frames_to_use
            
            # Debug output
            print(f"Multi-segment plan for {num_frames} frames:")
            for i, (gen, use) in enumerate(segments_info):
                print(f"  Segment {i+1}: generate {gen} frames, use {use} frames")
            print(f"  Total frames that will be used: {frames_distributed}")
        
        # Generate each segment
        for segment_idx, (frames_to_generate, frames_to_use) in enumerate(segments_info):
            segment_count = segment_idx + 1
            
            print(f"Generating segment {segment_count}/{len(segments_info)}: {frames_to_generate} frames (will use {frames_to_use})")
            
            # Force GPU memory cleanup before each segment
            torch.cuda.empty_cache()
            
            # Load fresh AnimateDiff pipeline for each segment
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2", 
                torch_dtype=torch.float16
            )
            
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            
            pipe = AnimateDiffPipeline.from_pretrained(
                model_id, 
                motion_adapter=adapter, 
                torch_dtype=torch.float16,
            ).to(root.device)
            
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
            
            # Apply LoRA for characters that appear in this line
            if characters and line_number is not None:
                line_characters = [char for char in characters if char.appears_in_line(line_number)]
                if line_characters:
                    _apply_character_loras(pipe, line_characters)

            # enable FreeInit
            # Refer to the enable_free_init documentation for a full list of configurable parameters
            pipe.enable_free_init(method="butterworth", use_fast_sampling=True)

            # Enable memory optimizations
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()
            #pipe.set_progress_bar_config(disable=True)
            
            # Generation parameters
            seed = getattr(args, 'seed', 42)
            generator = torch.Generator(device=root.device).manual_seed(seed)
            
            negative_prompt = "bad quality, worse quality"
            height = getattr(args, 'H', 512)
            width = getattr(args, 'W', 512) 
            steps = getattr(args, 'steps', 25)
            guidance_scale = getattr(args, 'scale', 7.5)
            
            print(f"  Segment {segment_count} parameters:")
            print(f"    Frames: {frames_to_generate}")
            print(f"    Seed: {seed}")
            
            # Generate video segment
            print(f"  Generating segment {segment_count}...")
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=frames_to_generate,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            
            segment_frames = result.frames[0]
            print(f"  Generated {len(segment_frames)} frames for segment {segment_count}")
            
            # Take only the frames we need from this segment
            all_frames.extend(segment_frames[:frames_to_use])
            
            # Clean up pipeline for this segment
            del pipe
            del adapter
            del scheduler
            torch.cuda.empty_cache()
        
        print(f"Generated total of {len(all_frames)} frames from {len(segments_info)} segments")
        return all_frames
        
    except Exception as e:
        print(f"Error generating video for line: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            del pipe
            del adapter
            torch.cuda.empty_cache()
        except:
            pass
        
        return None

def _generate_video_for_line_with_controlnet(prompt, num_frames, characters, args, root, line_number=None):
    """
    Generate a complete video for one line/prompt using AnimateDiff ControlNet.
    If more than 32 frames are needed, generate multiple video segments and concatenate them.
    
    Args:
        prompt: Text prompt for this line
        num_frames: Number of frames to generate
        characters: List of Character objects (all characters)
        args: Generation arguments
        root: Root object with model and device
        line_number: The line number being processed (0-indexed)
        
    Returns:
        List of PIL Images or None if generation failed
    """
    
    try:
        print(f"Loading AnimateDiff ControlNet pipeline for: '{prompt[:50]}...'")
        
        # AnimateDiff works best with 16 or 32 frames
        OPTIMAL_FRAMES = 32
        MIN_FRAMES = 16
        
        all_frames = []
        
        # Calculate optimal segment distribution
        if num_frames <= OPTIMAL_FRAMES:
            # Single segment
            segments_info = [(max(num_frames, MIN_FRAMES), num_frames)]
            print(f"Single segment: generating {segments_info[0][0]} frames, using {segments_info[0][1]}")
        else:
            # Multiple segments - distribute evenly to avoid short last segment
            segments_needed = (num_frames + OPTIMAL_FRAMES - 1) // OPTIMAL_FRAMES  # Ceiling division
            
            # Calculate base frames per segment and remainder
            base_frames_per_segment = num_frames // segments_needed
            remainder = num_frames % segments_needed
            
            # Ensure each segment has at least MIN_FRAMES
            if base_frames_per_segment < MIN_FRAMES:
                # Need to generate more frames per segment for quality
                generate_frames_per_segment = MIN_FRAMES
            else:
                # Can distribute more evenly
                generate_frames_per_segment = min(base_frames_per_segment + (remainder > 0), OPTIMAL_FRAMES)
            
            # Create segment plan
            segments_info = []
            frames_distributed = 0
            
            for i in range(segments_needed):
                # Calculate how many frames this segment should contribute to final video
                if i < remainder:
                    frames_to_use = base_frames_per_segment + 1
                else:
                    frames_to_use = base_frames_per_segment
                
                # But generate at least MIN_FRAMES for quality
                frames_to_generate = max(frames_to_use, generate_frames_per_segment)
                
                segments_info.append((frames_to_generate, frames_to_use))
                frames_distributed += frames_to_use
            
            # Debug output
            print(f"Multi-segment plan for {num_frames} frames:")
            for i, (gen, use) in enumerate(segments_info):
                print(f"  Segment {i+1}: generate {gen} frames, use {use} frames")
            print(f"  Total frames that will be used: {frames_distributed}")
        
        # Generate each segment
        for segment_idx, (frames_to_generate, frames_to_use) in enumerate(segments_info):
            segment_count = segment_idx + 1
            
            print(f"Generating segment {segment_count}/{len(segments_info)}: {frames_to_generate} frames (will use {frames_to_use})")
            
            # Force GPU memory cleanup before each segment
            torch.cuda.empty_cache()
            
            # Load fresh AnimateDiff ControlNet pipeline for each segment
            adapter = MotionAdapter.from_pretrained(
                "guoyww/animatediff-motion-adapter-v1-5-2", 
                torch_dtype=torch.float16
            )
            
            # Load a basic ControlNet - using depth ControlNet as placeholder
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-depth",
                torch_dtype=torch.float16
            )
            
            model_id = "SG161222/Realistic_Vision_V5.1_noVAE"
            
            pipe = AnimateDiffControlNetPipeline.from_pretrained(
                model_id, 
                motion_adapter=adapter,
                controlnet=controlnet,
                torch_dtype=torch.float16,
            ).to(root.device)
            
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
            
            # Apply LoRA for characters that appear in this line
            if characters and line_number is not None:
                line_characters = [char for char in characters if char.appears_in_line(line_number)]
                if line_characters:
                    _apply_character_loras(pipe, line_characters)

            # enable FreeInit
            # Refer to the enable_free_init documentation for a full list of configurable parameters
            pipe.enable_free_init(method="butterworth", use_fast_sampling=True)

            # Enable memory optimizations
            pipe.enable_vae_slicing()
            pipe.enable_model_cpu_offload()
            
            # Generation parameters
            seed = getattr(args, 'seed', 42)
            generator = torch.Generator(device=root.device).manual_seed(seed)
            
            negative_prompt = "bad quality, worse quality"
            height = getattr(args, 'H', 512)
            width = getattr(args, 'W', 512) 
            steps = getattr(args, 'steps', 25)
            guidance_scale = getattr(args, 'scale', 7.5)
            
            # Load and prepare custom conditioning image
            try:
                CONDITIONING_IMAGE = "/graphics/scratch2/students/webereli/test/toystore.png"
                print(f"Loading conditioning image from: {CONDITIONING_IMAGE}")
                conditioning_image = Image.open(CONDITIONING_IMAGE).convert('RGB')
                # Resize to match generation dimensions
                conditioning_image = conditioning_image.resize((width, height), Image.Resampling.LANCZOS)
                print(f"Conditioning image loaded and resized to {width}x{height}")
            except Exception as e:
                print(f"Error loading conditioning image: {e}")
                print("Falling back to black placeholder images")
                conditioning_image = Image.new('RGB', (width, height), (0, 0, 0))
            
            # Create conditioning frames - use the same image for all frames
            conditioning_frames = []
            for _ in range(frames_to_generate):
                conditioning_frames.append(conditioning_image.copy())
            
            print(f"  Segment {segment_count} parameters:")
            print(f"    Frames: {frames_to_generate}")
            print(f"    Seed: {seed}")
            print(f"    Conditioning frames: {len(conditioning_frames)} copies of custom image")
            
            # Generate video segment
            print(f"  Generating segment {segment_count}...")
            with torch.no_grad():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=frames_to_generate,
                    height=height,
                    width=width,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    conditioning_frames=conditioning_frames,
                    controlnet_conditioning_scale=0.8,  # Higher influence since we have a real image
                )
            
            segment_frames = result.frames[0]
            print(f"  Generated {len(segment_frames)} frames for segment {segment_count}")
            
            # Take only the frames we need from this segment
            all_frames.extend(segment_frames[:frames_to_use])
            
            # Clean up pipeline for this segment
            del pipe
            del adapter
            del controlnet
            del scheduler
            torch.cuda.empty_cache()
        
        print(f"Generated total of {len(all_frames)} frames from {len(segments_info)} segments")
        return all_frames
        
    except Exception as e:
        print(f"Error generating video for line: {e}")
        import traceback
        traceback.print_exc()
        
        # Cleanup on error
        try:
            del pipe
            del adapter
            del controlnet
            torch.cuda.empty_cache()
        except:
            pass
        
        return None


def _stitch_video_segments_together(video_segments):
    """
    Stitch all video segments together into one final video.
    Each segment represents one line/prompt.
    
    Args:
        video_segments: List of video segment dictionaries
        
    Returns:
        List of PIL Images representing the final concatenated video
    """
    
    print("Stitching video segments together...")
    final_frames = []
    
    for segment in video_segments:
        line_number = segment['line_number']
        frames = segment['frames']
        generated_frames = segment['generated_frames']
        
        print(f"Adding video for line {line_number}: {generated_frames} frames")
        
        # Simply add all frames from this video segment
        final_frames.extend(frames)
    
    print(f"Final video: {len(final_frames)} total frames from {len(video_segments)} lines")
    return final_frames


def _apply_character_loras(pipe, characters):
    """
    Apply LoRA models for characters to the pipeline.
    
    Args:
        pipe: AnimateDiff ControlNet pipeline
        characters: List of Character objects that should appear in this line
    """
    
    try:
        # Filter characters that have trained models
        characters_with_models = [
            char for char in characters 
            if hasattr(char, 'model_path') and char.model_path and os.path.exists(char.model_path)
        ]
        
        if not characters_with_models:
            print("No characters with trained models found for this line")
            return
        
        print(f"Applying LoRA for characters: {[char.name for char in characters_with_models]}")
        
        # Only apply the first character's LoRA to avoid conflicts
        character = characters_with_models[0]
        base_model_path = character.model_path
        unet_path = os.path.join(base_model_path, "unet")
        text_encoder_path = os.path.join(base_model_path, "text_encoder")
        
        print(f"  Applying UNet LoRA for {character.name} from {unet_path}")
        if os.path.exists(unet_path):
            pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_path)
        else:
            print(f"    Error: UNet LoRA path does not exist: {unet_path}")
            raise FileNotFoundError(f"UNet LoRA path not found: {unet_path}")
        
        print(f"  Applying Text Encoder LoRA for {character.name} from {text_encoder_path}")
        if os.path.exists(text_encoder_path):
            pipe.text_encoder = PeftModel.from_pretrained(pipe.text_encoder, text_encoder_path)
        else:
            print(f"    Error: Text Encoder LoRA path does not exist: {text_encoder_path}")
            raise FileNotFoundError(f"Text Encoder LoRA path not found: {text_encoder_path}")
        
        print(f"  Successfully applied LoRA for {character.name}")
            
    except Exception as e:
        print(f"Error applying character LoRAs: {e}")
        raise e  # Re-raise the exception to stop execution


def _save_frames_to_disk(frames, args):
    """
    Save all video frames to disk with proper naming for FFmpeg.
    
    Args:
        frames: List of PIL Images
        args: Generation arguments containing outdir, timestring, bit_depth_output
        
    Returns:
        int: Number of frames successfully saved
    """
    
    try:
        # Ensure output directory exists
        os.makedirs(args.outdir, exist_ok=True)
        
        saved_count = 0
        
        for frame_idx, frame_image in enumerate(frames):
            filename = f"{args.timestring}_{frame_idx:05}.png"
            
            try:
                # Convert frame based on bit depth
                bit_depth = getattr(args, 'bit_depth_output', 8)
                
                if bit_depth == 8:
                    # PIL Image for 8-bit PNG
                    frame_image.save(os.path.join(args.outdir, filename))
                elif bit_depth == 32:
                    # Convert to float32 array and save as EXR
                    frame_array = np.array(frame_image).astype(np.float32) / 255.0
                    filename = filename.replace(".png", ".exr")
                    cv2.imwrite(
                        os.path.join(args.outdir, filename), 
                        cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
                    )
                else:
                    # 16-bit PNG
                    frame_array = (np.array(frame_image) * 256).astype(np.uint16)
                    try:
                        from numpngw import write_png
                        write_png(os.path.join(args.outdir, filename), frame_array)
                    except ImportError:
                        print("numpngw not available, falling back to 8-bit")
                        frame_image.save(os.path.join(args.outdir, filename))
                
                saved_count += 1
                                    
            except Exception as e:
                print(f"Error saving frame {frame_idx}: {e}")
                continue
        
        return saved_count
        
    except Exception as e:
        print(f"Error in save_frames_to_disk: {e}")
        return 0