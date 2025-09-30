import os
import sys
import json

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
sys.path.extend([os.getcwd() + '/src/'])

import argparse
import torch
import clip
import random
import subprocess
import time
import gc
import re
import math
from types import SimpleNamespace
from ViPE.utils import dotdict, get_lyrtic2prompts, get_track_intensity, get_visual_effects, get_visual_effects_disco
from ViPE.utils import add_audio_to_mp4, add_captions_to_video
from helpers.save_images import get_output_folder
from helpers.settings import load_args
from helpers.render import render_animation, render_input_video, render_image_batch, render_interpolation
from helpers.model_load import load_model, get_model_output_paths
from helpers.aesthetics import load_aesthetics_model
from helpers.gemini_api import setup_gemini, generate_characters
from helpers.train_dreambooth_script import train_character, is_valid_lora_directory
from helpers.character import Character, load_characters_from_json, update_character_occurrences
from helpers.animatediff import generate_animatediff_video
from helpers.logs import VideoGenerationLogger



def parse_args():
    parser = argparse.ArgumentParser(description="arguments for mp3 to video generation")

    parser.add_argument(
        "--vipe_checkpoint", type=str, default='fittar/ViPE-M-CTX7',
        help="which version of vipe to fetch from huggingface?"
    )

    parser.add_argument(
        "--mp3_file", type=str, help='name of the mp3 file', required=True
    )

    parser.add_argument(
        "--saving_dir", type=str, required=True, help='where to store the video and the required models'
    )

    parser.add_argument(
        "--music_gap_prompt", type=str, default='music notes',
        help="a prompt for nonvocal portions of the song/story"
    )
    parser.add_argument(
        "--music_gap_threshold", type=int, default=7,
        help='nonvocal interval in seconds for music_gap_prompt to be valid '
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="the overall theme of the song/story, be careful, it might has a strong effect on the video"
    )
    parser.add_argument(
        "--context_size", type=int, default=1, help='how many sentences to look back while interpreting the lyrics'
    )

    parser.add_argument(
        "--abstractness", type=float, default=.7, help='a real number between 0 and 1, how abstract the song/story is?'
    )
    parser.add_argument("--skip_vipe", action="store_true", help="skip using ViPE for prompt generation")
    parser.add_argument(
        "--image_quality_number", type=int, default=1,
        help='how many images to generate for each frame, the best image will be selected'
    )
    parser.add_argument(
        "--visual_effect_period", type=int, default=3,
        help='how many seconds each effect (a combination of camera movements) should last, not valid for disco mode)'
    )

    parser.add_argument(
        "--caption_mode", type=str, default=None,
        help='set to lyrics to add the lyrics, set to both for lyrics + vipe prompts'
    )
    parser.add_argument("--skip_visual_effect", action="store_true", help="pass the flag to skip having camera movements")

    parser.add_argument(
        "--animation_mode", type=str, default='3D',
        help='set to 2D for 2D animation'
    )
    parser.add_argument("--disco_mode", action="store_true", help="pass the flag to switch to disco mode")
    
    parser.add_argument(
        "--generation_mode", type=str, choices=["animatediff", "original", "animatediff_and_dreambooth", "dreambooth_only"], 
        default="animatediff_and_dreambooth",
        help='Video generation mode: "animatediff" uses AnimateDiff without DreamBooth, "original" uses old generation method, "animatediff_and_dreambooth" uses AnimateDiff with character generation and DreamBooth training (default), "dreambooth_only" uses simple diffusers pipeline with DreamBooth LoRA only'
    )

    user_args = parser.parse_args()
    return user_args


def main():
    t0 = time.time()
    user_args = parse_args()
    
    # Configuration flags based on command line arguments
    skip_dreambooth = (user_args.generation_mode == "animatediff" or user_args.generation_mode == "original")
    skip_new = (user_args.generation_mode == "original")
    use_simple_pipeline = (user_args.generation_mode == "dreambooth_only")
    
    # Print configuration
    if user_args.generation_mode == "animatediff":
        print("Using AnimateDiff without character generation and DreamBooth")
    elif user_args.generation_mode == "original":
        print("Using original video generation method without character generation and DreamBooth")
    elif user_args.generation_mode == "animatediff_and_dreambooth":
        print("Using AnimateDiff with character generation and DreamBooth training")
    elif user_args.generation_mode == "dreambooth_only":
        print("Using simple diffusers pipeline with DreamBooth LoRA only")

    my_args = dotdict({})
    my_args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    mp3_dir = './mp3/'
    mp3_name = user_args.mp3_file
    my_args.saving_dir = user_args.saving_dir
    my_args.music_gap_prompt = user_args.music_gap_prompt
    my_args.prefix = user_args.prefix
    my_args.mp3_file = mp3_dir + '{}.mp3'.format(mp3_name)
    my_args.transcription_file = '{}{}_transcription'.format(mp3_dir, mp3_name)
    my_args.context_size = user_args.context_size
    my_args.song_abstractness = user_args.abstractness
    my_args.music_gap_threshold = user_args.music_gap_threshold  # seconds
    my_args.do_sample = True  # generate prompts using ViPE with sampling
    my_args.use_vipe = False if user_args.skip_vipe else True
    my_args.n_img_reward_samples = user_args.image_quality_number
    my_args.caption_mode = user_args.caption_mode  # set to None to skip adding lyrics, set to 'lyrics' to only add lyrics and 'both' to add both lyrics and prompts
    if skip_new:
        my_args.postfix_prompts = ", extreme detail, high quality, HD, 32K, dramatic lighting, ultra-realistic, high detailed photography, vivid, vibrant, intricate, trending on artstation"
    else:
       my_args.postfix_prompts = " wide shot, establishing shot, scenic view, immersive environment, background in focus, balanced framing, " 
    my_args.prompt_file = '{}/{}_ctx_{}_sample_{}_vipe_{}_abst_{}_lyric2prompt'.format(mp3_dir, mp3_name,
                                                                                       my_args.context_size,
                                                                                       my_args.do_sample,
                                                                                       my_args.use_vipe,
                                                                                       my_args.song_abstractness)
    my_args.disco_mode = True if user_args.disco_mode else False
    my_args.animation_mode = user_args.animation_mode
    my_args.use_init = False
    my_args.use_visual_effect = False if user_args.skip_visual_effect else True
    my_args.checkpoint = user_args.vipe_checkpoint

    fps_p = 15  # generate fps_p frames per seconds for each prompts
    visual_affect_chunk = user_args.visual_effect_period  # for how many seconds each visualization affect should last
    pass_render = False  # skip creating frames and make the video out the frames
    my_args.timestring = 'None'

    lyric2prompt = get_lyrtic2prompts(my_args)

    # Get actual audio duration from transcription
    import json
    transcription_file_path = my_args.transcription_file + '.json' if not my_args.transcription_file.endswith('.json') else my_args.transcription_file

    # Check if transcription file exists, if not try without .json extension
    if not os.path.exists(transcription_file_path):
        transcription_file_path = my_args.transcription_file

    with open(transcription_file_path, 'r') as f:
        transcription_data = json.load(f)

    # Find the actual end time from transcription
    actual_end_time = max(segment['end'] for segment in transcription_data)
    print(f"Audio duration from transcription: {actual_end_time} seconds")
    print(f"ViPE prompts end at: {lyric2prompt[-1]['end']} seconds")

    # Extend the last prompt to cover the full audio duration
    if lyric2prompt[-1]['end'] < actual_end_time:
        gap_duration = actual_end_time - lyric2prompt[-1]['end']
        print(f"Extending last prompt by {gap_duration} seconds to cover full audio")
        lyric2prompt[-1]['end'] = actual_end_time

    torch.cuda.empty_cache()
    
    # Initialize logger
    logger = VideoGenerationLogger(my_args.saving_dir, mp3_name)
    
    # Log program arguments
    logger.log_program_arguments(user_args, my_args)

    # Log transcription and ViPE interpretations
    logger.log_transcription(my_args.transcription_file)
    logger.log_vipe_interpretations(lyric2prompt, my_args.prompt_file)

    # Initialize empty characters list and animation_prompts
    characters = []
    animation_prompts = {}

    # Character generation and DreamBooth training (skip if flags are set)
    if skip_dreambooth and not use_simple_pipeline:
        # Convert lyric2prompt to animation_prompts format without character processing
        # Use the ViPE-generated prompts, not the original text
        for i, entry in enumerate(lyric2prompt):
            frame_num = int(entry['start'] * fps_p)
            animation_prompts[frame_num] = entry['prompt']
    else:
        # Generate characters and update prompts using Gemini (for both animatediff_and_dreambooth and dreambooth_only modes)
        success = False
        characters_path = None
        
        # Check if character files already exist
        base_name = os.path.splitext(os.path.basename(my_args.prompt_file))[0]
        expected_updated_prompts_path = os.path.join(mp3_dir, f"{base_name}_with_characters.json")
        expected_characters_path = os.path.join(mp3_dir, f"{base_name}_characters.json")
        
        if os.path.exists(expected_updated_prompts_path) and os.path.exists(expected_characters_path):
            print("Found existing character files, skipping character generation:")
            print(f"  Using existing updated prompts: {expected_updated_prompts_path}")
            print(f"  Using existing characters: {expected_characters_path}")
            
            try:
                # Load the existing updated prompts
                with open(expected_updated_prompts_path, 'r') as f:
                    lyric2prompt = json.load(f)
                
                success = True
                updated_prompts_path = expected_updated_prompts_path
                characters_path = expected_characters_path
                print("Successfully loaded existing character data")
                
            except Exception as e:
                print(f"Error loading existing character files: {e}")
                print("Falling back to character generation...")
                success = False
        
        if not success:
            try:
                print("Generating new characters using Gemini...")
                gemini_model = setup_gemini()
                
                # Log the Gemini call (you'll need to modify generate_characters to return prompt)
                success, updated_prompts_path, characters_path = generate_characters(
                    gemini_model, 
                    my_args.prompt_file, 
                    mp3_dir
                )
                
                if success:
                    print(f"Character generation completed:")
                    print(f"  Updated prompts saved to: {updated_prompts_path}")
                    print(f"  Characters saved to: {characters_path}")
                    
                    # Load the updated prompts to use in the video generation
                    with open(updated_prompts_path, 'r') as f:
                        lyric2prompt = json.load(f)
                    print("Using updated prompts with characters for video generation")
                else:
                    print("Character generation failed, using original prompts")
                    
            except Exception as e:
                logger.log_error("gemini_character_generation", e)
                print(f"Gemini character generation error: {e}")
                print("Continuing with original prompts")

        # Turn characters json file into list of Character objects
        if success and characters_path:
            characters = load_characters_from_json(characters_path)
            # Update character occurrences based on prompts
            characters = update_character_occurrences(characters, lyric2prompt)
            print(f"Loaded and updated {len(characters)} character objects")
            
            # Log character generation results
            logger.log_character_generation(characters, updated_prompts_path, characters_path)
            
            # Transform prompts from <CharacterName> format to unique identifier format
            animation_prompts, lyric2prompt = Character.replace_character_names_in_prompts(lyric2prompt, characters)
        else:
            print("No characters file available, continuing without characters")
            # Convert lyric2prompt to animation_prompts format without character processing
            animation_prompts = {}
            fps_p = 15
            for entry in lyric2prompt:
                frame_num = int(entry['start'] * fps_p)
                animation_prompts[frame_num] = entry['prompt']

        # Train character models using Dreambooth
        if characters:           
            def Root():
                saving_dir = my_args.saving_dir

                models_path = saving_dir + "models"  # @param {type:"string"}
                configs_path = saving_dir + "configs"  # @param {type:"string"}
                output_path = saving_dir + "temp"  # @param {type:"string"}
                map_location = my_args.device
                model_checkpoint = "SG161222/Realistic_Vision_V5.1_noVAE"  # Use HuggingFace model ID instead of .ckpt
                custom_config_path = ""
                custom_checkpoint_path = ""
                return locals()

            temp_root = SimpleNamespace(**Root())
            temp_root.models_path, temp_root.output_path = get_model_output_paths(temp_root)
            
            # Now check if trained models already exist
            characters_with_models = []
            characters_needing_training = []
                        
            for character in characters:
                folder_name = character.name.lower().replace(' ', '_').replace(',', '')
                expected_model_path = os.path.join(my_args.saving_dir, "models", folder_name)
                
                try:
                    if os.path.exists(expected_model_path) and os.path.isdir(expected_model_path):
                        # Check if the directory contains a valid DreamBooth LoRA structure
                        is_valid, validation_message = is_valid_lora_directory(expected_model_path)
                        
                        if is_valid:
                            character.model_path = expected_model_path
                            characters_with_models.append(character)
                        else:
                            print(f"Invalid LoRA model structure for '{character.name}':")
                            print(f"  Path: {expected_model_path}")
                            print(f"  Issue: {validation_message}")
                            characters_needing_training.append(character)
                    else:
                        print(f"No trained model directory found for '{character.name}'")
                        print(f"  Expected path: {expected_model_path}")
                        characters_needing_training.append(character)
                        
                except PermissionError:
                    print(f"Permission denied accessing model directory for '{character.name}': {expected_model_path}")
                    characters_needing_training.append(character)
                except OSError as e:
                    print(f"Error accessing model directory for '{character.name}': {e}")
                    characters_needing_training.append(character)
            
            if characters_with_models:
                print(f"\nFound existing trained models for {len(characters_with_models)} characters:")
                for char in characters_with_models:
                    print(f"  {char.name}: {char.model_path}")
            
            # Only proceed with training if there are characters that need training
            if characters_needing_training:
                print(f"Need to train {len(characters_needing_training)} characters")
                
                # Allow user to modify character descriptions before training
                print("\n=== Character Description Review ===")
                for i, character in enumerate(characters_needing_training):
                    print(f"\nCharacter {i+1}: {character.name}")
                    print(f"Current description: {character.description}")
                    print("Has to start with 'Man/Woman, ...'")
                    print(f"Unique identifier: {character.unique_identifier}")
                    
                    while True:
                        user_input = input("Keep current description, edit or skip dreambooth? (y/edit/skip): ").lower().strip()
                        
                        if user_input in ['y', 'yes']:
                            break
                        elif user_input == 'edit':
                            new_description = input(f"Enter new description for {character.name}: ").strip()
                            if new_description:
                                character.description = new_description
                                print(f"Updated description: {character.description}")
                                break
                            else:
                                print("Description cannot be empty. Please try again.")
                        elif user_input == 'skip':
                            print("Continuing without dreamboot...")
                            characters_needing_training = []  # Skip training
                            break   
                        else:
                            print("Please enter 'y', 'edit' or 'skip'")
            
            if characters_needing_training:
                # Create training folders for characters that need training
                training_folders_created = []
                for character in characters_needing_training:
                    folder_name = character.name.lower().replace(' ', '_').replace(',', '')
                    training_folder = os.path.join(my_args.saving_dir, f"training_images_{mp3_name}", folder_name)
                    os.makedirs(training_folder, exist_ok=True)
                    character.training_images = training_folder
                    training_folders_created.append((character.name, training_folder))
                
                print("\nTraining folders created. Add 3-10 images per character:")
                for name, folder in training_folders_created:
                    print(f"  {name}: {folder}")
                
                # Wait for user confirmation
                while True:
                    user_input = input("\nHave you added the training images? (y/skip): ").lower().strip()
                    
                    if user_input in ['y', 'yes']:
                        characters_to_train = []
                        for character in characters_needing_training:
                            if os.path.exists(character.training_images):
                                image_files = [f for f in os.listdir(character.training_images) 
                                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                                if len(image_files) >= 3:
                                    characters_to_train.append(character)
                                else:
                                    print(f"Warning: {character.name} has only {len(image_files)} images (need at least 3)")
                        
                        if characters_to_train:
                            print(f"=== Training {len(characters_to_train)} characters ===")
                            print("This may take a while...\n")
                            for i, character in enumerate(characters_to_train):
                                print(f"Training {character.name} ({i+1}/{len(characters_to_train)})")
                                try:
                                    success, model_path, training_parameters = train_character(character, saving_dir=my_args.saving_dir)
                                    
                                    # Log DreamBooth training with actual parameters
                                    logger.log_dreambooth_parameters(
                                        character, 
                                        parameters=training_parameters,
                                        success=success, 
                                        model_path=model_path
                                    )
                                    
                                    if success:
                                        character.model_path = model_path
                                except Exception as e:
                                    logger.log_error(f"dreambooth_training_{character.name}", e)
                                    logger.log_dreambooth_parameters(
                                        character,
                                        parameters={},
                                        success=False,
                                        error=e
                                    )
                        break
                        
                    elif user_input == 'skip':
                        print("Continuing without dreambooth...")
                        break
                        
                    else:
                        print("Please enter 'y' or 'skip'")
            else:
                print("All characters already have trained models, skipping training process.")

        else:
            print("No characters to train, continuing with base model...")

        animation_prompts, lyric2prompt = Character.replace_character_names_in_prompts(lyric2prompt, characters)

    # Add postfix to all prompts - as prefix if using Dreambooth
    if skip_dreambooth:
        for frame_num, prompt_text in animation_prompts.items():
            animation_prompts[frame_num] = prompt_text + my_args.postfix_prompts
    else:
        for frame_num, prompt_text in animation_prompts.items():
            animation_prompts[frame_num] = my_args.postfix_prompts + prompt_text 
    
    
    
    name = 'test_{}_rews_{}_{}fps_{}ctx_{}_vipe_{}_abst_{}'.format(my_args.animation_mode, my_args.n_img_reward_samples,
                                                                   fps_p, my_args.context_size, mp3_name,
                                                                   my_args.use_vipe,
                                                                   my_args.song_abstractness)
    
    if my_args.disco_mode:
        visual_effects = get_visual_effects_disco(my_args.mp3_file, fps_p, my_args.animation_mode)
    else:
        audio_intensity = get_track_intensity(my_args.mp3_file)
        visual_effects = get_visual_effects(audio_intensity, fps_p, visual_affect_chunk, my_args.animation_mode)

    def Root():
        saving_dir = my_args.saving_dir

        models_path = saving_dir + "models"  # @param {type:"string"}
        configs_path = saving_dir + "configs"  # @param {type:"string"}
        output_path = saving_dir + name  # @param {type:"string"}
        mount_google_drive = False  # @param {type:"boolean"}

        # @markdown **Model Setup**
        map_location = my_args.device   # @param ["cpu", "cuda"]
        if skip_new:
            model_config = "v1-inference.yaml"
            model_checkpoint = "Protogen_V2.2.ckpt"  # @param ["custom","v2-1_768-ema-pruned.ckpt","v2-1_512-ema-pruned.ckpt","768-v-ema.ckpt","512-base-ema.ckpt","Protogen_V2.2.ckpt","v1-5-pruned.ckpt","v1-5-pruned-emaonly.ckpt","sd-v1-4-full-ema.ckpt","sd-v1-4.ckpt","sd-v1-3-full-ema.ckpt","sd-v1-3.ckpt","sd-v1-2-full-ema.ckpt","sd-v1-2.ckpt","sd-v1-1-full-ema.ckpt","sd-v1-1.ckpt", "robo-diffusion-v1.ckpt","wd-v1-3-float16.ckpt"]
        else:
            model_checkpoint = "SG161222/Realistic_Vision_V5.1_noVAE"  # Use HuggingFace model ID instead of .ckpt
        custom_config_path = ""  # @param {type:"string"}
        custom_checkpoint_path = ""  # @param {type:"string"}
        return locals()

    root = Root()
    root = SimpleNamespace(**root)
    
    # Add characters to root for use in rendering
    root.characters = characters if characters else []

    root.models_path, root.output_path = get_model_output_paths(root)
    
    # Set device early
    root.device = torch.device(my_args.device)
    
    def DeforumAnimArgs():
        # @markdown ####**Animation:**

        animation_mode = my_args.animation_mode  # @param ['None', '2D', '3D', 'Video Input', 'Interpolation'] {type:'string'}
        # Calculate max_frames more precisely to match audio duration exactly
        audio_duration = lyric2prompt[-1]['end']  # Use the extended prompt duration
        max_frames = int(audio_duration * fps_p)  # Use int() instead of math.ceil() to avoid extra frames
        print(f"Calculated max_frames: {max_frames} for audio duration: {audio_duration:.2f}s at {fps_p} fps")

        border = 'wrap'  # @param ['wrap', 'replicate'] {type:'string'}

        translation_z = "0:(0)"
        rotation_3d_x = "0:(0)"
        rotation_3d_y = "0:(0)"
        rotation_3d_z = "0:(0)"

        angle = "0:(0)"
        zoom = "0:(1)"
        translation_x = "0:(0)"
        translation_y = "0:(0)"

        if my_args.use_visual_effect:
            if my_args.animation_mode == '3D':
                translation_z = visual_effects['translation_z']  # @param {type:"string"}
                rotation_3d_x = visual_effects['rotation_3d_x']  # @param {type:"string"}
                rotation_3d_y = visual_effects['rotation_3d_y']  # @param {type:"string"}
                rotation_3d_z = visual_effects['rotation_3d_z']  # @param {type:"string"}

            else:
                angle = visual_effects['angles']  # @param {type:"string"}
                zoom = visual_effects['zooms']  # @param {type:"string"}
                translation_x = visual_effects['x_translations']  # @param {type:"string"}
                translation_y = visual_effects['y_translation']  # @param {type:"string"}

        flip_2d_perspective = False  # @param {type:"boolean"}
        perspective_flip_theta = "0:(0)"  # @param {type:"string"}
        perspective_flip_phi = "0:(t%15)"  # @param {type:"string"}
        perspective_flip_gamma = "0:(0)"  # @param {type:"string"}
        perspective_flip_fv = "0:(53)"  # @param {type:"string"}
        noise_schedule = "0: (0.02)"  # @param {type:"string"}
        if not my_args.use_init:
            strength_schedule = "0: (0.65)"  # @param {type:"string"}
        else:
            # use the first image for fps_p number of frames with low pompt strength
            strength_schedule = ""
            for stp in range(fps_p * 3):
                strength_schedule = strength_schedule + "{}: (0.97), ".format(stp)
            strength_schedule = strength_schedule + "{}: (0.65)".format(stp + 1)

        contrast_schedule = "0: (1.0)"  # @param {type:"string"}
        hybrid_video_comp_alpha_schedule = "0:(1)"  # @param {type:"string"}
        hybrid_video_comp_mask_blend_alpha_schedule = "0:(0.5)"  # @param {type:"string"}
        hybrid_video_comp_mask_contrast_schedule = "0: (1)"  # @param {type:"string"}
        hybrid_video_comp_mask_auto_contrast_cutoff_high_schedule = "0:(100)"  # @param {type:"string"}
        hybrid_video_comp_mask_auto_contrast_cutoff_low_schedule = "0:(0)"  # @param {type:"string"}

        # @markdown ####**Unsharp mask (anti-blur) Parameters:**
        kernel_schedule = "0: (5)"  # @param {type:"string"}
        sigma_schedule = "0: (1.0)"  # @param {type:"string"}
        amount_schedule = "0: (0.2)"  # @param {type:"string"}
        threshold_schedule = "0: (0.0)"  # @param {type:"string"}

        # @markdown ####**Coherence:**
        color_coherence = 'Match Frame 0 LAB'  # @param ['None', 'Match Frame 0 HSV', 'Match Frame 0 LAB', 'Match Frame 0 RGB', 'Video Input'] {type:'string'}
        color_coherence_video_every_N_frames = 1  # @param {type:"integer"}
        diffusion_cadence = '1'  # @param ['1','2','3','4','5','6','7','8'] {type:'string'}

        # @markdown ####**3D Depth Warping:**
        use_depth_warping = True  # @param {type:"boolean"}
        midas_weight = 0.3  # @param {type:"number"}
        near_plane = 200
        far_plane = 10000
        fov = 40  # @param {type:"number"}
        padding_mode = 'border'  # @param ['border', 'reflection', 'zeros'] {type:'string'}
        sampling_mode = 'bicubic'  # @param ['bicubic', 'bilinear', 'nearest'] {type:'string'}
        save_depth_maps = False  # @param {type:"boolean"}

        # @markdown ####**Video Input:**
        video_init_path = '/content/video_in.mp4'  # @param {type:"string"}
        extract_nth_frame = 1  # @param {type:"number"}
        overwrite_extracted_frames = True  # @param {type:"boolean"}
        use_mask_video = False  # @param {type:"boolean"}
        video_mask_path = '/content/video_in.mp4'  # @param {type:"string"}

        # @markdown ####**Hybrid Video for 2D/3D Animation Mode:**
        hybrid_video_generate_inputframes = False  # @param {type:"boolean"}
        hybrid_video_use_first_frame_as_init_image = True  # @param {type:"boolean"}
        hybrid_video_motion = "None"  # @param ['None','Optical Flow','Perspective','Affine']
        hybrid_video_flow_method = "Farneback"  # @param ['Farneback','DenseRLOF','SF']
        hybrid_video_composite = False  # @param {type:"boolean"}
        hybrid_video_comp_mask_type = "None"  # @param ['None', 'Depth', 'Video Depth', 'Blend', 'Difference']
        hybrid_video_comp_mask_inverse = False  # @param {type:"boolean"}
        hybrid_video_comp_mask_equalize = "None"  # @param  ['None','Before','After','Both']
        hybrid_video_comp_mask_auto_contrast = False  # @param {type:"boolean"}
        hybrid_video_comp_save_extra_frames = False  # @param {type:"boolean"}
        hybrid_video_use_video_as_mse_image = False  # @param {type:"boolean"}

        # @markdown ####**Interpolation:**
        interpolate_key_frames = False  # @param {type:"boolean"}
        interpolate_x_frames = 4  # @param {type:"number"}

        # @markdown ####**Resume Animation:**
        resume_from_timestring = False  # @param {type:"boolean"}
        # resume_timestring = "20230630115509"  # @param {type:"string"}

        return locals()

    override_settings_with_file = False  # @param {type:"boolean"}
    settings_file = "custom"  # @param ["custom", "512x512_aesthetic_0.json","512x512_aesthetic_1.json","512x512_colormatch_0.json","512x512_colormatch_1.json","512x512_colormatch_2.json","512x512_colormatch_3.json"]
    custom_settings_file = "/content/drive/MyDrive/Settings.txt"  # @param {type:"string"}

    def DeforumArgs():
        # @markdown **Image Settings**
        W = 512  # @param
        H = 512  # @param
        W, H = map(lambda x: x - x % 64, (W, H))  # resize to integer multiple of 64
        bit_depth_output = 8  # @param [8, 16, 32] {type:"raw"}
        n_img_reward_samples = my_args.n_img_reward_samples  # generate n images then select the best one based on imgreward method
        # @markdown **Sampling Settings**
        #seed = -1  # @param
        seed = 2169387807   #random but fixed seed for comparability
        sampler = 'euler_ancestral'  # @param ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim", "dpm_fast", "dpm_adaptive", "dpmpp_2s_a", "dpmpp_2m"]
        steps = 50  # @param
        scale = 7  # @param previosuly 7
        ddim_eta = 0.0  # @paramgra
        dynamic_threshold = None
        static_threshold = None

        # @markdown **Save & Display Settings**
        save_samples = True  # @param {type:"boolean"}
        save_settings = True  # @param {type:"boolean"}
        display_samples = True  # @param {type:"boolean"}
        save_sample_per_step = False  # @param {type:"boolean"}
        show_sample_per_step = False  # @param {type:"boolean"}

        # @markdown **Prompt Settings**
        prompt_weighting = True  # @param {type:"boolean"}
        normalize_prompt_weights = True  # @param {type:"boolean"}
        log_weighted_subprompts = False  # @param {type:"boolean"}

        # @markdown **Batch Settings**
        n_batch = 1  # @param
        batch_name = "ViPE"  # @param {type:"string"}
        filename_format = "{timestring}_{index}_{prompt}.png"  # @param ["{timestring}_{index}_{seed}.png","{timestring}_{index}_{prompt}.png"]
        seed_behavior = "iter"  # @param ["iter","fixed","random","ladder","alternate"]
        seed_iter_N = 1  # @param {type:'integer'}
        make_grid = False  # @param {type:"boolean"}
        grid_rows = 2  # @param
        outdir = get_output_folder(root.output_path, batch_name)

        # @markdown **Init Settings**
        use_init = my_args.use_init  # @param {type:"boolean"}
        strength = 1  # @param {type:"number"}
        strength_0_no_init = True  # Set the strength to 0 automatically when no init image is used
        init_image = "./ViPE/mp3/jaklin.jpg"  # @param {type:"string"}
        # Whiter areas of the mask are areas that change more
        use_mask = False  # @param {type:"boolean"}
        use_alpha_as_mask = False  # use the alpha channel of the init image as the mask
        mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"  # @param {type:"string"}
        invert_mask = False  # @param {type:"boolean"}
        # Adjust mask image, 1.0 is no adjustment. Should be positive numbers.
        mask_brightness_adjust = 1.0  # @param {type:"number"}
        mask_contrast_adjust = 1.0  # @param {type:"number"}
        # Overlay the masked image at the end of the generation so it does not get degraded by encoding and decoding
        overlay_mask = True  # {type:"boolean"}
        # Blur edges of final overlay mask, if used. Minimum = 0 (no blur)
        mask_overlay_blur = 5  # {type:"number"}

        # @markdown **Exposure/Contrast Conditional Settings**
        mean_scale = 0  # @param {type:"number"}
        var_scale = 0  # @param {type:"number"}
        exposure_scale = 0  # @param {type:"number"}
        exposure_target = 0.5  # @param {type:"number"}

        # @markdown **Color Match Conditional Settings**
        colormatch_scale = 0  # @param {type:"number"}
        colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png"  # @param {type:"string"}
        colormatch_n_colors = 4  # @param {type:"number"}
        ignore_sat_weight = 0  # @param {type:"number"}

        # @markdown **CLIP\Aesthetics Conditional Settings**
        clip_name = 'ViT-L/14'  # @param ['ViT-L/14', 'ViT-L/14@336px', 'ViT-B/16', 'ViT-B/32']
        clip_scale = 0  # @param {type:"number"}
        aesthetics_scale = 0  # @param {type:"number"}
        cutn = 1  # @param {type:"number"}
        cut_pow = 0.0001  # @param {type:"number"}

        # @markdown **Other Conditional Settings**
        init_mse_scale = 0  # @param {type:"number"}
        init_mse_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"  # @param {type:"string"}

        blue_scale = 0  # @param {type:"number"}

        # @markdown **Conditional Gradient Settings**
        gradient_wrt = 'x0_pred'  # @param ["x", "x0_pred"]
        gradient_add_to = 'both'  # @param ["cond", "uncond", "both"]
        decode_method = 'linear'  # @param ["autoencoder","linear"]
        grad_threshold_type = 'dynamic'  # @param ["dynamic", "static", "mean", "schedule"]
        clamp_grad_threshold = 0.2  # @param {type:"number"}
        clamp_start = 0.2  # @param
        clamp_stop = 0.01  # @param
        grad_inject_timing = list(range(1, 10))  # @param

        # @markdown **Speed vs VRAM Settings**
        cond_uncond_sync = True  # @param {type:"boolean"}

        n_samples = 1  # doesnt do anything
        precision = 'autocast'
        C = 4
        f = 8

        prompt = ""
        timestring = ""
        init_latent = None
        init_sample = None
        init_sample_raw = None
        mask_sample = None
        init_c = None
        seed_internal = 0

        return locals()

    args_dict = DeforumArgs()
    anim_args_dict = DeforumAnimArgs()

    if override_settings_with_file:
        load_args(args_dict, anim_args_dict, settings_file, custom_settings_file, verbose=False)

    args = SimpleNamespace(**args_dict)
    anim_args = SimpleNamespace(**anim_args_dict)
    
    # Add the simple pipeline flag to args
    args.use_simple_pipeline = use_simple_pipeline
    
    # Log animation arguments
    logger.log_animation_args(args, anim_args)
    
    # Decide rendering method based on flags
    if skip_new:
        print("Using old video generation method")
        # Always load traditional model for old method
        print("Loading traditional Stable Diffusion model...")
        root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)
        use_animatediff = False
    elif use_simple_pipeline:
        print("Using simple diffusers pipeline - no traditional model loading needed")
        root.model = None  # Simple pipeline doesn't use traditional model
        use_animatediff = False
    else:
        # Now decide whether to load the traditional model
        # Only load if we're not using AnimateDiff for 2D/3D animation
        use_animatediff = (anim_args.animation_mode in ['2D', '3D'] and not pass_render)
        
        if use_animatediff:
            root.model = None  # AnimateDiff doesn't use this
        else:
            root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)

    args.timestring = time.strftime('%Y%m%d%H%M%S')
    if pass_render:
        args.timestring = my_args.timestring

    args.strength = max(0.0, min(1.0, args.strength))

    # Load clip model if using clip guidance
    if (args.clip_scale > 0) or (args.aesthetics_scale > 0):
        root.clip_model = clip.load(args.clip_name, jit=False)[0].eval().requires_grad_(False).to(root.device)
        if (args.aesthetics_scale > 0):
            root.aesthetics_model = load_aesthetics_model(args, root)

    if args.seed == -1:
        args.seed = random.randint(0, 2 ** 32 - 1)
    if not args.use_init:
        args.init_image = None
    if args.sampler == 'plms' and (args.use_init or anim_args.animation_mode != 'None'):
        print(f"Init images aren't supported with PLMS yet, switching to KLMS")
        args.sampler = 'klms'
    if args.sampler != 'ddim':
        args.ddim_eta = 0

    if anim_args.animation_mode == 'None':
        anim_args.max_frames = 1
    elif anim_args.animation_mode == 'Video Input':
        args.use_init = True

    # clean up unused memory
    gc.collect()
    torch.cuda.empty_cache()

    # dispatch to appropriate renderer
    if anim_args.animation_mode == '2D' or anim_args.animation_mode == '3D':
        if not pass_render and not skip_new and not use_simple_pipeline:
            # Use AnimateDiff instead of traditional rendering
            print("Using AnimateDiff for video generation...")
            print(animation_prompts)

         # Check if character occurrences need to be updated and update them
            if characters:
                needs_update = any(len(char.line_occurrences) == 0 for char in characters)
                if needs_update:
                    print("Updating character line occurrences...")
                    characters = update_character_occurrences(characters, lyric2prompt)
                    
                    # Debug: Print updated occurrences
                    for char in characters:
                        print(f"Character {char.name} appears in lines: {char.line_occurrences}")
            
            # Pass all characters to AnimateDiff instead of line-specific ones
            video_output_dir = generate_animatediff_video(args, anim_args, animation_prompts, root, characters)
            
            if video_output_dir is None:
                logger.log_fallback("AnimateDiff", "traditional_rendering", "AnimateDiff generation failed")
                print("AnimateDiff generation failed, falling back to traditional rendering")
                # Load the model now if we need to fall back
                if root.model is None:
                    print("Loading traditional model for fallback...")
                    root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)
                render_animation(args, anim_args, animation_prompts, root)
        else:
            # Traditional rendering (or simple pipeline) - make sure model is loaded if needed
            if use_simple_pipeline:
                print("Using simple diffusers pipeline for rendering...")
            else:
                print("Using traditional rendering...")
                if root.model is None:
                    print("Loading traditional model for rendering...")
                    root.model, root.device = load_model(root, load_on_run_all=True, check_sha256=True, map_location=root.map_location)
            render_animation(args, anim_args, animation_prompts, root)
    elif anim_args.animation_mode == 'Video Input':
        render_input_video(args, anim_args, animation_prompts, root)
    elif anim_args.animation_mode == 'Interpolation':
        render_interpolation(args, anim_args, animation_prompts, root)
    else:
        render_image_batch(args, animation_prompts, root)

    """
    # Create Video From Frames
    """

    skip_video_for_run_all = False  # @param {type: 'boolean'}
    fps = fps_p  # @param {type:"number"}
    use_manual_settings = False  # @param {type:"boolean"}
    render_steps = False  # @param {type: 'boolean'}
    path_name_modifier = "x0_pred"  # @param ["x0_pred","x"]
    make_gif = False
    bitdepth_extension = "exr" if args.bit_depth_output == 32 else "png"

    if skip_video_for_run_all == True:
        print('Skipping video creation, uncheck skip_video_for_run_all if you want to run it')
    else:

        if use_manual_settings:
            max_frames = "200"  # @param {type:"string"}
        else:
            if render_steps:  # render steps from a single image
                fname = f"{path_name_modifier}_%05d.png"
                all_step_dirs = [os.path.join(args.outdir, d) for d in os.listdir(args.outdir) if
                                 os.path.isdir(os.path.join(args.outdir, d))]
                newest_dir = max(all_step_dirs, key=os.path.getmtime)
                image_path = os.path.join(newest_dir, fname)
                print(f"Reading images from {image_path}")
                mp4_path = os.path.join(newest_dir, f"{args.timestring}_{path_name_modifier}.mp4")
                max_frames = str(args.steps)
            else:  # render images for a video
                image_path = os.path.join(args.outdir, f"{args.timestring}_%05d.{bitdepth_extension}")
                mp4_path = os.path.join(root.output_path, f"{mp3_name}_mute_.mp4")
                max_frames = str(anim_args.max_frames)

        # make video
        # Calculate precise duration to match audio
        video_duration = lyric2prompt[-1]['end']
        
        cmd = [
            'ffmpeg',
            '-y',
            '-framerate', str(fps),  # Use -framerate instead of -r for input
            '-i', image_path,
            '-t', str(video_duration),  # Use duration instead of frame count for more precise timing
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '17',
            '-preset', 'veryfast',
            mp4_path
        ]

        print(f"Creating video with duration: {video_duration:.2f}s using {max_frames} frames at {fps} fps")

        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(stderr)
            raise RuntimeError(stderr)

    if my_args.caption_mode is not None:
        add_captions_to_video(mp4_path, lyric2prompt, os.path.join(root.output_path, f"{mp3_name}_lyrics.mp4"),
                              my_args.caption_mode, my_args.add_fittar)

        add_audio_to_mp4(os.path.join(root.output_path, f"{mp3_name}_lyrics.mp4"), my_args.mp3_file,
                         os.path.join(root.output_path, f"{mp3_name}.mp4"))
        print('done adding the lyrics, prompts, and audio')
    else:
        add_audio_to_mp4(os.path.join(root.output_path, f"{mp3_name}_mute_.mp4"), my_args.mp3_file,
                         os.path.join(root.output_path, f"{mp3_name}.mp4"))
    t1 = time.time()
    print('video generation took, ', (t1 - t0) / 60, ' mins')
    
    # Generate run summary
    summary = logger.get_run_summary()
    if summary:
        print(f"\n=== Run Summary ===")
        print(f"Run ID: {summary['run_id']}")
        print(f"Logs saved to: {summary['logs_directory']}")
        print(f"Files created: {len(summary['files_created'])}")

    # Log frame-to-prompt mapping for evaluation
    if skip_new:
        # Old video generation method
        logger.log_frame_to_prompt_mapping(animation_prompts, anim_args.max_frames, method="old")
    else:
        # AnimateDiff method
        logger.log_frame_to_prompt_mapping(animation_prompts, anim_args.max_frames, method="animatediff")


if __name__ == "__main__":
    main()
