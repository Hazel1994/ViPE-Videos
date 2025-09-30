import os
import json
import datetime
import traceback
import shutil
from pathlib import Path

class VideoGenerationLogger:
    def __init__(self, saving_dir, mp3_name):
        """Initialize logger with logs directory in saving_dir"""
        self.saving_dir = saving_dir
        self.mp3_name = mp3_name
        self.logs_dir = os.path.join(saving_dir, "logs")
        
        # Create timestamp for this run
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.logs_dir, f"{mp3_name}_{self.timestamp}")
        
        # Create directories
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "errors"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "dreambooth"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "gemini"), exist_ok=True)
        
        print(f"Logging to: {self.run_dir}")
    
    def log_program_arguments(self, user_args, my_args):
        """Log the arguments used when running the program"""
        try:
            args_data = {
                "timestamp": self.timestamp,
                "user_args": {
                    "vipe_checkpoint": user_args.vipe_checkpoint,
                    "mp3_file": user_args.mp3_file,
                    "saving_dir": user_args.saving_dir,
                    "music_gap_prompt": user_args.music_gap_prompt,
                    "music_gap_threshold": user_args.music_gap_threshold,
                    "prefix": user_args.prefix,
                    "context_size": user_args.context_size,
                    "abstractness": user_args.abstractness,
                    "skip_vipe": user_args.skip_vipe,
                    "image_quality_number": user_args.image_quality_number,
                    "visual_effect_period": user_args.visual_effect_period,
                    "caption_mode": user_args.caption_mode,
                    "skip_visual_effect": user_args.skip_visual_effect,
                    "animation_mode": user_args.animation_mode,
                    "disco_mode": user_args.disco_mode,
                    "generation_mode": user_args.generation_mode
                },
                "processed_args": {
                    "device": my_args.device,
                    "saving_dir": my_args.saving_dir,
                    "music_gap_prompt": my_args.music_gap_prompt,
                    "prefix": my_args.prefix,
                    "mp3_file": my_args.mp3_file,
                    "transcription_file": my_args.transcription_file,
                    "context_size": my_args.context_size,
                    "song_abstractness": my_args.song_abstractness,
                    "music_gap_threshold": my_args.music_gap_threshold,
                    "do_sample": my_args.do_sample,
                    "use_vipe": my_args.use_vipe,
                    "n_img_reward_samples": my_args.n_img_reward_samples,
                    "caption_mode": my_args.caption_mode,
                    "postfix_prompts": my_args.postfix_prompts,
                    "prompt_file": my_args.prompt_file,
                    "disco_mode": my_args.disco_mode,
                    "animation_mode": my_args.animation_mode,
                    "use_init": my_args.use_init,
                    "use_visual_effect": my_args.use_visual_effect,
                    "checkpoint": my_args.checkpoint,
                    "timestring": getattr(my_args, 'timestring', 'None')
                }
            }
            
            with open(os.path.join(self.run_dir, "program_arguments.json"), 'w') as f:
                json.dump(args_data, f, indent=2)

        except Exception as e:
            self.log_error("program_arguments", e)
    
    def log_transcription(self, transcription_file_path):
        """Copy transcription file to logs"""
        try:
            if os.path.exists(transcription_file_path):
                dest_path = os.path.join(self.run_dir, "transcription.txt")
                shutil.copy2(transcription_file_path, dest_path)
            else:
                self.log_info("transcription", f"Transcription file not found: {transcription_file_path}")
        except Exception as e:
            self.log_error("transcription", e)
    
    def log_vipe_interpretations(self, lyric2prompt, prompt_file_path):
        """Log ViPE interpretations"""
        try:
            vipe_data = {
                "timestamp": self.timestamp,
                "prompt_file_path": prompt_file_path,
                "total_entries": len(lyric2prompt),
                "interpretations": lyric2prompt
            }
            
            with open(os.path.join(self.run_dir, "vipe_interpretations.json"), 'w') as f:
                json.dump(vipe_data, f, indent=2)
                        
        except Exception as e:
            self.log_error("vipe_interpretations", e)
    
    def log_gemini_prompt_and_response(self, prompt, response, success=True, error=None, iteration=1):
        """Log Gemini API interaction with iteration support"""
        timestamp = datetime.datetime.now().isoformat()
        
        log_entry = {
            "timestamp": timestamp,
            "component": "gemini_api",
            "iteration": iteration,
            "success": success,
            "prompt_length": len(prompt) if prompt else 0,
            "response_length": len(response) if response else 0,
            "error": str(error) if error else None
        }
        
        # Save detailed prompt and response to separate files
        if prompt:
            prompt_file = os.path.join(self.detailed_logs_dir, f"gemini_prompt_iteration_{iteration}.txt")
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(prompt)
            log_entry["prompt_file"] = prompt_file
        
        if response:
            response_file = os.path.join(self.detailed_logs_dir, f"gemini_response_iteration_{iteration}.txt")
            with open(response_file, 'w', encoding='utf-8') as f:
                f.write(response)
            log_entry["response_file"] = response_file
        
        self._write_log_entry("gemini_interactions", log_entry)
    
    def log_character_generation(self, characters, updated_prompts_path=None, characters_path=None):
        """Log character generation results"""
        try:
            char_data = {
                "timestamp": self.timestamp,
                "total_characters": len(characters) if characters else 0,
                "updated_prompts_path": updated_prompts_path,
                "characters_path": characters_path,
                "characters": [],
                "updated_prompts_content": None
            }
            
            # Read and store the content of updated_prompts file
            if updated_prompts_path and os.path.exists(updated_prompts_path):
                try:
                    with open(updated_prompts_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # Try to parse as JSON if possible
                        try:
                            char_data["updated_prompts_content"] = json.loads(content)
                        except json.JSONDecodeError:
                            # If not valid JSON, store as plain text
                            char_data["updated_prompts_content"] = content
                except Exception as e:
                    char_data["updated_prompts_content"] = f"Error reading file: {str(e)}"
            
            if characters:
                for char in characters:
                    char_info = {
                        "name": char.name,
                        "description": char.description,
                        "unique_identifier": char.unique_identifier,
                        "occurrences": getattr(char, 'occurrences', 0),
                        "training_images": getattr(char, 'training_images', None),
                        "model_path": getattr(char, 'model_path', None)
                    }
                    char_data["characters"].append(char_info)
            
            with open(os.path.join(self.run_dir, "gemini", "character_generation.json"), 'w') as f:
                json.dump(char_data, f, indent=2)
                    
        except Exception as e:
            self.log_error("character_generation", e)
    
    def log_dreambooth_parameters(self, character, parameters, success=None, model_path=None, error=None):
        """Log DreamBooth training parameters and results"""
        try:
            dreambooth_data = {
                "timestamp": self.timestamp,
                "character_name": character.name,
                "character_description": character.description,
                "unique_identifier": character.unique_identifier,
                "training_images_path": getattr(character, 'training_images', None),
                "parameters": parameters,
                "success": success,
                "model_path": model_path,
                "error": str(error) if error else None
            }
            
            filename = f"dreambooth_{character.name.lower().replace(' ', '_')}.json"
            with open(os.path.join(self.run_dir, "dreambooth", filename), 'w') as f:
                json.dump(dreambooth_data, f, indent=2)
                        
        except Exception as e:
            self.log_error("dreambooth_logging", e)
    
    def log_animation_args(self, args, anim_args):
        """Log animation arguments"""
        try:
            # Convert SimpleNamespace to dict
            args_dict = vars(args) if hasattr(args, '__dict__') else args
            anim_args_dict = vars(anim_args) if hasattr(anim_args, '__dict__') else anim_args
            
            anim_data = {
                "timestamp": self.timestamp,
                "deform_args": args_dict,
                "animation_args": anim_args_dict
            }
            
            with open(os.path.join(self.run_dir, "animation_arguments.json"), 'w') as f:
                json.dump(anim_data, f, indent=2, default=str)
            
        except Exception as e:
            self.log_error("animation_args", e)
    
    def log_diffusion_model(self, model_info, use_animatediff=False, traditional_model=None):
        """Log information about the diffusion model used"""
        try:
            model_data = {
                "timestamp": self.timestamp,
                "use_animatediff": use_animatediff,
                "model_checkpoint": getattr(model_info, 'model_checkpoint', None),
                "model_config": getattr(model_info, 'model_config', None),
                "device": str(getattr(model_info, 'device', None)),
                "traditional_model_loaded": traditional_model is not None,
                "models_path": getattr(model_info, 'models_path', None),
                "output_path": getattr(model_info, 'output_path', None)
            }
            
            with open(os.path.join(self.run_dir, "diffusion_model.json"), 'w') as f:
                json.dump(model_data, f, indent=2, default=str)
                        
        except Exception as e:
            self.log_error("diffusion_model", e)
    
    def log_error(self, context, error):
        """Log errors and exceptions"""
        try:
            error_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "context": context,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc()
            }
            
            filename = f"error_{context}_{datetime.datetime.now().strftime('%H%M%S')}.json"
            with open(os.path.join(self.run_dir, "errors", filename), 'w') as f:
                json.dump(error_data, f, indent=2)
            
            print(f"Logged error in {context}: {error}")
            
        except Exception as e:
            print(f"Failed to log error: {e}")
    
    def log_fallback(self, from_method, to_method, reason):
        """Log fallbacks between different methods"""
        try:
            fallback_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "from_method": from_method,
                "to_method": to_method,
                "reason": reason
            }
            
            # Append to fallbacks file
            fallbacks_file = os.path.join(self.run_dir, "fallbacks.json")
            
            if os.path.exists(fallbacks_file):
                with open(fallbacks_file, 'r') as f:
                    fallbacks = json.load(f)
            else:
                fallbacks = []
            
            fallbacks.append(fallback_data)
            
            with open(fallbacks_file, 'w') as f:
                json.dump(fallbacks, f, indent=2)
            
            print(f"Logged fallback: {from_method} → {to_method} ({reason})")
            
        except Exception as e:
            self.log_error("fallback_logging", e)
    
    def log_info(self, context, message):
        """Log general information"""
        try:
            info_data = {
                "timestamp": datetime.datetime.now().isoformat(),
                "context": context,
                "message": message
            }
            
            # Append to info file
            info_file = os.path.join(self.run_dir, "info.json")
            
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    info_entries = json.load(f)
            else:
                info_entries = []
            
            info_entries.append(info_data)
            
            with open(info_file, 'w') as f:
                json.dump(info_entries, f, indent=2)
            
            print(f"Logged info: {context} - {message}")
            
        except Exception as e:
            print(f"Failed to log info: {e}")
    
    def get_run_summary(self):
        """Generate a summary of this run"""
        try:
            summary = {
                "run_id": f"{self.mp3_name}_{self.timestamp}",
                "logs_directory": self.run_dir,
                "files_created": []
            }
            
            # List all files created in this run
            for root, dirs, files in os.walk(self.run_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), self.run_dir)
                    summary["files_created"].append(rel_path)
            
            with open(os.path.join(self.run_dir, "run_summary.json"), 'w') as f:
                json.dump(summary, f, indent=2)
            
            return summary
            
        except Exception as e:
            self.log_error("run_summary", e)
            return None
    
    def log_frame_to_prompt_mapping(self, animation_prompts, max_frames, method="auto"):
        """Log mapping of frames to prompts for evaluation - works for both AnimateDiff and old method"""
        try:
            mapping_data = {
                "timestamp": self.timestamp,
                "total_frames": max_frames,
                "generation_method": method,
                "frame_to_prompt_mapping": []
            }
            
            if method == "animatediff":
                # Use AnimateDiff extraction logic
                from helpers.animatediff import _extract_lines_from_animation_prompts
                lines = _extract_lines_from_animation_prompts(animation_prompts, max_frames)
                
                for i, line in enumerate(lines):
                    mapping_entry = {
                        "prompt_index": i,
                        "text": line.get('prompt', ''),
                        "prompt": line.get('prompt', ''),
                        "start_frame": line['start_frame'],
                        "end_frame": line['end_frame'] - 1,  # Make end_frame inclusive
                        "frame_count": line['duration']
                    }
                    mapping_data["frame_to_prompt_mapping"].append(mapping_entry)
                    
            else:
                # Use old method logic - animation_prompts is {frame_num: prompt_text}
                # Convert to frame-by-frame mapping
                frame_to_prompt = {}
                
                # Fill in the prompts for each frame
                for frame_num, prompt_text in animation_prompts.items():
                    frame_to_prompt[int(frame_num)] = prompt_text
                
                # Forward fill missing frames
                current_prompt = None
                for frame_idx in range(max_frames):
                    if frame_idx in frame_to_prompt:
                        current_prompt = frame_to_prompt[frame_idx]
                    elif current_prompt is not None:
                        frame_to_prompt[frame_idx] = current_prompt
                
                # Group consecutive frames with same prompt
                current_frame = 0
                prompt_index = 0
                
                while current_frame < max_frames:
                    if current_frame not in frame_to_prompt:
                        current_frame += 1
                        continue
                        
                    current_prompt = frame_to_prompt[current_frame]
                    start_frame = current_frame
                    
                    # Find end of this prompt
                    while current_frame < max_frames and frame_to_prompt.get(current_frame) == current_prompt:
                        current_frame += 1
                    
                    end_frame = current_frame - 1
                    frame_count = end_frame - start_frame + 1
                    
                    mapping_entry = {
                        "prompt_index": prompt_index,
                        "text": current_prompt,
                        "prompt": current_prompt,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "frame_count": frame_count
                    }
                    mapping_data["frame_to_prompt_mapping"].append(mapping_entry)
                    prompt_index += 1
            
            # Save the mapping
            with open(os.path.join(self.run_dir, "frame_to_prompt_mapping.json"), 'w') as f:
                json.dump(mapping_data, f, indent=2)
            
            print(f"Frame-to-prompt mapping logged to: {os.path.join(self.run_dir, 'frame_to_prompt_mapping.json')}")
            
        except Exception as e:
            self.log_error("frame_to_prompt_mapping", e)