#!/usr/bin/env python3
"""
DreamBooth Parameter Testing Script - One-at-a-time Testing
Tests one parameter at a time against a baseline configuration
"""
# USE THIS COMAND IN BASH TO GET THE CURRENT STATUS
# jq '{total: .total_combinations, completed: .completed_count, failed: [.combinations[] | select(.status=="failed")] | length, pending: [.combinations[] | select(.status=="pending")] | length, running: [.combinations[] | select(.status=="running")] | length}' /graphics/scratch2/students/webereli/parameter_testing/dreambooth_config.json

import os
import torch
import json
import time
import fcntl
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import sys
import os

# Add the helpers directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../helpers')))

from train_dreambooth_script import train_dreambooth, is_valid_lora_directory

def load_lora_pipeline(lora_model_path, base_model_name="SG161222/Realistic_Vision_V5.1_noVAE", device="cuda"):
    """Load a Stable Diffusion pipeline with LoRA weights applied"""
    print(f"Loading base model: {base_model_name}")
    
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
    
    pipe.unet.half()
    pipe.text_encoder.half()
    
    print("LoRA pipeline loaded successfully!")
    return pipe

def generate_test_image(pipe, prompt, output_path, num_inference_steps=50, guidance_scale=7.5):
    """Generate a single test image"""
    print(f"Generating image with prompt: '{prompt}'")
    
    with torch.no_grad():
        image = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=512,
            width=512
        ).images[0]
    
    image.save(output_path)
    print(f"Image saved to: {output_path}")
    return image

def get_baseline_parameters():
    """Get baseline parameters from train_dreambooth_script.py train_character() function"""
    return {
        'num_class_images': 200,      # From train_character()
        'max_train_steps': 800,       # From train_character()
        'learning_rate': 0.0002,      # From train_character() (2e-4)
        'lora_r': 16,                 # From train_dreambooth() call in train_character()
        'lora_alpha': 16,             # Set equal to lora_r (changed from 27)
        'prior_loss_weight': 1.0,     # New parameter to test
    }

def generate_test_configurations():
    """Generate one-at-a-time test configurations based on baseline"""
    baseline = get_baseline_parameters()
    
    # Define parameter variations to test
    parameter_variations = {
        'num_class_images': [50, 100, 200, 400, 600, 800],
        'max_train_steps': [200, 400, 1200, 1600],
        'learning_rate': [5e-5, 1e-4, 5e-4, 1e-3],
        'lora_r': [8, 16, 32, 64],
        'prior_loss_weight': [0.1, 0.5, 2.0, 5.0],
    }
    
    configurations = []
    
    # Add baseline configuration first
    configurations.append({
        'name': 'baseline',
        'parameters': baseline.copy(),
        'description': 'Baseline configuration with lora_alpha=lora_r'
    })
    
    # Generate one-at-a-time variations
    for param_name, variations in parameter_variations.items():
        for value in variations:
            # Skip if this value is already the baseline
            if value == baseline[param_name]:
                continue
                
            # Create new configuration with one parameter changed
            config = baseline.copy()
            config[param_name] = value
            
            # Always keep lora_alpha equal to lora_r
            config['lora_alpha'] = config['lora_r']
            
            configurations.append({
                'name': f'{param_name}_{value}',
                'parameters': config,
                'description': f'Test {param_name}={value} (baseline: {baseline[param_name]}, lora_alpha=lora_r)'
            })
    
    return configurations

def initialize_config_file(config_file_path):
    """Initialize the configuration file with one-at-a-time test configurations"""
    
    configurations = generate_test_configurations()
    
    # Create configuration data
    config_data = {
        "test_type": "one_at_a_time",
        "baseline_parameters": get_baseline_parameters(),
        "total_combinations": len(configurations),
        "completed_count": 0,
        "combinations": []
    }
    
    for i, config in enumerate(configurations):
        param_folder = create_parameter_folder_name(
            config['parameters'], 
            config['name'] if config['name'] != 'baseline' else None
        )
        
        config_data["combinations"].append({
            "id": i,
            "name": config['name'],
            "folder_name": param_folder,
            "parameters": config['parameters'],
            "description": config['description'],
            "status": "pending",
            "machine_id": None,
            "start_time": None,
            "end_time": None
        })
    # Write configuration file
    with open(config_file_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Initialized one-at-a-time testing with {len(configurations)} configurations:")
    print(f"  - 1 baseline configuration")
    for param_name in ['num_class_images', 'max_train_steps', 'learning_rate', 'lora_r', 'lora_alpha']:
        variations = [c for c in configurations if c['name'].startswith(param_name)]
        print(f"  - {len(variations)} {param_name} variations")
    
    return config_data

def get_next_task(config_file_path, machine_id):
    """Get the next available task from the configuration file"""
    
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            with open(config_file_path, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    config_data = json.load(f)
                    
                    # Find first pending task
                    next_task = None
                    for task in config_data["combinations"]:
                        if task["status"] == "pending":
                            next_task = task
                            break
                    
                    if next_task is None:
                        print("No pending tasks available")
                        return None
                    
                    # Mark task as running
                    next_task["status"] = "running"
                    next_task["machine_id"] = machine_id
                    next_task["start_time"] = time.time()
                    
                    # Write back to file
                    f.seek(0)
                    f.truncate()
                    json.dump(config_data, f, indent=2)
                    
                    print(f"Claimed task {next_task['id']}: {next_task['name']} ({next_task['description']})")
                    return next_task
                    
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error accessing config file (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Failed to access config file after all retries")
                return None
    
    return None

def mark_task_completed(config_file_path, task_id, success, training_time=None, generation_time=None):
    """Mark a task as completed in the configuration file"""
    
    max_retries = 5
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            with open(config_file_path, 'r+') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                
                try:
                    config_data = json.load(f)
                    
                    # Find and update the task
                    for task in config_data["combinations"]:
                        if task["id"] == task_id:
                            task["status"] = "completed" if success else "failed"
                            task["end_time"] = time.time()
                            task["training_time_seconds"] = training_time
                            task["generation_time_seconds"] = generation_time
                            if training_time:
                                task["training_time_formatted"] = f"{training_time:.2f}s ({training_time/60:.1f}m)"
                            if generation_time:
                                task["generation_time_formatted"] = f"{generation_time:.2f}s ({generation_time/60:.1f}m)"
                            break
                    
                    # Update completed count
                    if success:
                        config_data["completed_count"] += 1
                    
                    # Write back to file
                    f.seek(0)
                    f.truncate()
                    json.dump(config_data, f, indent=2)
                    
                    status = "completed" if success else "failed"
                    print(f"Marked task {task_id} as {status}")
                    if training_time:
                        print(f"  Training time: {training_time:.2f}s ({training_time/60:.1f}m)")
                    if generation_time:
                        print(f"  Generation time: {generation_time:.2f}s ({generation_time/60:.1f}m)")
                    return
                    
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                    
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error updating config file (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                print("Failed to update config file after all retries")
                return

def train_and_test_parameters(instance_dir, base_output_dir, unique_identifier, device, parameter_set, test_name):
    """Train a model with specific parameters and test it"""
    
    # Create parameter-specific folder name
    param_folder = create_parameter_folder_name(parameter_set, test_name if test_name != 'baseline' else None)
    
    # Create directories
    model_output_dir = os.path.join(base_output_dir, "models", param_folder)
    images_output_dir = os.path.join(base_output_dir, "test_images", param_folder)
    class_dir = os.path.join(base_output_dir, "class_images", param_folder)
    
    os.makedirs(model_output_dir, exist_ok=True)
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(class_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"TESTING: {test_name.upper()}")
    print(f"{'='*60}")
    print(f"Parameters:")
    baseline = get_baseline_parameters()
    for key, value in parameter_set.items():
        baseline_val = baseline[key]
        if value != baseline_val:
            print(f"  {key}: {value} (CHANGED from baseline: {baseline_val})")
        else:
            print(f"  {key}: {value}")
    print(f"Model output: {model_output_dir}")
    print(f"Images output: {images_output_dir}")
    
    training_time = 0.0
    generation_time = 0.0
    
    # Check if model already exists
    model_already_exists = False
    if os.path.exists(model_output_dir):
        is_valid, message = is_valid_lora_directory(model_output_dir)
        if is_valid:
            print(f"Model already exists and is valid, skipping training")
            model_already_exists = True
        else:
            print(f"Model exists but is invalid ({message}), retraining...")
            import shutil
            shutil.rmtree(model_output_dir)
            os.makedirs(model_output_dir, exist_ok=True)
    
    # Train the model if it doesn't exist or was invalid
    if not model_already_exists and not os.path.exists(os.path.join(model_output_dir, "unet", "adapter_config.json")):
        print("Starting training...")
        
        # Prepare prompts using unique identifier
        instance_prompt = f"a photo of {unique_identifier}"
        class_prompt = f"a photo of {unique_identifier.split()[1]}"
        
        print(f"Instance prompt: {instance_prompt}")
        print(f"Class prompt: {class_prompt}")
        print(f"Prior loss weight: {parameter_set['prior_loss_weight']}")
        
        training_start_time = time.time()
        
        try:
            success, model_path = train_dreambooth(
                model_name="SG161222/Realistic_Vision_V5.1_noVAE",
                instance_dir=instance_dir,
                class_dir=class_dir,
                output_dir=model_output_dir,
                instance_prompt=instance_prompt,
                class_prompt=class_prompt,
                resolution=512,
                train_batch_size=1,
                learning_rate=parameter_set['learning_rate'],
                max_train_steps=parameter_set['max_train_steps'],
                lora_r=parameter_set['lora_r'],
                lora_alpha=parameter_set['lora_alpha'],
                lora_text_encoder_r=parameter_set['lora_r'],
                lora_text_encoder_alpha=parameter_set['lora_alpha'],
                num_class_images=parameter_set['num_class_images'],
                prior_loss_weight=parameter_set['prior_loss_weight']  # Pass the configurable parameter
            )
            
            training_time = time.time() - training_start_time
            print(f"Training completed in {training_time:.2f}s ({training_time/60:.1f}m)")
            
            if not success:
                print(f"Training failed for parameter set: {param_folder}")
                return False, training_time, generation_time
                
        except Exception as e:
            training_time = time.time() - training_start_time
            print(f"Error during training: {e}")
            print(f"Training failed after {training_time:.2f}s ({training_time/60:.1f}m)")
            return False, training_time, generation_time
    else:
        print("Skipping training (model already exists)")
    
    # Test the trained model
    print("Testing trained model...")
    generation_start_time = time.time()
    
    try:
        # Load pipeline with trained LoRA
        pipe = load_lora_pipeline(model_output_dir, device=device)
        
        # Generate 25 test images for evaluation
        test_prompts = [
            f"portrait of {unique_identifier}, professional photography, studio lighting",
            f"close-up of {unique_identifier} smiling, natural lighting",
            f"headshot of {unique_identifier}, soft focus background",
            f"{unique_identifier} face, dramatic lighting, high contrast",
            f"{unique_identifier} looking directly at camera, intense gaze",
            f"profile view of {unique_identifier}, elegant pose",
            f"{unique_identifier} with wind in hair, outdoor portrait",
            f"close-up {unique_identifier} laughing, genuine happiness",
            f"{unique_identifier} face in golden hour light, warm tones",
            f"portrait {unique_identifier} in candlelight, romantic mood",
            f"{unique_identifier} face with raindrops, wet hair",
            f"headshot {unique_identifier} with red lipstick, glamorous",
            f"{unique_identifier} face in morning light, fresh awakening",
            f"close-up {unique_identifier} biting lip, thoughtful gesture",
            f"{unique_identifier} face with makeup, evening ready",
            f"portrait {unique_identifier} in candlelight, romantic mood",
            f"{unique_identifier} face showing surprise, wide eyes",
            f"{unique_identifier} standing in a park, full body shot",
            f"{unique_identifier} walking on a beach, sunset background",
            f"{unique_identifier} sitting on a bench, reading book",
            f"{unique_identifier} in a coffee shop, casual setting",
            f"{unique_identifier} dancing in a meadow, joyful movement",
            f"close-up {unique_identifier} with hat, fashionable style",
            f"{unique_identifier} face in mirror reflection, self-portrait",
            f"headshot {unique_identifier} with messy hair, casual look"
        ]
        
        # Generate test images
        for i, prompt in enumerate(test_prompts):
            output_file = os.path.join(images_output_dir, f"test_{i+1:02d}.png")
            generate_test_image(pipe, prompt, output_file)
        
        generation_time = time.time() - generation_start_time
        print(f"Generated {len(test_prompts)} test images successfully in {generation_time:.2f}s ({generation_time/60:.1f}m)")
        
        # Clean up pipeline to free memory
        del pipe
        torch.cuda.empty_cache()
        
        return True, training_time, generation_time
        
    except Exception as e:
        generation_time = time.time() - generation_start_time
        print(f"Error during testing: {e}")
        print(f"Generation failed after {generation_time:.2f}s ({generation_time/60:.1f}m)")
        return False, training_time, generation_time

def create_parameter_folder_name(params, test_name=None):
    """Create a folder name based on parameter values"""
    if test_name:
        return f"{test_name}_cls{params['num_class_images']}_steps{params['max_train_steps']}_r{params['lora_r']}_alpha{params['lora_alpha']}_lr{params['learning_rate']:.0e}_prior{params['prior_loss_weight']}"
    else:
        return f"baseline_cls{params['num_class_images']}_steps{params['max_train_steps']}_r{params['lora_r']}_alpha{params['lora_alpha']}_lr{params['learning_rate']:.0e}_prior{params['prior_loss_weight']}"

def main():
    """Main function to process one-at-a-time parameter testing"""
    
    # =============================================================================
    # CONFIGURATION - EDIT THESE VALUES
    # =============================================================================
    
    NAME = "alice"
    DESCRIPTION = "woman, young, black curly hair"
    INSTANCE_DIR = f"/graphics/scratch2/students/webereli/parameter_testing/training_images_test_file/{NAME}"
    BASE_OUTPUT_DIR = "/graphics/scratch2/students/webereli/parameter_testing/"
    CONFIG_FILE = os.path.join(BASE_OUTPUT_DIR, "dreambooth_config.json")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate unique identifier for character
    def _sanitize_name_for_identifier(name):
        import re
        return re.sub(r'[^a-z0-9]', '', name.lower())
    
    sanitized_name = _sanitize_name_for_identifier(NAME)
    first_descriptor = DESCRIPTION.split(',')[0].strip() if DESCRIPTION else ""
    UNIQUE_IDENTIFIER = f"sks{sanitized_name} {first_descriptor}"
    
    # Generate a unique machine ID
    MACHINE_ID = f"machine_{os.uname().nodename}_{os.getpid()}"
    
    # =============================================================================
    
    print("=== DreamBooth One-at-a-Time Parameter Testing ===")
    print(f"Machine ID: {MACHINE_ID}")
    print(f"Character: {NAME}")
    print(f"Unique Identifier: {UNIQUE_IDENTIFIER}")
    print(f"Instance directory: {INSTANCE_DIR}")
    print(f"Base output directory: {BASE_OUTPUT_DIR}")
    print(f"Config file: {CONFIG_FILE}")
    print(f"Device: {DEVICE}")
    
    baseline = get_baseline_parameters()
    print(f"\nBaseline parameters (from train_dreambooth_script.py):")
    for key, value in baseline.items():
        print(f"  {key}: {value}")
    
    # Validate instance directory
    if not os.path.exists(INSTANCE_DIR):
        print(f"Error: Instance directory does not exist: {INSTANCE_DIR}")
        return
    
    image_files = [f for f in os.listdir(INSTANCE_DIR) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) < 3:
        print(f"Error: Need at least 3 training images, found {len(image_files)}")
        return
    
    print(f"Found {len(image_files)} training images")
    
    # Create base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    
    # Initialize configuration file if it doesn't exist
    if not os.path.exists(CONFIG_FILE):
        print("Configuration file not found, initializing...")
        initialize_config_file(CONFIG_FILE)
    else:
        print("Using existing configuration file")
    
    # Process tasks until none are available
    total_processed = 0
    successful_tasks = 0
    failed_tasks = 0
    
    while True:
        # Get next available task
        task = get_next_task(CONFIG_FILE, MACHINE_ID)
        
        if task is None:
            print("No more tasks available. Exiting.")
            break
        
        print(f"\nProcessing task {task['id'] + 1}: {task['name']}")
        print(f"Description: {task['description']}")
        
        # Execute the task
        try:
            success, training_time, generation_time = train_and_test_parameters(
                INSTANCE_DIR,
                BASE_OUTPUT_DIR,
                UNIQUE_IDENTIFIER,
                DEVICE,
                task['parameters'],
                task['name']
            )
            
            # Mark task as completed with timing information
            mark_task_completed(CONFIG_FILE, task['id'], success, training_time, generation_time)
            
            total_processed += 1
            if success:
                successful_tasks += 1
                print(f"✓ Task {task['id'] + 1} ({task['name']}) completed successfully")
            else:
                failed_tasks += 1
                print(f"✗ Task {task['id'] + 1} ({task['name']}) failed")
                
        except Exception as e:
            print(f"Error processing task {task['id'] + 1}: {e}")
            mark_task_completed(CONFIG_FILE, task['id'], False, None, None)
            failed_tasks += 1
            total_processed += 1
    
    print(f"\n{'='*60}")
    print(f"ONE-AT-A-TIME TESTING COMPLETED!")
    print(f"{'='*60}")
    print(f"Total tasks processed: {total_processed}")
    print(f"Successful: {successful_tasks}")
    print(f"Failed: {failed_tasks}")

if __name__ == "__main__":
    main()