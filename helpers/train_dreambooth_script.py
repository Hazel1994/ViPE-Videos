#!/usr/bin/env python3
# filepath: /home/webereli/bar/ViPE-Videos/helpers/train_dreambooth_script.py

import subprocess
import sys
import os
from pathlib import Path
from typing import Tuple

def train_dreambooth(
    model_name,
    instance_dir,
    class_dir, 
    output_dir,
    instance_prompt,
    class_prompt,
    resolution,
    train_batch_size,
    learning_rate,
    max_train_steps,
    lora_r,
    lora_alpha,
    lora_text_encoder_r,
    lora_text_encoder_alpha,
    num_class_images,
    prior_loss_weight=1.0
):
    """
    Train a Dreambooth model with LoRA
    
    All parameters are required and passed from train_character()
    """
    
    # Validate directories exist
    if not os.path.exists(instance_dir):
        raise ValueError(f"Instance directory does not exist: {instance_dir}")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(class_dir).mkdir(parents=True, exist_ok=True)
    
    # Path to your existing dreambooth training script
    script_path = "dreambooth/examples/lora_dreambooth/train_dreambooth.py"
    
    # Build command using your existing script with configurable prior_loss_weight
    cmd = [
        "accelerate", "launch", script_path,
        f"--pretrained_model_name_or_path={model_name}",
        f"--instance_data_dir={instance_dir}",
        f"--class_data_dir={class_dir}",
        f"--output_dir={output_dir}",
        "--train_text_encoder",
        "--with_prior_preservation",
        f"--prior_loss_weight={prior_loss_weight}",
        f"--instance_prompt={instance_prompt}",
        f"--class_prompt={class_prompt}",
        f"--resolution={resolution}",
        f"--train_batch_size={train_batch_size}",
        f"--num_class_images={num_class_images}",
        "--use_lora",
        f"--lora_r={lora_r}",
        f"--lora_alpha={lora_alpha}",
        f"--lora_text_encoder_r={lora_text_encoder_r}",
        f"--lora_text_encoder_alpha={lora_text_encoder_alpha}",
        f"--learning_rate={learning_rate}",
        f"--max_train_steps={max_train_steps}",
        "--mixed_precision=fp16",
        "--gradient_accumulation_steps=2",
        "--checkpointing_steps=200",
        "--enable_xformers_memory_efficient_attention",
        "--seed=42"
    ]
    
    print("Starting Dreambooth training...")
    print(f"Base model: {model_name}")
    print(f"Training steps: {max_train_steps}")
    print(f"Learning rate: {learning_rate}")
    print(f"LoRA rank: {lora_r}")
    print(f"Prior loss weight: {prior_loss_weight}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        print("LoRA adapters saved to:", output_dir)
        return True, output_dir
    except subprocess.CalledProcessError as e:
        print(f"Training failed: {e}")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        return False, None
    except FileNotFoundError:
        print("Error: 'accelerate' command not found. Please install accelerate:")
        print("pip install accelerate")
        return False, None

def train_character(character, saving_dir):
    """
    Train a character using DreamBooth LoRA with fixed parameters
    """
    print(f"Training character: {character.name}")
    
    # Fixed training parameters - change here to modify training settings
    num_class_images = 200
    max_train_steps = 800
    learning_rate = 0.0002
    lora_r = 16
    lora_alpha = 27
    lora_text_encoder_r = 16
    lora_text_encoder_alpha = 17
    prior_loss_weight = 1.0
    
    # Store parameters for logging
    training_parameters = {
        "model_name": "SG161222/Realistic_Vision_V5.1_noVAE",
        "num_class_images": num_class_images,
        "max_train_steps": max_train_steps,
        "learning_rate": learning_rate,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_text_encoder_r": lora_text_encoder_r,
        "lora_text_encoder_alpha": lora_text_encoder_alpha,
        "prior_loss_weight": prior_loss_weight,
        "resolution": 512,
        "train_batch_size": 1,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 2,
        "checkpointing_steps": 200,
        "seed": 42
    }
    
    # Prepare character-specific paths and prompts
    folder_name = character.name.lower().replace(' ', '_').replace(',', '')
    output_dir = os.path.join(saving_dir, "models", folder_name)
    class_dir = os.path.join(saving_dir, "class_images")
    
    # Use the character's unique identifier for instance prompt
    # Use just the descriptor for class prompt to avoid duplication
    instance_prompt = f"a photo of {character.unique_identifier}"
    class_prompt = f"a photo of {character.description.split(',')[0].strip()}"
    
    # Add prompts to parameters
    training_parameters["instance_prompt"] = instance_prompt
    training_parameters["class_prompt"] = class_prompt
    training_parameters["output_dir"] = output_dir
    training_parameters["class_dir"] = class_dir
    
    print(f"Instance prompt: {instance_prompt}")
    print(f"Class prompt: {class_prompt}")
    print(f"Output directory: {output_dir}")
    print(f"Training with: {num_class_images} class images, {max_train_steps} steps, lr={learning_rate}")
    
    # Validate training data
    if not hasattr(character, 'training_images') or not os.path.exists(character.training_images):
        print(f"Error: No training images found for {character.name}")
        return False, None, training_parameters
    
    image_files = [f for f in os.listdir(character.training_images) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if len(image_files) < 3:
        print(f"Error: Need at least 3 training images for {character.name}, found {len(image_files)}")
        return False, None, training_parameters
    
    print(f"Found {len(image_files)} training images")
    training_parameters["num_training_images"] = len(image_files)
    
    # Call the training function with all required parameters
    try:
        success, model_path = train_dreambooth(
            model_name=training_parameters["model_name"],
            instance_dir=character.training_images,
            class_dir=class_dir,
            output_dir=output_dir,
            instance_prompt=instance_prompt,
            class_prompt=class_prompt,
            resolution=training_parameters["resolution"],
            train_batch_size=training_parameters["train_batch_size"],
            learning_rate=learning_rate,
            max_train_steps=max_train_steps,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_text_encoder_r=lora_text_encoder_r,
            lora_text_encoder_alpha=lora_text_encoder_alpha,
            num_class_images=num_class_images,
            prior_loss_weight=prior_loss_weight
        )
        
        if success:
            print(f"Successfully trained character: {character.name}")
            return True, model_path, training_parameters
        else:
            print(f"Failed to train character: {character.name}")
            return False, None, training_parameters
            
    except Exception as e:
        print(f"Error training character {character.name}: {e}")
        return False, None, training_parameters

def is_valid_lora_directory(model_path: str) -> Tuple[bool, str]:
    """Check if a directory contains a valid DreamBooth LoRA model structure."""
    try:
        if not os.path.exists(model_path) or not os.path.isdir(model_path):
            return False, "Directory does not exist"
        
        # Check for required subdirectories
        unet_dir = os.path.join(model_path, "unet")
        text_encoder_dir = os.path.join(model_path, "text_encoder")
        
        if not os.path.exists(unet_dir) or not os.path.isdir(unet_dir):
            return False, "Missing 'unet' subdirectory"
            
        if not os.path.exists(text_encoder_dir) or not os.path.isdir(text_encoder_dir):
            return False, "Missing 'text_encoder' subdirectory"
        
        # Check for required files in each subdirectory
        required_files = ["adapter_config.json", "adapter_model.safetensors"]
        
        for subdir, subdir_name in [(unet_dir, "unet"), (text_encoder_dir, "text_encoder")]:
            for required_file in required_files:
                file_path = os.path.join(subdir, required_file)
                if not os.path.exists(file_path):
                    return False, f"Missing {required_file} in {subdir_name} directory"
                
                # Check if adapter_model.safetensors is not empty
                if required_file == "adapter_model.safetensors":
                    if os.path.getsize(file_path) == 0:
                        return False, f"Empty {required_file} in {subdir_name} directory"
        
        return True, "Valid LoRA model structure"
        
    except (OSError, IOError, PermissionError) as e:
        return False, f"Error accessing directory: {e}"