#!/usr/bin/env python3
"""
Video Generator Wrapper Script

This script calls generate_video.py with different generation modes and manages
file organization afterwards.

Usage:
cd evaluation/
python video_generator_wrapper.py --saving_dir /path/to/save --mp3_file song_name

The script will:
1. Generate video with different generation modes (original, animatediff, dreambooth_only, animatediff_and_dreambooth)
2. Move generated files to organized log folders
3. Create a summary of which stamp corresponds to which generation mode
"""

import os
import sys
import argparse
import subprocess
import re
import shutil
import json
import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Wrapper for generate_video.py with file management")
    parser.add_argument("--saving_dir", type=str, required=True, 
                       help="Directory where videos and models will be stored")
    parser.add_argument("--mp3_file", type=str, required=True,
                       help="Name of the mp3 file (without .mp3 extension)")
    parser.add_argument("--additional_args", type=str, default="",
                       help="Additional arguments to pass to generate_video.py")
    return parser.parse_args()


def extract_logs_directory(output_text):
    """Extract the logs directory path from the generate_video.py output"""
    # Look for the "Logs saved to:" line in the output
    pattern = r"Logs saved to:\s+(.+)"
    match = re.search(pattern, output_text)
    if match:
        return match.group(1).strip()
    
    # Fallback: look for "Logging to:" line
    pattern = r"Logging to:\s+(.+)"
    match = re.search(pattern, output_text)
    if match:
        return match.group(1).strip()
    
    return None


def extract_stamp_from_logs_dir(logs_dir):
    """Extract the stamp (folder name) from the logs directory path"""
    return os.path.basename(logs_dir)


def get_date_from_stamp(stamp, mp3_file):
    """Extract date in YYYY-MM format from stamp"""
    # Pattern: mp3_file_YYYYMMDD_HHMMSS
    pattern = rf"{re.escape(mp3_file)}_(\d{{8}})_\d{{6}}"
    match = re.search(pattern, stamp)
    if match:
        date_str = match.group(1)  # YYYYMMDD
        return f"{date_str[:4]}-{date_str[4:6]}"  # YYYY-MM
    return None


def setup_character_files(mp3_file, character_id):
    """
    Move character-specific files before generation
    
    Args:
        mp3_file: Name of the mp3 file (without extension)
        character_id: Character identifier (character_1, character_2, etc.)
    """
    print(f"Setting up files for {character_id}")
    
    # Source directories
    character_dir = f"/home/webereli/ViPE-Videos/evaluation/character_specific_files/{character_id}"
    models_source = os.path.join(character_dir, "models")
    
    # Destination directories
    mp3_dest_dir = "/home/webereli/ViPE-Videos/mp3"
    models_dest_dir = f"/graphics/scratch2/students/webereli/{mp3_file}/models"
    
    # File names to move
    json_files = [
        f"{mp3_file}_ctx_1_sample_True_vipe_True_abst_0_characters.json",
        f"{mp3_file}_ctx_1_sample_True_vipe_True_abst_0_with_characters.json"
    ]
    
    # Move JSON files
    for json_file in json_files:
        source_path = os.path.join(character_dir, json_file)
        dest_path = os.path.join(mp3_dest_dir, json_file)
        
        if os.path.exists(source_path):
            print(f"Moving {source_path} to {dest_path}")
            # Create destination directory if it doesn't exist
            os.makedirs(mp3_dest_dir, exist_ok=True)
            shutil.copy2(source_path, dest_path)
        else:
            print(f"Warning: JSON file not found: {source_path}")
    
    # Move models directory
    if os.path.exists(models_source):
        print(f"Moving models from {models_source} to {models_dest_dir}")
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(models_dest_dir), exist_ok=True)
        
        # Remove existing models directory if it exists
        if os.path.exists(models_dest_dir):
            shutil.rmtree(models_dest_dir)
        
        # Copy the models directory
        shutil.copytree(models_source, models_dest_dir)
    else:
        print(f"Warning: Models directory not found: {models_source}")
    
    print(f"Character {character_id} setup completed")


def run_generate_video(saving_dir, mp3_file, generation_mode=None, additional_args=""):
    """Run generate_video.py with specified parameters"""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to generate_video.py in the parent directory
    generate_video_path = os.path.join(script_dir, "..", "generate_video.py")
    
    cmd = ["python", generate_video_path, 
           "--saving_dir", saving_dir,
           "--mp3_file", mp3_file,
           "--caption_mode", "both"]
        
    if generation_mode:
        cmd.extend(["--generation_mode", generation_mode])
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args.split())
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Change to the generate_video.py directory to ensure proper imports
    original_cwd = os.getcwd()
    target_cwd = os.path.dirname(generate_video_path)
    
    try:
        # Run the command and capture output from the correct directory
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, 
                              cwd=target_cwd)
        return result.stdout, result.stderr, True
    except subprocess.CalledProcessError as e:
        print(f"Error running generate_video.py: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return e.stdout, e.stderr, False


def move_files(saving_dir, mp3_file, stamp, logs_base_dir):
    """Move the generated files to the logs directory"""
    # Extract date from stamp
    date_folder = get_date_from_stamp(stamp, mp3_file)
    if not date_folder:
        print(f"Warning: Could not extract date from stamp {stamp}")
        return False
    
    # Find the actual output directory (it has a complex naming pattern)
    # Look for directories that contain the mp3_file name and start with "test_"
    output_dir = None
    for item in os.listdir(saving_dir):
        item_path = os.path.join(saving_dir, item)
        if os.path.isdir(item_path) and mp3_file in item and item.startswith("test_"):
            output_dir = item_path
            break
    
    if not output_dir:
        print(f"Warning: Could not find output directory in {saving_dir}")
        return False
    
    print(f"Found output directory: {output_dir}")
    
    # Source paths - now look in the actual output directory
    mp4_file = f"{mp3_file}.mp4"
    mp4_source = os.path.join(output_dir, mp4_file)
    vipe_folder_source = os.path.join(output_dir, date_folder, "ViPE")
    
    # Destination directory
    stamp_dest_dir = os.path.join(logs_base_dir, stamp)
    
    # Create destination directory if it doesn't exist
    os.makedirs(stamp_dest_dir, exist_ok=True)
    
    # Move MP4 file
    if os.path.exists(mp4_source):
        mp4_dest = os.path.join(stamp_dest_dir, mp4_file)
        print(f"Moving {mp4_source} to {mp4_dest}")
        shutil.move(mp4_source, mp4_dest)
    else:
        print(f"Warning: MP4 file not found at {mp4_source}")
    
    # Move ViPE folder
    if os.path.exists(vipe_folder_source):
        vipe_dest = os.path.join(stamp_dest_dir, date_folder, "ViPE")
        os.makedirs(os.path.dirname(vipe_dest), exist_ok=True)
        print(f"Moving {vipe_folder_source} to {vipe_dest}")
        shutil.move(vipe_folder_source, vipe_dest)
    else:
        print(f"Warning: ViPE folder not found at {vipe_folder_source}")
    
    return True


def main():
    args = parse_args()
    
    # Prepare logs base directory
    logs_base_dir = os.path.join(args.saving_dir, "logs")
    os.makedirs(logs_base_dir, exist_ok=True)
    
    # Summary data
    summary = {
        "generated_at": datetime.datetime.now().isoformat(),
        "mp3_file": args.mp3_file,
        "saving_dir": args.saving_dir,
        "runs": []
    }
    
    # List of runs to perform
    runs = [
        {"name": "original_generation", "generation_mode": "original", "description": "Original video generation method", "character": None},
        {"name": "animatediff_only", "generation_mode": "animatediff", "description": "AnimateDiff without DreamBooth", "character": None}
    ]
    
    # Add character-specific runs for dreambooth_only and animatediff_and_dreambooth
    for character_num in range(1, 6):  # character_1 to character_5
        character_id = f"character_{character_num}"
        
        # Add dreambooth_only run for this character
        runs.append({
            "name": f"dreambooth_only_{character_id}",
            "generation_mode": "dreambooth_only",
            "description": f"Simple diffusers pipeline with DreamBooth LoRA only - {character_id}",
            "character": character_id
        })
        
        # Add animatediff_and_dreambooth run for this character
        runs.append({
            "name": f"animatediff_and_dreambooth_{character_id}",
            "generation_mode": "animatediff_and_dreambooth", 
            "description": f"Full generation with AnimateDiff and DreamBooth - {character_id}",
            "character": character_id
        })
    
    for run_config in runs:
        print(f"\n{'='*60}")
        print(f"Starting run: {run_config['name']}")
        print(f"Description: {run_config['description']}")
        if run_config.get('character'):
            print(f"Character: {run_config['character']}")
        print(f"{'='*60}")
        
        # Setup character files if needed
        if run_config.get('character'):
            setup_character_files(args.mp3_file, run_config['character'])
        
        # Run generate_video.py
        stdout, stderr, success = run_generate_video(
            args.saving_dir, 
            args.mp3_file, 
            run_config["generation_mode"],
            args.additional_args
        )
        
        if not success:
            print(f"Failed to generate video for run: {run_config['name']}")
            run_summary = {
                "name": run_config["name"],
                "generation_mode": run_config["generation_mode"],
                "success": False,
                "error": "Video generation failed",
                "description": run_config["description"]
            }
            if run_config.get('character'):
                run_summary["character"] = run_config["character"]
            summary["runs"].append(run_summary)
            continue
        
        # Extract logs directory
        logs_dir = extract_logs_directory(stdout)
        if not logs_dir:
            print(f"Warning: Could not extract logs directory from output")
            run_summary = {
                "name": run_config["name"],
                "generation_mode": run_config["generation_mode"],
                "success": False,
                "error": "Could not extract logs directory",
                "description": run_config["description"]
            }
            if run_config.get('character'):
                run_summary["character"] = run_config["character"]
            summary["runs"].append(run_summary)
            continue
        
        # Extract stamp
        stamp = extract_stamp_from_logs_dir(logs_dir)
        print(f"Generated stamp: {stamp}")
        print(f"Logs directory: {logs_dir}")
        
        # Move files
        move_success = move_files(args.saving_dir, args.mp3_file, stamp, logs_base_dir)
        
        # Record in summary
        run_summary = {
            "name": run_config["name"],
            "generation_mode": run_config["generation_mode"],
            "stamp": stamp,
            "logs_directory": logs_dir,
            "success": success and move_success,
            "description": run_config["description"]
        }
        if run_config.get('character'):
            run_summary["character"] = run_config["character"]
        summary["runs"].append(run_summary)
        
        print(f"Completed run: {run_config['name']} with stamp: {stamp}")
    
    # Save summary file
    summary_file = os.path.join(logs_base_dir, f"{args.mp3_file}_generation_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print("GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Summary saved to: {summary_file}")
    print("\nRuns completed:")
    for run in summary["runs"]:
        status = "✓" if run["success"] else "✗"
        mode_info = f"--generation_mode {run['generation_mode']}" if run["generation_mode"] else "default mode"
        character_info = f" (Character: {run['character']})" if run.get('character') else ""
        print(f"{status} {run['name']}: {mode_info}{character_info} -> {run.get('stamp', 'N/A')}")
    
    print(f"\nAll generated files have been organized in: {logs_base_dir}")


if __name__ == "__main__":
    main()
