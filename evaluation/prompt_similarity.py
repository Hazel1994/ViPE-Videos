import json
import os
import torch
import warnings
from PIL import Image
import numpy as np
import argparse
import glob
from pathlib import Path
import re
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

# Suppress warnings
warnings.filterwarnings("ignore")

# Local model path - no downloads
LOCAL_CLIP_MODEL_PATH = "/graphics/scratch2/students/webereli/evaluation/prompt_similarity/clip-vit-base-patch16"

def initialize_clip_model():
    """Initialize CLIP model from local files only"""
    print(f"Loading CLIP model from local path: {LOCAL_CLIP_MODEL_PATH}")
    
    if not os.path.exists(LOCAL_CLIP_MODEL_PATH):
        raise ValueError(f"Local CLIP model not found at: {LOCAL_CLIP_MODEL_PATH}")
    
    # Force offline mode
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    
    try:
        # Load model and processor from local directory
        model = CLIPModel.from_pretrained(LOCAL_CLIP_MODEL_PATH, local_files_only=True)
        processor = CLIPProcessor.from_pretrained(LOCAL_CLIP_MODEL_PATH, local_files_only=True)
        
        # Move model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        print(f"CLIP model loaded successfully on device: {device}")
        return model, processor, device
        
    except Exception as e:
        raise RuntimeError(f"Failed to load CLIP model from {LOCAL_CLIP_MODEL_PATH}: {e}")

def calculate_clip_score_manual(image, prompt, model, processor, device):
    """Calculate CLIP score manually using the loaded model"""
    try:
        # Process image and text
        inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            # Convert to similarity score (0-100 scale like torchmetrics clip_score)
            score = logits_per_image.squeeze().cpu().item()
            # Scale to match torchmetrics output format
            scaled_score = score / 100.0 * 2.5  # Approximate scaling to match torchmetrics
            
        return float(scaled_score)
        
    except Exception as e:
        print(f"Error calculating CLIP score: {e}")
        return 0.0

def clean_prompt_text(prompt_text):
    """
    Remove unique identifiers from Dreambooth prompts.
    Converts "sksalex Woman dancing in the street" to "Woman dancing in the street"
    """
    # Remove common unique identifier patterns (e.g., sksalex, sks_person, etc.)
    # Pattern matches: word starting with 'sks' followed by alphanumeric characters
    cleaned = re.sub(r'\bsks\w*\s+', '', prompt_text, flags=re.IGNORECASE)
    
    # Remove other common unique identifier patterns
    # Pattern for tokens like "a_unique_token" at the beginning
    cleaned = re.sub(r'^\w*_\w*\s+', '', cleaned)
    
    # Remove extra whitespace
    cleaned = ' '.join(cleaned.split())
    
    return cleaned

def calculate_dual_clip_scores(images, original_prompt, cleaned_prompt, model, processor, device):
    """
    Calculate CLIP scores for both original and cleaned prompts using the same images.
    Returns scores for both versions efficiently.
    """
    original_scores = []
    cleaned_scores = []
    
    for image_array in images:
        # Convert numpy array to PIL Image
        image_pil = Image.fromarray((image_array * 255).astype('uint8'))
        
        # Calculate score with original prompt (including unique identifiers)
        original_score = calculate_clip_score_manual(image_pil, original_prompt, model, processor, device)
        original_scores.append(original_score)
        
        # Calculate score with cleaned prompt (without unique identifiers)
        cleaned_score = calculate_clip_score_manual(image_pil, cleaned_prompt, model, processor, device)
        cleaned_scores.append(cleaned_score)
    
    return original_scores, cleaned_scores

def find_directories(stamp):
    """Find the image directory and frame mapping file based on the stamp"""
    # Extract the prefix from the stamp (e.g., "jump" from "jump_20250915_074937")
    stamp_prefix = stamp.split('_')[0]
    
    base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    
    if not os.path.exists(base_path):
        raise ValueError(f"Base path does not exist: {base_path}")
    
    # Find the date folder (should be a folder like "2025-09", "2024-12", etc.)
    date_folders = []
    for d in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, d)):
            # Check if it matches date pattern (YYYY-MM)
            if re.match(r'^\d{4}-\d{2}$', d):
                date_folders.append(d)
    
    if not date_folders:
        # Fallback: look for any folder that might contain ViPE
        all_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        for folder in all_folders:
            potential_vipe_dir = os.path.join(base_path, folder, "ViPE")
            if os.path.exists(potential_vipe_dir):
                date_folders.append(folder)
                break
        
        if not date_folders:
            raise ValueError(f"No date folder (YYYY-MM format) or folder with ViPE directory found in: {base_path}")
    
    if len(date_folders) > 1:
        # Sort date folders and use the most recent one
        date_folders.sort(reverse=True)
        print(f"Warning: Multiple date folders found: {date_folders}. Using most recent: {date_folders[0]}")
    
    date_folder = date_folders[0]
    image_dir = os.path.join(base_path, date_folder, "ViPE")
    
    if not os.path.exists(image_dir):
        raise ValueError(f"ViPE directory does not exist: {image_dir}")
    
    # Find frame mapping file
    mapping_file = os.path.join(base_path, "frame_to_prompt_mapping.json")
    if not os.path.exists(mapping_file):
        raise ValueError(f"Frame mapping file does not exist: {mapping_file}")
    
    return image_dir, mapping_file

def load_frame_mapping(mapping_file):
    """Load frame to prompt mapping from JSON file"""
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    return data["frame_to_prompt_mapping"]

def get_frame_files_for_prompt(image_dir, start_frame, end_frame):
    """Get frame files for a specific prompt based on frame range"""
    frame_files = []
    
    # Get any PNG file to detect the actual timestamp pattern
    all_files = os.listdir(image_dir)
    png_files = [f for f in all_files if f.endswith('.png')]
    
    if not png_files:
        print(f"Warning: No PNG files found in {image_dir}")
        return frame_files
    
    # Extract the actual timestamp prefix from any existing file
    sample_file = png_files[0]
    if '_' in sample_file:
        actual_timestamp_prefix = sample_file.split('_')[0]
    else:
        print(f"Warning: Unexpected filename format: {sample_file}")
        return frame_files
    
    # Now just look for the frame numbers with the correct prefix
    for frame_idx in range(start_frame, end_frame + 1):
        filename = f"{actual_timestamp_prefix}_{frame_idx:05d}.png"
        full_path = os.path.join(image_dir, filename)
        
        if os.path.exists(full_path):
            frame_files.append(full_path)
    
    frame_files.sort()
    return frame_files

def load_image(image_path):
    """Load and normalize image"""
    if not os.path.exists(image_path):
        raise ValueError(f"Image file does not exist: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    return image_array

def prepare_images_all_frames(image_dir, start_frame, end_frame):
    """Load all images for a given frame range"""
    frame_files = get_frame_files_for_prompt(image_dir, start_frame, end_frame)
    
    if not frame_files:
        return None, []
    
    images = []
    loaded_files = []
    
    for frame_file in frame_files:
        try:
            image = load_image(frame_file)
            images.append(image)
            loaded_files.append(frame_file)
        except Exception as e:
            print(f"Warning: Could not load image {frame_file}: {e}")
            continue
    
    return images if images else None, loaded_files

def prepare_images_median_frame(image_dir, start_frame, end_frame):
    """Load only the median frame for a given frame range"""
    frame_files = get_frame_files_for_prompt(image_dir, start_frame, end_frame)
    
    if not frame_files:
        return None, []
    
    # Get median frame
    median_idx = len(frame_files) // 2
    median_frame_file = frame_files[median_idx]
    
    try:
        image = load_image(median_frame_file)
        return [image], [median_frame_file]
    except Exception as e:
        print(f"Warning: Could not load median frame {median_frame_file}: {e}")
        return None, []

def calculate_prompt_similarity_all_frames_dual(image_dir, frame_mappings, model, processor, device):
    """Calculate prompt similarity using all frames with both original and cleaned prompts"""
    results = []
    
    print("Calculating CLIP scores using all frames (dual version: with and without unique identifiers)...")
    
    for i, prompt_entry in enumerate(frame_mappings):
        prompt_index = prompt_entry["prompt_index"]
        start_frame = prompt_entry["start_frame"]
        end_frame = prompt_entry["end_frame"]
        original_prompt = prompt_entry["prompt"]
        frame_count = prompt_entry["frame_count"]
        
        # Clean the prompt text (remove unique identifiers)
        cleaned_prompt = clean_prompt_text(original_prompt)
        
        print(f"Processing prompt {prompt_index + 1}/{len(frame_mappings)}: frames {start_frame}-{end_frame}")
        print(f"  Original prompt: {original_prompt}")
        print(f"  Cleaned prompt: {cleaned_prompt}")
        
        # Load images (only once!)
        images, loaded_files = prepare_images_all_frames(image_dir, start_frame, end_frame)
        
        if images is None or len(images) == 0:
            print(f"Warning: No images found for prompt {prompt_index + 1}")
            results.append({
                "prompt_index": prompt_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "original_prompt": original_prompt,
                "cleaned_prompt": cleaned_prompt,
                "original_clip_score": None,
                "cleaned_clip_score": None,
                "original_individual_scores": [],
                "cleaned_individual_scores": [],
                "num_frames_used": 0,
                "num_frames_found": 0,
                "error": "No images found"
            })
            continue
        
        try:
            # Calculate CLIP scores for both versions using the same images
            original_scores, cleaned_scores = calculate_dual_clip_scores(
                images, original_prompt, cleaned_prompt, model, processor, device
            )
            
            # Calculate averages
            original_avg = np.mean(original_scores)
            cleaned_avg = np.mean(cleaned_scores)
            
            results.append({
                "prompt_index": prompt_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "original_prompt": original_prompt,
                "cleaned_prompt": cleaned_prompt,
                "original_clip_score": round(original_avg, 4),
                "cleaned_clip_score": round(cleaned_avg, 4),
                "score_difference": round(original_avg - cleaned_avg, 4),
                "original_individual_scores": [round(s, 4) for s in original_scores],
                "cleaned_individual_scores": [round(s, 4) for s in cleaned_scores],
                "num_frames_used": len(images),
                "num_frames_found": len(loaded_files),
                "method": "all_frames_normalized_dual"
            })
            
            print(f"  Original CLIP score: {original_avg:.4f}")
            print(f"  Cleaned CLIP score: {cleaned_avg:.4f}")
            print(f"  Difference: {original_avg - cleaned_avg:.4f}")
            
        except Exception as e:
            print(f"Error calculating CLIP score for prompt {prompt_index + 1}: {e}")
            results.append({
                "prompt_index": prompt_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "original_prompt": original_prompt,
                "cleaned_prompt": cleaned_prompt,
                "original_clip_score": None,
                "cleaned_clip_score": None,
                "original_individual_scores": [],
                "cleaned_individual_scores": [],
                "num_frames_used": 0,
                "num_frames_found": 0,
                "error": str(e)
            })
    
    return results

def calculate_prompt_similarity_median_frame_dual(image_dir, frame_mappings, model, processor, device):
    """Calculate prompt similarity using median frames with both original and cleaned prompts"""
    results = []
    
    print("Calculating CLIP scores using median frames (dual version: with and without unique identifiers)...")
    
    for i, prompt_entry in enumerate(frame_mappings):
        prompt_index = prompt_entry["prompt_index"]
        start_frame = prompt_entry["start_frame"]
        end_frame = prompt_entry["end_frame"]
        original_prompt = prompt_entry["prompt"]
        frame_count = prompt_entry["frame_count"]
        
        # Clean the prompt text (remove unique identifiers)
        cleaned_prompt = clean_prompt_text(original_prompt)
        
        print(f"Processing prompt {prompt_index + 1}/{len(frame_mappings)}: frames {start_frame}-{end_frame}")
        print(f"  Original prompt: {original_prompt}")
        print(f"  Cleaned prompt: {cleaned_prompt}")
        
        # Load median frame (only once!)
        images, loaded_files = prepare_images_median_frame(image_dir, start_frame, end_frame)
        
        if images is None or len(images) == 0:
            print(f"Warning: No median frame found for prompt {prompt_index + 1}")
            results.append({
                "prompt_index": prompt_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "original_prompt": original_prompt,
                "cleaned_prompt": cleaned_prompt,
                "original_clip_score": None,
                "cleaned_clip_score": None,
                "median_frame_file": None,
                "num_frames_available": 0,
                "error": "No median frame found"
            })
            continue
        
        try:
            # Calculate CLIP scores for both versions using the same median frame
            median_image_array = images[0]
            median_image_pil = Image.fromarray((median_image_array * 255).astype('uint8'))
            
            original_score = calculate_clip_score_manual(median_image_pil, original_prompt, model, processor, device)
            cleaned_score = calculate_clip_score_manual(median_image_pil, cleaned_prompt, model, processor, device)
            
            results.append({
                "prompt_index": prompt_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "original_prompt": original_prompt,
                "cleaned_prompt": cleaned_prompt,
                "original_clip_score": round(original_score, 4),
                "cleaned_clip_score": round(cleaned_score, 4),
                "score_difference": round(original_score - cleaned_score, 4),
                "median_frame_file": os.path.basename(loaded_files[0]) if loaded_files else None,
                "num_frames_available": len(get_frame_files_for_prompt(image_dir, start_frame, end_frame)),
                "method": "median_frame_dual"
            })
            
            print(f"  Original CLIP score: {original_score:.4f}")
            print(f"  Cleaned CLIP score: {cleaned_score:.4f}")
            print(f"  Difference: {original_score - cleaned_score:.4f}")
            
        except Exception as e:
            print(f"Error calculating CLIP score for prompt {prompt_index + 1}: {e}")
            results.append({
                "prompt_index": prompt_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "original_prompt": original_prompt,
                "cleaned_prompt": cleaned_prompt,
                "original_clip_score": None,
                "cleaned_clip_score": None,
                "median_frame_file": None,
                "num_frames_available": len(get_frame_files_for_prompt(image_dir, start_frame, end_frame)),
                "error": str(e)
            })
    
    return results

def calculate_dual_statistics(results, method_name):
    """Calculate statistics for both original and cleaned versions"""
    valid_results = [r for r in results if r["original_clip_score"] is not None and r["cleaned_clip_score"] is not None]
    
    if not valid_results:
        return {
            "method": method_name,
            "total_prompts": len(results),
            "valid_prompts": 0,
            "original_stats": {
                "average_clip_score": None,
                "min_clip_score": None,
                "max_clip_score": None,
                "std_clip_score": None,
                "q1_lower_quartile": None,
                "q2_median": None,
                "q3_upper_quartile": None
            },
            "cleaned_stats": {
                "average_clip_score": None,
                "min_clip_score": None,
                "max_clip_score": None,
                "std_clip_score": None,
                "q1_lower_quartile": None,
                "q2_median": None,
                "q3_upper_quartile": None
            },
            "difference_stats": {
                "average_difference": None,
                "min_difference": None,
                "max_difference": None,
                "std_difference": None,
                "q1_lower_quartile": None,
                "q2_median": None,
                "q3_upper_quartile": None
            }
        }
    
    original_scores = [r["original_clip_score"] for r in valid_results]
    cleaned_scores = [r["cleaned_clip_score"] for r in valid_results]
    differences = [r["score_difference"] for r in valid_results]
    
    # Calculate quartiles for original scores
    sorted_original = sorted(original_scores)
    n_orig = len(sorted_original)
    orig_q1 = sorted_original[int(n_orig * 0.25)] if n_orig > 0 else None
    orig_q2 = sorted_original[int(n_orig * 0.5)] if n_orig > 0 else None
    orig_q3 = sorted_original[int(n_orig * 0.75)] if n_orig > 0 else None
    
    # Calculate quartiles for cleaned scores
    sorted_cleaned = sorted(cleaned_scores)
    n_clean = len(sorted_cleaned)
    clean_q1 = sorted_cleaned[int(n_clean * 0.25)] if n_clean > 0 else None
    clean_q2 = sorted_cleaned[int(n_clean * 0.5)] if n_clean > 0 else None
    clean_q3 = sorted_cleaned[int(n_clean * 0.75)] if n_clean > 0 else None
    
    # Calculate quartiles for differences
    sorted_diff = sorted(differences)
    n_diff = len(sorted_diff)
    diff_q1 = sorted_diff[int(n_diff * 0.25)] if n_diff > 0 else None
    diff_q2 = sorted_diff[int(n_diff * 0.5)] if n_diff > 0 else None
    diff_q3 = sorted_diff[int(n_diff * 0.75)] if n_diff > 0 else None
    
    return {
        "method": method_name,
        "total_prompts": len(results),
        "valid_prompts": len(valid_results),
        "original_stats": {
            "average_clip_score": round(np.mean(original_scores), 4),
            "min_clip_score": round(min(original_scores), 4),
            "max_clip_score": round(max(original_scores), 4),
            "std_clip_score": round(np.std(original_scores), 4),
            "q1_lower_quartile": round(orig_q1, 4) if orig_q1 is not None else None,
            "q2_median": round(orig_q2, 4) if orig_q2 is not None else None,
            "q3_upper_quartile": round(orig_q3, 4) if orig_q3 is not None else None
        },
        "cleaned_stats": {
            "average_clip_score": round(np.mean(cleaned_scores), 4),
            "min_clip_score": round(min(cleaned_scores), 4),
            "max_clip_score": round(max(cleaned_scores), 4),
            "std_clip_score": round(np.std(cleaned_scores), 4),
            "q1_lower_quartile": round(clean_q1, 4) if clean_q1 is not None else None,
            "q2_median": round(clean_q2, 4) if clean_q2 is not None else None,
            "q3_upper_quartile": round(clean_q3, 4) if clean_q3 is not None else None
        },
        "difference_stats": {
            "average_difference": round(np.mean(differences), 4),
            "min_difference": round(min(differences), 4),
            "max_difference": round(max(differences), 4),
            "std_difference": round(np.std(differences), 4),
            "q1_lower_quartile": round(diff_q1, 4) if diff_q1 is not None else None,
            "q2_median": round(diff_q2, 4) if diff_q2 is not None else None,
            "q3_upper_quartile": round(diff_q3, 4) if diff_q3 is not None else None
        }
    }

def save_results(all_frames_results, median_results, all_frames_stats, median_stats, stamp):
    """Save all results to a single JSON file with summary at the top"""
    # Extract the prefix from the stamp (e.g., "jump" from "jump_20250915_074937")
    stamp_prefix = stamp.split('_')[0]
    
    base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    
    # Create summary with most relevant values at the top
    summary = {
        "evaluation_summary": {
            "stamp": stamp,
            "description": "CLIP scores calculated for both original prompts (with unique identifiers) and cleaned prompts (without unique identifiers)",
            "key_findings": {
                "all_frames_method": {
                    "total_prompts": all_frames_stats['total_prompts'] if all_frames_stats else 0,
                    "valid_prompts": all_frames_stats['valid_prompts'] if all_frames_stats else 0,
                    "original_average_clip_score": all_frames_stats['original_stats']['average_clip_score'] if all_frames_stats and all_frames_stats['original_stats']['average_clip_score'] else None,
                    "cleaned_average_clip_score": all_frames_stats['cleaned_stats']['average_clip_score'] if all_frames_stats and all_frames_stats['cleaned_stats']['average_clip_score'] else None,
                    "average_score_difference": all_frames_stats['difference_stats']['average_difference'] if all_frames_stats and all_frames_stats['difference_stats']['average_difference'] else None
                },
                "median_frame_method": {
                    "total_prompts": median_stats['total_prompts'] if median_stats else 0,
                    "valid_prompts": median_stats['valid_prompts'] if median_stats else 0,
                    "original_average_clip_score": median_stats['original_stats']['average_clip_score'] if median_stats and median_stats['original_stats']['average_clip_score'] else None,
                    "cleaned_average_clip_score": median_stats['cleaned_stats']['average_clip_score'] if median_stats and median_stats['cleaned_stats']['average_clip_score'] else None,
                    "average_score_difference": median_stats['difference_stats']['average_difference'] if median_stats and median_stats['difference_stats']['average_difference'] else None
                }
            }
        },
        "detailed_statistics": {
            "all_frames_method": all_frames_stats,
            "median_frame_method": median_stats
        },
        "detailed_results": {
            "all_frames_method": all_frames_results,
            "median_frame_method": median_results
        }
    }
    
    # Save single comprehensive results file
    output_file = os.path.join(base_path, "prompt_similarity_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Print summary to console
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    if all_frames_stats and all_frames_stats['original_stats']['average_clip_score'] is not None:
        print("All Frames Method:")
        print(f"  Valid prompts: {all_frames_stats['valid_prompts']}/{all_frames_stats['total_prompts']}")
        print(f"  Original (with identifiers):  {all_frames_stats['original_stats']['average_clip_score']:.4f}")
        print(f"  Cleaned (without identifiers): {all_frames_stats['cleaned_stats']['average_clip_score']:.4f}")
        print(f"  Average difference:           {all_frames_stats['difference_stats']['average_difference']:.4f}")
    
    if median_stats and median_stats['original_stats']['average_clip_score'] is not None:
        print("\nMedian Frame Method:")
        print(f"  Valid prompts: {median_stats['valid_prompts']}/{median_stats['total_prompts']}")
        print(f"  Original (with identifiers):  {median_stats['original_stats']['average_clip_score']:.4f}")
        print(f"  Cleaned (without identifiers): {median_stats['cleaned_stats']['average_clip_score']:.4f}")
        print(f"  Average difference:           {median_stats['difference_stats']['average_difference']:.4f}")
    
    return output_file

def update_existing_results_with_quartiles(stamp):
    """Update existing prompt similarity results to include quartiles without recalculating everything"""
    stamp_prefix = stamp.split('_')[0]
    base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    results_file = os.path.join(base_path, "prompt_similarity_results.json")
    
    if not os.path.exists(results_file):
        print(f"No existing results file found at {results_file}")
        return False
    
    try:
        # Load existing results
        with open(results_file, 'r') as f:
            existing_data = json.load(f)
        
        print(f"Updating existing results with quartiles for {stamp}...")
        
        # Check if quartiles already exist
        detailed_stats = existing_data.get("detailed_statistics", {})
        all_frames_stats = detailed_stats.get("all_frames_method", {})
        
        if (all_frames_stats.get("original_stats", {}).get("q1_lower_quartile") is not None):
            print(f"Quartiles already exist for {stamp}, skipping update.")
            return True
        
        # Get detailed results to recalculate statistics with quartiles
        detailed_results = existing_data.get("detailed_results", {})
        all_frames_results = detailed_results.get("all_frames_method", [])
        median_results = detailed_results.get("median_frame_method", [])
        
        # Recalculate statistics with quartiles
        updated_all_frames_stats = calculate_dual_statistics(all_frames_results, "all_frames_normalized_dual") if all_frames_results else None
        updated_median_stats = calculate_dual_statistics(median_results, "median_frame_dual") if median_results else None
        
        # Update the data structure
        if updated_all_frames_stats:
            existing_data["detailed_statistics"]["all_frames_method"] = updated_all_frames_stats
            # Also update evaluation summary key findings
            if "evaluation_summary" in existing_data and "key_findings" in existing_data["evaluation_summary"]:
                kf = existing_data["evaluation_summary"]["key_findings"]
                if "all_frames_method" in kf:
                    # Keep existing values, just don't overwrite
                    pass
        
        if updated_median_stats:
            existing_data["detailed_statistics"]["median_frame_method"] = updated_median_stats
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"Successfully updated {results_file} with quartiles")
        return True
        
    except Exception as e:
        print(f"Error updating results with quartiles for {stamp}: {e}")
        return False

def main():
    """Main function to calculate prompt similarity using both methods with dual comparison"""
    parser = argparse.ArgumentParser(description='Calculate prompt similarity using CLIP scores (dual version: with and without unique identifiers)')
    parser.add_argument('--stamp', required=True, help='Timestamp stamp like "apt_20250916_160720"')
    parser.add_argument('--method', choices=['all', 'median', 'both'], default='both',
                      help='Method to use: all (all frames), median (median frame), or both')
    parser.add_argument('--force_rerun', action='store_true',
                      help='Force a complete rerun even if results already exist')
    parser.add_argument('--update-quartiles-only', action='store_true', 
                      help='Only update existing results with quartiles, do not recalculate')
    
    args = parser.parse_args()
    stamp = args.stamp
    
    # If only updating quartiles, do that and exit
    if args.update_quartiles_only:
        success = update_existing_results_with_quartiles(stamp)
        if success:
            print("Quartiles update completed successfully!")
            return {"update_only": True, "success": True}
        else:
            print("Failed to update quartiles.")
            return {"update_only": True, "success": False}
    
    try:
        print(f"Processing stamp: {stamp}")
        print(f"Method: {args.method}")
        print("Running dual comparison (with and without unique identifiers)")
        
        # Initialize CLIP model from local files
        print("Initializing CLIP model from local files...")
        model, processor, device = initialize_clip_model()
        
        # Find directories and load mappings
        image_dir, mapping_file = find_directories(stamp)
        print(f"Image directory: {image_dir}")
        print(f"Frame mapping file: {mapping_file}")
        
        frame_mappings = load_frame_mapping(mapping_file)
        print(f"Found {len(frame_mappings)} prompts in mapping file")
        
        all_frames_results = None
        median_results = None
        all_frames_stats = None
        median_stats = None
        
        # Calculate using all frames method
        if args.method in ['all', 'both']:
            print("\n" + "="*50)
            print("CALCULATING USING ALL FRAMES METHOD (DUAL)")
            print("="*50)
            all_frames_results = calculate_prompt_similarity_all_frames_dual(
                image_dir, frame_mappings, model, processor, device
            )
            all_frames_stats = calculate_dual_statistics(all_frames_results, "all_frames_normalized_dual")
        
        # Calculate using median frame method
        if args.method in ['median', 'both']:
            print("\n" + "="*50)
            print("CALCULATING USING MEDIAN FRAME METHOD (DUAL)")
            print("="*50)
            median_results = calculate_prompt_similarity_median_frame_dual(
                image_dir, frame_mappings, model, processor, device
            )
            median_stats = calculate_dual_statistics(median_results, "median_frame_dual")
        
        # Print results
        print("\n" + "="*50)
        print("PROMPT SIMILARITY RESULTS (DUAL COMPARISON)")
        print("="*50)
        
        def print_dual_stats(stats, method_name):
            print(f"{method_name}:")
            print(f"  Total prompts: {stats['total_prompts']}")
            print(f"  Valid prompts: {stats['valid_prompts']}")
            
            if stats['original_stats']['average_clip_score'] is not None:
                print("  Original Prompts (with unique identifiers):")
                print(f"    Average CLIP score: {stats['original_stats']['average_clip_score']}")
                print(f"    Min CLIP score: {stats['original_stats']['min_clip_score']}")
                print(f"    Max CLIP score: {stats['original_stats']['max_clip_score']}")
                print(f"    Std CLIP score: {stats['original_stats']['std_clip_score']}")
                
                print("  Cleaned Prompts (without unique identifiers):")
                print(f"    Average CLIP score: {stats['cleaned_stats']['average_clip_score']}")
                print(f"    Min CLIP score: {stats['cleaned_stats']['min_clip_score']}")
                print(f"    Max CLIP score: {stats['cleaned_stats']['max_clip_score']}")
                print(f"    Std CLIP score: {stats['cleaned_stats']['std_clip_score']}")
                
                print("  Score Differences (Original - Cleaned):")
                print(f"    Average difference: {stats['difference_stats']['average_difference']}")
                print(f"    Min difference: {stats['difference_stats']['min_difference']}")
                print(f"    Max difference: {stats['difference_stats']['max_difference']}")
                print(f"    Std difference: {stats['difference_stats']['std_difference']}")
            else:
                print("  No valid results")
        
        if all_frames_stats:
            print_dual_stats(all_frames_stats, "All Frames Method (Normalized)")
        
        if median_stats:
            print("\n")
            print_dual_stats(median_stats, "Median Frame Method")
        
        # Save results
        if args.method == 'both' or (all_frames_results and median_results):
            output_file = save_results(all_frames_results, median_results, all_frames_stats, median_stats, stamp)
            print(f"\nResults saved to: {output_file}")
        elif all_frames_results:
            # Save only all frames results
            stamp_prefix = stamp.split('_')[0]
            base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
            output_file = os.path.join(base_path, "prompt_similarity_all_frames_dual.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    "statistics": all_frames_stats,
                    "detailed_results": all_frames_results
                }, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        elif median_results:
            # Save only median results
            stamp_prefix = stamp.split('_')[0]
            base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
            output_file = os.path.join(base_path, "prompt_similarity_median_frame_dual.json")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump({
                    "statistics": median_stats,
                    "detailed_results": median_results
                }, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        
        # Update existing results with quartiles if they don't have them
        if not args.force_rerun:  # Only if not forcing a complete rerun
            update_existing_results_with_quartiles(stamp)
        
        print(f"\nProcessing completed successfully!")
        
        return {
            "all_frames_results": all_frames_results,
            "median_results": median_results,
            "all_frames_stats": all_frames_stats,
            "median_stats": median_stats
        }
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    results = main()