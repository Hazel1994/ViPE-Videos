import json
import os
import torch
import warnings
from PIL import Image
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial

# Suppress the transformers processor warning
warnings.filterwarnings("ignore", message=".*use_fast.*", category=UserWarning)

IMAGE_DIR_PATH = "/graphics/scratch2/students/webereli/evaluation/2025-09/ViPE"
IMAGE_STAMP_PREFIX ="20250904144032"
FRAME_TO_PROMPT_MAPPING_PATH = "/graphics/scratch2/students/webereli/evaluation/frame_to_prompt_mapping.json"
FPS = 15

# Cache directory for the model
CACHE_DIR = "/graphics/scratch2/students/webereli/.cache/huggingface"

# Initialize with cache directory and offline mode as fallback
try:
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
except:
    # Fallback: try with cache directory
    os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
    os.environ["HF_HOME"] = CACHE_DIR
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score_single(image, prompt):
    """Calculate CLIP score between a single image and prompt"""
    image_int = (image * 255).astype("uint8")
    # Add batch dimension and convert to tensor
    image_tensor = torch.from_numpy(image_int).permute(2, 0, 1).unsqueeze(0)
    clip_score_val = clip_score_fn(image_tensor, [prompt]).detach()
    return float(clip_score_val)

def calculate_average_clip_score(images, prompt):
    """Calculate average CLIP score between multiple images and a prompt"""
    scores = []
    for image in images:
        score = calculate_clip_score_single(image, prompt)
        scores.append(score)
    
    avg_score = np.mean(scores)
    return round(avg_score, 4), scores

def prepare_images(start_frame, end_frame):
    """Load images for a given frame range, excluding first and last 5 frames"""
    # Exclude first and last 5 frames
    actual_start = start_frame + 5
    actual_end = end_frame - 5
    
    if actual_start >= actual_end:
        # If range is too small, use at least one frame in the middle
        middle_frame = (start_frame + end_frame) // 2
        actual_start = middle_frame
        actual_end = middle_frame + 1
    
    images = []
    for frame_idx in range(actual_start, actual_end):
        image_path = os.path.join(IMAGE_DIR_PATH, f"{IMAGE_STAMP_PREFIX}_{frame_idx:05d}.png")
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            images.append(image_array)
    
    return images if images else None

def prepare_frame_to_prompt_mapping():
    """Load frame to prompt mapping from JSON file"""
    with open(FRAME_TO_PROMPT_MAPPING_PATH, 'r') as f:
        data = json.load(f)
    return data

def get_frame_range(start_time, end_time, fps):
    """Convert time range to frame range - DEPRECATED, use mapping instead"""
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    return start_frame, end_frame

def main():
    """Main function to calculate CLIP scores for each prompt using frame mapping"""
    mapping_data = prepare_frame_to_prompt_mapping()
    frame_mappings = mapping_data["frame_to_prompt_mapping"]
    
    all_scores = []
    
    for prompt_entry in frame_mappings:
        prompt_index = prompt_entry["prompt_index"]
        start_frame = prompt_entry["start_frame"]
        end_frame = prompt_entry["end_frame"]
        prompt_text = prompt_entry["prompt"]
        frame_count = prompt_entry["frame_count"]
        
        print(f"Processing prompt {prompt_index + 1}: frames {start_frame}-{end_frame} ({frame_count} frames)")
        
        # Load corresponding images
        images = prepare_images(start_frame, end_frame)
        
        if images is None or len(images) == 0:
            print(f"Warning: No images found for prompt {prompt_index + 1}")
            continue
        
        # Calculate average CLIP score across all images for this prompt
        try:
            avg_score, individual_scores = calculate_average_clip_score(images, prompt_text)
            all_scores.append({
                "prompt_index": prompt_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "prompt": prompt_text,
                "avg_clip_score": avg_score,
                "individual_scores": individual_scores,
                "num_frames_used": len(images)
            })
            print(f"Average CLIP score: {avg_score} (using {len(images)} frames)")
        except Exception as e:
            print(f"Error calculating CLIP score for prompt {prompt_index + 1}: {e}")
    
    # Calculate and print overall statistics
    if all_scores:
        scores_only = [s["avg_clip_score"] for s in all_scores]
        avg_score = np.mean(scores_only)
        print(f"\nOverall Results:")
        print(f"Average CLIP score: {avg_score:.4f}")
        print(f"Min CLIP score: {min(scores_only):.4f}")
        print(f"Max CLIP score: {max(scores_only):.4f}")
        print(f"Processed {len(all_scores)}/{len(frame_mappings)} prompts successfully")
        
        # Save results to file
        results_path = "clip_scores_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_scores, f, indent=2)
        print(f"Results saved to {results_path}")
    
    return all_scores

if __name__ == "__main__":
    results = main()

