import json
import os
import cv2
import numpy as np
import argparse
import glob
from pathlib import Path

# Import insightface components
try:
    import insightface
    from insightface.app import FaceAnalysis
    
    # Initialize face analysis model
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # Use 'CPUExecutionProvider' for CPU
    app.prepare(ctx_id=0)  # ctx_id=-1 for CPU, 0 for GPU
    
    FACE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: insightface not available. Error: {e}")
    print("Please install insightface: pip install insightface")
    FACE_ANALYSIS_AVAILABLE = False

# Target character names to look for in prompts (case insensitive)
TARGET_CHARACTERS = ["elara", "alex", "jordan"]

def get_face_embedding(image_path):
    """Extract face embedding from an image path"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print(f"Warning: Multiple faces detected in {image_path}. Using first detected face")
    
    return faces[0].embedding

def calculate_cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embeddings"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return float(similarity)

def get_baseline_character(character):
    """Get the baseline character for comparison (next character in sequence, wrapping around)"""
    # Extract the character number
    char_num = int(character.split('_')[1])
    
    # Calculate baseline character (next in sequence, wrap around from 5 to 1)
    baseline_num = (char_num % 5) + 1
    baseline_character = f"character_{baseline_num}"
    
    return baseline_character

def find_date_folder(stamp_dir):
    """Find the date folder (like '2025-09') in the stamp directory"""
    if not os.path.exists(stamp_dir):
        raise ValueError(f"Stamp directory does not exist: {stamp_dir}")
    
    # Look for folders that match date pattern (YYYY-MM)
    date_folders = []
    for item in os.listdir(stamp_dir):
        item_path = os.path.join(stamp_dir, item)
        if os.path.isdir(item_path) and len(item) == 7 and item[4] == '-':
            try:
                # Check if it's a valid date format
                year, month = item.split('-')
                if len(year) == 4 and len(month) == 2 and year.isdigit() and month.isdigit():
                    date_folders.append(item)
            except:
                continue
    
    if not date_folders:
        raise ValueError(f"No date folder found in {stamp_dir}")
    
    if len(date_folders) > 1:
        print(f"Warning: Multiple date folders found: {date_folders}. Using the first one: {date_folders[0]}")
    
    return date_folders[0]

def find_directories(stamp):
    """Find the image directory and frame mapping file based on the stamp"""
    # Extract the prefix from the stamp (e.g., "apt" from "apt_20250915_091637")
    stamp_prefix = stamp.split('_')[0]
    base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    
    if not os.path.exists(base_path):
        raise ValueError(f"Base path does not exist: {base_path}")
    
    # Find the date folder
    date_folder = find_date_folder(base_path)
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

def load_training_images(character):
    """Load training images for a specific character"""
    training_dir = "/home/webereli/ViPE-Videos/evaluation/training_img"
    
    if not os.path.exists(training_dir):
        raise ValueError(f"Training images directory does not exist: {training_dir}")
    
    # Find all training images for the character
    pattern = os.path.join(training_dir, f"{character}_*.png")
    training_files = glob.glob(pattern)
    
    if not training_files:
        raise ValueError(f"No training images found for {character} in {training_dir}")
    
    print(f"Found {len(training_files)} training images for {character}")
    
    # Extract face embeddings from training images
    training_embeddings = []
    valid_files = []
    
    for img_file in training_files:
        try:
            embedding = get_face_embedding(img_file)
            training_embeddings.append(embedding)
            valid_files.append(os.path.basename(img_file))
            print(f"Successfully extracted embedding from: {os.path.basename(img_file)}")
        except Exception as e:
            print(f"Warning: Could not extract face from {os.path.basename(img_file)}: {e}")
            continue
    
    if not training_embeddings:
        raise ValueError(f"No valid face embeddings could be extracted from training images for {character}")
    
    print(f"Successfully loaded {len(training_embeddings)} training face embeddings for {character}")
    return training_embeddings, valid_files

def get_frame_files_for_prompt(image_dir, start_frame, end_frame):
    """Get frame files for a specific prompt based on frame range"""
    # Get all frame files in the directory
    frame_files = []
    for file in os.listdir(image_dir):
        if file.endswith('.png') and '_' in file:
            frame_files.append(os.path.join(image_dir, file))
    
    if not frame_files:
        return []
    
    # Sort by the frame number (last 5 digits before .png)
    frame_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    # Get frames for the specified range
    selected_frames = []
    for frame_idx in range(start_frame, end_frame + 1):
        if frame_idx < len(frame_files):
            selected_frames.append(frame_files[frame_idx])
    
    return selected_frames

def calculate_face_similarity_for_prompt(image_dir, start_frame, end_frame, training_embeddings):
    """Calculate average face similarity for all frames in a prompt"""
    frame_files = get_frame_files_for_prompt(image_dir, start_frame, end_frame)
    
    if not frame_files:
        print(f"Warning: No frame files found for frames {start_frame}-{end_frame}")
        return 0.0, 0, 0, []
    
    all_similarities = []
    successful_extractions = 0
    
    for frame_file in frame_files:
        try:
            frame_embedding = get_face_embedding(frame_file)
            
            # Calculate similarity with each training embedding
            frame_similarities = []
            for training_emb in training_embeddings:
                similarity = calculate_cosine_similarity(frame_embedding, training_emb)
                if similarity > 0.0:  # Ignore 0.0 scores
                    frame_similarities.append(similarity)
            
            if frame_similarities:
                # Use the maximum similarity across all training images for this frame
                max_similarity = max(frame_similarities)
                all_similarities.append(max_similarity)
                successful_extractions += 1
            
        except Exception as e:
            print(f"Warning: Could not extract face from {os.path.basename(frame_file)}: {e}")
            continue
    
    # Calculate average similarity (ignoring 0.0 scores)
    valid_similarities = [s for s in all_similarities if s > 0.0]
    average_similarity = np.mean(valid_similarities) if valid_similarities else 0.0
    
    return float(average_similarity), len(frame_files), successful_extractions, all_similarities

def prompt_contains_target_character(prompt_text):
    """Check if prompt contains any of the target character names (case insensitive)"""
    prompt_lower = prompt_text.lower()
    for character in TARGET_CHARACTERS:
        if character.lower() in prompt_lower:
            return True
    return False

def calculate_face_similarity_statistics(similarities):
    """Calculate statistics including quartiles for face similarity results"""
    if not similarities:
        return {
            "total_prompts_processed": 0,
            "average_face_similarity": None,
            "min_similarity": None,
            "max_similarity": None,
            "std_similarity": None,
            "q1_lower_quartile": None,
            "q2_median": None,
            "q3_upper_quartile": None
        }
    
    # Calculate quartiles
    sorted_similarities = sorted(similarities)
    n = len(sorted_similarities)
    q1 = sorted_similarities[int(n * 0.25)] if n > 0 else None
    q2_median = sorted_similarities[int(n * 0.5)] if n > 0 else None
    q3 = sorted_similarities[int(n * 0.75)] if n > 0 else None
    
    return {
        "total_prompts_processed": len(similarities),
        "average_face_similarity": float(np.mean(similarities)),
        "min_similarity": float(min(similarities)),
        "max_similarity": float(max(similarities)),
        "std_similarity": float(np.std(similarities)),
        "q1_lower_quartile": round(q1, 4) if q1 is not None else None,
        "q2_median": round(q2_median, 4) if q2_median is not None else None,
        "q3_upper_quartile": round(q3, 4) if q3 is not None else None
    }

def calculate_face_identity_coherence(image_dir, frame_mappings, character, baseline_character):
    """Calculate face identity coherence using training images for both target and baseline characters"""
    if not FACE_ANALYSIS_AVAILABLE:
        raise RuntimeError("Face analysis not available. Please install insightface.")
    
    # Load training images for both characters
    print(f"Loading training images for target character: {character}")
    training_embeddings, training_files = load_training_images(character)
    
    print(f"Loading training images for baseline character: {baseline_character}")
    baseline_embeddings, baseline_files = load_training_images(baseline_character)
    
    # Process each prompt
    prompt_results = []
    baseline_results = []
    prompt_similarities = []
    baseline_similarities = []
    skipped_prompts = 0
    
    print(f"Processing prompts containing target characters: {TARGET_CHARACTERS}")
    
    for i, prompt_entry in enumerate(frame_mappings):
        prompt_index = prompt_entry["prompt_index"]
        start_frame = prompt_entry["start_frame"]
        end_frame = prompt_entry["end_frame"]
        prompt_text = prompt_entry["prompt"]
        frame_count = prompt_entry["frame_count"]
        
        # Check if prompt contains any target character names
        if not prompt_contains_target_character(prompt_text):
            print(f"Skipping prompt {prompt_index + 1}: does not contain target characters")
            skipped_prompts += 1
            continue
        
        print(f"Processing prompt {prompt_index + 1}/{len(frame_mappings)}: frames {start_frame}-{end_frame}")
        
        # Calculate similarities for target character
        avg_similarity, total_frames, successful_frames, individual_similarities = calculate_face_similarity_for_prompt(
            image_dir, start_frame, end_frame, training_embeddings
        )
        
        # Calculate similarities for baseline character
        baseline_avg_similarity, baseline_total_frames, baseline_successful_frames, baseline_individual_similarities = calculate_face_similarity_for_prompt(
            image_dir, start_frame, end_frame, baseline_embeddings
        )
        
        # Store target character results
        prompt_detail = {
            "prompt_index": prompt_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": frame_count,
            "prompt": prompt_text,
            "character": character,
            "training_images_used": training_files,
            "total_frames_found": total_frames,
            "successful_face_detections": successful_frames,
            "face_detection_rate": successful_frames / total_frames if total_frames > 0 else 0.0,
            "average_face_similarity": avg_similarity,
            "individual_similarities": individual_similarities,
            "valid_similarities_count": len([s for s in individual_similarities if s > 0.0])
        }
        
        # Store baseline character results
        baseline_detail = {
            "prompt_index": prompt_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": frame_count,
            "prompt": prompt_text,
            "character": baseline_character,
            "training_images_used": baseline_files,
            "total_frames_found": baseline_total_frames,
            "successful_face_detections": baseline_successful_frames,
            "face_detection_rate": baseline_successful_frames / baseline_total_frames if baseline_total_frames > 0 else 0.0,
            "average_face_similarity": baseline_avg_similarity,
            "individual_similarities": baseline_individual_similarities,
            "valid_similarities_count": len([s for s in baseline_individual_similarities if s > 0.0])
        }
        
        prompt_results.append(prompt_detail)
        baseline_results.append(baseline_detail)
        
        if avg_similarity > 0.0:  # Only include non-zero similarities
            prompt_similarities.append(avg_similarity)
        
        if baseline_avg_similarity > 0.0:  # Only include non-zero similarities
            baseline_similarities.append(baseline_avg_similarity)
        
        print(f"  Target character ({character}): Found {total_frames} frames, {successful_frames} successful face detections, avg similarity: {avg_similarity:.4f}")
        print(f"  Baseline character ({baseline_character}): Found {baseline_total_frames} frames, {baseline_successful_frames} successful face detections, avg similarity: {baseline_avg_similarity:.4f}")
    
    print(f"Processed {len(prompt_results)} prompts containing target characters")
    print(f"Skipped {skipped_prompts} prompts (no target characters)")
    print(f"Target character valid similarities (>0.0): {len(prompt_similarities)}")
    print(f"Baseline character valid similarities (>0.0): {len(baseline_similarities)}")
    
    if not prompt_similarities:
        raise ValueError("No valid face similarities found for target character. Need at least one prompt with valid face detections.")
    
    if not baseline_similarities:
        print("Warning: No valid face similarities found for baseline character.")
    
    # Calculate statistics for both target and baseline
    target_stats = calculate_face_similarity_statistics(prompt_similarities)
    baseline_stats = calculate_face_similarity_statistics(baseline_similarities)
    
    results = {
        "character": character,
        "baseline_character": baseline_character,
        "training_images_used": training_files,
        "baseline_training_images_used": baseline_files,
        "num_training_images": len(training_files),
        "num_baseline_training_images": len(baseline_files),
        "target_characters_searched": TARGET_CHARACTERS,
        "total_prompts_in_mapping": len(frame_mappings),
        "prompts_with_target_characters": len(prompt_results),
        "skipped_prompts": skipped_prompts,
        "prompt_similarities": prompt_similarities,
        "baseline_similarities": baseline_similarities,
        "prompt_details": prompt_results,
        "baseline_details": baseline_results
    }
    
    # Add target statistics to results with prefix
    for key, value in target_stats.items():
        results[f"target_{key}"] = value
    
    # Add baseline statistics to results with prefix
    for key, value in baseline_stats.items():
        results[f"baseline_{key}"] = value
    
    return results

def save_results(results, stamp):
    """Save results to JSON file in the evaluation logs directory with statistics at the beginning"""
    stamp_prefix = stamp.split('_')[0]
    output_dir = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "face_similarity_results.json")
    
    # Reorganize results to put statistics first
    ordered_results = {}
    
    # Add basic info first
    ordered_results["character"] = results["character"]
    ordered_results["baseline_character"] = results["baseline_character"]
    
    # Add all target statistics together
    target_stat_keys = [
        "target_total_prompts_processed",
        "target_average_face_similarity", 
        "target_min_similarity",
        "target_max_similarity",
        "target_std_similarity",
        "target_q1_lower_quartile",
        "target_q2_median",
        "target_q3_upper_quartile"
    ]
    
    for key in target_stat_keys:
        if key in results:
            ordered_results[key] = results[key]
    
    # Add all baseline statistics together
    baseline_stat_keys = [
        "baseline_total_prompts_processed",
        "baseline_average_face_similarity",
        "baseline_min_similarity", 
        "baseline_max_similarity",
        "baseline_std_similarity",
        "baseline_q1_lower_quartile",
        "baseline_q2_median",
        "baseline_q3_upper_quartile"
    ]
    
    for key in baseline_stat_keys:
        if key in results:
            ordered_results[key] = results[key]
    
    # Add remaining metadata
    metadata_keys = [
        "training_images_used",
        "baseline_training_images_used", 
        "num_training_images",
        "num_baseline_training_images",
        "target_characters_searched",
        "total_prompts_in_mapping",
        "prompts_with_target_characters",
        "skipped_prompts"
    ]
    
    for key in metadata_keys:
        if key in results:
            ordered_results[key] = results[key]
    
    # Add similarity arrays
    ordered_results["prompt_similarities"] = results["prompt_similarities"]
    ordered_results["baseline_similarities"] = results["baseline_similarities"]
    
    # Add detailed results last
    ordered_results["prompt_details"] = results["prompt_details"]
    ordered_results["baseline_details"] = results["baseline_details"]
    
    with open(output_file, 'w') as f:
        json.dump(ordered_results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return output_file

def main():
    """Main function to calculate face identity coherence"""
    parser = argparse.ArgumentParser(description='Calculate face identity coherence using training images')
    parser.add_argument('--stamp', required=True, help='Timestamp stamp like "test_file_20250910_160439"')
    parser.add_argument('--character', required=True, help='Character name like "character_1"')
    
    args = parser.parse_args()
    stamp = args.stamp
    character = args.character
    
    # Get baseline character
    baseline_character = get_baseline_character(character)
    
    try:
        print(f"Processing stamp: {stamp}")
        print(f"Target character: {character}")
        print(f"Baseline character: {baseline_character}")
        
        # Find the image directory and mapping file
        image_dir, mapping_file = find_directories(stamp)
        print(f"Image directory: {image_dir}")
        print(f"Frame mapping file: {mapping_file}")
        
        # Load frame mappings
        frame_mappings = load_frame_mapping(mapping_file)
        print(f"Found {len(frame_mappings)} prompts in mapping file")
        
        # Calculate face identity coherence
        results = calculate_face_identity_coherence(image_dir, frame_mappings, character, baseline_character)
        
        # Print summary
        print("\n" + "="*60)
        print("FACE IDENTITY COHERENCE RESULTS")
        print("="*60)
        print(f"Target character: {results['character']}")
        print(f"Baseline character: {results['baseline_character']}")
        print(f"Target training images used: {results['num_training_images']}")
        for img_file in results['training_images_used']:
            print(f"  - {img_file}")
        print(f"Baseline training images used: {results['num_baseline_training_images']}")
        for img_file in results['baseline_training_images_used']:
            print(f"  - {img_file}")
        print(f"Target characters searched: {results['target_characters_searched']}")
        print(f"Total prompts in mapping: {results['total_prompts_in_mapping']}")
        print(f"Prompts with target characters: {results['prompts_with_target_characters']}")
        print(f"Skipped prompts: {results['skipped_prompts']}")
        
        print("\n" + "-"*30 + " TARGET CHARACTER " + "-"*30)
        print(f"Prompts processed (valid similarities): {results['target_total_prompts_processed']}")
        if results['target_average_face_similarity'] is not None:
            print(f"Average face similarity: {results['target_average_face_similarity']:.4f}")
            print(f"Min similarity: {results['target_min_similarity']:.4f}")
            print(f"Max similarity: {results['target_max_similarity']:.4f}")
            print(f"Q1 (Lower Quartile): {results['target_q1_lower_quartile']:.4f}")
            print(f"Q2 (Median): {results['target_q2_median']:.4f}")
            print(f"Q3 (Upper Quartile): {results['target_q3_upper_quartile']:.4f}")
            print(f"Std similarity: {results['target_std_similarity']:.4f}")
        
        print("\n" + "-"*30 + " BASELINE CHARACTER " + "-"*29)
        print(f"Prompts processed (valid similarities): {results['baseline_total_prompts_processed']}")
        if results['baseline_average_face_similarity'] is not None:
            print(f"Average face similarity: {results['baseline_average_face_similarity']:.4f}")
            print(f"Min similarity: {results['baseline_min_similarity']:.4f}")
            print(f"Max similarity: {results['baseline_max_similarity']:.4f}")
            print(f"Q1 (Lower Quartile): {results['baseline_q1_lower_quartile']:.4f}")
            print(f"Q2 (Median): {results['baseline_q2_median']:.4f}")
            print(f"Q3 (Upper Quartile): {results['baseline_q3_upper_quartile']:.4f}")
            print(f"Std similarity: {results['baseline_std_similarity']:.4f}")
        
        # Save results
        output_file = save_results(results, stamp)
        
        print(f"\nProcessing completed successfully!")
        print(f"Results saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    results = main()