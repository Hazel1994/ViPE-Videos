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
    
    # Find the date folder using the same logic as face_identity.py
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

def get_frame_files_for_prompt(image_dir, start_frame, end_frame):
    """Get frame files for a specific prompt based on frame range"""
    # Get all frame files in the directory (like face_identity.py does)
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

def extract_timestamp_from_stamp(stamp):
    """Extract timestamp pattern from stamp"""
    # Assuming format like "test_file_20250910_160439"
    parts = stamp.split('_')
    if len(parts) >= 2:
        # Try to find date and time parts
        timestamp_parts = []
        for part in parts:
            if part.isdigit() and len(part) >= 6:
                timestamp_parts.append(part)
        
        if len(timestamp_parts) >= 2:
            return timestamp_parts[0] + timestamp_parts[1]  # e.g., "20250910160439"
        elif len(timestamp_parts) == 1:
            return timestamp_parts[0]
    
    # Fallback: use the stamp as is
    return stamp.replace('_', '')

def calculate_average_embedding_for_prompt(image_dir, start_frame, end_frame):
    """Calculate average face embedding for all frames in a prompt"""
    frame_files = get_frame_files_for_prompt(image_dir, start_frame, end_frame)
    
    if not frame_files:
        print(f"Warning: No frame files found for frames {start_frame}-{end_frame}")
        return None, 0, 0
    
    embeddings = []
    successful_extractions = 0
    
    for frame_file in frame_files:
        try:
            embedding = get_face_embedding(frame_file)
            embeddings.append(embedding)
            successful_extractions += 1
        except Exception as e:
            print(f"Warning: Could not extract face from {os.path.basename(frame_file)}: {e}")
            continue
    
    if not embeddings:
        print(f"Warning: No valid face embeddings extracted for frames {start_frame}-{end_frame}")
        return None, len(frame_files), 0
    
    # Calculate average embedding
    average_embedding = np.mean(embeddings, axis=0)
    
    return average_embedding, len(frame_files), successful_extractions

def calculate_prompt_coherence(image_dir, frame_mappings):
    """Calculate face embedding coherence between consecutive prompts"""
    if not FACE_ANALYSIS_AVAILABLE:
        raise RuntimeError("Face analysis not available. Please install insightface.")
    
    prompt_embeddings = []
    prompt_details = []
    
    # Extract average embeddings for each prompt
    print("Extracting average face embeddings for each prompt...")
    for i, prompt_entry in enumerate(frame_mappings):
        prompt_index = prompt_entry["prompt_index"]
        start_frame = prompt_entry["start_frame"]
        end_frame = prompt_entry["end_frame"]
        prompt_text = prompt_entry["prompt"]
        frame_count = prompt_entry["frame_count"]
        
        print(f"Processing prompt {prompt_index + 1}/{len(frame_mappings)}: frames {start_frame}-{end_frame}")
        
        avg_embedding, total_frames, successful_frames = calculate_average_embedding_for_prompt(
            image_dir, start_frame, end_frame
        )
        
        prompt_detail = {
            "prompt_index": prompt_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": frame_count,
            "prompt": prompt_text,
            "total_frames_found": total_frames,
            "successful_face_detections": successful_frames,
            "face_detection_rate": successful_frames / total_frames if total_frames > 0 else 0.0,
            "has_valid_embedding": avg_embedding is not None
        }
        
        if avg_embedding is not None:
            prompt_embeddings.append(avg_embedding)
            prompt_detail["embedding_index"] = len(prompt_embeddings) - 1
        else:
            prompt_detail["embedding_index"] = -1
        
        prompt_details.append(prompt_detail)
        
        print(f"  Found {total_frames} frames, {successful_frames} successful face detections")
    
    print(f"Successfully extracted embeddings from {len(prompt_embeddings)} out of {len(frame_mappings)} prompts")
    
    if len(prompt_embeddings) < 2:
        raise ValueError("Need at least 2 valid prompt embeddings to calculate coherence")
    
    # Calculate cosine similarities between consecutive prompts
    similarities = []
    comparison_details = []
    
    print("Calculating cosine similarities between consecutive prompts...")
    for i in range(len(prompt_embeddings) - 1):
        similarity = calculate_cosine_similarity(prompt_embeddings[i], prompt_embeddings[i + 1])
        similarities.append(similarity)
        
        # Find the corresponding prompt details
        prompt_1_details = None
        prompt_2_details = None
        
        for detail in prompt_details:
            if detail["embedding_index"] == i:
                prompt_1_details = detail
            elif detail["embedding_index"] == i + 1:
                prompt_2_details = detail
        
        comparison_details.append({
            "prompt_1_index": prompt_1_details["prompt_index"] if prompt_1_details else -1,
            "prompt_2_index": prompt_2_details["prompt_index"] if prompt_2_details else -1,
            "prompt_1_frames": f"{prompt_1_details['start_frame']}-{prompt_1_details['end_frame']}" if prompt_1_details else "unknown",
            "prompt_2_frames": f"{prompt_2_details['start_frame']}-{prompt_2_details['end_frame']}" if prompt_2_details else "unknown",
            "cosine_similarity": similarity
        })
    
    # Calculate average similarity
    average_similarity = np.mean(similarities) if similarities else 0.0
    
    results = {
        "total_prompts": len(frame_mappings),
        "prompts_with_valid_embeddings": len(prompt_embeddings),
        "consecutive_comparisons": len(similarities),
        "average_cosine_similarity": float(average_similarity),
        "min_similarity": float(min(similarities)) if similarities else 0.0,
        "max_similarity": float(max(similarities)) if similarities else 0.0,
        "std_similarity": float(np.std(similarities)) if similarities else 0.0,
        "individual_similarities": similarities,
        "comparison_details": comparison_details,
        "prompt_details": prompt_details
    }
    
    return results

def save_results(results, stamp):
    """Save results to JSON file in the logs directory"""
    # Extract the prefix from the stamp (e.g., "apt" from "apt_20250915_091637")
    stamp_prefix = stamp.split('_')[0]
    base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    output_file = os.path.join(base_path, "entity_coherence_prompt_results.json")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return output_file

def calculate_prompt_coherence_statistics(similarities):
    """Calculate statistics including quartiles for prompt coherence results"""
    if not similarities:
        return {
            "consecutive_comparisons": 0,
            "average_cosine_similarity": None,
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
        "consecutive_comparisons": len(similarities),
        "average_cosine_similarity": float(np.mean(similarities)),
        "min_similarity": float(min(similarities)),
        "max_similarity": float(max(similarities)),
        "std_similarity": float(np.std(similarities)),
        "q1_lower_quartile": round(q1, 4) if q1 is not None else None,
        "q2_median": round(q2_median, 4) if q2_median is not None else None,
        "q3_upper_quartile": round(q3, 4) if q3 is not None else None
    }

def update_existing_results_with_quartiles(stamp):
    """Update existing entity coherence prompt results to include quartiles without recalculating everything"""
    stamp_prefix = stamp.split('_')[0]
    base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    results_file = os.path.join(base_path, "entity_coherence_prompt_results.json")
    
    if not os.path.exists(results_file):
        print(f"No existing results file found at {results_file}")
        return False
    
    try:
        # Load existing results
        with open(results_file, 'r') as f:
            existing_data = json.load(f)
        
        print(f"Updating existing prompt coherence results with quartiles for {stamp}...")
        
        # Check if quartiles already exist
        if existing_data.get("q1_lower_quartile") is not None:
            print(f"Quartiles already exist for {stamp}, skipping update.")
            return True
        
        # Get individual similarities to recalculate statistics with quartiles
        similarities = existing_data.get("individual_similarities", [])
        
        if not similarities:
            print(f"No individual similarities found in {results_file}")
            return False
        
        # Recalculate statistics with quartiles
        updated_stats = calculate_prompt_coherence_statistics(similarities)
        
        # Update the data structure with new quartile information
        for key, value in updated_stats.items():
            existing_data[key] = value
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"Successfully updated {results_file} with quartiles")
        return True
        
    except Exception as e:
        print(f"Error updating prompt coherence results with quartiles for {stamp}: {e}")
        return False

def main():
    """Main function to calculate entity coherence at prompt level"""
    parser = argparse.ArgumentParser(description='Calculate entity coherence at prompt level using face embeddings')
    parser.add_argument('--stamp', required=True, help='Timestamp stamp like "test_file_20250910_160439"')
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
        
        # Find the image directory and mapping file
        image_dir, mapping_file = find_directories(stamp)
        print(f"Image directory: {image_dir}")
        print(f"Frame mapping file: {mapping_file}")
        
        # Load frame mappings
        frame_mappings = load_frame_mapping(mapping_file)
        print(f"Found {len(frame_mappings)} prompts in mapping file")
        
        # Calculate prompt coherence
        results = calculate_prompt_coherence(image_dir, frame_mappings)
        
        # Calculate statistics with quartiles
        stats = calculate_prompt_coherence_statistics(results["individual_similarities"])
        
        # Update results with quartile statistics
        results.update(stats)
        
        # Print summary including quartiles
        print("\n" + "="*50)
        print("ENTITY COHERENCE RESULTS (PROMPT LEVEL)")
        print("="*50)
        print(f"Total prompts: {results['total_prompts']}")
        print(f"Prompts with valid embeddings: {results['prompts_with_valid_embeddings']}")
        print(f"Consecutive comparisons: {results['consecutive_comparisons']}")
        print(f"Average cosine similarity: {results['average_cosine_similarity']:.4f}")
        print(f"Min similarity: {results['min_similarity']:.4f}")
        print(f"Max similarity: {results['max_similarity']:.4f}")
        print(f"Q1 (Lower Quartile): {results['q1_lower_quartile']:.4f}")
        print(f"Q2 (Median): {results['q2_median']:.4f}")
        print(f"Q3 (Upper Quartile): {results['q3_upper_quartile']:.4f}")
        print(f"Std similarity: {results['std_similarity']:.4f}")
        
        # Print prompt-level details
        print("\nPrompt Details:")
        for detail in results['prompt_details']:
            status = "✓" if detail['has_valid_embedding'] else "✗"
            print(f"  {status} Prompt {detail['prompt_index'] + 1}: frames {detail['start_frame']}-{detail['end_frame']}, "
                  f"{detail['successful_face_detections']}/{detail['total_frames_found']} faces detected")
        
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
