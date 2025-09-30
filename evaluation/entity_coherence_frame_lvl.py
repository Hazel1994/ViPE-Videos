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

def find_image_directory(stamp):
    """Find the image directory based on the stamp"""
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
    
    return image_dir

def get_frame_files(image_dir, stamp):
    """Get all frame files sorted by frame number"""
    # Get all frame files in the directory (like face_identity.py does)
    frame_files = []
    for file in os.listdir(image_dir):
        if file.endswith('.png') and '_' in file:
            frame_files.append(os.path.join(image_dir, file))
    
    if not frame_files:
        raise ValueError(f"No frame files found in: {image_dir}")
    
    # Sort by the frame number (last 5 digits before .png)
    frame_files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]))
    
    print(f"Found {len(frame_files)} frame files")
    print(f"First file: {os.path.basename(frame_files[0])}")
    print(f"Last file: {os.path.basename(frame_files[-1])}")
    
    return frame_files

def calculate_frame_coherence(frame_files):
    """Calculate face embedding coherence between consecutive frames"""
    if not FACE_ANALYSIS_AVAILABLE:
        raise RuntimeError("Face analysis not available. Please install insightface.")
    
    embeddings = []
    valid_frame_indices = []
    
    # Extract embeddings from all frames
    print("Extracting face embeddings from frames...")
    for i, frame_file in enumerate(frame_files):
        try:
            embedding = get_face_embedding(frame_file)
            embeddings.append(embedding)
            valid_frame_indices.append(i)
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(frame_files)} frames")
        except Exception as e:
            print(f"Warning: Could not extract face from frame {i} ({os.path.basename(frame_file)}): {e}")
            continue
    
    print(f"Successfully extracted embeddings from {len(embeddings)} out of {len(frame_files)} frames")
    
    if len(embeddings) < 2:
        raise ValueError("Need at least 2 valid face embeddings to calculate coherence")
    
    # Calculate cosine similarities between consecutive frames
    similarities = []
    comparison_details = []
    
    print("Calculating cosine similarities between consecutive frames...")
    for i in range(len(embeddings) - 1):
        similarity = calculate_cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(similarity)
        
        frame_idx_1 = valid_frame_indices[i]
        frame_idx_2 = valid_frame_indices[i + 1]
        
        comparison_details.append({
            "frame_1_index": frame_idx_1,
            "frame_2_index": frame_idx_2,
            "frame_1_file": os.path.basename(frame_files[frame_idx_1]),
            "frame_2_file": os.path.basename(frame_files[frame_idx_2]),
            "cosine_similarity": similarity
        })
    
    # Calculate average similarity
    average_similarity = np.mean(similarities) if similarities else 0.0
    
    results = {
        "total_frames": len(frame_files),
        "valid_faces_detected": len(embeddings),
        "face_detection_rate": len(embeddings) / len(frame_files) if frame_files else 0,
        "consecutive_comparisons": len(similarities),
        "average_cosine_similarity": float(average_similarity),
        "min_similarity": float(min(similarities)) if similarities else 0.0,
        "max_similarity": float(max(similarities)) if similarities else 0.0,
        "std_similarity": float(np.std(similarities)) if similarities else 0.0,
        "individual_similarities": similarities,
        "comparison_details": comparison_details
    }
    
    return results

def save_results(results, stamp):
    """Save results to JSON file in the logs directory"""
    # Extract the prefix from the stamp (e.g., "apt" from "apt_20250915_091637")
    stamp_prefix = stamp.split('_')[0]
    base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    output_file = os.path.join(base_path, "entity_coherence_frame_results.json")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    return output_file

def calculate_frame_coherence_statistics(similarities):
    """Calculate statistics including quartiles for frame coherence results"""
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
    """Update existing entity coherence frame results to include quartiles without recalculating everything"""
    stamp_prefix = stamp.split('_')[0]
    base_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    results_file = os.path.join(base_path, "entity_coherence_frame_results.json")
    
    if not os.path.exists(results_file):
        print(f"No existing results file found at {results_file}")
        return False
    
    try:
        # Load existing results
        with open(results_file, 'r') as f:
            existing_data = json.load(f)
        
        print(f"Updating existing frame coherence results with quartiles for {stamp}...")
        
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
        updated_stats = calculate_frame_coherence_statistics(similarities)
        
        # Update the data structure with new quartile information
        for key, value in updated_stats.items():
            existing_data[key] = value
        
        # Save updated results
        with open(results_file, 'w') as f:
            json.dump(existing_data, f, indent=2)
        
        print(f"Successfully updated {results_file} with quartiles")
        return True
        
    except Exception as e:
        print(f"Error updating frame coherence results with quartiles for {stamp}: {e}")
        return False

# Update the main() function to use the new statistics calculation:
def main():
    """Main function to calculate entity coherence at frame level"""
    parser = argparse.ArgumentParser(description='Calculate entity coherence at frame level using face embeddings')
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
        
        # Find the image directory
        image_dir = find_image_directory(stamp)
        print(f"Image directory: {image_dir}")
        
        # Get frame files
        frame_files = get_frame_files(image_dir, stamp)
        
        # Calculate frame coherence
        results = calculate_frame_coherence(frame_files)
        
        # Calculate statistics with quartiles
        stats = calculate_frame_coherence_statistics(results["individual_similarities"])
        
        # Update results with quartile statistics
        results.update(stats)
        
        # Print summary including quartiles
        print("\n" + "="*50)
        print("ENTITY COHERENCE RESULTS (FRAME LEVEL)")
        print("="*50)
        print(f"Total frames: {results['total_frames']}")
        print(f"Valid faces detected: {results['valid_faces_detected']}")
        print(f"Face detection rate: {results['face_detection_rate']:.4f}")
        print(f"Consecutive comparisons: {results['consecutive_comparisons']}")
        print(f"Average cosine similarity: {results['average_cosine_similarity']:.4f}")
        print(f"Min similarity: {results['min_similarity']:.4f}")
        print(f"Max similarity: {results['max_similarity']:.4f}")
        print(f"Q1 (Lower Quartile): {results['q1_lower_quartile']:.4f}")
        print(f"Q2 (Median): {results['q2_median']:.4f}")
        print(f"Q3 (Upper Quartile): {results['q3_upper_quartile']:.4f}")
        print(f"Std similarity: {results['std_similarity']:.4f}")
        
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
