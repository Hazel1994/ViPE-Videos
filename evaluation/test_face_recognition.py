import json
import os
import cv2
import numpy as np
import warnings
from PIL import Image

# Configuration variables
UNIQUE_IDENTIFIER = "sksalex"
REFERENCE_IMAGE_DIR_PATH = "/graphics/scratch2/students/webereli/apt/training_images_apt/alex"  # Change this to your reference image
IMAGE_DIR_PATH = "/graphics/scratch2/students/webereli/evaluation/2025-09/ViPE"
IMAGE_STAMP_PREFIX = "20250904144032"
FRAME_TO_PROMPT_MAPPING_PATH = "/graphics/scratch2/students/webereli/evaluation/frame_to_prompt_mapping.json"
FPS = 15

# Face recognition parameters
SIMILARITY_THRESHOLD = 0.65  # Adjust this threshold according to your usecase

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

def get_face_embedding(image_path_or_array):
    """Extract face embedding from an image path or numpy array"""
    if isinstance(image_path_or_array, str):
        # It's a file path
        img = cv2.imread(image_path_or_array)
        if img is None:
            raise ValueError(f"Could not read image: {image_path_or_array}")
    else:
        # It's a numpy array (PIL Image converted)
        # Convert from RGB to BGR for OpenCV
        if image_path_or_array.max() <= 1.0:
            # Denormalize if values are between 0-1
            image_path_or_array = (image_path_or_array * 255).astype(np.uint8)
        img = cv2.cvtColor(image_path_or_array, cv2.COLOR_RGB2BGR)
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding

def compare_faces(emb1, emb2, threshold=SIMILARITY_THRESHOLD):
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

def load_reference_embeddings(reference_dir_path):
    """Load face embeddings from all PNG images in the reference directory"""
    reference_embeddings = []
    reference_files = []
    
    if not os.path.exists(reference_dir_path):
        raise ValueError(f"Reference directory does not exist: {reference_dir_path}")
    
    # Get all PNG files in the directory
    png_files = [f for f in os.listdir(reference_dir_path) if f.lower().endswith('.png')]
    
    if not png_files:
        raise ValueError(f"No PNG files found in reference directory: {reference_dir_path}")
    
    print(f"Found {len(png_files)} PNG files in reference directory")
    
    for png_file in png_files:
        file_path = os.path.join(reference_dir_path, png_file)
        try:
            print(f"Processing reference image: {png_file}")
            embedding = get_face_embedding(file_path)
            reference_embeddings.append(embedding)
            reference_files.append(png_file)
            print(f"Successfully extracted embedding from: {png_file}")
        except Exception as e:
            print(f"Warning: Could not extract face from {png_file}: {e}")
            continue
    
    if not reference_embeddings:
        raise ValueError("No valid face embeddings could be extracted from reference images")
    
    print(f"Successfully loaded {len(reference_embeddings)} reference face embeddings")
    return reference_embeddings, reference_files

def calculate_face_similarity_single(image_array, reference_embeddings):
    """Calculate face similarity between a single image and multiple reference embeddings"""
    try:
        image_embedding = get_face_embedding(image_array)
        similarities = []
        
        # Compare against each reference embedding
        for ref_embedding in reference_embeddings:
            similarity, is_match = compare_faces(ref_embedding, image_embedding)
            similarities.append(float(similarity))
        
        # Return the average similarity across all reference images
        avg_similarity = np.mean(similarities) if similarities else 0.0
        return float(avg_similarity)
    except Exception as e:
        print(f"Warning: Could not extract face from image: {e}")
        return 0.0  # Return 0 similarity if no face detected

def calculate_average_face_similarity(images, reference_embeddings):
    """Calculate average face similarity between multiple images and reference embeddings"""
    similarities = []
    for image in images:
        similarity = calculate_face_similarity_single(image, reference_embeddings)
        similarities.append(similarity)
    
    # Filter out zero similarities (failed detections) for average calculation
    valid_similarities = [s for s in similarities if s > 0]
    
    if not valid_similarities:
        avg_similarity = 0.0
        print("Warning: No valid face detections in any frames")
    else:
        avg_similarity = np.mean(valid_similarities)
    
    return round(avg_similarity, 4), similarities

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
            image_array = np.array(image)  # Keep as 0-255 uint8 for face detection
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
    """Main function to calculate face similarities for each prompt using frame mapping"""
    if not FACE_ANALYSIS_AVAILABLE:
        print("Error: Face analysis not available. Please install insightface.")
        return None
    
    # Get reference face embeddings from all images in directory
    try:
        print(f"Loading reference images from directory: {REFERENCE_IMAGE_DIR_PATH}")
        reference_embeddings, reference_files = load_reference_embeddings(REFERENCE_IMAGE_DIR_PATH)
        print(f"Reference face embeddings extracted successfully from {len(reference_files)} images:")
        for ref_file in reference_files:
            print(f"  - {ref_file}")
    except Exception as e:
        print(f"Error loading reference images: {e}")
        return None
    
    mapping_data = prepare_frame_to_prompt_mapping()
    frame_mappings = mapping_data["frame_to_prompt_mapping"]
    
    all_similarities = []
    skipped_prompts = 0
    
    for prompt_entry in frame_mappings:
        prompt_index = prompt_entry["prompt_index"]
        start_frame = prompt_entry["start_frame"]
        end_frame = prompt_entry["end_frame"]
        prompt_text = prompt_entry["prompt"]
        frame_count = prompt_entry["frame_count"]
        
        # Check if prompt contains the unique identifier
        if UNIQUE_IDENTIFIER not in prompt_text:
            print(f"Skipping prompt {prompt_index + 1}: does not contain unique identifier '{UNIQUE_IDENTIFIER}'")
            skipped_prompts += 1
            continue
        
        print(f"Processing prompt {prompt_index + 1}: frames {start_frame}-{end_frame} ({frame_count} frames)")
        
        # Load corresponding images
        images = prepare_images(start_frame, end_frame)
        
        if images is None or len(images) == 0:
            print(f"Warning: No images found for prompt {prompt_index + 1}")
            continue
        
        # Calculate average face similarity across all images for this prompt
        try:
            avg_similarity, individual_similarities = calculate_average_face_similarity(images, reference_embeddings)
            
            # Count successful face detections
            successful_detections = len([s for s in individual_similarities if s > 0])
            
            all_similarities.append({
                "prompt_index": prompt_index,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "frame_count": frame_count,
                "prompt": prompt_text,
                "avg_face_similarity": avg_similarity,
                "individual_similarities": individual_similarities,
                "num_frames_used": len(images),
                "successful_face_detections": successful_detections,
                "face_detection_rate": round(successful_detections / len(images), 4) if len(images) > 0 else 0,
                "reference_images_used": reference_files,
                "num_reference_images": len(reference_files)
            })
            print(f"Average face similarity: {avg_similarity} (using {len(images)} frames, {successful_detections} successful detections)")
        except Exception as e:
            print(f"Error calculating face similarity for prompt {prompt_index + 1}: {e}")
    
    # Calculate and print overall statistics
    if all_similarities:
        similarities_only = [s["avg_face_similarity"] for s in all_similarities]
        detection_rates = [s["face_detection_rate"] for s in all_similarities]
        
        avg_similarity = np.mean(similarities_only)
        avg_detection_rate = np.mean(detection_rates)
        
        print(f"\nOverall Results:")
        print(f"Reference images used: {len(reference_files)} images from {REFERENCE_IMAGE_DIR_PATH}")
        for ref_file in reference_files:
            print(f"  - {ref_file}")
        print(f"Total prompts: {len(frame_mappings)}")
        print(f"Prompts with unique identifier '{UNIQUE_IDENTIFIER}': {len(all_similarities)}")
        print(f"Skipped prompts (no unique identifier): {skipped_prompts}")
        print(f"Average face similarity: {avg_similarity:.4f}")
        print(f"Min face similarity: {min(similarities_only):.4f}")
        print(f"Max face similarity: {max(similarities_only):.4f}")
        print(f"Average face detection rate: {avg_detection_rate:.4f}")
        print(f"Processed {len(all_similarities)}/{len(frame_mappings)} prompts successfully")
        
        # Save results to file
        results_path = "face_similarity_results.json"
        with open(results_path, 'w') as f:
            json.dump(all_similarities, f, indent=2)
        print(f"Results saved to {results_path}")
    
    return all_similarities

if __name__ == "__main__":
    results = main()
