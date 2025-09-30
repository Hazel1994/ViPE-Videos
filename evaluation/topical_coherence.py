import numpy as np
import fasttext
import nltk
from typing import List
import os
import urllib.request
import zipfile
import json
import re
import argparse

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
MODEL_DIR = "/graphics/scratch2/students/webereli/topical_coherence"

os.makedirs(MODEL_DIR, exist_ok=True)
nltk.data.path.append(MODEL_DIR)
nltk.download("punkt", download_dir=MODEL_DIR)
nltk.download("punkt_tab", download_dir=MODEL_DIR)


def remove_unique_identifier(text: str, unique_identifier: str) -> str:
    """
    Remove all occurrences of the unique identifier from the text.
    
    Args:
        text: Input text to clean
        unique_identifier: The identifier to remove (case-insensitive)
        
    Returns:
        Cleaned text with the identifier removed
    """
    if not unique_identifier:
        return text
    
    # Create a pattern that matches the identifier as a whole word (case-insensitive)
    pattern = r'\b' + re.escape(unique_identifier) + r'\b'
    cleaned_text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # Clean up extra spaces that might be left behind
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def replace_unique_identifier(text: str, unique_identifier: str, replacement: str = "Man") -> str:
    """
    Replace all occurrences of the unique identifier with a replacement term.
    
    Args:
        text: Input text to modify
        unique_identifier: The identifier to replace (case-insensitive)
        replacement: The term to replace with (default: "Man")
        
    Returns:
        Text with the identifier replaced by the replacement term
    """
    if not unique_identifier:
        return text
    
    # Create a pattern that matches the identifier as a whole word (case-insensitive)
    pattern = r'\b' + re.escape(unique_identifier) + r'\b'
    cleaned_text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    
    # Clean up extra spaces that might be left behind
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def format_prompts_from_json(json_file_path: str, use_vipe: bool = False, use_original_transcription: bool = False, unique_identifier: str = "", replace_mode: str = "none") -> tuple[str, int, list]:
    """
    Load prompts from JSON file and format them as a single text for coherence measurement.
    
    Args:
        json_file_path: Path to the JSON file containing prompts
        use_vipe: If True, use vipe_interpretations.json format, else use character_generation.json format
        use_original_transcription: If True, use "text" field instead of "prompt" field from vipe_interpretations.json
        unique_identifier: The identifier to remove/replace from prompts
        replace_mode: "none" (no change), "remove" (remove identifier), "replace" (replace with "Man")
        
    Returns:
        A tuple containing:
        - A single string with all prompts concatenated, treating each prompt as a sentence
        - The number of prompts processed
        - List of individual prompts
    """
    if not os.path.exists(json_file_path):
        raise FileNotFoundError(f"JSON file not found: {json_file_path}")
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract prompts from the JSON data based on file format
    prompts = []
    
    if use_vipe or use_original_transcription:
        # vipe_interpretations.json format - prompts are in "interpretations" array
        items = data.get('interpretations', [])
        
        # Choose field based on whether we want original transcription or vipe interpretation
        field_name = "text" if use_original_transcription else "prompt"
        
        for item in items:
            if field_name in item and item[field_name]:
                # Clean up the text (remove leading/trailing whitespace)
                prompt = item[field_name].strip()
                
                # Apply identifier transformation based on mode
                if unique_identifier and replace_mode == "remove":
                    prompt = remove_unique_identifier(prompt, unique_identifier)
                elif unique_identifier and replace_mode == "replace":
                    prompt = replace_unique_identifier(prompt, unique_identifier, "Man")
                # If replace_mode == "none", leave prompt unchanged
                
                # Ensure the prompt ends with a period if it doesn't already end with punctuation
                if prompt and not prompt[-1] in '.!?':
                    prompt += '.'
                prompts.append(prompt)
    else:
        # character_generation.json format - prompts are in "updated_prompts_content" array
        items = data.get('updated_prompts_content', [])
        
        for item in items:
            if 'prompt' in item and item['prompt']:
                # Clean up the prompt text (remove leading/trailing whitespace)
                prompt = item['prompt'].strip()
                
                # Apply identifier transformation based on mode
                if unique_identifier and replace_mode == "remove":
                    prompt = remove_unique_identifier(prompt, unique_identifier)
                elif unique_identifier and replace_mode == "replace":
                    prompt = replace_unique_identifier(prompt, unique_identifier, "Man")
                # If replace_mode == "none", leave prompt unchanged
                
                # Ensure the prompt ends with a period if it doesn't already end with punctuation
                if prompt and not prompt[-1] in '.!?':
                    prompt += '.'
                prompts.append(prompt)
    
    # Join all prompts into a single text
    formatted_text = ' '.join(prompts)
    return formatted_text, len(prompts), prompts


def evaluate_coherence_version(coherence_evaluator, json_file_path: str, use_vipe: bool = False, use_original_transcription: bool = False, unique_identifier: str = "", replace_mode: str = "none"):
    """
    Evaluate coherence for a specific version.
    
    Args:
        use_original_transcription: If True, use "text" field instead of "prompt" field from vipe_interpretations.json
        replace_mode: "none" (no change), "remove" (remove identifier), "replace" (replace with "Man")
    
    Returns:
        Dictionary with evaluation results for this version
    """
    try:
        formatted_prompts, num_prompts, individual_prompts = format_prompts_from_json(
            json_file_path, use_vipe, use_original_transcription, unique_identifier, replace_mode
        )
        scores = coherence_evaluator.first_order_coherence(formatted_prompts)
        mean_score = np.nanmean(scores)
        
        # Filter out NaN values for quartile calculations
        valid_scores = [s for s in scores if not np.isnan(s)]
        
        # Calculate quartiles if we have valid scores
        q1, q2_median, q3 = None, None, None
        if valid_scores:
            sorted_scores = sorted(valid_scores)
            n = len(sorted_scores)
            q1 = sorted_scores[int(n * 0.25)] if n > 0 else None
            q2_median = sorted_scores[int(n * 0.5)] if n > 0 else None
            q3 = sorted_scores[int(n * 0.75)] if n > 0 else None
        
        return {
            "total_prompts": num_prompts,
            "total_similarities": len(scores),
            "mean_coherence": float(mean_score) if not np.isnan(mean_score) else None,
            "coherence_similarities": [float(sim) if not np.isnan(sim) else None for sim in scores],
            "prompts": individual_prompts,
            "statistics": {
                "min_similarity": float(np.nanmin(scores)) if len(scores) > 0 and not np.isnan(np.nanmin(scores)) else None,
                "max_similarity": float(np.nanmax(scores)) if len(scores) > 0 and not np.isnan(np.nanmax(scores)) else None,
                "std_similarity": float(np.nanstd(scores)) if len(scores) > 0 and not np.isnan(np.nanstd(scores)) else None,
                "q1_lower_quartile": float(q1) if q1 is not None else None,
                "q2_median": float(q2_median) if q2_median is not None else None,
                "q3_upper_quartile": float(q3) if q3 is not None else None
            }
        }
    except Exception as e:
        print(f"Error evaluating version: {e}")
        return {
            "error": str(e),
            "total_prompts": 0,
            "total_similarities": 0,
            "mean_coherence": None,
            "coherence_similarities": [],
            "prompts": [],
            "statistics": {
                "min_similarity": None,
                "max_similarity": None,
                "std_similarity": None,
                "q1_lower_quartile": None,
                "q2_median": None,
                "q3_upper_quartile": None
            }
        }


def save_all_coherence_results(vipe_results, original_transcription_results, gemini_with_names, gemini_without_names, gemini_replaced_names, output_path: str, stamp: str, name: str):
    """
    Save all five coherence evaluation results to a single JSON file.
    """
    results = {
        "evaluation_metadata": {
            "evaluation_type": "topical_coherence",
            "stamp": stamp,
            "character_name": name,
            "generated_at": "2025-09-23T12:00:00"  # You can use datetime.now().isoformat() if needed
        },
        "evaluations": {
            "vipe_interpretations": {
                "description": "Coherence evaluation using ViPE interpretations (no Gemini processing)",
                "source_file": "vipe_interpretations.json",
                "source_field": "prompt",
                **vipe_results
            },
            "original_transcription": {
                "description": "Coherence evaluation using original transcription text",
                "source_file": "vipe_interpretations.json",
                "source_field": "text",
                **original_transcription_results
            },
            "gemini_with_names": {
                "description": "Coherence evaluation using Gemini-processed prompts with character names",
                "source_file": "gemini/character_generation.json",
                "character_names_removed": False,
                **gemini_with_names
            },
            "gemini_without_names": {
                "description": "Coherence evaluation using Gemini-processed prompts with character names removed",
                "source_file": "gemini/character_generation.json", 
                "character_names_removed": True,
                **gemini_without_names
            },
            "gemini_replaced_names": {
                "description": "Coherence evaluation using Gemini-processed prompts with character names replaced by 'Man'",
                "source_file": "gemini/character_generation.json", 
                "character_names_replaced": True,
                "replacement_term": "Man",
                **gemini_replaced_names
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"All coherence results saved to: {output_path}")


class FirstOrderCoherence:
    def __init__(self, model_dir: str = "./fasttext_models"):
        """
        Initialize coherence evaluator.

        Args:
            model_dir: Directory where the fastText model will be stored.
        """
        os.makedirs(model_dir, exist_ok=True)
        model_bin_path = os.path.join(model_dir, "crawl-300d-2M-subword.bin")
        zip_path = os.path.join(model_dir, "crawl-300d-2M-subword.zip")

        # Download and extract model if not available
        if not os.path.exists(model_bin_path):
            if not os.path.exists(zip_path):
                print("Downloading fastText model... This may take a while (~4GB).")
                urllib.request.urlretrieve(FASTTEXT_URL, zip_path)
                print("Download complete.")
            else:
                print("Found cached zip file, skipping download.")

            print("Extracting model...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(model_dir)
            print("Extraction complete.")
        else:
            print("Model already available, skipping download and extraction.")

        self.model = fasttext.load_model(model_bin_path)

    def sentence_embedding(self, sentence: str) -> np.ndarray:
        """Get the average fastText embedding for a sentence."""
        return self.model.get_sentence_vector(sentence)

    def first_order_coherence(self, text: str) -> List[float]:
        """
        Compute first-order coherence (cosine similarity between consecutive sentences).

        Args:
            text: Input text.

        Returns:
            List of cosine similarities between consecutive sentence embeddings.
        """
        sentences = nltk.sent_tokenize(text, language="english")
        if len(sentences) < 2:
            return [np.nan]

        embeddings = [self.sentence_embedding(sent) for sent in sentences]

        similarities = []
        for i in range(len(embeddings) - 1):
            v1, v2 = embeddings[i], embeddings[i + 1]
            sim = self.cosine_similarity(v1, v2)
            similarities.append(sim)
        return similarities

    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return np.nan
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Evaluate topical coherence of prompts (all five versions)')
    parser.add_argument('--stamp', type=str, required=True, help='Timestamp or identifier for the evaluation')
    parser.add_argument('--name', type=str, required=True, help='Character name identifier for the evaluation')

    args = parser.parse_args()
    
    # Use the provided arguments
    stamp = args.stamp
    name = args.name
    
    # Extract the prefix from the stamp (e.g., "apt" from "apt_20250915_091637")
    stamp_prefix = stamp.split('_')[0]
    
    # Define output path for results
    results_dir = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
    output_path = os.path.join(results_dir, "topical_coherence_results.json")

    # Define file paths
    vipe_file_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}/vipe_interpretations.json"
    gemini_file_path = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}/gemini/character_generation.json"

    print(f"Running all coherence evaluations for stamp: {stamp}, character: {name}")
    print(f"ViPE file: {vipe_file_path}")
    print(f"Gemini file: {gemini_file_path}")
    
    # Ensure output directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize coherence evaluator (only once for all evaluations)
    coherence = FirstOrderCoherence(model_dir=MODEL_DIR)
    
    print("\n" + "="*60)
    print("1. Evaluating ViPE interpretations...")
    print("="*60)
    vipe_results = evaluate_coherence_version(coherence, vipe_file_path, use_vipe=True, use_original_transcription=False, unique_identifier="", replace_mode="none")
    if "error" not in vipe_results:
        print(f"ViPE - Mean coherence: {vipe_results['mean_coherence']}")
        print(f"ViPE - Processed {vipe_results['total_prompts']} prompts")
    
    print("\n" + "="*60)
    print("2. Evaluating Original Transcription...")
    print("="*60)
    original_transcription_results = evaluate_coherence_version(coherence, vipe_file_path, use_vipe=False, use_original_transcription=True, unique_identifier="", replace_mode="none")
    if "error" not in original_transcription_results:
        print(f"Original Transcription - Mean coherence: {original_transcription_results['mean_coherence']}")
        print(f"Original Transcription - Processed {original_transcription_results['total_prompts']} prompts")
    
    print("\n" + "="*60)
    print("3. Evaluating Gemini prompts WITH character names...")
    print("="*60)
    gemini_with_names = evaluate_coherence_version(coherence, gemini_file_path, use_vipe=False, use_original_transcription=False, unique_identifier="", replace_mode="none")
    if "error" not in gemini_with_names:
        print(f"Gemini (with names) - Mean coherence: {gemini_with_names['mean_coherence']}")
        print(f"Gemini (with names) - Processed {gemini_with_names['total_prompts']} prompts")
    
    print("\n" + "="*60)
    print("4. Evaluating Gemini prompts WITHOUT character names...")
    print("="*60)
    gemini_without_names = evaluate_coherence_version(coherence, gemini_file_path, use_vipe=False, use_original_transcription=False, unique_identifier=name, replace_mode="remove")
    if "error" not in gemini_without_names:
        print(f"Gemini (without names) - Mean coherence: {gemini_without_names['mean_coherence']}")
        print(f"Gemini (without names) - Processed {gemini_without_names['total_prompts']} prompts")
    
    print("\n" + "="*60)
    print("5. Evaluating Gemini prompts with character names replaced by 'Man'...")
    print("="*60)
    gemini_replaced_names = evaluate_coherence_version(coherence, gemini_file_path, use_vipe=False, use_original_transcription=False, unique_identifier=name, replace_mode="replace")
    if "error" not in gemini_replaced_names:
        print(f"Gemini (names → 'Man') - Mean coherence: {gemini_replaced_names['mean_coherence']}")
        print(f"Gemini (names → 'Man') - Processed {gemini_replaced_names['total_prompts']} prompts")
    
    # Save all results to a single JSON file
    save_all_coherence_results(vipe_results, original_transcription_results, gemini_with_names, gemini_without_names, gemini_replaced_names, output_path, stamp, name)
    
    print("\n" + "="*60)
    print("COHERENCE EVALUATION SUMMARY")
    print("="*60)
    print(f"ViPE Mean Coherence: {vipe_results.get('mean_coherence', 'N/A')}")
    print(f"Original Transcription Mean Coherence: {original_transcription_results.get('mean_coherence', 'N/A')}")
    print(f"Gemini (with names) Mean Coherence: {gemini_with_names.get('mean_coherence', 'N/A')}")
    print(f"Gemini (without names) Mean Coherence: {gemini_without_names.get('mean_coherence', 'N/A')}")
    print(f"Gemini (names → 'Man') Mean Coherence: {gemini_replaced_names.get('mean_coherence', 'N/A')}")
    print(f"Results saved to: {output_path}")