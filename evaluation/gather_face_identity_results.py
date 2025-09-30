import json
import os
from datetime import datetime
from pathlib import Path

def find_face_identity_summary_files():
    """Find all face identity summary files across all datasets"""
    base_path = "/graphics/scratch2/students/webereli"
    datasets = ["apt", "jump", "sledgehammer", "thriller", "vogue", "walkthisway", "teenspirit"]
    
    summary_files = []
    
    for dataset in datasets:
        dataset_logs_path = os.path.join(base_path, dataset, "logs")
        
        if not os.path.exists(dataset_logs_path):
            print(f"Warning: Dataset logs path not found: {dataset_logs_path}")
            continue
        
        # Look for the face identity summary file
        summary_file = os.path.join(dataset_logs_path, f"{dataset}_combined_face_identity_summary.json")
        if os.path.exists(summary_file):
            summary_files.append({
                "dataset": dataset,
                "file_path": summary_file
            })
        else:
            print(f"Warning: Face identity summary not found for {dataset}: {summary_file}")
    
    return summary_files

def extract_key_information(summary_data, dataset):
    """Extract key information from a face identity summary file"""
    
    combined_results = summary_data.get("combined_face_identity_results", [])
    
    # Group results by generation mode and character
    results_by_mode = {}
    results_by_character = {}
    
    for result in combined_results:
        mode = result.get("generation_mode")
        character = result.get("character")
        face_result = result.get("face_identity_result", {})
        
        # Initialize mode if not exists
        if mode not in results_by_mode:
            results_by_mode[mode] = {
                "target_similarities": [],
                "baseline_similarities": [],
                "runs": []
            }
        
        # Initialize character if not exists
        if character not in results_by_character:
            results_by_character[character] = {
                "target_similarities": [],
                "baseline_similarities": [],
                "runs": []
            }
        
        # Extract target and baseline statistics
        target_avg = face_result.get("target_average_face_similarity")
        baseline_avg = face_result.get("baseline_average_face_similarity")
        
        if target_avg is not None:
            results_by_mode[mode]["target_similarities"].append(target_avg)
            results_by_character[character]["target_similarities"].append(target_avg)
        
        if baseline_avg is not None:
            results_by_mode[mode]["baseline_similarities"].append(baseline_avg)
            results_by_character[character]["baseline_similarities"].append(baseline_avg)
        
        results_by_mode[mode]["runs"].append(result)
        results_by_character[character]["runs"].append(result)
    
    return {
        "dataset": dataset,
        "total_runs_evaluated": len(combined_results),
        "results_by_mode": results_by_mode,
        "results_by_character": results_by_character,
        "individual_results": combined_results
    }

def calculate_statistics(values):
    """Calculate statistics including quartiles for a list of values"""
    if not values:
        return {
            "total_runs": 0,
            "average": None,
            "min": None,
            "max": None,
            "std": None,
            "q1_lower_quartile": None,
            "q2_median": None,
            "q3_upper_quartile": None
        }
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    q1 = sorted_values[int(n * 0.25)] if n > 0 else None
    q2_median = sorted_values[int(n * 0.5)] if n > 0 else None
    q3 = sorted_values[int(n * 0.75)] if n > 0 else None
    
    mean = sum(values) / len(values)
    std = (sum([(x - mean)**2 for x in values]) / len(values)) ** 0.5
    
    return {
        "total_runs": len(values),
        "average": round(mean, 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "std": round(std, 4),
        "q1_lower_quartile": round(q1, 4) if q1 is not None else None,
        "q2_median": round(q2_median, 4) if q2_median is not None else None,
        "q3_upper_quartile": round(q3, 4) if q3 is not None else None
    }

def calculate_overall_averages_across_datasets(all_dataset_summaries):
    """Calculate overall averages across all datasets for each generation mode and similarity type"""
    
    # Initialize structure for overall averages
    overall_averages = {
        "animatediff_and_dreambooth": {
            "target_similarities": [],
            "baseline_similarities": []
        },
        "dreambooth_only": {
            "target_similarities": [],
            "baseline_similarities": []
        }
    }
    
    overall_by_character = {
        "character_1": {"target_similarities": [], "baseline_similarities": []},
        "character_2": {"target_similarities": [], "baseline_similarities": []},
        "character_3": {"target_similarities": [], "baseline_similarities": []},
        "character_4": {"target_similarities": [], "baseline_similarities": []},
        "character_5": {"target_similarities": [], "baseline_similarities": []}
    }
    
    # Collect all scores from all datasets
    for dataset_summary in all_dataset_summaries:
        results_by_mode = dataset_summary.get("results_by_mode", {})
        results_by_character = dataset_summary.get("results_by_character", {})
        
        # Collect by mode
        for mode, mode_data in results_by_mode.items():
            if mode in overall_averages:
                overall_averages[mode]["target_similarities"].extend(mode_data.get("target_similarities", []))
                overall_averages[mode]["baseline_similarities"].extend(mode_data.get("baseline_similarities", []))
        
        # Collect by character
        for character, char_data in results_by_character.items():
            if character in overall_by_character:
                overall_by_character[character]["target_similarities"].extend(char_data.get("target_similarities", []))
                overall_by_character[character]["baseline_similarities"].extend(char_data.get("baseline_similarities", []))
    
    # Calculate statistics for modes
    final_averages_by_mode = {}
    for mode, similarity_types in overall_averages.items():
        final_averages_by_mode[mode] = {}
        for similarity_type, scores in similarity_types.items():
            final_averages_by_mode[mode][similarity_type] = calculate_statistics(scores)
    
    # Calculate statistics for characters
    final_averages_by_character = {}
    for character, similarity_types in overall_by_character.items():
        final_averages_by_character[character] = {}
        for similarity_type, scores in similarity_types.items():
            final_averages_by_character[character][similarity_type] = calculate_statistics(scores)
    
    return final_averages_by_mode, final_averages_by_character

def create_comprehensive_face_identity_summary():
    """Create a comprehensive summary of all face identity results"""
    print("="*80)
    print("GATHERING FACE IDENTITY RESULTS")
    print("="*80)
    
    # Find all summary files
    summary_files = find_face_identity_summary_files()
    print(f"Found {len(summary_files)} face identity summary files")
    
    if not summary_files:
        print("No face identity summary files found!")
        return
    
    # Process each summary file
    all_dataset_summaries = []
    
    for file_info in summary_files:
        print(f"Processing: {file_info['dataset']}")
        
        try:
            with open(file_info['file_path'], 'r') as f:
                summary_data = json.load(f)
            
            key_info = extract_key_information(summary_data, file_info['dataset'])
            all_dataset_summaries.append(key_info)
            
        except Exception as e:
            print(f"Error processing {file_info['file_path']}: {e}")
            continue
    
    # Calculate overall averages across all datasets
    overall_averages_by_mode, overall_averages_by_character = calculate_overall_averages_across_datasets(all_dataset_summaries)
    
    # Create comprehensive summary
    summary = {
        "summary_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_datasets": len(all_dataset_summaries),
            "datasets_processed": [ds["dataset"] for ds in all_dataset_summaries],
            "total_runs_across_datasets": sum(ds.get("total_runs_evaluated", 0) for ds in all_dataset_summaries),
            "generation_modes": ["animatediff_and_dreambooth", "dreambooth_only"],
            "similarity_types": ["target_similarities", "baseline_similarities"],
            "characters": ["character_1", "character_2", "character_3", "character_4", "character_5"]
        },
        "overall_averages_by_mode": overall_averages_by_mode,
        "overall_averages_by_character": overall_averages_by_character,
        "dataset_summaries": all_dataset_summaries
    }
    
    # Save comprehensive summary
    output_path = "/graphics/scratch2/students/webereli/face_identity_results_summary.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComprehensive face identity summary saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("FACE IDENTITY SUMMARY STATISTICS")
    print("="*80)
    
    print(f"Total Datasets: {len(all_dataset_summaries)}")
    print(f"Total Runs Across All Datasets: {sum(ds.get('total_runs_evaluated', 0) for ds in all_dataset_summaries)}")
    
    print("\nOVERALL AVERAGES BY GENERATION MODE:")
    for mode, similarity_types in overall_averages_by_mode.items():
        print(f"  {mode.upper().replace('_', ' ')}:")
        
        for similarity_type, stats in similarity_types.items():
            similarity_name = similarity_type.replace('_', ' ').title()
            print(f"    {similarity_name}:")
            
            if stats["average"] is not None:
                print(f"      Average: {stats['average']:.4f}")
                print(f"      Min: {stats['min']:.4f}")
                print(f"      Max: {stats['max']:.4f}")
                print(f"      Q1 (Lower Quartile): {stats['q1_lower_quartile']:.4f}")
                print(f"      Q2 (Median): {stats['q2_median']:.4f}")
                print(f"      Q3 (Upper Quartile): {stats['q3_upper_quartile']:.4f}")
                print(f"      Total Runs: {stats['total_runs']}")
            else:
                print("      No valid evaluations")
    
    print("\nOVERALL AVERAGES BY CHARACTER:")
    for character, similarity_types in overall_averages_by_character.items():
        print(f"  {character.upper()}:")
        
        for similarity_type, stats in similarity_types.items():
            similarity_name = similarity_type.replace('_', ' ').title()
            print(f"    {similarity_name}:")
            
            if stats["average"] is not None:
                print(f"      Average: {stats['average']:.4f}")
                print(f"      Min: {stats['min']:.4f}")
                print(f"      Max: {stats['max']:.4f}")
                print(f"      Q1 (Lower Quartile): {stats['q1_lower_quartile']:.4f}")
                print(f"      Q2 (Median): {stats['q2_median']:.4f}")
                print(f"      Q3 (Upper Quartile): {stats['q3_upper_quartile']:.4f}")
                print(f"      Total Runs: {stats['total_runs']}")
            else:
                print("      No valid evaluations")
    
    print("\nDATASET BREAKDOWN:")
    for ds in all_dataset_summaries:
        print(f"  {ds['dataset'].upper()}: {ds['total_runs_evaluated']} runs")

if __name__ == "__main__":
    create_comprehensive_face_identity_summary()