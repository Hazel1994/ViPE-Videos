import json
import os
from datetime import datetime
from pathlib import Path

def find_coherence_results_files():
    """Find all topical_coherence_results.json files across all datasets"""
    base_path = "/graphics/scratch2/students/webereli"
    datasets = ["apt", "jump", "sledgehammer", "thriller", "vogue", "walkthisway", "teenspirit"]
    
    results_files = []
    
    for dataset in datasets:
        dataset_logs_path = os.path.join(base_path, dataset, "logs")
        
        if not os.path.exists(dataset_logs_path):
            print(f"Warning: Dataset logs path not found: {dataset_logs_path}")
            continue
        
        # Find all stamp directories in this dataset
        for item in os.listdir(dataset_logs_path):
            item_path = os.path.join(dataset_logs_path, item)
            if os.path.isdir(item_path) and item.startswith(dataset):
                # Look for topical_coherence_results.json in this stamp directory
                coherence_file = os.path.join(item_path, "topical_coherence_results.json")
                if os.path.exists(coherence_file):
                    results_files.append({
                        "dataset": dataset,
                        "stamp": item,
                        "file_path": coherence_file
                    })
    
    return results_files

def extract_key_information(coherence_data, dataset, stamp):
    """Extract key information from a coherence results file"""
    evaluation_metadata = coherence_data.get("evaluation_metadata", {})
    evaluations = coherence_data.get("evaluations", {})
    
    # Extract key metrics for each evaluation type
    summary = {
        "dataset": dataset,
        "stamp": stamp,
        "character_name": evaluation_metadata.get("character_name"),
        "evaluation_results": {}
    }
    
    for eval_type, eval_data in evaluations.items():
        if "error" not in eval_data:
            # Extract statistics including quartiles
            statistics = eval_data.get("statistics", {})
            
            summary["evaluation_results"][eval_type] = {
                "description": eval_data.get("description", ""),
                "total_prompts": eval_data.get("total_prompts", 0),
                "mean_coherence": eval_data.get("mean_coherence"),
                "statistics": {
                    "min_similarity": statistics.get("min_similarity"),
                    "max_similarity": statistics.get("max_similarity"), 
                    "std_similarity": statistics.get("std_similarity"),
                    "q1_lower_quartile": statistics.get("q1_lower_quartile"),
                    "q2_median": statistics.get("q2_median"),
                    "q3_upper_quartile": statistics.get("q3_upper_quartile")
                }
            }
        else:
            summary["evaluation_results"][eval_type] = {
                "error": eval_data.get("error"),
                "total_prompts": 0,
                "mean_coherence": None,
                "statistics": {
                    "min_similarity": None,
                    "max_similarity": None,
                    "std_similarity": None,
                    "q1_lower_quartile": None,
                    "q2_median": None,
                    "q3_upper_quartile": None
                }
            }
    
    return summary

def calculate_overall_averages(all_results):
    """Calculate overall averages across all datasets and characters"""
    eval_types = ["vipe_interpretations", "original_transcription", "gemini_with_names", "gemini_without_names", "gemini_replaced_names"]
    overall_averages = {}
    
    for eval_type in eval_types:
        coherence_scores = []
        
        for result in all_results:
            eval_data = result["evaluation_results"].get(eval_type, {})
            if eval_data.get("mean_coherence") is not None:
                coherence_scores.append(eval_data["mean_coherence"])
        
        if coherence_scores:
            # Sort scores for quartile calculations
            sorted_scores = sorted(coherence_scores)
            n = len(sorted_scores)
            
            # Calculate quartiles
            q1 = sorted_scores[int(n * 0.25)] if n > 0 else None
            q2_median = sorted_scores[int(n * 0.5)] if n > 0 else None  
            q3 = sorted_scores[int(n * 0.75)] if n > 0 else None
            
            overall_averages[eval_type] = {
                "total_evaluations": len(coherence_scores),
                "average_coherence": round(sum(coherence_scores) / len(coherence_scores), 4),
                "min_coherence": round(min(coherence_scores), 4),
                "max_coherence": round(max(coherence_scores), 4),
                "std_coherence": round(sum([(x - sum(coherence_scores)/len(coherence_scores))**2 for x in coherence_scores]) / len(coherence_scores), 4) ** 0.5,
                "q1_lower_quartile": round(q1, 4) if q1 is not None else None,
                "q2_median": round(q2_median, 4) if q2_median is not None else None,
                "q3_upper_quartile": round(q3, 4) if q3 is not None else None
            }
        else:
            overall_averages[eval_type] = {
                "total_evaluations": 0,
                "average_coherence": None,
                "min_coherence": None,
                "max_coherence": None,
                "std_coherence": None,
                "q1_lower_quartile": None,
                "q2_median": None,
                "q3_upper_quartile": None
            }
    
    return overall_averages

def calculate_dataset_averages(dataset_results):
    """Calculate average coherence scores across all characters for each dataset"""
    dataset_averages = {}
    
    for dataset, results in dataset_results.items():
        dataset_averages[dataset] = {
            "total_evaluations": len(results),
            "characters": [r["character_name"] for r in results],
            "evaluation_averages": {}
        }
        
        # Calculate averages for each evaluation type (includes original_transcription)
        eval_types = ["vipe_interpretations", "original_transcription", "gemini_with_names", "gemini_without_names", "gemini_replaced_names"]
        
        for eval_type in eval_types:
            coherence_scores = []
            valid_evaluations = 0
            
            for result in results:
                eval_data = result["evaluation_results"].get(eval_type, {})
                if eval_data.get("mean_coherence") is not None:
                    coherence_scores.append(eval_data["mean_coherence"])
                    valid_evaluations += 1
            
            if coherence_scores:
                # Sort scores for quartile calculations
                sorted_scores = sorted(coherence_scores)
                n = len(sorted_scores)
                
                # Calculate quartiles
                q1 = sorted_scores[int(n * 0.25)] if n > 0 else None
                q2_median = sorted_scores[int(n * 0.5)] if n > 0 else None
                q3 = sorted_scores[int(n * 0.75)] if n > 0 else None
                
                dataset_averages[dataset]["evaluation_averages"][eval_type] = {
                    "valid_evaluations": valid_evaluations,
                    "average_coherence": round(sum(coherence_scores) / len(coherence_scores), 4),
                    "min_coherence": round(min(coherence_scores), 4),
                    "max_coherence": round(max(coherence_scores), 4),
                    "q1_lower_quartile": round(q1, 4) if q1 is not None else None,
                    "q2_median": round(q2_median, 4) if q2_median is not None else None,
                    "q3_upper_quartile": round(q3, 4) if q3 is not None else None
                }
            else:
                dataset_averages[dataset]["evaluation_averages"][eval_type] = {
                    "valid_evaluations": 0,
                    "average_coherence": None,
                    "min_coherence": None,
                    "max_coherence": None,
                    "q1_lower_quartile": None,
                    "q2_median": None,
                    "q3_upper_quartile": None
                }
    
    return dataset_averages

def create_comprehensive_summary():
    """Create a comprehensive summary of all topical coherence results"""
    print("="*80)
    print("GATHERING TOPICAL COHERENCE RESULTS")
    print("="*80)
    
    # Find all results files
    results_files = find_coherence_results_files()
    print(f"Found {len(results_files)} coherence results files")
    
    if not results_files:
        print("No coherence results files found!")
        return
    
    # Process each results file
    all_results = []
    dataset_results = {}
    
    for file_info in results_files:
        print(f"Processing: {file_info['dataset']} - {file_info['stamp']}")
        
        try:
            with open(file_info['file_path'], 'r') as f:
                coherence_data = json.load(f)
            
            key_info = extract_key_information(coherence_data, file_info['dataset'], file_info['stamp'])
            all_results.append(key_info)
            
            # Group by dataset
            dataset = file_info['dataset']
            if dataset not in dataset_results:
                dataset_results[dataset] = []
            dataset_results[dataset].append(key_info)
            
        except Exception as e:
            print(f"Error processing {file_info['file_path']}: {e}")
            continue
    
    # Calculate averages
    dataset_averages = calculate_dataset_averages(dataset_results)
    overall_averages = calculate_overall_averages(all_results)
    
    # Create comprehensive summary
    summary = {
        "summary_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_datasets": len(dataset_results),
            "total_evaluations": len(all_results),
            "datasets_processed": list(dataset_results.keys()),
            "evaluation_types": ["vipe_interpretations", "original_transcription", "gemini_with_names", "gemini_without_names", "gemini_replaced_names"]  # ADDED original_transcription
        },
        "overall_averages": overall_averages,
        "dataset_averages": dataset_averages,
        "individual_results": all_results
    }
    
    # Save summary
    output_path = "/graphics/scratch2/students/webereli/topical_coherence_results_summary.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComprehensive summary saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("TOPICAL COHERENCE SUMMARY STATISTICS")
    print("="*80)
    
    print(f"Total Datasets: {len(dataset_results)}")
    print(f"Total Evaluations: {len(all_results)}")
    
    print("\nOVERALL AVERAGES:")
    for eval_type, stats in overall_averages.items():
        print(f"  {eval_type.replace('_', ' ').title()}:")
        if stats["average_coherence"] is not None:
            print(f"    Average Coherence: {stats['average_coherence']:.4f}")
            print(f"    Min Coherence: {stats['min_coherence']:.4f}")
            print(f"    Max Coherence: {stats['max_coherence']:.4f}")
            print(f"    Std Dev: {stats['std_coherence']:.4f}")
            print(f"    Q1 (Lower Quartile): {stats['q1_lower_quartile']:.4f}")
            print(f"    Q2 (Median): {stats['q2_median']:.4f}")
            print(f"    Q3 (Upper Quartile): {stats['q3_upper_quartile']:.4f}")
            print(f"    Total Evaluations: {stats['total_evaluations']}")
        else:
            print("    No valid evaluations")
    
    print("\nDATASET AVERAGES:")
    for dataset, averages in dataset_averages.items():
        print(f"  {dataset.upper()}:")
        print(f"    Characters: {', '.join(averages['characters'])}")
        print(f"    Total Evaluations: {averages['total_evaluations']}")
        
        for eval_type, stats in averages["evaluation_averages"].items():
            if stats["average_coherence"] is not None:
                print(f"    {eval_type.replace('_', ' ').title()}: Avg={stats['average_coherence']:.4f}, Q1={stats['q1_lower_quartile']:.4f}, Q2={stats['q2_median']:.4f}, Q3={stats['q3_upper_quartile']:.4f}")

if __name__ == "__main__":
    create_comprehensive_summary()