import json
import os
from datetime import datetime
from pathlib import Path

def find_prompt_similarity_summary_files():
    """Find all combined prompt similarity summary files across all datasets"""
    base_path = "/graphics/scratch2/students/webereli"
    datasets = ["apt", "jump", "sledgehammer", "thriller", "vogue", "walkthisway", "teenspirit"]
    
    summary_files = []
    
    for dataset in datasets:
        dataset_logs_path = os.path.join(base_path, dataset, "logs")
        
        if not os.path.exists(dataset_logs_path):
            print(f"Warning: Dataset logs path not found: {dataset_logs_path}")
            continue
        
        # Look for the combined summary file
        summary_file = os.path.join(dataset_logs_path, f"{dataset}_combined_prompt_similarity_summary.json")
        if os.path.exists(summary_file):
            summary_files.append({
                "dataset": dataset,
                "file_path": summary_file
            })
        else:
            print(f"Warning: Combined summary not found for {dataset}: {summary_file}")
    
    return summary_files

def extract_key_information(summary_data, dataset):
    """Extract key information from a combined prompt similarity summary file"""
    
    # Extract metadata
    metadata = summary_data.get("evaluation_metadata", {})
    mode_averages = summary_data.get("mode_averages", {})
    individual_findings = summary_data.get("individual_run_findings", [])
    
    return {
        "dataset": dataset,
        "mp3_file": metadata.get("mp3_file"),
        "total_runs_evaluated": metadata.get("total_runs_evaluated", 0),
        "last_updated": metadata.get("last_updated"),
        "mode_averages": mode_averages,
        "individual_runs": len(individual_findings),
        "individual_findings": individual_findings
    }

def calculate_overall_averages_across_datasets(all_dataset_summaries):
    """Calculate overall averages across all datasets for each generation mode and method"""
    
    # Initialize structure for overall averages
    overall_averages = {
        "dreambooth_only": {
            "all_frames_method": {"original": [], "cleaned": [], "differences": []},
            "median_frame_method": {"original": [], "cleaned": [], "differences": []}
        },
        "animatediff_and_dreambooth": {
            "all_frames_method": {"original": [], "cleaned": [], "differences": []},
            "median_frame_method": {"original": [], "cleaned": [], "differences": []}
        },
        "animatediff": {
            "all_frames_method": {"original": [], "cleaned": [], "differences": []},
            "median_frame_method": {"original": [], "cleaned": [], "differences": []}
        },
        "original": {
            "all_frames_method": {"original": [], "cleaned": [], "differences": []},
            "median_frame_method": {"original": [], "cleaned": [], "differences": []}
        }
    }
    
    # Collect all scores from all datasets
    for dataset_summary in all_dataset_summaries:
        mode_averages = dataset_summary.get("mode_averages", {})
        
        for mode, mode_data in mode_averages.items():
            if mode in overall_averages:
                # All frames method
                all_frames = mode_data.get("all_frames_method", {})
                if all_frames.get("average_original_score") is not None:
                    overall_averages[mode]["all_frames_method"]["original"].append(all_frames["average_original_score"])
                    overall_averages[mode]["all_frames_method"]["cleaned"].append(all_frames["average_cleaned_score"])
                    if all_frames.get("average_difference") is not None:
                        overall_averages[mode]["all_frames_method"]["differences"].append(all_frames["average_difference"])
                
                # Median frame method
                median_frames = mode_data.get("median_frame_method", {})
                if median_frames.get("average_original_score") is not None:
                    overall_averages[mode]["median_frame_method"]["original"].append(median_frames["average_original_score"])
                    overall_averages[mode]["median_frame_method"]["cleaned"].append(median_frames["average_cleaned_score"])
                    if median_frames.get("average_difference") is not None:
                        overall_averages[mode]["median_frame_method"]["differences"].append(median_frames["average_difference"])
    
    # Calculate statistics including quartiles for each mode and method
    final_averages = {}
    
    for mode, methods in overall_averages.items():
        final_averages[mode] = {}
        
        for method, score_types in methods.items():
            final_averages[mode][method] = {}
            
            for score_type, scores in score_types.items():
                if scores:
                    sorted_scores = sorted(scores)
                    n = len(sorted_scores)
                    
                    q1 = sorted_scores[int(n * 0.25)] if n > 0 else None
                    q2_median = sorted_scores[int(n * 0.5)] if n > 0 else None
                    q3 = sorted_scores[int(n * 0.75)] if n > 0 else None
                    
                    final_averages[mode][method][f"{score_type}_stats"] = {
                        "total_datasets": len(scores),
                        "average": round(sum(scores) / len(scores), 4),
                        "min": round(min(scores), 4),
                        "max": round(max(scores), 4),
                        "std": round(sum([(x - sum(scores)/len(scores))**2 for x in scores]) / len(scores), 4) ** 0.5,
                        "q1_lower_quartile": round(q1, 4) if q1 is not None else None,
                        "q2_median": round(q2_median, 4) if q2_median is not None else None,
                        "q3_upper_quartile": round(q3, 4) if q3 is not None else None
                    }
                else:
                    final_averages[mode][method][f"{score_type}_stats"] = {
                        "total_datasets": 0,
                        "average": None,
                        "min": None,
                        "max": None,
                        "std": None,
                        "q1_lower_quartile": None,
                        "q2_median": None,
                        "q3_upper_quartile": None
                    }
    
    return final_averages

def create_comprehensive_prompt_similarity_summary():
    """Create a comprehensive summary of all prompt similarity results"""
    print("="*80)
    print("GATHERING PROMPT SIMILARITY RESULTS")
    print("="*80)
    
    # Find all summary files
    summary_files = find_prompt_similarity_summary_files()
    print(f"Found {len(summary_files)} prompt similarity summary files")
    
    if not summary_files:
        print("No prompt similarity summary files found!")
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
    overall_averages = calculate_overall_averages_across_datasets(all_dataset_summaries)
    
    # Create comprehensive summary
    summary = {
        "summary_metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_datasets": len(all_dataset_summaries),
            "datasets_processed": [ds["dataset"] for ds in all_dataset_summaries],
            "total_runs_across_datasets": sum(ds.get("total_runs_evaluated", 0) for ds in all_dataset_summaries),
            "generation_modes": ["dreambooth_only", "animatediff_and_dreambooth", "animatediff", "original"],
            "evaluation_methods": ["all_frames_method", "median_frame_method"]
        },
        "overall_averages_across_datasets": overall_averages,
        "dataset_summaries": all_dataset_summaries
    }
    
    # Save comprehensive summary
    output_path = "/graphics/scratch2/students/webereli/prompt_similarity_results_summary.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComprehensive prompt similarity summary saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("PROMPT SIMILARITY SUMMARY STATISTICS")
    print("="*80)
    
    print(f"Total Datasets: {len(all_dataset_summaries)}")
    print(f"Total Runs Across All Datasets: {sum(ds.get('total_runs_evaluated', 0) for ds in all_dataset_summaries)}")
    
    print("\nOVERALL AVERAGES ACROSS DATASETS:")
    for mode, methods in overall_averages.items():
        print(f"  {mode.upper().replace('_', ' ')}:")
        
        for method, score_types in methods.items():
            method_name = method.replace('_', ' ').title()
            print(f"    {method_name}:")
            
            for score_type, stats in score_types.items():
                if stats["average"] is not None:
                    score_name = score_type.replace('_stats', '').replace('_', ' ').title()
                    print(f"      {score_name}: Avg={stats['average']:.4f}, Q1={stats['q1_lower_quartile']:.4f}, Q2={stats['q2_median']:.4f}, Q3={stats['q3_upper_quartile']:.4f}")
    
    print("\nDATASET BREAKDOWN:")
    for ds in all_dataset_summaries:
        print(f"  {ds['dataset'].upper()}: {ds['total_runs_evaluated']} runs, Last updated: {ds.get('last_updated', 'N/A')}")

if __name__ == "__main__":
    create_comprehensive_prompt_similarity_summary()