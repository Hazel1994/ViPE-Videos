import json
import os
from datetime import datetime
from pathlib import Path

def find_entity_coherence_summary_files():
    """Find all entity coherence summary files across all datasets"""
    base_path = "/graphics/scratch2/students/webereli"
    datasets = ["apt", "jump", "sledgehammer", "thriller", "vogue", "walkthisway", "teenspirit"]
    
    summary_files = []
    
    for dataset in datasets:
        dataset_logs_path = os.path.join(base_path, dataset, "logs")
        
        if not os.path.exists(dataset_logs_path):
            print(f"Warning: Dataset logs path not found: {dataset_logs_path}")
            continue
        
        # Look for the entity coherence summary file
        summary_file = os.path.join(dataset_logs_path, f"{dataset}_entity_coherence_summary.json")
        if os.path.exists(summary_file):
            summary_files.append({
                "dataset": dataset,
                "file_path": summary_file
            })
        else:
            print(f"Warning: Entity coherence summary not found for {dataset}: {summary_file}")
    
    return summary_files

def extract_key_information(summary_data, dataset):
    """Extract key information from an entity coherence summary file"""
    
    # Extract metadata
    metadata = summary_data.get("summary_metadata", {})
    results_by_mode = summary_data.get("results_by_mode", {})
    individual_results = summary_data.get("individual_results", [])
    
    return {
        "dataset": dataset,
        "mp3_file": metadata.get("dataset"),
        "total_runs_evaluated": metadata.get("total_runs_evaluated", 0),
        "generated_at": metadata.get("generated_at"),
        "results_by_mode": results_by_mode,
        "individual_runs": len(individual_results),
        "individual_results": individual_results
    }

def calculate_overall_averages_across_datasets(all_dataset_summaries):
    """Calculate overall averages across all datasets for each generation mode and coherence type"""
    
    # Initialize structure for overall averages
    overall_averages = {
        "animatediff_and_dreambooth": {
            "prompt_coherence": [],
            "frame_coherence": []
        },
        "animatediff": {
            "prompt_coherence": [],
            "frame_coherence": []
        },
        "original": {
            "frame_coherence": []
        }
    }
    
    # Collect all scores from all datasets
    for dataset_summary in all_dataset_summaries:
        results_by_mode = dataset_summary.get("results_by_mode", {})
        
        for mode, mode_data in results_by_mode.items():
            if mode in overall_averages:
                # Prompt coherence (only for animatediff modes)
                if "prompt_coherence" in overall_averages[mode]:
                    prompt_coherence = mode_data.get("prompt_coherence", {})
                    if prompt_coherence.get("average") is not None:
                        overall_averages[mode]["prompt_coherence"].append(prompt_coherence["average"])
                
                # Frame coherence
                if "frame_coherence" in overall_averages[mode]:
                    frame_coherence = mode_data.get("frame_coherence", {})
                    if frame_coherence.get("average") is not None:
                        overall_averages[mode]["frame_coherence"].append(frame_coherence["average"])
    
    # Calculate statistics including quartiles for each mode and coherence type
    final_averages = {}
    
    for mode, coherence_types in overall_averages.items():
        final_averages[mode] = {}
        
        for coherence_type, scores in coherence_types.items():
            if scores:
                sorted_scores = sorted(scores)
                n = len(sorted_scores)
                
                q1 = sorted_scores[int(n * 0.25)] if n > 0 else None
                q2_median = sorted_scores[int(n * 0.5)] if n > 0 else None
                q3 = sorted_scores[int(n * 0.75)] if n > 0 else None
                
                final_averages[mode][coherence_type] = {
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
                final_averages[mode][coherence_type] = {
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

def create_comprehensive_entity_coherence_summary():
    """Create a comprehensive summary of all entity coherence results"""
    print("="*80)
    print("GATHERING ENTITY COHERENCE RESULTS")
    print("="*80)
    
    # Find all summary files
    summary_files = find_entity_coherence_summary_files()
    print(f"Found {len(summary_files)} entity coherence summary files")
    
    if not summary_files:
        print("No entity coherence summary files found!")
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
            "generation_modes": ["animatediff_and_dreambooth", "animatediff", "original"],
            "coherence_types": ["prompt_coherence", "frame_coherence"]
        },
        "overall_averages_across_datasets": overall_averages,
        "dataset_summaries": all_dataset_summaries
    }
    
    # Save comprehensive summary
    output_path = "/graphics/scratch2/students/webereli/entity_coherence_results_summary.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nComprehensive entity coherence summary saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("ENTITY COHERENCE SUMMARY STATISTICS")
    print("="*80)
    
    print(f"Total Datasets: {len(all_dataset_summaries)}")
    print(f"Total Runs Across All Datasets: {sum(ds.get('total_runs_evaluated', 0) for ds in all_dataset_summaries)}")
    
    print("\nOVERALL AVERAGES ACROSS DATASETS:")
    for mode, coherence_types in overall_averages.items():
        print(f"  {mode.upper().replace('_', ' ')}:")
        
        for coherence_type, stats in coherence_types.items():
            coherence_name = coherence_type.replace('_', ' ').title()
            print(f"    {coherence_name}:")
            
            if stats["average"] is not None:
                print(f"      Average: {stats['average']:.4f}")
                print(f"      Min: {stats['min']:.4f}")
                print(f"      Max: {stats['max']:.4f}")
                print(f"      Q1 (Lower Quartile): {stats['q1_lower_quartile']:.4f}")
                print(f"      Q2 (Median): {stats['q2_median']:.4f}")
                print(f"      Q3 (Upper Quartile): {stats['q3_upper_quartile']:.4f}")
                print(f"      Total Datasets: {stats['total_datasets']}")
            else:
                print("      No valid evaluations")
    
    print("\nDATASET BREAKDOWN:")
    for ds in all_dataset_summaries:
        print(f"  {ds['dataset'].upper()}: {ds['total_runs_evaluated']} runs, Generated at: {ds.get('generated_at', 'N/A')}")

if __name__ == "__main__":
    create_comprehensive_entity_coherence_summary()