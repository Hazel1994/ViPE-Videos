import subprocess
import sys
import os
import json
from datetime import datetime

def run_entity_coherence_evaluation(stamp, generation_mode, coherence_type):
    """Run entity coherence evaluation for a specific stamp and type"""
    print(f"\n{'='*80}")
    print(f"STARTING ENTITY COHERENCE EVALUATION ({coherence_type.upper()})")
    print(f"STAMP: {stamp}")
    print(f"GENERATION MODE: {generation_mode}")
    print(f"{'='*80}")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if coherence_type == "prompt":
        script_name = "entity_coherence_prompt_lvl.py"
    elif coherence_type == "frame":
        script_name = "entity_coherence_frame_lvl.py"
    else:
        raise ValueError(f"Unknown coherence type: {coherence_type}")
    
    coherence_script = os.path.join(script_dir, script_name)
    
    if not os.path.exists(coherence_script):
        raise FileNotFoundError(f"{script_name} not found at: {coherence_script}")
    
    try:
        # Run the entity coherence script
        cmd = [sys.executable, coherence_script, "--stamp", stamp]
        
        print(f"Executing: {' '.join(cmd)}")
        start_time = datetime.now()
        
        result = subprocess.run(cmd, cwd=script_dir, capture_output=True, text=True)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"SUCCESS: {coherence_type} coherence for {stamp} completed in {duration}")
            return True
        else:
            print(f"FAILED: {coherence_type} coherence for {stamp} failed with return code {result.returncode}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"EXCEPTION: Error running {coherence_type} coherence evaluation for {stamp}: {e}")
        return False

def load_generation_summary(summary_file):
    """Load the generation summary file"""
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Generation summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        return json.load(f)

def filter_runs_for_coherence(generation_data):
    """Filter runs based on generation mode requirements"""
    relevant_runs = []
    
    for run in generation_data.get("runs", []):
        if not run.get("success", False):
            continue
        
        generation_mode = run.get("generation_mode", "")
        stamp = run.get("stamp", "")
        
        if not stamp:
            continue
        
        # Determine which coherence types are applicable
        applicable_types = []
        
        # Prompt-level coherence: only for animatediff modes
        if generation_mode in ["animatediff_and_dreambooth", "animatediff"]:
            applicable_types.append("prompt")
        
        # Frame-level coherence: for animatediff modes and original
        if generation_mode in ["animatediff_and_dreambooth", "animatediff", "original"]:
            applicable_types.append("frame")
        
        if applicable_types:
            relevant_runs.append({
                "stamp": stamp,
                "generation_mode": generation_mode,
                "character": run.get("character", ""),
                "name": run.get("name", ""),
                "applicable_coherence_types": applicable_types
            })
    
    return relevant_runs

def create_entity_coherence_summary(generation_data, successful_runs, output_dir):
    """Create a combined summary of all entity coherence results"""
    
    # Extract dataset name from generation_data
    mp3_file = generation_data.get("mp3_file", "unknown")
    
    # Collect all results
    all_results = []
    
    for run in successful_runs:
        stamp = run["stamp"]
        stamp_prefix = stamp.split('_')[0]
        results_dir = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
        
        run_result = {
            "stamp": stamp,
            "generation_mode": run["generation_mode"],
            "character": run["character"],
            "name": run["name"],
            "prompt_coherence": None,
            "frame_coherence": None
        }
        
        # Load prompt coherence results if available
        prompt_file = os.path.join(results_dir, "entity_coherence_prompt_results.json")
        if os.path.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                prompt_data = json.load(f)
                run_result["prompt_coherence"] = prompt_data.get("average_cosine_similarity")
        
        # Load frame coherence results if available
        frame_file = os.path.join(results_dir, "entity_coherence_frame_results.json")
        if os.path.exists(frame_file):
            with open(frame_file, 'r') as f:
                frame_data = json.load(f)
                run_result["frame_coherence"] = frame_data.get("average_cosine_similarity")
        
        all_results.append(run_result)
    
    # Create summary
    summary = {
        "summary_metadata": {
            "dataset": mp3_file,
            "generated_at": datetime.now().isoformat(),
            "total_runs_evaluated": len(all_results)
        },
        "results_by_mode": {},
        "individual_results": all_results
    }
    
    # Group by generation mode
    for mode in ["animatediff_and_dreambooth", "animatediff", "original"]:
        mode_results = [r for r in all_results if r["generation_mode"] == mode]
        
        if mode_results:
            prompt_scores = [r["prompt_coherence"] for r in mode_results if r["prompt_coherence"] is not None]
            frame_scores = [r["frame_coherence"] for r in mode_results if r["frame_coherence"] is not None]
            
            summary["results_by_mode"][mode] = {
                "total_runs": len(mode_results),
                "prompt_coherence": {
                    "available": len(prompt_scores),
                    "average": sum(prompt_scores) / len(prompt_scores) if prompt_scores else None
                },
                "frame_coherence": {
                    "available": len(frame_scores),
                    "average": sum(frame_scores) / len(frame_scores) if frame_scores else None
                }
            }
    
    # Save summary
    summary_file = os.path.join(output_dir, f"{mp3_file}_entity_coherence_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Entity coherence summary saved to: {summary_file}")
    return summary_file

def update_quartiles_for_existing_results(stamp, coherence_type):
    """Update existing results with quartiles without recalculating everything"""
    print(f"\n{'='*60}")
    print(f"UPDATING QUARTILES FOR: {stamp} - Type: {coherence_type}")
    print(f"{'='*60}")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if coherence_type == "prompt":
        script_name = "entity_coherence_prompt_lvl.py"
    elif coherence_type == "frame":
        script_name = "entity_coherence_frame_lvl.py"
    else:
        raise ValueError(f"Unknown coherence type: {coherence_type}")
    
    coherence_script = os.path.join(script_dir, script_name)
    
    if not os.path.exists(coherence_script):
        raise FileNotFoundError(f"{script_name} not found at: {coherence_script}")
    
    try:
        # Run the entity coherence script with --update-quartiles-only flag
        cmd = [sys.executable, coherence_script, "--stamp", stamp, "--update-quartiles-only"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode != 0:
            print(f"Error updating quartiles for {stamp} ({coherence_type}):")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return False
        
        print(f"Successfully updated quartiles for {stamp} ({coherence_type})")
        return True
            
    except Exception as e:
        print(f"Exception updating quartiles for {stamp} ({coherence_type}): {e}")
        return False

def main():
    """Main function to run entity coherence evaluation wrapper"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Entity Coherence Evaluation Wrapper')
    parser.add_argument('--summary-file', required=True, help='Path to generation summary JSON file')
    parser.add_argument('--force-rerun', action='store_true', help='Force re-evaluation even if results exist')
    parser.add_argument('--update-quartiles-only', action='store_true', 
                      help='Only update existing results with quartiles, do not run new evaluations')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENTITY COHERENCE EVALUATION WRAPPER")
    print("="*80)
    print(f"Summary file: {args.summary_file}")
    print(f"Force rerun: {args.force_rerun}")
    
    try:
        # Load generation data
        generation_data = load_generation_summary(args.summary_file)
        print(f"Loaded generation summary with {len(generation_data.get('runs', []))} total runs")
        
        # Filter relevant runs
        relevant_runs = filter_runs_for_coherence(generation_data)
        print(f"Found {len(relevant_runs)} runs applicable for entity coherence evaluation")
        
        if not relevant_runs:
            print("No applicable runs found for entity coherence evaluation")
            return 0
        
        # If only updating quartiles, do that and exit
        if args.update_quartiles_only:
            print(f"\nUpdating quartiles for existing entity coherence results...")
            successful_updates = 0
            failed_updates = 0
            
            for i, run in enumerate(relevant_runs, 1):
                stamp = run["stamp"]
                applicable_types = run["applicable_coherence_types"]
                
                print(f"\n[{i}/{len(relevant_runs)}] Processing run: {run['name']} ({stamp})")
                print(f"Applicable coherence types: {', '.join(applicable_types)}")
                
                for coherence_type in applicable_types:
                    print(f"  Updating {coherence_type} coherence quartiles...")
                    success = update_quartiles_for_existing_results(stamp, coherence_type)
                    if success:
                        successful_updates += 1
                    else:
                        failed_updates += 1
            
            print(f"\n{'='*80}")
            print(f"QUARTILES UPDATE COMPLETED")
            print(f"{'='*80}")
            print(f"Total evaluations processed: {successful_updates + failed_updates}")
            print(f"Successful updates: {successful_updates}")
            print(f"Failed updates: {failed_updates}")
            
            return 0 if failed_updates == 0 else 1
        
        # Process each run
        total_evaluations = 0
        successful_evaluations = 0
        failed_evaluations = 0
        
        for i, run in enumerate(relevant_runs, 1):
            stamp = run["stamp"]
            generation_mode = run["generation_mode"]
            applicable_types = run["applicable_coherence_types"]
            
            print(f"\n[{i}/{len(relevant_runs)}] Processing run: {run['name']} ({stamp})")
            print(f"Generation mode: {generation_mode}")
            print(f"Applicable coherence types: {', '.join(applicable_types)}")
            
            # Check if results already exist (unless force rerun)
            stamp_prefix = stamp.split('_')[0]
            results_dir = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
            
            for coherence_type in applicable_types:
                total_evaluations += 1
                
                if coherence_type == "prompt":
                    results_file = os.path.join(results_dir, "entity_coherence_prompt_results.json")
                else:  # frame
                    results_file = os.path.join(results_dir, "entity_coherence_frame_results.json")
                
                if os.path.exists(results_file) and not args.force_rerun:
                    print(f"  Skipping {coherence_type} coherence (results already exist): {results_file}")
                    successful_evaluations += 1
                    continue
                
                print(f"  Running {coherence_type} coherence evaluation...")
                success = run_entity_coherence_evaluation(stamp, generation_mode, coherence_type)
                
                if success:
                    successful_evaluations += 1
                else:
                    failed_evaluations += 1
        
        # Final summary
        print("\n" + "="*80)
        print("ENTITY COHERENCE EVALUATION COMPLETED")
        print("="*80)
        print(f"Total evaluations: {total_evaluations}")
        print(f"Successful evaluations: {successful_evaluations}")
        print(f"Failed evaluations: {failed_evaluations}")
        
        if failed_evaluations > 0:
            print(f"Some evaluations failed. Check logs above for details.")
            return 1
        else:
            print(f"All entity coherence evaluations completed successfully!")
            
            # Create final summary
            output_dir = os.path.dirname(args.summary_file)
            create_entity_coherence_summary(generation_data, relevant_runs, output_dir)
            
            return 0
            
    except Exception as e:
        print(f"Error in entity coherence wrapper: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)