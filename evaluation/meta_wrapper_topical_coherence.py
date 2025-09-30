import subprocess
import sys
import os
import json
from datetime import datetime

def run_topical_coherence_evaluation(dataset_name, character_id, character_name):
    """Run the topical coherence evaluation for a specific dataset and character"""
    print(f"\n{'='*80}")
    print(f"STARTING TOPICAL COHERENCE EVALUATION FOR: {dataset_name.upper()}")
    print(f"CHARACTER ID: {character_id} (Name: {character_name})")
    print(f"{'='*80}")
    
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    coherence_script = os.path.join(script_dir, "topical_coherence.py")
    
    if not os.path.exists(coherence_script):
        raise FileNotFoundError(f"topical_coherence.py not found at: {coherence_script}")
    
    # Find a valid stamp for this dataset by looking for generation summary
    summary_file = f"/graphics/scratch2/students/webereli/{dataset_name}/logs/{dataset_name}_generation_summary.json"
    
    if not os.path.exists(summary_file):
        print(f"WARNING: Summary file not found: {summary_file}")
        print(f"Skipping {dataset_name} - {character_id}")
        return False
    
    # Load the generation summary to find a valid stamp
    try:
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        # Find a successful run with the specified character_id
        valid_stamp = None
        for run in summary_data.get("runs", []):
            if (run.get("success") == True and 
                run.get("character") == character_id):
                valid_stamp = run.get("stamp")
                break
        
        if not valid_stamp:
            print(f"WARNING: No successful run found for character '{character_id}' in {dataset_name}")
            print(f"Skipping {dataset_name} - {character_id}")
            return False
            
    except Exception as e:
        print(f"ERROR: Could not read generation summary for {dataset_name}: {e}")
        return False
    
    try:
        # Run the topical coherence script with the actual character name (not character_id)
        cmd = [sys.executable, coherence_script, "--stamp", valid_stamp, "--name", character_name]
        
        print(f"Executing: {' '.join(cmd)}")
        start_time = datetime.now()
        
        result = subprocess.run(cmd, cwd=script_dir)
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"\nSUCCESS: {dataset_name} - {character_id} ({character_name}) completed in {duration}")
            return True
        else:
            print(f"\nFAILED: {dataset_name} - {character_id} ({character_name}) failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\nEXCEPTION: Error running topical coherence evaluation for {dataset_name} - {character_id}: {e}")
        return False

def get_character_name(character_id):
    """Get the actual character name for a character_id"""
    # Define character mappings
    character_mappings = {
        "character_1": "Alex",
        "character_2": "Alex", 
        "character_3": "Elara",
        "character_4": "Elara",
        "character_5": "Jordan"
    }
    
    return character_mappings.get(character_id, "Unknown")

def get_all_characters_for_dataset(dataset_name):
    """Get all characters that exist for a given dataset"""
    summary_file = f"/graphics/scratch2/students/webereli/{dataset_name}/logs/{dataset_name}_generation_summary.json"
    
    if not os.path.exists(summary_file):
        return []
    
    try:
        with open(summary_file, 'r') as f:
            summary_data = json.load(f)
        
        # Find all unique characters that have successful runs
        characters = set()
        for run in summary_data.get("runs", []):
            if run.get("success") == True and run.get("character"):
                characters.add(run.get("character"))
        
        return sorted(list(characters))
        
    except Exception as e:
        print(f"ERROR: Could not read generation summary for {dataset_name}: {e}")
        return []

def main():
    """Main function to run topical coherence evaluation for all datasets and characters"""
    print("="*80)
    print("META TOPICAL COHERENCE WRAPPER - PROCESSING ALL DATASETS & CHARACTERS")
    print("="*80)
    
    # List of all datasets to process
    datasets = [
        "apt",
        "jump", 
        "sledgehammer",
        "thriller",
        "vogue",
        "walkthisway",
        "teenspirit"
    ]
    
    start_time = datetime.now()
    successful_evaluations = []
    failed_evaluations = []
    
    print(f"Processing {len(datasets)} datasets: {', '.join(datasets)}")
    print(f"Started at: {start_time}")
    
    # Process each dataset
    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}] Processing dataset: {dataset}")
        
        # Get all characters for this dataset
        characters = get_all_characters_for_dataset(dataset)
        
        if not characters:
            print(f"No characters found for {dataset}, skipping...")
            failed_evaluations.append(f"{dataset} (no characters)")
            continue
        
        print(f"Found characters for {dataset}: {characters}")
        
        # Process each character in this dataset
        for j, character_id in enumerate(characters, 1):
            character_name = get_character_name(character_id)
            print(f"\n  [{j}/{len(characters)}] Processing: {dataset} - {character_id} ({character_name})")
            
            success = run_topical_coherence_evaluation(dataset, character_id, character_name)
            
            evaluation_key = f"{dataset} - {character_id} ({character_name})"
            if success:
                successful_evaluations.append(evaluation_key)
            else:
                failed_evaluations.append(evaluation_key)
    
    # Final summary
    end_time = datetime.now()
    total_duration = end_time - start_time
    
    print("\n" + "="*80)
    print("META TOPICAL COHERENCE EVALUATION COMPLETED")
    print("="*80)
    print(f"Total processing time: {total_duration}")
    print(f"Started: {start_time}")
    print(f"Finished: {end_time}")
    
    print(f"\nSUCCESSFUL ({len(successful_evaluations)} evaluations):")
    for evaluation in successful_evaluations:
        print(f"  - {evaluation}")
    
    if failed_evaluations:
        print(f"\nFAILED ({len(failed_evaluations)} evaluations):")
        for evaluation in failed_evaluations:
            print(f"  - {evaluation}")
    else:
        print(f"\nAll evaluations completed successfully!")
    
    print(f"\nResults saved to individual coherence results files:")
    for evaluation in successful_evaluations:
        dataset = evaluation.split(" - ")[0]
        results_file = f"/graphics/scratch2/students/webereli/{dataset}/logs/[STAMP]/topical_coherence_results.json"
        print(f"  - {results_file}")
    
    # Return exit code based on results
    if failed_evaluations:
        print(f"\nSome evaluations failed. Check logs above for details.")
        return 1
    else:
        print(f"\nAll topical coherence evaluations completed successfully!")
        return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)