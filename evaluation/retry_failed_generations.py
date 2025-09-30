#!/usr/bin/env python3
"""
Retry Failed Generations Script

This script reads a generation summary JSON file, identifies failed runs,
and retries them. It tracks retry attempts to avoid infinite loops.

Usage:
python retry_failed_generations.py --summary_file /path/to/summary.json [--max_retries 2]
"""

import os
import sys
import argparse
import json
import datetime
import shutil
from pathlib import Path

# Import functions from the main wrapper
from video_generator_wrapper import (
    setup_character_files, 
    run_generate_video, 
    extract_logs_directory, 
    extract_stamp_from_logs_dir, 
    move_files
)


def parse_args():
    parser = argparse.ArgumentParser(description="Retry failed video generations")
    parser.add_argument("--summary_file", type=str, required=True,
                       help="Path to the generation summary JSON file")
    parser.add_argument("--max_retries", type=int, default=2,
                       help="Maximum number of retry attempts per job (default: 2)")
    parser.add_argument("--additional_args", type=str, default="",
                       help="Additional arguments to pass to generate_video.py")
    return parser.parse_args()


def load_summary(summary_file):
    """Load and validate the generation summary JSON file"""
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary = json.load(f)
    
    # Validate required fields
    required_fields = ["mp3_file", "saving_dir", "runs"]
    for field in required_fields:
        if field not in summary:
            raise ValueError(f"Missing required field in summary: {field}")
    
    return summary


def get_failed_runs(summary):
    """Get list of failed runs from the summary"""
    failed_runs = []
    for run in summary["runs"]:
        if not run.get("success", False):
            failed_runs.append(run)
    return failed_runs


def initialize_retry_tracking(summary):
    """Initialize retry tracking for runs if not already present"""
    for run in summary["runs"]:
        if "retry_count" not in run:
            run["retry_count"] = 0
        if "retry_history" not in run:
            run["retry_history"] = []


def can_retry(run, max_retries):
    """Check if a run can be retried based on retry count"""
    return run.get("retry_count", 0) < max_retries


def retry_run(run, summary, additional_args=""):
    """Retry a single failed run"""
    print(f"\n{'='*60}")
    print(f"Retrying run: {run['name']}")
    print(f"Description: {run['description']}")
    if run.get('character'):
        print(f"Character: {run['character']}")
    print(f"Previous attempts: {run.get('retry_count', 0)}")
    print(f"{'='*60}")
    
    # Setup character files if needed
    if run.get('character'):
        setup_character_files(summary["mp3_file"], run['character'])
    
    # Run generate_video.py
    stdout, stderr, success = run_generate_video(
        summary["saving_dir"], 
        summary["mp3_file"], 
        run["generation_mode"],
        additional_args
    )
    
    # Update retry tracking
    run["retry_count"] = run.get("retry_count", 0) + 1
    retry_entry = {
        "attempt": run["retry_count"],
        "timestamp": datetime.datetime.now().isoformat(),
        "success": success
    }
    
    if not success:
        retry_entry["error"] = "Video generation failed"
        retry_entry["stderr"] = stderr
        run["retry_history"].append(retry_entry)
        run["success"] = False
        run["error"] = f"Failed after {run['retry_count']} attempts"
        print(f"Retry failed for run: {run['name']}")
        return False
    
    # Extract logs directory and stamp
    logs_dir = extract_logs_directory(stdout)
    if not logs_dir:
        retry_entry["error"] = "Could not extract logs directory"
        run["retry_history"].append(retry_entry)
        run["success"] = False
        run["error"] = "Could not extract logs directory"
        print(f"Could not extract logs directory for run: {run['name']}")
        return False
    
    stamp = extract_stamp_from_logs_dir(logs_dir)
    print(f"Generated stamp: {stamp}")
    
    # Move files
    logs_base_dir = os.path.join(summary["saving_dir"], "logs")
    move_success = move_files(summary["saving_dir"], summary["mp3_file"], stamp, logs_base_dir)
    
    if not move_success:
        retry_entry["error"] = "Failed to move generated files"
        run["retry_history"].append(retry_entry)
        run["success"] = False
        run["error"] = "Failed to move generated files"
        print(f"Failed to move files for run: {run['name']}")
        return False
    
    # Update run with success information
    retry_entry["stamp"] = stamp
    retry_entry["logs_directory"] = logs_dir
    run["retry_history"].append(retry_entry)
    run["success"] = True
    run["stamp"] = stamp
    run["logs_directory"] = logs_dir
    
    # Clear error field if it exists
    if "error" in run:
        del run["error"]
    
    print(f"Successfully retried run: {run['name']} with stamp: {stamp}")
    return True


def save_summary(summary, summary_file):
    """Save the updated summary back to the file"""
    # Create backup of original file
    backup_file = f"{summary_file}.backup.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(summary_file, backup_file)
    print(f"Created backup: {backup_file}")
    
    # Update the summary metadata
    summary["last_retry_run"] = datetime.datetime.now().isoformat()
    
    # Save updated summary
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Updated summary saved to: {summary_file}")


def print_retry_summary(failed_runs, retried_runs, successful_retries, max_retries_exceeded):
    """Print a summary of the retry operation"""
    print(f"\n{'='*60}")
    print("RETRY SUMMARY")
    print(f"{'='*60}")
    print(f"Total failed runs found: {len(failed_runs)}")
    print(f"Runs attempted for retry: {len(retried_runs)}")
    print(f"Successful retries: {len(successful_retries)}")
    print(f"Runs exceeding max retries: {len(max_retries_exceeded)}")
    
    if successful_retries:
        print(f"\n✓ Successfully retried:")
        for run in successful_retries:
            character_info = f" (Character: {run['character']})" if run.get('character') else ""
            print(f"  - {run['name']}{character_info} -> {run.get('stamp', 'N/A')}")
    
    if max_retries_exceeded:
        print(f"\n✗ Runs that exceeded max retries:")
        for run in max_retries_exceeded:
            character_info = f" (Character: {run['character']})" if run.get('character') else ""
            print(f"  - {run['name']}{character_info} (Attempts: {run.get('retry_count', 0)})")
    
    remaining_failures = [run for run in failed_runs if run not in successful_retries and run not in max_retries_exceeded]
    if remaining_failures:
        print(f"\n⚠ Runs still failing but can be retried:")
        for run in remaining_failures:
            character_info = f" (Character: {run['character']})" if run.get('character') else ""
            print(f"  - {run['name']}{character_info} (Attempts: {run.get('retry_count', 0)})")


def main():
    args = parse_args()
    
    # Load summary
    try:
        summary = load_summary(args.summary_file)
    except Exception as e:
        print(f"Error loading summary file: {e}")
        sys.exit(1)
    
    # Initialize retry tracking
    initialize_retry_tracking(summary)
    
    # Get failed runs
    failed_runs = get_failed_runs(summary)
    
    if not failed_runs:
        print("No failed runs found in the summary file.")
        return
    
    print(f"Found {len(failed_runs)} failed runs")
    
    # Separate runs that can be retried vs those that have exceeded max retries
    retryable_runs = [run for run in failed_runs if can_retry(run, args.max_retries)]
    max_retries_exceeded = [run for run in failed_runs if not can_retry(run, args.max_retries)]
    
    if max_retries_exceeded:
        print(f"\nSkipping {len(max_retries_exceeded)} runs that have exceeded max retries ({args.max_retries})")
        for run in max_retries_exceeded:
            character_info = f" (Character: {run['character']})" if run.get('character') else ""
            print(f"  - {run['name']}{character_info} (Attempts: {run.get('retry_count', 0)})")
    
    if not retryable_runs:
        print("\nNo runs available for retry.")
        return
    
    print(f"\nWill retry {len(retryable_runs)} runs:")
    for run in retryable_runs:
        character_info = f" (Character: {run['character']})" if run.get('character') else ""
        print(f"  - {run['name']}{character_info} (Previous attempts: {run.get('retry_count', 0)})")
    
    # Retry runs
    retried_runs = []
    successful_retries = []
    
    for run in retryable_runs:
        retried_runs.append(run)
        success = retry_run(run, summary, args.additional_args)
        if success:
            successful_retries.append(run)
    
    # Save updated summary
    save_summary(summary, args.summary_file)
    
    # Print summary
    print_retry_summary(failed_runs, retried_runs, successful_retries, max_retries_exceeded)


if __name__ == "__main__":
    main()