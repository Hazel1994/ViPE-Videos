import os
import json
import subprocess
import argparse

def run_face_identity(stamp, character):
    """Run face_identity.py for a given stamp and character, return parsed results or None."""
    cmd = [
        "python3",
        "face_identity.py",
        "--stamp", stamp,
        "--character", character
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Try to find the output file
        stamp_prefix = stamp.split('_')[0]
        output_dir = f"/graphics/scratch2/students/webereli/{stamp_prefix}/logs/{stamp}"
        output_file = os.path.join(output_dir, "face_similarity_results.json")
        if os.path.exists(output_file):
            with open(output_file, "r") as f:
                return json.load(f)
        else:
            print(f"Warning: Output file not found for {stamp} {character}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running face_identity.py for {stamp} {character}: {e}")
        print(e.stdout)
        print(e.stderr)
        return None

def main():
    parser = argparse.ArgumentParser(description="Wrapper for running face_identity.py on multiple generations")
    parser.add_argument("--summary", required=True, help="Path to {stamp_prefix}_generation_summary.json")
    args = parser.parse_args()

    summary_path = args.summary
    if not os.path.exists(summary_path):
        print(f"Summary file does not exist: {summary_path}")
        return

    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    # Determine output summary path
    stamp_prefix = os.path.basename(summary_path).split("_generation_summary.json")[0]
    combined_summary_path = os.path.join(
        os.path.dirname(summary_path),
        f"{stamp_prefix}_combined_face_identity_summary.json"
    )

    # Collect results - note: using "runs" instead of "generations"
    combined_results = []
    for run in summary_data.get("runs", []):
        mode = run.get("generation_mode", "")
        # Fixed typo: "animatediff_and_dreambooth" not "anmiatediff_and_dreambooth"
        if mode not in ["animatediff_and_dreambooth", "dreambooth_only"]:
            continue
        stamp = run.get("stamp")
        character = run.get("character")
        if not stamp or not character:
            print(f"Skipping entry with missing stamp or character: {run}")
            continue
        print(f"Processing stamp={stamp}, character={character}, mode={mode}")
        result = run_face_identity(stamp, character)
        if result is not None:
            combined_results.append({
                "stamp": stamp,
                "character": character,
                "generation_mode": mode,
                "face_identity_result": result
            })

    # Save combined summary
    with open(combined_summary_path, "w") as f:
        json.dump({"combined_face_identity_results": combined_results}, f, indent=2)
    print(f"Combined summary saved to: {combined_summary_path}")

if __name__ == "__main__":
    main()