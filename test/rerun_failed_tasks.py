import os
import json
import sys
import time

# Add the helpers directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../helpers')))
from train_dreambooth_script import train_dreambooth, is_valid_lora_directory

from dreambooth_parameters import (
    train_and_test_parameters,
    mark_task_completed,
    get_baseline_parameters,
    create_parameter_folder_name
)

def rerun_failed_tasks(config_file_path, instance_dir, base_output_dir, unique_identifier, device):
    with open(config_file_path, 'r') as f:
        config_data = json.load(f)

    failed_tasks = [task for task in config_data["combinations"] if task["status"] == "failed"]

    print(f"Found {len(failed_tasks)} failed tasks to rerun.")

    for task in failed_tasks:
        print(f"\nRe-running failed task {task['id']}: {task['name']}")
        try:
            success, training_time, generation_time = train_and_test_parameters(
                instance_dir,
                base_output_dir,
                unique_identifier,
                device,
                task['parameters'],
                task['name']
            )
            mark_task_completed(config_file_path, task['id'], success, training_time, generation_time)
            if success:
                print(f"✓ Task {task['id']} ({task['name']}) completed successfully on rerun")
            else:
                print(f"✗ Task {task['id']} ({task['name']}) failed again")
        except Exception as e:
            print(f"Error re-running task {task['id']}: {e}")
            mark_task_completed(config_file_path, task['id'], False, None, None)

if __name__ == "__main__":
    # CONFIGURATION - match these to your main script
    NAME = "alice"
    DESCRIPTION = "woman, young, black curly hair"
    INSTANCE_DIR = f"/graphics/scratch2/students/webereli/parameter_testing/training_images_test_file/{NAME}"
    BASE_OUTPUT_DIR = "/graphics/scratch2/students/webereli/parameter_testing/"
    CONFIG_FILE = os.path.join(BASE_OUTPUT_DIR, "dreambooth_config.json")
    DEVICE = "cuda" if "cuda" in sys.argv or (hasattr(os, "environ") and os.environ.get("CUDA_VISIBLE_DEVICES")) else "cpu"

    def _sanitize_name_for_identifier(name):
        import re
        return re.sub(r'[^a-z0-9]', '', name.lower())

    sanitized_name = _sanitize_name_for_identifier(NAME)
    first_descriptor = DESCRIPTION.split(',')[0].strip() if DESCRIPTION else ""
    UNIQUE_IDENTIFIER = f"sks{sanitized_name} {first_descriptor}"

    rerun_failed_tasks(CONFIG_FILE, INSTANCE_DIR, BASE_OUTPUT_DIR, UNIQUE_IDENTIFIER, DEVICE)