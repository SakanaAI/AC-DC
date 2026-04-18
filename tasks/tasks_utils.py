import importlib.util
import json
import os
import sys
from typing import Optional


def update_metadata(task_dir: str, new_metadata: dict):
    metadata_path = os.path.join(task_dir, "metadata.json")
    # Read existing metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    # Update metadata
    metadata.update(new_metadata)
    # Write updated metadata back to file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def update_task_metadata(task_dir: str, new_metadata: dict):
    metadata_path = os.path.join(task_dir, "task.json")
    # Read existing metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}
    # Update metadata
    metadata.update(new_metadata)
    # Write updated metadata back to file
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)


def save_task_to_disk(
    new_task_dir: str, response: dict, metadata: Optional[dict] = None
):
    if metadata is None:
        metadata = {}
    new_task_code = response["task_family"]
    new_task_python_file = os.path.join(new_task_dir, "task.py")
    new_task_json_descriptor = os.path.join(new_task_dir, "task.json")
    with open(new_task_python_file, "w") as f:
        f.write(str(new_task_code))

    with open(new_task_json_descriptor, "w") as f:
        response_filtered = {
            k: v
            for k, v in response.items()
            if k
            in [
                "name_of_task",
                "description_of_task",
                "capability_being_measured",
                "estimated_human_difficulty",
                "example_instruction",
            ]
        }
        json.dump(response_filtered, f, indent=4)

    if metadata:
        update_metadata(new_task_dir, metadata)


def load_task_family(task_dir: str):
    task_module_path = os.path.join(task_dir, "task.py")
    module_name = f"task_{os.path.basename(task_dir)}"
    spec = importlib.util.spec_from_file_location(module_name, task_module_path)
    task_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = task_module
    spec.loader.exec_module(task_module)
    task_family = task_module.TaskFamily

    # Check task family has required methods.
    for method in ["get_tasks", "get_instructions", "score"]:
        if not hasattr(task_family, method):
            raise AttributeError(f"Task family must define a {method} method.")

    return task_family

