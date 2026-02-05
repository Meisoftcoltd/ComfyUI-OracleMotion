import os
import json
import uuid
import gc
import torch
import re

def get_temp_dir():
    """
    Creates a unique temporary directory for the current batch run.
    Follows the structure: ComfyUI/output/OracleMotion_Temp/<uuid>/
    """
    # Attempt to locate ComfyUI base directory
    # Assumes this file is in ComfyUI/custom_nodes/ComfyUI-OracleMotion/utils.py
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Traverse up to find ComfyUI root (usually 2 levels up from custom_nodes/repo)
    # structure: root/custom_nodes/ComfyUI-OracleMotion/utils.py
    # root is 2 levels up from the directory containing this file
    comfy_root = os.path.dirname(os.path.dirname(current_dir))

    # Verify if 'output' exists there, otherwise just use a local output folder to be safe
    output_base = os.path.join(comfy_root, "output")
    if not os.path.exists(output_base):
        # Fallback if we are not in standard structure
        output_base = os.path.join(current_dir, "output")

    temp_base = os.path.join(output_base, "OracleMotion_Temp")
    unique_run_id = str(uuid.uuid4())
    run_dir = os.path.join(temp_base, unique_run_id)

    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def cleanup_vram():
    """
    Force garbage collection and clear CUDA cache.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def parse_json_output(text):
    """
    Robustly parses JSON from LLM output, handling markdown code blocks.
    """
    try:
        # If text contains markdown code blocks, extract the content
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
             # Try generic code block
            match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1)

        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        # Return empty list or raise, depending on desired behavior.
        # Returning None to signal failure.
        return None
