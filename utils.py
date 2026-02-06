import os
import json
import uuid
import gc
import torch
import re
import numpy as np
from PIL import Image

def get_temp_dir():
    """
    Creates a unique temporary directory for the current batch run.
    Follows the structure: ComfyUI/output/OracleMotion_Project_{UUID}/
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_root = os.path.dirname(os.path.dirname(current_dir))
    output_base = os.path.join(comfy_root, "output")

    if not os.path.exists(output_base):
        output_base = os.path.join(current_dir, "output")

    unique_run_id = str(uuid.uuid4())
    run_dir = os.path.join(output_base, f"OracleMotion_Project_{unique_run_id}")

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
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1)

        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return []

def load_image_from_path(path):
    """
    Robustly loads an image from a path.
    Returns a PIL Image or None if not found/failed.
    """
    if not os.path.exists(path):
        print(f"Error: Image not found at {path}")
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"Error loading image from {path}: {e}")
        return None

# Alias for consistency if called elsewhere
load_image_as_pil = load_image_from_path

def make_grid(keyframe_paths):
    """
    Creates a contact sheet (grid) from a list of image paths.
    Returns: torch.Tensor [1, H, W, C] suitable for ComfyUI IMAGE output.
    """
    images = []
    for path in keyframe_paths:
        img = load_image_from_path(path)
        if img:
            images.append(img)

    if not images:
        return torch.zeros((1, 512, 512, 3)) # Return black square if no images

    # Assume all images are same size for simplicity, or resize to first image size
    w, h = images[0].size
    grid_w = w * len(images)
    grid_h = h

    grid = Image.new('RGB', (grid_w, grid_h))
    for i, img in enumerate(images):
        if img.size != (w, h):
            img = img.resize((w, h))
        grid.paste(img, (i * w, 0))

    # Convert PIL to Tensor [1, H, W, C] (0-1 float)
    grid_np = np.array(grid).astype(np.float32) / 255.0
    grid_tensor = torch.from_numpy(grid_np).unsqueeze(0)

    return grid_tensor

def get_llm_models():
    """
    Scans for GGUF models in standard locations.
    Returns a list of filenames.
    """
    import folder_paths

    # Try getting from ComfyUI defined paths if available
    try:
        models = folder_paths.get_filename_list("LLM")
        if models:
            return [m for m in models if m.endswith(".gguf")]
    except:
        pass

    # Fallback/Additional manual scan
    # Assuming utils.py is in custom_nodes/ComfyUI-OracleMotion
    current_dir = os.path.dirname(os.path.abspath(__file__))
    comfy_root = os.path.dirname(os.path.dirname(current_dir))

    possible_paths = [
        os.path.join(comfy_root, "models", "LLM"),
        os.path.join(comfy_root, "models", "llama"), # Common alternative
    ]

    found_models = []
    for p in possible_paths:
        if os.path.exists(p):
            for f in os.listdir(p):
                if f.endswith(".gguf"):
                    found_models.append(f)

    return sorted(list(set(found_models)))
