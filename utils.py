import os
import json
import uuid
import gc
import torch
import re
import numpy as np
from PIL import Image

def get_temp_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if "custom_nodes" in current_dir:
        base = current_dir.split("custom_nodes")[0]
        output_dir = os.path.join(base, "output", "OracleMotion_Project_" + str(uuid.uuid4())[:8])
    else:
        output_dir = os.path.join(current_dir, "output")

    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def cleanup_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def parse_json_output(text):
    try:
        # Try to find JSON block in markdown
        match = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # Try to find any code block
            match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
            if match:
                text = match.group(1)

        # Clean up any potential trailing/leading whitespace or artifacts
        text = text.strip()
        return json.loads(text)
    except:
        # Fallback: try to parse the whole text if it's raw JSON
        try:
            return json.loads(text)
        except:
            return []

def load_image_from_path(path):
    if os.path.exists(path):
        return Image.open(path).convert("RGB")
    return None

load_image_as_pil = load_image_from_path

def make_grid(keyframe_paths):
    images = [load_image_from_path(p) for p in keyframe_paths if os.path.exists(p)]
    if not images:
        return torch.zeros((1, 512, 512, 3))

    w, h = images[0].size
    grid = Image.new('RGB', (w * len(images), h))
    for i, img in enumerate(images):
        grid.paste(img.resize((w, h)), (i * w, 0))

    return torch.from_numpy(np.array(grid).astype(np.float32) / 255.0).unsqueeze(0)

def get_llm_models():
    import folder_paths
    try:
        models = folder_paths.get_filename_list("LLM")
        if models:
            return [m for m in models if m.endswith(".gguf")]
    except:
        pass
    return ["Put_GGUF_In_Models_LLM_Folder.gguf"]

def get_font_path():
    current = os.path.dirname(os.path.abspath(__file__))
    font_dir = os.path.join(current, "fonts")
    if os.path.exists(font_dir):
        for f in os.listdir(font_dir):
            if f.endswith(".ttf") or f.endswith(".otf"):
                return os.path.join(font_dir, f)
    return "arial.ttf"

def validate_path(base_dir, filename):
    import os
    base_abs = os.path.abspath(base_dir)
    full_path = os.path.abspath(os.path.join(base_abs, filename))
    if os.path.commonpath([base_abs, full_path]) == base_abs:
        return full_path
    return None
