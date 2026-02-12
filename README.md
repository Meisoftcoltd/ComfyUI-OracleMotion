# 🔮 ComfyUI-OracleMotion (Studio Edition)
The Ultimate Audio-Driven Animation Studio for ComfyUI. Local LLMs | Local TTS | Wan 2.1 Agnostic Support | Viral Captions

## 🔌 How to Connect (The Wiring)

### Phase 1: The Script & Voice (Audio-First)
**🧠 Oracle Brain (Local)** `[storyboard_json]` --> **🎙️ Oracle Voice (Kokoro)** `[storyboard_json]`
*Note: The Voice node calculates the exact duration of every scene.*

### Phase 2: The Director (Timeline)
**🎙️ Oracle Voice** `[enhanced_json]` --> **🪬 Oracle Director** `[storyboard_json]`
*Action: Use the Visual Timeline here to edit text or drag-and-drop reference images.*

### Phase 3: The Visuals (Assets)
**🪬 Oracle Director** `[finalized_json]` --> **🎨 Oracle Visualizer** `[storyboard_json]`
*Input: Connect your Checkpoint (SDXL) and Base Image here.*

### Phase 4: The Engine (Animation)
**🎨 Oracle Visualizer** `[keyframe_paths]` --> **🎬 Oracle Engine** `[keyframe_paths]`
**🪬 Oracle Director** `[finalized_json]` --> **🎬 Oracle Engine** `[storyboard_json]` *(Critical for duration syncing)*
*Input: Connect your Video Model (Wan 2.1 GGUF), VAE, and CLIP here.*

### Phase 5: Post-Production (Viral Editor)
**🎬 Oracle Engine** `[video_paths]` --> **✂️ Oracle Post-Production** `[video_paths]`
**🪬 Oracle Director** `[finalized_json]` --> **✂️ Oracle Post-Production** `[enhanced_storyboard_json]`
*Features: Enable preview_mode to check caption placement before full render.*

## 🚀 Installation & Setup Guide

### 1. 📦 Python Dependencies
The requirements file excludes `torch` to prevent breaking your existing ComfyUI environment.
```bash
pip install -r requirements.txt
```

### 2. 📂 Required Assets (Manual Setup)
Due to licensing and file size, you must manually place the following files:

**🔤 Fonts:**
- Download a bold .ttf font (e.g., Komika Axis or Impact).
- Place it in: `custom_nodes/ComfyUI-OracleMotion/fonts/`
- *Fallback: If missing, the node will try to use a system font (Arial).*

**🗣️ Voice Models (Kokoro):**
- Download `kokoro-v0_19.onnx` and `voices.json` (Search HuggingFace for "hexgrad/Kokoro-82M").
- Place them in: `ComfyUI/models/Kokoro/` (Create folder if needed).

**🧠 Brain Models (LLM):**
- Download your preferred GGUF model (e.g., `Mistral-7B-Instruct-v0.3.Q6_K.gguf`).
- Place it in: `ComfyUI/models/LLM/`

### 3. 🔌 Standard Workflow
1. **Brain**: Generates the Script (.json).
2. **Voice**: Generates Audio & Calibrates Duration (.json update).
3. **Director**: Visual Timeline for manual tweaks.
4. **Visualizer**: Generates Keyframes (Images).
5. **Engine**: Animates based on Audio Duration (Video).
6. **Post-Production**: Burns Viral Captions & Stitches Audio.

🎉 **Ready to animate! Restart ComfyUI to load the nodes.**
