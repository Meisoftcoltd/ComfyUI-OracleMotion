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
