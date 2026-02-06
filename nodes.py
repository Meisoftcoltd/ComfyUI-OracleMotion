import os
import torch
import json
import uuid
import nodes
import folder_paths
from .utils import get_temp_dir, cleanup_vram, parse_json_output, make_grid

# Lazy imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

class OracleBrainAPI:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "narrative_text": ("STRING", {"multiline": True, "default": "A cyberpunk detective walking through a rainy neon city."}),
                "available_voices": ("STRING", {"default": "Bella, Sarul, QwenUser"}),
                "model_name": ("STRING", {"default": "gpt-4-turbo"}),
                "base_url": ("STRING", {"default": "https://api.openai.com/v1"}),
                "api_key": ("STRING", {"default": "sk-..."}),
            },
            "optional": {
                "audio_path": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("storyboard_json",)
    FUNCTION = "generate_storyboard"
    CATEGORY = "🪬 OracleMotion"

    def generate_storyboard(self, narrative_text, available_voices, model_name, base_url, api_key, audio_path=""):
        text_input = narrative_text

        # 1. Audio Transcription (if provided)
        if audio_path and os.path.exists(audio_path):
            if WhisperModel is None:
                print("Warning: faster_whisper not found. Skipping audio transcription.")
            else:
                print(f"Transcribing audio from {audio_path}...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                try:
                    model = WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")
                    segments, _ = model.transcribe(audio_path)
                    transcribed_text = " ".join([segment.text for segment in segments])
                    print(f"Transcription: {transcribed_text}")
                    text_input += "\n\nAudio Transcript: " + transcribed_text
                except Exception as e:
                    print(f"Error during transcription: {e}")

        # 2. LLM Generation
        if OpenAI is None:
            raise ImportError("openai library is required. Please install it.")

        client = OpenAI(api_key=api_key, base_url=base_url)

        SYSTEM_PROMPT = f"""
You are a Director. Output a strict JSON list.
Your task is to convert the user's narrative into a structured visual storyboard including dialogue and direction.
Available Voice Actors: {available_voices}

RULES:
1. Output MUST be a valid JSON list of objects.
2. Each object represents a scene.
3. Keys required per object:
   - "scene_id": (int) sequential index.
   - "dialogue": (string) Text to be spoken (empty if silent).
   - "audio_emotion": (string) Adjective (e.g., Happy, Sad, Whispering, Shouting).
   - "voice_name": (string) Name of the voice actor (pick from available voices).
   - "visual_prompt": (string) SDXL Prompt.
   - "action_description": (string) Movement description.
   - "reference_path": (string) leave empty.

Ensure the prompts are descriptive and consistent.
"""

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text_input}
                ]
            )
            content = response.choices[0].message.content
            parsed_json = parse_json_output(content)
        except Exception as e:
            print(f"LLM Error: {e}")
            parsed_json = []

        if not parsed_json:
            parsed_json = []

        return (json.dumps(parsed_json, indent=2),)

class OracleBrainLocal:
    @classmethod
    def INPUT_TYPES(s):
        from .utils import get_llm_models
        models = get_llm_models()
        return {
            "required": {
                "llm_model": (models if models else ["No models found"],),
                "narrative_text": ("STRING", {"multiline": True, "default": "A cyberpunk detective walking through a rainy neon city."}),
                "available_voices": ("STRING", {"default": "Bella, Sarul, QwenUser"}),
                "context_window": ("INT", {"default": 8192, "min": 2048, "max": 32768}),
                "max_tokens": ("INT", {"default": 2048, "min": 128, "max": 8192}),
                "gpu_layers": ("INT", {"default": 33, "min": 0, "max": 100}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("storyboard_json",)
    FUNCTION = "generate_storyboard_local"
    CATEGORY = "🪬 OracleMotion"

    def generate_storyboard_local(self, llm_model, narrative_text, available_voices, context_window, max_tokens, gpu_layers, temperature):
        from .utils import parse_json_output
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required for local LLM support. Please install it.")

        import folder_paths

        # Locate the model
        model_path = folder_paths.get_full_path("LLM", llm_model)
        # Fallback if folder_paths doesn't find it but we scanned it
        if not model_path or not os.path.exists(model_path):
             # Try to construct path based on scan logic in utils
             # This is a bit redundant but safe
             current_dir = os.path.dirname(os.path.abspath(__file__))
             comfy_root = os.path.dirname(os.path.dirname(current_dir))
             possible_paths = [
                os.path.join(comfy_root, "models", "LLM", llm_model),
                os.path.join(comfy_root, "models", "llama", llm_model),
             ]
             for p in possible_paths:
                 if os.path.exists(p):
                     model_path = p
                     break

        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(f"Model file not found: {llm_model}")

        print(f"Loading Local LLM: {model_path} with {gpu_layers} GPU layers...")
        llm = Llama(
            model_path=model_path,
            n_ctx=context_window,
            n_gpu_layers=gpu_layers,
            verbose=False
        )

        SYSTEM_PROMPT = f"""
You are a Director. Output a strict JSON list.
Your task is to convert the user's narrative into a structured visual storyboard including dialogue and direction.
Available Voice Actors: {available_voices}

RULES:
1. Output MUST be a valid JSON list of objects.
2. Each object represents a scene.
3. Keys required per object:
   - "scene_id": (int) sequential index.
   - "dialogue": (string) Text to be spoken (empty if silent).
   - "audio_emotion": (string) Adjective (e.g., Happy, Sad, Whispering, Shouting).
   - "voice_name": (string) Name of the voice actor (pick from available voices).
   - "visual_prompt": (string) SDXL Prompt.
   - "action_description": (string) Movement description.
   - "reference_path": (string) leave empty.

Ensure the prompts are descriptive and consistent.
"""

        # Construct Prompt (Generic Template)
        prompt = f"System: {SYSTEM_PROMPT}\nUser: {narrative_text}\nAssistant:"

        # Attempt to constrain to JSON if possible
        try:
            # New API
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": narrative_text}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"} # Try to enforce JSON
            )
            content = response["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Chat API failed or not supported, falling back to completion: {e}")
            # Fallback to completion
            prompt = f"{SYSTEM_PROMPT}\n\nUSER REQUEST: {narrative_text}\n\nOUTPUT JSON:"
            response = llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["User:", "System:"]
            )
            content = response["choices"][0]["text"]

        parsed_json = parse_json_output(content)
        if not parsed_json:
            parsed_json = []

        # Clean up
        del llm
        cleanup_vram()

        return (json.dumps(parsed_json, indent=2),)

class OracleDirector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "storyboard_json": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "user_edits": ("STRING", {"default": "[]"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("finalized_json",)
    FUNCTION = "direct_scenes"
    CATEGORY = "🪬 OracleMotion"

    def direct_scenes(self, storyboard_json, user_edits="[]"):
        # Load AI generated storyboard
        try:
            ai_scenes = json.loads(storyboard_json)
        except:
            ai_scenes = []

        # Load User Edits (from the JS Widget)
        try:
            user_scenes = json.loads(user_edits)
        except:
            user_scenes = []

        # Merge Logic: User edits override AI
        final_scenes = []

        if user_scenes and len(user_scenes) > 0:
            print("OracleDirector: Using User Edits.")
            final_scenes = user_scenes
        else:
            print("OracleDirector: Using AI Storyboard.")
            final_scenes = ai_scenes

        # Validate structure
        validated_scenes = []
        for i, scene in enumerate(final_scenes):
            validated_scene = {
                "scene_id": scene.get("scene_id", i),
                "frame": scene.get("frame", i * 40), # keeping frame for timeline viz
                "dialogue": scene.get("dialogue", ""),
                "audio_emotion": scene.get("audio_emotion", ""),
                "voice_name": scene.get("voice_name", ""),
                "visual_prompt": scene.get("visual_prompt", scene.get("prompt", "")), # mapping old key
                "action_description": scene.get("action_description", scene.get("action", "")), # mapping old key
                "reference_path": scene.get("reference_path", scene.get("path", "")) # mapping old key
            }
            validated_scenes.append(validated_scene)

        # Sort by scene_id
        validated_scenes.sort(key=lambda x: int(x["scene_id"]))

        return (json.dumps(validated_scenes, indent=2),)

class OracleVisualizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "storyboard_json": ("STRING", {"forceInput": True}),
                "sdxl_ckpt": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "global_style_prompt": ("STRING", {"multiline": True, "default": "Cinematic lighting, 8k resolution, photorealistic"}),
            }
        }

    RETURN_TYPES = ("LIST", "IMAGE")
    RETURN_NAMES = ("keyframe_paths", "preview_image")
    FUNCTION = "generate_keyframes"
    CATEGORY = "🪬 OracleMotion"

    def generate_keyframes(self, storyboard_json, sdxl_ckpt, global_style_prompt):
        # Lazy imports
        from diffusers import StableDiffusionXLPipeline
        from transformers import CLIPVisionModelWithProjection
        from PIL import Image
        import numpy as np

        scenes = json.loads(storyboard_json)
        temp_dir = get_temp_dir()
        keyframe_paths = []

        print(f"Loading SDXL from {sdxl_ckpt}...")
        try:
            if os.path.isfile(sdxl_ckpt):
                pipe = StableDiffusionXLPipeline.from_single_file(
                    sdxl_ckpt,
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    sdxl_ckpt,
                    torch_dtype=torch.float16
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load SDXL model: {e}")

        # Optimization
        pipe.enable_model_cpu_offload()

        try:
            for i, scene in enumerate(scenes):
                scene_id = scene.get("scene_id", i)
                visual_prompt = scene.get("visual_prompt", "")
                audio_emotion = scene.get("audio_emotion", "")
                ref_path = scene.get("reference_path", "")

                # Append emotion to prompt
                if audio_emotion:
                    visual_prompt = f"{visual_prompt}, {audio_emotion} expression, {audio_emotion} atmosphere"

                full_prompt = f"{visual_prompt}, {global_style_prompt}"

                print(f"Generating keyframe for scene {scene_id}...")

                # If valid ref path provided by user, use it.
                from .utils import load_image_as_pil
                if ref_path and os.path.exists(ref_path):
                     print(f"Using provided reference image: {ref_path}")
                     image = Image.open(ref_path).convert("RGB")
                else:
                     print(f"Generating new keyframe via SDXL...")
                     image = pipe(prompt=full_prompt, output_type="pil").images[0]

                filename = f"keyframe_{scene_id}_{uuid.uuid4().hex[:6]}.png"
                filepath = os.path.join(temp_dir, filename)
                image.save(filepath)
                keyframe_paths.append(filepath)

                cleanup_vram()

        finally:
            del pipe
            cleanup_vram()

        # Create Contact Sheet
        preview_image = make_grid(keyframe_paths)

        return (keyframe_paths, preview_image)

class OracleEngine:
    @classmethod
    def INPUT_TYPES(s):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except ImportError:
            samplers = ["euler", "euler_ancestral", "heun", "heunpp2", "dpm_2", "dpm_2_ancestral", "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu", "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm", "ddim", "uni_pc", "uni_pc_bh2"]
            schedulers = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]

        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "keyframe_paths": ("LIST",),
                "fps": ("INT", {"default": 16, "min": 1}), # Changed from frames to fps
                "steps": ("INT", {"default": 20, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "step": 0.1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "storyboard_json": ("STRING", {"forceInput": True}), # Added to get duration
            },
            "optional": {
                 "positive": ("STRING", {"default": "high quality, motion, animation", "multiline": True}),
                 "negative": ("STRING", {"default": "low quality, static, watermark", "multiline": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("video_paths",)
    FUNCTION = "animate_scenes"
    CATEGORY = "🪬 OracleMotion"

    def animate_scenes(self, model, vae, clip, keyframe_paths, fps, steps, cfg, sampler_name, scheduler, denoise, storyboard_json, positive="high quality", negative="low quality"):
        from diffusers.utils import export_to_video

        scenes = json.loads(storyboard_json)

        # Setup helpers
        temp_dir = get_temp_dir()
        video_paths = []

        # Conditioning
        tokens_pos = clip.tokenize(positive)
        cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
        cond_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

        tokens_neg = clip.tokenize(negative)
        cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
        cond_neg = [[cond_neg, {"pooled_output": pooled_neg}]]

        def load_image_as_latent(path, vae):
            from PIL import Image
            import numpy as np

            img = Image.open(path).convert("RGB")
            w, h = img.size
            w = (w // 8) * 8
            h = (h // 8) * 8
            img = img.resize((w, h))

            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0) # [1, H, W, 3]

            try:
                if hasattr(vae, "device"):
                    target_device = vae.device
                elif hasattr(vae, "first_stage_model"):
                    target_device = vae.first_stage_model.device
                else:
                    target_device = model.load_device if hasattr(model, "load_device") else "cuda"

                img_tensor = img_tensor.to(target_device)
            except Exception as e:
                print(f"Warning: Could not move tensor to VAE device: {e}")

            latent = vae.encode(img_tensor)
            if isinstance(latent, torch.Tensor):
                 return {"samples": latent}
            return latent

        def run_sampler(model, vae, latent_input, steps, cfg, sampler_name, scheduler, positive, negative, denoise):
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            try:
                result_latent = nodes.common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent_input, denoise=denoise
                )[0]
            except Exception as e:
                print(f"Sampler Error: {e}")
                raise e

            pixels = vae.decode(result_latent["samples"])
            return pixels

        current_input_path = None
        last_generated_frame_path = None

        for i, keyframe_path in enumerate(keyframe_paths):
            print(f"Animating Scene {i}...")

            # Get duration from storyboard
            scene_duration = 5.0 # default
            if i < len(scenes):
                scene_duration = scenes[i].get("duration", 5.0)

            frames_count = int(scene_duration * fps)
            if frames_count < 1: frames_count = 16 # min fallback

            # Determine Input
            if i == 0:
                current_input_path = keyframe_path
            else:
                current_input_path = last_generated_frame_path

            if not current_input_path or not os.path.exists(current_input_path):
                print(f"Skipping scene {i}: Input path missing {current_input_path}")
                continue

            # 1. Encode Input Image to Latent
            start_latent = load_image_as_latent(current_input_path, vae)

            # 2. Expand Latent for Video
            lat_tensor = start_latent["samples"]
            lat_batch = lat_tensor.repeat(frames_count, 1, 1, 1)
            latent_input = {"samples": lat_batch}

            # 3. Run Sampler
            pixel_frames = run_sampler(model, vae, latent_input, steps, cfg, sampler_name, scheduler, cond_pos, cond_neg, denoise)

            # 4. Save Video
            import numpy as np
            from PIL import Image

            frames_pil = []
            for f in range(pixel_frames.shape[0]):
                 p = pixel_frames[f].cpu().numpy()
                 p = np.clip(p, 0, 1) * 255
                 p = p.astype(np.uint8)
                 frames_pil.append(Image.fromarray(p))

            video_filename = f"scene_{i}_{uuid.uuid4().hex[:6]}.mp4"
            video_filepath = os.path.join(temp_dir, video_filename)

            export_to_video(frames_pil, video_filepath, fps=fps)
            video_paths.append(video_filepath)

            # 5. Save Last Frame for Next Scene
            last_frame_pil = frames_pil[-1]
            last_frame_filename = f"last_frame_scene_{i}_{uuid.uuid4().hex[:6]}.png"
            last_generated_frame_path = os.path.join(temp_dir, last_frame_filename)
            last_frame_pil.save(last_generated_frame_path)

            cleanup_vram()

        return (video_paths,)

class OraclePostProduction:
    @classmethod
    def INPUT_TYPES(s):
        from .utils import get_font_path
        font_path = get_font_path()
        return {
            "required": {
                "enhanced_storyboard_json": ("STRING", {"forceInput": True}),
                "font_size": ("INT", {"default": 60, "min": 10}),
                "font_color": ("STRING", {"default": "#FFD700"}),
                "stroke_width": ("INT", {"default": 4, "min": 0}),
                "position_y": ("INT", {"default": 100, "min": 0}),
                "preview_mode": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "video_paths": ("LIST",),
                "preview_background": ("IMAGE",),
                "font_path": ("STRING", {"default": font_path}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("final_video_path", "preview_image")
    FUNCTION = "post_production"
    CATEGORY = "🪬 OracleMotion"

    def post_production(self, enhanced_storyboard_json, font_size, font_color, stroke_width, position_y, preview_mode, video_paths=None, preview_background=None, font_path=None):
        from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np

        scenes = json.loads(enhanced_storyboard_json)
        temp_dir = get_temp_dir()

        # Helper to hex to rgb
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            if len(hex_color) == 6:
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return (255, 215, 0) # Gold default

        text_color = hex_to_rgb(font_color)
        stroke_color = (0, 0, 0)

        # Setup Font
        try:
            if font_path and os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                from .utils import get_font_path
                fallback = get_font_path()
                font = ImageFont.truetype(fallback, font_size)
        except Exception as e:
            print(f"Font loading error: {e}. Using default.")
            font = ImageFont.load_default()

        # --- LOGIC BRANCH 1: PREVIEW MODE ---
        if preview_mode:
            print("Running OraclePostProduction in PREVIEW MODE")
            # Create canvas
            W, H = 1080, 1920
            if preview_background is not None:
                # ComfyUI Image is Tensor [1, H, W, 3]
                # Convert first image to PIL
                p_img = 255. * preview_background[0].cpu().numpy()
                img = Image.fromarray(np.clip(p_img, 0, 255).astype(np.uint8)).convert("RGB")
                W, H = img.size
            else:
                img = Image.new("RGB", (W, H), (0, 0, 0))

            draw = ImageDraw.Draw(img)

            # Find sample text
            sample_text = "SAMPLE CAPTION TEXT"
            for scene in scenes:
                if scene.get("dialogue"):
                    sample_text = scene.get("dialogue")
                    break

            # Draw Text (Centered X, Position Y from bottom)
            try:
                # PIL < 10 bbox; PIL >= 10 getbbox?
                # Using simple approach
                left, top, right, bottom = draw.textbbox((0, 0), sample_text, font=font)
                text_width = right - left
                text_height = bottom - top
            except:
                 text_width, text_height = font.getsize(sample_text)

            x = (W - text_width) / 2
            y = H - position_y - text_height

            # Stroke
            draw.text((x, y), sample_text, font=font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_color)

            # Return Image Tensor
            img_np = np.array(img).astype(np.float32) / 255.0
            preview_tensor = torch.from_numpy(img_np).unsqueeze(0)

            return ("", preview_tensor)

        # --- LOGIC BRANCH 2: RENDER MODE ---
        print("Running OraclePostProduction in RENDER MODE")
        if not video_paths:
            raise RuntimeError("No video_paths provided for render mode.")

        # 1. Stitch Videos & Audio
        clips = []
        try:
            for i, path in enumerate(video_paths):
                if os.path.exists(path):
                    clip = VideoFileClip(path)
                    # Sync Audio
                    if i < len(scenes):
                        audio_path = scenes[i].get("audio_path", "")
                        if audio_path and os.path.exists(audio_path):
                            audio_clip = AudioFileClip(audio_path)
                            clip = clip.set_audio(audio_clip)
                            if abs(clip.duration - audio_clip.duration) > 0.1:
                                clip = clip.set_duration(audio_clip.duration)
                    clips.append(clip)
        except Exception as e:
            raise RuntimeError(f"Error loading clips: {e}")

        if not clips:
             raise RuntimeError("No clips found.")

        final_clip = concatenate_videoclips(clips, method="compose")

        # 2. Prepare Timings for Captions
        # Create a list of (start_time, end_time, text)
        captions = []
        current_t = 0.0
        for i, clip in enumerate(clips):
            if i < len(scenes):
                text = scenes[i].get("dialogue", "").strip()
                if text:
                    captions.append({
                        "start": current_t,
                        "end": current_t + clip.duration,
                        "text": text
                    })
            current_t += clip.duration

        # 3. Viral Burn Filter
        def burn_text(get_frame, t):
            frame = get_frame(t) # Numpy array [H, W, 3]

            # Find active caption
            active_text = None
            for cap in captions:
                if cap["start"] <= t < cap["end"]:
                    active_text = cap["text"]
                    break

            if active_text:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                W, H = img.size

                # Measure
                try:
                    left, top, right, bottom = draw.textbbox((0, 0), active_text, font=font)
                    tw = right - left
                    th = bottom - top
                except:
                     tw, th = font.getsize(active_text)

                x = (W - tw) / 2
                y = H - position_y - th

                draw.text((x, y), active_text, font=font, fill=text_color, stroke_width=stroke_width, stroke_fill=stroke_color)
                return np.array(img)

            return frame

        # Apply filter
        burned_clip = final_clip.fl(burn_text)

        output_filename = f"final_viral_{uuid.uuid4().hex[:6]}.mp4"
        output_path = os.path.join(temp_dir, output_filename)

        burned_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        # Cleanup
        for c in clips:
            try: c.close()
            except: pass

        return (output_path, torch.zeros((1, 512, 512, 3)))

class OracleVoiceKokoro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "storyboard_json": ("STRING", {"forceInput": True}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("storyboard_json_with_audio",)
    FUNCTION = "generate_voice"
    CATEGORY = "🪬 OracleMotion"

    def generate_voice(self, storyboard_json):
        # Implementation for Kokoro TTS
        # Logic: Iterate scenes, find voice_name, generate .wav
        # Update scene["audio_path"] and scene["duration"]
        try:
             import soundfile as sf
             from kokoro_onnx import Kokoro
        except ImportError:
             raise ImportError("kokoro-onnx and soundfile are required.")

        scenes = json.loads(storyboard_json)
        temp_dir = get_temp_dir()

        # Initialize Kokoro (Assuming model path logic)
        import folder_paths
        kokoro_path = os.path.join(folder_paths.models_dir, "Kokoro", "kokoro-v0_19.onnx")
        voices_path = os.path.join(folder_paths.models_dir, "Kokoro", "voices.json")

        # Fallback manual check
        if not os.path.exists(kokoro_path):
             print(f"Kokoro model not found at {kokoro_path}. Skipping audio generation.")
             return (json.dumps(scenes, indent=2),)

        kokoro = Kokoro(kokoro_path, voices_path)

        for scene in scenes:
            dialogue = scene.get("dialogue", "")
            voice_name = scene.get("voice_name", "af_sarah") # default voice
            emotion = scene.get("audio_emotion", "")

            if dialogue:
                # Map emotion to speed logic (simple heuristic)
                speed = 1.0
                if "sad" in emotion.lower(): speed = 0.8
                if "happy" in emotion.lower() or "excited" in emotion.lower(): speed = 1.2

                # Generate
                try:
                    samples, sample_rate = kokoro.create(dialogue, voice=voice_name, speed=speed, lang="en-us")

                    filename = f"audio_{scene.get('scene_id', 0)}_{uuid.uuid4().hex[:6]}.wav"
                    filepath = os.path.join(temp_dir, filename)

                    sf.write(filepath, samples, sample_rate)

                    # Update scene data
                    scene["audio_path"] = filepath
                    scene["duration"] = len(samples) / sample_rate

                except Exception as e:
                    print(f"Error generating voice for scene {scene.get('scene_id')}: {e}")
            else:
                # Silent scene default duration
                scene["duration"] = 3.0

        return (json.dumps(scenes, indent=2),)

class OracleVoiceInjector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "storyboard_json": ("STRING", {"forceInput": True}),
                "audio_batch": ("AUDIO",),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("storyboard_json_with_audio",)
    FUNCTION = "inject_audio"
    CATEGORY = "🪬 OracleMotion"

    def inject_audio(self, storyboard_json, audio_batch):
        # audio_batch from ComfyUI is usually {"waveform": tensor [B, C, N], "sample_rate": int}
        # Iterate scenes and assign audio clips sequentially

        scenes = json.loads(storyboard_json)
        temp_dir = get_temp_dir()

        waveform = audio_batch["waveform"]
        sample_rate = audio_batch["sample_rate"]

        # Check batch size
        num_clips = waveform.shape[0]

        import soundfile as sf
        import numpy as np

        for i, scene in enumerate(scenes):
            if i < num_clips:
                clip_waveform = waveform[i] # [C, N]
                # Convert to numpy for saving
                if clip_waveform.shape[0] > 1:
                    # Stereo to mono or keep stereo? sf.write handles multi-channel
                    audio_np = clip_waveform.cpu().numpy().T # [N, C]
                else:
                    audio_np = clip_waveform.squeeze().cpu().numpy()

                filename = f"injected_audio_{i}_{uuid.uuid4().hex[:6]}.wav"
                filepath = os.path.join(temp_dir, filename)

                sf.write(filepath, audio_np, sample_rate)

                duration = audio_np.shape[0] / sample_rate

                scene["audio_path"] = filepath
                scene["duration"] = duration
            else:
                 # No audio for this scene
                 pass

        return (json.dumps(scenes, indent=2),)
