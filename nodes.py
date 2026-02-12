import os
import torch
import json
import uuid
import nodes
import folder_paths
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .utils import get_temp_dir, cleanup_vram, parse_json_output, make_grid

# --- IMPORTS FOR QWEN ---
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None
    print("⚠️ [OracleMotion] Warning: 'qwen-tts' not found. Qwen nodes will fail.")

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
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a Director. Output a strict JSON list."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("storyboard_json",)
    FUNCTION = "generate_storyboard"
    CATEGORY = "🪬 OracleMotion"

    def generate_storyboard(self, narrative_text, available_voices, model_name, base_url, api_key, audio_path="", system_prompt=""):
        text_input = narrative_text

        # 1. Audio Transcription
        if audio_path and os.path.exists(audio_path):
            if WhisperModel is None:
                print("Warning: faster_whisper not found. Skipping transcription.")
            else:
                try:
                    print(f"Transcribing {audio_path}...")
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    model = WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")
                    segments, _ = model.transcribe(audio_path)
                    transcribed_text = " ".join([segment.text for segment in segments])
                    text_input += "\n\nAudio Transcript: " + transcribed_text
                except Exception as e:
                    print(f"Transcription Error: {e}")

        # 2. LLM Generation
        if OpenAI is None:
            raise ImportError("openai library required.")

        client = OpenAI(api_key=api_key, base_url=base_url)

        FULL_SYSTEM_PROMPT = f"""
{system_prompt}
Your task is to convert the user's narrative into a structured visual storyboard.
Available Voice Actors: {available_voices}

RULES:
Output MUST be a valid JSON list of objects.
Each object represents a scene.
Keys required per object:
"scene_id": (int) sequential index.
"dialogue": (string) Text to be spoken (empty if silent).
"audio_emotion": (string) Adjective (e.g., Happy, Sad, Whispering).
"voice_name": (string) Name of the voice actor.
"visual_prompt": (string) SDXL Prompt.
"action_description": (string) Movement description.
"""

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": FULL_SYSTEM_PROMPT},
                    {"role": "user", "content": text_input}
                ]
            )
            content = response.choices[0].message.content
            parsed_json = parse_json_output(content)
        except Exception as e:
            print(f"LLM Error: {e}")
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
                "context_window": ("INT", {"default": 8192}),
                "max_tokens": ("INT", {"default": 2048}),
                "gpu_layers": ("INT", {"default": 33}),
                "temperature": ("FLOAT", {"default": 0.7}),
            },
            "optional": {
                "system_prompt": ("STRING", {"multiline": True, "default": "You are a Director. Output a strict JSON list."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("storyboard_json",)
    FUNCTION = "generate_storyboard_local"
    CATEGORY = "🪬 OracleMotion"

    def generate_storyboard_local(self, llm_model, narrative_text, available_voices, context_window, max_tokens, gpu_layers, temperature, system_prompt=""):
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError("llama-cpp-python is required.")

        # Locate Model
        model_path = folder_paths.get_full_path("LLM", llm_model)
        if not model_path:
             # Fallback manual check
             base_path = os.path.join(folder_paths.models_dir, "LLM")
             model_path = os.path.join(base_path, llm_model)

        if not os.path.exists(model_path):
            raise RuntimeError(f"Model not found: {model_path}")

        print(f"Loading Local LLM: {model_path}")
        llm = Llama(
            model_path=model_path,
            n_ctx=context_window,
            n_gpu_layers=gpu_layers,
            verbose=False
        )

        FULL_SYSTEM_PROMPT = f"""
{system_prompt}
Convert narrative to JSON storyboard.
Available Voices: {available_voices}
Required Keys: scene_id, dialogue, audio_emotion, voice_name, visual_prompt, action_description.
Output ONLY JSON.
"""

        # Simple Prompt Template
        prompt = f"System: {FULL_SYSTEM_PROMPT}\nUser: {narrative_text}\nAssistant:"

        try:
            response = llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": FULL_SYSTEM_PROMPT},
                    {"role": "user", "content": narrative_text}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"}
            )
            content = response["choices"][0]["message"]["content"]
        except:
            # Fallback
            response = llm(prompt, max_tokens=max_tokens, temperature=temperature)
            content = response["choices"][0]["text"]

        parsed_json = parse_json_output(content)
        del llm
        cleanup_vram()
        return (json.dumps(parsed_json, indent=2),)

class OracleDirector:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "storyboard_json": ("STRING", {"forceInput": True})
            },
            "hidden": {
                "user_edits": ("STRING", {"default": "[]"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("finalized_json",)
    FUNCTION = "direct_scenes"
    CATEGORY = "🪬 OracleMotion"

    def direct_scenes(self, storyboard_json, user_edits="[]"):
        try: ai_scenes = json.loads(storyboard_json)
        except: ai_scenes = []
        try: user_scenes = json.loads(user_edits)
        except: user_scenes = []

        final_scenes = user_scenes if user_scenes else ai_scenes

        # Validation & Normalization
        validated = []
        for i, s in enumerate(final_scenes):
            validated.append({
                "scene_id": s.get("scene_id", i),
                "dialogue": s.get("dialogue", ""),
                "audio_emotion": s.get("audio_emotion", ""),
                "voice_name": s.get("voice_name", ""),
                "visual_prompt": s.get("visual_prompt", s.get("prompt", "")),
                "action_description": s.get("action_description", s.get("action", "")),
                "reference_path": s.get("reference_path", ""),
                "duration": s.get("duration", 3.0), # Important for Engine
                "audio_path": s.get("audio_path", "")
            })

        return (json.dumps(validated, indent=2),)

class OracleVisualizer:
    @classmethod
    def INPUT_TYPES(s):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except:
            samplers = ["euler"]
            schedulers = ["normal"]

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "storyboard_json": ("STRING", {"forceInput": True}),
                "width": ("INT", {"default": 1024}),
                "height": ("INT", {"default": 1024}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 8.0}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "global_style_prompt": ("STRING", {"multiline": True, "default": "Cinematic lighting, 8k, masterpiece"}),
                "negative_prompt": ("STRING", {"default": "text, watermark, blurry, low quality", "multiline": True}),
            }
        }

    RETURN_TYPES = ("LIST", "IMAGE")
    RETURN_NAMES = ("keyframe_paths", "preview_image")
    FUNCTION = "generate_keyframes"
    CATEGORY = "🪬 OracleMotion"

    def generate_keyframes(self, model, clip, vae, storyboard_json, width, height, steps, cfg, sampler_name, scheduler, global_style_prompt, negative_prompt):
        scenes = json.loads(storyboard_json)
        temp_dir = get_temp_dir()
        keyframe_paths = []

        # 1. Pre-calculate Negative Conditioning (Static for all scenes)
        tokens_neg = clip.tokenize(negative_prompt)
        cond_neg = [[clip.encode_from_tokens(tokens_neg, return_pooled=True)[0], {"pooled_output": clip.encode_from_tokens(tokens_neg, return_pooled=True)[1]}]]

        for i, scene in enumerate(scenes):
            # 2. Construct Prompt per Scene
            scene_prompt = scene.get('visual_prompt', '')
            emotion = scene.get('audio_emotion', '')
            full_prompt = f"{scene_prompt}, {emotion}, {global_style_prompt}"

            print(f"🎨 Visualizer Generating Scene {i}: {full_prompt}")

            # 3. Encode Positive
            tokens = clip.tokenize(full_prompt)
            cond_pos = [[clip.encode_from_tokens(tokens, return_pooled=True)[0], {"pooled_output": clip.encode_from_tokens(tokens, return_pooled=True)[1]}]]

            # 4. Create Empty Latent
            latent = torch.zeros([1, 4, height // 8, width // 8])

            # 5. Sample (Standard Comfy KSampler)
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            try:
                # Common KSampler Wrapper
                samples = nodes.common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    cond_pos, cond_neg, {"samples": latent}, denoise=1.0
                )[0]["samples"]
            except Exception as e:
                print(f"Sampler Error: {e}")
                continue

            # 6. Decode VAE
            pixels = vae.decode(samples) # [1, H, W, 3]

            # 7. Save Image
            # Convert Tensor to PIL
            i_np = 255. * pixels.cpu().numpy()[0]
            img = Image.fromarray(np.clip(i_np, 0, 255).astype(np.uint8))

            filename = f"keyframe_{i}_{uuid.uuid4().hex[:6]}.png"
            filepath = os.path.join(temp_dir, filename)
            img.save(filepath)
            keyframe_paths.append(filepath)

            cleanup_vram()

        # Generate Grid Preview
        preview = make_grid(keyframe_paths)
        return (keyframe_paths, preview)

class OracleEngine:
    @classmethod
    def INPUT_TYPES(s):
        try:
            import comfy.samplers
            samplers = comfy.samplers.KSampler.SAMPLERS
            schedulers = comfy.samplers.KSampler.SCHEDULERS
        except:
            samplers = ["euler"]
            schedulers = ["normal"]

        return {
            "required": {
                "model": ("MODEL",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "keyframe_paths": ("LIST",),
                "storyboard_json": ("STRING", {"forceInput": True}),
                "fps": ("INT", {"default": 16}),
                "steps": ("INT", {"default": 20}),
                "cfg": ("FLOAT", {"default": 6.0}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 1.0}),
                "motion_strength": ("FLOAT", {"default": 1.0}),
            },
            "optional": {
                "positive": ("STRING", {"default": "high quality motion", "multiline": True}),
                "negative": ("STRING", {"default": "static, watermark", "multiline": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("video_paths",)
    FUNCTION = "animate_scenes"
    CATEGORY = "🪬 OracleMotion"

    def animate_scenes(self, model, vae, clip, keyframe_paths, storyboard_json, fps, steps, cfg, sampler_name, scheduler, denoise, motion_strength, positive, negative):
        from diffusers.utils import export_to_video
        scenes = json.loads(storyboard_json)
        temp_dir = get_temp_dir()
        video_paths = []

        # Conditioning
        tokens = clip.tokenize(positive)
        cond_pos = [[clip.encode_from_tokens(tokens, return_pooled=True)[0], {"pooled_output": clip.encode_from_tokens(tokens, return_pooled=True)[1]}]]
        tokens_neg = clip.tokenize(negative)
        cond_neg = [[clip.encode_from_tokens(tokens_neg, return_pooled=True)[0], {"pooled_output": clip.encode_from_tokens(tokens_neg, return_pooled=True)[1]}]]

        last_latent = None

        for i, path in enumerate(keyframe_paths):
            duration = scenes[i].get("duration", 4.0) if i < len(scenes) else 4.0
            num_frames = max(16, int(duration * fps))

            img = Image.open(path).convert("RGB")
            w, h = img.size
            w, h = (w // 8) * 8, (h // 8) * 8
            img = img.resize((w, h))
            img_tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)

            # Encode VAE Safe
            try:
                latent = vae.encode(img_tensor[:,:,:,:3]) # [1, 4, H/8, W/8]
            except:
                latent = vae.encode(img_tensor)

            # Expand for Video
            lat_sample = latent["samples"]
            lat_batch = lat_sample.repeat(num_frames, 1, 1, 1)

            # Sample
            seed = torch.randint(0, 2**32 - 1, (1,)).item()
            samples = nodes.common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                cond_pos, cond_neg, {"samples": lat_batch}, denoise=denoise
            )[0]["samples"]

            # Decode
            pixels = vae.decode(samples) # [F, H, W, 3]

            # Save
            frames = []
            for f in range(pixels.shape[0]):
                p = (pixels[f].cpu().numpy() * 255).astype(np.uint8)
                frames.append(Image.fromarray(p))

            out_name = f"scene_{i}_{uuid.uuid4().hex[:6]}.mp4"
            out_path = os.path.join(temp_dir, out_name)
            export_to_video(frames, out_path, fps=fps)
            video_paths.append(out_path)

            cleanup_vram()

        return (video_paths,)

class OracleVoiceKokoro:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "storyboard_json": ("STRING", {"forceInput": True})
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_json",)
    FUNCTION = "gen_voice"
    CATEGORY = "🪬 OracleMotion"

    def gen_voice(self, storyboard_json):
        try:
             import soundfile as sf
             from kokoro_onnx import Kokoro
        except ImportError:
             raise ImportError("Missing requirements: soundfile, kokoro-onnx")

        scenes = json.loads(storyboard_json)
        temp_dir = get_temp_dir()

        # Path logic
        base_kokoro = os.path.join(folder_paths.models_dir, "Kokoro")
        model_path = os.path.join(base_kokoro, "kokoro-v0_19.onnx")
        voices_path = os.path.join(base_kokoro, "voices.json")

        if not os.path.exists(model_path):
             print(f"Kokoro not found at {model_path}. Please download it.")
             return (storyboard_json,)

        kokoro = Kokoro(model_path, voices_path)

        for scene in scenes:
            text = scene.get("dialogue", "")
            if text:
                voice = scene.get("voice_name", "af_bella")
                try:
                    samples, sample_rate = kokoro.create(text, voice=voice, speed=1.0)
                    duration = len(samples) / sample_rate

                    fname = f"audio_{scene['scene_id']}_{uuid.uuid4().hex[:6]}.wav"
                    fpath = os.path.join(temp_dir, fname)
                    sf.write(fpath, samples, sample_rate)

                    scene["audio_path"] = fpath
                    scene["duration"] = duration + 0.5 # Padding
                except Exception as e:
                    print(f"Voice Gen Error: {e}")
                    scene["duration"] = 4.0
            else:
                scene["duration"] = 4.0

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
    RETURN_NAMES = ("enhanced_json",)
    FUNCTION = "inject"
    CATEGORY = "🪬 OracleMotion"

    def inject(self, storyboard_json, audio_batch):
        scenes = json.loads(storyboard_json)
        temp_dir = get_temp_dir()

        # ComfyUI Audio: {"waveform": [Batch, Channels, Samples], "sample_rate": int}
        waveforms = audio_batch["waveform"]
        sr = audio_batch["sample_rate"]

        import soundfile as sf

        for i, scene in enumerate(scenes):
            if i < waveforms.shape[0]:
                clip = waveforms[i]
                # Robust Mono/Stereo
                if clip.dim() == 1:
                    clip = clip.unsqueeze(0)

                audio_np = clip.cpu().numpy().T
                duration = audio_np.shape[0] / sr

                fname = f"injected_{i}_{uuid.uuid4().hex[:6]}.wav"
                fpath = os.path.join(temp_dir, fname)
                sf.write(fpath, audio_np, sr)

                scene["audio_path"] = fpath
                scene["duration"] = duration
            else:
                pass

        return (json.dumps(scenes, indent=2),)

class OraclePostProduction:
    @classmethod
    def INPUT_TYPES(s):
        from .utils import get_font_path
        return {
            "required": {
                "enhanced_storyboard_json": ("STRING", {"forceInput": True}),
                "font_size": ("INT", {"default": 60}),
                "font_color": ("STRING", {"default": "#FFD700"}),
                "stroke_width": ("INT", {"default": 4}),
                "position_y": ("INT", {"default": 100}),
                "preview_mode": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "video_paths": ("LIST",),
                "preview_background": ("IMAGE",),
                "font_path": ("STRING", {"default": get_font_path()}),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("final_video", "preview_image")
    FUNCTION = "process"
    CATEGORY = "🪬 OracleMotion"
    OUTPUT_NODE = True

    def process(self, enhanced_storyboard_json, font_size, font_color, stroke_width, position_y, preview_mode, video_paths=None, preview_background=None, font_path=None):
        scenes = json.loads(enhanced_storyboard_json)

        # Setup Font
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        def get_text_size(draw, text, font):
            # Pillow 10+ Compatible
            try:
                left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
                return right - left, bottom - top
            except:
                # Fallback
                return font.getsize(text)

        # --- PREVIEW ---
        if preview_mode:
            # Create Canvas
            W, H = 1080, 1920
            if preview_background is not None:
                # Comfy [1, H, W, 3]
                i = 255. * preview_background[0].cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8)).convert("RGB")
                W, H = img.size
            else:
                img = Image.new("RGB", (W, H), (0, 0, 0))

            draw = ImageDraw.Draw(img)

            # Find Text
            text = "SAMPLE CAPTION"
            for s in scenes:
                if s.get("dialogue"):
                    text = s.get("dialogue")
                    break

            tw, th = get_text_size(draw, text, font)
            x = (W - tw) / 2
            y = H - position_y - th

            draw.text((x, y), text, font=font, fill=font_color, stroke_width=stroke_width, stroke_fill="black")

            res = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
            return ("", res)

        # --- RENDER ---
        from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip

        clips = []
        for i, path in enumerate(video_paths):
            if os.path.exists(path):
                clip = VideoFileClip(path)
                if i < len(scenes):
                    apath = scenes[i].get("audio_path")
                    if apath and os.path.exists(apath):
                        aclip = AudioFileClip(apath)
                        clip = clip.set_audio(aclip)
                        if abs(clip.duration - aclip.duration) > 0.5:
                            clip = clip.set_duration(aclip.duration)
                clips.append(clip)

        if not clips: return ("", torch.zeros((1,512,512,3)))

        final = concatenate_videoclips(clips, method="compose")

        # Prepare Captions
        captions = []
        curr = 0
        for i, c in enumerate(clips):
            txt = scenes[i].get("dialogue", "")
            if txt:
                captions.append({"start": curr, "end": curr+c.duration, "text": txt})
            curr += c.duration

        def burn(get_frame, t):
            frame = get_frame(t)
            active = None
            for c in captions:
                if c["start"] <= t < c["end"]:
                    active = c["text"]
                    break

            if active:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                W, H = img.size
                tw, th = get_text_size(draw, active, font)
                x = (W - tw) / 2
                y = H - position_y - th
                draw.text((x, y), active, font=font, fill=font_color, stroke_width=stroke_width, stroke_fill="black")
                return np.array(img)
            return frame

        final_burned = final.fl(burn)

        out_name = f"viral_{uuid.uuid4().hex[:6]}.mp4"
        out_path = os.path.join(get_temp_dir(), out_name)

        final_burned.write_videofile(out_path, fps=24, codec="libx264", audio_codec="aac", temp_audiofile="temp-audio.m4a", remove_temp=True)

        final.close()
        for c in clips: c.close()

        return (out_path, torch.zeros((1,512,512,3)))

# --- MERGED NODE: QWEN LOADER (FT + DEBUG + AUTO-DIR) ---
class OracleQwenLoader:
    @classmethod
    def INPUT_TYPES(s):
        import folder_paths

        # 1. Define and Create Paths
        base_tts_path = os.path.join(folder_paths.models_dir, "tts")
        ft_dir = os.path.join(base_tts_path, "finetuned_model")

        # Auto-create directories to prevent errors
        os.makedirs(base_tts_path, exist_ok=True)
        os.makedirs(ft_dir, exist_ok=True)

        # 2. Scan for Fine-Tunes
        ft_models = ["None (Use Base Model)"]
        try:
            if os.path.exists(ft_dir):
                for item in os.listdir(ft_dir):
                    item_path = os.path.join(ft_dir, item)
                    if os.path.isdir(item_path):
                        ft_models.append(item)
                        # Scan one level deep for "Speaker/Version" structure
                        for sub in os.listdir(item_path):
                            if os.path.isdir(os.path.join(item_path, sub)):
                                ft_models.append(f"{item}/{sub}")
        except Exception as e:
            print(f"⚠️ [OracleMotion] Error scanning fine-tunes: {e}")

        return {
            "required": {
                "repo_id": (["Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"], {"default": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"}),
                "fine_tuned_model": (sorted(ft_models),),
                "precision": (["bf16", "fp16", "fp32"], {"default": "bf16"}),
            }
        }

    RETURN_TYPES = ("QWEN_MODEL",)
    RETURN_NAMES = ("qwen_model",)
    FUNCTION = "load_model"
    CATEGORY = "🪬 OracleMotion"

    def load_model(self, repo_id, fine_tuned_model, precision):
        print(f"\n[OracleMotion:Loader] 🔵 Initializing Qwen3-TTS System...")

        if Qwen3TTSModel is None:
            print(f"[OracleMotion:Loader] ❌ CRITICAL: 'qwen-tts' library missing.")
            raise ImportError("Please install 'qwen-tts' via requirements.txt")

        import folder_paths
        import comfy.model_management as mm
        from huggingface_hub import snapshot_download

        device = mm.get_torch_device()
        print(f"[OracleMotion:Loader] ⚙️ Device: {device} | Precision: {precision}")

        # 1. Load Base Model
        save_dir = os.path.join(folder_paths.models_dir, "tts", repo_id.split("/")[-1])
        if not os.path.exists(save_dir):
            print(f"[OracleMotion:Loader] ⬇️ Downloading Base Model: {repo_id}...")
            snapshot_download(repo_id, local_dir=save_dir)
            print(f"[OracleMotion:Loader] ✅ Download Complete.")

        dtype = torch.float32
        if precision == "bf16": dtype = torch.bfloat16
        elif precision == "fp16": dtype = torch.float16

        print(f"[OracleMotion:Loader] 🚀 Loading Base Model into VRAM...")
        try:
            model = Qwen3TTSModel.from_pretrained(save_dir, device_map=device, dtype=dtype)
        except Exception as e:
            print(f"[OracleMotion:Loader] ❌ Failed to load base model: {e}")
            raise e

        # 2. Apply Fine-Tune (If selected)
        if fine_tuned_model != "None (Use Base Model)":
            ft_base_path = os.path.join(folder_paths.models_dir, "tts", "finetuned_model")

            # Handle path resolution
            if "/" in fine_tuned_model:
                parts = fine_tuned_model.split("/")
                ckpt_path = os.path.join(ft_base_path, *parts)
            else:
                ckpt_path = os.path.join(ft_base_path, fine_tuned_model)

            bin_file = os.path.join(ckpt_path, "pytorch_model.bin")

            if os.path.exists(bin_file):
                print(f"[OracleMotion:Loader] ♻️ Applying Fine-Tune weights: {fine_tuned_model}")
                try:
                    state_dict = torch.load(bin_file, map_location="cpu")
                    keys = model.model.load_state_dict(state_dict, strict=False)
                    print(f"[OracleMotion:Loader] ✅ Weights merged. (Missing keys: {len(keys.missing_keys)} - expected for PEFT)")

                    # Inspect Config for Custom Speaker Names
                    cfg_file = os.path.join(ckpt_path, "config.json")
                    if os.path.exists(cfg_file):
                        with open(cfg_file, 'r') as f:
                            cfg_data = json.load(f)
                            if "talker_config" in cfg_data and "spk_id" in cfg_data["talker_config"]:
                                spk_ids = cfg_data["talker_config"]["spk_id"]
                                print(f"[OracleMotion:Loader] ℹ️ Custom Speakers Found: {list(spk_ids.keys())}")
                except Exception as e:
                    print(f"[OracleMotion:Loader] ❌ Error applying fine-tune: {e}")
            else:
                print(f"[OracleMotion:Loader] ⚠️ Warning: pytorch_model.bin not found at {ckpt_path}")

        return (model,)

# --- NODE: QWEN VOICE (DEBUG ENABLED) ---
class OracleVoiceQwen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "storyboard_json": ("STRING", {"forceInput": True}),
                "qwen_model": ("QWEN_MODEL",),
                "default_gender": (["Female", "Male"], {"default": "Female"}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("enhanced_json",)
    FUNCTION = "gen_voice_qwen"
    CATEGORY = "🪬 OracleMotion"

    def gen_voice_qwen(self, storyboard_json, qwen_model, default_gender, seed):
        import soundfile as sf
        print(f"\n[OracleMotion:Voice] 🎙️ Starting Voice Generation Batch...")

        try:
            scenes = json.loads(storyboard_json)
            print(f"[OracleMotion:Voice] 📄 Parsed {len(scenes)} scenes from JSON.")
        except Exception as e:
            print(f"[OracleMotion:Voice] ❌ JSON Parsing Error: {e}")
            return (storyboard_json,)

        temp_dir = get_temp_dir()

        # Set Seed
        print(f"[OracleMotion:Voice] 🎲 Applying Seed: {seed}")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for i, scene in enumerate(scenes):
            scene_id = scene.get('scene_id', i)
            text = scene.get("dialogue", "")

            print(f"--- [Scene {scene_id}] ---")

            if text:
                # 1. Construct the Acting Instruction
                emotion = scene.get("audio_emotion", "Neutral")
                voice_name = scene.get("voice_name", default_gender)
                action = scene.get("action_description", "")

                # Qwen VoiceDesign Prompt format
                instruct = f"Gender: {voice_name}\nEmotion: {emotion}\nLanguage: Auto"
                if action:
                    instruct += f"\nContext: {action}"

                print(f"[OracleMotion:Voice] 📝 Instruction:\n   Gender: {voice_name}\n   Emotion: {emotion}")
                print(f"[OracleMotion:Voice] 🗣️ Text: \"{text[:30]}...\"")

                try:
                    # 2. Generate
                    print(f"[OracleMotion:Voice] ⏳ Generating Audio...")
                    wavs, sr = qwen_model.generate_voice_design(
                        text=text,
                        instruct=instruct,
                        output_dir=None
                    )

                    if not wavs or len(wavs) == 0:
                        raise ValueError("Model returned empty audio list.")

                    # 3. Process Output
                    audio_data = wavs[0]
                    duration = len(audio_data) / sr
                    print(f"[OracleMotion:Voice] ✅ Generated. Duration: {duration:.2f}s")

                    # 4. Save
                    fname = f"qwen_audio_{scene_id}_{uuid.uuid4().hex[:6]}.wav"
                    fpath = os.path.join(temp_dir, fname)
                    sf.write(fpath, audio_data, sr)
                    print(f"[OracleMotion:Voice] 💾 Saved to: {fname}")

                    # 5. Update JSON
                    scene["audio_path"] = fpath
                    scene["duration"] = duration + 0.2 # Slight padding

                except Exception as e:
                    print(f"[OracleMotion:Voice] ❌ Generation Error in Scene {scene_id}: {e}")
                    print(f"[OracleMotion:Voice] ⚠️ Fallback to silent 3.0s duration.")
                    scene["duration"] = 3.0
            else:
                print(f"[OracleMotion:Voice] 🔇 No dialogue detected. Skipping generation.")
                scene["duration"] = 3.0

            # VRAM Cleanup after each scene to prevent OOM
            cleanup_vram()

        print(f"[OracleMotion:Voice] 🎉 Batch Complete. Returning updated JSON.\n")
        return (json.dumps(scenes, indent=2),)
