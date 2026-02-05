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

class OracleBrain:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "narrative_text": ("STRING", {"multiline": True, "default": "A cyberpunk detective walking through a rainy neon city."}),
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
    CATEGORY = "OracleMotion"

    def generate_storyboard(self, narrative_text, model_name, base_url, api_key, audio_path=""):
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

        SYSTEM_PROMPT = """
You are an expert Director of Photography and Screenwriter.
Your task is to convert the user's narrative or idea into a structured visual storyboard.

RULES:
1. Output MUST be a valid JSON list of objects.
2. Each object represents a scene (keyframe).
3. Keys required per object:
   - "frame": (int) frame number (assume 24fps, so 0, 48, 96...).
   - "prompt": (string) detailed visual description for image generation.
   - "action": (string) description of movement for video generation.
   - "path": (string) leave empty, used for reference image path later.

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
    CATEGORY = "OracleMotion"

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
        # If user_scenes is empty, return ai_scenes.
        # If user_scenes has content, we trust the user.
        # But maybe we should try to merge by frame index?
        # The prompt says: "If user_edits is not empty, it overrides storyboard_json".
        # This implies a full override or a smart merge.
        # Let's assume full override if user has "touched" the timeline.
        # If the user_edits came from the widget which was populated by storyboard_json initially...
        # The JS widget logic I wrote initializes from user_edits widget value.
        # If user_edits is "[]" (default), we should probably output ai_scenes.
        # But we also want the JS side to SEE the ai_scenes.
        # This is a bit of a chicken-egg problem in ComfyUI standard nodes without extra message passing.
        # Usually, one would output the JSON, and the user would see it in the next run?
        # Or, the JS populates itself from the input?

        # Given the constraint: "If user_edits is not empty, it overrides"

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
                "frame": scene.get("frame", i * 40), # Default spacing if missing
                "prompt": scene.get("prompt", ""),
                "action": scene.get("action", ""),
                "path": scene.get("path", "")
            }
            validated_scenes.append(validated_scene)

        # Sort by frame just in case
        validated_scenes.sort(key=lambda x: int(x["frame"]))

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
    CATEGORY = "OracleMotion"

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
                frame_idx = scene.get("frame", i * 24)
                prompt = scene.get("prompt", "")
                ref_path = scene.get("path", "")

                full_prompt = f"{prompt}, {global_style_prompt}"

                print(f"Generating keyframe for frame {frame_idx}...")

                # Check if we have a valid reference path to load (for img2img or controlnet if implemented)
                # Current scope says "Static Keyframe Gen" using SDXL logic.
                # The instructions didn't specify ControlNet/Adapter for this specific pass,
                # but earlier plan mentioned OracleVisualizer (SDXL + IP-Adapter).
                # However, the "Final Instructions" reduced complexity to just "SDXL logic" and "Keep it self-contained".
                # I will stick to text-to-image for now unless IP-Adapter is strictly required by the prompt.
                # The prompt "Use the existing StableDiffusionXLPipeline logic" suggests keeping what was there if valid.
                # But the user also said "Stick to diffusers (SDXL) for this node to keep it self-contained."
                # I will do standard T2I. If ref_path is present, maybe I should load it?
                # The plan for Visualizer doesn't explicitly mention IP-Adapter in the final instruction block.
                # It just says "Generate static keyframes".
                # I will stick to T2I to ensure stability, but I'll add support for init image if I can?
                # No, let's just do T2I for keyframes as per prompt "Static Keyframe Gen".

                # If valid ref path provided by user, use it.
                from .utils import load_image_as_pil
                if ref_path and os.path.exists(ref_path):
                     print(f"Using provided reference image: {ref_path}")
                     # Just copy/load it. We might want to ensure it's RGB or resized?
                     # For now, just load and save to temp to ensure uniformity in list
                     image = Image.open(ref_path).convert("RGB")
                else:
                     print(f"Generating new keyframe via SDXL...")
                     image = pipe(prompt=full_prompt, output_type="pil").images[0]

                filename = f"keyframe_{frame_idx}_{uuid.uuid4().hex[:6]}.png"
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
                "frames": ("INT", {"default": 81, "min": 1}),
                "steps": ("INT", {"default": 20, "min": 1}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "step": 0.1}),
                "sampler_name": (samplers,),
                "scheduler": (schedulers,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                 "positive": ("STRING", {"default": "high quality, motion, animation", "multiline": True}),
                 "negative": ("STRING", {"default": "low quality, static, watermark", "multiline": True}),
            }
        }

    RETURN_TYPES = ("LIST",)
    RETURN_NAMES = ("video_paths",)
    FUNCTION = "animate_scenes"
    CATEGORY = "OracleMotion"

    def animate_scenes(self, model, vae, clip, keyframe_paths, frames, steps, cfg, sampler_name, scheduler, denoise, positive="high quality", negative="low quality"):
        from diffusers.utils import export_to_video

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
            # Resize if needed? Assuming model handles it or VAE does.
            # Ideally should match model expectations (e.g. 1024x1024 or similar)
            # For now, keeping original size or ensuring multiple of 8
            w, h = img.size
            # simplistic resize to nearest 8
            w = (w // 8) * 8
            h = (h // 8) * 8
            img = img.resize((w, h))

            img_np = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0) # [1, H, W, 3]

            # VAE Encode expects [B, H, W, C] ? No, Comfy VAE Encode expects [B, H, W, C] in nodes.VAEEncode
            # But internally: vae.encode(pixels)
            # pixels should be [B, H, W, C]

            # Use standard VAE Encode logic
            pixel_samples = img_tensor

            # Move to VAE device to avoid runtime error
            # ComfyUI VAE objects usually have a 'device' attribute or 'first_stage_model.device'
            # We try to detect it safely.
            try:
                # Common Comfy VAE wrapper
                if hasattr(vae, "device"):
                    target_device = vae.device
                elif hasattr(vae, "first_stage_model"):
                    target_device = vae.first_stage_model.device
                else:
                    # Fallback to model's load device if available, or cuda
                    target_device = model.load_device if hasattr(model, "load_device") else "cuda"

                pixel_samples = pixel_samples.to(target_device)
            except Exception as e:
                print(f"Warning: Could not move tensor to VAE device: {e}")

            # Check if vae.encode returns a dict or tensor
            latent = vae.encode(pixel_samples)
            if isinstance(latent, torch.Tensor):
                 return {"samples": latent}
            return latent # {"samples": tensor [1, 4, h, w]}

        def run_sampler(model, vae, latent_input, steps, cfg, sampler_name, scheduler, positive, negative, denoise):
            # latent_input is {"samples": tensor}

            # Run KSampler
            # nodes.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0)
            # Seed: random
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

            try:
                # common_ksampler returns (latent, )
                result_latent = nodes.common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, latent_input, denoise=denoise
                )[0]
            except Exception as e:
                print(f"Sampler Error: {e}")
                raise e

            # Decode
            # vae.decode(latent["samples"]) returns pixels [B, H, W, C]?
            # Comfy VAE decode: vae.decode(samples) -> pixels
            pixels = vae.decode(result_latent["samples"])
            return pixels # Tensor [B, H, W, C]

        current_input_path = None
        last_generated_frame_path = None

        for i, keyframe_path in enumerate(keyframe_paths):
            print(f"Animating Scene {i}...")

            # Determine Input
            if i == 0:
                current_input_path = keyframe_path
            else:
                # Use last frame of previous scene
                current_input_path = last_generated_frame_path

            if not current_input_path or not os.path.exists(current_input_path):
                print(f"Skipping scene {i}: Input path missing {current_input_path}")
                continue

            # 1. Encode Input Image to Latent
            start_latent = load_image_as_latent(current_input_path, vae)

            # 2. Expand Latent for Video (Batch Size = frames)
            # start_latent["samples"] is [1, 4, H, W]
            # We want [frames, 4, H, W]
            # This effectively makes a "video" where every frame is the start image.
            # KSampler with denoise=1.0 will hallucinate motion from this (or use it as condition).

            lat_tensor = start_latent["samples"]
            lat_batch = lat_tensor.repeat(frames, 1, 1, 1)
            latent_input = {"samples": lat_batch}

            # 3. Run Sampler
            # Note: denoise is passed from input (default 1.0)
            pixel_frames = run_sampler(model, vae, latent_input, steps, cfg, sampler_name, scheduler, cond_pos, cond_neg, denoise)

            # 4. Save Video
            # pixel_frames is [frames, H, W, 3] tensor
            # Convert to List of PIL
            import numpy as np
            from PIL import Image

            frames_pil = []
            for f in range(pixel_frames.shape[0]):
                 # clip 0-1, convert to 255 byte
                 p = pixel_frames[f].cpu().numpy()
                 p = np.clip(p, 0, 1) * 255
                 p = p.astype(np.uint8)
                 frames_pil.append(Image.fromarray(p))

            video_filename = f"scene_{i}_{uuid.uuid4().hex[:6]}.mp4"
            video_filepath = os.path.join(temp_dir, video_filename)

            # Use export_to_video from diffusers (simplest) or moviepy
            export_to_video(frames_pil, video_filepath, fps=16)
            video_paths.append(video_filepath)

            # 5. Save Last Frame for Next Scene
            last_frame_pil = frames_pil[-1]
            last_frame_filename = f"last_frame_scene_{i}_{uuid.uuid4().hex[:6]}.png"
            last_generated_frame_path = os.path.join(temp_dir, last_frame_filename)
            last_frame_pil.save(last_generated_frame_path)

            cleanup_vram()

        return (video_paths,)

class OracleEditor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_paths": ("LIST",),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("final_video_path",)
    FUNCTION = "stitch_videos"
    CATEGORY = "OracleMotion"

    def stitch_videos(self, video_paths):
        from moviepy.editor import VideoFileClip, concatenate_videoclips

        temp_dir = get_temp_dir()
        clips = []

        try:
            for path in video_paths:
                if os.path.exists(path):
                    clips.append(VideoFileClip(path))
                else:
                    print(f"Warning: Video clip not found at {path}, skipping.")

            if not clips:
                raise RuntimeError("No valid video clips to stitch.")

            # method="compose" is safer if clips have slight inconsistencies
            final_clip = concatenate_videoclips(clips, method="compose")

            output_filename = f"final_movie_{uuid.uuid4().hex[:6]}.mp4"
            output_path = os.path.join(temp_dir, output_filename)

            # Write video file. Using libx264 for compatibility.
            # FPS: If not specified, moviepy uses fps of first clip.
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        except Exception as e:
            raise RuntimeError(f"Failed to stitch videos: {e}")
        finally:
            # Close clips to release file handles
            for clip in clips:
                try: clip.close()
                except: pass

        return (output_path,)
