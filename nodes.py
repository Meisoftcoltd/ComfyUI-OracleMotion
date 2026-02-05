import os
import torch
import json
import uuid
from .utils import get_temp_dir, cleanup_vram, parse_json_output

# Lazy imports to avoid hard crashes if dependencies are missing during initial load
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

class OracleBrain:
    def __init__(self):
        pass

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
                raise ImportError("faster_whisper is required for audio processing. Please install it.")

            print(f"Transcribing audio from {audio_path}...")
            # Run on CPU or GPU depending on availability, but 'cuda' is safer if torch is there
            # However, faster_whisper manages its own device. Let's try cuda if available.
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = WhisperModel("base", device=device, compute_type="float16" if device=="cuda" else "int8")
            segments, _ = model.transcribe(audio_path)
            transcribed_text = " ".join([segment.text for segment in segments])
            print(f"Transcription: {transcribed_text}")
            text_input += "\n\nAudio Transcript: " + transcribed_text

        # 2. LLM Generation
        if OpenAI is None:
            raise ImportError("openai library is required. Please install it.")

        client = OpenAI(api_key=api_key, base_url=base_url)

        SYSTEM_PROMPT = """
You are an expert Director of Photography and Screenwriter.
Your task is to convert the user's narrative or idea into a structured visual storyboard.

RULES:
1. Output MUST be a valid JSON list of objects.
2. Each object represents a scene.
3. Keys required per object:
   - "scene_id": (int) sequential index.
   - "visual_prompt": (string) detailed visual description for an image generator (Subject, Action, Lighting, Style).
   - "action_description": (string) Brief description of movement for the video generator.
   - "reference_idx": (int) The index of the reference image to use for this scene (0, 1, 2...). Set to -1 if no specific reference is needed.
   - "duration": (float) estimated duration in seconds.

Ensure the visual prompts are descriptive and consistent in style.
"""

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_input}
            ]
        )

        content = response.choices[0].message.content
        parsed_json = parse_json_output(content)

        if parsed_json is None:
            # Fallback or error
            print("Failed to parse JSON from LLM output. Raw output:")
            print(content)
            # Return strict empty list structure as fallback
            return (json.dumps([]),)

        return (json.dumps(parsed_json, indent=2),)

class OracleVisualizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "storyboard_json": ("STRING",),
                "base_character_image": ("IMAGE",),
                "sdxl_ckpt": ("STRING", {"default": "stabilityai/stable-diffusion-xl-base-1.0"}),
                "global_style_prompt": ("STRING", {"multiline": True, "default": "Cinematic lighting, 8k resolution, photorealistic"}),
            },
            "optional": {
                "reference_images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("LIST",) # List of file paths
    RETURN_NAMES = ("keyframe_paths",)
    FUNCTION = "generate_keyframes"
    CATEGORY = "OracleMotion"

    def generate_keyframes(self, storyboard_json, base_character_image, sdxl_ckpt, global_style_prompt, reference_images=None):
        # Lazy imports
        from diffusers import StableDiffusionXLPipeline
        from transformers import CLIPVisionModelWithProjection
        import numpy as np
        from PIL import Image

        # Helper to convert ComfyUI Image Tensor [B,H,W,C] to List of PIL Images
        def tensor2pil(image_tensor):
            if image_tensor is None:
                return []
            images = []
            for i in range(image_tensor.shape[0]):
                img = 255. * image_tensor[i].cpu().numpy()
                img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                images.append(img)
            return images

        scenes = json.loads(storyboard_json)
        temp_dir = get_temp_dir()
        keyframe_paths = []

        # Load Image Encoder
        print("Loading Image Encoder...")
        try:
             image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                 "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                 torch_dtype=torch.float16
             )
        except Exception as e:
             raise RuntimeError(f"Failed to load Image Encoder: {e}")

        # Load SDXL
        print(f"Loading SDXL from {sdxl_ckpt}...")
        try:
            if os.path.isfile(sdxl_ckpt):
                pipe = StableDiffusionXLPipeline.from_single_file(
                    sdxl_ckpt,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    image_encoder=image_encoder
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    sdxl_ckpt,
                    torch_dtype=torch.float16,
                    image_encoder=image_encoder
                )
        except Exception as e:
            raise RuntimeError(f"Failed to load SDXL model: {e}")

        # Enable offloading
        pipe.enable_model_cpu_offload()

        # Load IP-Adapter
        print("Loading IP-Adapter...")
        try:
            pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus-sdxl_vit-h.safetensors")
            pipe.set_ip_adapter_scale(0.7)
        except Exception as e:
            print(f"Warning: Failed to load IP-Adapter ({e}). Continuing without it.")

        # Prepare Base Character Image (List of PIL)
        base_char_pil_list = tensor2pil(base_character_image)
        base_char_pil = base_char_pil_list[0] if base_char_pil_list else None

        # Prepare Reference Images
        ref_images_pil = tensor2pil(reference_images) if reference_images is not None else []

        try:
            for i, scene in enumerate(scenes):
                scene_id = scene.get("scene_id", i)
                visual_prompt = scene.get("visual_prompt", "")
                ref_idx = scene.get("reference_idx", -1)

                full_prompt = f"{visual_prompt}, {global_style_prompt}"

                # Determine IP-Adapter Images
                ip_images = []
                if base_char_pil:
                    ip_images.append(base_char_pil)

                if 0 <= ref_idx < len(ref_images_pil):
                    print(f"Scene {scene_id}: Using reference image {ref_idx}")
                    ip_images.append(ref_images_pil[ref_idx])

                # Generate
                print(f"Generating scene {scene_id}...")

                # pass ip_adapter_image only if we have images
                kwargs = {"prompt": full_prompt, "output_type": "pil"}
                if ip_images:
                     kwargs["ip_adapter_image"] = ip_images

                image = pipe(**kwargs).images[0]

                filename = f"keyframe_{scene_id}_{uuid.uuid4().hex[:6]}.png"
                filepath = os.path.join(temp_dir, filename)
                image.save(filepath)
                keyframe_paths.append(filepath)

                # Lightweight cleanup between frames
                cleanup_vram()

        finally:
            # Heavy cleanup after batch
            del pipe
            cleanup_vram()

        return (keyframe_paths,)

class OracleEngine:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keyframe_paths": ("LIST",),
                "wan_ckpt": ("STRING", {"default": "Wan-AI/Wan2.1-I2V-1.3B-Diffusers"}),
            }
        }
    RETURN_TYPES = ("LIST",) # List of video paths
    RETURN_NAMES = ("video_paths",)
    FUNCTION = "generate_videos"
    CATEGORY = "OracleMotion"

    def generate_videos(self, keyframe_paths, wan_ckpt):
        from diffusers import DiffusionPipeline
        from diffusers.utils import export_to_video
        from PIL import Image

        temp_dir = get_temp_dir()
        video_paths = []

        print(f"Loading Wan 2.1 from {wan_ckpt}...")
        try:
             # trust_remote_code=True is CRITICAL
             pipe = DiffusionPipeline.from_pretrained(wan_ckpt, trust_remote_code=True, torch_dtype=torch.float16)
        except Exception as e:
             raise RuntimeError(f"Failed to load Wan 2.1 model: {e}")

        # Enable offloading
        pipe.enable_model_cpu_offload()

        try:
            for i, frame_path in enumerate(keyframe_paths):
                if not os.path.exists(frame_path):
                    print(f"Warning: Keyframe not found at {frame_path}, skipping.")
                    continue

                print(f"Animating keyframe {i}: {frame_path}")
                image = Image.open(frame_path).convert("RGB")

                # Generate
                # Wan 2.1 specific params might vary. Assuming standard I2V args.
                # Usually takes 'image'.
                # We generate 81 frames (standard for Wan 5s)

                output = pipe(image=image, num_frames=81).frames[0]

                video_filename = f"scene_{i}_{uuid.uuid4().hex[:6]}.mp4"
                video_filepath = os.path.join(temp_dir, video_filename)

                export_to_video(output, video_filepath, fps=16)
                video_paths.append(video_filepath)

                cleanup_vram()

        finally:
            del pipe
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
