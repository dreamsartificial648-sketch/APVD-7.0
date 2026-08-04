"""
APVD Dream Canvas
A standalone companion app for APVD v7.0.

Drop this file next to your APVD app.py, model.py, utils.py, Models/, Memory/ folders,
then run:
    python apvd_dream_canvas.py

Main idea:
- Type a prompt like "chair" or "car".
- The app searches Models/ for a matching APVD .pt checkpoint.
- It generates a dream-object variant and places it on a movable layer canvas.
- It can also import normal images or Memory images.
- Each layer can be moved, scaled, rotated, cut out, softened, blended, duplicated, deleted, and exported.

This app is intentionally separate from app.py so it is safer to test.
"""
from __future__ import annotations

import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
    from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps, ImageTk
except Exception as exc:
    print("Missing required package:", exc)
    input("Press Enter to exit...")
    raise SystemExit(5) from exc

try:
    import torch
except Exception:
    torch = None

# Optional APVD project imports. The app still opens without them, but .pt generation needs them.
try:
    from model import VAE, get_device
except Exception:
    VAE = None

    def get_device():
        if torch is not None and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu") if torch is not None else None

try:
    from utils import select_model_path_for_prompt, tensor_to_pil, wavelet_to_rgb, rgb_to_wavelet
except Exception:
    select_model_path_for_prompt = None
    tensor_to_pil = None
    wavelet_to_rgb = None
    rgb_to_wavelet = None

APP_BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_BASE_DIR / "Models"
MEMORY_DIR = APP_BASE_DIR / "Memory"
OUTPUTS_DIR = APP_BASE_DIR / "Dream_Canvas_Output"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
MODEL_EXTENSIONS = {".pt", ".pth"}


def safe_torch_load(path: Path, *, map_location=None):
    if torch is None:
        raise RuntimeError("PyTorch is not installed, so APVD .pt models cannot be loaded.")
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def slugify(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._ -]+", "_", text).strip()
    text = re.sub(r"\s+", "_", text)
    return text or "dream_canvas"


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS], key=lambda p: str(p).lower())


def list_models(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.rglob("*") if p.suffix.lower() in MODEL_EXTENSIONS], key=lambda p: str(p).lower())


def score_path_for_prompt(path: Path, prompt: str) -> float:
    """Small fallback prompt matcher if APVD's utils.select_model_path_for_prompt is unavailable."""
    prompt_words = [w for w in re.split(r"[^a-z0-9]+", prompt.lower()) if w]
    haystack = " ".join(part.lower() for part in path.parts[-5:])
    if not prompt_words:
        return 0.0
    score = 0.0
    for word in prompt_words:
        if word in haystack:
            score += 3.0
        for token in re.split(r"[^a-z0-9]+", haystack):
            if token and (word in token or token in word):
                score += 0.75
    if path.stem.lower() == prompt.lower().strip():
        score += 10.0
    return score


def find_model_for_prompt(prompt: str, models_dir: Path = MODELS_DIR) -> tuple[Path, str]:
    if select_model_path_for_prompt is not None:
        try:
            return select_model_path_for_prompt(models_dir, prompt)
        except Exception:
            pass

    models = list_models(models_dir)
    if not models:
        raise FileNotFoundError(f"No APVD .pt/.pth models found in {models_dir.resolve()}")
    ranked = sorted(((score_path_for_prompt(p, prompt), p) for p in models), key=lambda x: x[0], reverse=True)
    best_score, best_path = ranked[0]
    if best_score <= 0:
        raise FileNotFoundError(f"No model looked related to prompt: {prompt!r}")
    return best_path, f"fallback filename match score {best_score:.2f}"


def guess_checkpoint_mode(checkpoint: dict) -> str:
    mode = checkpoint.get("reconstruction_mode") if isinstance(checkpoint, dict) else None
    if mode in {"RGB VAE", "Wavelet"}:
        return mode
    saved_in_channels = checkpoint.get("in_channels") if isinstance(checkpoint, dict) else None
    if saved_in_channels is None and isinstance(checkpoint, dict):
        state = checkpoint.get("model_state_dict", {})
        first_weight = state.get("encoder.0.weight") if isinstance(state, dict) else None
        if hasattr(first_weight, "shape") and len(first_weight.shape) >= 2:
            saved_in_channels = int(first_weight.shape[1])
    return "Wavelet" if int(saved_in_channels or 3) == 12 else "RGB VAE"


def load_apvd_model(path: Path, device):
    if torch is None or VAE is None:
        raise RuntimeError("APVD model loading needs torch plus your local model.py next to this file.")
    checkpoint = safe_torch_load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("This checkpoint does not look like an APVD VAE checkpoint with model_state_dict.")

    mode = guess_checkpoint_mode(checkpoint)
    in_ch = 12 if mode == "Wavelet" else 3
    out_ch = 12 if mode == "Wavelet" else 3
    output_size = tuple(checkpoint.get("output_size", (256, 256)))
    output_activation = checkpoint.get("output_activation", "identity" if mode == "Wavelet" else "sigmoid")
    model = VAE(
        latent_dim=int(checkpoint.get("latent_dim", 256)),
        in_channels=in_ch,
        out_channels=out_ch,
        output_size=output_size,
        output_activation=output_activation,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    return model, checkpoint, mode


def decode_output_to_pil(output, mode: str) -> Image.Image:
    if torch is None:
        raise RuntimeError("PyTorch is required for decoding APVD output.")
    with torch.no_grad():
        tensor = output.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if mode == "Wavelet":
            if wavelet_to_rgb is None:
                raise RuntimeError("This is a Wavelet checkpoint, but utils.wavelet_to_rgb could not be imported.")
            tensor = wavelet_to_rgb(tensor).clamp(0.0, 1.0)
        else:
            tensor = tensor.clamp(0.0, 1.0)
        if tensor_to_pil is not None:
            return tensor_to_pil(tensor).convert("RGBA")
        arr = (tensor.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
        return Image.fromarray(arr, "RGB").convert("RGBA")


def generate_from_apvd_checkpoint(model_path: Path, intensity: float, device) -> Image.Image:
    if torch is None:
        raise RuntimeError("PyTorch is required for APVD generation.")
    model, _checkpoint, mode = load_apvd_model(model_path, device)
    latent_dim = int(getattr(model, "latent_dim", 256))
    with torch.no_grad():
        latent = torch.randn(1, latent_dim, device=device) * float(intensity)
        if hasattr(model, "decode"):
            output = model.decode(latent)
        else:
            raise RuntimeError("Loaded VAE model has no decode() method.")
    if getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()
    return decode_output_to_pil(output, mode)


def _checkpoint_rgb_size(checkpoint: dict, mode: str) -> tuple[int, int]:
    output_size = tuple(checkpoint.get("output_size", (256, 256))) if isinstance(checkpoint, dict) else (256, 256)
    if len(output_size) != 2:
        output_size = (256, 256)
    w, h = int(output_size[0]), int(output_size[1])
    if mode == "Wavelet":
        return max(32, w * 2), max(32, h * 2)
    return max(32, w), max(32, h)


def _pil_to_apvd_batch(image: Image.Image, size: tuple[int, int], mode: str, device):
    if torch is None:
        raise RuntimeError("PyTorch is required for APVD reconstruction.")
    img = image.convert("RGB").resize(size, Image.Resampling.BICUBIC)
    arr = torch.from_numpy(__import__("numpy").asarray(img)).float() / 255.0
    tensor = arr.permute(2, 0, 1).contiguous()
    if mode == "Wavelet":
        if rgb_to_wavelet is None:
            raise RuntimeError("This is a Wavelet checkpoint, but utils.rgb_to_wavelet could not be imported.")
        tensor = rgb_to_wavelet(tensor)
    return tensor.unsqueeze(0).to(device)


def reconstruct_with_apvd_checkpoint(model_path: Path, image: Image.Image, variation: float, cleanup: float, device) -> Image.Image:
    """Reconstruct an input image through the selected APVD memory model.

    This is the conditioning trick used by Dream Expand: the model sees the larger
    white/blank canvas plus the original image in the middle, then the output is
    used as the imagined outside area. The original image is pasted back afterward.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required for APVD reconstruction.")
    model, checkpoint, mode = load_apvd_model(model_path, device)
    rgb_size = _checkpoint_rgb_size(checkpoint, mode)
    batch = _pil_to_apvd_batch(image, rgb_size, mode, device)
    variation = max(0.0, float(variation))
    cleanup = max(0.0, min(1.0, float(cleanup)))

    with torch.no_grad():
        if hasattr(model, "encode") and hasattr(model, "decode"):
            encoded = model.encode(batch)
            if isinstance(encoded, (tuple, list)) and len(encoded) >= 2:
                mu, logvar = encoded[0], encoded[1]
            else:
                mu, logvar = encoded, None
            noise = torch.randn_like(mu) * (variation * 0.55)
            latent = mu + noise
            output = model.decode(latent)
        else:
            raw = model(batch)
            output = raw[0] if isinstance(raw, (tuple, list)) else raw

        # Optional extra reconstruction passes. This works like a small "memory cleanup"
        # loop: higher cleanup re-feeds the output so it becomes less pasted/noisy.
        extra_passes = int(round(cleanup * 3))
        for _ in range(extra_passes):
            cleaned_batch = output.detach()
            if hasattr(model, "encode") and hasattr(model, "decode"):
                encoded = model.encode(cleaned_batch)
                if isinstance(encoded, (tuple, list)) and len(encoded) >= 2:
                    mu = encoded[0]
                else:
                    mu = encoded
                output = model.decode(mu)
            else:
                raw = model(cleaned_batch)
                output = raw[0] if isinstance(raw, (tuple, list)) else raw

    if getattr(device, "type", "") == "cuda":
        torch.cuda.empty_cache()
    result = decode_output_to_pil(output, mode).resize(image.size, Image.Resampling.BICUBIC)
    if cleanup > 0:
        # Light local polish so the filled area has less crunchy VAE fuzz.
        smooth = result.filter(ImageFilter.GaussianBlur(radius=0.35 + cleanup * 0.65))
        result = Image.blend(result, smooth, cleanup * 0.22)
        result = ImageEnhance.Contrast(result).enhance(1.0 + cleanup * 0.08)
        result = ImageEnhance.Color(result).enhance(1.0 - cleanup * 0.04)
    return result.convert("RGBA")


def build_expanded_memory_image(
    original: Image.Image,
    model_path: Path,
    device,
    *,
    border_px: int = 128,
    variation: float = 0.45,
    cleanup: float = 0.65,
    feather_px: int = 40,
    void_color: str = "#ffffff",
) -> Image.Image:
    """Create a larger canvas and ask APVD to imagine the outside area."""
    original = original.convert("RGBA")
    border_px = max(1, int(border_px))
    feather_px = max(0, int(feather_px))
    new_w = max(32, original.width + border_px * 2)
    new_h = max(32, original.height + border_px * 2)

    try:
        seed = Image.new("RGBA", (new_w, new_h), void_color)
    except Exception:
        seed = Image.new("RGBA", (new_w, new_h), "#ffffff")
    seed.alpha_composite(original, dest=(border_px, border_px))

    dream = reconstruct_with_apvd_checkpoint(model_path, seed, variation, cleanup, device)

    # Preserve the user's original pixels in the center, while feathering the border
    # so APVD's imagined surroundings can softly meet the source image.
    preserve = Image.new("L", (new_w, new_h), 0)
    original_alpha = original.getchannel("A")
    preserve.paste(original_alpha, (border_px, border_px))
    if feather_px > 0:
        preserve = preserve.filter(ImageFilter.GaussianBlur(radius=feather_px))
    final = Image.composite(seed, dream, preserve).convert("RGBA")

    # A tiny seam polish around the old image boundary.
    seam = Image.new("L", (new_w, new_h), 0)
    seam_box = (border_px, border_px, border_px + original.width, border_px + original.height)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(seam)
    draw.rectangle(seam_box, outline=255, width=max(2, min(12, feather_px // 3 + 2)))
    seam = seam.filter(ImageFilter.GaussianBlur(radius=max(1, feather_px // 4)))
    softened = final.filter(ImageFilter.GaussianBlur(radius=0.7))
    final = Image.composite(softened, final, seam).convert("RGBA")
    final.alpha_composite(original, dest=(border_px, border_px))
    return final


def create_alpha_from_background(image: Image.Image, threshold: int = 38, feather: int = 2) -> Image.Image:
    """Corner-color background removal. Crude, but good for dream-object stickers."""
    img = image.convert("RGBA")
    w, h = img.size
    px = img.load()
    sample_points = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2),
    ]
    colors = [px[x, y][:3] for x, y in sample_points]
    bg = tuple(int(sum(c[i] for c in colors) / len(colors)) for i in range(3))

    rgb = img.convert("RGB")
    alpha = Image.new("L", img.size, 255)
    apx = alpha.load()
    r0, g0, b0 = bg
    for y in range(h):
        for x in range(w):
            r, g, b = rgb.getpixel((x, y))
            dist = math.sqrt((r - r0) ** 2 + (g - g0) ** 2 + (b - b0) ** 2)
            if dist < threshold:
                apx[x, y] = 0
            elif dist < threshold * 1.8:
                apx[x, y] = int(255 * ((dist - threshold) / max(1, threshold * 0.8)))
    if feather > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=feather))
    result = img.copy()
    result.putalpha(alpha)
    return result


def auto_crop_alpha(image: Image.Image, padding: int = 6) -> Image.Image:
    img = image.convert("RGBA")
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    if not bbox:
        return img
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(img.width, x1 + padding)
    y1 = min(img.height, y1 + padding)
    return img.crop((x0, y0, x1, y1))


def soften_alpha(image: Image.Image, radius: float = 1.4) -> Image.Image:
    img = image.convert("RGBA")
    alpha = img.getchannel("A").filter(ImageFilter.GaussianBlur(radius=radius))
    img.putalpha(alpha)
    return img


def dream_edge_blend(layer: Image.Image, background: Image.Image, x: int, y: int, strength: float = 0.22) -> Image.Image:
    """Subtle color/texture match near transparent edges so layers look less pasted."""
    layer = layer.convert("RGBA")
    bg = background.convert("RGB")
    crop = Image.new("RGB", layer.size, (127, 127, 127))
    crop.paste(bg.crop((max(0, x), max(0, y), min(bg.width, x + layer.width), min(bg.height, y + layer.height))), (max(0, -x), max(0, -y)))

    alpha = layer.getchannel("A")
    edge = alpha.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(radius=2.0))
    edge = ImageEnhance.Contrast(edge).enhance(1.8)

    layer_rgb = layer.convert("RGB")
    mixed = Image.blend(layer_rgb, crop.filter(ImageFilter.GaussianBlur(radius=1.2)), max(0.0, min(0.6, strength)))
    out = Image.composite(mixed, layer_rgb, edge).convert("RGBA")
    out.putalpha(alpha)
    return out


@dataclass
class DreamLayer:
    name: str
    source_image: Image.Image
    x: float = 384
    y: float = 384
    scale: float = 1.0
    rotation: float = 0.0
    visible: bool = True
    opacity: float = 1.0
    selected: bool = False
    meta: dict = field(default_factory=dict)
    rendered_cache_key: tuple = field(default_factory=tuple)
    rendered_cache: Optional[Image.Image] = None

    def render(self) -> Image.Image:
        key = (self.source_image.size, round(self.scale, 4), round(self.rotation, 2), round(self.opacity, 3), self.visible)
        if self.rendered_cache is not None and self.rendered_cache_key == key:
            return self.rendered_cache
        img = self.source_image.convert("RGBA")
        if self.opacity < 0.999:
            alpha = img.getchannel("A").point(lambda p: int(p * max(0.0, min(1.0, self.opacity))))
            img.putalpha(alpha)
        new_w = max(1, int(img.width * max(0.03, self.scale)))
        new_h = max(1, int(img.height * max(0.03, self.scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        if abs(self.rotation) > 0.01:
            img = img.rotate(self.rotation, expand=True, resample=Image.Resampling.BICUBIC)
        self.rendered_cache_key = key
        self.rendered_cache = img
        return img

    def invalidate(self):
        self.rendered_cache_key = tuple()
        self.rendered_cache = None


class APVDDreamCanvas:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("APVD Dream Canvas")
        self.root.geometry("1180x820")
        self.root.minsize(980, 650)
        self.device = get_device() if torch is not None else None

        self.canvas_w = tk.IntVar(value=768)
        self.canvas_h = tk.IntVar(value=768)
        self.prompt_var = tk.StringVar(value="chair")
        self.status_var = tk.StringVar(value="Ready. Type a prompt and add a dream object.")
        self.intensity_var = tk.DoubleVar(value=1.0)
        self.cutout_var = tk.BooleanVar(value=True)
        self.crop_var = tk.BooleanVar(value=True)
        self.edge_blend_var = tk.BooleanVar(value=True)
        self.background_color_var = tk.StringVar(value="#181625")
        self.zoom_var = tk.DoubleVar(value=0.85)
        self.expand_border_var = tk.IntVar(value=128)
        self.expand_variation_var = tk.DoubleVar(value=0.45)
        self.expand_cleanup_var = tk.DoubleVar(value=0.65)
        self.expand_feather_var = tk.IntVar(value=40)
        self.expand_void_color_var = tk.StringVar(value="#ffffff")
        self.expand_model_path_var = tk.StringVar(value="No expansion model loaded")
        self.loaded_expand_model_path: Optional[Path] = None

        self.layers: list[DreamLayer] = []
        self.selected_index: Optional[int] = None
        self._canvas_photo: Optional[ImageTk.PhotoImage] = None
        self._drag_start: Optional[tuple[int, int, float, float]] = None
        self._last_render: Optional[Image.Image] = None
        self._generation_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._bind_shortcuts()
        self.refresh_canvas()

    def _build_ui(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        center = ttk.Frame(outer)
        center.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))

        ttk.Label(left, text="APVD Dream Canvas", font=("Segoe UI", 16, "bold")).pack(anchor=tk.W, pady=(0, 8))
        ttk.Label(left, text="Prompt / object").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.prompt_var, width=28).pack(fill=tk.X, pady=(2, 6))
        ttk.Button(left, text="Add Dream Object", command=self.add_prompt_object).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Generate Variant", command=self.generate_variant).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Import Image Layer", command=self.import_image_layer).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Import Random Memory", command=self.import_random_memory).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Dream Expand / Outpaint", font=("Segoe UI", 10, "bold")).pack(anchor=tk.W)
        ttk.Button(left, text="Import Image For Expand", command=self.import_image_for_expand).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Load Expansion Model", command=self.load_expansion_model).pack(fill=tk.X, pady=2)
        ttk.Label(left, textvariable=self.expand_model_path_var, wraplength=210).pack(anchor=tk.W, pady=(0, 2))
        ttk.Button(left, text="Dream Expand Selected Image", command=self.dream_expand_selected).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Dream Expand Whole Canvas", command=self.dream_expand_whole_canvas).pack(fill=tk.X, pady=2)
        self._labeled_spin(left, "Border pixels", self.expand_border_var, 8, 1536)
        self._labeled_scale(left, "Expansion variation", self.expand_variation_var, 0.0, 2.0)
        self._labeled_scale(left, "Expansion cleanup", self.expand_cleanup_var, 0.0, 1.0)
        self._labeled_spin(left, "Edge feather", self.expand_feather_var, 0, 256)
        ttk.Button(left, text="Set Void Color", command=self.choose_expand_void_color).pack(fill=tk.X, pady=2)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Generation").pack(anchor=tk.W)
        self._labeled_scale(left, "Dream intensity", self.intensity_var, 0.1, 4.0)
        ttk.Checkbutton(left, text="Smart edge cutout", variable=self.cutout_var).pack(anchor=tk.W)
        ttk.Checkbutton(left, text="Auto crop after cutout", variable=self.crop_var).pack(anchor=tk.W)
        ttk.Checkbutton(left, text="Blend edge into scene", variable=self.edge_blend_var).pack(anchor=tk.W)

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Label(left, text="Canvas").pack(anchor=tk.W)
        row = ttk.Frame(left)
        row.pack(fill=tk.X, pady=2)
        ttk.Label(row, text="W").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=128, to=4096, textvariable=self.canvas_w, width=7, command=self.resize_canvas).pack(side=tk.LEFT, padx=(3, 8))
        ttk.Label(row, text="H").pack(side=tk.LEFT)
        ttk.Spinbox(row, from_=128, to=4096, textvariable=self.canvas_h, width=7, command=self.resize_canvas).pack(side=tk.LEFT, padx=(3, 0))
        ttk.Button(left, text="Resize Canvas", command=self.resize_canvas).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Set Background Color", command=self.choose_bg_color).pack(fill=tk.X, pady=2)
        self._labeled_scale(left, "Preview zoom", self.zoom_var, 0.25, 1.5, command=lambda _v=None: self.refresh_canvas())

        ttk.Separator(left).pack(fill=tk.X, pady=10)
        ttk.Button(left, text="Dreamify Whole Canvas", command=self.dreamify_whole_canvas).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Export PNG", command=self.export_png).pack(fill=tk.X, pady=2)
        ttk.Button(left, text="Clear Scene", command=self.clear_scene).pack(fill=tk.X, pady=2)

        self.canvas = tk.Canvas(center, bg="#0c0c10", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_press)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self.scale_selected(1.05))
        self.canvas.bind("<Button-5>", lambda e: self.scale_selected(0.95))

        ttk.Label(right, text="Layers", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        self.layer_list = tk.Listbox(right, width=32, height=15, exportselection=False)
        self.layer_list.pack(fill=tk.BOTH, expand=False, pady=(2, 6))
        self.layer_list.bind("<<ListboxSelect>>", self.on_layer_select)

        layer_buttons = ttk.Frame(right)
        layer_buttons.pack(fill=tk.X)
        ttk.Button(layer_buttons, text="Up", command=lambda: self.move_layer(-1)).grid(row=0, column=0, sticky="ew", padx=1, pady=1)
        ttk.Button(layer_buttons, text="Down", command=lambda: self.move_layer(1)).grid(row=0, column=1, sticky="ew", padx=1, pady=1)
        ttk.Button(layer_buttons, text="Duplicate", command=self.duplicate_layer).grid(row=1, column=0, sticky="ew", padx=1, pady=1)
        ttk.Button(layer_buttons, text="Delete", command=self.delete_selected).grid(row=1, column=1, sticky="ew", padx=1, pady=1)
        layer_buttons.columnconfigure(0, weight=1)
        layer_buttons.columnconfigure(1, weight=1)

        ttk.Separator(right).pack(fill=tk.X, pady=10)
        ttk.Label(right, text="Selected layer controls", font=("Segoe UI", 12, "bold")).pack(anchor=tk.W)
        ttk.Button(right, text="Smart Cutout Selected", command=self.smart_cutout_selected).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Soften Edges", command=self.soften_selected).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Auto Crop", command=self.crop_selected).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Blend Selected Edges", command=self.blend_selected_edges).pack(fill=tk.X, pady=2)
        ttk.Button(right, text="Reset Transform", command=self.reset_selected_transform).pack(fill=tk.X, pady=2)

        self.scale_var = tk.DoubleVar(value=1.0)
        self.rotation_var = tk.DoubleVar(value=0.0)
        self.opacity_var = tk.DoubleVar(value=1.0)
        self._labeled_scale(right, "Scale", self.scale_var, 0.05, 4.0, command=self.apply_selected_controls)
        self._labeled_scale(right, "Rotation", self.rotation_var, -180, 180, command=self.apply_selected_controls)
        self._labeled_scale(right, "Opacity", self.opacity_var, 0.05, 1.0, command=self.apply_selected_controls)

        ttk.Label(right, text="Tips:\nDrag layers with mouse.\nMouse wheel scales selected.\nQ/E rotates. Delete removes.\nCtrl+S exports.", justify=tk.LEFT).pack(anchor=tk.W, pady=(14, 0))

        status = ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W)
        status.pack(side=tk.BOTTOM, fill=tk.X)

    def _labeled_scale(self, parent, label, var, from_, to, command=None):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text=label).pack(anchor=tk.W)
        scale = ttk.Scale(frame, variable=var, from_=from_, to=to, orient=tk.HORIZONTAL, command=command)
        scale.pack(fill=tk.X)
        return scale

    def _labeled_spin(self, parent, label, var, from_, to):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=3)
        ttk.Label(frame, text=label).pack(side=tk.LEFT)
        spin = ttk.Spinbox(frame, from_=from_, to=to, textvariable=var, width=8)
        spin.pack(side=tk.RIGHT)
        return spin

    def _bind_shortcuts(self):
        self.root.bind("<Delete>", lambda _e: self.delete_selected())
        self.root.bind("<Control-s>", lambda _e: self.export_png())
        self.root.bind("<Control-S>", lambda _e: self.export_png())
        self.root.bind("q", lambda _e: self.rotate_selected(-5))
        self.root.bind("e", lambda _e: self.rotate_selected(5))
        self.root.bind("<plus>", lambda _e: self.scale_selected(1.05))
        self.root.bind("<minus>", lambda _e: self.scale_selected(0.95))

    def choose_bg_color(self):
        color = simpledialog.askstring("Background Color", "Enter hex color:", initialvalue=self.background_color_var.get(), parent=self.root)
        if color:
            self.background_color_var.set(color.strip())
            self.refresh_canvas()

    def choose_expand_void_color(self):
        color = simpledialog.askstring("Dream Expand Void Color", "Blank area color before APVD imagines it:", initialvalue=self.expand_void_color_var.get(), parent=self.root)
        if color:
            self.expand_void_color_var.set(color.strip())
            self.status_var.set(f"Dream Expand void color set to {color.strip()}")

    def load_expansion_model(self):
        path = filedialog.askopenfilename(
            title="Load APVD expansion model",
            initialdir=str(MODELS_DIR.resolve()) if MODELS_DIR.exists() else str(APP_BASE_DIR.resolve()),
            filetypes=[("APVD model", "*.pt *.pth"), ("All files", "*.*")],
        )
        if not path:
            return
        self.loaded_expand_model_path = Path(path)
        self.expand_model_path_var.set(f"Expansion model: {self.loaded_expand_model_path.name}")
        self.status_var.set(f"Loaded expansion model: {self.loaded_expand_model_path.name}")

    def _resolve_expansion_model_path(self) -> Path:
        if self.loaded_expand_model_path is not None and self.loaded_expand_model_path.exists():
            return self.loaded_expand_model_path
        prompt = self.prompt_var.get().strip() or "dream"
        model_path, reason = find_model_for_prompt(prompt, MODELS_DIR)
        self.loaded_expand_model_path = Path(model_path)
        self.expand_model_path_var.set(f"Expansion model: {model_path.name} ({reason})")
        return Path(model_path)

    def resize_canvas(self):
        try:
            self.canvas_w.set(max(128, int(self.canvas_w.get())))
            self.canvas_h.set(max(128, int(self.canvas_h.get())))
        except Exception:
            self.canvas_w.set(768)
            self.canvas_h.set(768)
        self.refresh_canvas()

    def clear_scene(self):
        if self.layers and not messagebox.askyesno("Clear Scene", "Delete all layers?"):
            return
        self.layers.clear()
        self.selected_index = None
        self.refresh_layer_list()
        self.refresh_canvas()

    def set_busy(self, text: str):
        self.status_var.set(text)
        self.root.update_idletasks()

    def add_prompt_object(self):
        prompt = self.prompt_var.get().strip()
        if not prompt:
            messagebox.showerror("Prompt", "Type something first, like chair, car, person, tree, etc.")
            return
        if self._generation_thread and self._generation_thread.is_alive():
            messagebox.showinfo("Generation", "A generation is already running.")
            return
        self.set_busy(f"Searching APVD models for {prompt!r}...")

        def job():
            try:
                model_path, reason = find_model_for_prompt(prompt, MODELS_DIR)
                image = generate_from_apvd_checkpoint(model_path, self.intensity_var.get(), self.device)
                self.root.after(0, lambda: self.add_image_as_layer(image, prompt, {"model": str(model_path), "reason": reason}))
                self.root.after(0, lambda: self.status_var.set(f"Added {prompt!r} from {model_path.name} ({reason})"))
            except Exception as exc:
                self.root.after(0, lambda msg=str(exc): messagebox.showerror("Add Dream Object", msg))
                self.root.after(0, lambda: self.status_var.set("Could not add dream object."))

        self._generation_thread = threading.Thread(target=job, daemon=True)
        self._generation_thread.start()

    def generate_variant(self):
        if self.selected_index is not None:
            layer = self.layers[self.selected_index]
            model_path = layer.meta.get("model")
            prompt = layer.name
            if model_path:
                self.prompt_var.set(prompt)
                self.set_busy(f"Generating variant for {prompt!r}...")

                def job():
                    try:
                        image = generate_from_apvd_checkpoint(Path(model_path), self.intensity_var.get(), self.device)
                        self.root.after(0, lambda: self.add_image_as_layer(image, prompt + " variant", {"model": model_path, "variant_of": layer.name}))
                        self.root.after(0, lambda: self.status_var.set(f"Variant added for {prompt!r}."))
                    except Exception as exc:
                        self.root.after(0, lambda msg=str(exc): messagebox.showerror("Generate Variant", msg))
                threading.Thread(target=job, daemon=True).start()
                return
        self.add_prompt_object()

    def prepare_imported_image(self, image: Image.Image) -> Image.Image:
        img = image.convert("RGBA")
        if self.cutout_var.get():
            img = create_alpha_from_background(img, threshold=38, feather=2)
        if self.crop_var.get():
            img = auto_crop_alpha(img)
        return img

    def add_image_as_layer(self, image: Image.Image, name: str, meta: Optional[dict] = None):
        img = self.prepare_imported_image(image)
        # Keep imported things manageable on the canvas.
        max_side = max(img.size)
        if max_side > 360:
            ratio = 360 / max_side
            img = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.Resampling.LANCZOS)
        layer = DreamLayer(
            name=name,
            source_image=img,
            x=self.canvas_w.get() / 2,
            y=self.canvas_h.get() / 2,
            scale=1.0,
            rotation=0.0,
            opacity=1.0,
            meta=meta or {},
        )
        self.layers.append(layer)
        self.selected_index = len(self.layers) - 1
        self.refresh_layer_list()
        self.sync_controls_from_selected()
        self.refresh_canvas()

    def import_image_layer(self):
        path = filedialog.askopenfilename(title="Import image", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.gif")])
        if not path:
            return
        try:
            with Image.open(path) as img:
                self.add_image_as_layer(img, Path(path).stem, {"file": path})
            self.status_var.set(f"Imported layer: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Import Image", str(exc))

    def import_image_for_expand(self):
        """Import a normal rectangular image without sticker cutout/crop, then center it for outpainting."""
        path = filedialog.askopenfilename(title="Import image to Dream Expand", filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp *.gif")])
        if not path:
            return
        try:
            with Image.open(path) as img:
                rgba = img.convert("RGBA")
            max_side = max(rgba.size)
            if max_side > 720:
                ratio = 720 / max_side
                rgba = rgba.resize((max(1, int(rgba.width * ratio)), max(1, int(rgba.height * ratio))), Image.Resampling.LANCZOS)
            self.canvas_w.set(max(int(self.canvas_w.get()), rgba.width + 64))
            self.canvas_h.set(max(int(self.canvas_h.get()), rgba.height + 64))
            layer = DreamLayer(
                name=Path(path).stem + " expand source",
                source_image=rgba,
                x=self.canvas_w.get() / 2,
                y=self.canvas_h.get() / 2,
                scale=1.0,
                rotation=0.0,
                opacity=1.0,
                meta={"file": path, "expand_source": True},
            )
            self.layers.append(layer)
            self.selected_index = len(self.layers) - 1
            self.refresh_layer_list()
            self.sync_controls_from_selected()
            self.refresh_canvas()
            self.status_var.set(f"Imported expansion source: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Import Image For Expand", str(exc))

    def import_random_memory(self):
        images = list_images(MEMORY_DIR)
        if not images:
            messagebox.showerror("Memory", f"No images found in {MEMORY_DIR.resolve()}")
            return
        path = random.choice(images)
        try:
            with Image.open(path) as img:
                self.add_image_as_layer(img, path.stem, {"memory": str(path)})
            self.status_var.set(f"Imported random memory: {path.name}")
        except Exception as exc:
            messagebox.showerror("Memory", str(exc))

    def refresh_layer_list(self):
        self.layer_list.delete(0, tk.END)
        # Display top layer first but store real index in reverse lookup.
        self._listbox_to_layer_index = []
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            label = f"{'👁' if layer.visible else ' '} {layer.name}"
            self.layer_list.insert(tk.END, label)
            self._listbox_to_layer_index.append(i)
        if self.selected_index is not None and self.selected_index in self._listbox_to_layer_index:
            lb_index = self._listbox_to_layer_index.index(self.selected_index)
            self.layer_list.selection_clear(0, tk.END)
            self.layer_list.selection_set(lb_index)
            self.layer_list.activate(lb_index)

    def on_layer_select(self, _event=None):
        selection = self.layer_list.curselection()
        if not selection:
            return
        self.selected_index = self._listbox_to_layer_index[int(selection[0])]
        self.sync_controls_from_selected()
        self.refresh_canvas()

    def sync_controls_from_selected(self):
        layer = self.selected_layer()
        if layer is None:
            return
        self.scale_var.set(layer.scale)
        self.rotation_var.set(layer.rotation)
        self.opacity_var.set(layer.opacity)

    def selected_layer(self) -> Optional[DreamLayer]:
        if self.selected_index is None:
            return None
        if 0 <= self.selected_index < len(self.layers):
            return self.layers[self.selected_index]
        return None

    def apply_selected_controls(self, _value=None):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.scale = float(self.scale_var.get())
        layer.rotation = float(self.rotation_var.get())
        layer.opacity = float(self.opacity_var.get())
        layer.invalidate()
        self.refresh_canvas()

    def move_layer(self, delta: int):
        if self.selected_index is None:
            return
        new_index = self.selected_index + delta
        if not (0 <= new_index < len(self.layers)):
            return
        self.layers[self.selected_index], self.layers[new_index] = self.layers[new_index], self.layers[self.selected_index]
        self.selected_index = new_index
        self.refresh_layer_list()
        self.refresh_canvas()

    def duplicate_layer(self):
        layer = self.selected_layer()
        if layer is None:
            return
        copy = DreamLayer(
            name=layer.name + " copy",
            source_image=layer.source_image.copy(),
            x=layer.x + 24,
            y=layer.y + 24,
            scale=layer.scale,
            rotation=layer.rotation,
            visible=layer.visible,
            opacity=layer.opacity,
            meta=dict(layer.meta),
        )
        self.layers.append(copy)
        self.selected_index = len(self.layers) - 1
        self.refresh_layer_list()
        self.sync_controls_from_selected()
        self.refresh_canvas()

    def delete_selected(self):
        if self.selected_index is None:
            return
        del self.layers[self.selected_index]
        if not self.layers:
            self.selected_index = None
        else:
            self.selected_index = min(self.selected_index, len(self.layers) - 1)
        self.refresh_layer_list()
        self.sync_controls_from_selected()
        self.refresh_canvas()

    def reset_selected_transform(self):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.scale = 1.0
        layer.rotation = 0.0
        layer.opacity = 1.0
        layer.invalidate()
        self.sync_controls_from_selected()
        self.refresh_canvas()

    def smart_cutout_selected(self):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.source_image = create_alpha_from_background(layer.source_image, threshold=38, feather=2)
        layer.invalidate()
        self.refresh_canvas()

    def soften_selected(self):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.source_image = soften_alpha(layer.source_image)
        layer.invalidate()
        self.refresh_canvas()

    def crop_selected(self):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.source_image = auto_crop_alpha(layer.source_image)
        layer.invalidate()
        self.refresh_canvas()

    def blend_selected_edges(self):
        layer = self.selected_layer()
        if layer is None:
            return
        base = self.render_scene(skip_index=self.selected_index)
        rendered = layer.render()
        x = int(layer.x - rendered.width / 2)
        y = int(layer.y - rendered.height / 2)
        blended = dream_edge_blend(rendered, base, x, y, strength=0.25)
        # Store blended result as the source at current transform, then reset transform to preserve appearance.
        layer.source_image = blended
        layer.scale = 1.0
        layer.rotation = 0.0
        layer.opacity = 1.0
        layer.invalidate()
        self.sync_controls_from_selected()
        self.refresh_canvas()

    def rotate_selected(self, amount: float):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.rotation += amount
        layer.rotation = ((layer.rotation + 180) % 360) - 180
        layer.invalidate()
        self.sync_controls_from_selected()
        self.refresh_canvas()

    def scale_selected(self, factor: float):
        layer = self.selected_layer()
        if layer is None:
            return
        layer.scale = max(0.03, min(8.0, layer.scale * factor))
        layer.invalidate()
        self.sync_controls_from_selected()
        self.refresh_canvas()

    def on_mousewheel(self, event):
        if event.delta > 0:
            self.scale_selected(1.05)
        elif event.delta < 0:
            self.scale_selected(0.95)

    def event_to_canvas_coords(self, event) -> tuple[float, float]:
        zoom = max(0.05, float(self.zoom_var.get()))
        return event.x / zoom, event.y / zoom

    def hit_test(self, cx: float, cy: float) -> Optional[int]:
        # Topmost first.
        for i in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[i]
            img = layer.render()
            x0 = layer.x - img.width / 2
            y0 = layer.y - img.height / 2
            if x0 <= cx <= x0 + img.width and y0 <= cy <= y0 + img.height:
                local_x = int(cx - x0)
                local_y = int(cy - y0)
                try:
                    if img.getchannel("A").getpixel((local_x, local_y)) > 8:
                        return i
                except Exception:
                    return i
        return None

    def on_canvas_press(self, event):
        cx, cy = self.event_to_canvas_coords(event)
        hit = self.hit_test(cx, cy)
        if hit is not None:
            self.selected_index = hit
            layer = self.layers[hit]
            self._drag_start = (event.x, event.y, layer.x, layer.y)
            self.refresh_layer_list()
            self.sync_controls_from_selected()
            self.refresh_canvas()
        else:
            self.selected_index = None
            self._drag_start = None
            self.refresh_layer_list()
            self.refresh_canvas()

    def on_canvas_drag(self, event):
        if self._drag_start is None or self.selected_index is None:
            return
        sx, sy, lx, ly = self._drag_start
        zoom = max(0.05, float(self.zoom_var.get()))
        dx = (event.x - sx) / zoom
        dy = (event.y - sy) / zoom
        layer = self.selected_layer()
        if layer is None:
            return
        layer.x = lx + dx
        layer.y = ly + dy
        self.refresh_canvas()

    def on_canvas_release(self, _event):
        self._drag_start = None

    def render_scene(self, skip_index: Optional[int] = None) -> Image.Image:
        bg_color = self.background_color_var.get().strip() or "#181625"
        try:
            scene = Image.new("RGBA", (int(self.canvas_w.get()), int(self.canvas_h.get())), bg_color)
        except Exception:
            scene = Image.new("RGBA", (int(self.canvas_w.get()), int(self.canvas_h.get())), "#181625")
        for i, layer in enumerate(self.layers):
            if i == skip_index or not layer.visible:
                continue
            img = layer.render()
            if self.edge_blend_var.get() and layer.meta.get("auto_edge_blend", False):
                img = dream_edge_blend(img, scene, int(layer.x - img.width / 2), int(layer.y - img.height / 2), strength=0.16)
            x = int(layer.x - img.width / 2)
            y = int(layer.y - img.height / 2)
            scene.alpha_composite(img, dest=(x, y))
        return scene

    def refresh_canvas(self):
        scene = self.render_scene()
        self._last_render = scene.copy()
        zoom = max(0.05, float(self.zoom_var.get()))
        preview = scene.resize((max(1, int(scene.width * zoom)), max(1, int(scene.height * zoom))), Image.Resampling.BILINEAR)

        # Selection box.
        if self.selected_index is not None and 0 <= self.selected_index < len(self.layers):
            draw_img = preview.copy()
            try:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(draw_img)
                layer = self.layers[self.selected_index]
                rendered = layer.render()
                x0 = int((layer.x - rendered.width / 2) * zoom)
                y0 = int((layer.y - rendered.height / 2) * zoom)
                x1 = int((layer.x + rendered.width / 2) * zoom)
                y1 = int((layer.y + rendered.height / 2) * zoom)
                draw.rectangle((x0, y0, x1, y1), outline=(255, 255, 255, 210), width=2)
                preview = draw_img
            except Exception:
                pass

        self._canvas_photo = ImageTk.PhotoImage(preview, master=self.root)
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, preview.width, preview.height))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._canvas_photo)

    def _dream_expand_image_async(self, source: Image.Image, name: str):
        if self._generation_thread and self._generation_thread.is_alive():
            messagebox.showinfo("Dream Expand", "A generation/expansion is already running.")
            return
        try:
            border = max(1, int(self.expand_border_var.get()))
            feather = max(0, int(self.expand_feather_var.get()))
            variation = float(self.expand_variation_var.get())
            cleanup = float(self.expand_cleanup_var.get())
            void_color = self.expand_void_color_var.get().strip() or "#ffffff"
        except Exception:
            messagebox.showerror("Dream Expand", "Check the border, variation, cleanup, and feather settings.")
            return
        self.set_busy("Dream Expand is imagining beyond the image borders...")

        def job():
            try:
                model_path = self._resolve_expansion_model_path()
                expanded = build_expanded_memory_image(
                    source,
                    model_path,
                    self.device,
                    border_px=border,
                    variation=variation,
                    cleanup=cleanup,
                    feather_px=feather,
                    void_color=void_color,
                )
                def finish():
                    self.canvas_w.set(expanded.width)
                    self.canvas_h.set(expanded.height)
                    self.layers = [DreamLayer(name=name, source_image=expanded, x=expanded.width / 2, y=expanded.height / 2, meta={"expanded_with": str(model_path)})]
                    self.selected_index = 0
                    self.refresh_layer_list()
                    self.sync_controls_from_selected()
                    self.refresh_canvas()
                    self.status_var.set(f"Dream Expanded with {model_path.name}. Original pixels preserved; outside is APVD memory fill.")
                self.root.after(0, finish)
            except Exception as exc:
                self.root.after(0, lambda msg=str(exc): messagebox.showerror("Dream Expand", msg))
                self.root.after(0, lambda: self.status_var.set("Dream Expand failed."))

        self._generation_thread = threading.Thread(target=job, daemon=True)
        self._generation_thread.start()

    def dream_expand_selected(self):
        layer = self.selected_layer()
        if layer is None:
            messagebox.showinfo("Dream Expand", "Select or import an image layer first.")
            return
        if abs(layer.rotation) > 0.01 or abs(layer.scale - 1.0) > 0.001 or layer.opacity < 0.999:
            source = layer.render()
        else:
            source = layer.source_image.copy()
        self._dream_expand_image_async(source.convert("RGBA"), layer.name + " - Dream Expanded")

    def dream_expand_whole_canvas(self):
        if not self.layers:
            messagebox.showinfo("Dream Expand", "Add or import something first.")
            return
        source = self.render_scene().convert("RGBA")
        self._dream_expand_image_async(source, "Whole Canvas - Dream Expanded")

    def dreamify_whole_canvas(self):
        """Simple unifier pass: soft blur/noise/color squeeze. This is not model-based yet."""
        if not self.layers:
            messagebox.showinfo("Dreamify", "Add some layers first.")
            return
        scene = self.render_scene()
        rgb = scene.convert("RGB")
        soft = rgb.filter(ImageFilter.GaussianBlur(radius=0.6))
        sharp = Image.blend(rgb, soft, 0.28)
        sharp = ImageEnhance.Color(sharp).enhance(0.92)
        sharp = ImageEnhance.Contrast(sharp).enhance(1.08)
        unified = sharp.convert("RGBA")
        self.layers = [DreamLayer(name="Dreamified whole scene", source_image=unified, x=self.canvas_w.get()/2, y=self.canvas_h.get()/2)]
        self.selected_index = 0
        self.refresh_layer_list()
        self.sync_controls_from_selected()
        self.refresh_canvas()
        self.status_var.set("Scene flattened and dreamified. Basically: cursed glue applied.")

    def export_png(self):
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        default = OUTPUTS_DIR / f"apvd_dream_canvas_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = filedialog.asksaveasfilename(
            title="Export Dream Canvas PNG",
            defaultextension=".png",
            initialfile=default.name,
            initialdir=str(OUTPUTS_DIR.resolve()),
            filetypes=[("PNG", "*.png")],
        )
        if not path:
            return
        try:
            image = self.render_scene()
            image.save(path)
            self.status_var.set(f"Exported: {Path(path).name}")
            messagebox.showinfo("Export", f"Saved Dream Canvas image to:\n{Path(path).resolve()}")
        except Exception as exc:
            messagebox.showerror("Export", str(exc))

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    APVDDreamCanvas().run()
