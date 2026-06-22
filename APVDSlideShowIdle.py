"""
APVDSlideShowIdle
A lightweight fullscreen idle slideshow for APVD models and memories.

Put this file next to your APVD project files, especially:
- model.py
- utils.py
- memory_system.py

Expected folders beside this script:
- Models/  -> .pt/.pth APVD checkpoints
- Memory/  -> APVD memory images/latents, plus regular images if you add them

Controls:
- Esc or Q: quit
- Space: pause/resume
- R: rescan folders
- M: switch mode
- N: next model
- F11: toggle fullscreen
"""

import os
import random
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    import torch
    from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageDraw, ImageOps
except Exception as exc:
    print("Missing dependency:", exc)
    input("Press Enter to exit...")
    raise SystemExit(1)

try:
    from model import VAE, get_device
    from utils import tensor_to_pil
except Exception as exc:
    print("Could not import APVD project files. Put this script beside model.py and utils.py.")
    print(exc)
    input("Press Enter to exit...")
    raise SystemExit(2)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}
MODEL_EXTS = {".pt", ".pth"}


def safe_torch_load(path, *, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def normalize_apvd_image(image: Image.Image, background: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Return RGB while safely preserving transparency from palette PNG/GIF images."""
    image = ImageOps.exif_transpose(image)
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    elif image.mode in {"RGBA", "LA"}:
        image = image.convert("RGBA")
    else:
        image = image.convert("RGB")
    if image.mode == "RGBA":
        bg = Image.new("RGBA", image.size, (*background, 255))
        bg.alpha_composite(image)
        image = bg.convert("RGB")
    return image


def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True).clamp_min(1e-8)
    high_norm = high / torch.norm(high, dim=1, keepdim=True).clamp_min(1e-8)
    dot = torch.clamp((low_norm * high_norm).sum(1), -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if torch.all(so < 1e-6):
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high


class APVDSlideShowIdle:
    def __init__(self):
        self.base_dir = Path(__file__).resolve().parent
        self.models_dir = self.base_dir / "Models"
        self.memory_dir = self.base_dir / "Memory"
        self.device = get_device()
        self.model = None
        self.model_path = None
        self.model_paths = []
        self.memory_images = []
        self.memory_latents = []
        self.current_latent = None
        self.target_latent = None
        self.step = 0
        self.steps_per_dream = 30
        self.paused = False
        self.fullscreen = True
        self.mode = "Dream Morph"  # Dream Morph, Random Pop, Memory Images
        self.worker_busy = False
        self.pending_image = None
        self.last_status = "Starting..."

        self.root = tk.Tk()
        self.root.title("APVDSlideShowIdle")
        self.root.geometry("1280x720")
        self.root.configure(bg="black")
        self.root.attributes("-fullscreen", True)

        self.canvas = tk.Canvas(self.root, width=1280, height=720, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.photo = None

        self.root.bind("<Escape>", lambda e: self.root.destroy())
        self.root.bind("q", lambda e: self.root.destroy())
        self.root.bind("Q", lambda e: self.root.destroy())
        self.root.bind("<space>", self.toggle_pause)
        self.root.bind("r", lambda e: self.rescan())
        self.root.bind("R", lambda e: self.rescan())
        self.root.bind("m", lambda e: self.next_mode())
        self.root.bind("M", lambda e: self.next_mode())
        self.root.bind("n", lambda e: self.load_random_model())
        self.root.bind("N", lambda e: self.load_random_model())
        self.root.bind("<F11>", self.toggle_fullscreen)

        self.rescan()
        if self.model_paths:
            self.load_random_model()
        self.loop()

    def rescan(self):
        self.models_dir.mkdir(exist_ok=True)
        self.memory_dir.mkdir(exist_ok=True)
        self.model_paths = sorted([p for p in self.models_dir.rglob("*") if p.suffix.lower() in MODEL_EXTS])
        self.memory_images = sorted([p for p in self.memory_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        self.memory_latents = sorted([p for p in self.memory_dir.rglob("*") if p.suffix.lower() in MODEL_EXTS])
        self.last_status = f"Found {len(self.model_paths)} model(s), {len(self.memory_images)} memory image(s), {len(self.memory_latents)} latent file(s)."
        self.draw_status_only()

    def load_random_model(self):
        if not self.model_paths:
            self.model = None
            self.last_status = "No models found. Put .pt/.pth files in the Models folder."
            return
        path = random.choice(self.model_paths)
        try:
            checkpoint = safe_torch_load(path, map_location=self.device)
            output_size = tuple(checkpoint.get("output_size", (256, 256)))
            latent_dim = int(checkpoint.get("latent_dim", 256))
            model = VAE(latent_dim=latent_dim, output_size=output_size).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.eval()
            self.model = model
            self.model_path = path
            self.current_latent = None
            self.target_latent = None
            self.step = 0
            self.last_status = f"Loaded: {path.name}"
        except Exception as exc:
            self.last_status = f"Failed to load {path.name}: {exc}"

    def random_latent(self):
        if self.memory_latents and random.random() < 0.35:
            path = random.choice(self.memory_latents)
            try:
                data = safe_torch_load(path, map_location=self.device)
                if isinstance(data, torch.Tensor):
                    z = data.float().to(self.device)
                    if z.ndim == 1:
                        z = z.unsqueeze(0)
                    if z.shape[1] == self.model.latent_dim:
                        return z
            except Exception:
                pass
        intensity = random.uniform(0.08, 1.25)
        return torch.randn(1, self.model.latent_dim, device=self.device) * intensity

    def decode_latent(self, z):
        with torch.no_grad():
            recon = self.model.decode(z)
            return tensor_to_pil(recon).convert("RGB")

    def make_frame(self):
        if self.mode == "Memory Images" and self.memory_images:
            img = normalize_apvd_image(Image.open(random.choice(self.memory_images)))
            return self.apply_idle_fx(img)

        if self.model is None:
            if self.memory_images:
                img = normalize_apvd_image(Image.open(random.choice(self.memory_images)))
                return self.apply_idle_fx(img)
            return None

        if self.mode == "Random Pop":
            z = self.random_latent()
            return self.apply_idle_fx(self.decode_latent(z))

        if self.current_latent is None or self.target_latent is None or self.step >= self.steps_per_dream:
            self.current_latent = self.target_latent if self.target_latent is not None else self.random_latent()
            self.target_latent = self.random_latent()
            self.step = 0
            if random.random() < 0.12:
                self.load_random_model()

        alpha = self.step / max(1, self.steps_per_dream)
        alpha = 0.5 - 0.5 * torch.cos(torch.tensor(alpha * 3.14159265)).item()
        z = slerp(alpha, self.current_latent, self.target_latent)
        self.step += 1
        return self.apply_idle_fx(self.decode_latent(z))

    def apply_idle_fx(self, img):
        img = normalize_apvd_image(img)
        w, h = self.canvas.winfo_width() or 1280, self.canvas.winfo_height() or 720
        scale = max(w / img.width, h / img.height) * 1.04
        nw, nh = int(img.width * scale), int(img.height * scale)
        img = img.resize((nw, nh), Image.Resampling.LANCZOS)
        x = (nw - w) // 2
        y = (nh - h) // 2
        img = img.crop((x, y, x + w, y + h))
        img = ImageEnhance.Contrast(img).enhance(1.08)
        img = ImageEnhance.Color(img).enhance(1.05)
        if random.random() < 0.08:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.6))
        overlay = Image.new("RGB", img.size, (0, 0, 0))
        img = Image.blend(img, overlay, 0.08)
        draw = ImageDraw.Draw(img)
        title = self.model_path.name if self.model_path else "No model loaded"
        text = f"APVDSlideShowIdle  |  {self.mode}  |  {title}"
        draw.rectangle((12, h - 42, min(w - 12, 12 + len(text) * 8), h - 14), fill=(0, 0, 0))
        draw.text((22, h - 36), text, fill=(220, 230, 255))
        return img

    def worker(self):
        try:
            self.pending_image = self.make_frame()
        except Exception as exc:
            self.last_status = f"Generation error: {exc}"
            self.pending_image = None
        finally:
            self.worker_busy = False

    def loop(self):
        if not self.paused and not self.worker_busy:
            if self.pending_image is not None:
                self.show_image(self.pending_image)
                self.pending_image = None
            self.worker_busy = True
            threading.Thread(target=self.worker, daemon=True).start()
        self.root.after(120 if self.mode == "Dream Morph" else 1800, self.loop)

    def show_image(self, img):
        self.photo = ImageTk.PhotoImage(img, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def draw_status_only(self):
        w, h = 1280, 720
        img = Image.new("RGB", (w, h), (5, 7, 14))
        draw = ImageDraw.Draw(img)
        draw.text((40, 40), "APVDSlideShowIdle", fill=(235, 240, 255))
        draw.text((40, 72), self.last_status, fill=(170, 180, 210))
        draw.text((40, 104), f"Models folder: {self.models_dir}", fill=(120, 130, 165))
        draw.text((40, 132), f"Memory folder: {self.memory_dir}", fill=(120, 130, 165))
        draw.text((40, 180), "Esc/Q quit | Space pause | R rescan | M mode | N next model | F11 fullscreen", fill=(170, 180, 210))
        self.show_image(img)

    def toggle_pause(self, _event=None):
        self.paused = not self.paused
        self.last_status = "Paused" if self.paused else "Running"

    def toggle_fullscreen(self, _event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def next_mode(self):
        modes = ["Dream Morph", "Random Pop", "Memory Images"]
        self.mode = modes[(modes.index(self.mode) + 1) % len(modes)]
        self.current_latent = None
        self.target_latent = None
        self.step = 0
        self.last_status = f"Mode: {self.mode}"

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    APVDSlideShowIdle().run()
