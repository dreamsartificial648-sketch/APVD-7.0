"""
APVD v7.0 - AI Pixel Value Determinator
Tkinter GUI for VAE-based image variation generation with mini latent diffusion.
Features:
- Training controls (Epochs, Save/Load)
- Generation Tools (Unique, Chaos Mode, Auto-Cycle)
- Dream Cycle: Smoothly morphs between latent points using Slerp for constant velocity.
- Mini Diffusion: Iterative latent denoising for more structured generations.
- Memory evolution, latent presets, interactive breeding, and latent map recall.
- Threaded Training with Stop Functionality
"""
from __future__ import annotations

import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import zipfile
from collections import OrderedDict
from datetime import datetime
from io import BytesIO
from pathlib import Path

try:
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F
    import tkinter as tk
    from PIL import Image, ImageDraw, ImageFile
    from tkinter import filedialog, messagebox, simpledialog, ttk
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
except Exception as e:
    print("A required package is missing:", e)
    input("Press Enter to exit...")
    raise SystemExit(5) from e

from memory_system import MemoryBank, breed_latents, parse_selection_indices, summarize_memory
from model import VAE, vae_loss, latent_denoiser_loss, get_device
from model_merger import merge_checkpoints
from reconstruction_video import render_reconstruction_video
from scene_composer import LayeredScene, ParsedScene, generate_scene_from_prompt
from latent_diffusion import (
    DiffusionConfig,
    DiffusionModel,
    apvd_diffusion_polish,
    apvd_reconstruction,
    build_default_diffusion_for_apvd,
    pure_diffusion_generation,
    train_diffusion_model,
)
from utils import (
    get_image_paths,
    list_image_members,
    list_model_paths,
    select_model_path_for_prompt,
    load_training_images_from_archive_entries,
    load_training_images_from_paths,
    load_training_images_from_videos,
    tensor_to_pil,
    rgb_to_wavelet,
    wavelet_to_rgb
)

APP_BASE_DIR = Path(__file__).resolve().parent
MEMORY_DIR = APP_BASE_DIR / "Memory"
MODELS_DIR = APP_BASE_DIR / "Models"
OUTPUTS_DIR = APP_BASE_DIR / "Outputs"
DREAMIFY_OUTPUT_DIR = APP_BASE_DIR / "Dreamify_Output"
DREAM_VIDEOS_DIR = APP_BASE_DIR / "Dream_Videos"
TIMELAPSES_DIR = APP_BASE_DIR / "Timelapses"
APP_SETTINGS_PATH = APP_BASE_DIR / "apvd_user_settings.json"

def _read_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default

# Number of decoded/resized training tensors to keep warm in RAM.
# This is treated as an approximate TOTAL cache budget and is split across
# DataLoader workers so 8 workers does not accidentally become 8x the RAM use.
DEFAULT_DATASET_CACHE_ITEMS = max(0, _read_int_env("APVD_DATASET_CACHE_ITEMS", 4096))

logging.basicConfig(
    level=os.environ.get("APVD_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("apvd")

try:
    from reconstruction_judge import ReconstructionJudge
    from memory_finder import MemoryFinder
except Exception:
    logger.debug("Optional APVD judge/finder modules were not loaded.", exc_info=True)
    ReconstructionJudge = None
    MemoryFinder = None

MAX_TRAINING_PREVIEW_IMAGES = 256
TRAINING_VISUAL_PREVIEW_INTERVAL = 10
TIMELAPSE_MAX_FRAMES = 900
TIMELAPSE_FPS = 12
TRAINING_INTENSITY_CHOICES = [
    "10% Background",
    "25% Light",
    "50% Balanced",
    "75% Fast",
    "100% Maximum",
]
PRECISION_MODE_CHOICES = [
    "Auto Recommended",
    "FP32 Stable",
    "FP16 Fast",
    "BF16 Balanced",
]
LOSS_MODE_CHOICES = [
    "Classic VAE",
    "Structure Stable",
    "Sharp Detail",
    "Experimental Structure",
]
LOSS_MODE_HELP = {
    "Classic VAE": "Current APVD loss. Best for normal dream/generalized behavior.",
    "Structure Stable": "Adds gentle L1 shape pressure so objects stay more like themselves.",
    "Sharp Detail": "Adds L1 plus edge loss for stronger outlines and details.",
    "Experimental Structure": "Heavier structure mix using L1 and edge loss. Useful for cars/characters, but more experimental.",
}

USER_LEVEL_CHOICES = ["Newbie", "Amateur", "Expert", "Nerd"]
USER_LEVEL_HELP = {
    "Newbie": "Basic training, model loading, preview, and the safest visible controls.",
    "Amateur": "Adds generation, Dreamify, prompts, and the common tuning controls.",
    "Expert": "Adds DDPM diffusion, evolution, memory tools, and deeper model controls.",
    "Nerd": "Shows everything. No seatbelt. Full cockpit mode.",
}
USER_LEVEL_SECTION_VISIBILITY = {
    "Newbie": {"Hardware", "Sources And Model", "Training Settings", "Preview"},
    "Amateur": {"Hardware", "Sources And Model", "Training Settings", "Generation Tools", "Generation Settings", "Prompt And Personality", "Preview"},
    "Expert": {"Hardware", "Sources And Model", "Training Settings", "Generation Tools", "Generation Settings", "Latent DDPM Diffusion", "Prompt And Personality", "Evolution And Memory", "Preview"},
    "Nerd": "ALL",
}

AUDIO_MEMORY_MODE_CHOICES = [
    "Silent",
    "Original Audio",
    "Audio Memory Reconstruction",
    "Dreamy Memory Audio",
    "Corrupted Memory Audio",
]
AUDIO_MEMORY_DREAM_MODES = {
    "Audio Memory Reconstruction",
    "Dreamy Memory Audio",
    "Corrupted Memory Audio",
}
AUDIO_MEMORY_PROFILE_DIR = APP_BASE_DIR / "Audio_Memory"
ImageFile.LOAD_TRUNCATED_IMAGES = True
PERSONALITY_PRESETS = {
    "Manual": {
        "intensity": 1.0,
        "blend": False,
        "blend_count": 2,
        "iterations": 3,
        "use_diffusion": True,
        "diffusion_steps": 8,
        "diffusion_strength": 0.85,
    },
    "Dreamy": {
        "intensity": 0.8,
        "blend": True,
        "blend_count": 4,
        "iterations": 2,
        "use_diffusion": True,
        "diffusion_steps": 10,
        "diffusion_strength": 0.55,
    },
    "Chaotic": {
        "intensity": 8.5,
        "blend": False,
        "blend_count": 2,
        "iterations": 0,
        "use_diffusion": True,
        "diffusion_steps": 14,
        "diffusion_strength": 1.25,
    },
    "Nostalgic": {
        "intensity": 0.9,
        "blend": True,
        "blend_count": 3,
        "iterations": 4,
        "use_diffusion": True,
        "diffusion_steps": 7,
        "diffusion_strength": 0.7,
    },
    "Hybrid": {
        "intensity": 2.2,
        "blend": True,
        "blend_count": 6,
        "iterations": 3,
        "use_diffusion": True,
        "diffusion_steps": 12,
        "diffusion_strength": 0.95,
    },
    "Corruption": {
        "intensity": 6.0,
        "blend": False,
        "blend_count": 2,
        "iterations": 10,
        "use_diffusion": True,
        "diffusion_steps": 18,
        "diffusion_strength": 1.35,
    },
}

def safe_torch_load(path, *, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def slerp(val, low, high):
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm * high_norm).sum(1)
    dot = torch.clamp(dot, -1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if torch.all(so < 1e-6):
        return (1.0 - val) * low + val * high
    res = (torch.sin((1.0 - val) * omega) / so).unsqueeze(1) * low + (torch.sin(val * omega) / so).unsqueeze(1) * high
    return res

class APVDDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        target_resolution: int,
        archive_entries: list[tuple[Path, str]] | None = None,
        video_paths: list[Path] | None = None,
        video_stride: int = 30,
        video_max_frames: int = 0,
        wavelet_mode: bool = False,
        cache_limit: int | None = None,
    ):
        self.image_paths = [Path(path) for path in image_paths]
        self.target_size = (int(target_resolution), int(target_resolution))
        self.wavelet_mode = wavelet_mode
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    self.target_size,
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.ToTensor(),
            ]
        )
        self.fallback = torch.zeros(
            12 if wavelet_mode else 3,
            self.target_size[1] // 2 if wavelet_mode else self.target_size[1],
            self.target_size[0] // 2 if wavelet_mode else self.target_size[0],
            dtype=torch.float32,
        )

        self.video_stride = max(1, int(video_stride))
        self.video_max_frames = None if int(video_max_frames) <= 0 else int(video_max_frames)
        self.samples: list[tuple[str, object]] = [("path", path) for path in self.image_paths]
        if archive_entries:
            self.samples.extend(("archive", (Path(ap), member)) for ap, member in archive_entries)
        if video_paths:
            self.samples.extend(self._build_video_samples(video_paths))

        self.cache_limit = max(0, int(DEFAULT_DATASET_CACHE_ITEMS if cache_limit is None else cache_limit))
        self._cache: OrderedDict[tuple[str, str], torch.Tensor] = OrderedDict()

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _is_safe_archive_member(member_name: str) -> bool:
        norm = member_name.replace("\\", "/").strip("/")
        if not norm:
            return False
        parts = norm.split("/")
        if ".." in parts:
            return False
        if parts[0].endswith(":"):
            return False
        return True

    @staticmethod
    def _skip_macosx_path(member_name: str) -> bool:
        parts = member_name.replace("\\", "/").split("/")
        return any(part == "__MACOSX" for part in parts)

    @staticmethod
    def _member_is_image_file(member_name: str) -> bool:
        return Path(member_name).suffix.lower() in {
            ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp",
        }

    def _build_video_samples(self, video_paths: list[Path]) -> list[tuple[str, object]]:
        samples: list[tuple[str, object]] = []
        remaining = self.video_max_frames
        for video_path in video_paths:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                continue
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if frame_count <= 0:
                continue
            for frame_index in range(0, frame_count, self.video_stride):
                samples.append(("video", (Path(video_path), frame_index)))
                if remaining is not None:
                    remaining -= 1
                    if remaining <= 0:
                        return samples
        return samples

    def _cache_key(self, kind: str, payload: object) -> tuple[str, str]:
        return kind, str(payload)

    def _get_cached(self, kind: str, payload: object) -> torch.Tensor | None:
        if self.cache_limit <= 0:
            return None
        key = self._cache_key(kind, payload)
        tensor = self._cache.get(key)
        if tensor is not None:
            self._cache.move_to_end(key)
        return tensor

    def _put_cached(self, kind: str, payload: object, tensor: torch.Tensor) -> torch.Tensor:
        if self.cache_limit <= 0:
            return tensor
        key = self._cache_key(kind, payload)
        self._cache[key] = tensor.detach().cpu()
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_limit:
            self._cache.popitem(last=False)
        return tensor

    def _pil_to_training_tensor(self, image: Image.Image) -> torch.Tensor:
        try:
            image.draft("RGB", self.target_size)
        except Exception:
            pass
        tensor = self.transform(image.convert("RGB"))
        if self.wavelet_mode:
            tensor = rgb_to_wavelet(tensor)
        return tensor

    def _load_archive_member(self, archive_path: Path, member_name: str) -> torch.Tensor:
        if not self._is_safe_archive_member(member_name) or self._skip_macosx_path(member_name):
            return self.fallback.clone()

        try:
            if archive_path.suffix.lower() == ".zip" or archive_path.name.lower().endswith((".zip",)):
                with zipfile.ZipFile(archive_path, "r") as zf:
                    with zf.open(member_name, "r") as f:
                        data = f.read()
            else:
                with tarfile.open(archive_path, "r:*") as tf:
                    info = tf.getmember(member_name)
                    if not info.isfile():
                        return self.fallback.clone()
                    reader = tf.extractfile(info)
                    if reader is None:
                        return self.fallback.clone()
                    data = reader.read()
            with Image.open(BytesIO(data)) as img:
                return self._pil_to_training_tensor(img)
        except Exception:
            logger.debug("Skipping unreadable archive image %s in %s", member_name, archive_path, exc_info=True)
            return self.fallback.clone()

    def _load_video_frame(self, video_path: Path, frame_index: int) -> torch.Tensor:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return self.fallback.clone()
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok or frame is None:
                return self.fallback.clone()
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(np.ascontiguousarray(rgb))
            pil = pil.convert("RGB")
            tensor = self.transform(pil)
            if self.wavelet_mode:
                tensor = rgb_to_wavelet(tensor)
            return tensor
        except Exception:
            logger.debug("Skipping unreadable video frame %s:%s", video_path, frame_index, exc_info=True)
            return self.fallback.clone()
        finally:
            cap.release()

    def __getitem__(self, index: int) -> torch.Tensor:
        kind, payload = self.samples[index]
        cached = self._get_cached(kind, payload)
        if cached is not None:
            return cached
        try:
            if kind == "path":
                path = Path(payload)
                with Image.open(path) as img:
                    return self._put_cached(kind, payload, self._pil_to_training_tensor(img))
            if kind == "archive":
                archive_path, member_name = payload
                return self._put_cached(kind, payload, self._load_archive_member(Path(archive_path), str(member_name)))
            if kind == "video":
                video_path, frame_index = payload
                return self._put_cached(kind, payload, self._load_video_frame(Path(video_path), int(frame_index)))
            with Image.open(Path(payload)) as img:
                return self._put_cached(kind, payload, self._pil_to_training_tensor(img))
        except Exception:
            logger.debug("Skipping unreadable training sample %s", payload, exc_info=True)
            return self.fallback.clone()

class APVDApp:
    def __init__(self):
        self.root = tk.Tk()
        self._closing = False
        self._after_ids: set[str] = set()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.title("APVD v7.0 - AI Pixel Value Determinator")
        self.root.geometry("800x1050")
        self.root.minsize(750, 1000)

        self.device = get_device()
        self.model_lock = threading.RLock()
        self.model: VAE | None = None
        self.latent_diffusion: DiffusionModel | None = None
        self.loaded_latent_diffusion_path: Path | None = None
        self.loaded_model_path: Path | None = None
        self.training_folder: Path | None = None
        self.training_paths: list[Path] | None = None
        self.batch_training_root: Path | None = None
        self.batch_training_folders: list[Path] = []
        self.archive_entries: list[tuple[Path, str]] = []
        self.video_paths: list[Path] = []
        self.training_tensors: torch.Tensor | None = None
        
        # State Variables
        self.epochs_var = tk.IntVar(value=100)
        self.resolution_var = tk.IntVar(value=256)
        self.batch_size_var = tk.IntVar(value=16)
        self.loader_workers_var = tk.IntVar(value=4)
        self.prefetch_batches_var = tk.IntVar(value=2)
        self.dataset_cache_items_var = tk.IntVar(value=DEFAULT_DATASET_CACHE_ITEMS)
        self.training_intensity_var = tk.StringVar(value="50% Balanced")
        self.learning_rate_var = tk.DoubleVar(value=2e-4)
        # Precision Mode replaces the old mixed-precision checkbox, but
        # mixed_precision_var is kept as a compatibility flag for older code/metadata.
        self.precision_mode_var = tk.StringVar(value="Auto Recommended")
        self.loss_mode_var = tk.StringVar(value="Classic VAE")
        self.loss_mode_help_var = tk.StringVar(value=LOSS_MODE_HELP["Classic VAE"])
        self.mixed_precision_var = tk.BooleanVar(value=True)
        self.nan_guard_var = tk.BooleanVar(value=True)
        self.video_stride_var = tk.IntVar(value=30)
        self.video_max_frames_var = tk.IntVar(value=0)
        self.iterations_var = tk.IntVar(value=3)
        self.show_iterations_var = tk.BooleanVar(value=True)
        self.auto_cycle_var = tk.BooleanVar(value=False)
        self.dream_cycle_var = tk.BooleanVar(value=False)
        self.blend_mode_var = tk.BooleanVar(value=False)
        self.blend_count_var = tk.IntVar(value=2)
        self.output_count_var = tk.IntVar(value=1)
        self.use_mini_diffusion_var = tk.BooleanVar(value=True)
        self.diffusion_steps_var = tk.IntVar(value=8)
        self.diffusion_strength_var = tk.DoubleVar(value=0.85)
        self.dream_strength_var = tk.DoubleVar(value=0.35)
        self.memory_pull_var = tk.DoubleVar(value=0.25)
        self.dreamify_frame_skip_var = tk.IntVar(value=1)
        self.keep_original_audio_var = tk.BooleanVar(value=False)
        self.dreamify_audio_mode_var = tk.StringVar(value="Silent")
        self.audio_memory_source_var = tk.StringVar(value="")
        self.audio_memory_epochs_var = tk.IntVar(value=60)
        self.audio_memory_mel_bins_var = tk.IntVar(value=128)
        self.audio_memory_strength_var = tk.DoubleVar(value=0.85)
        self.audio_memory_noise_var = tk.DoubleVar(value=0.06)
        self.audio_memory_keep_rhythm_var = tk.BooleanVar(value=True)
        self.audio_memory_cleanup_var = tk.BooleanVar(value=True)
        self.live_resolution_var = tk.StringVar(value="192")
        self.live_target_fps_var = tk.IntVar(value=10)
        self.live_dream_strength_var = tk.DoubleVar(value=0.25)
        self.live_memory_pull_var = tk.DoubleVar(value=0.15)
        self.live_capture_mode_var = tk.StringVar(value="Full Screen")
        self.live_show_fps_var = tk.BooleanVar(value=True)
        self.live_mini_diffusion_var = tk.BooleanVar(value=False)
        self.live_display_mode_var = tk.StringVar(value="Challenge Window")
        self.live_overlay_opacity_var = tk.DoubleVar(value=0.85)
        self.live_clickthrough_overlay_var = tk.BooleanVar(value=True)
        self.use_latent_diffusion_var = tk.BooleanVar(value=False)
        self.latent_diffusion_timesteps_var = tk.IntVar(value=75)
        self.latent_diffusion_strength_var = tk.DoubleVar(value=0.25)
        self.personality_var = tk.StringVar(value="Manual")
        self.generation_prompt_var = tk.StringVar(value="")
        self.include_memory_training_var = tk.BooleanVar(value=True)
        self.memory_training_limit_var = tk.IntVar(value=32)
        self.memory_recent_weight_var = tk.DoubleVar(value=0.7)
        self.dream_fps_var = tk.IntVar(value=16)
        self.dream_video_seconds_var = tk.IntVar(value=20)
        self.dream_video_fps_var = tk.IntVar(value=16)
        self.dream_video_audio_mode_var = tk.StringVar(value="Silent")
        self.dream_video_audio_source_var = tk.StringVar(value="")
        self.dream_structure_video_source_var = tk.StringVar(value="")
        self.dream_structure_guidance_var = tk.DoubleVar(value=0.72)
        self.dream_autoregressive_var = tk.BooleanVar(value=True)
        self.dream_feedback_strength_var = tk.DoubleVar(value=0.35)
        self.latent_drift_var = tk.DoubleVar(value=0.18)
        self.motion_smoothness_var = tk.DoubleVar(value=0.92)
        self.dream_instability_var = tk.DoubleVar(value=0.05)
        self.evolution_count_var = tk.IntVar(value=6)
        self.judge_top_k_var = tk.IntVar(value=2)
        self.judge_min_score_var = tk.DoubleVar(value=0.45)
        self.memory_finder_top_k_var = tk.IntVar(value=6)
        self.evolution_selection_var = tk.StringVar(value="")
        self.theme_mode_var = tk.StringVar(value="Auto")
        self.training_preview_enabled_var = tk.BooleanVar(value=False)
        self.timelapse_enabled_var = tk.BooleanVar(value=False)
        self.training_video_layout_var = tk.StringVar(value="Horizontal")
        self.device_choice_var = tk.StringVar(value=self._device_choice_from_device(self.device))
        self.hardware_status_var = tk.StringVar(value="")
        self.header_device_var = tk.StringVar(value="")
        self.user_level_var = tk.StringVar(value="Amateur")
        self.user_level_help_var = tk.StringVar(value=USER_LEVEL_HELP["Amateur"])
        
        # Reconstruction Mode
        self.reconstruction_mode_var = tk.StringVar(value="RGB VAE")
        self._settings_save_after_id: str | None = None
        self._settings_traces_installed = False
        
        self._load_app_settings()
        
        self.is_training = False
        self.training_thread: threading.Thread | None = None
        self.training_pause_event = threading.Event()
        self.training_pause_event.set()
        self.is_latent_diffusion_training = False
        self.is_dream_video_generating = False
        self.is_audio_memory_training = False
        self.audio_memory_profile: dict[str, object] | None = None
        self.loaded_audio_memory_path: Path | None = None
        self.realtime_dreamify_active = False
        self.realtime_dreamify_thread: threading.Thread | None = None
        self.realtime_dreamify_settings: dict[str, object] = {}
        self.model_cycle_paths: list[Path] = []
        self.model_cycle_queue: list[Path] = []
        self.model_cycle_active = False
        self.model_cycle_delay_ms = 2000

        self.current_latent = None
        self.target_latent = None
        self.interpolation_step = 0
        self.total_interpolation_steps = 20
        self.last_generated_latents: list[torch.Tensor] = []
        self.recent_memory_records = []
        self.evolution_candidates: list[dict] = []
        self.latent_map_points: list[dict] = []
        self.latent_map_window: tk.Toplevel | None = None
        self.latent_map_canvas: tk.Canvas | None = None
        self.model_map_window: tk.Toplevel | None = None
        self.model_map_tree: ttk.Treeview | None = None
        self.model_map_details_var = tk.StringVar(value="Select a .pt model to inspect its training details.")
        self.model_map_item_paths: dict[str, Path] = {}
        self.last_training_metadata: dict = {}
        self.memory_bank = MemoryBank(MEMORY_DIR)
        self.reconstruction_judge = None
        self.memory_finder = None

        self.output_window: tk.Toplevel | None = None
        self.output_canvas: tk.Canvas | None = None
        self._output_photo = None
        self.live_overlay_window: tk.Toplevel | None = None
        self.live_overlay_canvas: tk.Canvas | None = None
        self._live_overlay_photo = None
        self.live_challenge_window: tk.Toplevel | None = None
        self.live_challenge_canvas: tk.Canvas | None = None
        self._live_challenge_photo = None
        self._live_region: dict | None = None
        self.timelapse_run_dir: Path | None = None
        self.timelapse_frames_dir: Path | None = None
        self.timelapse_frame_count = 0
        self.timelapse_capture_interval = 1
        self.timelapse_last_capture_step = -1
        self.timelapse_start_time = 0.0
        self.timelapse_video_path: Path | None = None
        self.timelapse_is_encoding = False
        self.timelapse_layout = "Horizontal"
        self._theme_widgets: list[tuple[tk.Widget, str]] = []
        self._section_states: dict[str, tk.BooleanVar] = {}
        self._section_buttons: dict[str, ttk.Button] = {}
        self._section_containers: dict[str, ttk.LabelFrame] = {}
        self._section_pack_options: dict[str, dict[str, object]] = {}
        self._section_order: list[str] = []
        self._theme_palette = self._palette_for_mode()

        self._configure_styles()
        self._build_ui()
        self._install_settings_autosave()
        self._apply_user_level(update_status=False)
        self._refresh_memory_list()
        self._create_output_window()

    def _after(self, delay_ms: int, callback=None, *args):
        if callback is None:
            return self.root.after(delay_ms)
        if self._closing:
            return None

        after_id = None

        def _run_callback():
            if after_id is not None:
                self._after_ids.discard(after_id)
            if self._closing:
                return
            try:
                callback(*args)
            except tk.TclError:
                if not self._closing:
                    raise

        after_id = self.root.after(delay_ms, _run_callback)
        self._after_ids.add(after_id)
        return after_id

    def _cancel_after_events(self):
        for after_id in list(self._after_ids):
            try:
                self.root.after_cancel(after_id)
            except tk.TclError:
                pass
            finally:
                self._after_ids.discard(after_id)

    def _close(self):
        self._save_app_settings()
        self._closing = True
        self.is_training = False
        self.training_pause_event.set()
        self._stop_realtime_dreamify(update_status=False)
        train_thread = self.training_thread
        if train_thread is not None and train_thread.is_alive() and threading.current_thread() is not train_thread:
            train_thread.join(timeout=2.0)
        self.training_thread = None
        self.auto_cycle_var.set(False)
        self.dream_cycle_var.set(False)
        self.model_cycle_active = False
        self._cancel_after_events()
        if self.output_window is not None:
            try:
                self.output_window.destroy()
            except tk.TclError:
                pass
            self.output_window = None
            self.output_canvas = None
            self._output_photo = None
        self._destroy_live_overlay()
        self._destroy_live_challenge_window()
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _get_reconstruction_mode(self) -> str:
        return self.reconstruction_mode_var.get()

    def _is_wavelet_mode(self) -> bool:
        return self._get_reconstruction_mode() == "Wavelet"

    def _is_rgb_mode(self) -> bool:
        return self._get_reconstruction_mode() == "RGB VAE"

    @staticmethod
    def _wavelet_size_for_resolution(resolution: int) -> tuple[int, int]:
        resolution = max(32, int(resolution))
        if resolution % 2 != 0:
            resolution += 1
        return (resolution // 2, resolution // 2)

    @staticmethod
    def _rgb_size_for_wavelet_output(output_size: tuple[int, int]) -> tuple[int, int]:
        if len(output_size) != 2:
            return (256, 256)
        return (int(output_size[0]) * 2, int(output_size[1]) * 2)

    def _model_uses_wavelet(self) -> bool:
        return self.model is not None and int(getattr(self.model, "in_channels", 3)) == 12

    def _model_rgb_input_size(self) -> tuple[int, int]:
        if self.model is None:
            size = int(self.resolution_var.get())
            return (size, size)
        output_size = tuple(getattr(self.model, "output_size", (int(self.resolution_var.get()), int(self.resolution_var.get()))))
        if self._model_uses_wavelet():
            return self._rgb_size_for_wavelet_output(output_size)
        return (int(output_size[0]), int(output_size[1]))

    def _sanitize_model_batch(self, tensor: torch.Tensor, *, is_wavelet: bool) -> torch.Tensor:
        if is_wavelet:
            return torch.nan_to_num(tensor, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
        return torch.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    def _decode_model_output_to_rgb(self, output_tensor: torch.Tensor, mode: str | None = None) -> torch.Tensor:
        use_wavelet = (mode == "Wavelet") if mode is not None else (
            self._model_uses_wavelet()
            or (output_tensor.ndim >= 3 and int(output_tensor.shape[-3]) == 12)
        )
        if use_wavelet:
            return wavelet_to_rgb(output_tensor).clamp(0.0, 1.0)
        return output_tensor.clamp(0.0, 1.0)

    def _output_display_size_for_image(self, image_size: tuple[int, int] | None = None) -> tuple[int, int]:
        """Return the output-window canvas size without lying about old model sizes.

        256 models get a 256 preview, 512 models get a 512 preview, and Wavelet
        checkpoints are converted back to their real RGB display size. Very large
        models are capped to fit on-screen, but the image is never stretched upward.
        """
        if image_size is not None and len(image_size) == 2:
            width, height = int(image_size[0]), int(image_size[1])
        else:
            width, height = self._model_rgb_input_size()

        width = max(32, int(width))
        height = max(32, int(height))
        try:
            max_w = max(256, int(self.root.winfo_screenwidth()) - 80)
            max_h = max(256, int(self.root.winfo_screenheight()) - 140)
        except tk.TclError:
            max_w, max_h = 1024, 1024

        scale = min(1.0, max_w / max(1, width), max_h / max(1, height))
        return max(32, int(width * scale)), max(32, int(height * scale))

    def _resize_output_window(self, image_size: tuple[int, int] | None = None, *, keep_position: bool = True) -> tuple[int, int]:
        inner_w, inner_h = self._output_display_size_for_image(image_size)
        pad = 8
        win = self.output_window
        canvas = self.output_canvas
        if win is None or canvas is None:
            return inner_w, inner_h
        try:
            old_x, old_y = win.winfo_x(), win.winfo_y()
            canvas.configure(width=inner_w, height=inner_h)
            win.minsize(inner_w + pad * 2, inner_h + pad * 2 + 24)
            rgb_w, rgb_h = image_size if image_size is not None else self._model_rgb_input_size()
            win.title(f"APVD Output Display - {int(rgb_w)}x{int(rgb_h)}")
            geometry = f"{inner_w + pad * 2}x{inner_h + pad * 2 + 24}"
            if keep_position:
                geometry += f"+{old_x}+{old_y}"
            win.geometry(geometry)
        except tk.TclError:
            self.output_window = None
            self.output_canvas = None
            self._output_photo = None
        return inner_w, inner_h

    def _create_output_window(self):
        inner_w, inner_h = self._output_display_size_for_image()
        pad = 8

        if self.output_window is not None:
            try:
                if self.output_window.winfo_exists():
                    self._resize_output_window()
                    self.output_window.deiconify()
                    self.output_window.lift()
                    return
            except tk.TclError:
                pass
            self.output_window = None
            self.output_canvas = None

        w = tk.Toplevel(self.root)
        model_w, model_h = self._model_rgb_input_size()
        w.title(f"APVD Output Display - {int(model_w)}x{int(model_h)}")
        w.minsize(inner_w + pad * 2, inner_h + pad * 2 + 24)
        w.geometry(f"{inner_w + pad * 2}x{inner_h + pad * 2 + 24}")

        self.root.update_idletasks()
        try:
            x = self.root.winfo_x() + self.root.winfo_width() + 12
            y = self.root.winfo_y()
            w.geometry(f"{inner_w + pad * 2}x{inner_h + pad * 2 + 24}+{x}+{y}")
        except tk.TclError:
            pass

        outer = ttk.Frame(w, padding=pad)
        outer.pack(fill=tk.BOTH, expand=True)
        cv = tk.Canvas(
            outer,
            width=inner_w,
            height=inner_h,
            bg=self._theme_palette["canvas"],
            highlightthickness=0,
        )
        cv.pack()
        self._register_theme_widget(cv, "output_canvas")

        self.output_window = w
        self.output_canvas = cv

        def _on_close():
            if self.output_window is not None:
                try:
                    self.output_window.destroy()
                except tk.TclError:
                    pass
            self.output_window = None
            self.output_canvas = None
            self._output_photo = None

        w.protocol("WM_DELETE_WINDOW", _on_close)

    def _create_live_overlay_window(self, *, opacity: float, clickthrough: bool):
        width = max(640, int(self.root.winfo_screenwidth()))
        height = max(480, int(self.root.winfo_screenheight()))
        if self.live_overlay_window is not None:
            try:
                if self.live_overlay_window.winfo_exists():
                    self.live_overlay_window.geometry(f"{width}x{height}+0+0")
                    self.live_overlay_window.deiconify()
                    self.live_overlay_window.lift()
                    self.live_overlay_window.attributes("-alpha", 1.0)
                    self.live_overlay_canvas.configure(width=width, height=height)
                    self._apply_live_overlay_clickthrough(clickthrough)
                    return
            except tk.TclError:
                pass
            self.live_overlay_window = None
            self.live_overlay_canvas = None
            self._live_overlay_photo = None

        w = tk.Toplevel(self.root)
        w.title("APVD Real-Time Dreamify Overlay")
        w.configure(bg="black")
        w.overrideredirect(True)
        w.attributes("-topmost", True)
        w.attributes("-alpha", 1.0)
        w.geometry(f"{width}x{height}+0+0")

        cv = tk.Canvas(w, width=width, height=height, bg="black", highlightthickness=0, bd=0)
        cv.pack(fill=tk.BOTH, expand=True)
        w.bind("<Escape>", lambda _event=None: self._stop_realtime_dreamify())
        self.live_overlay_window = w
        self.live_overlay_canvas = cv
        try:
            w.update_idletasks()
        except tk.TclError:
            pass
        self._apply_live_overlay_clickthrough(clickthrough)

    def _destroy_live_overlay(self):
        if self.live_overlay_window is not None:
            try:
                self.live_overlay_window.destroy()
            except tk.TclError:
                pass
        self.live_overlay_window = None
        self.live_overlay_canvas = None
        self._live_overlay_photo = None

    def _create_live_challenge_window(self):
        if self.live_challenge_window is not None:
            try:
                if self.live_challenge_window.winfo_exists():
                    self.live_challenge_window.deiconify()
                    self.live_challenge_window.lift()
                    return
            except tk.TclError:
                pass
            self.live_challenge_window = None
            self.live_challenge_canvas = None
            self._live_challenge_photo = None

        w = tk.Toplevel(self.root)
        w.title("APVD Real-Time Dreamify - Challenge Window")
        w.attributes("-topmost", True)
        w.resizable(True, True)
        try:
            self.root.update_idletasks()
            rx = self.root.winfo_x() + self.root.winfo_width() + 12
            ry = self.root.winfo_y()
            w.geometry(f"960x540+{rx}+{ry}")
        except tk.TclError:
            w.geometry("960x540")
        w.configure(bg="black")

        tip = ttk.Label(
            w,
            text="Challenge Window — position this outside your capture region to avoid feedback. Esc stops.",
            style="SurfaceMuted.TLabel",
            anchor=tk.CENTER,
        )
        tip.pack(side=tk.TOP, fill=tk.X, padx=4, pady=(2, 0))

        cv = tk.Canvas(w, bg="black", highlightthickness=0, bd=0)
        cv.pack(fill=tk.BOTH, expand=True)
        w.bind("<Escape>", lambda _e=None: self._stop_realtime_dreamify())
        self.live_challenge_window = w
        self.live_challenge_canvas = cv
        try:
            w.update_idletasks()
        except tk.TclError:
            pass

        def _on_close():
            self._stop_realtime_dreamify()
            self._destroy_live_challenge_window()
        w.protocol("WM_DELETE_WINDOW", _on_close)

    def _destroy_live_challenge_window(self):
        if self.live_challenge_window is not None:
            try:
                self.live_challenge_window.destroy()
            except tk.TclError:
                pass
        self.live_challenge_window = None
        self.live_challenge_canvas = None
        self._live_challenge_photo = None

    def _display_challenge_frame(self, pil_img: Image.Image):
        from PIL import ImageTk
        if self._closing:
            return
        win = self.live_challenge_window
        canvas = self.live_challenge_canvas
        if win is None or canvas is None:
            return
        try:
            if not win.winfo_exists():
                self.live_challenge_window = None
                self.live_challenge_canvas = None
                self._live_challenge_photo = None
                return
            cw = max(1, canvas.winfo_width())
            ch = max(1, canvas.winfo_height())
            img = pil_img.convert("RGB")
            iw, ih = img.size
            scale = min(cw / max(1, iw), ch / max(1, ih))
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            fitted = img.resize((nw, nh), Image.Resampling.BILINEAR)
            bg = Image.new("RGB", (cw, ch), (0, 0, 0))
            bg.paste(fitted, ((cw - nw) // 2, (ch - nh) // 2))
            self._live_challenge_photo = ImageTk.PhotoImage(bg, master=win)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=self._live_challenge_photo)
        except tk.TclError:
            self.live_challenge_window = None
            self.live_challenge_canvas = None
            self._live_challenge_photo = None

    def _prompt_capture_region(self) -> dict | None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Set Capture Region")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self._theme_palette["bg"])
        try:
            dialog.geometry(f"+{self.root.winfo_x() + 40}+{self.root.winfo_y() + 40}")
        except tk.TclError:
            pass

        result: dict | None = None
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()

        frame = ttk.Frame(dialog, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Enter the screen region to capture:").grid(row=0, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))
        fields = {}
        defaults = {"X": 0, "Y": 0, "Width": sw, "Height": sh}
        if self._live_region is not None:
            prev = self._live_region
            defaults = {"X": prev.get("left", 0), "Y": prev.get("top", 0), "Width": prev.get("width", sw), "Height": prev.get("height", sh)}
        for col, (label, default) in enumerate(defaults.items()):
            ttk.Label(frame, text=label).grid(row=1, column=col * 2, sticky=tk.E, padx=(0, 4))
            var = tk.StringVar(value=str(default))
            ttk.Entry(frame, textvariable=var, width=7).grid(row=1, column=col * 2 + 1, padx=(0, 10))
            fields[label] = var

        warn_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=warn_var, foreground="orange").grid(row=2, column=0, columnspan=4, sticky=tk.W, pady=(6, 0))

        def _ok():
            nonlocal result
            try:
                x = max(0, int(fields["X"].get()))
                y = max(0, int(fields["Y"].get()))
                w = max(64, int(fields["Width"].get()))
                h = max(64, int(fields["Height"].get()))
            except ValueError:
                warn_var.set("Please enter valid integers for all fields.")
                return
            region = {"left": x, "top": y, "width": w, "height": h}
            cwin = self.live_challenge_window
            if cwin is not None:
                try:
                    if cwin.winfo_exists():
                        cx, cy = cwin.winfo_x(), cwin.winfo_y()
                        cw2, ch2 = cwin.winfo_width(), cwin.winfo_height()
                        overlap = not (cx + cw2 <= x or cx >= x + w or cy + ch2 <= y or cy >= y + h)
                        if overlap:
                            warn_var.set("Challenge Window overlaps capture region — move it to avoid feedback flicker.")
                            return
                except tk.TclError:
                    pass
            result = region
            dialog.destroy()

        def _cancel():
            dialog.destroy()

        btn_row = ttk.Frame(frame)
        btn_row.grid(row=3, column=0, columnspan=4, pady=(14, 0), sticky=tk.E)
        ttk.Button(btn_row, text="OK", command=_ok, style="Accent.TButton").pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row, text="Cancel", command=_cancel).pack(side=tk.LEFT)
        dialog.wait_window()
        return result

    def _apply_live_overlay_clickthrough(self, enabled: bool):
        window = self.live_overlay_window
        if window is None:
            return
        self._set_window_clickthrough(window, enabled)
        self._after(100, lambda: self._set_window_clickthrough(window, enabled))

    def _set_window_clickthrough(self, window: tk.Toplevel, enabled: bool):
        system = platform.system()
        if system == "Windows":
            try:
                import ctypes
                window.update_idletasks()
                hwnd = window.winfo_id()
                user32 = ctypes.windll.user32
                gwl_exstyle = -20
                ws_ex_layered = 0x00080000
                ws_ex_transparent = 0x00000020
                swp_nosize = 0x0001
                swp_nomove = 0x0002
                swp_nozorder = 0x0004
                swp_framechanged = 0x0020
                get_style = getattr(user32, "GetWindowLongPtrW", user32.GetWindowLongW)
                set_style = getattr(user32, "SetWindowLongPtrW", user32.SetWindowLongW)
                style = get_style(hwnd, gwl_exstyle)
                style |= ws_ex_layered
                if enabled:
                    style |= ws_ex_transparent
                else:
                    style &= ~ws_ex_transparent
                set_style(hwnd, gwl_exstyle, style)
                user32.SetWindowPos(hwnd, -1, 0, 0, 0, 0, swp_nomove | swp_nosize | swp_nozorder | swp_framechanged)
            except Exception:
                pass
        elif system == "Darwin":
            try:
                if enabled:
                    window.attributes("-transparent", True)
                    window.tk.call("wm", "attributes", window, "-type", "splash")
                else:
                    window.attributes("-transparent", False)
            except Exception:
                pass
        else:
            try:
                if enabled:
                    window.tk.call("wm", "attributes", window, "-type", "splash")
                    window.attributes("-alpha", window.attributes("-alpha"))
                else:
                    window.tk.call("wm", "attributes", window, "-type", "normal")
            except Exception:
                pass

    def _exclude_overlay_from_capture(self):
        if os.name != "nt":
            return
        win = self.live_overlay_window
        if win is None:
            return
        try:
            import ctypes
            WDA_EXCLUDEFROMCAPTURE = 0x00000011
            hwnd = win.winfo_id()
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
        except Exception:
            pass

    def _live_escape_is_down(self) -> bool:
        if os.name != "nt":
            return False
        try:
            import ctypes
            return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)
        except Exception:
            return False

    def _palette_for_mode(self) -> dict[str, str]:
        mode = self.theme_mode_var.get() if hasattr(self, "theme_mode_var") else "Auto"
        if mode == "Auto":
            hour = datetime.now().hour
            mode = "Day" if 7 <= hour < 19 else "Night"

        if mode == "Day":
            return {
                "bg": "#eef2f7", "surface": "#ffffff", "surface_alt": "#f7f9fc",
                "text": "#1c2430", "muted": "#596579", "accent": "#2d6cdf",
                "accent_text": "#ffffff", "border": "#cfd8e6", "canvas": "#e7edf6",
                "canvas_border": "#9baac0", "list_bg": "#ffffff", "entry": "#ffffff",
            }
        return {
            "bg": "#121421", "surface": "#1c2033", "surface_alt": "#252a43",
            "text": "#edf1ff", "muted": "#aab3cf", "accent": "#8b7cf6",
            "accent_text": "#ffffff", "border": "#3b4266", "canvas": "#101323",
            "canvas_border": "#59608a", "list_bg": "#171b2d", "entry": "#20253a",
        }

    def _configure_styles(self) -> None:
        palette = self._theme_palette
        self.root.configure(bg=palette["bg"])
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".", background=palette["bg"], foreground=palette["text"], fieldbackground=palette["entry"])
        style.configure("TFrame", background=palette["bg"])
        style.configure("Surface.TFrame", background=palette["surface"])
        style.configure("TLabel", background=palette["bg"], foreground=palette["text"])
        style.configure("Surface.TLabel", background=palette["surface"], foreground=palette["text"])
        style.configure("Category.TLabel", background=palette["surface"], foreground=palette["accent"], font=("", 9, "bold"))
        style.configure("Muted.TLabel", background=palette["bg"], foreground=palette["muted"])
        style.configure("SurfaceMuted.TLabel", background=palette["surface"], foreground=palette["muted"])
        style.configure("Status.TLabel", background=palette["bg"], foreground=palette["text"], font=("", 10, "bold"))
        style.configure("TLabelframe", background=palette["surface"], bordercolor=palette["border"], relief="solid")
        style.configure("TLabelframe.Label", background=palette["surface"], foreground=palette["text"], font=("", 10, "bold"))
        style.configure("TButton", background=palette["surface_alt"], foreground=palette["text"], bordercolor=palette["border"], focusthickness=0, padding=(10, 6))
        style.map("TButton", background=[("active", palette["accent"]), ("pressed", palette["accent"])], foreground=[("active", palette["accent_text"]), ("pressed", palette["accent_text"])])
        style.configure("Accent.TButton", background=palette["accent"], foreground=palette["accent_text"], bordercolor=palette["accent"])
        style.map("Accent.TButton", background=[("active", palette["accent"]), ("pressed", palette["accent"])])
        style.configure("TCheckbutton", background=palette["surface"], foreground=palette["text"])
        style.map("TCheckbutton", background=[("active", palette["surface_alt"])])
        style.configure("TCombobox", fieldbackground=palette["entry"], background=palette["surface_alt"], foreground=palette["text"])
        style.configure("TSpinbox", fieldbackground=palette["entry"], background=palette["surface_alt"], foreground=palette["text"], arrowsize=13)
        style.configure("TEntry", fieldbackground=palette["entry"], foreground=palette["text"], insertcolor=palette["text"])
        style.configure("Horizontal.TProgressbar", background=palette["accent"], troughcolor=palette["surface_alt"], bordercolor=palette["border"])

    def _register_theme_widget(self, widget: tk.Widget, role: str) -> tk.Widget:
        self._theme_widgets.append((widget, role))
        return widget

    def _apply_registered_theme(self) -> None:
        palette = self._theme_palette
        for widget, role in list(self._theme_widgets):
            try:
                if role == "canvas":
                    widget.configure(bg=palette["canvas"], highlightbackground=palette["canvas_border"])
                elif role == "scroll_canvas":
                    widget.configure(bg=palette["bg"])
                elif role == "output_canvas":
                    widget.configure(bg=palette["canvas"])
                elif role == "listbox":
                    widget.configure(bg=palette["list_bg"], fg=palette["text"], selectbackground=palette["accent"], selectforeground=palette["accent_text"], highlightbackground=palette["border"])
                elif role == "scale":
                    widget.configure(bg=palette["surface"], fg=palette["text"], troughcolor=palette["surface_alt"], highlightthickness=0, activebackground=palette["accent"])
                elif role == "spinbox":
                    widget.configure(bg=palette["entry"], fg=palette["text"], buttonbackground=palette["surface_alt"], insertbackground=palette["text"], highlightbackground=palette["border"])
            except tk.TclError:
                continue

    def _set_theme(self, *_args) -> None:
        self._theme_palette = self._palette_for_mode()
        self._configure_styles()
        self._apply_registered_theme()
        if self.output_canvas is not None:
            try:
                self.output_canvas.configure(bg=self._theme_palette["canvas"])
            except tk.TclError:
                pass
        if self.live_overlay_canvas is not None:
            try:
                self.live_overlay_canvas.configure(bg="black")
            except tk.TclError:
                pass

    def _settings_variables(self) -> dict[str, tk.Variable]:
        names = [
            "epochs_var", "resolution_var", "batch_size_var", "loader_workers_var",
            "prefetch_batches_var", "dataset_cache_items_var", "training_intensity_var",
            "learning_rate_var", "precision_mode_var", "loss_mode_var", "mixed_precision_var",
            "nan_guard_var", "video_stride_var", "video_max_frames_var", "iterations_var",
            "show_iterations_var", "auto_cycle_var", "dream_cycle_var", "blend_mode_var",
            "blend_count_var", "output_count_var", "use_mini_diffusion_var",
            "diffusion_steps_var", "diffusion_strength_var", "dream_strength_var",
            "memory_pull_var", "dreamify_frame_skip_var", "keep_original_audio_var",
            "dreamify_audio_mode_var", "audio_memory_source_var", "audio_memory_epochs_var",
            "audio_memory_mel_bins_var", "audio_memory_strength_var", "audio_memory_noise_var",
            "audio_memory_keep_rhythm_var", "audio_memory_cleanup_var",
            "live_resolution_var", "live_target_fps_var", "live_dream_strength_var",
            "live_memory_pull_var", "live_capture_mode_var", "live_show_fps_var",
            "live_mini_diffusion_var", "live_display_mode_var", "live_overlay_opacity_var",
            "live_clickthrough_overlay_var", "use_latent_diffusion_var",
            "latent_diffusion_timesteps_var", "latent_diffusion_strength_var",
            "personality_var", "generation_prompt_var", "include_memory_training_var",
            "memory_training_limit_var", "memory_recent_weight_var", "dream_fps_var",
            "dream_video_seconds_var", "dream_video_fps_var", "dream_video_audio_mode_var",
            "dream_video_audio_source_var", "dream_structure_video_source_var",
            "dream_structure_guidance_var", "dream_autoregressive_var",
            "dream_feedback_strength_var", "latent_drift_var", "motion_smoothness_var",
            "dream_instability_var", "evolution_count_var", "judge_top_k_var",
            "judge_min_score_var", "memory_finder_top_k_var", "evolution_selection_var",
            "theme_mode_var", "training_preview_enabled_var", "timelapse_enabled_var",
            "training_video_layout_var", "device_choice_var", "user_level_var",
            "reconstruction_mode_var",
        ]
        return {name: getattr(self, name) for name in names if hasattr(self, name)}

    def _load_app_settings(self) -> None:
        try:
            if not APP_SETTINGS_PATH.exists():
                return
            data = json.loads(APP_SETTINGS_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
        except Exception:
            logger.debug("Could not load APVD user settings.", exc_info=True)
            return

        saved_vars = data.get("variables", {})
        if isinstance(saved_vars, dict):
            for name, value in saved_vars.items():
                var = getattr(self, name, None)
                if not isinstance(var, tk.Variable):
                    continue
                try:
                    if name == "device_choice_var" and value not in self._available_device_choices():
                        continue
                    if name == "user_level_var" and value not in USER_LEVEL_CHOICES:
                        continue
                    if name == "theme_mode_var" and value not in ("Auto", "Day", "Night"):
                        continue
                    if name == "reconstruction_mode_var" and value not in ("RGB VAE", "Wavelet"):
                        continue
                    var.set(value)
                except Exception:
                    logger.debug("Skipping saved setting %s=%r", name, value, exc_info=True)

        scales = data.get("scales", {})
        if isinstance(scales, dict):
            self._pending_scale_settings = scales
        else:
            self._pending_scale_settings = {}
        self._pending_section_states = data.get("sections", {}) if isinstance(data.get("sections", {}), dict) else {}
        self.user_level_help_var.set(USER_LEVEL_HELP.get(self.user_level_var.get(), USER_LEVEL_HELP["Amateur"]))

    def _collect_app_settings(self) -> dict[str, object]:
        variables = {}
        for name, var in self._settings_variables().items():
            try:
                variables[name] = var.get()
            except Exception:
                continue
        scales = {}
        for name in ("var_scale", "speed_scale"):
            scale = getattr(self, name, None)
            if scale is not None:
                try:
                    scales[name] = float(scale.get())
                except Exception:
                    pass
        sections = {}
        for title, state in self._section_states.items():
            try:
                sections[title] = bool(state.get())
            except Exception:
                pass
        return {"version": 1, "saved_at": datetime.now().isoformat(timespec="seconds"), "variables": variables, "scales": scales, "sections": sections}

    def _save_app_settings(self) -> None:
        try:
            APP_SETTINGS_PATH.write_text(json.dumps(self._collect_app_settings(), indent=2), encoding="utf-8")
        except Exception:
            logger.debug("Could not save APVD user settings.", exc_info=True)

    def _queue_settings_save(self, *_args) -> None:
        if self._closing or not getattr(self, "_settings_traces_installed", False):
            return
        if self._settings_save_after_id is not None:
            try:
                self.root.after_cancel(self._settings_save_after_id)
            except tk.TclError:
                pass
        self._settings_save_after_id = self.root.after(500, self._save_app_settings)

    def _install_settings_autosave(self) -> None:
        if self._settings_traces_installed:
            return
        for var in self._settings_variables().values():
            try:
                var.trace_add("write", self._queue_settings_save)
            except Exception:
                pass
        for name in ("var_scale", "speed_scale"):
            scale = getattr(self, name, None)
            if scale is not None:
                scale.configure(command=lambda _value, self=self: self._queue_settings_save())
        self._settings_traces_installed = True

    def _restore_post_ui_settings(self) -> None:
        scales = getattr(self, "_pending_scale_settings", {})
        if isinstance(scales, dict):
            for name, value in scales.items():
                scale = getattr(self, name, None)
                if scale is not None:
                    try:
                        scale.set(float(value))
                    except Exception:
                        pass
        sections = getattr(self, "_pending_section_states", {})
        if isinstance(sections, dict):
            for title, open_state in sections.items():
                state = self._section_states.get(title)
                button = self._section_buttons.get(title)
                container = self._section_containers.get(title)
                if state is None or button is None or container is None:
                    continue
                try:
                    body = container.winfo_children()[1]
                    if bool(open_state):
                        body.pack(fill=tk.X, pady=(8, 0))
                        button.configure(text="-")
                    else:
                        body.forget()
                        button.configure(text="+")
                    state.set(bool(open_state))
                except Exception:
                    pass

    def _apply_user_level(self, *_args, update_status: bool = True) -> None:
        level = self.user_level_var.get()
        if level not in USER_LEVEL_CHOICES:
            level = "Amateur"
            self.user_level_var.set(level)
        visible = USER_LEVEL_SECTION_VISIBILITY.get(level, "ALL")
        self.user_level_help_var.set(USER_LEVEL_HELP.get(level, USER_LEVEL_HELP["Amateur"]))
        for container in self._section_containers.values():
            try:
                container.pack_forget()
            except tk.TclError:
                pass
        for title in self._section_order:
            container = self._section_containers.get(title)
            if container is None:
                continue
            should_show = visible == "ALL" or title in visible
            if not should_show:
                continue
            try:
                container.pack(**self._section_pack_options.get(title, {"fill": tk.X, "padx": 12, "pady": 6}))
            except tk.TclError:
                pass
        if update_status and hasattr(self, "status_var"):
            self.status_var.set(f"User level set to {level}: {self.user_level_help_var.get()}")
        self._queue_settings_save()

    def _category_label(self, parent: tk.Widget, text: str, row: int) -> None:
        ttk.Label(parent, text=text, style="Category.TLabel").grid(row=row, column=0, columnspan=6, sticky=tk.W, pady=(10, 4))

    @staticmethod
    def _device_choice_from_device(device: torch.device) -> str:
        device_type = getattr(device, "type", str(device)).lower()
        if device_type == "cuda":
            return "CUDA"
        if device_type == "mps":
            return "MPS"
        return "CPU"

    def _available_device_choices(self) -> list[str]:
        choices = ["CPU"]
        if torch.cuda.is_available():
            choices.append("CUDA")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            choices.append("MPS")
        return choices

    @staticmethod
    def _device_from_choice(choice: str) -> torch.device:
        if choice == "CUDA":
            return torch.device("cuda")
        if choice == "MPS":
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _format_gb(byte_count: int) -> str:
        return f"{byte_count / (1024 ** 3):.1f} GB"

    def _describe_device_choice(self, choice: str) -> str:
        if choice == "CUDA":
            if not torch.cuda.is_available():
                return "CUDA was not found on this computer."
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            return f"CUDA GPU: {torch.cuda.get_device_name(index)} | VRAM: {self._format_gb(props.total_memory)} | GPU {index + 1}/{torch.cuda.device_count()}"
        if choice == "MPS":
            return "Apple Metal GPU acceleration is available."
        cpu_name = platform.processor() or platform.machine() or "CPU"
        return f"CPU: {cpu_name}"

    def _cuda_compute_capability(self) -> tuple[int, int]:
        if not torch.cuda.is_available():
            return (0, 0)
        try:
            major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
            return int(major), int(minor)
        except Exception:
            return (0, 0)

    def _cuda_gpu_name_lower(self) -> str:
        if not torch.cuda.is_available():
            return ""
        try:
            return str(torch.cuda.get_device_name(torch.cuda.current_device())).lower()
        except Exception:
            return ""

    def _bf16_supported(self) -> bool:
        if getattr(self.device, "type", "") != "cuda" or not torch.cuda.is_available():
            return False
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            major, _minor = self._cuda_compute_capability()
            return major >= 8

    def _resolve_auto_precision_mode(self) -> str:
        """Choose a safe default for unknown consumer hardware. Users can override this."""
        if getattr(self.device, "type", "") != "cuda" or not torch.cuda.is_available():
            return "FP32 Stable"
        name = self._cuda_gpu_name_lower()
        major, _minor = self._cuda_compute_capability()
        if self._bf16_supported():
            return "BF16 Balanced"
        # Older CUDA cards and many laptop/mobile GPUs are more likely to be cranky
        # with FP16, so Auto chooses stability there. Dedicated RTX desktop GPUs can
        # still pick FP16 Fast manually if desired.
        if major < 7 or any(word in name for word in ("laptop", "mobile", "mx", "quadro m", "gtx 9", "gtx 10")):
            return "FP32 Stable"
        return "FP16 Fast"

    def _resolve_precision_settings(self) -> dict[str, object]:
        requested = self.precision_mode_var.get() if hasattr(self, "precision_mode_var") else "Auto Recommended"
        if requested not in PRECISION_MODE_CHOICES:
            requested = "Auto Recommended"
        resolved = self._resolve_auto_precision_mode() if requested == "Auto Recommended" else requested
        device_type = getattr(self.device, "type", "")
        if device_type != "cuda":
            resolved = "FP32 Stable"

        autocast_enabled = False
        autocast_dtype = None
        scaler_enabled = False
        if device_type == "cuda" and resolved == "FP16 Fast":
            autocast_enabled = True
            autocast_dtype = torch.float16
            scaler_enabled = True
        elif device_type == "cuda" and resolved == "BF16 Balanced":
            if self._bf16_supported():
                autocast_enabled = True
                autocast_dtype = torch.bfloat16
                scaler_enabled = False
            else:
                resolved = "FP32 Stable"

        self.mixed_precision_var.set(bool(autocast_enabled))
        dtype_label = "fp32"
        if autocast_dtype is torch.float16:
            dtype_label = "fp16"
        elif autocast_dtype is torch.bfloat16:
            dtype_label = "bf16"
        return {
            "requested": requested,
            "resolved": resolved,
            "autocast_enabled": autocast_enabled,
            "autocast_dtype": autocast_dtype,
            "scaler_enabled": scaler_enabled,
            "non_blocking": device_type == "cuda",
            "dtype_label": dtype_label,
        }

    def _on_precision_mode_changed(self, *_args) -> None:
        settings = self._resolve_precision_settings()
        if hasattr(self, "status_var"):
            requested = str(settings["requested"])
            resolved = str(settings["resolved"])
            if requested == "Auto Recommended":
                self.status_var.set(f"Precision Mode: Auto Recommended → {resolved}.")
            else:
                self.status_var.set(f"Precision Mode: {resolved}.")

    def _cache_items_per_worker(self, requested_total: int, loader_workers: int) -> int:
        """Return a per-worker cache budget so the UI value acts like an approximate total RAM budget."""
        requested_total = max(0, int(requested_total))
        loader_workers = max(0, int(loader_workers))
        if requested_total <= 0:
            return 0
        if loader_workers <= 0:
            return requested_total
        return max(1, requested_total // loader_workers)

    def _build_training_dataloader(self, dataset: Dataset, *, batch_size: int, loader_workers: int, prefetch_batches: int, shuffle: bool = True) -> DataLoader:
        cuda_enabled = getattr(self.device, "type", "") == "cuda"
        loader_kwargs = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": loader_workers, "pin_memory": cuda_enabled}
        if loader_workers > 0:
            loader_kwargs.update({"persistent_workers": True, "prefetch_factor": prefetch_batches})
        logger.info(
            "Building DataLoader: samples=%s batch=%s workers=%s prefetch=%s cache_per_worker=%s",
            len(dataset),
            batch_size,
            loader_workers,
            prefetch_batches if loader_workers > 0 else 0,
            getattr(dataset, "cache_limit", 0),
        )
        return DataLoader(dataset, **loader_kwargs)

    def _on_loss_mode_changed(self, *_args) -> None:
        mode = self.loss_mode_var.get() if hasattr(self, "loss_mode_var") else "Classic VAE"
        if mode not in LOSS_MODE_CHOICES:
            mode = "Classic VAE"
            self.loss_mode_var.set(mode)
        if hasattr(self, "loss_mode_help_var"):
            self.loss_mode_help_var.set(LOSS_MODE_HELP.get(mode, LOSS_MODE_HELP["Classic VAE"]))
        if hasattr(self, "status_var"):
            self.status_var.set(f"Loss mode set to {mode}.")

    @staticmethod
    def _loss_mode_weights(mode: str) -> dict[str, float]:
        """Return blend weights for APVD training losses.

        base = existing VAE reconstruction+KL loss.
        l1 = direct shape-preserving L1 difference.
        edge = Sobel-style edge/outline difference.
        """
        if mode == "Structure Stable":
            return {"base": 0.80, "l1": 0.20, "edge": 0.00}
        if mode == "Sharp Detail":
            return {"base": 0.75, "l1": 0.15, "edge": 0.10}
        if mode == "Experimental Structure":
            return {"base": 0.65, "l1": 0.20, "edge": 0.15}
        return {"base": 1.00, "l1": 0.00, "edge": 0.00}

    @staticmethod
    def _sobel_edges_for_loss(tensor: torch.Tensor) -> torch.Tensor:
        """Small Sobel edge extractor used only for structure-aware loss modes."""
        if tensor.ndim != 4:
            return tensor
        channels = int(tensor.shape[1])
        if channels <= 0:
            return tensor
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        kernel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
            dtype=tensor.dtype,
            device=tensor.device,
        ).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        gx = F.conv2d(tensor, kernel_x, padding=1, groups=channels)
        gy = F.conv2d(tensor, kernel_y, padding=1, groups=channels)
        return torch.sqrt((gx * gx) + (gy * gy) + 1e-6)

    def _apvd_training_loss(
        self,
        recon: torch.Tensor,
        batch: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        *,
        is_wavelet: bool,
        loss_mode: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """APVD loss with optional structure/detail stabilizers.

        This intentionally does not turn APVD into a GAN. It keeps the normal VAE
        objective and optionally adds L1/edge pressure for shape consistency.
        """
        if loss_mode not in LOSS_MODE_CHOICES:
            loss_mode = "Classic VAE"
        weights = self._loss_mode_weights(loss_mode)
        base_loss = vae_loss(
            recon,
            batch,
            mu,
            logvar,
            reconstruction_loss="mse" if is_wavelet else "bce",
        )
        total = base_loss * float(weights["base"])
        l1_loss = recon.new_tensor(0.0)
        edge_loss = recon.new_tensor(0.0)
        if weights["l1"] > 0.0:
            recon_l1 = torch.nan_to_num(recon, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
            batch_l1 = torch.nan_to_num(batch, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
            l1_loss = F.l1_loss(recon_l1, batch_l1, reduction="sum")
            total = total + (l1_loss * float(weights["l1"]))
        if weights["edge"] > 0.0:
            recon_edges = self._sobel_edges_for_loss(recon)
            batch_edges = self._sobel_edges_for_loss(batch)
            edge_loss = F.l1_loss(recon_edges, batch_edges, reduction="sum")
            total = total + (edge_loss * float(weights["edge"]))
        metrics = {
            "base_loss": float(base_loss.detach().float().item()),
            "l1_loss": float(l1_loss.detach().float().item()),
            "edge_loss": float(edge_loss.detach().float().item()),
            "base_weight": float(weights["base"]),
            "l1_weight": float(weights["l1"]),
            "edge_weight": float(weights["edge"]),
        }
        return total, metrics

    @staticmethod
    def _training_intensity_percent(choice: str) -> int:
        match = re.match(r"\s*(\d+)", str(choice))
        if not match:
            return 50
        return max(10, min(100, int(match.group(1))))

    def _training_intensity_profile(self, choice: str | None = None) -> dict[str, float | int]:
        percent = self._training_intensity_percent(choice or self.training_intensity_var.get())
        max_workers = max(1, min(16, (os.cpu_count() or 4) - 1))
        if percent >= 100:
            workers, prefetch, throttle_cap = max_workers, 4, 0.0
        elif percent >= 75:
            workers, prefetch, throttle_cap = max(1, min(max_workers, 8)), 3, 0.25
        elif percent >= 50:
            workers, prefetch, throttle_cap = max(1, min(max_workers, 4)), 2, 0.75
        elif percent >= 25:
            workers, prefetch, throttle_cap = max(1, min(max_workers, 2)), 1, 1.25
        else:
            workers, prefetch, throttle_cap = 1, 1, 1.75
        return {"percent": percent, "workers": workers, "prefetch": prefetch, "throttle_cap": throttle_cap}

    def _apply_training_intensity(self, *_args) -> None:
        profile = self._training_intensity_profile()
        self.loader_workers_var.set(int(profile["workers"]))
        self.prefetch_batches_var.set(int(profile["prefetch"]))
        percent = int(profile["percent"])
        if hasattr(self, "status_var"):
            if percent >= 100:
                self.status_var.set("Training intensity set to 100%: maximum data loading with no throttling.")
            else:
                self.status_var.set(f"Training intensity set to {percent}%: data loading tuned down and GPU work throttled between batches.")

    def _refresh_hardware_status(self) -> None:
        choice = self._device_choice_from_device(self.device)
        if self.device_choice_var.get() != choice:
            self.device_choice_var.set(choice)
        status = f"Selected {choice} | {self._describe_device_choice(choice)}"
        self.hardware_status_var.set(status)
        self.header_device_var.set(status)

    def _optimize_for_gpu(self) -> None:
        if self.is_training or self.is_latent_diffusion_training or self.realtime_dreamify_active:
            messagebox.showwarning("Hardware", "Stop training and Real-Time Dreamify before optimizing settings.")
            return
        if getattr(self.device, "type", "") == "cuda" and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(torch.cuda.current_device())
            vram_gb = props.total_memory / (1024 ** 3)
            self.precision_mode_var.set("Auto Recommended")
            self._resolve_precision_settings()
            self.nan_guard_var.set(True)
            self.training_intensity_var.set("75% Fast")
            if vram_gb >= 11:
                self.resolution_var.set(384)
                self.batch_size_var.set(32)
                self.loader_workers_var.set(6)
                self.prefetch_batches_var.set(4)
                self.live_resolution_var.set("256")
                self.diffusion_steps_var.set(10)
                self.diffusion_strength_var.set(0.85)
                self.memory_training_limit_var.set(max(int(self.memory_training_limit_var.get()), 64))
                note = "RTX 3060 / 12GB-style preset applied: 384px, batch 32, Auto precision, stronger prefetch."
            elif vram_gb >= 7:
                self.resolution_var.set(320)
                self.batch_size_var.set(24)
                self.loader_workers_var.set(4)
                self.prefetch_batches_var.set(3)
                self.live_resolution_var.set("192")
                note = "Mid-VRAM CUDA preset applied."
            else:
                self.resolution_var.set(256)
                self.batch_size_var.set(16)
                self.loader_workers_var.set(3)
                self.prefetch_batches_var.set(2)
                self.live_resolution_var.set("192")
                note = "Low-VRAM CUDA preset applied."
            self.status_var.set(note)
            messagebox.showinfo("Optimize For My GPU", note)
            return
        self.precision_mode_var.set("FP32 Stable")
        self._resolve_precision_settings()
        self.nan_guard_var.set(True)
        self.resolution_var.set(256)
        self.batch_size_var.set(8)
        self.loader_workers_var.set(2)
        self.prefetch_batches_var.set(1)
        self.status_var.set("CPU-safe preset applied: 256px, batch 8, FP32 Stable.")

    def _change_device(self, *_args) -> None:
        if self.is_training or self.is_latent_diffusion_training or self.realtime_dreamify_active:
            messagebox.showwarning("Hardware", "Stop training and Real-Time Dreamify before switching hardware.")
            self._refresh_hardware_status()
            return
        choice = self.device_choice_var.get()
        if choice not in self._available_device_choices():
            messagebox.showerror("Hardware", f"{choice} is not available on this computer.")
            self._refresh_hardware_status()
            return
        new_device = self._device_from_choice(choice)
        try:
            with self.model_lock:
                if self.model is not None:
                    self.model = self.model.to(new_device)
                if self.latent_diffusion is not None:
                    self.latent_diffusion = self.latent_diffusion.to(new_device)
                if getattr(self.device, "type", "") == "cuda" and new_device.type != "cuda":
                    torch.cuda.empty_cache()
                self.device = new_device
            self._refresh_hardware_status()
            if hasattr(self, "status_var"):
                self.status_var.set(f"Hardware switched to {choice}.")
        except Exception as exc:
            messagebox.showerror("Hardware", str(exc))
            self._refresh_hardware_status()

    def _open_manual(self) -> None:
        w = tk.Toplevel(self.root)
        w.title("APVD Manual")
        w.geometry("760x680")
        w.minsize(620, 480)
        outer = ttk.Frame(w, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        text = tk.Text(outer, wrap=tk.WORD, bg=self._theme_palette["list_bg"], fg=self._theme_palette["text"], insertbackground=self._theme_palette["text"], relief=tk.FLAT, padx=12, pady=12, height=24)
        text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        text.configure(yscrollcommand=scroll.set)
        text.insert(tk.END, self._manual_text())
        text.configure(state=tk.DISABLED)

    def _manual_text(self) -> str:
        return """APVD Quick Manual

========================
 RECONSTRUCTION MODE
========================

RGB VAE
- Classic APVD mode.
- Trains directly on RGB pixels.
- Best compatibility with old APVD models.

Wavelet
- Experimental mode.
- Trains on a multi-frequency image representation.
- May preserve structure and edges better.
- May train slightly slower and creates models that are not compatible with RGB VAE checkpoints.


========================
 PRECISION MODE
========================

Auto Recommended
- Default consumer-friendly mode.
- Chooses a stable precision for the selected hardware.

FP32 Stable
- Safest math mode.
- Best for laptop GPUs, older CUDA cards, CPU training, unstable drivers, and NaN/crash problems.
- Slower, but reliable.

FP16 Fast
- Fast CUDA mixed precision.
- Best for GPUs that handle FP16 well.
- If training becomes unstable, switch back to FP32 Stable.

BF16 Balanced
- Uses bfloat16 autocast on supported CUDA GPUs.
- Often more stable than FP16 while still faster than full FP32.
- Falls back to FP32 Stable if unsupported.


========================
 LOSS MODE
========================

Classic VAE
- Original APVD training behavior.
- Best for normal dream/generalized reconstructions.

Structure Stable
- Adds gentle L1 shape loss.
- Helps cars, characters, and objects keep their silhouette.

Sharp Detail
- Adds L1 plus Sobel edge loss.
- Helps outlines, wheels, windows, and borders stay clearer.

Experimental Structure
- Stronger structure/detail mix.
- Can improve shape consistency, but may reduce some dreaminess or need tuning.



========================
 APVD MODEL GUIDE
========================

[SOURCES & MODELS]

Select Images / Folder
- Loads image datasets for training.

Batch Folder
- Trains separate models from folders inside a parent directory.

Select Video(s)
- Extracts frames from videos for training.

Select Archive(s)
- Loads images from ZIP/TAR archives.

Clear Sources
- Removes selected datasets.

Save / Load Model
- Saves or opens APVD checkpoints.

PyTorch Map
- Displays checkpoint and training information.

Merge Models
- Blends compatible APVD models together.

Prompt Generation
- Loads models using prompt matching.

Compose Scene
- Builds generated scenes from text prompts.

Output Display
- Opens a clean image preview window.


[TRAINING]

Train APVD
- Starts model training.

Pause / Resume Training
- Pauses or continues training.

Stop Training
- Ends the current training session.

Epochs
- Number of full dataset passes.

Resolution
- Training image size.

Batch Size
- Images processed together each step.

Learning Rate
- Training adjustment strength.

Batch Loads
- Worker processes for loading images.

Prefetch Batches
- Prepared batches kept ahead for smoother GPU usage.

Dataset Cache
- Approximate number of decoded/resized training images to keep warm in RAM.
- The cache budget is split across loader workers so high worker counts do not multiply RAM use unexpectedly.
- Higher values reduce repeated image decoding on later epochs but use more RAM.

Training Intensity
- Chooses how aggressively training uses CPU data loading and GPU time.

Memory Images
- Saved memories mixed into training.

Video Stride
- Frame spacing during video extraction.

Max Frames
- Maximum extracted video frames.
- 0 = unlimited.


[GENERATION]

Generate Unique
- Creates new APVD generations.

Reconstruction Video
- Exports reconstruction behavior as video.

Chaos Mode
- Increases randomness and variation.

Model Shuffle
- Cycles through multiple models.

Auto-Cycle
- Continuously generates images.

Dream Cycle
- Morphs between latent memories.

Dreamify Image/Video
- Reconstructs an uploaded image or video through the loaded APVD model memory.

Real-Time Dreamify
- Captures the screen live and reconstructs it through the loaded APVD model memory.

Dream Continuation Video
- Generates a short APVD dream video by smoothly walking through the loaded model's latent space.

Autoregressive Feedback
- Feeds each generated dream frame back through APVD's encoder so the next frame continues from what the model actually drew.

Feedback Strength
- Controls how strongly generated frames steer the next dream frame. Higher values make it more self-feeding but can drift or melt faster.

Blend Trained Images
- Mixes learned image anchors.

Variation Intensity
- Controls distance from learned examples.

Morph Smoothness
- Controls dream transition smoothness.

Output Image Count
- Number of generated images.

Cleanup Iterations
- Reconstructs outputs repeatedly for cleanup.


[DIFFUSION]

Use Mini Diffusion
- Enables lightweight latent denoising.

Diffusion Steps
- Number of denoise passes.

Denoise Strength
- Diffusion influence strength.

Use Latent DDPM
- Uses larger DDPM diffusion model.

Train DDPM
- Trains latent diffusion helper.

APVD Recon
- APVD-focused reconstruction mode.

DDPM Polish
- Uses DDPM mainly for refinement.

Pure DDPM
- Generates primarily through DDPM.


[PROMPT & PERSONALITY]

Prompt Tag
- Adds short concepts to memories.

Personality
- Loads preset generation styles.

Save / Load Seed
- Stores or restores latent seeds.

Latent Map
- Visualizes remembered latent space.

Memory Retrain
- Trains using saved memory images.


[EVOLUTION & MEMORY]

Evolution Round
- Generates selectable candidates.

Breed Favorites
- Combines selected outputs together.

Dream FPS
- Playback speed for Dream Cycle.

Recall Memory
- Loads saved memories.

Use As Prompt Tag
- Converts memory summaries into prompts.

Recent Bias
- Favors newer or older memories.


[HARDWARE]

CPU
- Universal but slower.

CUDA
- NVIDIA GPU acceleration.

MPS
- Apple Metal acceleration.

Hardware Status
- Displays detected acceleration device.
"""

    def _make_section(self, parent: tk.Widget, title: str, *, open_by_default: bool = True) -> ttk.Frame:
        container = ttk.LabelFrame(parent, padding=(10, 8))
        pack_options = {"fill": tk.X, "padx": 12, "pady": 6}
        container.pack(**pack_options)
        self._section_containers[title] = container
        self._section_pack_options[title] = pack_options
        self._section_order.append(title)
        header = ttk.Frame(container, style="Surface.TFrame")
        header.pack(fill=tk.X)
        body = ttk.Frame(container, style="Surface.TFrame")
        state = tk.BooleanVar(value=open_by_default)
        self._section_states[title] = state

        def toggle() -> None:
            if state.get():
                body.forget()
                state.set(False)
                button.configure(text="+")
            else:
                body.pack(fill=tk.X, pady=(8, 0))
                state.set(True)
                button.configure(text="-")
            self._queue_settings_save()

        button = ttk.Button(header, text="-" if open_by_default else "+", width=3, command=toggle)
        button.pack(side=tk.LEFT)
        ttk.Label(header, text=title, style="Surface.TLabel", font=("", 10, "bold")).pack(side=tk.LEFT, padx=(8, 0))
        self._section_buttons[title] = button
        if open_by_default:
            body.pack(fill=tk.X, pady=(8, 0))
        return body

    @staticmethod
    def _grid_row(parent: tk.Widget, row: int, label: str, widget: tk.Widget, *, column: int = 0, pad_y: int = 4) -> None:
        ttk.Label(parent, text=label, style="Surface.TLabel").grid(row=row, column=column, sticky=tk.W, padx=(0, 8), pady=pad_y)
        widget.grid(row=row, column=column + 1, sticky=tk.W, padx=(0, 18), pady=pad_y)

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        scroll_canvas = tk.Canvas(main, bg=self._theme_palette["bg"], highlightthickness=0)
        self._register_theme_widget(scroll_canvas, "scroll_canvas")
        scrollbar = ttk.Scrollbar(main, orient=tk.VERTICAL, command=scroll_canvas.yview)
        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=1, sticky=tk.NS)
        scroll_canvas.grid(row=0, column=0, sticky=tk.NSEW)

        content = ttk.Frame(scroll_canvas)
        content_id = scroll_canvas.create_window((0, 0), window=content, anchor=tk.NW)

        def _resize_content(event):
            scroll_canvas.itemconfigure(content_id, width=event.width)

        def _sync_scroll_region(_event=None):
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        scroll_canvas.bind("<Configure>", _resize_content)
        content.bind("<Configure>", _sync_scroll_region)
        scroll_canvas.bind_all("<MouseWheel>", lambda event: scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units"))

        header = ttk.Frame(content, padding=(12, 12, 12, 4))
        header.pack(fill=tk.X)
        ttk.Label(header, text="APVD Model Studio", font=("", 16, "bold")).pack(side=tk.LEFT)
        ttk.Label(header, textvariable=self.header_device_var, style="Muted.TLabel").pack(side=tk.LEFT, padx=(14, 0))
        ttk.Button(header, text="Manual", command=self._open_manual).pack(side=tk.RIGHT, padx=(12, 0))
        ttk.Label(header, text="Theme:").pack(side=tk.RIGHT, padx=(12, 6))
        theme_box = ttk.Combobox(header, textvariable=self.theme_mode_var, values=("Auto", "Day", "Night"), state="readonly", width=8)
        theme_box.pack(side=tk.RIGHT)
        theme_box.bind("<<ComboboxSelected>>", self._set_theme)
        ttk.Label(header, text="User Level:").pack(side=tk.RIGHT, padx=(12, 6))
        level_box = ttk.Combobox(header, textvariable=self.user_level_var, values=USER_LEVEL_CHOICES, state="readonly", width=9)
        level_box.pack(side=tk.RIGHT)
        level_box.bind("<<ComboboxSelected>>", self._apply_user_level)
        self._refresh_hardware_status()
        ttk.Label(content, textvariable=self.user_level_help_var, style="Muted.TLabel").pack(fill=tk.X, padx=12, pady=(0, 4))

        hardware_frame = self._make_section(content, "Hardware", open_by_default=True)
        self._category_label(hardware_frame, "Compute Device", 0)
        ttk.Label(hardware_frame, text="Run On:", style="Surface.TLabel").grid(row=1, column=0, sticky=tk.W, padx=(0, 8), pady=4)
        hardware_box = ttk.Combobox(hardware_frame, textvariable=self.device_choice_var, values=self._available_device_choices(), state="readonly", width=10)
        hardware_box.grid(row=1, column=1, sticky=tk.W, padx=(0, 12), pady=4)
        hardware_box.bind("<<ComboboxSelected>>", self._change_device)
        ttk.Button(hardware_frame, text="Refresh Hardware", command=self._refresh_hardware_status).grid(row=1, column=2, sticky=tk.W, padx=4, pady=4)
        ttk.Button(hardware_frame, text="Optimize For My GPU", command=self._optimize_for_gpu, style="Accent.TButton").grid(row=1, column=3, sticky=tk.W, padx=4, pady=4)
        ttk.Label(hardware_frame, textvariable=self.hardware_status_var, style="SurfaceMuted.TLabel").grid(row=2, column=0, columnspan=5, sticky=tk.W, pady=(6, 0))

        source_frame = self._make_section(content, "Sources And Model", open_by_default=True)
        self._category_label(source_frame, "Training Sources", 0)
        source_buttons = ttk.Frame(source_frame, style="Surface.TFrame")
        source_buttons.grid(row=1, column=0, sticky=tk.W)
        for idx, (text, command) in enumerate([
            ("Select Images", self._select_images),
            ("Select Folder", self._select_folder),
            ("Batch Folder", self._select_batch_folder),
            ("Select Video(s)", self._select_videos),
            ("Select Archive(s)", self._select_archives),
            ("Clear Sources", self._clear_training_sources),
        ]):
            ttk.Button(source_buttons, text=text, command=command).grid(row=idx // 3, column=idx % 3, padx=4, pady=4, sticky=tk.W)

        model_buttons = ttk.Frame(source_frame, style="Surface.TFrame")
        self._category_label(source_frame, "Model Tools", 2)
        model_buttons.grid(row=3, column=0, sticky=tk.W, pady=(4, 0))
        for idx, (text, command) in enumerate([
            ("Save Model", self._save_model),
            ("Load Model", self._load_model),
            ("PyTorch Map", self._open_model_map),
            ("Merge Models", self._merge_models),
            ("Prompt Generation", self._auto_load_model),
            ("Compose Scene", self._compose_scene_prompt),
            ("Output Display", self._create_output_window),
        ]):
            ttk.Button(model_buttons, text=text, command=command).grid(row=idx // 4, column=idx % 4, padx=4, pady=4, sticky=tk.W)

        training_frame = self._make_section(content, "Training Settings", open_by_default=True)
        self._category_label(training_frame, "Run Controls", 0)
        self.train_btn = ttk.Button(training_frame, text="Train APVD", command=self._train, style="Accent.TButton")
        self.train_btn.grid(row=1, column=0, padx=(0, 6), pady=(0, 10), sticky=tk.W)
        self.pause_btn = ttk.Button(training_frame, text="Pause Training", command=self._toggle_training_pause, state=tk.DISABLED)
        self.pause_btn.grid(row=1, column=1, padx=6, pady=(0, 10), sticky=tk.W)
        self.stop_btn = ttk.Button(training_frame, text="Stop Training", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.grid(row=1, column=2, padx=6, pady=(0, 10), sticky=tk.W)

        self._category_label(training_frame, "Reconstruction Mode", 2)
        rec_mode_box = ttk.Combobox(training_frame, textvariable=self.reconstruction_mode_var, values=("RGB VAE", "Wavelet"), state="readonly", width=12)
        rec_mode_box.grid(row=3, column=0, sticky=tk.W, padx=(0, 12), pady=4)
        rec_mode_box.bind("<<ComboboxSelected>>", lambda e: self.status_var.set(f"Reconstruction mode set to {self._get_reconstruction_mode()}."))
        ttk.Label(training_frame, text="RGB VAE: classic APVD pixel reconstruction.\nWavelet: experimental multi-frequency reconstruction for better structure/detail.", style="SurfaceMuted.TLabel").grid(row=3, column=1, columnspan=3, sticky=tk.W, pady=4)

        self._category_label(training_frame, "Model Learning", 4)
        self.epoch_spin = ttk.Spinbox(training_frame, from_=1, to=50000, increment=1, width=8, textvariable=self.epochs_var)
        self._grid_row(training_frame, 5, "Epochs:", self.epoch_spin)
        self.resolution_spin = ttk.Spinbox(training_frame, from_=32, to=1024, increment=16, width=8, textvariable=self.resolution_var)
        self._grid_row(training_frame, 5, "Resolution:", self.resolution_spin, column=2)
        self.batch_spin = ttk.Spinbox(training_frame, from_=1, to=256, increment=1, width=8, textvariable=self.batch_size_var)
        self._grid_row(training_frame, 6, "Batch Size:", self.batch_spin)
        self.lr_spin = ttk.Spinbox(training_frame, from_=0.000001, to=0.1, increment=0.00001, width=10, textvariable=self.learning_rate_var, format="%.6f")
        self._grid_row(training_frame, 6, "Learning Rate:", self.lr_spin, column=2)
        precision_box = ttk.Combobox(training_frame, textvariable=self.precision_mode_var, values=PRECISION_MODE_CHOICES, state="readonly", width=20)
        self._grid_row(training_frame, 7, "Precision Mode:", precision_box)
        precision_box.bind("<<ComboboxSelected>>", self._on_precision_mode_changed)
        ttk.Label(training_frame, text="FP32 = safest. FP16 = fastest when stable. BF16 = supported-GPU balance.", style="SurfaceMuted.TLabel").grid(row=7, column=2, columnspan=3, sticky=tk.W, pady=(8, 0))
        loss_mode_box = ttk.Combobox(training_frame, textvariable=self.loss_mode_var, values=LOSS_MODE_CHOICES, state="readonly", width=22)
        self._grid_row(training_frame, 8, "Loss Mode:", loss_mode_box)
        loss_mode_box.bind("<<ComboboxSelected>>", self._on_loss_mode_changed)
        ttk.Label(training_frame, textvariable=self.loss_mode_help_var, style="SurfaceMuted.TLabel").grid(row=8, column=2, columnspan=3, sticky=tk.W, pady=4)
        ttk.Checkbutton(training_frame, text="NaN Guard / Auto-Recover", variable=self.nan_guard_var).grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
        self._category_label(training_frame, "Data Loading", 10)
        self.loader_spin = ttk.Spinbox(training_frame, from_=0, to=32, increment=1, width=8, textvariable=self.loader_workers_var)
        self._grid_row(training_frame, 11, "Batch Loads:", self.loader_spin)
        self.prefetch_spin = ttk.Spinbox(training_frame, from_=1, to=512, increment=1, width=8, textvariable=self.prefetch_batches_var)
        self._grid_row(training_frame, 11, "Prefetch Batches:", self.prefetch_spin, column=2)
        cache_spin = ttk.Spinbox(training_frame, from_=0, to=200000, increment=512, width=8, textvariable=self.dataset_cache_items_var)
        self._grid_row(training_frame, 12, "Dataset Cache:", cache_spin)
        ttk.Label(training_frame, text="Approx. total decoded/resized images kept warm in RAM; split across workers.", style="SurfaceMuted.TLabel").grid(row=12, column=2, columnspan=3, sticky=tk.W, pady=4)
        self.intensity_box = ttk.Combobox(training_frame, textvariable=self.training_intensity_var, values=TRAINING_INTENSITY_CHOICES, state="readonly", width=18)
        self.intensity_box.bind("<<ComboboxSelected>>", self._apply_training_intensity)
        self._grid_row(training_frame, 13, "Training Intensity:", self.intensity_box)
        ttk.Checkbutton(training_frame, text="Blend memory into training", variable=self.include_memory_training_var).grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=(8, 0))
        memory_limit_spin = ttk.Spinbox(training_frame, from_=0, to=10000, increment=8, width=8, textvariable=self.memory_training_limit_var)
        self._grid_row(training_frame, 14, "Memory Images:", memory_limit_spin, column=2, pad_y=8)
        self._category_label(training_frame, "Video Sampling", 15)
        video_stride_spin = ttk.Spinbox(training_frame, from_=1, to=10000, increment=1, width=8, textvariable=self.video_stride_var)
        self._grid_row(training_frame, 16, "Video Stride:", video_stride_spin)
        max_frames_spin = ttk.Spinbox(training_frame, from_=0, to=1_000_000, increment=100, width=8, textvariable=self.video_max_frames_var)
        self._grid_row(training_frame, 16, "Max Frames:", max_frames_spin, column=2)

        gen_tools_frame = self._make_section(content, "Generation Tools", open_by_default=True)
        self._category_label(gen_tools_frame, "Generate And Cycle", 0)
        for idx, (text, command) in enumerate([
            ("Generate Unique", self._generate),
            ("Dreamify Image/Video", self._dreamify_media),
            ("Audio Memory", self._open_audio_memory_settings),
            ("Real-Time Dreamify", self._toggle_realtime_dreamify),
            ("Dream Continuation Video", self._generate_dream_continuation_video),
            ("Reconstruction Video", self._export_reconstruction_video),
            ("Chaos Mode", self._toggle_chaos),
            ("Model Shuffle", self._toggle_model_cycle),
        ]):
            ttk.Button(gen_tools_frame, text=text, command=command).grid(row=1, column=idx, padx=4, pady=4, sticky=tk.W)
        ttk.Checkbutton(gen_tools_frame, text="Auto-Cycle", variable=self.auto_cycle_var, command=self._toggle_auto_cycle).grid(row=2, column=0, padx=4, pady=8, sticky=tk.W)
        ttk.Checkbutton(gen_tools_frame, text="Dream Cycle (Morph)", variable=self.dream_cycle_var, command=self._toggle_dream_cycle).grid(row=2, column=1, padx=4, pady=8, sticky=tk.W)
        ttk.Checkbutton(gen_tools_frame, text="Blend Trained Images", variable=self.blend_mode_var).grid(row=2, column=2, padx=4, pady=8, sticky=tk.W)

        generation_settings = self._make_section(content, "Generation Settings", open_by_default=False)
        self._category_label(generation_settings, "Image Variation", 0)
        self.var_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=20.0, resolution=0.5, orient=tk.HORIZONTAL, length=280), "scale")
        self.var_scale.set(1.0)
        self._grid_row(generation_settings, 1, "Variation Intensity:", self.var_scale)
        self.speed_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=5, to=150, resolution=1, orient=tk.HORIZONTAL, length=280), "scale")
        self.speed_scale.set(40)
        self._grid_row(generation_settings, 2, "Morph Smoothness:", self.speed_scale)
        self.blend_spin = ttk.Spinbox(generation_settings, from_=2, to=16, increment=1, width=6, textvariable=self.blend_count_var)
        self._grid_row(generation_settings, 3, "Blend Image Count:", self.blend_spin)
        self.output_spin = ttk.Spinbox(generation_settings, from_=1, to=8, increment=1, width=6, textvariable=self.output_count_var)
        self._grid_row(generation_settings, 4, "Output Image Count:", self.output_spin)
        self._category_label(generation_settings, "Cleanup And Denoising", 5)
        self.iter_spin = ttk.Spinbox(generation_settings, from_=0, to=25, increment=1, width=6, textvariable=self.iterations_var)
        self._grid_row(generation_settings, 6, "Cleanup Iterations:", self.iter_spin)
        ttk.Checkbutton(generation_settings, text="Show each iteration", variable=self.show_iterations_var).grid(row=6, column=2, sticky=tk.W, padx=(0, 18), pady=4)
        ttk.Checkbutton(generation_settings, text="Use Mini Diffusion", variable=self.use_mini_diffusion_var).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=(10, 4))
        diffusion_steps_spin = ttk.Spinbox(generation_settings, from_=1, to=50, increment=1, width=6, textvariable=self.diffusion_steps_var)
        self._grid_row(generation_settings, 8, "Diffusion Steps:", diffusion_steps_spin)
        diffusion_strength_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.1, to=1.5, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.diffusion_strength_var), "scale")
        self._grid_row(generation_settings, 9, "Denoise Strength:", diffusion_strength_scale)
        self._category_label(generation_settings, "Dreamify", 10)
        dream_strength_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.dream_strength_var), "scale")
        self._grid_row(generation_settings, 11, "Dream Strength:", dream_strength_scale)
        memory_pull_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.memory_pull_var), "scale")
        self._grid_row(generation_settings, 12, "Memory Pull:", memory_pull_scale)
        dreamify_skip_spin = ttk.Spinbox(generation_settings, from_=1, to=30, increment=1, width=6, textvariable=self.dreamify_frame_skip_var)
        self._grid_row(generation_settings, 13, "Dreamify Frame Skip:", dreamify_skip_spin)
        dreamify_audio_box = ttk.Combobox(generation_settings, textvariable=self.dreamify_audio_mode_var, values=AUDIO_MEMORY_MODE_CHOICES, state="readonly", width=24)
        self._grid_row(generation_settings, 13, "Dreamify Audio:", dreamify_audio_box, column=2)
        ttk.Checkbutton(generation_settings, text="Keep Original Audio (legacy)", variable=self.keep_original_audio_var).grid(row=14, column=2, sticky=tk.W, padx=(0, 18), pady=4)
        ttk.Button(generation_settings, text="Audio Memory Settings", command=self._open_audio_memory_settings).grid(row=14, column=0, columnspan=2, sticky=tk.W, pady=4)
        self._category_label(generation_settings, "Real-Time Dreamify", 14)
        live_resolution_box = ttk.Combobox(generation_settings, textvariable=self.live_resolution_var, values=("128", "192", "256"), state="readonly", width=6)
        self._grid_row(generation_settings, 15, "Live Resolution:", live_resolution_box)
        live_fps_spin = ttk.Spinbox(generation_settings, from_=1, to=30, increment=1, width=6, textvariable=self.live_target_fps_var)
        self._grid_row(generation_settings, 15, "Target FPS:", live_fps_spin, column=2)
        live_dream_strength_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.live_dream_strength_var), "scale")
        self._grid_row(generation_settings, 16, "Live Dream Strength:", live_dream_strength_scale)
        live_memory_pull_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.live_memory_pull_var), "scale")
        self._grid_row(generation_settings, 17, "Live Memory Pull:", live_memory_pull_scale)
        live_capture_box = ttk.Combobox(generation_settings, textvariable=self.live_capture_mode_var, values=("Full Screen", "Region"), state="readonly", width=12)
        self._grid_row(generation_settings, 18, "Capture Mode:", live_capture_box)
        ttk.Checkbutton(generation_settings, text="Show Live FPS", variable=self.live_show_fps_var).grid(row=18, column=2, sticky=tk.W, padx=(0, 18), pady=4)
        live_display_box = ttk.Combobox(generation_settings, textvariable=self.live_display_mode_var, values=("Output Window", "Challenge Window", "Experimental Fullscreen Overlay"), state="readonly", width=18)
        self._grid_row(generation_settings, 19, "Live Display:", live_display_box)
        live_overlay_opacity_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.20, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.live_overlay_opacity_var), "scale")
        self._grid_row(generation_settings, 20, "Overlay Opacity:", live_overlay_opacity_scale)
        ttk.Checkbutton(generation_settings, text="Click-Through Overlay", variable=self.live_clickthrough_overlay_var).grid(row=20, column=2, sticky=tk.W, padx=(0, 18), pady=4)
        ttk.Checkbutton(generation_settings, text="Mini Diffusion Live", variable=self.live_mini_diffusion_var).grid(row=21, column=0, columnspan=2, sticky=tk.W, pady=(6, 4))
        self._category_label(generation_settings, "Dream Continuation Video", 22)
        dream_video_seconds_spin = ttk.Spinbox(generation_settings, from_=1, to=180, increment=1, width=6, textvariable=self.dream_video_seconds_var)
        self._grid_row(generation_settings, 23, "Dream Video Seconds:", dream_video_seconds_spin)
        dream_video_fps_spin = ttk.Spinbox(generation_settings, from_=4, to=30, increment=1, width=6, textvariable=self.dream_video_fps_var)
        self._grid_row(generation_settings, 23, "Dream Video FPS:", dream_video_fps_spin, column=2)
        dream_audio_box = ttk.Combobox(generation_settings, textvariable=self.dream_video_audio_mode_var, values=("Silent", "Use Source Video Audio", "Procedural Dream Audio", "Audio Memory Reconstruction", "Dreamy Memory Audio", "Corrupted Memory Audio"), state="readonly", width=24)
        self._grid_row(generation_settings, 24, "Dream Audio:", dream_audio_box)
        ttk.Button(generation_settings, text="Select Audio/Video", command=self._select_dream_audio_source).grid(row=24, column=2, sticky=tk.W, pady=4)
        ttk.Button(generation_settings, text="Select Structure Video", command=self._select_dream_structure_video).grid(row=25, column=0, columnspan=2, sticky=tk.W, pady=4)
        structure_guidance_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.dream_structure_guidance_var), "scale")
        self._grid_row(generation_settings, 25, "Structure Guidance:", structure_guidance_scale, column=2)
        latent_drift_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.latent_drift_var), "scale")
        self._grid_row(generation_settings, 26, "Latent Drift:", latent_drift_scale)
        motion_smoothness_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=0.99, resolution=0.01, orient=tk.HORIZONTAL, length=220, variable=self.motion_smoothness_var), "scale")
        self._grid_row(generation_settings, 27, "Motion Smoothness:", motion_smoothness_scale)
        dream_instability_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=1.5, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.dream_instability_var), "scale")
        self._grid_row(generation_settings, 28, "Dream Instability:", dream_instability_scale)
        ttk.Checkbutton(generation_settings, text="Autoregressive Feedback", variable=self.dream_autoregressive_var).grid(row=29, column=0, columnspan=2, sticky=tk.W, pady=(6, 4))
        feedback_strength_scale = self._register_theme_widget(tk.Scale(generation_settings, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.dream_feedback_strength_var), "scale")
        self._grid_row(generation_settings, 30, "Feedback Strength:", feedback_strength_scale)

        latent_diffusion_frame = self._make_section(content, "Latent DDPM Diffusion", open_by_default=False)
        self._category_label(latent_diffusion_frame, "DDPM Generation", 0)
        ttk.Checkbutton(latent_diffusion_frame, text="Use Latent DDPM on Generate", variable=self.use_latent_diffusion_var).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=4)
        latent_steps_spin = ttk.Spinbox(latent_diffusion_frame, from_=50, to=100, increment=5, width=6, textvariable=self.latent_diffusion_timesteps_var)
        self._grid_row(latent_diffusion_frame, 2, "Timesteps:", latent_steps_spin)
        latent_strength_scale = self._register_theme_widget(tk.Scale(latent_diffusion_frame, from_=0.05, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=220, variable=self.latent_diffusion_strength_var), "scale")
        self._grid_row(latent_diffusion_frame, 3, "Polish Strength:", latent_strength_scale)
        self._category_label(latent_diffusion_frame, "DDPM Tools", 4)
        self.latent_diffusion_train_btn = ttk.Button(latent_diffusion_frame, text="Train DDPM", command=self._train_latent_diffusion)
        for idx, button in enumerate([
            self.latent_diffusion_train_btn,
            ttk.Button(latent_diffusion_frame, text="Save DDPM", command=self._save_latent_diffusion),
            ttk.Button(latent_diffusion_frame, text="Load DDPM", command=self._load_latent_diffusion),
            ttk.Button(latent_diffusion_frame, text="APVD Recon", command=self._generate_apvd_reconstruction),
            ttk.Button(latent_diffusion_frame, text="DDPM Polish", command=self._generate_latent_diffusion_polish),
            ttk.Button(latent_diffusion_frame, text="Pure DDPM", command=self._generate_pure_latent_diffusion),
        ]):
            button.grid(row=5 + (idx // 3), column=idx % 3, padx=4, pady=(10 if idx < 3 else 4, 4), sticky=tk.W)

        prompt_frame = self._make_section(content, "Prompt And Personality", open_by_default=True)
        self._category_label(prompt_frame, "Prompt Controls", 0)
        ttk.Label(prompt_frame, text="Prompt tag:", style="Surface.TLabel").grid(row=1, column=0, sticky=tk.W)
        ttk.Entry(prompt_frame, textvariable=self.generation_prompt_var, width=42).grid(row=1, column=1, sticky=tk.EW, padx=(8, 12))
        ttk.Label(prompt_frame, text="Personality:", style="Surface.TLabel").grid(row=1, column=2, sticky=tk.W)
        ttk.Combobox(prompt_frame, textvariable=self.personality_var, values=list(PERSONALITY_PRESETS.keys()), state="readonly", width=14).grid(row=1, column=3, sticky=tk.W, padx=(8, 12))
        ttk.Button(prompt_frame, text="Apply Preset", command=self._apply_personality_preset).grid(row=1, column=4, sticky=tk.W)
        self._category_label(prompt_frame, "Seed And Memory Tools", 2)
        for idx, (text, command) in enumerate([
            ("Save Seed", self._save_current_seed),
            ("Load Seed", self._load_seed),
            ("Latent Map", self._open_latent_map),
            ("Memory Retrain", self._evolve_from_memory),
        ]):
            ttk.Button(prompt_frame, text=text, command=command).grid(row=3, column=idx, sticky=tk.W, pady=(10, 0), padx=(0 if idx == 0 else 8, 0))
        prompt_frame.columnconfigure(1, weight=1)

        evolution_frame = self._make_section(content, "Evolution And Memory", open_by_default=False)
        self._category_label(evolution_frame, "Interactive Evolution", 0)
        candidates_spin = ttk.Spinbox(evolution_frame, from_=4, to=8, increment=1, width=6, textvariable=self.evolution_count_var)
        self._grid_row(evolution_frame, 1, "Candidates:", candidates_spin)
        ttk.Label(evolution_frame, text="Favorites:", style="Surface.TLabel").grid(row=1, column=2, sticky=tk.W)
        ttk.Entry(evolution_frame, textvariable=self.evolution_selection_var, width=20).grid(row=1, column=3, sticky=tk.W, padx=(8, 12))
        ttk.Label(evolution_frame, text="Use 1,3,4 format", style="SurfaceMuted.TLabel").grid(row=1, column=4, sticky=tk.W)
        ttk.Button(evolution_frame, text="Evolution Round", command=self._generate_evolution_round).grid(row=2, column=0, sticky=tk.W, pady=(10, 0))
        ttk.Button(evolution_frame, text="Breed Favorites", command=self._breed_evolution_favorites).grid(row=2, column=1, sticky=tk.W, pady=(10, 0), padx=(8, 0))
        ttk.Button(evolution_frame, text="Judge Generate", command=self._judge_generate_candidates).grid(row=2, column=2, sticky=tk.W, pady=(10, 0), padx=(8, 0))
        ttk.Button(evolution_frame, text="Memory Finder", command=self._memory_finder_search).grid(row=2, column=3, sticky=tk.W, pady=(10, 0), padx=(8, 0))
        dream_fps_spin = ttk.Spinbox(evolution_frame, from_=1, to=60, increment=1, width=6, textvariable=self.dream_fps_var)
        self._grid_row(evolution_frame, 1, "Dream FPS:", dream_fps_spin, column=5, pad_y=4)
        self._category_label(evolution_frame, "Memory Stream", 3)
        self.memory_listbox = self._register_theme_widget(tk.Listbox(evolution_frame, height=6, exportselection=False), "listbox")
        self.memory_listbox.grid(row=4, column=0, columnspan=5, sticky=tk.EW, pady=(8, 0))
        ttk.Button(evolution_frame, text="Recall Memory", command=self._recall_selected_memory).grid(row=5, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Button(evolution_frame, text="Use As Prompt Tag", command=self._use_memory_prompt).grid(row=5, column=1, sticky=tk.W, pady=(8, 0))
        recent_bias_scale = self._register_theme_widget(tk.Scale(evolution_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=160, variable=self.memory_recent_weight_var), "scale")
        self._grid_row(evolution_frame, 5, "Recent Bias:", recent_bias_scale, column=2, pad_y=8)
        evolution_frame.columnconfigure(0, weight=1)

        status_frame = ttk.Frame(content, padding=(12, 4, 12, 8))
        status_frame.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel").pack(fill=tk.X)
        self.load_progress_var = tk.DoubleVar(value=0.0)
        self.load_progress = ttk.Progressbar(status_frame, maximum=100.0, variable=self.load_progress_var, mode="determinate")
        self.load_progress.pack(fill=tk.X, pady=(6, 0))

        preview_frame = self._make_section(content, "Preview", open_by_default=True)
        preview_tools = ttk.Frame(preview_frame, style="Surface.TFrame")
        preview_tools.pack(fill=tk.X, padx=10, pady=(10, 0))
        self.training_preview_btn = ttk.Button(preview_tools, text="Display Image Progress", command=self._toggle_training_progress_preview)
        self.training_preview_btn.pack(side=tk.LEFT, padx=(0, 8))
        self.timelapse_btn = ttk.Button(preview_tools, text="Timelapse", command=self._toggle_timelapse_recording)
        self.timelapse_btn.pack(side=tk.LEFT, padx=(0, 12))
        ttk.Label(preview_tools, text="Training Video:", style="Surface.TLabel").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Radiobutton(preview_tools, text="Horizontal", value="Horizontal", variable=self.training_video_layout_var).pack(side=tk.LEFT)
        ttk.Radiobutton(preview_tools, text="Vertical 9:16", value="Vertical", variable=self.training_video_layout_var).pack(side=tk.LEFT, padx=(4, 0))
        self.canvas_frame = ttk.Frame(preview_frame, padding=10, style="Surface.TFrame")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = self._register_theme_widget(tk.Canvas(self.canvas_frame, width=512, height=512, bg=self._theme_palette["canvas"], highlightthickness=1, highlightbackground=self._theme_palette["canvas_border"]), "canvas")
        self.canvas.pack()
        self._apply_registered_theme()
        self._restore_post_ui_settings()

    IMAGE_FILE_TYPES = [("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All files", "*.*")]
    FILE_TYPES = IMAGE_FILE_TYPES
    VIDEO_FILE_TYPES = [("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"), ("All files", "*.*")]
    AUDIO_FILE_TYPES = [("Audio files", "*.mp3 *.wav *.m4a *.aac *.ogg *.flac"), ("All files", "*.*")]
    MODEL_FILE_TYPES = [("PyTorch model", "*.pt"), ("PyTorch checkpoint", "*.pth"), ("All files", "*.*")]
    ARCHIVE_FILE_TYPES = [("Archives", "*.zip *.tar *.tar.gz *.tgz"), ("ZIP", "*.zip"), ("TAR / compressed", "*.tar *.tar.gz *.tgz"), ("All files", "*.*")]

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60: return f"{seconds:.1f}s"
        minutes = seconds / 60
        if minutes < 60: return f"{minutes:.1f}m"
        hours = minutes / 60
        return f"{hours:.1f}h"

    def _toggle_training_progress_preview(self):
        enabled = not self.training_preview_enabled_var.get()
        self.training_preview_enabled_var.set(enabled)
        label = "Progress: On" if enabled else "Display Image Progress"
        self.training_preview_btn.config(text=label)
        self.status_var.set("Training image progress preview enabled." if enabled else "Training image progress preview disabled.")

    def _toggle_timelapse_recording(self):
        enabled = not self.timelapse_enabled_var.get()
        self.timelapse_enabled_var.set(enabled)
        self.timelapse_btn.config(text="Timelapse: On" if enabled else "Timelapse")
        if enabled:
            self._start_timelapse_session()
            self.status_var.set("Timelapse recording armed. Frames will be captured during training.")
        else:
            self.status_var.set("Timelapse recording disabled.")
            if self.timelapse_frames_dir is not None:
                threading.Thread(target=self._finish_timelapse_session, daemon=True).start()

    def _start_timelapse_session(self) -> None:
        if self.timelapse_frames_dir is not None:
            return
        self.timelapse_is_encoding = False
        self.timelapse_layout = self._current_training_video_layout()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = TIMELAPSES_DIR / f"APVD_timelapse_{stamp}"
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        self.timelapse_run_dir = run_dir
        self.timelapse_frames_dir = frames_dir
        self.timelapse_frame_count = 0
        self.timelapse_capture_interval = 1
        self.timelapse_last_capture_step = -1
        self.timelapse_start_time = time.perf_counter()
        self.timelapse_video_path = run_dir / "apvd_training_timelapse.mp4"

    def _finish_timelapse_session(self) -> None:
        if self.timelapse_is_encoding:
            return
        frames_dir = self.timelapse_frames_dir
        video_path = self.timelapse_video_path
        frame_count = self.timelapse_frame_count
        if frames_dir is None or video_path is None:
            return
        self.timelapse_is_encoding = True

        def _clear_session():
            self.timelapse_frames_dir = None
            self.timelapse_run_dir = None
            self.timelapse_frame_count = 0
            self.timelapse_last_capture_step = -1
            self.timelapse_capture_interval = 1
            self.timelapse_is_encoding = False
            self.timelapse_enabled_var.set(False)
            self.timelapse_btn.config(text="Timelapse")
            self.timelapse_layout = self._current_training_video_layout()

        if frame_count <= 0:
            self._after(0, _clear_session)
            return

        try:
            self._after(0, lambda: self.status_var.set("Encoding APVD timelapse video..."))
            frame_paths = sorted(frames_dir.glob("*.jpg"))
            first = cv2.imread(str(frame_paths[0])) if frame_paths else None
            if first is None:
                raise RuntimeError("No timelapse frames could be read.")
            height, width = first.shape[:2]
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), TIMELAPSE_FPS, (width, height))
            if not writer.isOpened():
                raise RuntimeError("Could not open MP4 writer for timelapse export.")
            try:
                for frame_path in frame_paths:
                    frame = cv2.imread(str(frame_path))
                    if frame is None: continue
                    if frame.shape[:2] != (height, width):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    writer.write(frame)
            finally:
                writer.release()
            self._after(0, lambda p=video_path, n=len(frame_paths): self.status_var.set(f"Timelapse saved: {p} ({n} frames)."))
        except Exception as exc:
            self._after(0, lambda e=exc: messagebox.showerror("Timelapse", str(e)))
        finally:
            self._after(0, _clear_session)

    @staticmethod
    def _hex_to_rgb(value: str) -> tuple[int, int, int]:
        value = value.lstrip("#")
        if len(value) != 6:
            return (255, 255, 255)
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))

    def _current_training_video_layout(self) -> str:
        return "Vertical" if self.training_video_layout_var.get() == "Vertical" else "Horizontal"

    @staticmethod
    def _draw_training_progress_bar(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], progress: float, *, bg: tuple[int, int, int], border: tuple[int, int, int], accent: tuple[int, int, int], muted: tuple[int, int, int]) -> None:
        x0, y0, x1, y1 = box
        progress = min(100.0, max(0.0, progress))
        draw.rectangle((x0, y0, x1, y1), outline=border, fill=bg)
        fill_w = int((x1 - x0) * (progress / 100.0))
        draw.rectangle((x0, y0, x0 + fill_w, y1), fill=accent)
        draw.text((x0, y1 + 8), f"{progress:.1f}% complete", fill=muted)

    def _training_visual_frame(self, input_tensor: torch.Tensor, recon_tensor: torch.Tensor, metrics: dict, layout: str | None = None) -> Image.Image:
        palette = dict(self._theme_palette)
        bg = self._hex_to_rgb(palette["bg"])
        surface = self._hex_to_rgb(palette["surface"])
        accent = self._hex_to_rgb(palette["accent"])
        text = self._hex_to_rgb(palette["text"])
        muted = self._hex_to_rgb(palette["muted"])
        border = self._hex_to_rgb(palette["border"])

        layout = layout or self._current_training_video_layout()
        canvas_size = (720, 1280) if layout == "Vertical" else (1280, 720)
        canvas = Image.new("RGB", canvas_size, bg)
        draw = ImageDraw.Draw(canvas)

        progress = float(metrics.get("progress", 0.0))

        metric_lines = [
            f"Dataset: {metrics.get('dataset_label', 'Current dataset')}",
            f"Epoch: {metrics.get('epoch', 0)}/{metrics.get('epochs', 0)}",
            f"Batch: {metrics.get('batch', 0)}/{metrics.get('total_batches', 0)}",
            f"Loss: {float(metrics.get('loss', 0.0)):.2f}",
            f"Elapsed: {self._format_duration(float(metrics.get('elapsed', 0.0)))}",
            f"ETA: {self._format_duration(float(metrics.get('eta', 0.0)))}",
            f"Theme: {self._current_theme_label()}",
        ]

        if layout == "Vertical":
            draw.rectangle((0, 0, 720, 116), fill=surface)
            draw.rectangle((0, 114, 720, 116), fill=accent)
            draw.text((34, 22), "APVD Training Progress", fill=text)
            self._draw_training_progress_bar(draw, (34, 66, 686, 82), progress, bg=bg, border=border, accent=accent, muted=muted)

            image_size = 252
            input_img = tensor_to_pil(input_tensor.detach().cpu()).resize((image_size, image_size), Image.Resampling.LANCZOS)
            recon_img = tensor_to_pil(recon_tensor.detach().cpu()).resize((image_size, image_size), Image.Resampling.LANCZOS)
            noise = Image.fromarray(np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8))
            noise_blend = Image.blend(noise, recon_img, min(0.95, max(0.05, progress / 100.0)))
            panels = [("Input batch", input_img), ("Noise to image", noise_blend), ("Reconstruction", recon_img)]
            for idx, (label, image) in enumerate(panels):
                x = 34
                y = 144 + idx * 302
                draw.rectangle((x, y, 686, y + 278), fill=surface, outline=border)
                canvas.paste(image, (x + 18, y + 13))
                draw.text((x + 296, y + 26), label, fill=text)
                start = idx * 2
                for line_idx, line in enumerate(metric_lines[start : start + 2]):
                    draw.text((x + 296, y + 70 + line_idx * 38), line, fill=muted)
            for idx, line in enumerate(metric_lines[6:]):
                draw.text((54 + idx * 300, 1070), line, fill=muted)
        else:
            draw.rectangle((0, 0, 1280, 78), fill=surface)
            draw.rectangle((0, 76, 1280, 78), fill=accent)
            draw.text((34, 22), "APVD Training Progress", fill=text)
            self._draw_training_progress_bar(draw, (890, 30, 1220, 44), progress, bg=bg, border=border, accent=accent, muted=muted)

            input_img = tensor_to_pil(input_tensor.detach().cpu()).resize((300, 300), Image.Resampling.LANCZOS)
            recon_img = tensor_to_pil(recon_tensor.detach().cpu()).resize((300, 300), Image.Resampling.LANCZOS)
            noise = Image.fromarray(np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8))
            noise_blend = Image.blend(noise, recon_img, min(0.95, max(0.05, progress / 100.0)))
            panels = [("Input batch", input_img), ("Noise to image", noise_blend), ("Reconstruction", recon_img)]
            for idx, (label, image) in enumerate(panels):
                x = 48 + idx * 406
                y = 116
                draw.rectangle((x - 10, y - 10, x + 310, y + 346), fill=surface, outline=border)
                canvas.paste(image, (x, y))
                draw.text((x, y + 314), label, fill=text)
            y = 514
            for idx, line in enumerate(metric_lines):
                x = 58 + (idx % 4) * 300
                line_y = y + (idx // 4) * 42
                draw.text((x, line_y), line, fill=text if idx < 4 else muted)
        return canvas

    def _handle_training_visual_snapshot(self, batch: torch.Tensor, recon: torch.Tensor, metrics: dict, *, force: bool = False) -> None:
        if not self.training_preview_enabled_var.get() and not self.timelapse_enabled_var.get():
            return

        step = int(metrics.get("step", 0))
        total_steps = max(1, int(metrics.get("total_steps", 1)))
        if self.timelapse_enabled_var.get():
            if self.timelapse_frames_dir is None:
                self._start_timelapse_session()
            self.timelapse_capture_interval = max(1, total_steps // TIMELAPSE_MAX_FRAMES)

        should_preview = self.training_preview_enabled_var.get() and (force or step == 1 or step % TRAINING_VISUAL_PREVIEW_INTERVAL == 0 or step >= total_steps)
        should_timelapse = self.timelapse_enabled_var.get() and (force or step == 1 or step >= total_steps or step - self.timelapse_last_capture_step >= self.timelapse_capture_interval)
        if not should_preview and not should_timelapse:
            return

        try:
            input_tensor = batch[0].detach().cpu()
            recon_tensor = recon[0].detach().cpu()
            if self._is_wavelet_mode():
                input_tensor = self._decode_model_output_to_rgb(input_tensor)
                recon_tensor = self._decode_model_output_to_rgb(recon_tensor)
        except Exception:
            return

        if should_preview:
            try:
                preview = self._training_visual_frame(input_tensor, recon_tensor, metrics, self._current_training_video_layout())
                self._after(0, lambda img=preview: self._display_image(img))
            except Exception:
                pass

        if should_timelapse and self.timelapse_frames_dir is not None:
            try:
                frame = self._training_visual_frame(input_tensor, recon_tensor, metrics, self.timelapse_layout)
            except Exception:
                return
            self.timelapse_frame_count += 1
            self.timelapse_last_capture_step = step
            frame_path = self.timelapse_frames_dir / f"{self.timelapse_frame_count:06d}.jpg"
            try:
                frame.save(frame_path, quality=92)
            except Exception:
                pass

    def _current_theme_label(self) -> str:
        mode = self.theme_mode_var.get()
        if mode == "Auto":
            hour = datetime.now().hour
            return "Day" if 7 <= hour < 19 else "Night"
        return mode

    def _current_mode_label(self) -> str:
        if self.dream_cycle_var.get():
            return "dream_cycle"
        if self.auto_cycle_var.get():
            return "auto_cycle"
        if self.model_cycle_active:
            return "model_shuffle"
        return "generate"

    def _refresh_memory_list(self):
        self.recent_memory_records = self.memory_bank.load_memories(limit=24)
        self.memory_listbox.delete(0, tk.END)
        for record in self.recent_memory_records:
            self.memory_listbox.insert(tk.END, summarize_memory(record))

    def _apply_personality_preset(self):
        preset = PERSONALITY_PRESETS.get(self.personality_var.get(), PERSONALITY_PRESETS["Manual"])
        self.var_scale.set(preset["intensity"])
        self.blend_mode_var.set(bool(preset["blend"]))
        self.blend_count_var.set(int(preset["blend_count"]))
        self.iterations_var.set(int(preset["iterations"]))
        self.use_mini_diffusion_var.set(bool(preset["use_diffusion"]))
        self.diffusion_steps_var.set(int(preset["diffusion_steps"]))
        self.diffusion_strength_var.set(float(preset["diffusion_strength"]))
        self.status_var.set(f"Applied personality preset: {self.personality_var.get()}")

    def _save_current_seed(self):
        if not self.last_generated_latents:
            messagebox.showerror("Save Seed", "Generate at least one image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=[("Latent seed", "*.pt"), ("All files", "*.*")], parent=self.root)
        if not path:
            return
        payload = {
            "latent": self.last_generated_latents[0].detach().cpu(),
            "prompt": self.generation_prompt_var.get().strip(),
            "personality": self.personality_var.get(),
            "reconstruction_mode": self._get_reconstruction_mode(),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        torch.save(payload, path)
        self.status_var.set(f"Saved latent seed: {Path(path).name}")

    def _load_seed(self):
        if self.model is None:
            messagebox.showerror("Load Seed", "Load or train a model first.")
            return
        path = filedialog.askopenfilename(filetypes=[("Latent seed", "*.pt"), ("All files", "*.*")], parent=self.root)
        if not path:
            return
        payload = safe_torch_load(path, map_location="cpu")
        latent = payload["latent"] if isinstance(payload, dict) and "latent" in payload else payload
        
        if isinstance(payload, dict) and "reconstruction_mode" in payload:
            seed_mode = payload["reconstruction_mode"]
            if seed_mode != self._get_reconstruction_mode():
                messagebox.showwarning("Mode Mismatch", f"Seed was saved in {seed_mode} mode, but current mode is {self._get_reconstruction_mode()}.")

        self._render_latent_gallery([latent.detach().float()], mode="seed_recall", status_label=f"Loaded seed: {Path(path).name}", save_memory=False)
        if isinstance(payload, dict):
            prompt = str(payload.get("prompt", "")).strip()
            if prompt:
                self.generation_prompt_var.set(prompt)

    def _remember_generation(self, image: Image.Image, latent: torch.Tensor, *, mode: str, extra: dict | None = None):
        model_name = "unloaded"
        if self.loaded_model_path is not None:
            model_name = self.loaded_model_path.name
        elif self.model is not None:
            model_name = f"latent-{self.model.latent_dim}"
        record = self.memory_bank.save_memory(
            image, latent, prompt=self.generation_prompt_var.get().strip(),
            mode=mode, personality=self.personality_var.get(),
            model_name=model_name, metadata=extra or {}
        )
        self._refresh_memory_list()
        return record

    def _get_selected_memory_record(self):
        selection = self.memory_listbox.curselection()
        if not selection:
            return None
        idx = int(selection[0])
        if idx < 0 or idx >= len(self.recent_memory_records):
            return None
        return self.recent_memory_records[idx]

    def _recall_selected_memory(self):
        if self.model is None:
            messagebox.showerror("Recall Memory", "Load or train a model first.")
            return
        record = self._get_selected_memory_record()
        if record is None:
            messagebox.showerror("Recall Memory", "Select a memory first.")
            return
        latent = self.memory_bank.load_latent(record).detach().float()
        self._render_latent_gallery([latent], mode="memory_recall", status_label=f"Recalled memory: {record.memory_id}", save_memory=False)

    def _use_memory_prompt(self):
        record = self._get_selected_memory_record()
        if record is None:
            messagebox.showerror("Memory Stream", "Select a memory first.")
            return
        self.generation_prompt_var.set(record.prompt)
        self.personality_var.set(record.personality)
        self.status_var.set(f"Loaded prompt tag from {record.memory_id}")

    def _open_latent_map(self):
        self.latent_map_points = self.memory_bank.build_latent_map(limit=64)
        if not self.latent_map_points:
            messagebox.showerror("Latent Map", "At least two saved memories are required.")
            return

        if self.latent_map_window is None or not self.latent_map_window.winfo_exists():
            self.latent_map_window = tk.Toplevel(self.root)
            self.latent_map_window.title("Latent Space Map")
            self.latent_map_canvas = tk.Canvas(self.latent_map_window, width=560, height=360, bg="#10131a")
            self.latent_map_canvas.pack(fill=tk.BOTH, expand=True)
            self.latent_map_canvas.bind("<Button-1>", self._on_latent_map_click)

        self._draw_latent_map()

    def _draw_latent_map(self):
        if self.latent_map_canvas is None:
            return
        canvas = self.latent_map_canvas
        canvas.delete("all")
        canvas.create_text(12, 12, anchor=tk.NW, fill="#d9e2ff", text="Click a node to recall that memory")
        for idx, point in enumerate(self.latent_map_points, start=1):
            record = point["record"]
            radius = 6
            x = point["x"]
            y = point["y"]
            fill = "#ffb347" if record.personality == "Chaotic" else "#7fd6ff"
            if record.personality in {"Dreamy", "Nostalgic"}:
                fill = "#b0e57c"
            canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill=fill, outline="")
            canvas.create_text(x + 10, y - 10, anchor=tk.W, fill="#eef3ff", text=str(idx))

    def _on_latent_map_click(self, event):
        if not self.latent_map_points:
            return
        best_point = min(self.latent_map_points, key=lambda point: (point["x"] - event.x) ** 2 + (point["y"] - event.y) ** 2)
        record = best_point["record"]
        try:
            latent = self.memory_bank.load_latent(record).detach().float()
        except Exception as exc:
            messagebox.showerror("Latent Map", str(exc))
            return
        self._render_latent_gallery([latent], mode="latent_map_recall", status_label=f"Latent map jump: {record.memory_id}", save_memory=False)

    @staticmethod
    def _coerce_int(value) -> int | None:
        if value is None or isinstance(value, bool):
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_metadata_value(value) -> str:
        if value is None or value == "":
            return "Unknown"
        if isinstance(value, (list, tuple)):
            return ", ".join(str(item) for item in value) if value else "Unknown"
        return str(value)

    def _extract_model_training_metadata(self, checkpoint: object, model_path: Path) -> dict:
        metadata: dict = {}
        if isinstance(checkpoint, dict):
            raw_metadata = checkpoint.get("training_metadata")
            if isinstance(raw_metadata, dict):
                metadata.update(raw_metadata)
            legacy_metadata = checkpoint.get("metadata")
            if isinstance(legacy_metadata, dict):
                for key, value in legacy_metadata.items():
                    metadata.setdefault(key, value)
            metadata.setdefault("latent_dim", checkpoint.get("latent_dim"))
            metadata.setdefault("output_size", checkpoint.get("output_size"))
            metadata.setdefault("version", checkpoint.get("version"))
            metadata.setdefault("reconstruction_mode", checkpoint.get("reconstruction_mode", "RGB VAE"))
            metadata.setdefault("in_channels", checkpoint.get("in_channels", 3))
            metadata.setdefault("out_channels", checkpoint.get("out_channels", 3))

        epochs = self._coerce_int(metadata.get("epochs"))
        total_epochs = self._coerce_int(metadata.get("total_epochs"))
        if total_epochs is None:
            total_epochs = epochs
        image_count = self._coerce_int(metadata.get("dataset_image_count"))
        if image_count is None:
            image_count = self._coerce_int(metadata.get("image_count"))

        return {
            "dataset_label": metadata.get("dataset_label"),
            "dataset_image_count": image_count,
            "epochs": epochs,
            "total_epochs": total_epochs,
            "chunk_count": self._coerce_int(metadata.get("chunk_count")),
            "source_count": self._coerce_int(metadata.get("source_count")),
            "resolution": metadata.get("resolution") or metadata.get("output_size"),
            "saved_at": metadata.get("saved_at"),
            "version": metadata.get("version"),
            "reconstruction_mode": metadata.get("reconstruction_mode", "RGB VAE"),
            "path": model_path,
        }

    def _format_model_details(self, details: dict) -> str:
        relative_path = details["path"]
        try:
            relative_path = relative_path.relative_to(MODELS_DIR)
        except ValueError:
            relative_path = details["path"]

        lines = [
            f"Model: {details['path'].name}",
            f"Folder: {relative_path.parent if relative_path.parent != Path('.') else 'Models'}",
            f"Reconstruction Mode: {self._format_metadata_value(details.get('reconstruction_mode'))}",
            f"Dataset label: {self._format_metadata_value(details.get('dataset_label'))}",
            f"Dataset image count: {self._format_metadata_value(details.get('dataset_image_count'))}",
            f"Epochs per chunk: {self._format_metadata_value(details.get('epochs'))}",
            f"Total epochs trained: {self._format_metadata_value(details.get('total_epochs'))}",
            f"Training chunks: {self._format_metadata_value(details.get('chunk_count'))}",
            f"Source files used: {self._format_metadata_value(details.get('source_count'))}",
            f"Resolution: {self._format_metadata_value(details.get('resolution'))}",
            f"Saved at: {self._format_metadata_value(details.get('saved_at'))}",
            f"Checkpoint version: {self._format_metadata_value(details.get('version'))}",
        ]
        if details.get("dataset_image_count") is None or details.get("total_epochs") is None:
            lines.append("")
            lines.append("Legacy checkpoint: this model was saved before training metadata was tracked.")
        return "\n".join(lines)

    def _open_model_map(self):
        models_folder = MODELS_DIR
        if not models_folder.exists():
            messagebox.showerror("PyTorch Map", f"Model folder not found:\n{models_folder.resolve()}")
            return

        if self.model_map_window is None or not self.model_map_window.winfo_exists():
            self.model_map_window = tk.Toplevel(self.root)
            self.model_map_window.title("PyTorch Model Map")
            self.model_map_window.geometry("900x560")
            self.model_map_window.minsize(760, 440)

            container = ttk.Frame(self.model_map_window, padding=10)
            container.pack(fill=tk.BOTH, expand=True)
            container.columnconfigure(0, weight=3)
            container.columnconfigure(1, weight=2)
            container.rowconfigure(1, weight=1)

            ttk.Label(container, text="Browse trained checkpoints in Models and click a .pt file to inspect its training metadata.").grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

            tree_frame = ttk.Frame(container)
            tree_frame.grid(row=1, column=0, sticky=tk.NSEW, padx=(0, 10))
            tree_frame.columnconfigure(0, weight=1)
            tree_frame.rowconfigure(0, weight=1)

            self.model_map_tree = ttk.Treeview(tree_frame, show="tree")
            self.model_map_tree.grid(row=0, column=0, sticky=tk.NSEW)
            self.model_map_tree.bind("<<TreeviewSelect>>", self._on_model_map_select)

            tree_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.model_map_tree.yview)
            tree_scroll.grid(row=0, column=1, sticky=tk.NS)
            self.model_map_tree.configure(yscrollcommand=tree_scroll.set)

            detail_frame = ttk.LabelFrame(container, text="Model Details", padding=10)
            detail_frame.grid(row=1, column=1, sticky=tk.NSEW)
            detail_frame.columnconfigure(0, weight=1)
            detail_frame.rowconfigure(0, weight=1)

            ttk.Label(detail_frame, textvariable=self.model_map_details_var, justify=tk.LEFT, anchor=tk.NW).grid(row=0, column=0, sticky=tk.NSEW)

        self._refresh_model_map()
        self.model_map_window.deiconify()
        self.model_map_window.lift()

    def _refresh_model_map(self):
        if self.model_map_tree is None:
            return
        models_folder = MODELS_DIR
        self.model_map_tree.delete(*self.model_map_tree.get_children())
        self.model_map_item_paths = {}
        self.model_map_details_var.set("Select a .pt model to inspect its training details.")

        root_id = self.model_map_tree.insert("", tk.END, text="Models", open=True)

        def add_folder(parent_id: str, folder: Path) -> None:
            for child in sorted(folder.iterdir(), key=lambda path: (not path.is_dir(), path.name.lower())):
                if child.is_dir():
                    child_id = self.model_map_tree.insert(parent_id, tk.END, text=child.name, open=False)
                    add_folder(child_id, child)
                elif child.suffix.lower() == ".pt":
                    item_id = self.model_map_tree.insert(parent_id, tk.END, text=child.name, open=False)
                    self.model_map_item_paths[item_id] = child

        add_folder(root_id, models_folder)

    def _on_model_map_select(self, _event=None):
        if self.model_map_tree is None:
            return
        selection = self.model_map_tree.selection()
        if not selection:
            return

        model_path = self.model_map_item_paths.get(selection[0])
        if model_path is None:
            self.model_map_details_var.set("Select a .pt model to inspect its training details.")
            return

        try:
            checkpoint = safe_torch_load(model_path, map_location="cpu")
            details = self._extract_model_training_metadata(checkpoint, model_path)
            self.model_map_details_var.set(self._format_model_details(details))
        except Exception as exc:
            self.model_map_details_var.set(f"Could not read {model_path.name}:\n{exc}")

    def _evolve_from_memory(self):
        if self.model is None:
            messagebox.showerror("Memory Retrain", "Load or train a model first.")
            return
        memory_paths = self.memory_bank.get_weighted_image_paths(limit=32, recent_bias=float(self.memory_recent_weight_var.get()))
        if not memory_paths:
            messagebox.showerror("Memory Retrain", "No saved memories were found.")
            return

        self.is_training = True
        self.train_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        def _run():
            try:
                completed = self._train_dataset(
                    n_epochs=max(1, min(30, int(self.epochs_var.get() // 4) or 1)),
                    resolution=int(self.resolution_var.get()),
                    dataset_label="Memory evolution",
                    training_paths=memory_paths,
                    reset_model=False,
                )
                if completed:
                    self._after(0, lambda: self.status_var.set("Memory evolution training finished."))
            except Exception as exc:
                error_message = str(exc)
                self._after(0, lambda msg=error_message: messagebox.showerror("Memory Retrain", msg))
            finally:
                self._finish_training()

        threading.Thread(target=_run, daemon=True).start()

    def _generate_evolution_round(self):
        if self.model is None:
            messagebox.showerror("Evolution", "Load or train a model first.")
            return
        count = max(4, min(8, int(self.evolution_count_var.get())))
        self.evolution_selection_var.set("")
        self.evolution_candidates = [{"latent": self._get_random_latent()} for _ in range(count)]
        self._render_latent_gallery([candidate["latent"] for candidate in self.evolution_candidates], mode="evolution_round", numbered=True, status_label=f"Evolution round ready. Pick favorites from 1-{count}.")

    def _breed_evolution_favorites(self):
        if self.model is None:
            messagebox.showerror("Evolution", "Load or train a model first.")
            return
        if not self.evolution_candidates:
            messagebox.showerror("Evolution", "Run an evolution round first.")
            return
        selected = parse_selection_indices(self.evolution_selection_var.get(), upper_bound=len(self.evolution_candidates))
        if not selected:
            messagebox.showerror("Evolution", "Enter favorite candidate numbers like 1,3,4.")
            return
        parents = [self.evolution_candidates[idx]["latent"] for idx in selected]
        child_latents = breed_latents(parents, noise_scale=max(0.05, self.var_scale.get() / 12.0), child_count=len(self.evolution_candidates))
        self.evolution_candidates = [{"latent": latent} for latent in child_latents]
        self._render_latent_gallery(child_latents, mode="evolution_breed", numbered=True, status_label=f"Bred {len(child_latents)} children from favorites {self.evolution_selection_var.get()}")

    def _ensure_reconstruction_judge(self):
        if ReconstructionJudge is None:
            messagebox.showerror("Reconstruction Judge", "reconstruction_judge.py could not be imported.", parent=self.root)
            return None
        if self.reconstruction_judge is None:
            try:
                self.status_var.set("Loading Reconstruction Judge...")
                self.root.update_idletasks()
                self.reconstruction_judge = ReconstructionJudge(memory_bank=self.memory_bank, device=self.device)
            except Exception as exc:
                messagebox.showerror("Reconstruction Judge", str(exc), parent=self.root)
                return None
        return self.reconstruction_judge

    def _ensure_memory_finder(self):
        if MemoryFinder is None:
            messagebox.showerror("Memory Finder", "memory_finder.py could not be imported.", parent=self.root)
            return None
        if self.memory_finder is None:
            try:
                self.status_var.set("Loading Memory Finder...")
                self.root.update_idletasks()
                self.memory_finder = MemoryFinder(memory_bank=self.memory_bank, device=self.device)
            except Exception as exc:
                messagebox.showerror("Memory Finder", str(exc), parent=self.root)
                return None
        return self.memory_finder

    def _compose_judged_grid(self, ranked_items, columns: int = 3):
        if not ranked_items:
            raise ValueError("No judged images to compose.")
        width, height = ranked_items[0]["image"].size
        columns = max(1, columns)
        rows = math.ceil(len(ranked_items) / columns)
        pad = 12
        label_h = 56
        canvas = Image.new("RGB", (columns * width + (columns + 1) * pad, rows * (height + label_h) + (rows + 1) * pad), color=(8, 10, 18))
        draw = ImageDraw.Draw(canvas)
        for grid_idx, item in enumerate(ranked_items, start=1):
            row = (grid_idx - 1) // columns
            col = (grid_idx - 1) % columns
            x = pad + col * (width + pad)
            y = pad + row * (height + label_h + pad)
            img = item["image"].resize((width, height), Image.Resampling.LANCZOS)
            canvas.paste(img, (x, y))
            scores = item["scores"]
            label = f"#{grid_idx}  total {scores.combined_score:.2f}\nS {scores.structure_score:.2f}  ID {scores.identity_score:.2f}  C {scores.composition_score:.2f}  Free {scores.dream_freedom_score:.2f}"
            draw.text((x, y + height + 5), label, fill=(235, 240, 255))
        return canvas

    def _judge_generate_candidates(self):
        if self.model is None:
            messagebox.showerror("Reconstruction Judge", "Load or train an APVD model first.", parent=self.root)
            return
        judge = self._ensure_reconstruction_judge()
        if judge is None:
            return

        reference = None
        if messagebox.askyesno("Reconstruction Judge", "Use a reference image for structure/identity scoring?", parent=self.root):
            ref_path = filedialog.askopenfilename(title="Select reference image", filetypes=self.FILE_TYPES, parent=self.root)
            if ref_path:
                try:
                    reference = Image.open(ref_path).convert("RGB")
                except Exception as exc:
                    messagebox.showerror("Reconstruction Judge", f"Could not open reference image:\n{exc}", parent=self.root)
                    return

        count = max(4, min(8, int(self.evolution_count_var.get())))
        top_k = max(1, min(count, int(self.judge_top_k_var.get())))
        min_score = float(self.judge_min_score_var.get())
        latents = [self._get_random_latent() for _ in range(count)]
        candidates = []

        self.status_var.set(f"Judge is scoring {count} APVD dreams...")
        self.root.update_idletasks()
        with torch.no_grad():
            for latent in latents:
                latent_device = latent.to(self.device)
                image = tensor_to_pil(self._decode_model_output_to_rgb(self._decode_latent(latent_device, show_steps=False)))
                candidates.append((image, latent_device.detach().cpu()))

        ranked = judge.rank(candidates, reference=reference)
        selected_indices = {idx for idx, scores in ranked[:top_k] if scores.combined_score >= min_score}
        ranked_items = []
        saved = 0
        for original_idx, scores in ranked:
            image, latent = candidates[original_idx]
            ranked_items.append({"image": image, "latent": latent, "scores": scores})
            if original_idx in selected_indices:
                self._remember_generation(image, latent, mode="judge_selected", extra={"judge_scores": scores.to_dict()})
                saved += 1

        self.last_generated_latents = [item["latent"] for item in ranked_items]
        self.evolution_candidates = [{"latent": item["latent"], "judge_scores": item["scores"].to_dict()} for item in ranked_items]
        self._display_image(self._compose_judged_grid(ranked_items))
        self.status_var.set(f"Judge ranked {count} dreams and saved {saved} best candidate(s) to memory.")

    def _memory_finder_search(self):
        finder = self._ensure_memory_finder()
        if finder is None:
            return
        ref_path = filedialog.askopenfilename(title="Select reference image for Memory Finder", filetypes=self.FILE_TYPES, parent=self.root)
        if not ref_path:
            return
        try:
            reference = Image.open(ref_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("Memory Finder", f"Could not open reference image:\n{exc}", parent=self.root)
            return

        top_k = max(1, min(12, int(self.memory_finder_top_k_var.get())))
        self.status_var.set("Building/searching APVD memory index...")
        self.root.update_idletasks()
        try:
            if hasattr(finder, "invalidate_index"):
                finder.invalidate_index()
            matches = finder.find(reference, top_k=top_k, search_limit=256, output_image_size=256)
        except Exception as exc:
            messagebox.showerror("Memory Finder", str(exc), parent=self.root)
            return

        if not matches:
            messagebox.showinfo("Memory Finder", "No memory matches were found yet.", parent=self.root)
            self.status_var.set("Memory Finder found no matches.")
            return

        images = []
        labels = []
        for idx, match in enumerate(matches, start=1):
            try:
                img = Image.open(match.image_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
                if match.attention_region:
                    draw = ImageDraw.Draw(img)
                    x, y, w, h = match.attention_region
                    draw.rectangle((x, y, x + w, y + h), outline=(255, 255, 255), width=2)
                images.append(img)
                labels.append(f"#{idx} score {match.combined_score:.2f}  id {match.identity_similarity:.2f}  struct {match.structure_similarity:.2f}")
            except Exception:
                continue

        if not images:
            messagebox.showinfo("Memory Finder", "Matches existed, but their images could not be opened.", parent=self.root)
            return

        grid = self._compose_memory_finder_grid(images, labels)
        self._display_image(grid)
        best = matches[0]
        self.status_var.set(f"Memory Finder found {len(matches)} match(es). Best: {best.combined_score:.2f} from {Path(best.image_path).name}")

    def _compose_memory_finder_grid(self, images, labels, columns: int = 3):
        width, height = images[0].size
        rows = math.ceil(len(images) / columns)
        pad = 12
        label_h = 42
        canvas = Image.new("RGB", (columns * width + (columns + 1) * pad, rows * (height + label_h) + (rows + 1) * pad), color=(8, 10, 18))
        draw = ImageDraw.Draw(canvas)
        for idx, img in enumerate(images):
            row = idx // columns
            col = idx % columns
            x = pad + col * (width + pad)
            y = pad + row * (height + label_h + pad)
            canvas.paste(img, (x, y))
            draw.text((x, y + height + 5), labels[idx], fill=(235, 240, 255))
        return canvas

    def _select_images(self):
        paths = filedialog.askopenfilenames(title="Select training images", filetypes=self.FILE_TYPES)
        if not paths: return
        self.training_paths = [Path(p) for p in paths]
        self.training_folder = None
        self.batch_training_root = None
        self.batch_training_folders = []
        self.status_var.set(f"Selected {len(self.training_paths)} images.")

    def _select_videos(self):
        paths = filedialog.askopenfilenames(title="Select training video(s)", filetypes=self.VIDEO_FILE_TYPES)
        if not paths:
            return
        self.video_paths = [Path(p) for p in paths]
        self.status_var.set(f"Selected {len(self.video_paths)} video file(s).")

    def _select_batch_folder(self):
        folder = filedialog.askdirectory(title="Select parent folder with dataset subfolders")
        if not folder:
            return
        root = Path(folder)
        dataset_folders = self._find_batch_dataset_folders(root)
        if not dataset_folders:
            messagebox.showerror("Error", "No dataset subfolders with images were found.")
            return
        self.batch_training_root = root
        self.batch_training_folders = dataset_folders
        self.training_folder = None
        self.training_paths = None
        self.archive_entries = []
        self.video_paths = []
        self.status_var.set(f"Queued {len(dataset_folders)} dataset folder(s) for batch training.")

    def _select_archives(self):
        paths = filedialog.askopenfilenames(title="Select training archive(s)", filetypes=self.ARCHIVE_FILE_TYPES)
        if not paths:
            return
        new_entries: list[tuple[Path, str]] = []
        n_ok = 0
        for p in paths:
            ap = Path(p)
            members = list_image_members(ap)
            if not members:
                messagebox.showerror("Error", f"No supported images found in archive:\n{ap.name}")
                continue
            n_ok += 1
            new_entries.extend((ap, m) for m in members)
        if not new_entries:
            return
        self.archive_entries.extend(new_entries)
        self.status_var.set(f"Added {len(new_entries)} image(s) from {n_ok} archive(s).")

    def _clear_training_sources(self):
        self.training_paths = None
        self.training_folder = None
        self.batch_training_root = None
        self.batch_training_folders = []
        self.archive_entries = []
        self.video_paths = []
        self.status_var.set("Cleared image/video/archive sources.")

    def _select_folder(self):
        folder = filedialog.askdirectory(title="Select folder with training images")
        if not folder: return
        self.training_folder = Path(folder)
        paths = get_image_paths(self.training_folder)
        if not paths:
            messagebox.showerror("Error", "No images found.")
            return
        self.batch_training_root = None
        self.batch_training_folders = []
        self.training_paths = paths
        self.status_var.set(f"Found {len(paths)} images in folder.")

    def _export_reconstruction_video(self):
        frame_paths = filedialog.askopenfilenames(title="Select ordered latent traversal frames", filetypes=self.FILE_TYPES, parent=self.root)
        if not frame_paths:
            return
        if len(frame_paths) < 2:
            messagebox.showerror("Reconstruction Video", "Select at least 2 frame images.")
            return

        original_path = None
        if messagebox.askyesno("Reconstruction Video", "Include an original input image for the initialization and final comparison phases?", parent=self.root):
            picked = filedialog.askopenfilename(title="Select original input image", filetypes=self.FILE_TYPES, parent=self.root)
            if picked:
                original_path = picked

        audio_path = None
        if messagebox.askyesno("Reconstruction Video", "Add an ambient audio track if you have one?", parent=self.root):
            picked = filedialog.askopenfilename(title="Select optional audio track", filetypes=self.AUDIO_FILE_TYPES, parent=self.root)
            if picked:
                audio_path = picked

        resolution_text = simpledialog.askstring("Reconstruction Video", "Output resolution (`1920x1080` or `1280x720`):", initialvalue="1920x1080", parent=self.root)
        if not resolution_text:
            return
        resolution_text = resolution_text.strip().lower().replace(" ", "")
        resolution_map = {"1920x1080": (1920, 1080), "1280x720": (1280, 720)}
        resolution = resolution_map.get(resolution_text)
        if resolution is None:
            messagebox.showerror("Reconstruction Video", "Resolution must be `1920x1080` or `1280x720`.")
            return

        fps = simpledialog.askinteger("Reconstruction Video", "FPS (`30` or `60`):", initialvalue=30, minvalue=30, maxvalue=60, parent=self.root)
        if fps is None:
            return
        if fps not in {30, 60}:
            messagebox.showerror("Reconstruction Video", "FPS must be `30` or `60`.")
            return

        suggested_name = f"reconstruction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        output_path = filedialog.asksaveasfilename(title="Save reconstruction video", defaultextension=".mp4", initialdir=str(OUTPUTS_DIR.resolve()), initialfile=suggested_name, filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")], parent=self.root)
        if not output_path:
            return

        ordered_paths = [Path(path) for path in sorted(frame_paths, key=lambda value: [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", Path(value).name.lower())])]
        self.status_var.set("Rendering Reconstruction Video Mode...")

        def progress_update(done: int, total: int):
            self._after(0, lambda d=done, t=total: self.status_var.set(f"Rendering Reconstruction Video Mode... {int((d / max(1, t)) * 100)}%"))

        def render_job():
            try:
                result = render_reconstruction_video(
                    ordered_paths, Path(output_path),
                    original_image_path=Path(original_path) if original_path else None,
                    audio_path=Path(audio_path) if audio_path else None,
                    resolution=resolution, fps=fps, progress_callback=progress_update
                )
            except Exception as exc:
                self._after(0, lambda msg=str(exc): (self.status_var.set("Reconstruction video export failed."), messagebox.showerror("Reconstruction Video", msg, parent=self.root)))
                return

            codec_note = "H.264" if result.used_h264 else "MP4 fallback codec"
            audio_note = " with audio" if result.audio_included else ""
            self._after(0, lambda: (self.status_var.set(f"Saved reconstruction video: {result.output_path.name} ({result.duration_seconds:.1f}s, {codec_note}{audio_note})."), messagebox.showinfo("Reconstruction Video", f"Saved video to:\n{result.output_path}\n\nDuration: {result.duration_seconds:.1f}s\nCodec: {codec_note}", parent=self.root)))

        threading.Thread(target=render_job, daemon=True).start()

    def _train(self):
        has_batch_folders = bool(self.batch_training_folders)
        has_images = bool(self.training_paths) or bool(self.archive_entries)
        has_videos = bool(self.video_paths)
        if not has_batch_folders and not has_images and not has_videos:
            messagebox.showerror("Error", "Select images, a folder, a batch folder, archive(s), and/or video(s) first.")
            return
        self.is_training = True
        self.training_pause_event.set()
        self.train_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL, text="Pause Training")
        self.stop_btn.config(state=tk.NORMAL)
        if self.timelapse_enabled_var.get():
            self._start_timelapse_session()
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

    def _stop_training(self):
        self.is_training = False
        self.training_pause_event.set()
        self.pause_btn.config(state=tk.DISABLED, text="Pause Training")
        self.status_var.set("Stopping training...")

    def _toggle_training_pause(self):
        if not self.is_training:
            return
        if self.training_pause_event.is_set():
            self.training_pause_event.clear()
            self.pause_btn.config(text="Resume Training")
            self.status_var.set("Training paused.")
        else:
            self.training_pause_event.set()
            self.pause_btn.config(text="Pause Training")
            self.status_var.set("Training resumed.")

    def _wait_if_training_paused(self) -> bool:
        while self.is_training and not self.training_pause_event.is_set():
            time.sleep(0.1)
        return self.is_training

    @staticmethod
    def _sanitize_model_stem(name: str) -> str:
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", name).strip()
        sanitized = re.sub(r"\s+", " ", sanitized)
        return sanitized or "model"

    def _find_batch_dataset_folders(self, root: Path) -> list[Path]:
        dataset_folders: list[Path] = []
        for child in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name.lower()):
            if get_image_paths(child):
                dataset_folders.append(child)
        return dataset_folders

    def _build_model_checkpoint(self) -> dict:
        if self.model is None:
            raise ValueError("No model is loaded.")
        mode = "Wavelet" if self._model_uses_wavelet() else "RGB VAE"
        in_ch = int(getattr(self.model, "in_channels", 12 if mode == "Wavelet" else 3))
        out_ch = int(getattr(self.model, "out_channels", 12 if mode == "Wavelet" else 3))
        return {
            "model_state_dict": self.model.state_dict(),
            "latent_dim": self.model.latent_dim,
            "output_size": self.model.output_size,
            "in_channels": in_ch,
            "out_channels": out_ch,
            "output_activation": getattr(self.model, "output_activation", "identity" if mode == "Wavelet" else "sigmoid"),
            "reconstruction_mode": mode,
            "version": "2.3-mini-diffusion",
            "training_metadata": dict(self.last_training_metadata),
        }

    def _next_available_model_path(self, target_dir: Path, model_name: str) -> Path:
        stem = self._sanitize_model_stem(model_name)
        candidate = target_dir / f"{stem}.pt"
        if not candidate.exists():
            return candidate
        index = 2
        while True:
            candidate = target_dir / f"{stem} ({index}).pt"
            if not candidate.exists():
                return candidate
            index += 1

    def _save_model_to_path(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._build_model_checkpoint(), path)
        if self.model_map_window is not None and self.model_map_window.winfo_exists():
            self._refresh_model_map()

    @staticmethod
    def _sample_training_preview(tensors: torch.Tensor, max_items: int) -> torch.Tensor:
        if tensors.size(0) <= max_items:
            return tensors.detach().cpu()
        idx = torch.randperm(tensors.size(0))[:max_items]
        return tensors[idx].detach().cpu()

    def _train_dataset(
        self,
        n_epochs: int,
        resolution: int,
        dataset_label: str,
        training_paths: list[Path] | None = None,
        archive_entries: list[tuple[Path, str]] | None = None,
        video_paths: list[Path] | None = None,
        reset_model: bool = False,
        batch_index: int | None = None,
        batch_total: int | None = None,
    ) -> bool:
        if reset_model:
            with self.model_lock:
                self.model = None
                self.loaded_model_path = None

        image_paths = [Path(path) for path in (training_paths or [])]
        archive_entries = list(archive_entries or [])
        video_paths = [Path(path) for path in (video_paths or [])]
        if not image_paths and not archive_entries and not video_paths:
            raise ValueError("No training data sources were provided.")

        target_size = (resolution, resolution)
        is_wavelet = self._is_wavelet_mode()
        model_output_size = self._wavelet_size_for_resolution(resolution) if is_wavelet else target_size
        in_channels = 12 if is_wavelet else 3
        out_channels = 12 if is_wavelet else 3
        output_activation = "identity" if is_wavelet else "sigmoid"

        with self.model_lock:
            if (
                self.model is None
                or getattr(self.model, "output_size", model_output_size) != model_output_size
                or int(getattr(self.model, "in_channels", 3)) != in_channels
                or getattr(self.model, "output_activation", "sigmoid") != output_activation
            ):
                self.model = VAE(
                    latent_dim=256,
                    in_channels=in_channels,
                    out_channels=out_channels,
                    output_size=model_output_size,
                    output_activation=output_activation,
                ).to(self.device)
                self.loaded_model_path = None

        precision = self._resolve_precision_settings()
        loss_mode = self.loss_mode_var.get() if hasattr(self, "loss_mode_var") else "Classic VAE"
        if loss_mode not in LOSS_MODE_CHOICES:
            loss_mode = "Classic VAE"
        amp_enabled = bool(precision["autocast_enabled"])
        scaler_enabled = bool(precision["scaler_enabled"])
        autocast_dtype = precision["autocast_dtype"]
        non_blocking = bool(precision["non_blocking"])
        if getattr(self.device, "type", "") == "cuda":
            torch.backends.cudnn.benchmark = True

        batch_size = max(1, min(256, int(self.batch_size_var.get())))
        # Workers are CPU processes. Huge values usually slow Windows down, so keep this realistic.
        max_workers = max(0, min(32, (os.cpu_count() or 4) - 1))
        loader_workers = max(0, min(max_workers, int(self.loader_workers_var.get())))
        # Prefetch is allowed to be larger than the old 16 cap, but it still represents batches per worker.
        prefetch_batches = max(1, min(512, int(self.prefetch_batches_var.get())))
        cache_total = max(0, min(200000, int(self.dataset_cache_items_var.get())))
        cache_per_worker = self._cache_items_per_worker(cache_total, loader_workers)
        dataset = APVDDataset(
            image_paths, resolution,
            archive_entries=archive_entries,
            video_paths=video_paths,
            video_stride=self.video_stride_var.get(),
            video_max_frames=self.video_max_frames_var.get(),
            wavelet_mode=is_wavelet,
            cache_limit=cache_per_worker,
        )
        learning_rate = max(1e-7, min(0.1, float(self.learning_rate_var.get())))
        intensity_profile = self._training_intensity_profile()
        training_intensity = int(intensity_profile["percent"])
        throttle_cap = float(intensity_profile["throttle_cap"])
        self._after(0, lambda bs=batch_size, lw=loader_workers, pf=prefetch_batches, cache=cache_total, lr=learning_rate: (self.batch_size_var.set(bs), self.loader_workers_var.set(lw), self.prefetch_batches_var.set(pf), self.dataset_cache_items_var.set(cache), self.learning_rate_var.set(lr)))

        data_loader = self._build_training_dataloader(dataset, batch_size=batch_size, loader_workers=loader_workers, prefetch_batches=prefetch_batches, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        self.model.train()
        train_start = time.perf_counter()
        preview_parts: list[torch.Tensor] = []
        total_batches = max(1, len(data_loader))
        total_steps = max(1, n_epochs * total_batches)
        completed_steps = 0
        total_training_images = len(dataset)

        self._after(0, lambda: self.load_progress_var.set(0.0))
        self._after(0, lambda p=str(precision["requested"]), r=str(precision["resolved"]), lm=loss_mode: self.status_var.set((f"Training using precision: {p} → {r}" if p == "Auto Recommended" else f"Training using precision: {r}") + f" | Loss: {lm}."))

        prefix_parts: list[str] = []
        if batch_index is not None and batch_total is not None:
            prefix_parts.append(f"[{batch_index}/{batch_total}]")
        prefix = " ".join(prefix_parts)
        if prefix:
            prefix += " "

        for epoch in range(n_epochs):
            if not self.is_training:
                return False
            if not self._wait_if_training_paused():
                return False

            epoch_loss = 0.0
            for batch_idx, batch in enumerate(data_loader, start=1):
                if not self.is_training:
                    return False
                if not self._wait_if_training_paused():
                    return False

                step_start = time.perf_counter()
                batch = batch.to(self.device, non_blocking=non_blocking).float()
                batch = self._sanitize_model_batch(batch, is_wavelet=is_wavelet)

                if len(preview_parts) < MAX_TRAINING_PREVIEW_IMAGES:
                    remaining = MAX_TRAINING_PREVIEW_IMAGES - len(preview_parts)
                    preview_parts.extend(batch[:remaining].detach().cpu())

                optimizer.zero_grad(set_to_none=True)

                autocast_kwargs = {"enabled": amp_enabled}
                if autocast_dtype is not None:
                    autocast_kwargs["dtype"] = autocast_dtype
                with torch.amp.autocast("cuda", **autocast_kwargs):
                    recon, mu, logvar = self.model(batch)
                with torch.amp.autocast("cuda", enabled=False):
                    recon_loss, structure_loss_metrics = self._apvd_training_loss(
                        recon.float(),
                        batch.float(),
                        mu.float(),
                        logvar.float(),
                        is_wavelet=is_wavelet,
                        loss_mode=loss_mode,
                    )
                    denoise_loss = latent_denoiser_loss(self.model, mu.detach().float())
                    loss = recon_loss + (0.25 * denoise_loss)

                if self.nan_guard_var.get() and not torch.isfinite(loss.detach()):
                    old_lr = learning_rate
                    learning_rate = max(learning_rate * 0.5, 1e-7)
                    for group in optimizer.param_groups:
                        group["lr"] = learning_rate
                    if scaler_enabled:
                        scaler.update()
                    self._after(0, lambda old=old_lr, new=learning_rate: self.status_var.set(f"NaN/Inf loss caught. Skipped batch and lowered LR from {old:.6f} to {new:.6f}."))
                    continue

                scaler.scale(loss).backward()
                if self.nan_guard_var.get():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.detach().float().item()

                completed_steps += 1
                if training_intensity < 100 and throttle_cap > 0.0:
                    step_seconds = max(0.0, time.perf_counter() - step_start)
                    sleep_seconds = min(throttle_cap, step_seconds * ((100 - training_intensity) / max(1, training_intensity)))
                    if sleep_seconds > 0.001 and self.is_training:
                        time.sleep(sleep_seconds)
                progress = min(100.0, (completed_steps / total_steps) * 100.0)
                elapsed = time.perf_counter() - train_start
                eta = (total_steps - completed_steps) * (elapsed / max(1, completed_steps))
                running_avg = epoch_loss / max(1, batch_idx)
                metrics = {
                    "dataset_label": dataset_label, "epoch": epoch + 1, "epochs": n_epochs,
                    "batch": batch_idx, "total_batches": total_batches, "loss": running_avg,
                    "elapsed": elapsed, "eta": eta, "progress": progress,
                    "step": completed_steps, "total_steps": total_steps
                }
                self._handle_training_visual_snapshot(batch, recon.detach(), metrics, force=batch_idx == 1 or completed_steps >= total_steps)
                self._after(0, lambda p=progress: self.load_progress_var.set(p))
                if batch_idx == 1 or batch_idx % 350 == 0 or batch_idx == total_batches:
                    self._after(0, lambda e=epoch, b=batch_idx, a=running_avg, t=eta, p=prefix, label=dataset_label: self.status_var.set(f"{p}{label} | Epoch {e+1}/{n_epochs} | Batch {b}/{total_batches} | Loss: {a:.0f} | ETA: {self._format_duration(t)}"))

            avg = epoch_loss / total_batches
            elapsed = time.perf_counter() - train_start
            eta = (n_epochs - (epoch + 1)) * (elapsed / max(1, epoch + 1))
            self._after(0, lambda e=epoch, a=avg, t=eta, p=prefix, label=dataset_label: self.status_var.set(f"{p}{label} | Epoch {e+1}/{n_epochs} | Loss: {a:.0f} | ETA: {self._format_duration(t)}"))

        if amp_enabled:
            torch.cuda.empty_cache()

        self.model.eval()
        self.last_training_metadata = {
            "dataset_label": dataset_label, "dataset_image_count": total_training_images,
            "epochs": n_epochs, "total_epochs": n_epochs, "chunk_count": 1,
            "source_count": len(image_paths), "resolution": [resolution, resolution],
            "batch_size": batch_size, "learning_rate": learning_rate,
            "training_intensity": training_intensity,
            "precision_mode": str(precision["requested"]),
            "precision_resolved": str(precision["resolved"]),
            "precision_dtype": str(precision["dtype_label"]),
            "loss_mode": loss_mode,
            "loss_mode_weights": self._loss_mode_weights(loss_mode),
            "mixed_precision": bool(amp_enabled),
            "nan_guard": bool(self.nan_guard_var.get()), "batch_load_workers": loader_workers,
            "prefetch_batches": prefetch_batches, "saved_at": datetime.now().isoformat(timespec="seconds"),
            "version": "2.3-mini-diffusion",
            "reconstruction_mode": self._get_reconstruction_mode(),
            "input_channels": in_channels, "output_channels": out_channels
        }
        if preview_parts:
            self.training_tensors = torch.stack(preview_parts, dim=0)[:MAX_TRAINING_PREVIEW_IMAGES]
        return True

    def _training_loop(self):
        n_epochs = max(1, int(self.epochs_var.get()))
        resolution = int(self.resolution_var.get())
        resolution = max(32, min(1024, resolution))
        if self._is_wavelet_mode() and resolution % 2 != 0:
            resolution = min(1024, resolution + 1)
        self._after(0, lambda: self.resolution_var.set(resolution))

        try:
            if self.batch_training_folders:
                models_folder = MODELS_DIR
                models_folder.mkdir(parents=True, exist_ok=True)
                total = len(self.batch_training_folders)
                saved_paths: list[Path] = []

                for index, dataset_folder in enumerate(self.batch_training_folders, start=1):
                    if not self.is_training:
                        break
                    if not self._wait_if_training_paused():
                        break
                    dataset_paths = get_image_paths(dataset_folder)
                    if not dataset_paths:
                        continue

                    self._after(0, lambda i=index, t=total, name=dataset_folder.name: self.status_var.set(f"[{i}/{t}] Preparing dataset: {name}"))
                    completed = self._train_dataset(
                        n_epochs=n_epochs, resolution=resolution, dataset_label=dataset_folder.name,
                        training_paths=dataset_paths, reset_model=True, batch_index=index, batch_total=total
                    )
                    if not completed:
                        break

                    model_path = self._next_available_model_path(models_folder, dataset_folder.name)
                    self._save_model_to_path(model_path)
                    saved_paths.append(model_path)
                    self._after(0, lambda i=index, t=total, p=model_path: self.status_var.set(f"[{i}/{t}] Saved {p.name} to Models."))

                if saved_paths and self.is_training:
                    self._after(0, lambda count=len(saved_paths), last=saved_paths[-1].name: self.status_var.set(f"Batch training finished. Saved {count} model(s); last: {last}"))
            else:
                memory_paths: list[Path] = []
                if self.include_memory_training_var.get():
                    memory_limit = max(0, int(self.memory_training_limit_var.get()))
                    if memory_limit > 0:
                        memory_paths = self.memory_bank.get_weighted_image_paths(limit=memory_limit, recent_bias=float(self.memory_recent_weight_var.get()))
                merged_training_paths = list(self.training_paths or [])
                merged_training_paths.extend(memory_paths)
                self._train_dataset(
                    n_epochs=n_epochs, resolution=resolution, dataset_label="Current dataset",
                    training_paths=merged_training_paths, archive_entries=self.archive_entries,
                    video_paths=self.video_paths, reset_model=False
                )
        except Exception as e:
            error_message = str(e)
            self._after(0, lambda msg=error_message: messagebox.showerror("Error", msg))
        finally:
            if not self.is_training:
                self._after(0, lambda: self.status_var.set("Training stopped."))
            if self.timelapse_frames_dir is not None:
                self._finish_timelapse_session()
        self._finish_training()

    def _finish_training(self):
        def _finish():
            self.is_training = False
            self.training_thread = None
            self.training_pause_event.set()
            self.train_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED, text="Pause Training")
            self.stop_btn.config(state=tk.DISABLED)

        if threading.current_thread() is threading.main_thread():
            _finish()
        else:
            self._after(0, _finish)

    def _save_model(self):
        if self.model is None: return
        path = filedialog.asksaveasfilename(defaultextension=".pt", filetypes=self.MODEL_FILE_TYPES)
        if path:
            self._save_model_to_path(Path(path))

    def _load_model(self):
        path = filedialog.askopenfilename(filetypes=self.MODEL_FILE_TYPES)
        if not path: return
        self._load_model_file(Path(path))

    def _load_model_file(self, path: Path):
        if self.realtime_dreamify_active:
            self._stop_realtime_dreamify()
        checkpoint = safe_torch_load(path, map_location=self.device)
        
        mode = checkpoint.get("reconstruction_mode")
        if mode not in ["RGB VAE", "Wavelet"]:
            saved_in_channels = checkpoint.get("in_channels")
            if saved_in_channels is None:
                first_weight = checkpoint.get("model_state_dict", {}).get("encoder.0.weight")
                saved_in_channels = int(first_weight.shape[1]) if hasattr(first_weight, "shape") and len(first_weight.shape) >= 2 else 3
            mode = "Wavelet" if int(saved_in_channels) == 12 else "RGB VAE"
        self.reconstruction_mode_var.set(mode)
        
        if self._is_wavelet_mode():
            in_ch = 12
            out_ch = 12
        else:
            in_ch = 3
            out_ch = 3

        output_size = tuple(checkpoint.get("output_size", (256, 256)))
        output_activation = checkpoint.get("output_activation", "identity" if mode == "Wavelet" else "sigmoid")
        loaded_model = VAE(
            latent_dim=checkpoint.get("latent_dim", 256),
            in_channels=in_ch,
            out_channels=out_ch,
            output_size=output_size,
            output_activation=output_activation,
        ).to(self.device)
        load_result = loaded_model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        loaded_model.eval()
        
        with self.model_lock:
            self.model = loaded_model
            self.loaded_model_path = Path(path)
            self.last_training_metadata = self._extract_model_training_metadata(checkpoint, self.loaded_model_path)
        if len(output_size) == 2 and all(isinstance(v, int) for v in output_size):
            display_size = self._rgb_size_for_wavelet_output(output_size) if mode == "Wavelet" else output_size
            self.resolution_var.set(int(display_size[0]))
            self._resize_output_window((int(display_size[0]), int(display_size[1])))
        missing = list(getattr(load_result, "missing_keys", []))
        has_denoiser = not any(key.startswith("latent_denoiser.") for key in missing)
        if not has_denoiser:
            self.use_mini_diffusion_var.set(False)
        note = " Mini diffusion ready." if has_denoiser else " Legacy checkpoint loaded; mini diffusion disabled until retrained."
        self.status_var.set(f"Model loaded: {path.name}.{note}")

        if self.latent_diffusion is not None and self.latent_diffusion.config.latent_dim != self.model.latent_dim:
            self.latent_diffusion = None
            self.loaded_latent_diffusion_path = None
            self.use_latent_diffusion_var.set(False)
            self.status_var.set(f"Model loaded: {path.name}.{note} Loaded DDPM was cleared because latent dimensions did not match.")

    def _make_latent_diffusion(self) -> DiffusionModel:
        if self.model is None:
            raise ValueError("Load or train an APVD model first.")
        timesteps = max(50, min(100, int(self.latent_diffusion_timesteps_var.get())))
        self.latent_diffusion_timesteps_var.set(timesteps)
        config = DiffusionConfig(latent_dim=self.model.latent_dim, timesteps=timesteps)
        return DiffusionModel(config).to(self.device)

    def _ensure_latent_diffusion(self) -> DiffusionModel:
        if self.model is None:
            raise ValueError("Load or train an APVD model first.")
        if self.latent_diffusion is None:
            raise ValueError("Train or load a Latent DDPM checkpoint first.")
        if self.latent_diffusion.config.latent_dim != self.model.latent_dim:
            raise ValueError(f"Latent DDPM uses {self.latent_diffusion.config.latent_dim} dims, but APVD uses {self.model.latent_dim} dims.")
        return self.latent_diffusion.to(self.device).eval()

    def _build_latent_diffusion_loader(self, resolution: int) -> DataLoader:
        image_paths = list(self.training_paths or [])
        archive_entries = list(self.archive_entries or [])
        video_paths = list(self.video_paths or [])
        batch_size = max(1, min(256, int(self.batch_size_var.get())))

        if image_paths or archive_entries or video_paths:
            max_workers = max(0, min(32, (os.cpu_count() or 4) - 1))
            loader_workers = max(0, min(max_workers, int(self.loader_workers_var.get())))
            prefetch_batches = max(1, min(512, int(self.prefetch_batches_var.get())))
            cache_total = max(0, min(200000, int(self.dataset_cache_items_var.get())))
            dataset = APVDDataset(
                image_paths, resolution,
                archive_entries=archive_entries, video_paths=video_paths,
                video_stride=self.video_stride_var.get(), video_max_frames=self.video_max_frames_var.get(),
                wavelet_mode=self._is_wavelet_mode(),
                cache_limit=self._cache_items_per_worker(cache_total, loader_workers),
            )
            return self._build_training_dataloader(dataset, batch_size=batch_size, loader_workers=loader_workers, prefetch_batches=prefetch_batches, shuffle=True)

        if self.training_tensors is not None and self.training_tensors.size(0) > 0:
            tensors = self.training_tensors.detach().float()
            return DataLoader(torch.utils.data.TensorDataset(tensors), batch_size=batch_size, shuffle=True, num_workers=0)

        raise ValueError("Select training images/folder/archive/video, or train APVD first so preview tensors exist.")

    def _train_latent_diffusion(self):
        if self.model is None:
            messagebox.showerror("Latent DDPM", "Load or train an APVD model first.")
            return
        if self.is_latent_diffusion_training:
            messagebox.showinfo("Latent DDPM", "Latent DDPM training is already running.")
            return

        epochs = simpledialog.askinteger("Train Latent DDPM", "Epochs for latent diffusion training:", initialvalue=10, minvalue=1, maxvalue=50000, parent=self.root)
        if epochs is None:
            return

        try:
            resolution = int(self.resolution_var.get())
            data_loader = self._build_latent_diffusion_loader(resolution)
        except Exception as exc:
            messagebox.showerror("Latent DDPM", str(exc))
            return

        self.latent_diffusion = self._make_latent_diffusion()
        self.loaded_latent_diffusion_path = None
        self.is_latent_diffusion_training = True
        self.latent_diffusion_train_btn.config(state=tk.DISABLED)
        self.status_var.set("Latent DDPM training started...")
        self.load_progress_var.set(0.0)

        def progress(update: dict):
            epoch = int(update["epoch"])
            total_epochs = int(update["epochs"])
            loss = float(update["loss"])
            progress_value = min(100.0, (epoch / max(1, total_epochs)) * 100.0)
            self._after(0, lambda e=epoch, t=total_epochs, l=loss, p=progress_value: (self.load_progress_var.set(p), self.status_var.set(f"Latent DDPM | Epoch {e}/{t} | Loss: {l:.5f}")))

        def train_job():
            try:
                stats = train_diffusion_model(
                    self.model, self.latent_diffusion, data_loader,
                    epochs=int(epochs), lr=max(1e-7, min(0.1, float(self.learning_rate_var.get()))),
                    device=self.device, progress_callback=progress
                )
                self._after(0, lambda s=stats: (self.status_var.set(f"Latent DDPM trained. Final loss: {float(s['final_loss']):.5f}"), self.use_latent_diffusion_var.set(True)))
            except Exception as exc:
                self._after(0, lambda msg=str(exc): messagebox.showerror("Latent DDPM", msg))
            finally:
                def finish():
                    self.is_latent_diffusion_training = False
                    self.latent_diffusion_train_btn.config(state=tk.NORMAL)
                self._after(0, finish)

        threading.Thread(target=train_job, daemon=True).start()

    def _save_latent_diffusion(self):
        if self.latent_diffusion is None:
            messagebox.showerror("Latent DDPM", "No Latent DDPM is trained or loaded.")
            return
        path = filedialog.asksaveasfilename(title="Save Latent DDPM", defaultextension=".pt", initialfile="apvd_latent_ddpm.pt", filetypes=self.MODEL_FILE_TYPES, parent=self.root)
        if not path:
            return
        try:
            out = self.latent_diffusion.save(path, metadata={"apvd_model": str(self.loaded_model_path) if self.loaded_model_path else None, "apvd_latent_dim": self.model.latent_dim if self.model is not None else None, "saved_at": datetime.now().isoformat(timespec="seconds")})
            self.loaded_latent_diffusion_path = out
            self.status_var.set(f"Saved Latent DDPM: {out.name}")
        except Exception as exc:
            messagebox.showerror("Latent DDPM", str(exc))

    def _load_latent_diffusion(self):
        if self.model is None:
            messagebox.showerror("Latent DDPM", "Load or train an APVD model first.")
            return
        path = filedialog.askopenfilename(title="Load Latent DDPM", filetypes=self.MODEL_FILE_TYPES, parent=self.root)
        if not path:
            return
        try:
            diffusion, _checkpoint = DiffusionModel.load(path, device=self.device)
            if diffusion.config.latent_dim != self.model.latent_dim:
                raise ValueError(f"Checkpoint latent_dim={diffusion.config.latent_dim}, but loaded APVD latent_dim={self.model.latent_dim}.")
            self.latent_diffusion = diffusion.eval()
            self.loaded_latent_diffusion_path = Path(path)
            self.latent_diffusion_timesteps_var.set(diffusion.config.timesteps)
            self.use_latent_diffusion_var.set(True)
            self.status_var.set(f"Loaded Latent DDPM: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Latent DDPM", str(exc))

    def _select_generation_image_tensor(self) -> torch.Tensor | None:
        path = filedialog.askopenfilename(title="Select image", filetypes=self.FILE_TYPES, parent=self.root)
        if not path:
            return None
        target_size = getattr(self.model, "output_size", (self.resolution_var.get(), self.resolution_var.get()))
        transform = transforms.Compose([transforms.Resize(target_size), transforms.CenterCrop(target_size), transforms.ToTensor()])
        with Image.open(path) as img:
            tensor = transform(img.convert("RGB")).unsqueeze(0).to(self.device)
            if self._is_wavelet_mode():
                tensor = rgb_to_wavelet(tensor)
            return tensor

    def _show_generated_tensor(self, image_tensor: torch.Tensor, latent: torch.Tensor, *, mode: str):
        image = tensor_to_pil(self._decode_model_output_to_rgb(image_tensor))
        self._display_image(image)
        self.last_generated_latents = [latent.detach().cpu()]
        self._remember_generation(image, latent.detach().cpu(), mode=mode)

    def _generate_apvd_reconstruction(self):
        if self.model is None:
            messagebox.showerror("APVD Reconstruction", "Load or train an APVD model first.")
            return
        image_tensor = self._select_generation_image_tensor()
        if image_tensor is None:
            return
        try:
            output, latent = apvd_reconstruction(self.model, image_tensor, device=self.device)
            self._show_generated_tensor(output, latent, mode="apvd_reconstruction")
            self.status_var.set("APVD Reconstruction generated.")
        except Exception as exc:
            messagebox.showerror("APVD Reconstruction", str(exc))

    def _generate_latent_diffusion_polish(self):
        if self.model is None:
            messagebox.showerror("DDPM Polish", "Load or train an APVD model first.")
            return
        image_tensor = self._select_generation_image_tensor()
        if image_tensor is None:
            return
        try:
            diffusion = self._ensure_latent_diffusion()
            output, latent = apvd_diffusion_polish(self.model, diffusion, image_tensor, strength=float(self.latent_diffusion_strength_var.get()), device=self.device)
            self._show_generated_tensor(output, latent, mode="apvd_latent_ddpm_polish")
            self.status_var.set("APVD + Latent DDPM Polish generated.")
        except Exception as exc:
            messagebox.showerror("DDPM Polish", str(exc))

    def _generate_pure_latent_diffusion(self):
        if self.model is None:
            messagebox.showerror("Pure DDPM", "Load or train an APVD model first.")
            return
        try:
            diffusion = self._ensure_latent_diffusion()
            output_count = max(1, min(8, int(self.output_count_var.get())))
            output, latents = pure_diffusion_generation(self.model, diffusion, batch_size=output_count, device=self.device)
            images = []
            self.last_generated_latents = []
            for index in range(output.size(0)):
                image = tensor_to_pil(self._decode_model_output_to_rgb(output[index]))
                images.append(image)
                latent = latents[index : index + 1].detach().cpu()
                self.last_generated_latents.append(latent)
                self._remember_generation(image, latent, mode="pure_latent_ddpm")
            final_image = images[0] if len(images) == 1 else self._compose_side_by_side(images)
            self._display_image(final_image)
            self.status_var.set(f"Pure Latent DDPM generated {len(images)} output(s).")
        except Exception as exc:
            messagebox.showerror("Pure DDPM", str(exc))

    def _dreamify_media(self):
        if self.model is None:
            messagebox.showerror("Dreamify Image/Video", "Load or train an APVD model first.")
            return
        path = filedialog.askopenfilename(
            title="Select image or video to Dreamify",
            filetypes=[("Image and video files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp *.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"), *self.IMAGE_FILE_TYPES[:-1], *self.VIDEO_FILE_TYPES[:-1], ("All files", "*.*")],
            parent=self.root
        )
        if not path:
            return
        media_path = Path(path)
        if self._is_video_path(media_path):
            self._open_dreamify_video_preview(media_path)
            return
        if self._is_image_path(media_path):
            try:
                self._dreamify_image_file(media_path)
            except Exception as exc:
                messagebox.showerror("Dreamify Image/Video", str(exc))
            return
        messagebox.showerror("Dreamify Image/Video", "Select a supported image or video file.")

    def _dreamify_tensor(self, input_tensor: torch.Tensor, *, dream_strength: float | None = None, memory_pull: float | None = None) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Load or train an APVD model first.")
        self.model.eval()
        if dream_strength is None:
            dream_strength = float(self.dream_strength_var.get())
        if memory_pull is None:
            memory_pull = float(self.memory_pull_var.get())
        dream_strength = max(0.0, min(2.0, float(dream_strength)))
        memory_pull = max(0.0, min(1.0, float(memory_pull)))
        source = input_tensor.detach().float()
        if source.ndim == 3:
            source = source.unsqueeze(0)
        source = self._sanitize_model_batch(source, is_wavelet=self._model_uses_wavelet()).to(self.device)

        with torch.no_grad():
            mu, _logvar = self.model.encode(source)
            latent = mu
            if dream_strength > 0.0:
                latent = latent + (torch.randn_like(latent) * dream_strength)
            if memory_pull > 0.0:
                anchor = self._dreamify_memory_anchor(latent)
                if anchor is not None:
                    latent = torch.lerp(latent, anchor.to(latent.device), memory_pull)
            output = self._decode_latent(latent, show_steps=False)
            output = self._sanitize_model_batch(output, is_wavelet=self._model_uses_wavelet())
        return output.detach().cpu()

    def _dreamify_memory_anchor(self, latent: torch.Tensor) -> torch.Tensor | None:
        candidates = [item.to(self.device) for item in self.last_generated_latents if isinstance(item, torch.Tensor) and item.numel() == latent.numel()]
        if candidates:
            return random.choice(candidates).reshape_as(latent)
        anchor = self._get_memory_anchor()
        if anchor is not None and anchor.numel() == latent.numel():
            return anchor.reshape_as(latent)
        random_anchor = self._get_random_latent()
        if random_anchor is not None and random_anchor.numel() == latent.numel():
            return random_anchor.reshape_as(latent)
        return None

    def _toggle_realtime_dreamify(self):
        if self.realtime_dreamify_active:
            self._stop_realtime_dreamify()
        else:
            self._start_realtime_dreamify()

    def _start_realtime_dreamify(self):
        if self.model is None:
            messagebox.showerror("Real-Time Dreamify", "Load or train an APVD model first.")
            return
        if self.is_training or self.is_latent_diffusion_training:
            messagebox.showwarning("Real-Time Dreamify", "Stop training before starting Real-Time Dreamify.")
            return
        try:
            import mss
        except ImportError:
            messagebox.showerror("Real-Time Dreamify", "Install the screen capture dependency first:\n\npip install mss")
            return

        try:
            resolution = int(self.live_resolution_var.get())
        except Exception:
            resolution = 192
        if resolution not in {128, 192, 256}:
            resolution = 192
            self.live_resolution_var.set("192")
        try:
            target_fps = int(self.live_target_fps_var.get())
        except Exception:
            target_fps = 10
        target_fps = max(1, min(30, target_fps))
        self.live_target_fps_var.set(target_fps)
        dream_strength = max(0.0, min(2.0, float(self.live_dream_strength_var.get())))
        memory_pull = max(0.0, min(1.0, float(self.live_memory_pull_var.get())))
        capture_mode = self.live_capture_mode_var.get()
        if capture_mode not in {"Full Screen", "Region"}:
            capture_mode = "Full Screen"
            self.live_capture_mode_var.set(capture_mode)
        display_mode = self.live_display_mode_var.get()
        if display_mode not in {"Output Window", "Challenge Window", "Experimental Fullscreen Overlay"}:
            display_mode = "Challenge Window"
            self.live_display_mode_var.set(display_mode)
        overlay_opacity = max(0.20, min(1.0, float(self.live_overlay_opacity_var.get())))
        self.live_overlay_opacity_var.set(overlay_opacity)
        clickthrough_overlay = bool(self.live_clickthrough_overlay_var.get())

        if capture_mode == "Region":
            region = self._prompt_capture_region()
            if region is None:
                return 
            self._live_region = region
        else:
            self._live_region = None

        memory_latent = None
        latent_dim = int(getattr(self.model, "latent_dim", 0) or 0)
        if memory_pull > 0.0 and latent_dim > 0:
            for latent in self.last_generated_latents:
                if isinstance(latent, torch.Tensor) and latent.numel() == latent_dim:
                    memory_latent = latent.detach().float().reshape(1, latent_dim).cpu()
                    break
            if memory_latent is None:
                try:
                    anchor = self._get_memory_anchor()
                    if isinstance(anchor, torch.Tensor) and anchor.numel() == latent_dim:
                        memory_latent = anchor.detach().float().reshape(1, latent_dim).cpu()
                except Exception:
                    memory_latent = None
            if memory_latent is None:
                memory_latent = torch.randn(1, latent_dim)

        settings = {
            "resolution": resolution, "target_fps": target_fps, "dream_strength": dream_strength,
            "memory_pull": memory_pull, "capture_mode": capture_mode, "show_fps": bool(self.live_show_fps_var.get()),
            "mini_diffusion": bool(self.live_mini_diffusion_var.get()), "mini_steps": max(1, min(4, int(self.diffusion_steps_var.get()))),
            "memory_latent": memory_latent, "display_mode": display_mode, "overlay_opacity": overlay_opacity,
            "clickthrough_overlay": clickthrough_overlay, "is_wavelet": self._is_wavelet_mode()
        }
        settings["region"] = self._live_region
        self.realtime_dreamify_settings = settings
        self.model.eval()
        if display_mode == "Challenge Window":
            self._destroy_live_overlay()
            self._create_live_challenge_window()
        elif display_mode == "Experimental Fullscreen Overlay":
            self._destroy_live_challenge_window()
            self._create_live_overlay_window(opacity=overlay_opacity, clickthrough=clickthrough_overlay)
        else: 
            self._destroy_live_overlay()
            self._destroy_live_challenge_window()
            self._create_output_window()
        self.realtime_dreamify_active = True
        self.realtime_dreamify_thread = threading.Thread(target=self._realtime_dreamify_worker, args=(mss, settings), daemon=True)
        self.realtime_dreamify_thread.start()
        self.status_var.set(f"Real-Time Dreamify running at {target_fps} FPS")

    def _stop_realtime_dreamify(self, *, update_status: bool = True):
        was_active = self.realtime_dreamify_active
        self.realtime_dreamify_active = False
        thread = self.realtime_dreamify_thread
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=1.0)
        self.realtime_dreamify_thread = None
        self._destroy_live_overlay()
        self._destroy_live_challenge_window()
        if getattr(self.device, "type", "") == "cuda":
            torch.cuda.empty_cache()
        if update_status and was_active and hasattr(self, "status_var"):
            self.status_var.set("Real-Time Dreamify stopped")

    def _realtime_dreamify_worker(self, mss_module, settings: dict[str, object]):
        frame_interval = 1.0 / max(1, int(settings.get("target_fps", 10)))
        frames_since_status = 0
        last_status_time = time.perf_counter()
        displayed_fps = float(settings.get("target_fps", 10))
        display_mode = settings.get("display_mode", "Challenge Window")
        overlay_mode = display_mode == "Experimental Fullscreen Overlay"
        challenge_mode = display_mode == "Challenge Window"
        escape_armed = not self._live_escape_is_down()
        consecutive_black = 0 

        if overlay_mode and os.name == "nt":
            self._after(0, self._exclude_overlay_from_capture)

        try:
            with mss_module.mss() as sct:
                while self.realtime_dreamify_active and not self._closing:
                    if overlay_mode:
                        escape_down = self._live_escape_is_down()
                        if escape_down and escape_armed:
                            self.realtime_dreamify_active = False
                            self._after(0, lambda: (self._destroy_live_overlay(), self.status_var.set("Real-Time Dreamify stopped")))
                            break
                        if not escape_down:
                            escape_armed = True

                    frame_start = time.perf_counter()
                    raw_frame = self._capture_screen_frame(sct, settings)
                    input_tensor = self._live_pil_to_tensor(raw_frame)
                    if settings.get("is_wavelet", False):
                        input_tensor = rgb_to_wavelet(input_tensor)
                    output_tensor = self._dreamify_live_frame(input_tensor, settings)
                    if settings.get("is_wavelet", False):
                        output_tensor = wavelet_to_rgb(output_tensor)
                    output_image = tensor_to_pil(output_tensor)

                    frames_since_status += 1
                    now = time.perf_counter()
                    status_elapsed = now - last_status_time
                    if status_elapsed >= 1.0:
                        displayed_fps = frames_since_status / status_elapsed
                        frames_since_status = 0
                        last_status_time = now
                        self._after(0, lambda fps=displayed_fps: self.status_var.set(f"Real-Time Dreamify running at {fps:.1f} FPS"))

                    extrema = raw_frame.convert("L").getextrema()
                    if extrema[1] < 8:
                        consecutive_black += 1
                    else:
                        consecutive_black = 0
                    if consecutive_black == 5:
                        self._after(0, lambda: self.status_var.set("Capture appears black. Try Borderless Windowed mode or Region capture."))
                    if overlay_mode and consecutive_black >= 20:
                        self.realtime_dreamify_active = False
                        self._after(0, lambda: (self._destroy_live_overlay(), self.live_display_mode_var.set("Challenge Window"), messagebox.showwarning("Real-Time Dreamify", "Fullscreen overlay failed or captured itself.\n\nTry Challenge Window mode with Region capture instead."), self.status_var.set("Real-Time Dreamify stopped")))
                        break

                    if challenge_mode:
                        img_copy = output_image.copy()
                        self._after(0, lambda img=img_copy: self._display_challenge_frame(img))
                    else:
                        self._display_live_frame(output_image, raw_frame=raw_frame if overlay_mode else None, fps=displayed_fps, show_fps=bool(settings.get("show_fps", True)), settings=settings)
                    elapsed = time.perf_counter() - frame_start
                    sleep_time = frame_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except Exception as exc:
            self.realtime_dreamify_active = False
            self._after(0, lambda msg=str(exc): (self.status_var.set("Real-Time Dreamify stopped"), messagebox.showerror("Real-Time Dreamify", msg)))
        finally:
            if getattr(self.device, "type", "") == "cuda":
                torch.cuda.empty_cache()

    def _capture_screen_frame(self, sct, settings: dict[str, object]) -> Image.Image:
        monitors = getattr(sct, "monitors", [])
        if not monitors:
            raise RuntimeError("No screen monitors were found.")
        monitor = monitors[1] if len(monitors) > 1 else monitors[0]
        stored_region = settings.get("region")
        if stored_region is not None:
            region = dict(stored_region)
        else:
            region = {"left": int(monitor["left"]), "top": int(monitor["top"]), "width": int(monitor["width"]), "height": int(monitor["height"])}
        shot = sct.grab(region)
        frame = np.asarray(shot)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
        resolution = max(64, int(settings.get("resolution", 192)))
        return image.resize((resolution, resolution), Image.Resampling.BILINEAR)

    def _live_pil_to_tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    def _dreamify_live_frame(self, input_tensor: torch.Tensor, settings: dict[str, object]) -> torch.Tensor:
        with self.model_lock:
            if self.model is None:
                raise ValueError("Load or train an APVD model first.")
            model = self.model
            model_device = next(model.parameters()).device
            if model_device != self.device:
                model = model.to(self.device)
                self.model = model
            model.eval()
        dream_strength = max(0.0, min(2.0, float(settings.get("dream_strength", 0.25))))
        memory_pull = max(0.0, min(1.0, float(settings.get("memory_pull", 0.15))))
        source = input_tensor.detach().float()
        if source.ndim == 3:
            source = source.unsqueeze(0)
        source = self._sanitize_model_batch(source, is_wavelet=bool(settings.get("is_wavelet", False))).to(self.device)

        with torch.no_grad():
            if hasattr(model, "encode") and hasattr(model, "decode"):
                mu, _logvar = model.encode(source)
                latent = mu
                fallback_output = None
            else:
                result = model(source)
                if isinstance(result, tuple) and len(result) >= 3:
                    fallback_output, latent, _logvar = result[:3]
                else:
                    fallback_output = result
                    latent = None

            if latent is not None:
                if dream_strength > 0.0:
                    latent = latent + (torch.randn_like(latent) * dream_strength)
                if memory_pull > 0.0:
                    memory_latent = settings.get("memory_latent")
                    if isinstance(memory_latent, torch.Tensor) and memory_latent.numel() == latent.numel():
                        memory_latent = memory_latent.to(latent.device).reshape_as(latent)
                    else:
                        memory_latent = torch.randn_like(latent)
                    latent = torch.lerp(latent, memory_latent, memory_pull)

                if bool(settings.get("mini_diffusion", False)) and hasattr(model, "predict_latent_noise"):
                    steps = max(1, min(4, int(settings.get("mini_steps", 2))))
                    for step_idx in range(steps):
                        t_value = 1.0 if steps == 1 else 1.0 - (step_idx / (steps - 1))
                        t = torch.full((latent.size(0), 1), t_value, device=latent.device)
                        predicted_noise = torch.nan_to_num(model.predict_latent_noise(latent, t), nan=0.0, posinf=0.0, neginf=0.0)
                        latent = torch.nan_to_num(latent - (predicted_noise * 0.12 * t_value), nan=0.0, posinf=6.0, neginf=-6.0).clamp(-6.0, 6.0)

                output = model.decode(latent) if hasattr(model, "decode") else fallback_output
            else:
                output = fallback_output

            if output is None:
                raise ValueError("The loaded model did not return a reconstructable output.")
            output = self._sanitize_model_batch(output, is_wavelet=bool(settings.get("is_wavelet", False)))
        return output.detach().cpu()

    def _display_live_frame(self, pil_img: Image.Image, *, raw_frame: Image.Image | None = None, fps: float | None = None, show_fps: bool = True, settings: dict[str, object] | None = None):
        if self._closing:
            return
        image = pil_img.convert("RGB")
        if show_fps and fps is not None:
            image = image.copy()
            draw = ImageDraw.Draw(image)
            label = f"{fps:.1f} FPS"
            try:
                bbox = draw.textbbox((0, 0), label)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = draw.textlength(label), 12
            x, y = 8, 8
            draw.rectangle((x - 4, y - 3, x + int(text_w) + 6, y + int(text_h) + 6), fill=(0, 0, 0))
            draw.text((x, y), label, fill=(255, 255, 255))

        def show():
            if self._closing:
                return
            if settings is not None and settings.get("display_mode") == "Experimental Fullscreen Overlay":
                self._display_live_overlay_image(image, raw_frame=raw_frame, opacity=float(settings.get("overlay_opacity", 0.85)), clickthrough=bool(settings.get("clickthrough_overlay", True)))
                return
            if self.output_window is None:
                self._create_output_window()
            self._display_image(image)

        self._after(0, show)

    def _display_live_overlay_image(self, image: Image.Image, *, raw_frame: Image.Image | None = None, opacity: float, clickthrough: bool):
        from PIL import ImageTk
        if self.live_overlay_window is None or self.live_overlay_canvas is None:
            self._create_live_overlay_window(opacity=1.0, clickthrough=clickthrough)
        win = self.live_overlay_window
        canvas = self.live_overlay_canvas
        if win is None or canvas is None:
            return
        try:
            if not win.winfo_exists():
                self.live_overlay_window = None
                self.live_overlay_canvas = None
                self._live_overlay_photo = None
                return

            width = max(1, win.winfo_width() or self.root.winfo_screenwidth())
            height = max(1, win.winfo_height() or self.root.winfo_screenheight())
            vae_out = image.resize((width, height), Image.Resampling.BILINEAR).convert("RGB")

            if raw_frame is not None:
                bg = raw_frame.resize((width, height), Image.Resampling.BILINEAR).convert("RGB")
                blend_alpha = max(0.0, min(1.0, float(opacity)))
                fitted = Image.blend(bg, vae_out, blend_alpha)
            else:
                fitted = vae_out

            self._live_overlay_photo = ImageTk.PhotoImage(fitted, master=win)
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=self._live_overlay_photo)
        except tk.TclError:
            self.live_overlay_window = None
            self.live_overlay_canvas = None
            self._live_overlay_photo = None

    def _dreamify_image_file(self, path: Path):
        output_dir = DREAMIFY_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as img:
            image = img.convert("RGB")
        input_tensor = self._pil_to_apvd_tensor(image)
        output_tensor = self._dreamify_tensor(input_tensor)
        output_image = tensor_to_pil(self._decode_model_output_to_rgb(output_tensor))
        out_path = output_dir / f"dreamify_{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_image.save(out_path)
        self._display_image(output_image)
        self.status_var.set(f"Dreamified image saved: {out_path.name}")
        self.last_generated_latents = []

    def _open_dreamify_video_preview(self, path: Path):
        win = tk.Toplevel(self.root)
        win.title(f"Dreamify Preview - {path.name}")
        win.transient(self.root)
        win.geometry("760x620")
        win.minsize(620, 520)
        win.configure(bg=self._theme_palette["bg"])

        dream_strength_var = tk.DoubleVar(value=float(self.dream_strength_var.get()))
        memory_pull_var = tk.DoubleVar(value=float(self.memory_pull_var.get()))
        frame_skip_var = tk.IntVar(value=max(1, int(self.dreamify_frame_skip_var.get())))
        keep_audio_var = tk.BooleanVar(value=bool(self.keep_original_audio_var.get()))
        state = {"busy": False, "photo": None}

        body = ttk.Frame(win, padding=12, style="Surface.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        ttk.Label(body, text=path.name, style="Surface.TLabel").pack(anchor=tk.W, pady=(0, 8))
        preview_label = ttk.Label(body, text="Generating preview...", anchor=tk.CENTER, style="Surface.TLabel")
        preview_label.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        controls = ttk.Frame(body, style="Surface.TFrame")
        controls.pack(fill=tk.X)
        dream_scale = self._register_theme_widget(tk.Scale(controls, from_=0.0, to=2.0, resolution=0.05, orient=tk.HORIZONTAL, length=300, variable=dream_strength_var), "scale")
        self._grid_row(controls, 0, "Dream Strength:", dream_scale)
        memory_scale = self._register_theme_widget(tk.Scale(controls, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=300, variable=memory_pull_var), "scale")
        self._grid_row(controls, 1, "Memory Pull:", memory_scale)
        frame_skip_spin = ttk.Spinbox(controls, from_=1, to=30, increment=1, width=6, textvariable=frame_skip_var)
        self._grid_row(controls, 2, "Frame Skip:", frame_skip_spin)
        ttk.Checkbutton(controls, text="Keep Original Audio", variable=keep_audio_var).grid(row=2, column=2, sticky=tk.W, padx=(0, 18), pady=4)

        buttons = ttk.Frame(body, style="Surface.TFrame")
        buttons.pack(fill=tk.X, pady=(12, 0))

        def current_settings() -> tuple[float, float, int, bool]:
            return (max(0.0, min(2.0, float(dream_strength_var.get()))), max(0.0, min(1.0, float(memory_pull_var.get()))), max(1, min(30, int(frame_skip_var.get()))), bool(keep_audio_var.get()))

        def apply_settings():
            dream_strength, memory_pull, frame_skip, keep_audio = current_settings()
            self.dream_strength_var.set(dream_strength)
            self.memory_pull_var.set(memory_pull)
            self.dreamify_frame_skip_var.set(frame_skip)
            self.keep_original_audio_var.set(keep_audio)
            return dream_strength, memory_pull, frame_skip, keep_audio

        def set_busy(busy: bool):
            state["busy"] = busy
            refresh_btn.config(state=tk.DISABLED if busy else tk.NORMAL)
            render_btn.config(state=tk.DISABLED if busy else tk.NORMAL)

        def refresh_preview():
            if state["busy"]:
                return
            dream_strength, memory_pull, _frame_skip, _keep_audio = current_settings()
            set_busy(True)
            preview_label.config(text="Generating preview...", image="")
            self.status_var.set(f"Generating Dreamify preview: {path.name}")

            def worker():
                try:
                    image = self._dreamify_video_preview_image(path, dream_strength, memory_pull)
                except Exception as exc:
                    self._after(0, lambda msg=str(exc): (set_busy(False), messagebox.showerror("Dreamify Preview", msg)))
                    return

                def show():
                    if not win.winfo_exists():
                        return
                    from PIL import ImageTk
                    display = image.copy()
                    display.thumbnail((700, 360), Image.Resampling.LANCZOS)
                    state["photo"] = ImageTk.PhotoImage(display, master=win)
                    preview_label.config(image=state["photo"], text="")
                    self._display_image(image)
                    self.status_var.set("Dreamify preview ready. Adjust settings or render the full video.")
                    set_busy(False)

                self._after(0, show)

            threading.Thread(target=worker, daemon=True).start()

        def render_video():
            if state["busy"]:
                return
            dream_strength, memory_pull, frame_skip, keep_audio = apply_settings()
            win.destroy()
            self.status_var.set(f"Dreamifying video: {path.name}")
            threading.Thread(target=self._dreamify_video_file, args=(path,), kwargs={"dream_strength": dream_strength, "memory_pull": memory_pull, "frame_skip": frame_skip, "keep_audio": keep_audio}, daemon=True).start()

        refresh_btn = ttk.Button(buttons, text="Refresh Preview", command=refresh_preview)
        refresh_btn.pack(side=tk.LEFT, padx=(0, 8))
        render_btn = ttk.Button(buttons, text="Render Full Video", command=render_video)
        render_btn.pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(buttons, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)

        self._after(100, refresh_preview)

    def _dreamify_video_preview_image(self, path: Path, dream_strength: float, memory_pull: float) -> Image.Image:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video:\n{path}")
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(total_frames - 1, total_frames // 3)))
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if not ok:
                raise ValueError("No readable preview frame was found in the selected video.")
        finally:
            cap.release()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        source_image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
        output_tensor = self._dreamify_tensor(self._pil_to_apvd_tensor(source_image), dream_strength=dream_strength, memory_pull=memory_pull)
        output_image = tensor_to_pil(self._decode_model_output_to_rgb(output_tensor)).resize(source_image.size, Image.Resampling.LANCZOS)
        return self._compose_side_by_side([source_image, output_image])

    def _dreamify_video_file(self, path: Path, *, dream_strength: float | None = None, memory_pull: float | None = None, frame_skip: int | None = None, keep_audio: bool | None = None):
        output_dir = DREAMIFY_OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"dreamify_{path.stem}_{timestamp}.mp4"
        video_only_path = output_path
        audio_mode = self.dreamify_audio_mode_var.get() if hasattr(self, "dreamify_audio_mode_var") else "Original Audio"
        if keep_audio is None:
            keep_audio = bool(self.keep_original_audio_var.get()) or audio_mode == "Original Audio"
        wants_memory_audio = audio_mode in AUDIO_MEMORY_DREAM_MODES
        wants_any_audio = bool(keep_audio) or wants_memory_audio
        if wants_any_audio:
            video_only_path = output_dir / f"dreamify_{path.stem}_{timestamp}_video_only.mp4"

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            self._after(0, lambda: messagebox.showerror("Dreamify Image/Video", f"Could not open video:\n{path}"))
            return

        writer = None
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or int(self.resolution_var.get())
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or int(self.resolution_var.get())
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_skip is None:
                frame_skip = int(self.dreamify_frame_skip_var.get())
            frame_skip = max(1, min(30, int(frame_skip)))
            writer = cv2.VideoWriter(str(video_only_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
            if not writer.isOpened():
                raise ValueError(f"Could not create video writer:\n{video_only_path}")

            frame_index = 0
            last_output = None
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                try:
                    if frame_index % frame_skip == 0 or last_output is None:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
                        output_tensor = self._dreamify_tensor(self._pil_to_apvd_tensor(pil), dream_strength=dream_strength, memory_pull=memory_pull)
                        last_output = self._tensor_to_cv_frame(output_tensor, size=(width, height))
                    if last_output is not None:
                        writer.write(last_output)
                except Exception:
                    if last_output is not None:
                        writer.write(last_output)
                frame_index += 1
                if total_frames > 0 and (frame_index == 1 or frame_index % 10 == 0):
                    pct = min(100, int((frame_index / total_frames) * 100))
                    self._after(0, lambda p=pct: self.status_var.set(f"Dreamifying video: {p}%"))
                    self._after(0, lambda img=Image.fromarray(cv2.cvtColor(last_output, cv2.COLOR_BGR2RGB)) if last_output is not None else None: self._display_image(img) if img is not None else None)

            if frame_index == 0:
                raise ValueError("No readable frames were found in the selected video.")
        except Exception as exc:
            self._after(0, lambda msg=str(exc): messagebox.showerror("Dreamify Image/Video", msg))
            return
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if getattr(self.device, "type", "") == "cuda":
                torch.cuda.empty_cache()

        final_path = video_only_path
        if wants_memory_audio:
            try:
                audio_source = self._resolve_audio_memory_source(path)
                if audio_source is None:
                    raise ValueError("No audio/video source was available for Audio Memory Reconstruction.")
                audio_path = output_dir / f"dreamify_{path.stem}_{timestamp}_audio_memory.wav"
                rendered_audio = self._render_audio_memory_file(audio_source, audio_path, seconds=(frame_index / float(fps)) if fps else None, mode=audio_mode)
                final_path, _audio_included = self._mux_audio_into_video(video_only_path, output_path, rendered_audio)
            except Exception as exc:
                self._after(0, lambda msg=str(exc): self.status_var.set(f"Audio memory failed; saved video-only export. {msg}"))
                final_path = video_only_path
        else:
            final_path = self._mux_original_audio(path, video_only_path, output_path, keep_audio=keep_audio)
        self._after(0, lambda p=final_path: (self.status_var.set(f"Dreamified video saved: {p.name}"), messagebox.showinfo("Dreamify Image/Video", f"Saved Dreamify video to:\n{p.resolve()}")))

    def _mux_original_audio(self, source_path: Path, video_only_path: Path, output_path: Path, *, keep_audio: bool | None = None) -> Path:
        if keep_audio is None:
            keep_audio = bool(self.keep_original_audio_var.get())
        if video_only_path == output_path or not bool(keep_audio):
            return output_path
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            try:
                video_only_path.replace(output_path)
            except Exception:
                return video_only_path
            return output_path
        cmd = [ffmpeg, "-y", "-i", str(video_only_path), "-i", str(source_path), "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-shortest", str(output_path)]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            video_only_path.unlink(missing_ok=True)
            return output_path
        except Exception:
            return video_only_path

    def _pil_to_apvd_tensor(self, image: Image.Image) -> torch.Tensor:
        target_size = self._model_rgb_input_size()
        transform = transforms.Compose([transforms.Resize(target_size, interpolation=transforms.InterpolationMode.BICUBIC), transforms.ToTensor()])
        tensor = transform(image.convert("RGB")).unsqueeze(0)
        if self._model_uses_wavelet():
            tensor = rgb_to_wavelet(tensor)
        return tensor

    def _tensor_to_cv_frame(self, tensor: torch.Tensor, *, size: tuple[int, int]) -> np.ndarray:
        image = tensor_to_pil(self._decode_model_output_to_rgb(tensor)).resize(size, Image.Resampling.LANCZOS)
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    def _make_initial_dream_latent(self) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Load or train an APVD model first.")
        latent_dim = int(self.model.latent_dim)
        for latent in self.last_generated_latents:
            if isinstance(latent, torch.Tensor) and latent.numel() == latent_dim:
                return latent.detach().float().reshape(1, latent_dim).cpu()
        memory_anchor = self._get_memory_anchor()
        if isinstance(memory_anchor, torch.Tensor) and memory_anchor.numel() == latent_dim:
            return memory_anchor.detach().float().reshape(1, latent_dim).cpu()
        return self._get_random_latent().detach().float().reshape(1, latent_dim).cpu()

    def _decode_latent_to_pil(self, latent: torch.Tensor, *, output_size: tuple[int, int], refined: bool = True) -> Image.Image:
        if self.model is None:
            raise ValueError("Load or train an APVD model first.")
        latent = latent.to(self.device)
        if refined:
            image_tensor = self._decode_latent(latent, show_steps=False)
        else:
            image_tensor = self.model.decode(latent)
        return tensor_to_pil(self._decode_model_output_to_rgb(image_tensor)).resize(output_size, Image.Resampling.LANCZOS)

    def _refine_dream_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Load or train an APVD model first.")
        current = latent.detach().float().reshape(1, -1).to(self.device)
        with torch.no_grad():
            if self.use_latent_diffusion_var.get() and self.latent_diffusion is not None:
                diffusion = self._ensure_latent_diffusion()
                current = diffusion.polish_latent(current, strength=max(0.0, min(1.0, float(self.latent_diffusion_strength_var.get()))))
            if self.use_mini_diffusion_var.get() and hasattr(self.model, "predict_latent_noise"):
                current = self._mini_diffusion_refine(current, show_steps=False)
            current = torch.nan_to_num(current, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
        return current

    def _pil_to_cv_frame(self, image: Image.Image, *, size: tuple[int, int]) -> np.ndarray:
        resized = image.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        rgb = np.asarray(resized, dtype=np.uint8)
        return cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR)

    def _open_audio_memory_settings(self):
        window = tk.Toplevel(self.root)
        window.title("APVD Audio Memory / AFVD Lite")
        window.resizable(False, False)
        window.transient(self.root)
        try:
            window.geometry(f"+{self.root.winfo_x() + 80}+{self.root.winfo_y() + 80}")
        except tk.TclError:
            pass
        frame = ttk.Frame(window, padding=14)
        frame.pack(fill=tk.BOTH, expand=True)
        ttk.Label(frame, text="Audio Memory Reconstruction", font=("", 12, "bold")).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 8))
        ttk.Label(frame, text="A small AFVD-style audio memory for APVD exports. It keeps the important controls and auto-handles segment timing.", wraplength=520).grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(0, 10))

        ttk.Label(frame, text="Source:").grid(row=2, column=0, sticky=tk.W, pady=4)
        ttk.Entry(frame, textvariable=self.audio_memory_source_var, width=52).grid(row=2, column=1, sticky=tk.EW, padx=(8, 8), pady=4)
        ttk.Button(frame, text="Select Audio/Video", command=self._select_audio_memory_source).grid(row=2, column=2, sticky=tk.W, pady=4)

        ttk.Label(frame, text="Dreamify Audio Mode:").grid(row=3, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(frame, textvariable=self.dreamify_audio_mode_var, values=AUDIO_MEMORY_MODE_CHOICES, state="readonly", width=26).grid(row=3, column=1, sticky=tk.W, padx=(8, 8), pady=4)

        ttk.Label(frame, text="Dream Continuation Audio:").grid(row=4, column=0, sticky=tk.W, pady=4)
        ttk.Combobox(frame, textvariable=self.dream_video_audio_mode_var, values=("Silent", "Use Source Video Audio", "Procedural Dream Audio", "Audio Memory Reconstruction", "Dreamy Memory Audio", "Corrupted Memory Audio"), state="readonly", width=26).grid(row=4, column=1, sticky=tk.W, padx=(8, 8), pady=4)

        ttk.Label(frame, text="Epochs:").grid(row=5, column=0, sticky=tk.W, pady=4)
        ttk.Spinbox(frame, from_=1, to=5000, increment=1, width=8, textvariable=self.audio_memory_epochs_var).grid(row=5, column=1, sticky=tk.W, padx=(8, 8), pady=4)
        ttk.Label(frame, text="Mel bins:").grid(row=5, column=1, sticky=tk.W, padx=(120, 0), pady=4)
        ttk.Combobox(frame, textvariable=self.audio_memory_mel_bins_var, values=(64, 96, 128, 192, 256), state="readonly", width=8).grid(row=5, column=1, sticky=tk.W, padx=(190, 0), pady=4)

        strength_scale = self._register_theme_widget(tk.Scale(frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=260, variable=self.audio_memory_strength_var), "scale")
        self._grid_row(frame, 6, "Audio Strength:", strength_scale)
        noise_scale = self._register_theme_widget(tk.Scale(frame, from_=0.0, to=0.5, resolution=0.01, orient=tk.HORIZONTAL, length=260, variable=self.audio_memory_noise_var), "scale")
        self._grid_row(frame, 7, "Memory Noise:", noise_scale)

        ttk.Checkbutton(frame, text="Keep rhythm/timing from source", variable=self.audio_memory_keep_rhythm_var).grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(8, 2))
        ttk.Checkbutton(frame, text="Cleanup / reduce clicks", variable=self.audio_memory_cleanup_var).grid(row=9, column=0, columnspan=2, sticky=tk.W, pady=2)

        button_row = ttk.Frame(frame)
        button_row.grid(row=10, column=0, columnspan=3, sticky=tk.W, pady=(12, 0))
        ttk.Button(button_row, text="Train Audio Memory", command=self._train_audio_memory).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Save Audio Memory", command=self._save_audio_memory_profile).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Load Audio Memory", command=self._load_audio_memory_profile).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Preview Reconstructed Audio", command=self._preview_audio_memory).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Close", command=window.destroy).pack(side=tk.LEFT)
        frame.columnconfigure(1, weight=1)
        self._apply_registered_theme()

    def _select_audio_memory_source(self):
        path = filedialog.askopenfilename(
            title="Select audio or video for Audio Memory",
            filetypes=[("Audio / Video", "*.mp3 *.wav *.ogg *.m4a *.aac *.flac *.mp4 *.mov *.mkv *.avi *.webm"), ("All files", "*.*")],
            parent=self.root,
        )
        if path:
            self.audio_memory_source_var.set(path)
            if not self.dream_video_audio_source_var.get().strip():
                self.dream_video_audio_source_var.set(path)
            self.status_var.set(f"Audio memory source selected: {Path(path).name}")

    def _resolve_audio_memory_source(self, fallback: Path | None = None) -> Path | None:
        raw = self.audio_memory_source_var.get().strip()
        if raw:
            path = Path(raw)
            if path.exists():
                return path
        if fallback is not None and Path(fallback).exists():
            return Path(fallback)
        raw = self.dream_video_audio_source_var.get().strip()
        if raw and Path(raw).exists():
            return Path(raw)
        if self.video_paths:
            first = Path(self.video_paths[0])
            if first.exists():
                return first
        return None

    def _train_audio_memory(self):
        if self.is_audio_memory_training:
            messagebox.showinfo("Audio Memory", "Audio memory training is already running.")
            return
        source = self._resolve_audio_memory_source()
        if source is None:
            self._select_audio_memory_source()
            source = self._resolve_audio_memory_source()
            if source is None:
                return
        epochs = max(1, min(5000, int(self.audio_memory_epochs_var.get())))
        mel_bins = max(16, min(512, int(self.audio_memory_mel_bins_var.get())))
        self.is_audio_memory_training = True
        self.status_var.set("Training audio memory: starting...")
        threading.Thread(target=self._train_audio_memory_worker, args=(source, epochs, mel_bins), daemon=True).start()

    def _train_audio_memory_worker(self, source: Path, epochs: int, mel_bins: int) -> None:
        try:
            profile = self._build_audio_memory_profile(source, epochs=epochs, mel_bins=mel_bins)
            self.audio_memory_profile = profile
            self.loaded_audio_memory_path = None
            def _done():
                self.is_audio_memory_training = False
                self.status_var.set(f"Audio memory trained from {source.name} ({mel_bins} bins, {epochs} epochs).")
                messagebox.showinfo("Audio Memory", "Audio memory profile is ready. Use Audio Memory Reconstruction/Dreamy/Corrupted in Dreamify Audio.")
            self._after(0, _done)
        except Exception as exc:
            self._after(0, lambda msg=str(exc): (setattr(self, "is_audio_memory_training", False), self.status_var.set("Audio memory training failed."), messagebox.showerror("Audio Memory", msg)))

    def _extract_audio_to_wav(self, source_path: Path, output_wav: Path, *, sample_rate: int = 22050, duration: float | None = None) -> Path:
        source_path = Path(source_path)
        output_wav = Path(output_wav)
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            if source_path.suffix.lower() == ".wav":
                return source_path
            raise RuntimeError("FFmpeg is required to extract audio from video or non-WAV files.")
        cmd = [ffmpeg, "-y", "-i", str(source_path), "-vn", "-ac", "1", "-ar", str(int(sample_rate))]
        if duration is not None and duration > 0:
            cmd.extend(["-t", f"{float(duration):.3f}"])
        cmd.extend(["-f", "wav", str(output_wav)])
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_wav

    def _read_wav_mono(self, wav_path: Path) -> tuple[np.ndarray, int]:
        import wave
        with wave.open(str(wav_path), "rb") as wf:
            channels = wf.getnchannels()
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if sample_width == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")
        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
        return np.nan_to_num(data.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0), int(sample_rate)

    def _write_wav_mono(self, output_path: Path, audio: np.ndarray, sample_rate: int) -> Path:
        import wave
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        audio = np.clip(audio, -0.98, 0.98)
        pcm = (audio * 32767.0).astype(np.int16)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(pcm.tobytes())
        return output_path

    def _band_energy_profile(self, audio: np.ndarray, sample_rate: int, bins: int) -> tuple[list[float], float, float]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size < 1024:
            audio = np.pad(audio, (0, 1024 - audio.size))
        window = min(4096, max(1024, 2 ** int(np.floor(np.log2(max(1024, min(audio.size, 4096)))))))
        hop = max(256, window // 4)
        chunks = []
        win = np.hanning(window).astype(np.float32)
        for start in range(0, max(1, audio.size - window + 1), hop):
            chunk = audio[start:start + window]
            if chunk.size < window:
                chunk = np.pad(chunk, (0, window - chunk.size))
            spectrum = np.abs(np.fft.rfft(chunk * win)).astype(np.float32)
            band_values = [float(np.mean(band)) for band in np.array_split(spectrum, int(bins))]
            chunks.append(band_values)
            if len(chunks) >= 512:
                break
        arr = np.asarray(chunks or [[0.0] * int(bins)], dtype=np.float32)
        profile = np.log1p(arr.mean(axis=0))
        if float(profile.max()) > 0:
            profile = profile / float(profile.max())
        rms = float(np.sqrt(np.mean(np.square(audio))) + 1e-8)
        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio).astype(np.float32))))) if audio.size > 1 else 0.0
        return profile.astype(float).tolist(), rms, zcr

    def _build_audio_memory_profile(self, source: Path, *, epochs: int, mel_bins: int) -> dict[str, object]:
        import tempfile
        sample_rate = 22050
        with tempfile.TemporaryDirectory(prefix="apvd_audio_memory_") as tmp:
            wav_path = self._extract_audio_to_wav(source, Path(tmp) / "source.wav", sample_rate=sample_rate)
            audio, sr = self._read_wav_mono(wav_path)
        if audio.size < sr // 8:
            raise ValueError("The selected source did not contain enough readable audio.")
        profile, rms, zcr = self._band_energy_profile(audio, sr, mel_bins)
        stride = max(1, epochs // 20)
        for epoch in range(1, epochs + 1):
            if epoch == 1 or epoch == epochs or epoch % stride == 0:
                pct = int((epoch / max(1, epochs)) * 100)
                self._after(0, lambda p=pct: self.status_var.set(f"Training audio memory: {p}%"))
            if epoch % max(1, epochs // 8) == 0:
                time.sleep(0.001)
        return {
            "kind": "APVD_AFVD_LITE_AUDIO_MEMORY",
            "version": 1,
            "source_name": Path(source).name,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "sample_rate": int(sr),
            "mel_bins": int(mel_bins),
            "epochs": int(epochs),
            "band_profile": profile,
            "rms": rms,
            "zero_crossing_rate": zcr,
        }

    def _save_audio_memory_profile(self):
        if not self.audio_memory_profile:
            messagebox.showinfo("Audio Memory", "Train or load an audio memory profile first.")
            return
        AUDIO_MEMORY_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        default = f"audio_memory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = filedialog.asksaveasfilename(title="Save Audio Memory", defaultextension=".json", initialdir=str(AUDIO_MEMORY_PROFILE_DIR), initialfile=default, filetypes=[("Audio memory profile", "*.json"), ("All files", "*.*")], parent=self.root)
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.audio_memory_profile, f, indent=2)
        self.loaded_audio_memory_path = Path(path)
        self.status_var.set(f"Audio memory saved: {Path(path).name}")

    def _load_audio_memory_profile(self):
        AUDIO_MEMORY_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(title="Load Audio Memory", initialdir=str(AUDIO_MEMORY_PROFILE_DIR), filetypes=[("Audio memory profile", "*.json"), ("All files", "*.*")], parent=self.root)
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
        if profile.get("kind") != "APVD_AFVD_LITE_AUDIO_MEMORY":
            messagebox.showerror("Audio Memory", "That file does not look like an APVD AFVD Lite audio memory profile.")
            return
        self.audio_memory_profile = profile
        self.loaded_audio_memory_path = Path(path)
        self.audio_memory_mel_bins_var.set(int(profile.get("mel_bins", self.audio_memory_mel_bins_var.get())))
        self.audio_memory_epochs_var.set(int(profile.get("epochs", self.audio_memory_epochs_var.get())))
        self.status_var.set(f"Audio memory loaded: {Path(path).name}")

    def _preview_audio_memory(self):
        source = self._resolve_audio_memory_source()
        if source is None:
            self._select_audio_memory_source()
            source = self._resolve_audio_memory_source()
            if source is None:
                return
        output = AUDIO_MEMORY_PROFILE_DIR / "audio_memory_preview.wav"
        try:
            rendered = self._render_audio_memory_file(source, output, seconds=8.0, mode=self.dreamify_audio_mode_var.get())
            self.status_var.set(f"Audio memory preview saved: {rendered.name}")
            if os.name == "nt":
                os.startfile(str(rendered))
            else:
                opener = "open" if sys.platform == "darwin" else "xdg-open"
                subprocess.Popen([opener, str(rendered)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            messagebox.showerror("Audio Memory", str(exc))

    def _render_audio_memory_file(self, source: Path, output_wav: Path, *, seconds: float | None = None, mode: str | None = None) -> Path:
        import tempfile
        source = Path(source)
        mode = mode or "Audio Memory Reconstruction"
        sample_rate = 22050
        if self.audio_memory_profile is None:
            self.audio_memory_profile = self._build_audio_memory_profile(
                source,
                epochs=max(1, min(5000, int(self.audio_memory_epochs_var.get()))),
                mel_bins=max(16, min(512, int(self.audio_memory_mel_bins_var.get()))),
            )
        with tempfile.TemporaryDirectory(prefix="apvd_audio_render_") as tmp:
            wav_path = self._extract_audio_to_wav(source, Path(tmp) / "source.wav", sample_rate=sample_rate, duration=seconds)
            audio, sr = self._read_wav_mono(wav_path)
        if seconds is not None and seconds > 0:
            target = max(1, int(float(seconds) * sr))
            if audio.size < target:
                audio = np.pad(audio, (0, target - audio.size))
            elif audio.size > target:
                audio = audio[:target]
        processed = self._apply_audio_memory_effect(audio, sr, mode=mode)
        return self._write_wav_mono(output_wav, processed, sr)

    def _apply_audio_memory_effect(self, audio: np.ndarray, sample_rate: int, *, mode: str) -> np.ndarray:
        audio = np.nan_to_num(np.asarray(audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            return audio
        strength = max(0.0, min(1.0, float(self.audio_memory_strength_var.get())))
        noise_amount = max(0.0, min(0.5, float(self.audio_memory_noise_var.get())))
        if mode == "Dreamy Memory Audio":
            strength = min(1.0, strength + 0.10)
            noise_amount = min(0.5, noise_amount + 0.025)
        elif mode == "Corrupted Memory Audio":
            strength = min(1.0, strength + 0.20)
            noise_amount = min(0.5, noise_amount + 0.08)
        profile = self.audio_memory_profile or {}
        bands = np.asarray(profile.get("band_profile", []), dtype=np.float32)
        if bands.size == 0:
            bands = np.linspace(1.0, 0.35, 128, dtype=np.float32)
        bands = np.nan_to_num(bands, nan=0.0, posinf=1.0, neginf=0.0)
        bands = np.clip(bands, 0.02, 1.0)
        window = min(2048, max(512, 2 ** int(np.floor(np.log2(max(512, min(audio.size, 2048)))))))
        hop = max(128, window // 4)
        win = np.hanning(window).astype(np.float32)
        out = np.zeros(audio.size + window, dtype=np.float32)
        norm = np.zeros(audio.size + window, dtype=np.float32)
        rng = np.random.default_rng(int(time.time()) % 2_147_483_647)
        for start in range(0, audio.size, hop):
            chunk = audio[start:start + window]
            if chunk.size < window:
                chunk = np.pad(chunk, (0, window - chunk.size))
            spec = np.fft.rfft(chunk * win)
            gain = np.interp(np.linspace(0, 1, spec.size), np.linspace(0, 1, bands.size), bands)
            gain = 0.35 + (gain * 0.95)
            if mode == "Dreamy Memory Audio":
                gain *= np.linspace(1.05, 0.60, spec.size)
            elif mode == "Corrupted Memory Audio":
                wobble = 0.8 + 0.35 * np.sin(np.linspace(0, np.pi * 10.0, spec.size))
                gain *= wobble
            spec = spec * ((1.0 - strength * 0.75) + (gain * strength * 0.75))
            rebuilt = np.fft.irfft(spec, n=window).astype(np.float32) * win
            out[start:start + window] += rebuilt
            norm[start:start + window] += win * win
        out = out[:audio.size] / np.maximum(norm[:audio.size], 1e-4)
        if bool(self.audio_memory_keep_rhythm_var.get()):
            envelope_win = max(128, int(sample_rate * 0.025))
            kernel = np.ones(envelope_win, dtype=np.float32) / float(envelope_win)
            src_env = np.convolve(np.abs(audio), kernel, mode="same")
            out_env = np.convolve(np.abs(out), kernel, mode="same") + 1e-5
            out = out * np.clip(src_env / out_env, 0.25, 3.0)
        if noise_amount > 0.0:
            hiss = rng.normal(0.0, noise_amount * 0.06, audio.size).astype(np.float32)
            wobble = 1.0 + (noise_amount * 0.08 * np.sin(np.linspace(0, np.pi * 12.0, audio.size, dtype=np.float32)))
            out = (out * wobble) + hiss
        if mode == "Corrupted Memory Audio":
            crush = max(16.0, 256.0 - (strength * 180.0))
            out = np.round(out * crush) / crush
        if bool(self.audio_memory_cleanup_var.get()):
            kernel_size = 5 if mode != "Corrupted Memory Audio" else 3
            kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
            out = np.convolve(out, kernel, mode="same")
        mix = (audio * max(0.0, 1.0 - strength)) + (out * strength)
        peak = float(np.max(np.abs(mix))) if mix.size else 0.0
        if peak > 0.98:
            mix = mix / peak * 0.96
        return np.clip(mix, -0.98, 0.98).astype(np.float32)

    def _select_dream_audio_source(self):
        path = filedialog.askopenfilename(title="Select audio or video source for Dream Continuation", filetypes=[("Audio / Video", "*.mp3 *.wav *.ogg *.m4a *.aac *.flac *.mp4 *.mov *.mkv *.avi *.webm"), ("All files", "*.*")], parent=self.root)
        if path:
            self.dream_video_audio_source_var.set(path)
            self.status_var.set(f"Dream audio source selected: {Path(path).name}")

    def _select_dream_structure_video(self):
        path = filedialog.askopenfilename(title="Select source video for Dream Continuation structure", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.wmv"), ("All files", "*.*")], parent=self.root)
        if path:
            self.dream_structure_video_source_var.set(path)
            if not self.dream_video_audio_source_var.get().strip():
                self.dream_video_audio_source_var.set(path)
            self.status_var.set(f"Dream structure video selected: {Path(path).name}")

    def _pil_to_model_tensor(self, image: Image.Image, *, resolution: int) -> torch.Tensor:
        image = image.convert("RGB").resize((int(resolution), int(resolution)), Image.Resampling.LANCZOS)
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous().unsqueeze(0)
        # Convert RGB to Wavelet representation if needed
        if self.model is not None and self.model.in_channels == 12:
            tensor = rgb_to_wavelet(tensor)
        return tensor.to(self.device)

    def _encode_pil_to_latent(self, image: Image.Image, *, resolution: int) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Load or train an APVD model first.")
        tensor = self._pil_to_model_tensor(image, resolution=resolution)
        with torch.no_grad():
            mu, _logvar = self.model.encode(tensor)
        return torch.nan_to_num(mu.detach().float(), nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)

    def _load_dream_structure_latents(self, video_path: Path | None, *, resolution: int, max_anchors: int = 72) -> list[torch.Tensor]:
        if video_path is None or not Path(video_path).exists():
            return []
        if self.model is None:
            raise ValueError("Load or train an APVD model first.")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open structure video:\n{video_path}")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count <= 0:
                return []
            anchor_count = max(4, min(int(max_anchors), frame_count))
            indices = np.linspace(0, max(0, frame_count - 1), anchor_count, dtype=np.int64)
            latents: list[torch.Tensor] = []
            last_frame = None
            for frame_index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                small = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
                if last_frame is not None:
                    delta = float(np.mean(cv2.absdiff(small, last_frame)))
                    if delta < 1.5 and len(latents) > 4:
                        continue
                last_frame = small
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(np.ascontiguousarray(rgb)).convert("RGB")
                latents.append(self._encode_pil_to_latent(image, resolution=resolution).detach().cpu())
            if len(latents) < 2:
                return latents
            stack = torch.cat([z.reshape(1, -1) for z in latents], dim=0)
            common = stack.mean(dim=0, keepdim=True)
            generalized = []
            previous = None
            for z in latents:
                z = z.reshape(1, -1)
                z = torch.lerp(z, common, 0.28)
                if previous is not None:
                    z = torch.lerp(z, previous, 0.35)
                generalized.append(torch.nan_to_num(z, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0))
                previous = generalized[-1]
            return generalized
        finally:
            cap.release()

    def _interpolate_structure_anchor(self, anchors: list[torch.Tensor], timeline_t: float) -> torch.Tensor | None:
        if not anchors:
            return None
        if len(anchors) == 1:
            return anchors[0].detach().float().reshape(1, -1).to(self.device)
        timeline_t = max(0.0, min(1.0, float(timeline_t)))
        position = timeline_t * (len(anchors) - 1)
        idx = int(math.floor(position))
        idx2 = min(len(anchors) - 1, idx + 1)
        local_t = position - idx
        local_t = local_t * local_t * (3.0 - (2.0 * local_t))
        a = anchors[idx].detach().float().reshape(1, -1).to(self.device)
        b = anchors[idx2].detach().float().reshape(1, -1).to(self.device)
        if torch.norm(a).item() > 1e-6 and torch.norm(b).item() > 1e-6:
            return slerp(local_t, a, b)
        return torch.lerp(a, b, local_t)

    def _make_dream_scene_target(self, current_latent: torch.Tensor, *, drift: float, instability: float) -> torch.Tensor:
        if self.model is None:
            raise ValueError("Load or train an APVD model first.")
        latent_dim = int(self.model.latent_dim)
        anchor = None
        try:
            anchor = self._get_memory_anchor()
        except Exception:
            anchor = None
        if anchor is None:
            try:
                anchor = self._get_blended_anchor(max(2, min(6, int(self.blend_count_var.get()))))
            except Exception:
                anchor = None
        if anchor is None:
            try:
                anchor = self._get_random_latent()
            except Exception:
                anchor = None
        if not isinstance(anchor, torch.Tensor) or anchor.numel() != latent_dim:
            anchor = torch.randn(1, latent_dim, device=self.device)
        anchor = anchor.detach().float().reshape(1, latent_dim).to(self.device)
        current_latent = current_latent.detach().float().reshape(1, latent_dim).to(self.device)
        continuity = max(0.35, min(0.92, 1.0 - (float(drift) * 0.22)))
        target = torch.lerp(anchor, current_latent, continuity)
        noise_amount = (0.04 + float(drift) * 0.10 + float(instability) * 0.055)
        target = target + (torch.randn_like(target) * noise_amount)
        return torch.nan_to_num(target, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)

    def _apply_autoregressive_dream_feedback(
        self,
        generated_image: Image.Image,
        predicted_next_latent: torch.Tensor,
        *,
        resolution: int,
        feedback_strength: float,
        structure_guidance: float = 0.0,
        has_structure_anchors: bool = False,
    ) -> torch.Tensor:
        """Use the generated frame as the next-frame input, similar to autoregressive video generation.

        APVD still moves through latent memory, but this feedback step makes the next
        frame continue from what the decoder actually drew instead of only from the
        planned latent target. Structure-guided videos get a slightly lower feedback
        mix so source-video anchors do not get overwritten too quickly.
        """
        mix = max(0.0, min(1.0, float(feedback_strength)))
        if mix <= 0.0:
            return predicted_next_latent

        feedback_latent = self._encode_pil_to_latent(generated_image, resolution=resolution)
        feedback_latent = feedback_latent.detach().float().reshape_as(predicted_next_latent).to(self.device)
        if has_structure_anchors:
            mix *= max(0.20, 1.0 - (float(structure_guidance) * 0.55))

        if torch.norm(predicted_next_latent).item() > 1e-6 and torch.norm(feedback_latent).item() > 1e-6:
            updated = slerp(mix, predicted_next_latent, feedback_latent)
        else:
            updated = torch.lerp(predicted_next_latent, feedback_latent, mix)
        return torch.nan_to_num(updated, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)

    def _write_procedural_dream_audio(self, output_path: Path, *, seconds: float, seed: int = 1337) -> Path:
        import wave
        rng = random.Random(seed)
        sample_rate = 44100
        total = max(1, int(float(seconds) * sample_rate))
        t = np.linspace(0.0, float(seconds), total, endpoint=False, dtype=np.float32)
        base = rng.choice([55.0, 65.41, 73.42, 82.41, 98.0])
        signal = np.zeros_like(t)
        for i, mult in enumerate([1.0, 1.5, 2.0, 2.5, 3.0]):
            wobble = 0.7 + (0.35 * np.sin(2.0 * np.pi * (0.03 + i * 0.017) * t))
            signal += (0.13 / (i + 1)) * np.sin(2.0 * np.pi * base * mult * wobble * t)
        pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * 0.45 * t)
        hiss = np.random.default_rng(seed).normal(0.0, 0.018, total).astype(np.float32)
        signal = (signal * (0.45 + pulse * 0.35)) + hiss
        fade_len = min(total // 4, int(sample_rate * 1.5))
        if fade_len > 0:
            fade = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            signal[:fade_len] *= fade
            signal[-fade_len:] *= fade[::-1]
        signal = np.clip(signal, -0.95, 0.95)
        pcm = (signal * 32767.0).astype(np.int16)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(output_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())
        return output_path

    def _mux_audio_into_video(self, video_only_path: Path, output_path: Path, audio_path: Path | None) -> tuple[Path, bool]:
        if audio_path is None or not Path(audio_path).exists():
            if video_only_path != output_path:
                try:
                    video_only_path.replace(output_path)
                except Exception:
                    return video_only_path, False
            return output_path, False
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            if video_only_path != output_path:
                try:
                    video_only_path.replace(output_path)
                except Exception:
                    return video_only_path, False
            return output_path, False
        cmd = [ffmpeg, "-y", "-i", str(video_only_path), "-i", str(audio_path), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(output_path)]
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            video_only_path.unlink(missing_ok=True)
            return output_path, True
        except Exception:
            return video_only_path, False

    def _generate_dream_continuation_video(self):
        if self.model is None:
            messagebox.showerror("Dream Continuation Video", "Load or train an APVD model first.")
            return
        if self.is_dream_video_generating:
            messagebox.showinfo("Dream Continuation Video", "A dream video is already generating.")
            return

        try:
            initial_latent = self._make_initial_dream_latent()
            seconds = max(1, min(180, int(self.dream_video_seconds_var.get())))
            fps = max(4, min(30, int(self.dream_video_fps_var.get())))
            latent_drift = max(0.0, min(2.0, float(self.latent_drift_var.get())))
            motion_smoothness = max(0.0, min(0.99, float(self.motion_smoothness_var.get())))
            dream_instability = max(0.0, min(1.5, float(self.dream_instability_var.get())))
            resolution = max(32, min(1024, int(self.resolution_var.get())))
            audio_mode = self.dream_video_audio_mode_var.get()
            audio_source = Path(self.dream_video_audio_source_var.get()) if self.dream_video_audio_source_var.get().strip() else None
            structure_video = Path(self.dream_structure_video_source_var.get()) if self.dream_structure_video_source_var.get().strip() else None
            structure_guidance = max(0.0, min(1.0, float(self.dream_structure_guidance_var.get())))
            autoregressive_feedback = bool(self.dream_autoregressive_var.get())
            feedback_strength = max(0.0, min(1.0, float(self.dream_feedback_strength_var.get())))
            needs_source_audio = audio_mode == "Use Source Video Audio" or audio_mode in AUDIO_MEMORY_DREAM_MODES
            if needs_source_audio and audio_source is None and structure_video is not None:
                audio_source = structure_video
            if needs_source_audio and audio_source is None:
                audio_source = self._resolve_audio_memory_source()
            if needs_source_audio and audio_source is None and self.video_paths:
                audio_source = Path(self.video_paths[0])
            if needs_source_audio and audio_source is None:
                selected = filedialog.askopenfilename(title="Select a video/audio file for Dream Continuation audio", filetypes=[("Audio / Video", "*.mp3 *.wav *.ogg *.m4a *.aac *.flac *.mp4 *.mov *.mkv *.avi *.webm"), ("All files", "*.*")], parent=self.root)
                if not selected:
                    return
                audio_source = Path(selected)
                self.dream_video_audio_source_var.set(str(audio_source))
                self.audio_memory_source_var.set(str(audio_source))
        except Exception as exc:
            messagebox.showerror("Dream Continuation Video", str(exc))
            return

        self.dream_video_seconds_var.set(seconds)
        self.dream_video_fps_var.set(fps)
        self.latent_drift_var.set(latent_drift)
        self.motion_smoothness_var.set(motion_smoothness)
        self.dream_instability_var.set(dream_instability)
        self.dream_structure_guidance_var.set(structure_guidance)
        self.dream_autoregressive_var.set(autoregressive_feedback)
        self.dream_feedback_strength_var.set(feedback_strength)
        self.resolution_var.set(resolution)
        self.is_dream_video_generating = True
        self.status_var.set("Generating dream video: 0%")

        threading.Thread(target=self._generate_dream_video_worker, kwargs={
            "initial_latent": initial_latent, "seconds": seconds, "fps": fps, "latent_drift": latent_drift,
            "motion_smoothness": motion_smoothness, "dream_instability": dream_instability, "resolution": resolution,
            "audio_mode": audio_mode, "audio_source": audio_source, "structure_video": structure_video, "structure_guidance": structure_guidance,
            "autoregressive_feedback": autoregressive_feedback, "feedback_strength": feedback_strength
        }, daemon=True).start()

    def _generate_dream_video_worker(self, *, initial_latent: torch.Tensor, seconds: int, fps: int, latent_drift: float, motion_smoothness: float, dream_instability: float, resolution: int, audio_mode: str = "Silent", audio_source: Path | None = None, structure_video: Path | None = None, structure_guidance: float = 0.0, autoregressive_feedback: bool = True, feedback_strength: float = 0.35) -> None:
        output_dir = DREAM_VIDEOS_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = output_dir / f"dream_continuation_{timestamp}.mp4"
        video_only_path = output_dir / f"dream_continuation_{timestamp}_video_only.mp4"
        temp_audio_path = output_dir / f"dream_continuation_{timestamp}_procedural.wav"
        writer = None
        final_image: Image.Image | None = None
        final_latent: torch.Tensor | None = None
        audio_included = False

        try:
            if self.model is None:
                raise ValueError("Load or train an APVD model first.")

            self.model.to(self.device).eval()
            total_frames = max(1, int(seconds) * int(fps))
            size = (int(resolution), int(resolution))
            writer = cv2.VideoWriter(str(video_only_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), size)
            if not writer.isOpened():
                raise ValueError(f"Could not create video writer:\n{output_path}")

            latent = initial_latent.detach().float().reshape(1, -1).to(self.device)
            latent_dim = int(self.model.latent_dim)
            if latent.size(1) != latent_dim:
                raise ValueError(f"Dream latent has {latent.size(1)} dims, but APVD uses {latent_dim}.")

            structure_anchors = self._load_dream_structure_latents(structure_video, resolution=resolution, max_anchors=max(8, min(96, total_frames // max(1, fps // 2)))) if structure_video is not None and structure_guidance > 0.0 else []
            if structure_anchors:
                self._after(0, lambda n=len(structure_anchors): self.status_var.set(f"Loaded {n} structure anchors from source video."))
                first_anchor = structure_anchors[0].reshape(1, -1).to(self.device)
                latent = torch.lerp(latent, first_anchor, min(0.85, structure_guidance * 0.75))
                latent = self._refine_dream_latent(latent)

            alpha = max(0.02, min(0.65, 1.0 - float(motion_smoothness)))
            if structure_anchors:
                alpha = max(alpha, 0.10 + (structure_guidance * 0.22))
            segment_frames = max(6, int(fps * (1.0 + motion_smoothness * 3.0)))
            progress_stride = max(1, int(fps / 2))
            preview_stride = max(1, int(fps))
            target_latent = self._make_dream_scene_target(latent, drift=latent_drift, instability=dream_instability)
            velocity = torch.zeros_like(latent)

            with torch.no_grad():
                for frame_index in range(total_frames):
                    render_latent = self._refine_dream_latent(latent)
                    image = self._decode_latent_to_pil(render_latent, output_size=size, refined=True)
                    writer.write(self._pil_to_cv_frame(image, size=size))
                    final_image = image

                    if frame_index == 0 or frame_index == total_frames - 1 or frame_index % progress_stride == 0:
                        pct = min(100, int(((frame_index + 1) / total_frames) * 100))
                        label = "structure-guided" if structure_anchors else "memory-guided"
                        if autoregressive_feedback and feedback_strength > 0.0:
                            label += " autoregressive"
                        self._after(0, lambda p=pct, l=label: self.status_var.set(f"Generating dream continuation ({l}): {p}%"))
                    if frame_index == 0 or frame_index == total_frames - 1 or frame_index % preview_stride == 0:
                        self._after(0, lambda img=image.copy(): self._display_image(img))

                    timeline_t = (frame_index + 1) / max(1, total_frames - 1)
                    if structure_anchors:
                        source_target = self._interpolate_structure_anchor(structure_anchors, timeline_t)
                        memory_target = target_latent
                        if frame_index > 0 and frame_index % segment_frames == 0:
                            memory_target = self._make_dream_scene_target(latent, drift=latent_drift * 0.5, instability=dream_instability * 0.35)
                            target_latent = memory_target
                        target_latent = torch.lerp(memory_target, source_target, structure_guidance)
                    elif frame_index > 0 and frame_index % segment_frames == 0:
                        target_latent = self._make_dream_scene_target(latent, drift=latent_drift, instability=dream_instability)
                    else:
                        target_latent = target_latent + (torch.randn_like(target_latent) * float(latent_drift) * 0.0025)

                    if torch.norm(latent).item() > 1e-6 and torch.norm(target_latent).item() > 1e-6:
                        guided = slerp(alpha, latent, target_latent)
                    else:
                        guided = torch.lerp(latent, target_latent, alpha)

                    velocity = (velocity * float(motion_smoothness)) + ((guided - latent) * (1.0 - float(motion_smoothness)))
                    next_latent = latent + velocity

                    if dream_instability > 0.0:
                        noise_scale = dream_instability * (0.004 if structure_anchors else 0.009)
                        next_latent = next_latent + (torch.randn_like(next_latent) * noise_scale)

                    next_latent = torch.nan_to_num(next_latent, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
                    if autoregressive_feedback and feedback_strength > 0.0:
                        next_latent = self._apply_autoregressive_dream_feedback(
                            image,
                            next_latent,
                            resolution=resolution,
                            feedback_strength=feedback_strength,
                            structure_guidance=structure_guidance,
                            has_structure_anchors=bool(structure_anchors),
                        )

                    latent = torch.nan_to_num(next_latent, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)

            final_latent = self._refine_dream_latent(latent).detach().cpu()

            mux_audio_path = None
            if audio_mode == "Use Source Video Audio" and audio_source is not None:
                mux_audio_path = Path(audio_source)
            elif audio_mode == "Procedural Dream Audio":
                mux_audio_path = self._write_procedural_dream_audio(temp_audio_path, seconds=total_frames / float(fps), seed=int(time.time()) % 999999)
            elif audio_mode in AUDIO_MEMORY_DREAM_MODES and audio_source is not None:
                audio_memory_path = output_dir / f"dream_continuation_{timestamp}_audio_memory.wav"
                mux_audio_path = self._render_audio_memory_file(Path(audio_source), audio_memory_path, seconds=total_frames / float(fps), mode=audio_mode)
            output_path, audio_included = self._mux_audio_into_video(video_only_path, output_path, mux_audio_path)
        except Exception as exc:
            self._after(0, lambda msg=str(exc): (setattr(self, "is_dream_video_generating", False), self.status_var.set("Dream video export failed."), messagebox.showerror("Dream Continuation Video", msg)))
            return
        finally:
            if writer is not None:
                writer.release()
            if getattr(self.device, "type", "") == "cuda":
                torch.cuda.empty_cache()

        def _finish() -> None:
            self.is_dream_video_generating = False
            if final_latent is not None:
                self.last_generated_latents = [final_latent]
            if final_image is not None:
                self._display_image(final_image)
                if final_latent is not None:
                    self._remember_generation(final_image, final_latent, mode="dream_continuation_video")
            self.status_var.set(f"Dream video saved: {output_path.name}")
            messagebox.showinfo("Dream Continuation Video", f"Saved dream video to:\n{output_path.resolve()}")

        self._after(0, _finish)

    @classmethod
    def _is_video_path(cls, path: Path) -> bool:
        return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".wmv"}

    @classmethod
    def _is_image_path(cls, path: Path) -> bool:
        return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}

    def _auto_load_model(self):
        models_folder = MODELS_DIR
        if not models_folder.exists():
            messagebox.showerror("Error", f"Model folder not found:\n{models_folder.resolve()}")
            return
        prompt = simpledialog.askstring("Auto Load Model", "Enter a prompt, folder name, or nested category path:", parent=self.root)
        if prompt is None:
            return
        try:
            best_model_path, selection_reason = select_model_path_for_prompt(models_folder, prompt)
            self.generation_prompt_var.set(prompt.strip())
            self._load_model_file(best_model_path)
            self.status_var.set(f"Selected model: {best_model_path.name} ({selection_reason})")
            self._generate_image(self.model)
        except Exception as exc:
            messagebox.showerror("Auto Load Model", str(exc))

    def _compose_scene_prompt(self):
        models_folder = MODELS_DIR
        if not models_folder.exists():
            messagebox.showerror("Error", f"Model folder not found:\n{models_folder.resolve()}")
            return
        prompt = simpledialog.askstring("Compose Scene", "Enter a scene prompt:", parent=self.root)
        if prompt is None:
            return
        try:
            self.generation_prompt_var.set(prompt.strip())
            output_image, output_path, scene = generate_scene_from_prompt(prompt=prompt, models_folder=models_folder, device=self.device, output_dir=OUTPUTS_DIR, target_size=(self.resolution_var.get(), self.resolution_var.get()))
            self._display_image(output_image)
            description = self._describe_scene(scene)
            relation = "layered_depth" if isinstance(scene, LayeredScene) else scene.relation
            self.status_var.set(f"Composed scene: {description} ({relation}) -> {output_path.name}")
        except Exception as exc:
            messagebox.showerror("Compose Scene", str(exc))

    def _merge_models(self):
        models_folder = MODELS_DIR
        if not models_folder.exists():
            messagebox.showerror("Merge Models", f"Model folder not found:\n{models_folder.resolve()}")
            return

        selected = filedialog.askopenfilenames(title="Select model checkpoints to merge", initialdir=str(models_folder.resolve()), filetypes=self.MODEL_FILE_TYPES)
        if not selected:
            return
        checkpoint_paths = [Path(path) for path in selected]
        if len(checkpoint_paths) < 2:
            messagebox.showerror("Merge Models", "Select at least two checkpoints.")
            return

        # Safety Check: Prevent mixing RGB VAE and Wavelet
        checkpoint_modes = []
        for path in checkpoint_paths:
            try:
                checkpoint = safe_torch_load(path, map_location="cpu")
                mode = checkpoint.get("reconstruction_mode", "RGB VAE")
                checkpoint_modes.append(mode)
            except Exception:
                messagebox.showerror("Merge Models", f"Could not read metadata from {path.name}.")
                return
        
        if len(set(checkpoint_modes)) > 1:
            messagebox.showerror("Merge Models", "Cannot merge RGB VAE and Wavelet models because their reconstruction spaces are different.")
            return

        strategy = simpledialog.askstring("Merge Models", "Enter merge strategy: mean, weighted, or slerp", initialvalue="mean", parent=self.root)
        if strategy is None:
            return
        strategy = strategy.strip().lower()
        if strategy not in {"mean", "weighted", "slerp"}:
            messagebox.showerror("Merge Models", "Strategy must be mean, weighted, or slerp.")
            return

        weights: list[float] | None = None
        if strategy == "weighted":
            raw_weights = simpledialog.askstring("Merge Models", "Enter one comma-separated weight per selected checkpoint.", parent=self.root)
            if raw_weights is None:
                return
            try:
                weights = [float(value.strip()) for value in raw_weights.split(",") if value.strip()]
            except ValueError:
                messagebox.showerror("Merge Models", "Weights must be numeric values.")
                return
            if len(weights) != len(checkpoint_paths):
                messagebox.showerror("Merge Models", f"Expected {len(checkpoint_paths)} weights, received {len(weights)}.")
                return

        output_path = models_folder / f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
        try:
            merged_path = merge_checkpoints(checkpoint_paths=checkpoint_paths, output_path=output_path, strategy=strategy, weights=weights)
            if self.model_map_window is not None and self.model_map_window.winfo_exists():
                self._refresh_model_map()
            self.status_var.set(f"Merged {len(checkpoint_paths)} models -> {merged_path.name}")
            messagebox.showinfo("Merge Models", f"Merged checkpoint saved to:\n{merged_path.resolve()}")
        except Exception as exc:
            messagebox.showerror("Merge Models", str(exc))

    def _describe_scene(self, scene: ParsedScene | LayeredScene) -> str:
        if isinstance(scene, LayeredScene):
            labels = []
            for layer_name, scene_object in (("background", scene.background), ("midground", scene.midground), ("frontground", scene.frontground)):
                if scene_object is not None:
                    labels.append(f"{layer_name}:{scene_object.noun}")
            return ", ".join(labels) if labels else scene.prompt
        return " + ".join(obj.noun for obj in scene.objects)

    def _toggle_chaos(self):
        self.personality_var.set("Chaotic")
        self._apply_personality_preset()
        self.var_scale.set(random.uniform(12.0, 20.0))
        self.iterations_var.set(random.randint(0, 5))
        self.blend_count_var.set(random.randint(2, 6))
        self._generate()

    def _generate_image(self, model):
        if model is None:
            raise ValueError("No model is loaded.")
        self._generate()

    def _stop_model_cycle(self):
        self.model_cycle_active = False

    def _shuffle_model_cycle_queue(self):
        self.model_cycle_queue = list(self.model_cycle_paths)
        random.shuffle(self.model_cycle_queue)

    def _toggle_model_cycle(self):
        if self.model_cycle_active:
            self._stop_model_cycle()
            self.status_var.set("Model shuffle stopped.")
            return
        use_folder = messagebox.askyesnocancel("Model Shuffle", "Choose models from a folder?\n\nYes = select a folder\nNo = pick specific model files", parent=self.root)
        if use_folder is None:
            return

        selected_paths: list[Path] = []
        if use_folder:
            folder = filedialog.askdirectory(title="Select folder with model files", parent=self.root)
            if not folder:
                return
            selected_paths = list_model_paths(Path(folder))
        else:
            paths = filedialog.askopenfilenames(title="Select model files", filetypes=self.MODEL_FILE_TYPES, parent=self.root)
            if not paths:
                return
            selected_paths = [Path(p) for p in paths]

        if not selected_paths:
            messagebox.showerror("Model Shuffle", "No model files were found or selected.")
            return
        self.model_cycle_paths = sorted(selected_paths, key=lambda path: path.name.lower())
        self._shuffle_model_cycle_queue()
        self.model_cycle_active = True
        self.auto_cycle_var.set(False)
        self.dream_cycle_var.set(False)
        self.status_var.set(f"Model shuffle started with {len(self.model_cycle_paths)} model(s).")
        self._model_cycle_loop()

    def _model_cycle_loop(self):
        if not self.model_cycle_active:
            return
        if not self.model_cycle_queue:
            if not self.model_cycle_paths:
                self._stop_model_cycle()
                self.status_var.set("Model shuffle stopped: no model files available.")
                return
            self._shuffle_model_cycle_queue()

        model_path = self.model_cycle_queue.pop(0)
        try:
            self._load_model_file(model_path)
            self._generate()
            remaining = len(self.model_cycle_queue)
            self.status_var.set(f"Model shuffle: {model_path.name} | Remaining this round: {remaining}")
        except Exception as exc:
            self.status_var.set(f"Model shuffle skipped {model_path.name}: {exc}")

        if self.model_cycle_active:
            self._after(self.model_cycle_delay_ms, self._model_cycle_loop)

    def _toggle_auto_cycle(self):
        if self.auto_cycle_var.get():
            self._stop_model_cycle()
            self.dream_cycle_var.set(False)
            self._auto_generate_loop()

    def _toggle_dream_cycle(self):
        if self.dream_cycle_var.get():
            self._stop_model_cycle()
            self.auto_cycle_var.set(False)
            self.current_latent = None
            self.target_latent = None
            self.interpolation_step = 0
            self._dream_cycle_loop()

    def _auto_generate_loop(self):
        if not self.auto_cycle_var.get() or self.model is None: return
        self._generate()
        self._after(2000, self._auto_generate_loop)

    def _get_blended_anchor(self, blend_count: int):
        if self.training_tensors is None or self.training_tensors.size(0) == 0:
            return None
        total = self.training_tensors.size(0)
        count = max(2, min(int(blend_count), 64))
        if total >= count:
            idx = torch.randperm(total, device=self.training_tensors.device)[:count]
        else:
            idx = torch.randint(0, total, (count,), device=self.training_tensors.device)
        anchors = self.training_tensors[idx].to(self.device, non_blocking=(getattr(self.device, "type", "") == "cuda"))
        mu, _ = self.model.encode(anchors)
        weights = torch.rand(count, device=mu.device)
        weights = weights / weights.sum()
        return (mu * weights.unsqueeze(1)).sum(dim=0, keepdim=True)

    def _get_memory_anchor(self):
        memories = self.memory_bank.load_memories(limit=12)
        prompt_tag = self.generation_prompt_var.get().strip().lower()
        if prompt_tag:
            tagged = [record for record in memories if prompt_tag and prompt_tag in record.prompt.lower()]
            if tagged:
                memories = tagged
        if not memories:
            return None
        record = random.choice(memories[: max(1, min(6, len(memories)))])
        try:
            return self.memory_bank.load_latent(record).to(self.device)
        except Exception:
            return None

    def _get_random_latent(self):
        intensity = self.var_scale.get() / 10.0
        personality = self.personality_var.get()
        blend_enabled = bool(self.blend_mode_var.get())
        blend_count = max(2, int(self.blend_count_var.get()))
        with torch.no_grad():
            if personality in {"Dreamy", "Hybrid"}:
                memory_anchor = self._get_memory_anchor()
                if memory_anchor is not None:
                    noise = torch.randn_like(memory_anchor, device=memory_anchor.device)
                    return memory_anchor + (noise * max(0.05, intensity * 0.65))
            if blend_enabled:
                blended = self._get_blended_anchor(blend_count)
                if blended is not None:
                    noise = torch.randn_like(blended, device=blended.device)
                    return blended + (intensity * noise)
            if self.training_tensors is not None and self.training_tensors.size(0) > 0:
                total_items = self.training_tensors.size(0)
                if personality == "Nostalgic" and total_items > 1:
                    nostalgic_span = max(1, total_items // 3)
                    idx = torch.randint(0, nostalgic_span, (1,), device=self.training_tensors.device)
                else:
                    idx = torch.randint(0, total_items, (1,), device=self.training_tensors.device)
                img_batch = self.training_tensors[idx].to(self.device, non_blocking=(getattr(self.device, "type", "") == "cuda"))
                mu, _ = self.model.encode(img_batch)
                noise = torch.randn_like(mu, device=mu.device)
                if personality == "Chaotic":
                    noise = noise * 1.6
                elif personality == "Dreamy":
                    noise = noise * 0.6
                return mu + (intensity * noise)
            else:
                if personality == "Corruption":
                    intensity *= 1.4
                return torch.randn(1, self.model.latent_dim, device=self.device) * intensity

    def _compose_labeled_grid(self, images, columns: int = 3):
        if not images:
            raise ValueError("No images to compose.")
        width, height = images[0].size
        columns = max(1, columns)
        rows = math.ceil(len(images) / columns)
        pad = 12
        label_h = 28
        canvas = Image.new("RGB", (columns * width + (columns + 1) * pad, rows * (height + label_h) + (rows + 1) * pad), color=(8, 10, 18))
        draw = ImageDraw.Draw(canvas)
        for idx, image in enumerate(images, start=1):
            row = (idx - 1) // columns
            col = (idx - 1) % columns
            x = pad + col * (width + pad)
            y = pad + row * (height + label_h + pad)
            img = image.resize((width, height), Image.Resampling.LANCZOS) if image.size != (width, height) else image
            canvas.paste(img, (x, y))
            draw.text((x, y + height + 6), f"#{idx}", fill=(235, 240, 255))
        return canvas

    def _render_latent_gallery(self, latents, *, mode: str, numbered: bool = False, status_label: str = "", save_memory: bool = True):
        images = []
        self.last_generated_latents = []

        with torch.no_grad():
            for latent in latents:
                latent_device = latent.to(self.device)
                recon = self._decode_latent(latent_device, show_steps=(len(latents) == 1 and mode == "generate" and self.show_iterations_var.get()))
                image = tensor_to_pil(self._decode_model_output_to_rgb(recon))
                images.append(image)
                stored_latent = latent_device.detach().cpu()
                self.last_generated_latents.append(stored_latent)
                if save_memory:
                    self._remember_generation(image, stored_latent, mode=mode, extra={"diffusion_steps": int(self.diffusion_steps_var.get()), "diffusion_strength": float(self.diffusion_strength_var.get()), "iterations": int(self.iterations_var.get())})

        if len(images) == 1:
            final_image = images[0]
        elif numbered:
            final_image = self._compose_labeled_grid(images)
        else:
            final_image = self._compose_side_by_side(images)

        self._display_image(final_image)
        if status_label:
            self.status_var.set(status_label)
        return images

    def _dream_cycle_loop(self):
        if not self.dream_cycle_var.get() or self.model is None: return
        if self.current_latent is None:
            self.current_latent = self._get_random_latent()
            self.target_latent = self._get_random_latent()
            self.interpolation_step = 0
        if self.interpolation_step >= self.total_interpolation_steps:
            self.current_latent = self.target_latent
            self.target_latent = self._get_random_latent()
            self.interpolation_step = 0
        self.total_interpolation_steps = int(self.speed_scale.get())
        alpha = self.interpolation_step / self.total_interpolation_steps
        with torch.no_grad():
            interp_latent = slerp(alpha, self.current_latent, self.target_latent)
            recon = self._decode_latent(interp_latent, show_steps=False)
            image = tensor_to_pil(self._decode_model_output_to_rgb(recon))
            self._display_image(image)
            if self.interpolation_step == 0:
                self.last_generated_latents = [interp_latent.detach().cpu()]
                self._remember_generation(image, interp_latent.detach().cpu(), mode="dream_journal")

        self.interpolation_step += 1
        self.status_var.set("Dreaming Cycle active...")
        fps = max(1, int(self.dream_fps_var.get()))
        delay_ms = max(16, int(1000 / fps))
        self._after(delay_ms, self._dream_cycle_loop)

    def _generate(self):
        if self.model is None:
            messagebox.showerror("Error", "Load/Train model first.")
            return

        intensity = self.var_scale.get()
        iterations = max(0, int(self.iterations_var.get()))
        show_steps = self.show_iterations_var.get()
        output_count = max(1, min(8, int(self.output_count_var.get())))
        blend_enabled = bool(self.blend_mode_var.get())
        blend_count = max(2, int(self.blend_count_var.get()))
        use_diffusion = bool(self.use_mini_diffusion_var.get())
        diffusion_steps = max(1, int(self.diffusion_steps_var.get()))
        latents = []
        
        with torch.no_grad():
            for _ in range(output_count):
                latents.append(self._get_random_latent())

        self._render_latent_gallery(latents, mode=self._current_mode_label(), numbered=False, status_label="", save_memory=not self.auto_cycle_var.get())
        if not self.auto_cycle_var.get() and not self.dream_cycle_var.get():
            mode = f"Blend x{blend_count}" if blend_enabled else "Single anchor"
            self.status_var.set(f"Generated ({self.personality_var.get()} | Intensity: {intensity:.1f} | {mode} | Diffusion: {'on' if use_diffusion else 'off'} x{diffusion_steps} | Outputs: {output_count})")

    def _decode_latent(self, z, show_steps: bool = False):
        current = z
        if self.use_latent_diffusion_var.get() and self.latent_diffusion is not None:
            diffusion = self._ensure_latent_diffusion()
            current = diffusion.polish_latent(current, strength=float(self.latent_diffusion_strength_var.get()))
        if self.use_mini_diffusion_var.get():
            current = self._mini_diffusion_refine(current, show_steps=show_steps)

        recon = self.model.decode(current)
        iterations = max(0, int(self.iterations_var.get()))
        for _step in range(iterations):
            mu_step, _ = self.model.encode(recon)
            recon = self.model.decode(mu_step)
            if show_steps:
                self._display_image(tensor_to_pil(self._decode_model_output_to_rgb(recon)))
                self.root.update()
                self._after(50)
        return recon

    def _mini_diffusion_refine(self, z, show_steps: bool = False):
        steps = max(1, int(self.diffusion_steps_var.get()))
        strength = max(0.05, float(self.diffusion_strength_var.get()))
        intensity_scale = max(0.2, min(2.0, self.var_scale.get() / 10.0))
        current = torch.nan_to_num(z.clone(), nan=0.0, posinf=4.0, neginf=-4.0)

        for step_idx in range(steps):
            if steps == 1:
                t_value = 1.0
            else:
                t_value = 1.0 - (step_idx / (steps - 1))
            t = torch.full((current.size(0), 1), t_value, device=current.device)
            predicted_noise = torch.nan_to_num(self.model.predict_latent_noise(current, t), nan=0.0, posinf=0.0, neginf=0.0)
            step_scale = strength * intensity_scale * (0.2 + 0.8 * t_value)
            current = current - (predicted_noise * step_scale)
            if step_idx < steps - 1:
                residual_scale = 0.03 * intensity_scale * t_value
                current = current + (torch.randn_like(current) * residual_scale)
            current = torch.nan_to_num(current, nan=0.0, posinf=4.0, neginf=-4.0)

            if show_steps:
                preview = self.model.decode(current)
                self._display_image(tensor_to_pil(self._decode_model_output_to_rgb(preview)))
                self.root.update()
                self._after(50)

        return current

    def _compose_side_by_side(self, images):
        if not images:
            raise ValueError("No images to compose.")
        width, height = images[0].size
        canvas = Image.new("RGB", (width * len(images), height), color=(8, 8, 16))
        x_offset = 0
        for image in images:
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.LANCZOS)
            canvas.paste(image, (x_offset, 0))
            x_offset += width
        return canvas

    def _display_image(self, pil_img):
        from PIL import ImageTk, Image
        if self._closing:
            return

        source_img = pil_img.convert("RGB")

        # Main in-app preview stays 512x512 so the normal APVD window layout does
        # not explode. It does not upscale old 256 outputs; it centers them.
        preview_img = source_img.copy()
        preview_img.thumbnail((512, 512), Image.Resampling.LANCZOS)
        preview_fitted = Image.new("RGB", (512, 512), color=self._hex_to_rgb(self._theme_palette["canvas"]))
        preview_fitted.paste(preview_img, ((512 - preview_img.width) // 2, (512 - preview_img.height) // 2))
        self.photo = ImageTk.PhotoImage(preview_fitted, master=self.root)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        ow = self.output_canvas
        if ow is not None:
            try:
                win = self.output_window
                if win is not None and win.winfo_exists():
                    canvas_w, canvas_h = self._resize_output_window(source_img.size)
                    output_img = source_img.copy()
                    output_img.thumbnail((canvas_w, canvas_h), Image.Resampling.LANCZOS)
                    output_fitted = Image.new("RGB", (canvas_w, canvas_h), color=self._hex_to_rgb(self._theme_palette["canvas"]))
                    output_fitted.paste(output_img, ((canvas_w - output_img.width) // 2, (canvas_h - output_img.height) // 2))
                    self._output_photo = ImageTk.PhotoImage(output_fitted, master=win)
                    ow.delete("all")
                    ow.create_image(0, 0, anchor=tk.NW, image=self._output_photo)
            except tk.TclError:
                self.output_window = None
                self.output_canvas = None
                self._output_photo = None

def main():
    app = APVDApp()
    app.root.mainloop()

if __name__ == "__main__":
    main()