"""
Latent-space diffusion for APVD.

This module does not change APVD's encoder or decoder. It trains a small
diffusion denoiser on APVD latent vectors and then uses APVD's existing decoder
to turn denoised latents back into images.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class DiffusionConfig:
    """Configuration for a lightweight DDPM in APVD latent space."""

    latent_dim: int = 256
    timesteps: int = 75
    hidden_dim: int = 512
    time_embed_dim: int = 128
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "cosine"


def _make_beta_schedule(config: DiffusionConfig) -> Tensor:
    if config.timesteps < 2:
        raise ValueError("timesteps must be at least 2")

    if config.schedule == "linear":
        return torch.linspace(config.beta_start, config.beta_end, config.timesteps)

    if config.schedule == "cosine":
        # Cosine schedules are stable with only 50-100 latent diffusion steps.
        s = 0.008
        steps = config.timesteps + 1
        x = torch.linspace(0, config.timesteps, steps, dtype=torch.float64)
        alphas_cumprod = torch.cos(((x / config.timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-5, 0.999).float()

    raise ValueError("schedule must be 'linear' or 'cosine'")


def _ensure_latent_batch(latent: Tensor) -> Tensor:
    if latent.ndim == 1:
        latent = latent.unsqueeze(0)
    if latent.ndim > 2:
        latent = latent.reshape(latent.size(0), -1)
    if latent.ndim != 2:
        raise ValueError(f"Expected latent shape [B, D] or [D], got {tuple(latent.shape)}")
    return latent


def _timestep_tensor(timestep: int | Tensor, batch_size: int, device: torch.device) -> Tensor:
    if isinstance(timestep, Tensor):
        t = timestep.to(device=device, dtype=torch.long)
        if t.ndim == 0:
            t = t.repeat(batch_size)
        else:
            t = t.reshape(-1)
        if t.numel() != batch_size:
            raise ValueError(f"Expected {batch_size} timesteps, got {t.numel()}")
        return t
    return torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)


class NoiseScheduler(nn.Module):
    """Forward q(x_t | x_0) and reverse p(x_{t-1} | x_t) math for latent DDPM."""

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        betas = _make_beta_schedule(config)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.ones(1), alphas_cumprod[:-1]), dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance.clamp(min=1e-20))

    @property
    def timesteps(self) -> int:
        return self.config.timesteps

    def _extract(self, values: Tensor, timestep: Tensor, target_shape: torch.Size) -> Tensor:
        timestep = timestep.to(device=values.device, dtype=torch.long).clamp(0, self.timesteps - 1)
        out = values.gather(0, timestep)
        while out.ndim < len(target_shape):
            out = out.unsqueeze(-1)
        return out

    def add_noise(self, clean_latent: Tensor, timestep: Tensor, noise: Tensor | None = None) -> Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) eps."""
        clean_latent = _ensure_latent_batch(clean_latent)
        if noise is None:
            noise = torch.randn_like(clean_latent)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timestep, clean_latent.shape)
        sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, timestep, clean_latent.shape)
        return (sqrt_alpha * clean_latent) + (sqrt_one_minus_alpha * noise)

    def step(
        self,
        predicted_noise: Tensor,
        timestep: Tensor,
        noisy_latent: Tensor,
        *,
        noise: Tensor | None = None,
    ) -> Tensor:
        """Reverse diffusion step: predict x_{t-1} from x_t and predicted noise."""
        noisy_latent = _ensure_latent_batch(noisy_latent)
        predicted_noise = _ensure_latent_batch(predicted_noise)
        if noise is None:
            noise = torch.randn_like(noisy_latent)

        beta_t = self._extract(self.betas, timestep, noisy_latent.shape)
        sqrt_one_minus_alpha_bar_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod,
            timestep,
            noisy_latent.shape,
        )
        sqrt_recip_alpha_t = self._extract(self.sqrt_recip_alphas, timestep, noisy_latent.shape)

        model_mean = sqrt_recip_alpha_t * (
            noisy_latent - (beta_t / sqrt_one_minus_alpha_bar_t) * predicted_noise
        )
        posterior_variance_t = self._extract(self.posterior_variance, timestep, noisy_latent.shape)
        nonzero_mask = (timestep != 0).float().reshape(noisy_latent.size(0), *([1] * (noisy_latent.ndim - 1)))
        return model_mean + nonzero_mask * torch.sqrt(posterior_variance_t) * noise


class SinusoidalTimeEmbedding(nn.Module):
    """Stable timestep features for the MLP denoiser."""

    def __init__(self, dim: int, max_period: int = 10_000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timestep: Tensor) -> Tensor:
        timestep = timestep.reshape(-1).float()
        half = self.dim // 2
        frequencies = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=timestep.device, dtype=torch.float32)
            / max(1, half - 1)
        )
        args = timestep.unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=1)
        if self.dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ResidualMLPBlock(nn.Module):
    """A small residual block keeps the denoiser lightweight but trainable."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.net(x)


class MLPDenoiser(nn.Module):
    """Predicts Gaussian noise from a noisy APVD latent and a timestep."""

    def __init__(self, latent_dim: int, hidden_dim: int, time_embed_dim: int):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(latent_dim + time_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            ResidualMLPBlock(hidden_dim),
            ResidualMLPBlock(hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, noisy_latent: Tensor, timestep: Tensor) -> Tensor:
        noisy_latent = _ensure_latent_batch(noisy_latent)
        time_embedding = self.time_mlp(timestep)
        return self.net(torch.cat((noisy_latent, time_embedding), dim=1))


class DiffusionModel(nn.Module):
    """A compact latent DDPM for APVD's 256-D latent vectors."""

    def __init__(self, config: DiffusionConfig | None = None):
        super().__init__()
        self.config = config or DiffusionConfig()
        self.scheduler = NoiseScheduler(self.config)
        self.denoiser = MLPDenoiser(
            latent_dim=self.config.latent_dim,
            hidden_dim=self.config.hidden_dim,
            time_embed_dim=self.config.time_embed_dim,
        )

    def forward(self, noisy_latent: Tensor, timestep: int | Tensor) -> Tensor:
        noisy_latent = _ensure_latent_batch(noisy_latent)
        t = _timestep_tensor(timestep, noisy_latent.size(0), noisy_latent.device)
        return self.denoiser(noisy_latent, t)

    def training_loss(self, clean_latent: Tensor) -> Tensor:
        """MSE between predicted noise and the actual noise added by q(x_t | x_0)."""
        clean_latent = _ensure_latent_batch(clean_latent).float()
        if clean_latent.size(1) != self.config.latent_dim:
            raise ValueError(
                f"Diffusion latent_dim={self.config.latent_dim}, got latent size {clean_latent.size(1)}"
            )
        timestep = torch.randint(
            0,
            self.config.timesteps,
            (clean_latent.size(0),),
            device=clean_latent.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(clean_latent)
        noisy_latent = self.scheduler.add_noise(clean_latent, timestep, noise)
        predicted_noise = self(noisy_latent, timestep)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def denoise_latent(
        self,
        noisy_latent: Tensor,
        *,
        start_timestep: int | None = None,
        deterministic: bool = False,
        return_all_steps: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Run p(x_{t-1} | x_t) until t=0."""
        current = _ensure_latent_batch(noisy_latent).float()
        start = self.config.timesteps - 1 if start_timestep is None else int(start_timestep)
        start = max(0, min(start, self.config.timesteps - 1))
        steps: list[Tensor] = []

        for step_idx in range(start, -1, -1):
            timestep = torch.full((current.size(0),), step_idx, device=current.device, dtype=torch.long)
            predicted_noise = self(current, timestep)
            step_noise = torch.zeros_like(current) if deterministic else None
            current = self.scheduler.step(predicted_noise, timestep, current, noise=step_noise)
            if return_all_steps:
                steps.append(current.detach().clone())

        if return_all_steps:
            return current, steps
        return current

    @torch.no_grad()
    def polish_latent(
        self,
        clean_latent: Tensor,
        *,
        strength: float = 0.25,
        deterministic: bool = False,
        return_all_steps: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Image-to-image style refine: add light noise, then denoise it back."""
        clean_latent = _ensure_latent_batch(clean_latent).float()
        strength = max(0.0, min(1.0, float(strength)))
        start = int(round((self.config.timesteps - 1) * strength))
        if start <= 0:
            if return_all_steps:
                return clean_latent, [clean_latent.detach().clone()]
            return clean_latent

        timestep = torch.full((clean_latent.size(0),), start, device=clean_latent.device, dtype=torch.long)
        noised = self.scheduler.add_noise(clean_latent, timestep)
        return self.denoise_latent(
            noised,
            start_timestep=start,
            deterministic=deterministic,
            return_all_steps=return_all_steps,
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        *,
        device: torch.device | str | None = None,
        deterministic: bool = False,
        return_all_steps: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Generate new APVD latents by denoising pure Gaussian noise."""
        model_device = next(self.parameters()).device
        sample_device = torch.device(device) if device is not None else model_device
        latent = torch.randn(max(1, int(batch_size)), self.config.latent_dim, device=sample_device)
        return self.denoise_latent(latent, deterministic=deterministic, return_all_steps=return_all_steps)

    @torch.no_grad()
    def conditional_denoise(
        self,
        noisy_latent: Tensor,
        class_labels: Tensor,
        *,
        start_timestep: int | None = None,
        deterministic: bool = False,
        return_all_steps: bool = False,
    ) -> Tensor | tuple[Tensor, list[Tensor]]:
        """Conditional denoising using class labels."""
        current = _ensure_latent_batch(noisy_latent).float()
        start = self.config.timesteps - 1 if start_timestep is None else int(start_timestep)
        start = max(0, min(start, self.config.timesteps - 1))
        steps: list[Tensor] = []

        for step_idx in range(start, -1, -1):
            timestep = torch.full((current.size(0),), step_idx, device=current.device, dtype=torch.long)
            # For conditional denoising, we would modify the denoiser to accept class_labels
            # This is a placeholder implementation that just calls the regular denoising
            predicted_noise = self(current, timestep)
            step_noise = torch.zeros_like(current) if deterministic else None
            current = self.scheduler.step(predicted_noise, timestep, current, noise=step_noise)
            if return_all_steps:
                steps.append(current.detach().clone())

        if return_all_steps:
            return current, steps
        return current

    def save(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Save the diffusion model without bundling APVD encoder/decoder weights."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, Any] = {
            "version": 1,
            "config": asdict(self.config),
            "model_state_dict": self.state_dict(),
            "metadata": metadata or {},
        }
        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        torch.save(checkpoint, output_path)
        return output_path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> tuple["DiffusionModel", dict[str, Any]]:
        """Load a saved latent diffusion checkpoint."""
        map_location = torch.device(device) if device is not None else "cpu"
        checkpoint = torch.load(path, map_location=map_location)
        config = DiffusionConfig(**checkpoint.get("config", {}))
        model = cls(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        if device is not None:
            model = model.to(torch.device(device))
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return model, checkpoint


def build_default_diffusion_for_apvd(apvd_model: nn.Module, *, timesteps: int = 75) -> DiffusionModel:
    """Create a diffusion model whose latent size matches the loaded APVD model."""
    latent_dim = int(getattr(apvd_model, "latent_dim", 256))
    return DiffusionModel(DiffusionConfig(latent_dim=latent_dim, timesteps=timesteps))


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def encode_apvd_latent(apvd_model: nn.Module, images: Tensor, *, sample_latent: bool = False) -> Tensor:
    """Call APVD's encoder API and normalize the result to [B, latent_dim]."""
    if hasattr(apvd_model, "encode") and callable(getattr(apvd_model, "encode")):
        encoded = apvd_model.encode(images)
    elif hasattr(apvd_model, "encoder") and callable(getattr(apvd_model, "encoder")):
        encoded = apvd_model.encoder(images)
    else:
        raise AttributeError("APVD model must expose encode(images) or encoder(images)")

    if isinstance(encoded, tuple):
        mu = encoded[0]
        if sample_latent and len(encoded) > 1 and hasattr(apvd_model, "reparameterize"):
            return _ensure_latent_batch(apvd_model.reparameterize(mu, encoded[1]))
        return _ensure_latent_batch(mu)
    return _ensure_latent_batch(encoded)


def decode_apvd_latent(apvd_model: nn.Module, latent: Tensor) -> Tensor:
    """Call APVD's decoder API."""
    latent = _ensure_latent_batch(latent)
    if hasattr(apvd_model, "decode") and callable(getattr(apvd_model, "decode")):
        return apvd_model.decode(latent)
    if hasattr(apvd_model, "decoder") and callable(getattr(apvd_model, "decoder")):
        return apvd_model.decoder(latent)
    raise AttributeError("APVD model must expose decode(latent) or decoder(latent)")


def _extract_image_tensor(batch: Any) -> Tensor:
    if isinstance(batch, Tensor):
        return batch
    if isinstance(batch, dict):
        for key in ("image", "images", "x", "input", "pixel_values"):
            value = batch.get(key)
            if isinstance(value, Tensor):
                return value
    if isinstance(batch, (tuple, list)):
        for value in batch:
            if isinstance(value, Tensor):
                return value
    raise TypeError("Dataloader batch must contain an image tensor")


def train_diffusion_model(
    apvd_model: nn.Module,
    diffusion_model: DiffusionModel,
    data_loader: torch.utils.data.DataLoader,
    *,
    epochs: int = 10,
    optimizer: torch.optim.Optimizer | None = None,
    lr: float = 1e-4,
    device: torch.device | str | None = None,
    grad_clip: float | None = 1.0,
    sample_latent: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """
    Train diffusion on latents from APVD's encoder.

    Loop:
    1. image -> APVD latent
    2. random timestep noise is added by q(x_t | x_0)
    3. denoiser predicts the noise
    4. MSE loss is backpropagated into diffusion only
    """
    train_device = torch.device(device) if device is not None else _module_device(apvd_model)
    apvd_model = apvd_model.to(train_device)
    diffusion_model = diffusion_model.to(train_device)
    optimizer = optimizer or torch.optim.AdamW(diffusion_model.parameters(), lr=lr)

    original_requires_grad = [param.requires_grad for param in apvd_model.parameters()]
    for param in apvd_model.parameters():
        param.requires_grad_(False)

    apvd_model.eval()
    diffusion_model.train()
    epoch_losses: list[float] = []
    global_step = 0

    try:
        for epoch in range(max(1, int(epochs))):
            running = 0.0
            count = 0
            for batch in data_loader:
                images = _extract_image_tensor(batch).to(train_device, non_blocking=True).float()
                with torch.no_grad():
                    clean_latent = encode_apvd_latent(
                        apvd_model,
                        images,
                        sample_latent=sample_latent,
                    ).detach()

                optimizer.zero_grad(set_to_none=True)
                loss = diffusion_model.training_loss(clean_latent)
                loss.backward()
                if grad_clip is not None and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), float(grad_clip))
                optimizer.step()

                global_step += 1
                count += 1
                running += float(loss.detach().cpu())
                if progress_callback is not None:
                    progress_callback(
                        {
                            "epoch": epoch + 1,
                            "epochs": max(1, int(epochs)),
                            "step": global_step,
                            "loss": float(loss.detach().cpu()),
                        }
                    )

            epoch_losses.append(running / max(1, count))
    finally:
        for param, requires_grad in zip(apvd_model.parameters(), original_requires_grad):
            param.requires_grad_(requires_grad)

    diffusion_model.eval()
    return {
        "epoch_losses": epoch_losses,
        "final_loss": epoch_losses[-1] if epoch_losses else None,
        "steps": global_step,
    }


@torch.no_grad()
def diffusion_denoise(
    diffusion_model: DiffusionModel,
    latent: Tensor,
    *,
    strength: float | None = None,
    deterministic: bool = False,
) -> Tensor:
    """Convenience wrapper used by the generation modes below."""
    if strength is None:
        return diffusion_model.denoise_latent(latent, deterministic=deterministic)
    return diffusion_model.polish_latent(latent, strength=strength, deterministic=deterministic)


@torch.no_grad()
def apvd_reconstruction(
    apvd_model: nn.Module,
    image: Tensor,
    *,
    device: torch.device | str | None = None,
    sample_latent: bool = False,
) -> tuple[Tensor, Tensor]:
    """Mode 1: latent = encoder(image); output = decoder(latent)."""
    run_device = torch.device(device) if device is not None else _module_device(apvd_model)
    apvd_model = apvd_model.to(run_device).eval()
    image = image.to(run_device).float()
    latent = encode_apvd_latent(apvd_model, image, sample_latent=sample_latent)
    output = decode_apvd_latent(apvd_model, latent)
    return output, latent


@torch.no_grad()
def apvd_diffusion_polish(
    apvd_model: nn.Module,
    diffusion_model: DiffusionModel,
    image: Tensor,
    *,
    strength: float = 0.25,
    device: torch.device | str | None = None,
    deterministic: bool = False,
    sample_latent: bool = False,
) -> tuple[Tensor, Tensor]:
    """Mode 2: encode image, polish the latent with diffusion, then decode."""
    run_device = torch.device(device) if device is not None else _module_device(apvd_model)
    apvd_model = apvd_model.to(run_device).eval()
    diffusion_model = diffusion_model.to(run_device).eval()
    image = image.to(run_device).float()
    latent = encode_apvd_latent(apvd_model, image, sample_latent=sample_latent)
    polished_latent = diffusion_model.polish_latent(latent, strength=strength, deterministic=deterministic)
    output = decode_apvd_latent(apvd_model, polished_latent)
    return output, polished_latent


@torch.no_grad()
def pure_diffusion_generation(
    apvd_model: nn.Module,
    diffusion_model: DiffusionModel,
    *,
    batch_size: int = 1,
    device: torch.device | str | None = None,
    deterministic: bool = False,
) -> tuple[Tensor, Tensor]:
    """Mode 3: generate a latent from pure noise with diffusion, then decode it."""
    run_device = torch.device(device) if device is not None else _module_device(apvd_model)
    apvd_model = apvd_model.to(run_device).eval()
    diffusion_model = diffusion_model.to(run_device).eval()
    latent = diffusion_model.sample(batch_size=batch_size, device=run_device, deterministic=deterministic)
    output = decode_apvd_latent(apvd_model, latent)
    return output, latent


@torch.no_grad()
def interpolate_latents_with_diffusion(
    diffusion_model: DiffusionModel,
    latent_a: Tensor,
    latent_b: Tensor,
    mix: float,
    *,
    strength: float = 0.15,
    deterministic: bool = False,
) -> Tensor:
    """Optional mode: lerp two APVD latents and diffusion-polish the result."""
    latent_a = _ensure_latent_batch(latent_a)
    latent_b = _ensure_latent_batch(latent_b).to(latent_a.device)
    mix = max(0.0, min(1.0, float(mix)))
    blended = (latent_a * (1.0 - mix)) + (latent_b * mix)
    return diffusion_model.polish_latent(blended, strength=strength, deterministic=deterministic)


def example_usage() -> None:
    """
    Minimal integration sketch.

    This is intentionally not executed automatically because it needs your APVD
    checkpoint, dataloader, and image tensors.
    """
    from model import VAE, get_device

    device = get_device()
    apvd = VAE(latent_dim=256).to(device)
    diffusion = build_default_diffusion_for_apvd(apvd, timesteps=75).to(device)

    # Train:
    # stats = train_diffusion_model(apvd, diffusion, train_loader, epochs=10, device=device)
    # diffusion.save("Models/apvd_latent_diffusion.pt", metadata=stats)

    # Load later:
    # diffusion, _ = DiffusionModel.load("Models/apvd_latent_diffusion.pt", device=device)

    # Generate:
    # recon_img, recon_latent = apvd_reconstruction(apvd, image_tensor, device=device)
    # polished_img, polished_latent = apvd_diffusion_polish(apvd, diffusion, image_tensor, device=device)
    # new_img, new_latent = pure_diffusion_generation(apvd, diffusion, batch_size=1, device=device)
    _ = (apvd, diffusion)
