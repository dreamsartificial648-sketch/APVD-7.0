"""
APVD v2 - VAE model with a lightweight latent denoiser.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LatentDenoiser(nn.Module):
    """Small MLP that predicts latent noise from a noisy latent and a timestep."""

    def __init__(self, latent_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t.unsqueeze(1)
        t = t.clamp(0.0, 1.0)
        t_features = torch.cat(
            (
                t,
                torch.sin(t * math.pi),
                torch.cos(t * math.pi),
            ),
            dim=1,
        )
        return self.net(torch.cat((z, t_features), dim=1))


class PatternDecayDenoiser(nn.Module):
    """Tiny image-space denoiser for APVD Pattern Decay Reconstruction.

    It is deliberately much smaller than a DDPM UNet. APVD still does the
    memory reconstruction; this module only learns how corrupted image fields
    tend to decay back toward the dataset's visual patterns.
    """

    def __init__(self, channels: int = 3, hidden_dim: int = 48):
        super().__init__()
        channels = int(channels)
        hidden_dim = int(hidden_dim)
        self.channels = channels
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Conv2d(channels + 1, hidden_dim, kernel_size=3, padding=1)
        self.down1 = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)
        self.mid1 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)
        self.mid2 = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1)
        self.up1 = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.out_proj = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)

    def forward(self, noisy: Tensor, t: Tensor) -> Tensor:
        if t.ndim == 1:
            t = t.reshape(-1, 1, 1, 1)
        elif t.ndim == 2:
            t = t.reshape(t.size(0), 1, 1, 1)
        t = t.to(device=noisy.device, dtype=noisy.dtype).clamp(0.0, 1.0)
        t_map = t.expand(noisy.size(0), 1, noisy.size(-2), noisy.size(-1))
        h0 = F.silu(self.in_proj(torch.cat((noisy, t_map), dim=1)))
        h1 = F.silu(self.down1(h0))
        h1 = F.silu(self.mid1(h1))
        h1 = F.silu(self.mid2(h1))
        up = F.silu(self.up1(h1))
        if up.shape[-2:] != h0.shape[-2:]:
            up = F.interpolate(up, size=h0.shape[-2:], mode="bilinear", align_corners=False)
        residual = torch.tanh(self.out_proj(up + h0))
        # Residual is scaled by corruption level: early noisy steps can move
        # more, final steps make small corrections instead of repainting.
        return noisy + residual * (0.05 + 0.95 * t)


class HashGridAssist(nn.Module):
    """Optional learned multi-resolution-ish texture bias for APVD's decoder.

    This is intentionally tiny and optional. When disabled, the VAE architecture
    and checkpoint format stay the same as older APVD models. When enabled, the
    grid starts at zero so a standard checkpoint can be upgraded with
    load_state_dict(..., strict=False) and then fine-tuned.
    """

    def __init__(self, out_channels: int, grid_size: int = 32, strength: float = 0.08):
        super().__init__()
        grid_size = max(4, min(128, int(grid_size)))
        self.out_channels = int(out_channels)
        self.grid_size = int(grid_size)
        self.strength = float(max(0.0, min(1.0, strength)))
        self.grid = nn.Parameter(torch.zeros(1, self.out_channels, self.grid_size, self.grid_size))
        # Starts slightly open, but the zero grid means there is no image change
        # until training learns useful detail.
        self.gate = nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        grid = F.interpolate(self.grid, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gate = torch.sigmoid(self.gate).to(dtype=x.dtype)
        return x + (grid.to(dtype=x.dtype) * gate * self.strength)


class VAE(nn.Module):
    """Convolutional VAE with configurable output resolution and latent denoiser."""

    def __init__(
        self,
        latent_dim: int = 256,
        in_channels: int = 3,
        out_channels: int = 3,
        output_size: tuple[int, int] = (256, 256),
        output_activation: str = "sigmoid",
        decoder_mode: str = "Standard Decoder",
        hash_grid_size: int = 32,
        hash_grid_strength: float = 0.08,
        pattern_decay_hidden: int = 48,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_size = output_size
        self.output_activation = output_activation
        self.decoder_mode = "Hash Grid Assist" if decoder_mode == "Hash Grid Assist" else "Standard Decoder"
        self.hash_grid_size = max(4, min(128, int(hash_grid_size)))
        self.hash_grid_strength = float(max(0.0, min(1.0, hash_grid_strength)))
        self.pattern_decay_hidden = max(16, min(128, int(pattern_decay_hidden)))

        # Downsample while preserving flexibility for different input resolutions.
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
        )

        # Normalize encoder output to a fixed latent projection shape.
        self.enc_dim = 4 * 4 * 512

        self.fc_mu = nn.Linear(self.enc_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.enc_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.enc_dim)
        self.latent_denoiser = LatentDenoiser(latent_dim)

        # Decoder base output is 128x128; final interpolate reaches requested output size.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
        )
        self.hash_grid_assist = (
            HashGridAssist(out_channels, grid_size=self.hash_grid_size, strength=self.hash_grid_strength)
            if self.decoder_mode == "Hash Grid Assist"
            else None
        )
        self.pattern_decay_denoiser = PatternDecayDenoiser(out_channels, hidden_dim=self.pattern_decay_hidden)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        h = self.encoder(x)
        h = F.adaptive_avg_pool2d(h, output_size=(4, 4))
        h = h.reshape(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        logvar = torch.nan_to_num(logvar, nan=0.0, posinf=12.0, neginf=-12.0).clamp(-12.0, 12.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std, device=mu.device)
        return mu + eps * std

    def decode(self, z: Tensor) -> Tensor:
        h = self.fc_decode(z)
        h = h.reshape(h.size(0), 512, 4, 4)
        x = self.decoder(h)
        x = F.interpolate(x, size=self.output_size, mode="bilinear", align_corners=False)
        if self.hash_grid_assist is not None:
            x = self.hash_grid_assist(x)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(x)
        if self.output_activation == "tanh":
            return torch.tanh(x)
        return x

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def predict_latent_noise(self, z: Tensor, t: Tensor) -> Tensor:
        return self.latent_denoiser(z, t)

    def pattern_decay_reconstruct(self, noisy: Tensor, t: Tensor) -> Tensor:
        return self.pattern_decay_denoiser(noisy, t)


def vae_loss(
    recon: Tensor,
    x: Tensor,
    mu: Tensor,
    logvar: Tensor,
    reconstruction_loss: str = "bce",
) -> Tensor:
    """VAE loss = reconstruction loss + KL divergence.

    RGB APVD uses BCE on normalized pixels. Wavelet APVD should use MSE because
    Haar coefficients can be negative and the LL band can exceed 1.0.
    """
    mu = torch.nan_to_num(mu, nan=0.0, posinf=1e4, neginf=-1e4)
    logvar = torch.nan_to_num(logvar, nan=0.0, posinf=12.0, neginf=-12.0).clamp(-12.0, 12.0)

    if reconstruction_loss == "mse":
        recon_safe = torch.nan_to_num(recon, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
        x_safe = torch.nan_to_num(x, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
        recon_term = F.mse_loss(recon_safe, x_safe, reduction="sum")
    else:
        eps = 1e-6
        recon_safe = torch.nan_to_num(recon, nan=0.5, posinf=1.0, neginf=0.0).clamp(eps, 1.0 - eps)
        x_safe = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
        recon_term = F.binary_cross_entropy(recon_safe, x_safe, reduction="sum")

    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_term + kl


def latent_denoiser_loss(
    model: VAE,
    clean_latents: Tensor,
    max_noise_scale: float = 2.0,
) -> Tensor:
    """Train the latent denoiser to predict injected Gaussian noise."""
    batch = clean_latents.size(0)
    t = torch.rand(batch, 1, device=clean_latents.device)
    noise = torch.randn_like(clean_latents)
    noise_scale = 1.0 + (max_noise_scale - 1.0) * t
    noisy_latents = clean_latents + (noise * noise_scale)
    predicted_noise = model.predict_latent_noise(noisy_latents, t)
    return F.mse_loss(predicted_noise, noise)



def pattern_decay_denoiser_loss(
    model: VAE,
    clean_images: Tensor,
    max_noise_strength: float = 0.85,
    is_wavelet: bool = False,
) -> Tensor:
    """Train APVD's image-space Pattern Decay denoiser.

    A clean training tensor is corrupted with random noise strengths. The small
    denoiser learns to pull it back toward the original image/tensor. This gives
    APVD a DDPM-like image-field habit without replacing APVD with a full DDPM.
    """
    clean = torch.nan_to_num(clean_images.float(), nan=0.0, posinf=4.0, neginf=-4.0)
    batch = clean.size(0)
    t = torch.rand(batch, 1, device=clean.device, dtype=clean.dtype)
    noise = torch.randn_like(clean)
    strength = max(0.01, float(max_noise_strength))
    view_shape = (batch, 1, 1, 1)
    noisy = clean + noise * t.reshape(view_shape) * strength
    if is_wavelet:
        noisy = noisy.clamp(-4.0, 4.0)
        target = clean.clamp(-4.0, 4.0)
    else:
        noisy = noisy.clamp(0.0, 1.0)
        target = clean.clamp(0.0, 1.0)
    predicted = model.pattern_decay_reconstruct(noisy, t).float()
    if is_wavelet:
        predicted = torch.nan_to_num(predicted, nan=0.0, posinf=4.0, neginf=-4.0).clamp(-4.0, 4.0)
    else:
        predicted = torch.nan_to_num(predicted, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return F.mse_loss(predicted, target)


def get_device() -> torch.device:
    """Return CUDA, MPS, or CPU device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")