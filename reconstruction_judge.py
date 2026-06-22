"""
APVD - Reconstruction Quality Judge

Scores generated images on four axes without encouraging pixel-perfect copying.
Acts as a passive quality filter: it ranks outputs and feeds the best into
memory evolution. It does NOT backpropagate into the VAE.

Primary backbone: DINOv2-small (21M params, ViT-S/14)
    - Self-supervised patch features capture semantic structure without pixel bias.
    - CLS token captures global subject identity independent of color/background.
    - Runs inference-only; no gradient flow into APVD's VAE.

Fallback (if DINOv2 unavailable): edge-map + latent-space comparison
    - Uses only libraries already present in the APVD codebase.
    - No internet required after first DINOv2 download.

Four scores (all in [0.0, 1.0]):
    Structure Score       - Are major silhouettes/shapes in the right positions?
    Character Identity    - Does the subject still "read" as the same entity?
    Composition Score     - Is there a clear focal subject with reasonable variety?
    Dream Freedom Score   - How novel is this image vs stored memories?
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from memory_system import MemoryBank, MemoryRecord


# ---------------------------------------------------------------------------
# Score dataclass
# ---------------------------------------------------------------------------

@dataclass
class JudgeScores:
    """Four-axis quality scores for a single generated image. All in [0.0, 1.0]."""

    structure_score: float
    """Are major silhouettes and shapes in approximately the right positions?"""

    identity_score: float
    """Does the subject still 'read' as the same character or entity?"""

    composition_score: float
    """Is there a clear focal region with reasonable background variety?"""

    dream_freedom_score: float
    """How novel is this image relative to all stored memories?
    High = genuinely new (rewarded). Very low = near-exact copy (penalized)."""

    combined_score: float
    """Weighted combination of all four scores."""

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"JudgeScores("
            f"struct={self.structure_score:.3f}, "
            f"id={self.identity_score:.3f}, "
            f"comp={self.composition_score:.3f}, "
            f"freedom={self.dream_freedom_score:.3f}, "
            f"combined={self.combined_score:.3f})"
        )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _try_load_dino(variant: str = "dinov2_vits14") -> Any | None:
    """
    Attempt to load a DINOv2 model via torch.hub.
    Returns None silently if the model is unavailable.

    Recommended variants:
        dinov2_vits14  - 21M params, fastest, recommended for RTX 3060
        dinov2_vitb14  - 86M params, stronger features, still comfortable
    """
    try:
        model = torch.hub.load(
            "facebookresearch/dinov2", variant, pretrained=True
        )
        model.eval()
        return model
    except Exception:
        return None


def _image_to_tensor(image: Image.Image, size: int = 224) -> torch.Tensor:
    """Resize and ImageNet-normalize an image to a DINOv2-compatible tensor [1, 3, H, W]."""
    img = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def _extract_edge_map(image: Image.Image, size: int = 128) -> np.ndarray:
    """
    Extract a normalized Sobel edge magnitude map.
    Used for structure comparison when DINOv2 is unavailable.
    Captures silhouettes and major shapes without colour information.
    """
    gray = np.array(
        image.convert("L").resize((size, size), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    magnitude = np.sqrt(gx**2 + gy**2)
    peak = magnitude.max()
    if peak > 0:
        magnitude /= peak
    return magnitude


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two flattened arrays."""
    a_flat = a.reshape(-1).astype(np.float32)
    b_flat = b.reshape(-1).astype(np.float32)
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def _quadrant_layout_score(
    ref_patches: np.ndarray, gen_patches: np.ndarray
) -> float:
    """
    Coarse spatial similarity using 2×2 quadrant-mean patch features.

    Compares where "stuff is happening" semantically without caring about
    exact patch-level texture or colour. A character whose torso is in the
    lower-left quadrant should still score well if the generated image has
    something semantically similar there, even at a different pose.

    Both inputs have shape [N, D] where N = H*W patch count.
    """
    n = ref_patches.shape[0]
    side = int(math.isqrt(n))
    if side * side != n:
        # Non-square patch grid fallback
        return max(0.0, _cosine_similarity(ref_patches.mean(0), gen_patches.mean(0)))

    ref_grid = ref_patches.reshape(side, side, -1)
    gen_grid = gen_patches.reshape(side, side, -1)
    half = side // 2

    quadrant_scores: list[float] = []
    for r_slice in (slice(0, half), slice(half, None)):
        for c_slice in (slice(0, half), slice(half, None)):
            r_mean = ref_grid[r_slice, c_slice].mean(axis=(0, 1))
            g_mean = gen_grid[r_slice, c_slice].mean(axis=(0, 1))
            quadrant_scores.append(_cosine_similarity(r_mean, g_mean))

    return max(0.0, float(np.mean(quadrant_scores)))


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

class ReconstructionJudge:
    """
    Passive quality filter for APVD generated images.

    The judge does NOT modify the VAE or inject gradients. It scores
    generated images and selects which ones deserve to enter memory
    evolution — acting as a curatorial layer between generation and learning.

    Usage pattern:
        judge = ReconstructionJudge(memory_bank=bank, device=device)
        candidates = [(img1, latent1), (img2, latent2), ...]
        best = judge.select_for_memory(candidates, reference=ref_img)
        for idx, scores in best:
            # save candidates[idx] to memory bank
    """

    def __init__(
        self,
        memory_bank: MemoryBank | None = None,
        device: torch.device | None = None,
        *,
        dino_variant: str = "dinov2_vits14",
        structure_weight: float = 0.30,
        identity_weight: float = 0.25,
        composition_weight: float = 0.15,
        dream_freedom_weight: float = 0.30,
        # Distance below which a generated image is treated as a "near-copy"
        dream_freedom_margin: float = 0.12,
    ):
        self.memory_bank = memory_bank
        self.device = device or torch.device("cpu")
        self.dream_freedom_margin = float(dream_freedom_margin)
        self._weights = dict(
            structure=structure_weight,
            identity=identity_weight,
            composition=composition_weight,
            dream_freedom=dream_freedom_weight,
        )

        self._dino: Any | None = _try_load_dino(dino_variant)
        if self._dino is not None:
            self._dino = self._dino.to(self.device)

        # RAM cache: image path → (cls_token, patch_features)
        self._feature_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        generated: Image.Image,
        reference: Image.Image | None = None,
        generated_latent: torch.Tensor | None = None,
    ) -> JudgeScores:
        """
        Score a single generated image.

        Args:
            generated:        The newly generated APVD image.
            reference:        Optional source/prompt image for structural comparison.
                              When None, structure and identity scores default to 0.5.
            generated_latent: Optional VAE latent for this image. Enables fast
                              latent-space distance comparison for dream freedom.
        """
        s = self._score_structure(generated, reference)
        i = self._score_identity(generated, reference)
        c = self._score_composition(generated)
        d = self._score_dream_freedom(generated, generated_latent)

        w = self._weights
        combined = s * w["structure"] + i * w["identity"] + c * w["composition"] + d * w["dream_freedom"]

        return JudgeScores(
            structure_score=round(s, 4),
            identity_score=round(i, 4),
            composition_score=round(c, 4),
            dream_freedom_score=round(d, 4),
            combined_score=round(float(combined), 4),
        )

    def rank(
        self,
        candidates: list[tuple[Image.Image, torch.Tensor | None]],
        reference: Image.Image | None = None,
    ) -> list[tuple[int, JudgeScores]]:
        """
        Score all candidates and return them sorted best-to-worst.

        Returns:
            List of (original_index, JudgeScores), highest combined_score first.
        """
        scored = []
        for idx, (img, latent) in enumerate(candidates):
            scores = self.score(img, reference=reference, generated_latent=latent)
            scored.append((idx, scores))
        scored.sort(key=lambda pair: pair[1].combined_score, reverse=True)
        return scored

    def select_for_memory(
        self,
        candidates: list[tuple[Image.Image, torch.Tensor | None]],
        reference: Image.Image | None = None,
        *,
        top_k: int = 2,
        min_combined_score: float = 0.45,
    ) -> list[tuple[int, JudgeScores]]:
        """
        Return only the top-k candidates that exceed the quality threshold.

        These are the images the memory evolution system should absorb. Low
        scores mean either the structure is lost OR the image is too similar
        to existing memories (memorization risk).
        """
        ranked = self.rank(candidates, reference=reference)
        return [
            (idx, scores)
            for idx, scores in ranked[:top_k]
            if scores.combined_score >= min_combined_score
        ]

    def clear_cache(self) -> None:
        """Clear the in-RAM feature cache. Call after the memory bank changes significantly."""
        self._feature_cache.clear()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_dino_features(
        self, image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Extract DINOv2 CLS token [D] and patch features [N, D].
        Returns None if DINOv2 is not loaded.
        """
        if self._dino is None:
            return None
        tensor = _image_to_tensor(image).to(self.device)
        out = self._dino.forward_features(tensor)
        cls = out["x_norm_clstoken"].squeeze(0).float().cpu().numpy()
        patches = out["x_norm_patchtokens"].squeeze(0).float().cpu().numpy()
        return cls, patches

    def _extract_cached(
        self, image: Image.Image, key: str | None = None
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if key is not None and key in self._feature_cache:
            return self._feature_cache[key]
        result = self._extract_dino_features(image)
        if result is not None and key is not None:
            self._feature_cache[key] = result
        return result

    # ------------------------------------------------------------------
    # Individual scores (see docstrings for design rationale)
    # ------------------------------------------------------------------

    def _score_structure(
        self,
        generated: Image.Image,
        reference: Image.Image | None,
    ) -> float:
        """
        Structure Score: Are major shapes/silhouettes in approximately the right positions?

        With DINOv2: 2×2 quadrant-level patch feature comparison.
            This is deliberately coarse. Fine pixel locations, exact poses, and
            background regions are all ignored. Only the quadrant-level semantic
            layout matters: is the subject on the left? Are the shapes roughly aligned?

        Without DINOv2: Sobel edge-map spatial correlation.
            Captures silhouettes without any colour information.

        Score deliberately does NOT reward pixel-level spatial accuracy.
        A character in a new pose but same general position still scores well.
        """
        if reference is None:
            return 0.5

        if self._dino is not None:
            ref_f = self._extract_dino_features(reference)
            gen_f = self._extract_dino_features(generated)
            if ref_f is not None and gen_f is not None:
                return _quadrant_layout_score(ref_f[1], gen_f[1])

        # Edge-map fallback
        ref_edge = _extract_edge_map(reference)
        gen_edge = _extract_edge_map(generated)
        return max(0.0, _cosine_similarity(ref_edge, gen_edge))

    def _score_identity(
        self,
        generated: Image.Image,
        reference: Image.Image | None,
    ) -> float:
        """
        Character Identity Score: Does the generated image "read" as the same entity?

        Uses DINOv2 CLS token cosine similarity. The CLS token encodes global
        semantic content — what the image depicts — independently of colour,
        background, and exact texture. A character in a new outfit against a new
        background should still score well here.

        Soft anti-memorization penalty: similarities above 0.97 (near-identical
        global appearance) receive a downward push. This is the first line of
        defence against global appearance memorization.
        """
        if reference is None:
            return 0.5

        if self._dino is not None:
            ref_f = self._extract_dino_features(reference)
            gen_f = self._extract_dino_features(generated)
            if ref_f is not None and gen_f is not None:
                sim = _cosine_similarity(ref_f[0], gen_f[0])
                # Gentle penalty for near-exact global similarity
                if sim > 0.97:
                    sim = 0.97 - (sim - 0.97) * 2.5
                return max(0.0, min(1.0, sim))

        # Fallback: downsampled grayscale histogram comparison
        ref_gray = np.array(reference.convert("L").resize((64, 64)), dtype=np.float32)
        gen_gray = np.array(generated.convert("L").resize((64, 64)), dtype=np.float32)
        return max(0.0, _cosine_similarity(ref_gray, gen_gray))

    def _score_composition(self, generated: Image.Image) -> float:
        """
        Composition Score: Is there a clear focal subject with background variety?

        Uses the coefficient of variation (std/mean) of DINOv2 patch feature L2 norms.
        - Very uniform norms → flat/collapsed/featureless image → low score
        - Very high variation → chaotic noise with no clear subject → low score
        - Moderate variation → clear subject standing out from background → high score

        Rewards images that "have something to look at" without dictating what it is.
        """
        if self._dino is not None:
            features = self._extract_dino_features(generated)
            if features is not None:
                patch_norms = np.linalg.norm(features[1], axis=1)
                cv = float(patch_norms.std() / (patch_norms.mean() + 1e-9))
                # Gaussian bump centered at CV ≈ 0.4 (moderate, clear subject)
                score = math.exp(-((cv - 0.40) ** 2) / (2 * 0.22**2))
                return float(score)

        # Fallback: spatial luminance variance
        gray = np.array(
            generated.convert("L").resize((64, 64)), dtype=np.float32
        ) / 255.0
        local_std = float(gray.std())
        # Reward moderate contrast, not flat gray or pure noise
        score = math.exp(-((local_std - 0.18) ** 2) / (2 * 0.10**2))
        return float(score)

    def _score_dream_freedom(
        self,
        generated: Image.Image,
        generated_latent: torch.Tensor | None = None,
    ) -> float:
        """
        Dream Freedom Score: How different is this image from all stored memories?

        This is the primary anti-memorization mechanism. It penalizes images
        that are too similar to anything already in the memory bank.

        Two-tier comparison (both used when available, minimum is taken):
            1. Latent-space cosine distance (fast): generated_latent vs all stored .pt files.
               Already available in MemoryBank — no new model needed.
            2. DINOv2 CLS token distance (semantic): generated image vs cached memory images.

        Scoring curve (non-linear):
            dist < margin          → heavy penalty (0.0–0.10)  — near-copy, do not absorb
            margin < dist < 2×margin → linear ramp (0.10–0.80) — borderline
            dist > 2×margin        → reward, asymptote toward 1.0 — genuinely novel

        The margin parameter (default 0.12) controls sensitivity. Lower = stricter.
        """
        if self.memory_bank is None:
            return 0.65  # No reference bank: default reward for exploration

        margin = self.dream_freedom_margin
        distances: list[float] = []

        # --- Tier 1: latent-space distance (O(n) dot products, very fast) ---
        if generated_latent is not None:
            gen_flat = generated_latent.detach().float().cpu().reshape(-1).numpy()
            gen_norm = float(np.linalg.norm(gen_flat))
            if gen_norm > 1e-9:
                gen_unit = gen_flat / gen_norm
                for record in self.memory_bank.load_memories(limit=64):
                    try:
                        stored = self.memory_bank.load_latent(record)
                        stored_flat = stored.detach().float().cpu().reshape(-1).numpy()
                        stored_norm = float(np.linalg.norm(stored_flat))
                        if stored_norm > 1e-9:
                            sim = float(np.dot(gen_unit, stored_flat / stored_norm))
                            distances.append(1.0 - max(0.0, sim))
                    except Exception:
                        continue

        # --- Tier 2: DINOv2 semantic distance ---
        if self._dino is not None:
            gen_feats = self._extract_dino_features(generated)
            if gen_feats is not None:
                for path in self.memory_bank.get_weighted_image_paths(limit=32)[:32]:
                    try:
                        with Image.open(path) as mem_img:
                            mem_feats = self._extract_cached(
                                mem_img.convert("RGB"), key=str(path)
                            )
                        if mem_feats is not None:
                            sim = _cosine_similarity(gen_feats[0], mem_feats[0])
                            distances.append(1.0 - max(0.0, sim))
                    except Exception:
                        continue

        if not distances:
            return 0.65

        min_dist = float(min(distances))

        # Piecewise scoring curve
        if min_dist < margin:
            # Near-copy zone: heavy penalty, approaching 0
            score = 0.10 * (min_dist / max(1e-9, margin))
        elif min_dist < 2.0 * margin:
            # Ramp zone: linear from 0.10 to 0.80
            t = (min_dist - margin) / margin
            score = 0.10 + t * 0.70
        else:
            # Freedom zone: reward, asymptote toward 1.0
            overshoot = min_dist - 2.0 * margin
            score = 0.80 + 0.20 * (1.0 - math.exp(-overshoot * 5.0))

        return max(0.0, min(1.0, score))
