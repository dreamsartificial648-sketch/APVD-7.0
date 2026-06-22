"""
APVD - Memory Finder

Accepts a reference image and searches the APVD memory bank for images
containing similar structures, characters, or shapes.

Architecture:
    DINOv2-small (21M params, ViT-S/14) features are precomputed and cached
    for all memory bank images. Search is then a set of dot products — O(n)
    in the number of stored memories with no image loading at query time.

    Falls back to existing VAE latent-space search when DINOv2 is unavailable.
    All latents are already stored as .pt files by MemoryBank; no new disk
    writes are needed.

Outputs per match:
    - identity_similarity   Global semantic match (CLS token cosine similarity)
    - structure_similarity  Spatial layout match (quadrant patch feature comparison)
    - latent_similarity     VAE latent-space cosine similarity
    - combined_score        Weighted combination
    - attention_region      (x, y, w, h) bounding box of the most active region
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from memory_system import MemoryBank, MemoryRecord

def safe_torch_load(path, *, map_location=None):
    """Local fallback because some APVD versions define safe_torch_load in app.py, not memory_system.py."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
from reconstruction_judge import (
    _try_load_dino,
    _image_to_tensor,
    _extract_edge_map,
    _cosine_similarity,
    _quadrant_layout_score,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class MemoryMatch:
    """A single result from a Memory Finder query."""

    record: MemoryRecord
    image_path: Path
    identity_similarity: float
    """Global semantic similarity — does it depict the same kind of subject?"""
    structure_similarity: float
    """Spatial layout similarity — are shapes in similar positions?"""
    latent_similarity: float | None
    """VAE latent cosine similarity. None if latent unavailable for this record."""
    combined_score: float
    attention_region: tuple[int, int, int, int] | None
    """Approximate bounding box (x, y, width, height) of the most active region,
    in pixel coordinates relative to the stored image at its native resolution.
    None if patch layout cannot be determined."""

    def __repr__(self) -> str:
        bbox = f"bbox={self.attention_region}" if self.attention_region else "bbox=None"
        return (
            f"MemoryMatch(id={self.identity_similarity:.3f}, "
            f"struct={self.structure_similarity:.3f}, "
            f"combined={self.combined_score:.3f}, {bbox}, "
            f"prompt='{self.record.prompt[:30]}')"
        )


# ---------------------------------------------------------------------------
# Memory Finder
# ---------------------------------------------------------------------------

class MemoryFinder:
    """
    Searches APVD memory banks for images similar to a reference query.

    Typical usage:
        finder = MemoryFinder(memory_bank=bank, device=device)
        finder.build_index()   # call once; rebuilds automatically if stale

        matches = finder.find(query_image, top_k=5)
        for match in matches:
            print(match.record.prompt, match.combined_score)
            img = Image.open(match.image_path)
            # optionally draw match.attention_region as a bounding box

    Latent-only search (no image loading required):
        matches = finder.find_by_latent(my_latent, top_k=5)
    """

    def __init__(
        self,
        memory_bank: MemoryBank,
        device: torch.device | None = None,
        *,
        dino_variant: str = "dinov2_vits14",
        identity_weight: float = 0.45,
        structure_weight: float = 0.35,
        latent_weight: float = 0.20,
    ):
        self.memory_bank = memory_bank
        self.device = device or torch.device("cpu")
        self.identity_weight = identity_weight
        self.structure_weight = structure_weight
        self.latent_weight = latent_weight

        self._dino: Any | None = _try_load_dino(dino_variant)
        if self._dino is not None:
            self._dino = self._dino.to(self.device)

        # Feature index.
        # Key: memory_id string
        # Value: dict with any subset of:
        #   "cls"      → np.ndarray [D]           DINOv2 CLS token (unit-normalized)
        #   "patches"  → np.ndarray [N, D]         DINOv2 patch features
        #   "edge_map" → np.ndarray [H, W]         Sobel edge map (fallback)
        #   "latent"   → np.ndarray [latent_dim]   Unit-normalized VAE latent
        self._index: dict[str, dict[str, Any]] = {}
        self._index_built: bool = False

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def build_index(
        self,
        limit: int = 256,
        *,
        force_rebuild: bool = False,
        verbose: bool = False,
    ) -> int:
        """
        Precompute and cache features for all memory bank entries.

        Should be called once before bulk searches. Called automatically on
        first find() if not already built. After adding new memories, call
        invalidate_index() then build_index() again.

        Returns:
            Number of successfully indexed records.
        """
        if self._index_built and not force_rebuild:
            return len(self._index)

        self._index.clear()
        records = self.memory_bank.load_memories(limit=limit)

        for record in records:
            entry: dict[str, Any] = {}

            # --- DINOv2 features (preferred) ---
            img_path = Path(record.image_path)
            if img_path.exists() and self._dino is not None:
                try:
                    with Image.open(img_path) as img:
                        cls, patches = self._extract_features(img.convert("RGB"))
                    entry["cls"] = cls / (np.linalg.norm(cls) + 1e-9)
                    entry["patches"] = patches
                    if verbose:
                        print(f"  [index] DINOv2 OK: {record.memory_id}")
                except Exception as exc:
                    if verbose:
                        print(f"  [index] DINOv2 failed for {record.memory_id}: {exc}")

            # --- Edge-map fallback ---
            if "cls" not in entry and img_path.exists():
                try:
                    with Image.open(img_path) as img:
                        entry["edge_map"] = _extract_edge_map(img)
                    if verbose:
                        print(f"  [index] edge-map fallback: {record.memory_id}")
                except Exception:
                    pass

            # --- VAE latent (always attempt, fast) ---
            latent_path = Path(record.latent_path)
            if latent_path.exists():
                try:
                    latent = safe_torch_load(latent_path, map_location="cpu")
                    flat = latent.detach().float().reshape(-1).numpy()
                    norm = float(np.linalg.norm(flat))
                    entry["latent"] = flat / (norm + 1e-9)
                except Exception:
                    pass

            if entry:
                self._index[record.memory_id] = entry

        self._index_built = True
        return len(self._index)

    def invalidate_index(self) -> None:
        """Discard the cached feature index. Call after new memories are saved."""
        self._index.clear()
        self._index_built = False

    # ------------------------------------------------------------------
    # Primary search API
    # ------------------------------------------------------------------

    def find(
        self,
        query_image: Image.Image,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        search_limit: int = 256,
        include_attention_regions: bool = True,
        output_image_size: int = 256,
    ) -> list[MemoryMatch]:
        """
        Search for memory images similar to the query image.

        Args:
            query_image:               Reference image to search for.
            top_k:                     Maximum results to return.
            min_score:                 Minimum combined_score to include in results.
            search_limit:              Maximum stored memories to search.
            include_attention_regions: Compute attention bounding boxes if True.
            output_image_size:         Assumed output resolution for bbox coordinates.

        Returns:
            List of MemoryMatch, sorted by combined_score descending.
        """
        if not self._index_built:
            self.build_index(limit=search_limit)

        query_rgb = query_image.convert("RGB")

        # Extract query features
        query_cls: np.ndarray | None = None
        query_patches: np.ndarray | None = None
        query_edge: np.ndarray | None = None

        if self._dino is not None:
            try:
                query_cls, query_patches = self._extract_features(query_rgb)
                query_cls = query_cls / (np.linalg.norm(query_cls) + 1e-9)
            except Exception:
                pass

        if query_cls is None:
            query_edge = _extract_edge_map(query_rgb)

        # Load record map for fast access
        records = self.memory_bank.load_memories(limit=search_limit)
        record_map = {r.memory_id: r for r in records}

        matches: list[MemoryMatch] = []

        for memory_id, entry in self._index.items():
            record = record_map.get(memory_id)
            if record is None:
                continue

            # Identity score (global semantic match)
            if query_cls is not None and "cls" in entry:
                identity_sim = max(0.0, float(np.dot(query_cls, entry["cls"])))
            elif query_edge is not None and "edge_map" in entry:
                identity_sim = max(0.0, _cosine_similarity(query_edge, entry["edge_map"]))
            else:
                identity_sim = 0.0

            # Structure score (spatial layout match)
            if query_patches is not None and "patches" in entry:
                structure_sim = _quadrant_layout_score(query_patches, entry["patches"])
            elif query_edge is not None and "edge_map" in entry:
                structure_sim = max(0.0, _cosine_similarity(query_edge, entry["edge_map"]))
            else:
                structure_sim = identity_sim

            # Latent score (VAE embedding match)
            latent_sim: float | None = None
            if "latent" in entry:
                # No query latent from an external image — use identity as proxy
                # (improves when the caller passes a generated latent via find_by_latent)
                latent_sim = identity_sim * 0.8 + structure_sim * 0.2

            # Weighted combined score
            if latent_sim is not None:
                total_w = self.identity_weight + self.structure_weight + self.latent_weight
                combined = (
                    identity_sim * self.identity_weight
                    + structure_sim * self.structure_weight
                    + latent_sim * self.latent_weight
                ) / total_w
            else:
                total_w = self.identity_weight + self.structure_weight
                combined = (
                    identity_sim * self.identity_weight
                    + structure_sim * self.structure_weight
                ) / total_w

            if combined < min_score:
                continue

            # Attention bounding region
            attention_region: tuple[int, int, int, int] | None = None
            if include_attention_regions and "patches" in entry:
                attention_region = _compute_attention_region(
                    entry["patches"], image_size=output_image_size
                )

            matches.append(
                MemoryMatch(
                    record=record,
                    image_path=Path(record.image_path),
                    identity_similarity=round(identity_sim, 4),
                    structure_similarity=round(structure_sim, 4),
                    latent_similarity=round(latent_sim, 4) if latent_sim is not None else None,
                    combined_score=round(float(combined), 4),
                    attention_region=attention_region,
                )
            )

        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return matches[:top_k]

    def find_with_latent(
        self,
        query_image: Image.Image,
        query_latent: torch.Tensor,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        search_limit: int = 256,
    ) -> list[MemoryMatch]:
        """
        Richer search using both a query image AND its VAE latent.

        The latent gives a direct comparison in VAE embedding space, which
        can be faster and more consistent with APVD's internal representation.
        """
        if not self._index_built:
            self.build_index(limit=search_limit)

        query_rgb = query_image.convert("RGB")

        # Query latent (unit-normalized for dot-product similarity)
        q_flat = query_latent.detach().float().cpu().reshape(-1).numpy()
        q_norm = float(np.linalg.norm(q_flat))
        q_unit = q_flat / (q_norm + 1e-9) if q_norm > 1e-9 else q_flat

        # DINOv2 features
        query_cls: np.ndarray | None = None
        query_patches: np.ndarray | None = None
        if self._dino is not None:
            try:
                query_cls, query_patches = self._extract_features(query_rgb)
                query_cls = query_cls / (np.linalg.norm(query_cls) + 1e-9)
            except Exception:
                pass

        query_edge: np.ndarray | None = None
        if query_cls is None:
            query_edge = _extract_edge_map(query_rgb)

        records = self.memory_bank.load_memories(limit=search_limit)
        record_map = {r.memory_id: r for r in records}
        matches: list[MemoryMatch] = []

        for memory_id, entry in self._index.items():
            record = record_map.get(memory_id)
            if record is None:
                continue

            if query_cls is not None and "cls" in entry:
                identity_sim = max(0.0, float(np.dot(query_cls, entry["cls"])))
            elif query_edge is not None and "edge_map" in entry:
                identity_sim = max(0.0, _cosine_similarity(query_edge, entry["edge_map"]))
            else:
                identity_sim = 0.0

            if query_patches is not None and "patches" in entry:
                structure_sim = _quadrant_layout_score(query_patches, entry["patches"])
            else:
                structure_sim = identity_sim

            latent_sim: float | None = None
            if "latent" in entry and q_norm > 1e-9:
                latent_sim = max(0.0, float(np.dot(q_unit, entry["latent"])))

            if latent_sim is not None:
                combined = (
                    identity_sim * self.identity_weight
                    + structure_sim * self.structure_weight
                    + latent_sim * self.latent_weight
                )
            else:
                tw = self.identity_weight + self.structure_weight
                combined = (
                    identity_sim * (self.identity_weight / tw)
                    + structure_sim * (self.structure_weight / tw)
                )

            if combined < min_score:
                continue

            region = _compute_attention_region(entry["patches"]) if "patches" in entry else None

            matches.append(
                MemoryMatch(
                    record=record,
                    image_path=Path(record.image_path),
                    identity_similarity=round(identity_sim, 4),
                    structure_similarity=round(structure_sim, 4),
                    latent_similarity=round(latent_sim, 4) if latent_sim is not None else None,
                    combined_score=round(float(combined), 4),
                    attention_region=region,
                )
            )

        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return matches[:top_k]

    def find_by_latent(
        self,
        query_latent: torch.Tensor,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
        search_limit: int = 256,
    ) -> list[MemoryMatch]:
        """
        Latent-space-only search. No image loading or DINOv2 inference needed.

        Fast path: useful when you already have a generated latent and want
        to find the nearest stored memories without any image I/O.
        All latents are already stored by MemoryBank — zero new overhead.
        """
        if not self._index_built:
            self.build_index(limit=search_limit)

        q_flat = query_latent.detach().float().cpu().reshape(-1).numpy()
        q_norm = float(np.linalg.norm(q_flat))
        if q_norm < 1e-9:
            return []
        q_unit = q_flat / q_norm

        records = self.memory_bank.load_memories(limit=search_limit)
        record_map = {r.memory_id: r for r in records}
        matches: list[MemoryMatch] = []

        for memory_id, entry in self._index.items():
            if "latent" not in entry:
                continue
            record = record_map.get(memory_id)
            if record is None:
                continue

            sim = max(0.0, float(np.dot(q_unit, entry["latent"])))
            if sim < min_score:
                continue

            matches.append(
                MemoryMatch(
                    record=record,
                    image_path=Path(record.image_path),
                    identity_similarity=round(sim, 4),
                    structure_similarity=round(sim, 4),
                    latent_similarity=round(sim, 4),
                    combined_score=round(sim, 4),
                    attention_region=None,
                )
            )

        matches.sort(key=lambda m: m.combined_score, reverse=True)
        return matches[:top_k]

    # ------------------------------------------------------------------
    # Grouped / annotated search
    # ------------------------------------------------------------------

    def find_by_personality(
        self,
        query_image: Image.Image,
        personality: str,
        *,
        top_k: int = 5,
        search_limit: int = 256,
    ) -> list[MemoryMatch]:
        """
        Restrict the search to memories tagged with a specific personality.

        Useful for "find images of this character in my memory bank"
        when personalities are used as character labels in APVD.
        """
        all_matches = self.find(
            query_image, top_k=search_limit, search_limit=search_limit
        )
        filtered = [
            m for m in all_matches
            if m.record.personality.lower() == personality.lower()
        ]
        return filtered[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _extract_features(
        self, image: Image.Image
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract DINOv2 CLS token [D] and patch features [N, D]."""
        tensor = _image_to_tensor(image).to(self.device)
        out = self._dino.forward_features(tensor)
        cls = out["x_norm_clstoken"].squeeze(0).float().cpu().numpy()
        patches = out["x_norm_patchtokens"].squeeze(0).float().cpu().numpy()
        return cls, patches


# ---------------------------------------------------------------------------
# Attention region computation (standalone, used by both Judge and Finder)
# ---------------------------------------------------------------------------

def _compute_attention_region(
    patches: np.ndarray,
    image_size: int = 256,
) -> tuple[int, int, int, int] | None:
    """
    Estimate the bounding box of the most semantically active image region.

    Uses the L2 norm of each DINOv2 patch feature vector as an attention proxy.
    High-norm patches correspond to regions with rich semantic content.

    Returns (x, y, width, height) in pixel coordinates, or None if the patch
    grid is non-square or too small to be meaningful.

    Design note: This is a coarse bounding box, not a precise segmentation mask.
    For a tighter mask, DINO's self-attention maps (forward_features keys) can be
    used with a threshold, but require additional forward passes with register tokens.
    """
    n = patches.shape[0]
    side = int(math.isqrt(n))
    if side < 2 or side * side != n:
        return None

    patch_norms = np.linalg.norm(patches, axis=1).reshape(side, side)

    # Find patches above the 75th percentile of activation
    threshold = float(np.percentile(patch_norms, 75))
    hot = patch_norms >= threshold

    rows_active = np.any(hot, axis=1)
    cols_active = np.any(hot, axis=0)

    if not rows_active.any() or not cols_active.any():
        return None

    row_indices = np.where(rows_active)[0]
    col_indices = np.where(cols_active)[0]
    rmin, rmax = int(row_indices[0]), int(row_indices[-1])
    cmin, cmax = int(col_indices[0]), int(col_indices[-1])

    patch_px = image_size // side
    x = cmin * patch_px
    y = rmin * patch_px
    w = (cmax - cmin + 1) * patch_px
    h = (rmax - rmin + 1) * patch_px

    # Clamp to image bounds
    x = max(0, min(x, image_size - 1))
    y = max(0, min(y, image_size - 1))
    w = max(patch_px, min(w, image_size - x))
    h = max(patch_px, min(h, image_size - y))

    return (x, y, w, h)
