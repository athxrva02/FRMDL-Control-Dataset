"""Shared utilities for the JPEG re-encoding confound control dataset.

Holds the pieces every script needs: config loading + CLI overrides, the
deterministic seeding scheme, the wnid -> ImageNet-1k index label map (fetched
and verified, never hard-coded), the canonical 224x224 image step, and the
PNG/JPEG save helpers that are the *only* thing this project adds on top of the
original ImageNet-C corruption code.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PIL import Image

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT = Path(__file__).resolve().parent.parent


def resolve(path_like) -> Path:
    """Resolve a possibly-relative config path against the repo root."""
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class Config:
    seed: int = 0
    n_images: int = 500
    severities: list = field(default_factory=lambda: [1, 3, 5])
    jpeg_qualities: list = field(default_factory=lambda: [75, 90])
    imagenet_c_default_quality: int = 85
    data_dir: str = "data"
    results_dir: str = "results"
    imagenette_url: str = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
    imagenette_dirname: str = "imagenette2-320"
    imagenet_class_index_url: str = (
        "https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json"
    )
    models: list = field(default_factory=lambda: ["alexnet", "resnet50", "convnext_tiny"])
    batch_size: int = 184

    # ---- derived helpers -------------------------------------------------- #
    @property
    def effective_qualities(self) -> list:
        """Swept qualities unioned with the discovered ImageNet-C default."""
        qs = set(int(q) for q in self.jpeg_qualities)
        if self.imagenet_c_default_quality is not None:
            qs.add(int(self.imagenet_c_default_quality))
        return sorted(qs)

    def is_default_quality(self, q: int) -> bool:
        return int(q) == int(self.imagenet_c_default_quality)

    @property
    def data_path(self) -> Path:
        return resolve(self.data_dir)

    @property
    def results_path(self) -> Path:
        return resolve(self.results_dir)

    @property
    def imagenette_path(self) -> Path:
        return self.data_path / self.imagenette_dirname


def load_config(config_path: str = "config.yaml", overrides: Optional[dict] = None) -> Config:
    """Load config.yaml and apply CLI overrides (None values are ignored)."""
    with open(resolve(config_path)) as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config(**{k: v for k, v in raw.items() if k in Config.__dataclass_fields__})
    if overrides:
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
    return cfg


def add_common_cli(parser: argparse.ArgumentParser) -> None:
    """Register the config-override flags shared across scripts."""
    parser.add_argument("--config", default="config.yaml", help="path to config.yaml")
    parser.add_argument("--n-images", type=int, default=None, dest="n_images")
    parser.add_argument(
        "--severities", type=_int_list, default=None,
        help="comma-separated, e.g. 1,3,5",
    )
    parser.add_argument(
        "--jpeg-qualities", type=_int_list, default=None, dest="jpeg_qualities",
        help="comma-separated, e.g. 75,90",
    )
    parser.add_argument("--out-dir", default=None, dest="data_dir", help="override data_dir")
    parser.add_argument("--seed", type=int, default=None)


def _int_list(s: str) -> list:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def overrides_from_args(args: argparse.Namespace) -> dict:
    keys = ["n_images", "severities", "jpeg_qualities", "data_dir", "seed"]
    return {k: getattr(args, k, None) for k in keys}


# --------------------------------------------------------------------------- #
# Determinism (Section 5.4)
# --------------------------------------------------------------------------- #


def derive_seed(image_id: str, corruption_name: str, severity: int) -> int:
    """Order-independent 32-bit seed from (image, corruption, severity).

    crc32 of the joined key, masked to 32 bits. Used to seed both numpy and
    the stdlib RNG before every corrupt() call so a given (image, corruption,
    severity) array is reproducible regardless of generation order.
    """
    key = f"{image_id}|{corruption_name}|{severity}".encode("utf-8")
    return zlib.crc32(key) & 0xFFFFFFFF


def seed_for_corruption(image_id: str, corruption_name: str, severity: int) -> int:
    s = derive_seed(image_id, corruption_name, severity)
    np.random.seed(s)
    random.seed(s)
    return s


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


# --------------------------------------------------------------------------- #
# Label map (Section 4.2) — fetch + verify, never hard-code
# --------------------------------------------------------------------------- #


def resolve_label_map(cfg: Config, cache_dir: Optional[Path] = None) -> Optional[dict]:
    """Return a dict wnid -> ImageNet-1k index (0..999), or None if unresolved.

    Strategy:
      1. Fetch the standard imagenet_class_index.json, assert it has 1000 entries
         and index 0 == ["n01440764", "tench"]. Cache it on disk.
      2. Fallback: index == rank in the lexicographically sorted list of the 1000
         wnids (the torchvision.datasets.ImageNet convention).
      3. If neither works, return None; the pipeline then runs label-free only.
    """
    cache_dir = cache_dir or cfg.data_path
    cache_path = cache_dir / "imagenet_class_index.json"

    idx_to_pair = _load_class_index(cfg, cache_path)
    if idx_to_pair is not None:
        try:
            assert len(idx_to_pair) == 1000, f"expected 1000 entries, got {len(idx_to_pair)}"
            assert idx_to_pair["0"][0] == "n01440764" and idx_to_pair["0"][1] == "tench", (
                f"index 0 mismatch: {idx_to_pair['0']}"
            )
            wnid_to_index = {pair[0]: int(i) for i, pair in idx_to_pair.items()}
            assert len(wnid_to_index) == 1000
            return wnid_to_index
        except AssertionError as e:
            print(f"[label-map] class index failed verification ({e}); trying fallback.")

    # Fallback: lexicographic rank of wnids present in the ImageNette tree.
    wnids = _imagenette_wnids(cfg)
    if wnids is not None and len(wnids) == 10:
        # We only have ImageNette's 10 wnids on disk, but the 1000-way index is a
        # global rank. The lexicographic-rank rule requires *all* 1000 wnids, which
        # we do not have offline. So this fallback only applies when the full set is
        # derivable. Without it, return None and run label-free.
        print(
            "[label-map] could not fetch class index and cannot reconstruct the global "
            "1000-way rank offline; proceeding label-free."
        )
    return None


def _load_class_index(cfg: Config, cache_path: Path):
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception:
            pass
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(cfg.imagenet_class_index_url, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        with open(cache_path, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print(f"[label-map] fetch failed: {e!r}")
        return None


def _imagenette_wnids(cfg: Config):
    val = cfg.imagenette_path / "val"
    if not val.is_dir():
        return None
    return sorted(p.name for p in val.iterdir() if p.is_dir())


# --------------------------------------------------------------------------- #
# Canonical image step (Section 4.3) + save helpers
# --------------------------------------------------------------------------- #


def canonical_array(source_path) -> np.ndarray:
    """Load a source image -> canonical 224x224x3 uint8 array.

    1. open + convert RGB
    2. resize shorter side to 256 (bilinear)
    3. center-crop 224x224
    """
    img = Image.open(source_path).convert("RGB")
    w, h = img.size
    short = min(w, h)
    scale = 256 / short
    new_w, new_h = round(w * scale), round(h * scale)
    img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    left = (new_w - 224) // 2
    top = (new_h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))

    arr = np.asarray(img, dtype=np.uint8)
    assert arr.shape == (224, 224, 3), f"canonical shape {arr.shape}"
    return arr


def to_uint8(arr: np.ndarray) -> np.ndarray:
    """Clip to [0, 255] and cast to uint8 (Section 5.4)."""
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr, 0, 255).astype(np.uint8)


def save_png(arr: np.ndarray, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(arr)).save(path, format="PNG")


def save_jpeg(arr: np.ndarray, path, quality: int) -> None:
    """Save as JPEG at the given quality. PIL defaults for all other params
    (subsampling, qtables); see README."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(arr)).save(path, format="JPEG", quality=int(quality))


def load_array(path) -> np.ndarray:
    """Load an image file back into a uint8 RGB array (forces a real decode)."""
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Path layout helpers (Section 5.5) — single source of truth for filenames
# --------------------------------------------------------------------------- #


def base_png_path(cfg: Config, image_id: str) -> Path:
    return cfg.data_path / "base" / f"{image_id}.png"


def clean_jpeg_path(cfg: Config, image_id: str, q: int) -> Path:
    return cfg.data_path / "clean_saves" / f"jpeg_q{q}" / f"{image_id}.jpg"


def corrupt_png_path(cfg: Config, corruption: str, severity: int, image_id: str) -> Path:
    return cfg.data_path / "corrupted" / corruption / f"s{severity}" / "png" / f"{image_id}.png"


def corrupt_jpeg_path(cfg: Config, corruption: str, severity: int, image_id: str, q: int) -> Path:
    return (
        cfg.data_path / "corrupted" / corruption / f"s{severity}" / f"jpeg_q{q}" / f"{image_id}.jpg"
    )
