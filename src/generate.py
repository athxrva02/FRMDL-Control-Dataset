"""generate.py — build the control dataset (Section 6).

For each selected ImageNette image: produce the canonical 224x224 array once,
save it as the clean PNG (control arm) and as clean JPEGs. Then for each
(corruption, severity): generate the corrupted array exactly once under a
deterministic seed, and save that single array as a lossless PNG and as a JPEG
at each quality. The corruption is never regenerated per save format — that
one-array rule is what makes the PNG/JPEG gap attributable to the save step
alone.

Usage:
    python src/generate.py [--config config.yaml] [--n-images N]
        [--severities 1,3,5] [--jpeg-qualities 75,90] [--out-dir data]
        [--seed S] [--overwrite] [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

try:
    from imagecorruptions import corrupt, get_corruption_names
except Exception as e:  # pragma: no cover
    print(f"FATAL: cannot import imagecorruptions: {e!r}")
    raise

try:
    from tqdm import tqdm
except Exception:  # tqdm optional
    def tqdm(x, **k):
        return x


# --------------------------------------------------------------------------- #
# ImageNette download (Section 4.1)
# --------------------------------------------------------------------------- #


def ensure_imagenette(cfg: C.Config) -> Path:
    root = cfg.imagenette_path
    if (root / "val").is_dir():
        return root
    cfg.data_path.mkdir(parents=True, exist_ok=True)
    tgz = cfg.data_path / f"{cfg.imagenette_dirname}.tgz"
    if not tgz.exists():
        print(f"[data] downloading ImageNette from {cfg.imagenette_url} ...")
        _download(cfg.imagenette_url, tgz)
    print(f"[data] extracting {tgz.name} ...")
    with tarfile.open(tgz, "r:gz") as tar:
        _safe_extract(tar, cfg.data_path)
    if not (root / "val").is_dir():
        raise RuntimeError(f"expected {root/'val'} after extraction; not found")
    return root


def _download(url: str, dest: Path) -> None:
    state = {"last": -1}

    def hook(blocks, bs, total):
        if total > 0:
            done = min(blocks * bs, total)
            pct = int(100 * done / total)
            if pct >= state["last"] + 5:  # log every ~5%, not every block
                state["last"] = pct
                print(f"  {done/1e6:6.1f} / {total/1e6:6.1f} MB ({pct:3d}%)", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=hook)


def _safe_extract(tar: tarfile.TarFile, path: Path) -> None:
    base = path.resolve()
    for m in tar.getmembers():
        target = (path / m.name).resolve()
        if not str(target).startswith(str(base)):
            raise RuntimeError(f"unsafe path in tar: {m.name}")
    tar.extractall(path)


# --------------------------------------------------------------------------- #
# Deterministic stratified image selection (Section 4.2)
# --------------------------------------------------------------------------- #


def select_images(cfg: C.Config) -> list:
    """Return a list of (image_id, source_path, wnid), deterministic from seed."""
    val = cfg.imagenette_path / "val"
    class_dirs = sorted(p for p in val.iterdir() if p.is_dir())
    if not class_dirs:
        raise RuntimeError(f"no class dirs in {val}")

    rng = random.Random(cfg.seed)
    n_classes = len(class_dirs)
    base = cfg.n_images // n_classes
    rem = cfg.n_images - base * n_classes  # spread remainder over first `rem` classes

    selected = []  # (wnid, source_path)
    for i, cdir in enumerate(class_dirs):
        files = sorted(p for p in cdir.iterdir() if p.suffix.lower() in (".jpeg", ".jpg", ".png"))
        want = base + (1 if i < rem else 0)
        want = min(want, len(files))
        picks = rng.sample(files, want) if want < len(files) else list(files)
        for p in picks:
            selected.append((cdir.name, p))

    # Stable id assignment: sort by (wnid, filename), then assign sequential ids.
    selected.sort(key=lambda t: (t[0], t[1].name))
    out = []
    for k, (wnid, path) in enumerate(selected, start=1):
        out.append((f"img_{k:04d}", path, wnid))
    return out


def write_manifest(cfg: C.Config, rows: list, wnid_to_index) -> None:
    path = C.REPO_ROOT / "manifest.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "source_path", "wnid", "imagenet_index", "base_png_path"])
        for image_id, src, wnid in rows:
            idx = "" if wnid_to_index is None else wnid_to_index.get(wnid, "")
            rel_src = src.relative_to(C.REPO_ROOT) if str(src).startswith(str(C.REPO_ROOT)) else src
            rel_png = C.base_png_path(cfg, image_id).relative_to(C.REPO_ROOT)
            w.writerow([image_id, rel_src, wnid, idx, rel_png])
    print(f"[manifest] wrote {len(rows)} rows -> {path}")


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #


def generate(cfg: C.Config, overwrite: bool, smoke: bool) -> None:
    corruptions = get_corruption_names()
    if smoke:
        corruptions = ["gaussian_noise", "defocus_blur", "frost"]
        print(f"[smoke] restricting to {corruptions}")

    qualities = cfg.effective_qualities
    print(f"[config] n_images={cfg.n_images} severities={cfg.severities} "
          f"qualities={qualities} (ImageNet-C default={cfg.imagenet_c_default_quality})")

    ensure_imagenette(cfg)
    wnid_to_index = C.resolve_label_map(cfg)
    if wnid_to_index is None:
        print("[label-map] UNRESOLVED — manifest indices will be empty; labelled metrics skipped.")
    else:
        print(f"[label-map] resolved {len(wnid_to_index)} wnids.")

    images = select_images(cfg)
    write_manifest(cfg, images, wnid_to_index)

    stats = {"png": 0, "jpeg": 0, "skipped_exist": 0, "corruption_fail": {}}

    # ---- clean arms ---- #
    print("[gen] clean arms (base PNG + clean JPEGs)...")
    for image_id, src, _wnid in tqdm(images, desc="clean"):
        arr = C.canonical_array(src)
        bp = C.base_png_path(cfg, image_id)
        if overwrite or not bp.exists():
            C.save_png(arr, bp); stats["png"] += 1
        else:
            stats["skipped_exist"] += 1
        for q in qualities:
            jp = C.clean_jpeg_path(cfg, image_id, q)
            if overwrite or not jp.exists():
                C.save_jpeg(arr, jp, q); stats["jpeg"] += 1
            else:
                stats["skipped_exist"] += 1

    # ---- corrupted arms ---- #
    failed_corruptions = set()
    for corruption in corruptions:
        if corruption in failed_corruptions:
            continue
        for severity in cfg.severities:
            desc = f"{corruption} s{severity}"
            for image_id, src, _wnid in tqdm(images, desc=desc, leave=False):
                cpng = C.corrupt_png_path(cfg, corruption, severity, image_id)
                cjpgs = [C.corrupt_jpeg_path(cfg, corruption, severity, image_id, q) for q in qualities]
                if not overwrite and cpng.exists() and all(p.exists() for p in cjpgs):
                    stats["skipped_exist"] += 1 + len(cjpgs)
                    continue

                arr = C.canonical_array(src)
                C.seed_for_corruption(image_id, corruption, severity)
                try:
                    corrupted = corrupt(arr, severity=severity, corruption_name=corruption)
                except Exception as e:
                    msg = f"{corruption}: {e!r}"
                    if corruption not in stats["corruption_fail"]:
                        print(f"\n[WARN] corruption failed, skipping it: {msg}")
                    stats["corruption_fail"][corruption] = str(e)
                    failed_corruptions.add(corruption)
                    break  # abandon this corruption entirely
                corrupted = C.to_uint8(corrupted)

                if overwrite or not cpng.exists():
                    C.save_png(corrupted, cpng); stats["png"] += 1
                else:
                    stats["skipped_exist"] += 1
                for q, jp in zip(qualities, cjpgs):
                    if overwrite or not jp.exists():
                        C.save_jpeg(corrupted, jp, q); stats["jpeg"] += 1
                    else:
                        stats["skipped_exist"] += 1
            if corruption in failed_corruptions:
                break

    _summary(cfg, corruptions, failed_corruptions, stats)


def _summary(cfg, corruptions, failed, stats):
    disk = _dir_size(cfg.data_path / "corrupted") + _dir_size(cfg.data_path / "base") \
        + _dir_size(cfg.data_path / "clean_saves")
    ok = [c for c in corruptions if c not in failed]
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print(f"  PNG files written : {stats['png']}")
    print(f"  JPEG files written: {stats['jpeg']}")
    print(f"  skipped (existing): {stats['skipped_exist']}")
    print(f"  corruptions OK    : {len(ok)}/{len(corruptions)}")
    if failed:
        print(f"  corruptions FAILED: {sorted(failed)}")
        for c, e in stats["corruption_fail"].items():
            print(f"      - {c}: {e}")
    print(f"  dataset disk usage: {disk/1e6:.1f} MB")
    print("=" * 60)


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def main():
    p = argparse.ArgumentParser(description="Generate the JPEG-confound control dataset.")
    C.add_common_cli(p)
    p.add_argument("--overwrite", action="store_true", help="rewrite files that already exist")
    p.add_argument("--smoke", action="store_true",
                   help="tiny run: 3 corruptions only (combine with --n-images)")
    args = p.parse_args()
    cfg = C.load_config(args.config, C.overrides_from_args(args))
    generate(cfg, overwrite=args.overwrite, smoke=args.smoke)


if __name__ == "__main__":
    main()
