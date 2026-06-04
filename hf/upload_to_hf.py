#!/usr/bin/env python3
"""upload_to_hf.py — publish the control dataset to the Hugging Face Hub.

Builds `metadata.csv` (one row per arm file) from the on-disk dataset + manifest,
then uploads the dataset card, the small index/config files, and the image arms
to a Hub dataset repo. The ImageNette source download is never uploaded.

Prerequisites:
    pip install huggingface_hub pyyaml
    huggingface-cli login          # or pass --token / set HF_TOKEN

Examples:
    # dry run — build metadata.csv and report what would be uploaded
    python hf/upload_to_hf.py --repo-id <user>/jpeg-confound-control-dataset --dry-run

    # lightweight release: card + index + first 25 images' arms (good for a blog link)
    python hf/upload_to_hf.py --repo-id <user>/jpeg-confound-control-dataset --sample 25

    # full ~3.2 GB upload (private)
    python hf/upload_to_hf.py --repo-id <user>/jpeg-confound-control-dataset --private
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


# --------------------------------------------------------------------------- #
# Config / manifest
# --------------------------------------------------------------------------- #


def load_data_dir() -> Path:
    """Read data_dir from config.yaml (default 'data')."""
    cfg_path = REPO_ROOT / "config.yaml"
    data_dir = "data"
    if cfg_path.exists():
        try:
            import yaml
            with open(cfg_path) as f:
                data_dir = (yaml.safe_load(f) or {}).get("data_dir", "data")
        except Exception:
            pass
    return REPO_ROOT / data_dir


def load_manifest() -> dict:
    """image_id -> (wnid, imagenet_index)."""
    path = REPO_ROOT / "manifest.csv"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found — run generate.py first.")
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r["image_id"]] = (r.get("wnid", ""), r.get("imagenet_index", ""))
    return out


def ordered_image_ids() -> list:
    path = REPO_ROOT / "manifest.csv"
    with open(path) as f:
        return [r["image_id"] for r in csv.DictReader(f)]


# --------------------------------------------------------------------------- #
# metadata.csv (one row per arm file)
# --------------------------------------------------------------------------- #


def build_metadata(data_dir: Path) -> Path:
    """Walk the four arm trees and write metadata.csv at the repo root."""
    labels = load_manifest()
    rows = []

    def rel(p: Path) -> str:
        return str(p.relative_to(REPO_ROOT))

    def lab(image_id: str):
        return labels.get(image_id, ("", ""))

    # clean_png
    for p in sorted((data_dir / "base").glob("*.png")):
        wnid, idx = lab(p.stem)
        rows.append((rel(p), p.stem, "clean_png", "clean", 0, "", wnid, idx))

    # clean_jpeg
    for qdir in sorted((data_dir / "clean_saves").glob("jpeg_q*")):
        q = int(qdir.name.replace("jpeg_q", ""))
        for p in sorted(qdir.glob("*.jpg")):
            wnid, idx = lab(p.stem)
            rows.append((rel(p), p.stem, "clean_jpeg", "clean", 0, q, wnid, idx))

    # corrupt arms
    corr_root = data_dir / "corrupted"
    if corr_root.is_dir():
        for cdir in sorted(p for p in corr_root.iterdir() if p.is_dir()):
            corruption = cdir.name
            for sdir in sorted(cdir.glob("s*")):
                try:
                    sev = int(sdir.name[1:])
                except ValueError:
                    continue
                for p in sorted((sdir / "png").glob("*.png")):
                    wnid, idx = lab(p.stem)
                    rows.append((rel(p), p.stem, "corrupt_png", corruption, sev, "", wnid, idx))
                for qdir in sorted(sdir.glob("jpeg_q*")):
                    q = int(qdir.name.replace("jpeg_q", ""))
                    for p in sorted(qdir.glob("*.jpg")):
                        wnid, idx = lab(p.stem)
                        rows.append((rel(p), p.stem, "corrupt_jpeg", corruption, sev, q, wnid, idx))

    out = REPO_ROOT / "metadata.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file_name", "image_id", "arm", "corruption",
                    "severity", "jpeg_quality", "wnid", "imagenet_index"])
        w.writerows(rows)
    print(f"[metadata] wrote {len(rows)} rows -> {out}")
    return out


# --------------------------------------------------------------------------- #
# Upload
# --------------------------------------------------------------------------- #


def arm_allow_patterns(sample: int) -> list | None:
    """Glob patterns (relative to data/) selecting the first `sample` images' arms.
    None means upload every arm."""
    if not sample:
        return None
    ids = ordered_image_ids()[:sample]
    pats = []
    for iid in ids:
        pats += [f"**/{iid}.png", f"**/{iid}.jpg"]
    return pats


def main():
    p = argparse.ArgumentParser(description="Upload the control dataset to the Hugging Face Hub.")
    p.add_argument("--repo-id", required=True, help="e.g. <user>/jpeg-confound-control-dataset")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                   help="HF token (default: HF_TOKEN env or cached login)")
    p.add_argument("--private", action="store_true", help="create the repo as private")
    p.add_argument("--sample", type=int, default=0,
                   help="upload only the first N images' arms (0 = full dataset)")
    p.add_argument("--dry-run", action="store_true",
                   help="build metadata.csv and report; upload nothing")
    args = p.parse_args()

    data_dir = load_data_dir()
    if not (data_dir / "base").is_dir():
        sys.exit(f"ERROR: {data_dir}/base not found — run generate.py first.")

    build_metadata(data_dir)
    allow = arm_allow_patterns(args.sample)
    ignore = ["imagenette2-320/**", "*.tgz", "*.json"]  # never upload the source download/cache

    n_arms = sum(1 for _ in (data_dir.rglob("*.png"))) + sum(1 for _ in (data_dir.rglob("*.jpg")))
    scope = f"first {args.sample} images" if args.sample else "full dataset"
    print(f"[plan] repo={args.repo_id} ({'private' if args.private else 'public'})  "
          f"scope={scope}  arms on disk≈{n_arms}")
    print("[plan] uploading: README.md (card), config.yaml, manifest.csv, metadata.csv, data/ arms")

    if args.dry_run:
        print("[dry-run] nothing uploaded. metadata.csv has been written.")
        return

    from huggingface_hub import HfApi, create_repo

    create_repo(args.repo_id, repo_type="dataset", private=args.private,
                exist_ok=True, token=args.token)
    api = HfApi(token=args.token)

    # 1) card + small index/config files (explicit, so the project README is not used as the card)
    api.upload_file(path_or_fileobj=str(HERE / "README.md"), path_in_repo="README.md",
                    repo_id=args.repo_id, repo_type="dataset",
                    commit_message="Add dataset card")
    for fn in ["config.yaml", "manifest.csv", "metadata.csv"]:
        fp = REPO_ROOT / fn
        if fp.exists():
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=fn,
                            repo_id=args.repo_id, repo_type="dataset",
                            commit_message=f"Add {fn}")

    # 2) the image arms. The Hub allows at most 25k files per commit, so a full
    # upload (~92k files) must be split: one commit for the clean arms, then one
    # per corruption (~6k files each). Already-uploaded blobs are deduplicated.
    def push(patterns, msg):
        api.upload_folder(folder_path=str(data_dir), path_in_repo="data",
                          repo_id=args.repo_id, repo_type="dataset",
                          allow_patterns=patterns, ignore_patterns=ignore,
                          commit_message=msg)
        print(f"[upload] committed: {msg}")

    if args.sample:
        print("[upload] sample arms (single commit) ...")
        push(allow, f"Add image arms ({scope})")
    else:
        print("[upload] clean arms ...")
        push(["base/**", "clean_saves/**"], "Add clean arms (clean_png, clean_jpeg)")
        corr_root = data_dir / "corrupted"
        corruptions = sorted(p.name for p in corr_root.iterdir() if p.is_dir())
        for i, c in enumerate(corruptions, 1):
            print(f"[upload] corruption {i}/{len(corruptions)}: {c} ...")
            push([f"corrupted/{c}/**"], f"Add corrupt arms: {c}")

    print(f"[done] https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
