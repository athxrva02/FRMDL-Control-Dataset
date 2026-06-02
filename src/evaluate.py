"""evaluate.py — inference-only model evaluation (Section 7).

For each model and each base image, every arm is loaded *from its saved file*
(so JPEG arms pass through a real JPEG decode), batched together, and run once.
Per (model, image, corruption, severity, jpeg_quality) we record the PNG-arm and
JPEG-arm predictions plus the label-free arm-difference metrics (flip, KL, L2)
and, when labels are resolved, correctness.

Usage:
    python src/evaluate.py [--config config.yaml] [--models alexnet,resnet50,...]
        [--device mps|cpu|cuda] [--limit N] [--smoke]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tvm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k):
        return x

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Model name -> (constructor, weights enum). One modern model (ConvNeXt-Tiny)
# alongside AlexNet (the paper's CE normaliser) and ResNet-50.
MODEL_SPECS = {
    "alexnet": (tvm.alexnet, tvm.AlexNet_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
}


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_norm(weights):
    """Read (mean, std) from weights.transforms(); fall back to ImageNet std values."""
    try:
        t = weights.transforms()
        mean, std = list(t.mean), list(t.std)
        assert len(mean) == 3 and len(std) == 3
        return mean, std
    except Exception as e:
        print(f"[norm] falling back to standard ImageNet mean/std ({e!r})")
        return IMAGENET_MEAN, IMAGENET_STD


def normalize_batch(arrs: np.ndarray, mean, std, device) -> torch.Tensor:
    """arrs: (B,224,224,3) uint8 -> normalized (B,3,224,224) float tensor.
    Equivalent to ToTensor + Normalize; no resize/crop (already 224x224)."""
    x = torch.from_numpy(arrs).to(device=device, dtype=torch.float32) / 255.0
    x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
    m = torch.tensor(mean, device=device).view(1, 3, 1, 1)
    s = torch.tensor(std, device=device).view(1, 3, 1, 1)
    return (x - m) / s


def softmax_np(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=1).cpu().numpy()


# --------------------------------------------------------------------------- #
# Per-image arm enumeration
# --------------------------------------------------------------------------- #


def build_arm_plan(cfg: C.Config, corruptions, qualities):
    """Return a list describing, for one image, every (key, png_path, jpeg_paths).

    Each entry: dict(corruption, severity, png_path, jpegs={q: path}).
    Includes the clean pseudo-corruption (corruption='clean', severity=0) whose
    PNG arm is the base image and whose JPEG arms are the clean JPEG saves.
    """
    plan = []
    plan.append({"corruption": "clean", "severity": 0, "kind": "clean"})
    for corruption in corruptions:
        for s in cfg.severities:
            plan.append({"corruption": corruption, "severity": s, "kind": "corrupt"})
    return plan


def arm_paths(cfg, image_id, entry, qualities):
    if entry["kind"] == "clean":
        png = C.base_png_path(cfg, image_id)
        jpegs = {q: C.clean_jpeg_path(cfg, image_id, q) for q in qualities}
    else:
        c, s = entry["corruption"], entry["severity"]
        png = C.corrupt_png_path(cfg, c, s, image_id)
        jpegs = {q: C.corrupt_jpeg_path(cfg, c, s, image_id, q) for q in qualities}
    return png, jpegs


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def evaluate(cfg: C.Config, model_names, device, limit, smoke):
    from imagecorruptions import get_corruption_names
    corruptions = get_corruption_names()
    if smoke:
        corruptions = ["gaussian_noise", "defocus_blur", "frost"]

    # Only evaluate corruptions that were actually generated (handles skipped ones).
    corruptions = [c for c in corruptions
                   if (cfg.data_path / "corrupted" / c).is_dir()]
    qualities = cfg.effective_qualities

    images = _load_manifest(cfg, limit)
    wnid_to_index = C.resolve_label_map(cfg)
    has_labels = wnid_to_index is not None and any(r["imagenet_index"] != "" for r in images)
    if not has_labels:
        print("[labels] unresolved/empty — correctness columns will be blank.")

    plan = build_arm_plan(cfg, corruptions, qualities)
    cfg.results_path.mkdir(parents=True, exist_ok=True)
    out_path = cfg.results_path / "predictions.csv"

    fieldnames = [
        "model", "image_id", "corruption", "severity", "jpeg_quality",
        "pred_png", "pred_jpeg", "correct_png", "correct_jpeg",
        "maxprob_png", "maxprob_jpeg", "flip", "kl", "prob_l2",
    ]
    fout = open(out_path, "w", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    n_rows = 0
    for model_name in model_names:
        ctor, weights = MODEL_SPECS[model_name]
        mean, std = get_norm(weights)
        print(f"[model] {model_name} on {device} (mean={mean})")
        model = ctor(weights=weights).eval().to(device)

        for rec in tqdm(images, desc=model_name):
            image_id = rec["image_id"]
            label = rec["imagenet_index"]
            label = int(label) if (has_labels and label != "") else None

            # Gather every arm file for this image into one batch.
            batch_arrs, batch_meta = [], []
            for entry in plan:
                png, jpegs = arm_paths(cfg, image_id, entry, qualities)
                if not png.exists():
                    continue  # corruption/severity not generated
                missing_q = [q for q in qualities if not jpegs[q].exists()]
                batch_arrs.append(C.load_array(png))
                batch_meta.append(("png", entry, None))
                for q in qualities:
                    if q in missing_q:
                        continue
                    batch_arrs.append(C.load_array(jpegs[q]))
                    batch_meta.append(("jpeg", entry, q))

            if not batch_arrs:
                continue
            probs = _infer(model, np.stack(batch_arrs), mean, std, device, cfg.batch_size)

            # Index probs by (kind, corruption, severity, q).
            idx = {}
            for i, (kind, entry, q) in enumerate(batch_meta):
                idx[(kind, entry["corruption"], entry["severity"], q)] = i

            for entry in plan:
                c, s = entry["corruption"], entry["severity"]
                png_i = idx.get(("png", c, s, None))
                if png_i is None:
                    continue
                p_png = probs[png_i]
                pred_png = int(p_png.argmax())
                for q in qualities:
                    jpg_i = idx.get(("jpeg", c, s, q))
                    if jpg_i is None:
                        continue
                    p_jpg = probs[jpg_i]
                    pred_jpg = int(p_jpg.argmax())
                    writer.writerow({
                        "model": model_name,
                        "image_id": image_id,
                        "corruption": c,
                        "severity": s,
                        "jpeg_quality": q,
                        "pred_png": pred_png,
                        "pred_jpeg": pred_jpg,
                        "correct_png": "" if label is None else int(pred_png == label),
                        "correct_jpeg": "" if label is None else int(pred_jpg == label),
                        "maxprob_png": round(float(p_png[pred_png]), 6),
                        "maxprob_jpeg": round(float(p_jpg[pred_jpg]), 6),
                        "flip": int(pred_png != pred_jpg),
                        "kl": round(_kl(p_png, p_jpg), 6),
                        "prob_l2": round(float(np.linalg.norm(p_png - p_jpg)), 6),
                    })
                    n_rows += 1
        del model
        if device.type == "mps":
            torch.mps.empty_cache()

    fout.close()
    print(f"[done] wrote {n_rows} rows -> {out_path}")


def _infer(model, arrs, mean, std, device, batch_size):
    out = []
    with torch.no_grad():
        for i in range(0, len(arrs), batch_size):
            chunk = arrs[i:i + batch_size]
            x = normalize_batch(chunk, mean, std, device)
            logits = model(x)
            out.append(softmax_np(logits))
    return np.concatenate(out, axis=0)


def _kl(p, q, eps=1e-12):
    """KL(p || q) with p = png arm as reference (Section 7.3)."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def _load_manifest(cfg, limit):
    path = C.REPO_ROOT / "manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run generate.py first")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if limit:
        rows = rows[:limit]
    return rows


def main():
    p = argparse.ArgumentParser(description="Evaluate models on the control dataset.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--models", type=lambda s: s.split(","), default=None,
                   help="comma-separated subset of models")
    p.add_argument("--device", default=None, help="mps|cpu|cuda (auto if omitted)")
    p.add_argument("--limit", type=int, default=None, help="evaluate only first N images")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    cfg = C.load_config(args.config)
    model_names = args.models or cfg.models
    bad = [m for m in model_names if m not in MODEL_SPECS]
    if bad:
        raise SystemExit(f"unknown models {bad}; choices: {list(MODEL_SPECS)}")
    device = pick_device(args.device)
    evaluate(cfg, model_names, device, args.limit, args.smoke)


if __name__ == "__main__":
    main()
