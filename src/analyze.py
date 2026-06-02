"""analyze.py — metrics, ImageNet-C-style aggregation, and figures (Section 8).

Three blocks:
  8.1 Image-space characterization of the JPEG delta (label-free): MAE, RMSE,
      PSNR, and the radially averaged magnitude spectrum of (jpeg - png).
  8.2 Model-space aggregation per (model, corruption, jpeg_quality): error
      arms, save_gap, pure_jpeg_effect, interaction, flip_rate, mean_kl.
  8.3 ImageNet-C-style Corruption Error normalized by AlexNet, PNG vs JPEG arms,
      and the resulting mCE-style scores + rankings per model.

Usage:
    python src/analyze.py [--config config.yaml] [--image-sample N] [--smoke]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# Corruption -> family, for grouped figures (Section 8.4).
FAMILY = {
    "gaussian_noise": "noise", "shot_noise": "noise", "impulse_noise": "noise",
    "defocus_blur": "blur", "glass_blur": "blur", "motion_blur": "blur", "zoom_blur": "blur",
    "snow": "weather", "frost": "weather", "fog": "weather",
    "brightness": "digital", "contrast": "digital", "elastic_transform": "digital",
    "pixelate": "digital", "jpeg_compression": "digital",
}
NOISE = [c for c, f in FAMILY.items() if f == "noise"]
BLUR = [c for c, f in FAMILY.items() if f == "blur"]


# --------------------------------------------------------------------------- #
# 8.1 Image-space
# --------------------------------------------------------------------------- #


def radial_profile(mag2d: np.ndarray) -> np.ndarray:
    """Radially average a 2D (already fftshifted) magnitude spectrum -> 1D curve."""
    h, w = mag2d.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    tbin = np.bincount(r.ravel(), mag2d.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def luminance(arr: np.ndarray) -> np.ndarray:
    # Rec. 601 luma on a float array.
    return arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114


def image_space(cfg, corruptions, image_ids, qualities, sample):
    rows = []
    spectra = {}  # (corruption, severity, q) -> averaged radial curve
    sample_ids = image_ids[:sample]

    for corruption in corruptions:
        for s in cfg.severities:
            for q in qualities:
                maes, rmses, psnrs, curves = [], [], [], []
                for image_id in sample_ids:
                    png_p = C.corrupt_png_path(cfg, corruption, s, image_id)
                    jpg_p = C.corrupt_jpeg_path(cfg, corruption, s, image_id, q)
                    if not (png_p.exists() and jpg_p.exists()):
                        continue
                    png = C.load_array(png_p).astype(np.float64)
                    jpg = C.load_array(jpg_p).astype(np.float64)
                    delta = jpg - png
                    mae = np.mean(np.abs(delta))
                    rmse = np.sqrt(np.mean(delta ** 2))
                    psnr = 99.0 if rmse == 0 else 20 * np.log10(255.0 / rmse)
                    maes.append(mae); rmses.append(rmse); psnrs.append(psnr)

                    lum = luminance(delta)
                    F = np.fft.fftshift(np.fft.fft2(lum))
                    curves.append(radial_profile(np.abs(F)))

                if not maes:
                    continue
                rows.append({
                    "corruption": corruption, "family": FAMILY.get(corruption, "?"),
                    "severity": s, "jpeg_quality": q, "n_images": len(maes),
                    "mae": np.mean(maes), "rmse": np.mean(rmses), "psnr": np.mean(psnrs),
                })
                minlen = min(len(c) for c in curves)
                spectra[(corruption, s, q)] = np.mean([c[:minlen] for c in curves], axis=0)

    df = pd.DataFrame(rows)
    df.to_csv(cfg.results_path / "image_space.csv", index=False)
    print(f"[8.1] image_space.csv: {len(df)} rows")
    return df, spectra


# --------------------------------------------------------------------------- #
# 8.2 Model-space aggregation
# --------------------------------------------------------------------------- #


def model_space(cfg, pred: pd.DataFrame, has_labels: bool):
    corrupt = pred[pred.corruption != "clean"]
    clean = pred[pred.corruption == "clean"]

    # Pure JPEG effect on clean images, per (model, q): err_clean_jpeg - err_clean_png.
    clean_err = {}
    if has_labels:
        for (m, q), g in clean.groupby(["model", "jpeg_quality"]):
            clean_err[(m, q)] = (1 - g.correct_png.mean(), 1 - g.correct_jpeg.mean())

    rows = []
    for (m, c, q), g in corrupt.groupby(["model", "corruption", "jpeg_quality"]):
        row = {
            "model": m, "corruption": c, "family": FAMILY.get(c, "?"), "jpeg_quality": q,
            "n": len(g), "flip_rate": g.flip.mean(), "mean_kl": g.kl.mean(),
            "mean_prob_l2": g.prob_l2.mean(),
        }
        if has_labels:
            ecp = 1 - g.correct_png.mean()
            ecj = 1 - g.correct_jpeg.mean()
            ecln_p, ecln_j = clean_err.get((m, q), (np.nan, np.nan))
            row.update({
                "err_clean_png": ecln_p, "err_clean_jpeg": ecln_j,
                "err_corrupt_png": ecp, "err_corrupt_jpeg": ecj,
                "save_gap": ecj - ecp,
                "pure_jpeg_effect": ecln_j - ecln_p,
                "interaction": (ecj - ecp) - (ecln_j - ecln_p),
            })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["model", "family", "corruption", "jpeg_quality"])
    df.to_csv(cfg.results_path / "metrics_per_corruption.csv", index=False)
    print(f"[8.2] metrics_per_corruption.csv: {len(df)} rows")
    return df


# --------------------------------------------------------------------------- #
# 8.3 ImageNet-C-style mCE (normalized by AlexNet on THIS dataset)
# --------------------------------------------------------------------------- #


def mce_analysis(cfg, pred: pd.DataFrame, has_labels: bool):
    if not has_labels:
        print("[8.3] skipped (labels unresolved).")
        return None
    if "alexnet" not in pred.model.unique():
        print("[8.3] skipped (AlexNet not evaluated — it is the CE normaliser).")
        return None

    corrupt = pred[pred.corruption != "clean"]
    # CE_c^f = sum_s E_{s,c}^f / sum_s E_{s,c}^{alexnet}, summed over severities.
    # Build per (model, corruption, severity, arm) mean error, then sum over severities.
    def ce_table(arm_col):
        err = (1 - corrupt.groupby(["model", "corruption", "severity"])[arm_col].mean()) \
            .rename("err").reset_index()
        # sum over severities
        summed = err.groupby(["model", "corruption"]).err.sum().reset_index()
        alex = summed[summed.model == "alexnet"].set_index("corruption").err
        summed["ce"] = summed.apply(
            lambda r: r.err / alex[r.corruption] if alex[r.corruption] > 0 else np.nan, axis=1)
        return summed

    rows = []
    for arm, col in [("png", "correct_png"), ("jpeg", "correct_jpeg")]:
        t = ce_table(col)
        for m, g in t.groupby("model"):
            rows.append({"model": m, "arm": arm, "mCE": 100 * g.ce.mean()})
    mdf = pd.DataFrame(rows)
    wide = mdf.pivot(index="model", columns="arm", values="mCE").reset_index()
    wide = wide.rename(columns={"png": "mCE_png", "jpeg": "mCE_jpeg"})
    wide["delta"] = wide["mCE_jpeg"] - wide["mCE_png"]
    wide.to_csv(cfg.results_path / "mce.csv", index=False)

    rank_png = wide.sort_values("mCE_png").model.tolist()
    rank_jpeg = wide.sort_values("mCE_jpeg").model.tolist()
    print("[8.3] mCE (lower=better, AlexNet-normalized on this dataset):")
    print(wide.to_string(index=False))
    print(f"      ranking PNG : {rank_png}")
    print(f"      ranking JPEG: {rank_jpeg}")
    print(f"      ranking changed: {rank_png != rank_jpeg}")
    return wide


# --------------------------------------------------------------------------- #
# 8.4 Figures
# --------------------------------------------------------------------------- #


def figures(cfg, ms_df, img_df, spectra, mce_df, has_labels):
    figdir = cfg.results_path / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # (a) save_gap per corruption, grouped by family, faceted by quality.
    if has_labels and "save_gap" in ms_df.columns:
        for m in ms_df.model.unique():
            sub = ms_df[ms_df.model == m].sort_values(["family", "corruption"])
            qualities = sorted(sub.jpeg_quality.unique())
            fig, axes = plt.subplots(1, len(qualities), figsize=(6 * len(qualities), 5), squeeze=False)
            for ax, q in zip(axes[0], qualities):
                d = sub[sub.jpeg_quality == q]
                colors = {"noise": "C0", "blur": "C1", "weather": "C2", "digital": "C3"}
                ax.bar(d.corruption, 100 * d.save_gap, color=[colors[f] for f in d.family])
                ax.axhline(0, color="k", lw=0.6)
                ax.set_title(f"{m}  save_gap  q={q}")
                ax.set_ylabel("err(jpeg) - err(png)  [pp]")
                ax.tick_params(axis="x", rotation=90)
            fig.tight_layout()
            fig.savefig(figdir / f"save_gap_{m}.png", dpi=130)
            plt.close(fig)

    # (b) flip_rate per corruption (averaged over qualities) per model.
    fig, ax = plt.subplots(figsize=(11, 5))
    piv = ms_df.groupby(["corruption", "model"]).flip_rate.mean().unstack()
    piv = piv.reindex(sorted(piv.index, key=lambda c: (FAMILY.get(c, "z"), c)))
    piv.plot.bar(ax=ax)
    ax.set_ylabel("flip rate (png vs jpeg argmax)")
    ax.set_title("Per-corruption flip rate")
    fig.tight_layout()
    fig.savefig(figdir / "flip_rate.png", dpi=130)
    plt.close(fig)

    # (c) radial delta spectra: noise vs blur overlaid (highest severity, default q).
    q = cfg.imagenet_c_default_quality
    s = max(cfg.severities)
    fig, ax = plt.subplots(figsize=(8, 5))
    for group, color, label in [(NOISE, "C0", "noise"), (BLUR, "C1", "blur")]:
        curves = [spectra[(c, s, q)] for c in group if (c, s, q) in spectra]
        if not curves:
            continue
        minlen = min(len(c) for c in curves)
        avg = np.mean([c[:minlen] for c in curves], axis=0)
        ax.plot(np.arange(minlen), np.log1p(avg), color=color, label=label)
    ax.set_xlabel("radial spatial frequency (px^-1, low -> high)")
    ax.set_ylabel("log(1 + mean |FFT(delta)|)")
    ax.set_title(f"JPEG-delta radial spectrum (s={s}, q={q})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "radial_spectra_noise_vs_blur.png", dpi=130)
    plt.close(fig)

    # (d) mCE_png vs mCE_jpeg scatter across models.
    if mce_df is not None:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(mce_df.mCE_png, mce_df.mCE_jpeg)
        for _, r in mce_df.iterrows():
            ax.annotate(r.model, (r.mCE_png, r.mCE_jpeg))
        lim = [min(mce_df.mCE_png.min(), mce_df.mCE_jpeg.min()) - 2,
               max(mce_df.mCE_png.max(), mce_df.mCE_jpeg.max()) + 2]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_xlabel("mCE (PNG arm)")
        ax.set_ylabel("mCE (JPEG arm)")
        ax.set_title("mCE: does the JPEG save move the score?")
        fig.tight_layout()
        fig.savefig(figdir / "mce_scatter.png", dpi=130)
        plt.close(fig)

    print(f"[8.4] figures -> {figdir}")


# --------------------------------------------------------------------------- #


def main():
    p = argparse.ArgumentParser(description="Analyze predictions + dataset images.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--image-sample", type=int, default=50,
                   help="images per (corruption,severity,q) for image-space stats")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    cfg = C.load_config(args.config)

    pred_path = cfg.results_path / "predictions.csv"
    if not pred_path.exists():
        raise SystemExit(f"{pred_path} not found — run evaluate.py first")
    pred = pd.read_csv(pred_path)
    has_labels = pred.correct_png.notna().any()
    print(f"[load] predictions.csv: {len(pred)} rows | labels={'yes' if has_labels else 'no'}")

    corruptions = [c for c in pred.corruption.unique() if c != "clean"]
    if args.smoke:
        corruptions = [c for c in corruptions if c in ("gaussian_noise", "defocus_blur", "frost")]
    qualities = sorted(pred.jpeg_quality.unique())
    image_ids = sorted(pred.image_id.unique())

    img_df, spectra = image_space(cfg, corruptions, image_ids, qualities, args.image_sample)
    ms_df = model_space(cfg, pred, has_labels)
    mce_df = mce_analysis(cfg, pred, has_labels)
    figures(cfg, ms_df, img_df, spectra, mce_df, has_labels)
    print("[done] analysis complete.")


if __name__ == "__main__":
    main()
