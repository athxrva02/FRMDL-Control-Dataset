# JPEG Re-encoding Confound Control Dataset

A controlled-experiment dataset that isolates one acknowledged-but-unmeasured confound in
**ImageNet-C**. Hendrycks & Dietterich (*Benchmarking Neural Network Robustness to Common
Corruptions and Perturbations*, ICLR 2019, [arXiv:1903.12261](https://arxiv.org/abs/1903.12261))
state in **Section 4.1** that ImageNet-C images are saved as lightly compressed JPEGs, so an
image corrupted by Gaussian noise is *also* slightly corrupted by JPEG compression. The benchmark
therefore never measures a corruption `c` in isolation — it measures `JPEG(c(x))`. This project
quantifies **how much of each corruption's reported error is attributable to the corruption versus
the final JPEG save step**, and whether that leakage differs across corruption families (noise,
blur, weather, digital). It maps onto storyline question **5(a) — "is the score fair and
unbiased?"**

## The control structure (one isolated variable)

For each base image, reduced once to a canonical 224×224 RGB array, we build matched arms:

| arm | what it is |
|---|---|
| `clean_png` | canonical image saved losslessly as PNG — **the control arm** |
| `clean_jpeg_q` | canonical image saved as JPEG at quality `q` — isolates JPEG acting **alone** |
| `corrupt_png` | corrupted array saved losslessly as PNG |
| `corrupt_jpeg_q` | the **same** corrupted array saved as JPEG at quality `q` |

The **only** thing that differs between `corrupt_png` and `corrupt_jpeg_q` is the save format.
Same source image, same corruption, same severity, same generated array, same random seed. Any
measured difference is therefore attributable to the JPEG save step and nothing else. The
corruption array for a given `(image, corruption, severity)` is generated **exactly once** and then
saved every way — it is never regenerated per format. That one-array rule is the core of the
experiment's validity.

The corruption functions themselves are the **original** ImageNet-C code, used via the
[`imagecorruptions`](https://github.com/bethgelab/imagecorruptions) package
(`from imagecorruptions import corrupt`). The only logic this project adds is
`Image.save(format='PNG')` vs `Image.save(format='JPEG', quality=q)`.

## Methodological note (stated plainly, per design)

ImageNet source images — and therefore ImageNette — are themselves stored as JPEG. There is **no
lossless original** for ImageNet-derived data. This does **not** confound the experiment: the
pre-existing source compression is held identical across the `corrupt_png` and `corrupt_jpeg_q`
arms, so it cancels in the comparison. The variable this dataset isolates is specifically the
**final save-time JPEG re-encode** that the paper describes, not the source compression. Stating
this strengthens the design rather than weakening it.

## What was verified, not assumed

- **ImageNet-C JPEG quality.** Read directly from `make_imagenet_c.py` in
  [github.com/hendrycks/robustness](https://github.com/hendrycks/robustness):
  `Image.fromarray(np.uint8(img)).save(save_path, quality=85, optimize=True)`.
  The ImageNet-C default is therefore **quality 85**. We sweep `{75, 90}` **plus** 85, and label
  85 as the ImageNet-C default in all outputs. (We do not pass `optimize=True`; it changes Huffman
  table selection only, not the quantization that drives the lossy difference. All other JPEG save
  parameters — subsampling, qtables — are PIL defaults. Documented here as required.)
- **`corrupt()` contract.** Confirmed from the installed package: it takes a NumPy **uint8** array
  of shape `(H, W, 3)` in `[0, 255]`, severity in `1..5`, and returns a uint8 array of the same
  shape.
- **wnid → ImageNet-1k index map.** Fetched `imagenet_class_index.json` and asserted it has exactly
  1000 entries and index `0 == ["n01440764", "tench"]` before use (never hard-coded). Cached to
  `data/imagenet_class_index.json`. Fallbacks documented in `src/common.py`.
- **torchvision weights API.** `weights=` enums and `weights.transforms()` confirmed present;
  normalization mean/std read from the transform (all three models use the standard ImageNet
  values), with the standard-ImageNet fallback wired in.
- **ImageNette URL.** Taken from the official `fastai/imagenette` repo:
  `https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz` (320px variant).

## Accepted simplification

The canonical image is built **once** (open → RGB → resize shorter side to 256 bilinear →
center-crop 224×224). The model input transform applies **only** `ToTensor` + `Normalize` — no
second resize or crop. Different torchvision weights use slightly different official eval resize
sizes (256/232/236), so absolute accuracy is marginally off each model's official protocol. This
does not matter: the PNG and JPEG arms receive **identical** geometry, and the controlled contrast
is strictly between arms. Likewise, labelled correctness uses argmax over **all 1000 classes** (the
honest hard metric), not the 10 ImageNette classes.

## Compatibility notes

`imagecorruptions` is 2019-era and pulls `scikit-image`/`opencv`. It does not run cleanly on
NumPy 2.x (removed dtype aliases), so **NumPy is pinned `<2`** (`numpy==1.26.4`). The package is
installed **from source** rather than the PyPI wheel so the bundled `frost` overlay images are
present — the plain wheel ships without them and `frost` fails. With this setup all **15**
corruptions (including `frost`) generate successfully. Built and tested on macOS arm64, Python 3.11,
torch 2.12 with the **MPS** backend (CUDA/CPU also supported; the device is auto-selected).

## Installation

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

The whole pipeline is wrapped by a driver script:

```bash
./run_all.sh              # full default scope (N=500): generate -> evaluate -> analyze
./run_all.sh 100          # quicker real run at N=100 images
./run_all.sh 500 --fresh  # wipe previously generated arms first (clean slate)
```

Full default scope is ~2–4h on Apple-silicon MPS and ~3–4 GB of disk; per-stage logs land in
`logs/`. Or run the stages individually:

```bash
# 1. Generate the dataset (downloads ImageNette on first run).
python src/generate.py                        # 500 images, 15 corruptions, s∈{1,3,5}, q∈{75,85,90}
python src/generate.py --smoke --n-images 20  # quick sanity run: 3 corruptions

# 2. Evaluate three ImageNet-pretrained models (inference only, no training).
python src/evaluate.py                        # writes results/predictions.csv

# 3. Aggregate metrics + ImageNet-C-style mCE + figures.
python src/analyze.py                         # writes results/*.csv and results/figures/*.png
```

All scope knobs live in `config.yaml` and can be overridden on the CLI
(`--n-images`, `--severities`, `--jpeg-qualities`, `--seed`, `--out-dir`, `--models`, `--device`,
`--limit`). `generate.py` is idempotent: existing outputs are skipped unless `--overwrite` is set.

## Dataset layout

```
data/
  base/<image_id>.png                              # canonical 224×224 clean image == clean_png arm
  clean_saves/jpeg_q<q>/<image_id>.jpg             # clean_jpeg_q arm
  corrupted/<corruption>/s<severity>/
    png/<image_id>.png                             # corrupt_png arm
    jpeg_q<q>/<image_id>.jpg                        # corrupt_jpeg_q arm
manifest.csv                                       # one row per base image (schema below)
results/
  predictions.csv            # per (model, image, corruption, severity, q) arm predictions + metrics
  metrics_per_corruption.csv # per (model, corruption, q): save_gap, pure_jpeg_effect, interaction, ...
  image_space.csv            # per (corruption, severity, q): MAE/RMSE/PSNR of the JPEG delta
  mce.csv                    # mCE_png / mCE_jpeg / delta per model
  figures/                   # save_gap, flip_rate, radial spectra, mCE scatter
```

`<image_id>` is a stable id (e.g. `img_0001`) assigned deterministically at selection time. The
`base/` PNG doubles as the `clean_png` arm and is not duplicated.

**`manifest.csv` schema:** `image_id`, `source_path`, `wnid`, `imagenet_index` (0–999, empty if
labels unresolved), `base_png_path`.

## Determinism

- Image selection is seeded (`seed` in config); ids are assigned after a stable `(wnid, filename)`
  sort, so the same seed yields the same set and the same ids.
- Before every `corrupt()` call we derive an **order-independent** 32-bit seed from
  `(image_id, corruption, severity)` via `zlib.crc32`, then seed both `numpy.random` and `random`.
  Re-running `generate.py` with the same seed produces **byte-identical** PNG outputs.

## Metrics

**Label-free** (always computed): `flip` (argmax changes between arms), `kl` (KL divergence,
PNG arm as reference), `prob_l2` (L2 between softmax vectors). **Labelled** (skipped with a warning
if the wnid map is unresolved): top-1 `correct` over all 1000 classes for every arm. Aggregations:
`save_gap = err_corrupt_jpeg − err_corrupt_png`, `pure_jpeg_effect = err_clean_jpeg − err_clean_png`,
`interaction = save_gap − pure_jpeg_effect`, plus the ImageNet-C `mCE` normalized by AlexNet on this
same dataset, computed once on the PNG arm and once on the JPEG arm (with the model ranking under
each, to test whether the save step changes the score *or* the ordering).

## Pre-registered expected outcomes

Stated before running (van Gemert: state the expected answer first):

- **Pure JPEG on clean images at high quality:** near-zero error change.
- **Noise corruptions** (`gaussian_noise`, `shot_noise`, `impulse_noise`) **+ JPEG:** JPEG
  quantizes high-frequency content, so it should *attenuate* the noise → `corrupt_jpeg` slightly
  **easier** than `corrupt_png` (negative `save_gap`). Interpretation: ImageNet-C's JPEG save
  slightly **understates** pure noise severity.
- **Blur / weather / digital:** JPEG adds blocking on an already low-frequency image → a smaller,
  more mixed-sign `save_gap`.
- `flip_rate` and `mean_kl` clearly **non-zero** for noise corruptions — the save step measurably
  moves model behavior even where the error gap is modest.
- **Image-space:** the radially averaged spectrum of the `delta` array concentrates energy at
  **high frequencies** for noise corruptions.

The deliverable is the **heterogeneity across corruptions**, not one number. Results that contradict
these predictions are findings to report, not failures.

> Smoke-run sanity (20 images, 3 corruptions): JPEG-delta MAE was **46.5 for `gaussian_noise`** vs
> **0.83 for `defocus_blur`** at q=85, and noise `flip_rate` reached 0.2–0.47 — the predicted
> heterogeneity is already visible at tiny scale.

## Releasing (rubric)

The code + `config.yaml` + `requirements.txt` + `manifest.csv` + `README.md` fully specify the
dataset, which is reproducible from `generate.py` + the seed. A small sample of generated images as
a GitHub Release asset is an acceptable minimal release; the full set can be regenerated by anyone.
