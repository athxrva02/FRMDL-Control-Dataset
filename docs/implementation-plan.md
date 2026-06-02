# Implementation Spec: JPEG Re-encoding Confound Control Dataset

**Purpose of this document.** This is a complete, exact build spec to hand to Claude Code. It describes a control dataset and an evaluation pipeline that measure how much of ImageNet-C's reported corruption difficulty is attributable to the JPEG save step rather than the named corruption. Follow it literally. Where a value cannot be assumed, this spec says so explicitly and tells you to discover and verify it; do not fill those gaps from memory.

---

## 1. The property being tested

The paper *Benchmarking Neural Network Robustness to Common Corruptions and Perturbations* (Hendrycks and Dietterich, ICLR 2019, arXiv:1903.12261) states in Section 4.1 that ImageNet-C images are saved as lightly compressed JPEGs, and notes that this implies an image corrupted by Gaussian noise is also slightly corrupted by JPEG compression. The benchmark therefore never measures a corruption `c` in isolation; it measures `JPEG(c(x))`.

**Property to test:** for each ImageNet-C corruption, how much of the measured Corruption Error is attributable to the named corruption versus the final JPEG save step, and does this leakage differ in size and sign across corruption families (noise, blur, weather, digital)?

This maps onto experimental question 5(a) in the storyline ("is the score fair and unbiased?").

---

## 2. Scientific design

### 2.1 The arms

For each base image, after it has been reduced to a canonical 224x224 RGB array (Section 4.3):

- `clean_png`: the canonical image saved losslessly as PNG. This is the control arm.
- `clean_jpeg_q`: the canonical image saved as JPEG at quality `q`. Isolates the JPEG save step acting alone, with no corruption.
- For each corruption `c` and severity `s`: the corrupted array is generated once, then saved two ways:
  - `corrupt_png`: corrupted array saved losslessly as PNG.
  - `corrupt_jpeg_q`: the same corrupted array saved as JPEG at quality `q`.

### 2.2 The isolated variable

The only thing that differs between `corrupt_png` and `corrupt_jpeg_q` is the save format. Same source image, same corruption, same severity, same generated array, same random seed. Any measured difference between these two arms is therefore attributable to the JPEG save step and nothing else.

### 2.3 Methodological note that MUST be stated in the blog

ImageNet source images (and therefore ImageNette) are themselves stored as JPEG. There is no lossless original available for ImageNet-derived data. This is fine and does not confound the experiment: the pre-existing JPEG compression of the source is held identical across both the `corrupt_png` and `corrupt_jpeg_q` arms, so it cancels in the comparison. The variable this dataset isolates is specifically the *final save-time JPEG re-encode* that the paper describes, not the source compression. The blog must say this plainly. Stating it strengthens the submission rather than weakening it.

### 2.4 Why this is a valid control dataset

One isolated variable. Ground truth is correct by construction (the corruption and the save format are both fully specified by us). It has a true control arm (`clean_png`). It is fully scripted and reproducible from a seed. Expected outcomes are pre-registered in Section 11.

---

## 3. Technology stack: explicit and mandatory

Use exactly these. Do not substitute equivalents without flagging it.

| Component | Tool | Notes |
|---|---|---|
| Corruption functions | `imagecorruptions` (pip) | Packages Hendrycks and Dietterich's original ImageNet-C corruption code. Exposes `corrupt(image, corruption_name=..., severity=...)` and `get_corruption_names()`. |
| Base images | ImageNette (fast.ai) | 10-class ImageNet subset, validation split only. |
| Models | `torchvision.models` | ImageNet-pretrained weights via the `weights=` enum API. No training. |
| Image IO and saving | Pillow (`PIL`) | `Image.save(path, format='PNG')` and `Image.save(path, format='JPEG', quality=q)`. |
| Numerics | `numpy`, `scipy` | FFT for the image-space analysis. |
| Tables | `pandas` | All result CSVs. |
| Plots | `matplotlib` | All figures. |
| Progress | `tqdm` | Optional but recommended. |

`imagecorruptions` will pull its own dependencies (including `scikit-image` and `opencv`). Do not fight that; let pip resolve them.

### 3.1 requirements.txt

Produce a `requirements.txt`. Use these package names. Pin a working set after install; do not invent version numbers up front.

```
torch
torchvision
imagecorruptions
numpy
scipy
pandas
matplotlib
pillow
tqdm
pyyaml
```

---

## 4. Data

### 4.1 Base images: ImageNette

Download the ImageNette validation split. Obtain the current download URL from the official `fastai/imagenette` GitHub repository rather than hard-coding a URL, since the repo lists the maintained links. The 160px or 320px variant is sufficient; do not use the full-resolution variant (unnecessarily large).

Alternative if `torchvision` in the environment is recent enough: `torchvision.datasets.Imagenette(root, split='val', download=True)` may handle the download. If you use it, inspect the returned object's attributes to recover the per-class wnid; do not assume attribute names.

### 4.2 Labels

ImageNette folders are named by ImageNet wnid (for example `n01440764`). Models output 1000-way logits, so each image needs its integer ImageNet-1k class index.

Build the wnid to index map as follows, and **verify it with assertions**:

1. Obtain the standard ImageNet class index file (`imagenet_class_index.json`), which maps `index -> [wnid, human_label]` for all 1000 classes. This file is widely mirrored. Fetch it, then assert: it has exactly 1000 entries, and index `0` is `["n01440764", "tench"]`. If either assertion fails, do not proceed with that file.
2. Invert it to get `wnid -> index`.
3. Fallback if the file cannot be obtained: the ImageNet-1k index of a wnid equals its rank in the lexicographically sorted list of all 1000 wnids. This is the convention `torchvision.datasets.ImageNet` uses.
4. Fallback if labels still cannot be resolved: the pipeline must still run and produce all label-free metrics (Section 7.3). Labeled metrics are then skipped with a logged warning.

Sample a fixed number of base images: default `N_IMAGES = 500`, stratified roughly evenly across the 10 ImageNette classes. Selection must be deterministic (seeded). Record every selected image in the manifest.

### 4.3 Canonical image step

For every selected source image, produce one canonical 224x224 RGB uint8 array, exactly once, by:

1. `Image.open(path).convert('RGB')`.
2. Resize so the shorter side is 256 pixels, using bilinear resampling (`Image.Resampling.BILINEAR`).
3. Center-crop 224x224.

This canonical array is the input to every corruption and every save arm. Corruptions are applied at 224x224, matching ImageNet-C convention.

**Accepted simplification, state it in the README:** the model input transform (Section 7.1) applies only normalization, not a second resize or crop. Different torchvision weights use slightly different official eval resize sizes, so absolute accuracy here will be marginally off each model's official protocol. This does not matter, because the `png` and `jpeg` arms receive identical geometry, and the controlled contrast is between arms.

---

## 5. Dataset specification

### 5.1 Corruptions

Use the 15 main ImageNet-C corruptions. Retrieve the names from `imagecorruptions.get_corruption_names()` rather than typing them in. For reference they are: `gaussian_noise`, `shot_noise`, `impulse_noise`, `defocus_blur`, `glass_blur`, `motion_blur`, `zoom_blur`, `snow`, `frost`, `fog`, `brightness`, `contrast`, `elastic_transform`, `pixelate`, `jpeg_compression`.

Known issue to handle: the `frost` corruption requires `imagecorruptions` to be installed from source (it needs bundled frost overlay images) and may fail under a plain `pip install`. The generation script must catch a failure on any corruption, log a warning naming the corruption, and continue. Do not let one corruption abort the run. If `frost` fails, either install `imagecorruptions` from its source repository or proceed with 14 corruptions and note it.

The corruption `jpeg_compression` is itself a JPEG operation. Keep it in the set; it is a legitimate and interesting edge case for this study. Note it in the blog.

### 5.2 Severities

Use `SEVERITIES = [1, 3, 5]` by default (integers, the values `imagecorruptions` expects). Three levels keeps runtime within budget while spanning mild to severe. The script must accept a different list via config.

### 5.3 JPEG qualities

Use `JPEG_QUALITIES = [75, 90]` by default. Additionally:

- Inspect the ImageNet-C corruption-creation script in `github.com/hendrycks/robustness` (the file under the `ImageNet-C` directory that creates the corrupted images, e.g. a `make_imagenet_c`-style script) to find the exact `Image.save(...)` call and the JPEG quality it uses. **Do not assume a value.** If you find it, add that exact quality to the list and label it in outputs as the ImageNet-C default.
- If you cannot locate it, proceed with `[75, 90]` and state in the README that the ImageNet-C default quality could not be confirmed, which is why a range is swept.

Use PIL defaults for all other JPEG save parameters (subsampling, qtables). Document this.

### 5.4 Determinism

The corruption functions use random number generators internally. Make every corrupted array reproducible and order-independent:

- Before each `corrupt()` call, derive a deterministic 32-bit seed from the tuple `(image_id, corruption_name, severity)`, for example `zlib.crc32` of the joined string masked to 32 bits, then call both `numpy.random.seed(seed)` and `random.seed(seed)`.
- The corrupted array for a given `(image, corruption, severity)` is generated exactly once. The PNG arm and every JPEG-quality arm are produced by saving that one array. Never regenerate the corruption per save format. This rule is the core of the experiment's validity.

The corrupted array must be uint8 in `[0, 255]` before saving. If `corrupt()` returns a float array, clip to `[0, 255]` and cast to uint8.

### 5.5 Directory layout

```
jpeg-confound-control/
  README.md
  requirements.txt
  config.yaml
  src/
    common.py        # seeding, IO, label map, config loading
    generate.py
    evaluate.py
    analyze.py
  data/
    base/
      <image_id>.png                         # canonical 224x224 clean image
    clean_saves/
      jpeg_q<q>/<image_id>.jpg
    corrupted/
      <corruption>/
        s<severity>/
          png/<image_id>.png
          jpeg_q<q>/<image_id>.jpg
  manifest.csv
  results/
    predictions.csv
    metrics_per_corruption.csv
    image_space.csv
    figures/
```

`<image_id>` is a short stable id assigned at selection time (for example `img_0001`). The `base/` PNG doubles as the `clean_png` arm; do not duplicate it.

### 5.6 manifest.csv schema

One row per selected base image.

| column | meaning |
|---|---|
| `image_id` | assigned stable id |
| `source_path` | path within the downloaded ImageNette tree |
| `wnid` | ImageNet wnid from the folder name |
| `imagenet_index` | integer 0-999, or empty if labels unresolved |
| `base_png_path` | path to the canonical PNG |

---

## 6. generate.py specification

**Inputs:** `config.yaml` (with CLI overrides for any key).

**Algorithm:**

1. Load config. Resolve and verify the label map (Section 4.2).
2. Select `N_IMAGES` base images deterministically, stratified across the 10 classes. Write `manifest.csv`.
3. For each selected image: build the canonical 224x224 array (Section 4.3), save it to `data/base/<image_id>.png`, and save `clean_jpeg_q` for each quality.
4. For each `(image, corruption, severity)`:
   a. Seed deterministically (Section 5.4).
   b. Call `corrupt()` on the canonical array. On exception, log a warning with the corruption name and continue.
   c. Cast result to uint8 `[0, 255]`.
   d. Save once as `corrupt_png`, and once per JPEG quality as `corrupt_jpeg_q`.
5. Print a summary: counts of files written, corruptions skipped, total disk used.

**CLI args:** `--config`, plus overrides `--n-images`, `--severities`, `--jpeg-qualities`, `--out-dir`, `--seed`.

**Idempotence:** if an output file already exists and `--overwrite` is not set, skip it.

---

## 7. evaluate.py specification

No training. Inference only.

### 7.1 Models and preprocessing

Load three `torchvision` ImageNet-pretrained models via the `weights=` enum API:

- AlexNet (include it: the paper normalizes Corruption Error by AlexNet).
- ResNet-50.
- One modern model, for example ConvNeXt-Tiny or ViT-B/16.

For each model, obtain its preprocessing from `weights.transforms()`. From that transform object read the normalization `mean` and `std`. If those attributes are not accessible in the installed torchvision version, fall back to the standard ImageNet values `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`, and assert the transform is a standard ImageNet classification transform.

Model input transform: `ToTensor` then `Normalize(mean, std)`. No resize, no crop (already 224x224 from Section 4.3). Set every model to `.eval()` and run under `torch.no_grad()`. Use GPU if available, else CPU.

### 7.2 What to evaluate

Every arm must be loaded from its saved file on disk, so the JPEG arms actually pass through a real JPEG decode. For each `(model, image, corruption, severity)`:

- Run the model on `corrupt_png` and on each `corrupt_jpeg_q`.
- Also run it on `clean_png` and each `clean_jpeg_q` (once per image and model, not once per corruption).
- For each run, record the argmax class index, the top-1 softmax probability, and the full softmax vector in memory long enough to compute the metrics below (do not write 1000-dim vectors to CSV).

### 7.3 Metrics

**Label-free (must work even if labels are unresolved):**

- `flip`: 1 if `argmax(corrupt_png) != argmax(corrupt_jpeg_q)`, else 0.
- `kl`: KL divergence between the two softmax distributions, `corrupt_png` as reference.
- `logit_l2` or `prob_l2`: L2 distance between the two output vectors.

**Labeled (skip with a warning if labels unresolved):**

- `correct`: 1 if argmax over all 1000 classes equals `imagenet_index`. Compute for `clean_png`, `clean_jpeg_q`, `corrupt_png`, `corrupt_jpeg_q`.
- Note in the README that argmax is taken over all 1000 classes (the honest hard metric), not restricted to the 10 ImageNette classes.

### 7.4 Output: predictions.csv

Long format, one row per `(model, image, corruption, severity, jpeg_quality)`:

| column | meaning |
|---|---|
| `model`, `image_id`, `corruption`, `severity`, `jpeg_quality` | keys |
| `pred_png`, `pred_jpeg` | argmax class indices |
| `correct_png`, `correct_jpeg` | 0/1, or empty if labels unresolved |
| `maxprob_png`, `maxprob_jpeg` | top-1 softmax probability |
| `flip`, `kl`, `prob_l2` | label-free arm-difference metrics |

Clean-arm results (`clean_png` vs `clean_jpeg_q`) go in the same file with `corruption='clean'` and `severity=0`.

---

## 8. analyze.py specification

### 8.1 Image-space characterization (label-free)

For each `(corruption, severity, jpeg_quality)`, over all images, compute the difference array `delta = corrupt_jpeg.astype(float) - corrupt_png.astype(float)` and report:

- Mean absolute pixel difference (MAE) and RMSE.
- PSNR between the two arms.
- Radially averaged magnitude spectrum of `delta`: convert `delta` to luminance, take the 2D FFT, `fftshift`, magnitude, then radially average into a 1D curve. Average that curve across images.

Write `image_space.csv` with the scalar metrics. Save the radial spectra as a figure.

### 8.2 Model-space aggregation

Write `metrics_per_corruption.csv`, one row per `(model, corruption, jpeg_quality)` aggregated over severities and images:

- `err_clean_png`, `err_clean_jpeg`
- `err_corrupt_png`, `err_corrupt_jpeg`
- `save_gap = err_corrupt_jpeg - err_corrupt_png`
- `pure_jpeg_effect = err_clean_jpeg - err_clean_png`
- `interaction = save_gap - pure_jpeg_effect`
- `flip_rate`, `mean_kl` between the corrupt arms

### 8.3 ImageNet-C style aggregation (ties directly to the paper)

Using the paper's Corruption Error definition, compute, per model and per corruption, a Corruption Error normalized by AlexNet's errors on this same dataset, once using the PNG arm and once using the JPEG arm. Average across the 15 corruptions to get an mCE-style score `mCE_png` and `mCE_jpeg` per model. Report both numbers and the model ranking under each. The headline question is whether the JPEG save step changes either the numbers or the ranking.

### 8.4 Figures (save to results/figures/)

- Bar chart of `save_gap` per corruption, grouped by family, faceted by JPEG quality.
- Bar chart of `flip_rate` per corruption.
- Radial `delta` spectra, noise corruptions versus blur corruptions overlaid.
- Scatter of `mCE_png` versus `mCE_jpeg` across models.

---

## 9. README.md (in the repo, written by Claude Code)

Must include: one-paragraph statement of the property and the paper citation with the Section 4.1 reference; the exact commands to install, generate, evaluate, and analyze; the dataset directory layout; the determinism scheme; the accepted simplification from Section 4.3; the methodological note from Section 2.3; and whether the ImageNet-C default JPEG quality was confirmed.

For the assignment rubric, the dataset and code links are satisfied by: a public GitHub or GitLab repo containing all of `src/`, `config.yaml`, `requirements.txt`, `manifest.csv`, and `README.md`; and the generated images hosted as a GitHub Release asset or a dataset host. Because the dataset is fully reproducible from `generate.py` plus the seed plus `manifest.csv`, the script itself plus a small sample of images is an acceptable minimal release.

---

## 10. Acceptance criteria and sanity checks

The build is correct only if all of these hold:

1. Loading `corrupt_png` back from disk returns exactly the array that was saved (PNG is lossless). Assert this on a sample.
2. `corrupt_jpeg_q` differs from `corrupt_png` (JPEG is lossy). Assert non-zero MAE on a sample.
3. Re-running `generate.py` with the same seed produces byte-identical PNG outputs. Assert on a sample.
4. `pure_jpeg_effect` on clean images at high quality is small. A large value indicates a pipeline bug.
5. For at least the noise corruptions, `flip_rate` and `mean_kl` between the corrupt arms are clearly above zero. A flat zero indicates the JPEG arm is not actually being decoded from JPEG.
6. Every corruption that did not error appears in `metrics_per_corruption.csv` for every model.

---

## 11. Pre-registered expected outcomes

Write these into the blog before running anything (van Gemert: state the expected answer before the experiment).

- Pure JPEG on clean images at high quality: near-zero error change.
- Noise corruptions (`gaussian_noise`, `shot_noise`, `impulse_noise`) plus JPEG save: JPEG quantizes high-frequency content, so it should attenuate the noise. Expect `corrupt_jpeg` to be slightly easier than `corrupt_png`, that is a negative `save_gap`. Interpretation: the ImageNet-C JPEG save slightly understates pure noise severity.
- Blur, weather, and digital corruptions: JPEG adds blocking artifacts on top of an already low-frequency image. Expect a smaller and more mixed-sign `save_gap`.
- `flip_rate` and `mean_kl` clearly non-zero for noise corruptions, showing the save step measurably moves model behavior even when the error gap is modest.
- Image-space: the radially averaged spectrum of the `delta` array concentrates energy at high frequencies for noise corruptions.

If results contradict these predictions, that is a finding to report, not a failure.

---

## 12. Things Claude Code MUST verify, not assume

1. **JPEG quality used by ImageNet-C.** Discover it from the `hendrycks/robustness` corruption-creation script. Do not state a number from memory. Sweep a range regardless.
2. **The wnid to ImageNet-index label map.** Fetch the standard mapping file and verify it with the assertions in Section 4.2. Do not hard-code the mapping.
3. **`imagecorruptions` compatibility.** This package dates from 2019 and may not import or run cleanly against current NumPy or scikit-image (for example NumPy 2.x removed deprecated aliases). If it fails, pin compatible dependency versions (such as an older NumPy) in `requirements.txt`, or install `imagecorruptions` from source, and document what was needed.
4. **`corrupt()` input contract.** Confirm from the installed `imagecorruptions` package whether `corrupt()` expects a NumPy uint8 array or a PIL image, and the exact severity range. Do not guess.
5. **torchvision weights API.** Confirm the `weights=` enum names exist in the installed torchvision version and that `weights.transforms()` exposes the normalization parameters. Use the fallback in Section 7.1 if not.
6. **ImageNette download URL.** Take it from the official `fastai/imagenette` repository at build time rather than hard-coding a URL.

---

## 13. Time budget (target 10 to 15 hours)

| Task | Hours |
|---|---|
| Environment setup, dependency compatibility, ImageNette download | 1.5 |
| `common.py`, label map, image selection, manifest | 1.5 |
| `generate.py` and a generation run | 3.5 |
| `evaluate.py` and an evaluation run | 3.0 |
| `analyze.py`, metrics, figures | 2.0 |
| Blog draft (motivation, examples, generation description, links) | 3.0 |
| Total | 14.5 |

Keep scope at the defaults (500 images, 15 corruptions, severities [1,3,5], qualities [75,90] plus any discovered default). All scope knobs live in `config.yaml`; do not hard-code them in the scripts.
