---
license: other
license_name: imagenet-terms-of-access
license_link: https://image-net.org/download.php
pretty_name: JPEG Re-encoding Confound Control Dataset (ImageNet-C)
task_categories:
- image-classification
language:
- en
tags:
- robustness
- imagenet-c
- corruption-robustness
- jpeg-compression
- controlled-experiment
- confound
- benchmark-bias
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: metadata
    path: metadata.csv
---

# JPEG Re-encoding Confound Control Dataset

A **controlled-experiment** dataset that isolates one acknowledged-but-unmeasured confound in
**ImageNet-C**. Hendrycks & Dietterich (*Benchmarking Neural Network Robustness to Common
Corruptions and Perturbations*, ICLR 2019, [arXiv:1903.12261](https://arxiv.org/abs/1903.12261))
save every corrupted image as a lightly compressed JPEG. The benchmark therefore never measures a
corruption `c` applied to an image `x` in isolation — it measures `JPEG(c(x))`. This dataset lets
you quantify **how much of each corruption's reported error is attributable to the corruption
versus the final JPEG save step**, and whether that leakage differs across corruption families.

The single isolated variable is the **save format**: the same corrupted pixel array is saved once
losslessly (PNG) and once as JPEG. Everything else — image, corruption, severity, random seed — is
held identical, so any measured difference is attributable to the JPEG save step and nothing else.

## The control structure (four matched arms)

For each base image, reduced once to a canonical 224×224 RGB array:

| arm | description |
|---|---|
| `clean_png` | canonical image saved losslessly as PNG — **the control arm** |
| `clean_jpeg` | canonical image saved as JPEG at quality `q` — isolates JPEG acting **alone** |
| `corrupt_png` | corrupted array saved losslessly as PNG — the corruption in isolation |
| `corrupt_jpeg` | the **same** corrupted array saved as JPEG at quality `q` — what ImageNet-C stores |

The corruption array for a given `(image, corruption, severity)` is generated **exactly once** and
then saved every way; it is never regenerated per format. That one-array rule is the core of the
experiment's validity. The corruption functions are the **original** ImageNet-C code, used via the
[`imagecorruptions`](https://github.com/bethgelab/imagecorruptions) package — the only logic this
project adds is `Image.save(format='PNG')` vs `Image.save(format='JPEG', quality=q)`.

## Scope

- **15 corruptions** (families: noise, blur, weather, digital).
- **3 severities**: `{1, 3, 5}`.
- **3 JPEG qualities**: `{75, 85, 90}`. Quality **85** is the ImageNet-C default, read directly
  from `make_imagenet_c.py` in the Hendrycks repo (`save(..., quality=85, optimize=True)`); 75 and
  90 bracket it so results are not tied to one guess.
- **Base images**: the 10-class [ImageNette](https://github.com/fastai/imagenette) (320px)
  validation split, selected deterministically from a fixed seed.

## Files and layout

```
data/
  base/<image_id>.png                                # clean_png arm
  clean_saves/jpeg_q<q>/<image_id>.jpg               # clean_jpeg arm
  corrupted/<corruption>/s<severity>/
      png/<image_id>.png                             # corrupt_png arm
      jpeg_q<q>/<image_id>.jpg                        # corrupt_jpeg arm
manifest.csv     # one row per base image: image_id, source_path, wnid, imagenet_index, base_png_path
metadata.csv     # one row per arm file (index over the whole dataset; schema below)
config.yaml      # exact scope knobs used to generate this release
```

`<image_id>` (e.g. `img_0001`) is assigned deterministically at selection time. The `base/` PNG
doubles as the `clean_png` arm and is not duplicated.

### `metadata.csv` schema

One row per image file, so the dataset can be filtered without walking the tree:

| column | meaning |
|---|---|
| `file_name` | path relative to repo root, e.g. `data/corrupted/fog/s3/jpeg_q85/img_0001.jpg` |
| `image_id` | stable id, e.g. `img_0001` |
| `arm` | one of `clean_png`, `clean_jpeg`, `corrupt_png`, `corrupt_jpeg` |
| `corruption` | corruption name, or `clean` for the clean arms |
| `severity` | `1/3/5`, or `0` for clean arms |
| `jpeg_quality` | `75/85/90` for JPEG arms, empty for PNG arms |
| `wnid` | ImageNet wnid of the source class |
| `imagenet_index` | ImageNet-1k class index (0–999) |

## Usage

Pair each `*_png` arm with the matching `*_jpeg` arm (same `image_id`, `corruption`, `severity`)
and compare model behavior. The intended contrast is `corrupt_png` vs `corrupt_jpeg`; the
`clean_*` arms isolate JPEG acting alone (the control).

```python
import pandas as pd
from huggingface_hub import hf_hub_download

repo = "<user>/jpeg-confound-control-dataset"
meta = pd.read_csv(hf_hub_download(repo, "metadata.csv", repo_type="dataset"))

# all matched arms for fog at severity 3, quality 85
fog = meta[(meta.corruption == "fog") & (meta.severity == 3) &
           (meta.arm.isin(["corrupt_png", "corrupt_jpeg"])) &
           (meta.jpeg_quality.isin([85, ""]))]
```

Or pull the full tree:

```python
from huggingface_hub import snapshot_download
snapshot_download("<user>/jpeg-confound-control-dataset", repo_type="dataset")
```

## Reproducibility

The dataset regenerates **byte-identically** from the generation code plus the seed in
`config.yaml`: image selection is seeded, and before every corruption an order-independent 32-bit
seed is derived from `(image_id, corruption, severity)` via `zlib.crc32`. The full generator,
evaluator, and analysis are in the
[code repository](https://github.com/) (`generate.py` → `evaluate.py` → `analyze.py`).

## Intended use and limitations

- **Intended use**: measuring the JPEG save-step confound in corruption-robustness benchmarks;
  teaching controlled-experiment design. Research only.
- **No lossless ImageNet original exists** — ImageNet (and thus ImageNette) source images are
  themselves JPEG. This source compression is held identical across the PNG and JPEG arms, so it
  cancels in the comparison; the variable isolated here is the **final save-time re-encode**, not
  total JPEG exposure.
- Base images are the 10-class ImageNette subset with a single canonical crop, so absolute
  accuracies differ from the full ImageNet-C protocol; the controlled contrast between arms is
  unaffected.

## Licensing and source

- **Images** derive from ImageNette, a subset of ImageNet, and are subject to the
  [ImageNet terms of access](https://image-net.org/download.php) (non-commercial research use).
  Redistribute only as permitted by those terms; when in doubt, release **code + manifest + a
  small sample** and let users regenerate the full set.
- **Corruption functions**: [`imagecorruptions`](https://github.com/bethgelab/imagecorruptions)
  (Apache-2.0), packaging Hendrycks's original ImageNet-C code.
- **Packaging/generation code**: see the code repository's license.

## Citation

The corruptions and the confound this dataset measures:

```bibtex
@inproceedings{hendrycks2019benchmarking,
  title     = {Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author    = {Hendrycks, Dan and Dietterich, Thomas},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2019},
  url       = {https://arxiv.org/abs/1903.12261}
}
```
