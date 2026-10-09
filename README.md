# Insect Pose Estimation Models (SLEAP-NN, Top-Down)

This repository contains a series of trained **[SLEAP-NN](https://github.com/talmolab/sleap-nn)** models for **multi-animal pose estimation of insects** in video. The models track four body points on each animal:

```
head ── thorax ── abdomen_center ── abdomen_tip
```

The models use SLEAP's **top-down** approach, which runs two models together:

1. **Centroid model:** finds every insect in the frame by locating its **thorax** (the anchor point).
2. **Centered-instance model:** crops a small window around each detected thorax and predicts all four body points for that insect.

Every training round therefore has **one centroid run and one centered-instance run**. You need both of them to run inference.

---

## Repository structure

Each folder is one training run. Folder names follow SLEAP-NN's convention:

```
YYMMDD_HHMMSS.<model_type>.n=<number_of_labeled_frames>
```

| Round | Centroid run | Centered-instance run | Training labels file |

| 1 | `260818_004100.centroid.n=180` | `260818_033402.centered_instance.n=180` | `First_prediction_labeled_180.slp` |

---

## What each file in a run folder does

| File | Purpose |
|---|---|
| `best.ckpt` | **The trained model weights.** This is the checkpoint with the lowest validation loss during training. It is the file SLEAP-NN loads for inference. |
| `training_config.yaml` | **The full configuration the run actually used**: data paths, preprocessing, augmentation, network architecture, head type, optimizer, learning-rate schedule, early stopping and evaluation settings. SLEAP-NN reads this file together with `best.ckpt` when you load the model. |
| `initial_config.yaml` | The configuration as it was **before** training started (before SLEAP-NN filled in computed values such as the skeleton, parameter count and run name). Kept for reference and reproducibility. |
| `labels_gt.train.0.slp` | **Ground-truth labels for the training split**: the hand-labeled (or corrected) frames the model learned from. SLEAP labels format, openable in the SLEAP GUI. |
| `labels_gt.val.0.slp` | **Ground-truth labels for the validation split**: frames held out from training and used to measure accuracy. |
| `labels_pr.train.0.slp` | **The model's predictions** on the training frames after training. Compare with `labels_gt.train` to see how well the model fits its training data. |
| `labels_pr.val.0.slp` | **The model's predictions** on the validation frames. Compare with `labels_gt.val` to see how well the model generalizes. |
| `metrics.train.0.npz` | Evaluation metrics on the training split (NumPy archive): OKS mAP/mAR, mOKS, PCK, localization error percentiles and visibility metrics. |
| `metrics.val.0.npz` | The same metrics on the validation split. **Use these numbers to compare models.** |
| `training_log.csv` | Per-epoch training history: train/val loss, learning rate and epoch time. Centered-instance runs also log the loss for each body part (`train/confmaps/head`, etc.). Useful for plotting learning curves. |
| `viz/` | Prediction visualizations saved during training (`train.XXXX.png`, `validation.XXXX.png`). Only `260813_133426.centered_instance.n=63` kept its images; the other runs left this folder empty. |

## Model details

All models use a **UNet** backbone and take **grayscale** input (frames up to 1920×1080).

| | Centroid (all rounds) | Centered instance (rounds 1–5) | Centered instance (rounds 6–7) |
|---|---|---|---|
| Input scale | 0.5 | 1.0 | 1.0 |
| Crop size | – (full frame) | 192 px | 160 px |
| UNet filters / max stride | 16 / 16 | 24 / 32 | 32 / 16 |
| Output stride | 2 | 4 | 2 |
| Confidence-map sigma | 2.5 | 2.5 | 1.5 |
| Parameters | ~1.95 M | ~1.64 M | ~1.30 M |
| Batch size / learning rate | 4 / 1e-5 (rounds 1–5), 8 / 1e-4 (rounds 6–7) | 4 / 1e-5 | 8 / 1e-4 |
| Online hard keypoint mining | off (on in round 7) | off | on |

Common training settings: Adam optimizer, ReduceLROnPlateau scheduler, early stopping (patience 15), random rotation augmentation (±180°) and scale augmentation (0.9–1.1). Validation fraction was 10% (20% in round 7).

---

## Results

Validation metrics read from `metrics.val.0.npz`:

### Centroid models (insect detection)

| Round | n | OKS mAP | mOKS | mPCK |
| 7 | 180 | **0.970** | 0.985 | 0.973 |

### Centered-instance models (body-part localization)

| Round | n | Median error (px) | 90th pct. error (px) | OKS mAP | mOKS | PCK@10 |
| 7 | 180 | **1.58** | **3.76** | **0.305** | **0.690** | **0.961** |

**Key takeaway:** centroid detection was already reliable from about round 3. Body-part accuracy stayed flat until the round 6–7 architecture change (smaller 160 px crops, finer output stride, smaller sigma and hard keypoint mining). That change roughly halved the localization error.

---

## Usage

Install SLEAP-NN (these models were trained with `sleap-nn` **0.1.0**):

```bash
pip install sleap-nn
```

Run top-down inference on a video with the round 7 models (pass both model folders):

```bash
sleap-nn track \
  --data_path path/to/video.mp4 \
  --model_paths "260818_004100.centroid.n=180" \
  --model_paths "260818_033402.centered_instance.n=180" \
  --output_path predictions.slp
```

The output `.slp` file can be opened in the [SLEAP](https://sleap.ai) GUI for review, correction or export. Command-line flags can change between SLEAP-NN versions, so check `sleap-nn track --help` if a flag is not recognized.

### Reading the metrics in Python

```python
import numpy as np

m = np.load("260818_033402.centered_instance.n=180/metrics.val.0.npz",
            allow_pickle=True)["metrics"].item()
print(m["distance_metrics"]["p50"])     # median localization error (px)
print(m["voc_metrics"]["oks_voc.mAP"])  # OKS mAP
print(m["mOKS"])                        # mean OKS
```

### Retraining

Each `training_config.yaml` can be reused to retrain a model:

```bash
sleap-nn train --config-name training_config.yaml --config-dir "260818_004100.centroid.n=180"
```
