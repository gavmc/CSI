# WiFi CSI Human Pose Estimation

Estimating human body pose from WiFi Channel State Information (CSI) — no camera required at inference time. A multi-receiver ESP32 sensor mesh streams CSI to a transformer model trained with cross-modal supervision from a camera-based pose teacher.

> **Status:** Paused. Core pipeline (firmware → collection → training) is working end-to-end; current numbers are a smoke test, not a benchmark (see [Results & Honest Caveats](#results--honest-caveats)).

---

## Overview

Standard WiFi hardware exposes CSI — per-subcarrier amplitude/phase measurements derived from the OFDM preamble — which varies with multipath propagation in a room. A person moving through that space perturbs the multipath in ways that correlate with their pose. This project asks whether that signal is rich enough to recover 2D keypoints, using only radio.

Since there's no way to hand-label CSI with joint coordinates, the system uses **cross-modal supervision**: a synchronized webcam feed is run through a YOLO-pose model to generate keypoint "pseudo-labels," and the CSI transformer is trained to reproduce them. At inference time, the camera is dropped entirely — the trained model needs only CSI.

## Architecture

```
┌─────────────┐   50 Hz CSI (UDP)   ┌────────────────┐
│  4x ESP32   │ ──────────────────► │   Collector    │
│  receivers  │                     │ (clock sync,   │
└─────────────┘                     │  csi.npz)      │
       ▲                            └───────┬────────┘
       │ probe traffic                      │
┌─────────────┐                             ▼
│ Netgear X6  │                    ┌─────────────────┐      ┌──────────────────┐
│  (AP / Tx)  │                    │ Dataset builder │◄─────│ YOLO-pose teacher│
└─────────────┘                    │ resample/window │      │ (13 keypoints/fr)│
                                   │   /normalize    │      └──────────────────┘
                                   └───────┬─────────┘
                                           ▼
                                 ┌────────────────────┐
                                 │  CSIPoseTransformer│
                                 │  X [N,4,50,52] →   │
                                 │  Y [N,13,2]        │
                                 └────────────────────┘
```

## Hardware

- **4x ESP32-WROOM-32** as CSI receivers (custom firmware, ESP-IDF v6)
- **Netgear Nighthawk X6** repurposed as a dedicated 2.4 GHz sensing AP (fixed channel, HT20, no WAN)
- Webcam for ground-truth video (teacher signal only — not needed post-training)

## Firmware

FreeRTOS-based, per-ESP32:
- MAC-filtered promiscuous-mode CSI capture (`esp_wifi_set_csi_rx_cb`) with a 20-byte wire header
- Producer/consumer queue pattern — callback copies into a queue, a separate task drains and sends over UDP
- A dedicated pinger task generates steady probe traffic against the AP to keep the CSI callback rate up (~50 Hz achieved, ~92% match-to-receive ratio)
- Each captured frame is 128 bytes → 64 (imag, real) int8 pairs → the LLTF channel estimate in FFT bin order; DC and guard-band subcarriers are dropped, leaving 52 valid data subcarriers per frame

## Data Pipeline

1. **Collector** — desktop Python process receives packed CSI records over UDP from all 4 receivers, plus a synced webcam stream, and estimates clock offset via probe RTT.
2. **Labeling** — YOLO-pose runs over the video to produce 13 keypoints/frame with per-joint confidence. Low-confidence joints are masked out of the training loss rather than trained on noisy labels.
3. **Dataset builder** — resamples each receiver's async CSI stream onto a common 50 Hz grid (per-subcarrier linear interpolation), windows into `[R=4, T=50, S=52]` tensors (1s windows), and normalizes.

## Model

`CSIPoseTransformer` — ~1M params, sized for data-starved training on a single GPU:

- Per-frame subcarrier vector → linear embedding (`d_model=96`)
- Learned positional + per-receiver embeddings
- Shared-weight per-receiver temporal transformer encoder
- Cross-receiver fusion layers
- DETR-style decoder: 13 learned joint queries attend over the fused CSI representation
- Output: `[K=13, 3]` per window → (x, y, confidence) per joint

**Loss:** Smooth-L1 on visible-joint coordinates + BCE on the visibility/confidence head, masked by YOLO's per-joint confidence.

**Pretraining:** a masked-CSI-reconstruction objective (mask random time/subcarrier patches, reconstruct with the same encoder) is implemented for self-supervised pretraining on unlabeled CSI — useful because unlabeled CSI is free to collect, unlike labeled (camera-synced) sessions.

## Results & Honest Caveats

- Achieved ~50 Hz CSI capture, ~92% match-to-receive ratio across the 4-receiver mesh.
- First full training run: PCK@0.1 reached ~0.63, val loss bottomed at epoch ~32 (0.0268) with a textbook overfitting curve after.
- **That PCK number is inflated.** The run used a random train/val split with `STRIDE=10`, `WINDOW=50` — adjacent windows share 80% of their frames, so most val windows have a near-duplicate in train (same second, same room, same outfit). The fix is a **temporal split with a gap** (index-ordered, ~5-window gap = zero frame overlap) rather than a random split — implemented but not yet re-run at scale.
- Session count is currently limited to one short collection (~226s, ~11k CSI frames/sensor, ~6k camera frames), i.e. one room, one outfit, one furniture layout — not enough to know if this generalizes.

**Bottom line:** the pipeline (firmware → sync → labeling → training) is proven to work and there's real pose-correlated signal in the CSI, but there is no trustworthy accuracy number yet. Next real step is re-running eval with the temporal split, then collecting multiple sessions across conditions.

## Repo Structure

```
firmware/            ESP-IDF C++ project (per-ESP32 CSI capture + UDP streaming)
collector/            Desktop UDP collector, webcam capture, clock sync
labels.py             YOLO-pose labeling pass over collected video
build_dataset.py      Resample → window → normalize → csi.npz / labels.npz
csi_pose_transformer.py   Model definition + masked-CSI pretraining objective
train.py              Training loop (temporal split, Smooth-L1 + BCE loss)
```

## Roadmap

- [ ] Re-run evaluation with temporal (gapped) split for an honest PCK number
- [ ] Multi-session collection across rooms/outfits/furniture layouts
- [ ] Masked-CSI self-supervised pretraining on longer unlabeled sessions
- [ ] Investigate phase-derived features (currently amplitude-only; ESP32 phase is noisy without CFO compensation)
- [ ] Coarse localization as an intermediate milestone before full pose

## Key Learnings

- ESP32 single-antenna CSI has no reliable phase data without extra calibration — amplitude-only features were used throughout.
- Cross-modal (camera-teacher → radio-student) supervision is the practical way to label CSI at all, but it caps model accuracy at teacher accuracy — YOLO-pose jitter/misses propagate directly into training labels.
- Adjacent-window leakage from overlapping strides is an easy, easy-to-miss way to inflate offline metrics on time-series data; always check split strategy against window stride before trusting a number.