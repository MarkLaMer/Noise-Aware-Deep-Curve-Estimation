# NA-DCE: Noise-Aware Deep Curve Estimation

Noise-aware zero-reference low-light image enhancement, implemented from scratch in PyTorch. Extends Zero-DCE (Guo et al., CVPR 2020) with lightweight noise-aware loss terms: no paired training data, no added networks, no architectural changes.

**CISC 473 Deep Learning, Capstone Project P02, Queen's University, Fall 2026**

Bella Xu · Mark Nistor

📄 [Project proposal (PDF)](docs/CISC473_Group8_Proposal.pdf)
📄 [Midterm report (PDF)](docs/CISC473_Group8_Midterm.pdf)

## The idea

Zero-DCE enhances dark images by estimating per-pixel tonal curves, trained with zero reference images. But its losses contain no noise model, so brightening dark regions amplifies sensor noise. We quantify that failure, then test three noise-aware loss terms that keep the zero-reference property:

1. **Darkness-gated smoothness**: TV penalty weighted by input darkness
2. **Self-supervised denoising regulariser**: Neighbor2Neighbor-style consistency on outputs
3. **Curve-gain regularisation**: bounding effective per-pixel gain in low-SNR regions

Evaluated on LOL, ExDark, and a self-collected corpus of nighttime street photographs, including downstream object-detection performance (frozen YOLO, mAP@0.5) before vs. after enhancement.

All models are implemented from scratch; the official Zero-DCE repository is used only to cross-check our reproduction.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
pip install -e .
```

For GPU training, replace the CPU-only PyTorch build with the CUDA one (we used CUDA 13.0):

```bash
pip uninstall -y torch torchvision
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

## Data

Datasets are not committed. Download LOL (Wei et al., BMVC 2018) from https://daooshee.github.io/BMVC2018website/ and extract it to:

    data/LOL/our485/low/    485 training images (only the low-light images are used)
    data/LOL/eval15/low/    15 test inputs
    data/LOL/eval15/high/   15 test references (used only for metrics)

## Reproduce

Train three seeds (about 10 min each on an RTX 4070 Laptop GPU):

```bash
python scripts/train.py --seed 1 --out outputs/gpu_seed1
python scripts/train.py --seed 2 --out outputs/gpu_seed2
python scripts/train.py --seed 3 --out outputs/gpu_seed3
```

Evaluate on LOL eval15 (PSNR / SSIM), saving the enhanced images:

```bash
python scripts/evaluate.py --ckpt outputs/gpu_seed1/last.pth --save-dir outputs/eval_seed1
```

Official checkpoint for cross-checking (download `Epoch99.pth` from the official Zero-DCE repository; `--official` accounts for its opposite curve sign):

```bash
python scripts/evaluate.py --ckpt path/to/Epoch99.pth --official --save-dir outputs/eval_official
```

Noise-amplification study (needs `outputs/eval_seed1` and `outputs/eval_official`):

```bash
python scripts/noise_study.py
```

Sanity checks:

```bash
python tests/check_model.py
python tests/check_curves.py
python tests/check_losses.py
```

## Midterm results (LOL eval15)

| Model | PSNR (dB) | SSIM |
|---|---|---|
| Zero-DCE, official checkpoint (our evaluator) | 14.86 | 0.562 |
| Ours, 3 seeds | 14.96 ± 0.41 | 0.516 ± 0.003 |

In the darkest input regions, our model's output noise reaches 8.8× the reference noise level, versus 3.3× for the official checkpoint (`scripts/noise_study.py`).

## Repository structure

    src/nadce/     model, curves, losses, dataset
    scripts/       train.py, evaluate.py, noise_study.py
    tests/         sanity checks for model, curves and losses
    docs/          proposal and reports

## References

Built on: Zero-DCE (arXiv:2001.06826) · Zero-DCE++ (arXiv:2103.00860) · Neighbor2Neighbor (arXiv:2101.02824). Full reference list in the proposal.
