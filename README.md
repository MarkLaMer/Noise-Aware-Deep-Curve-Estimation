# NA-DCE — Noise-Aware Deep Curve Estimation

Noise-aware zero-reference low-light image enhancement (Zero-DCE extension), from scratch in PyTorch.

CISC 473 Deep Learning capstone, Queen's University, Fall 2026. Group: Mark Nistor and Bella Xu.

[Project proposal (PDF)](docs/CISC_473_Proposal.pdf)

## The Idea

Zero-DCE enhances dark images by estimating per-pixel tonal curves, trained with zero reference images, but its losses contain no noise model, so brightening dark regions amplifies sensor noise. We quantify that failure, then test three
noise-aware loss terms that keep the zero-reference property:

1. **Darkness-gated smoothness**: TV penalty weighted by input darkness
2. **Self-supervised denoising regulariser**: Neighbor2Neighbor-style consistency on outputs
3. **Curve-gain regularisation**: bounding effective per-pixel gain in low-SNR regions

Evaluated on LOL, ExDark, and a self-collected corpus of nighttime street
photographs, including downstream object-detection performance (frozen YOLO,
mAP@0.5) before vs. after enhancement.

All models are implemented from scratch; the official Zero-DCE repository is
used only to cross-check our reproduction.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows  (source .venv/bin/activate on Linux/Mac)
pip install -r requirements.txt
```

## Data

Datasets are not committed to the repo. See [data/README.md](data/README.md)
for download instructions (LOL, ExDark).

## Reproduce

Coming with the midterm release: `scripts/reproduce.sh` single-command training + evaluation with fixed seeds (mean ± std over 3 seeds).

## Repository Structure

    src/nadce/     model, losses, training
    scripts/       train / evaluate / reproduce entry points
    configs/       experiment configs incl. seeds
    docs/          proposal and reports
    notebooks/     exploratory analysis

## References

Built on: Zero-DCE (arXiv:2001.06826) · Zero-DCE++ (arXiv:2103.00860) ·
Neighbor2Neighbor (arXiv:2101.02824). Full reference list in the proposal.
