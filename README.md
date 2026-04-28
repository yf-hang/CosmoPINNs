# CosmoPINNs

CosmoPINNs is a research codebase for solving canonical differential equations (CDEs) of cosmological wavefunction integrals with physics-informed neural networks (PINNs).

The project is organized as a three-phase hierarchy:

- `Phase 0`: two-site chain
- `Phase 1`: one-loop bubble, trained by transfer learning from matched Phase-0 checkpoints
- `Phase 2`: two-loop sunset, also initialized from matched Phase-0 checkpoints
 
## What This Repository Contains

- `main.py`: main training entry point for fixed-`epsilon` runs
- `lib/`: neural-network models, losses, and training utilities
- `two_site_chain/`: Phase-0 problem definition
- `tl_two_site_bubble/`: Phase-1 transfer-learning setup
- `tl_two_site_sunset/`: Phase-2 transfer-learning setup
- `plot_tools/`: plotting and post-training diagnostic utilities

## Main Features

- PINN training for fixed-`epsilon` cosmological CDE systems
- Transfer-learning workflow for higher-loop targets
- Built-in diagnostics for:
  - training losses
  - relative `L1/L2` errors
  - log-ratio amplitude mismatch
  - cosine-similarity shape alignment
- Manuscript-ready summaries and result plots

## Quick Start

1. Adjust the run settings in `config.json`.
2. Launch training from the project root:

```bash
python main.py
```

## Notes
- The formulation implemented in this codebase originate from [arXiv:2410.17192](https://arxiv.org/abs/2410.17192), where the kinematic flow and CDEs for the relevant two-site loop-level cosmological wavefunction integrals were first analyzed and derived. We ask that papers using, discussing, or extending this formulation cite [arXiv:2410.17192](https://arxiv.org/abs/2410.17192) as the original work.
- The current production workflow is configuration-driven rather than packaged as a pip module.
- This repository is intended as research code, so some paths and runtime options are tuned for local or batch environments.
