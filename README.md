# CosmoPINNs

CosmoPINNs is a physics-informed neural network (PINN) framework for solving the canonical differential equations satisfied by cosmological wavefunction coefficients. The project focuses on the two-site family of cosmological wavefunction integrals in general power-law Friedmann–Robertson-Walker (FRW) backgrounds. It also incorporates transfer learning strategy, using models trained on lower-dimensional baseline systems (tree-level graphs) to initialize and improve the training of higher-dimensional target systems (loop-level graphs). This approach allows the model to reuse previously learned structures, helping reduce training difficulty and improve the efficiency and stability of solving more complex cosmological systems.

Some of the results are collected at: <https://yf-hang.github.io/CosmoPINNs/>.

## Overview

The core idea is to approximate the vector of master integrals (MIs) by a neural network and train it directly against the canonical differential system. The loss combines:

- loss of canonical differential equation (CDE) evaluated at collocation points;
- analytic boundary and anchor data that fix integration constants and
  normalization;
- post-training diagnostics against analytic solutions.

The code implements a three-phase hierarchy:

| Phase | Topology | Inputs | Real-sector MI dimension | Training role |
| --- | --- | --- | ---: | --- |
| Phase 1 | two-site chain, $\ell = 0$ | `(u1, u2)` | 4 | source model |
| Phase 2 | one-loop bubble, $\ell = 1$ | `(u1, u2, u3)` | 10 | transfer target |
| Phase 3 | two-loop sunset, $\ell = 2$ | `(u1, u2, u3, u4)` | 22 | transfer target |

For transfer learning, the Phase-1 hidden representation is copied into the
target model, frozen, and paired with new input and output layers matching the
loop-level topology.

## Scientific Setup

The default numerical setup follows the manuscript:

| Loop order $\ell$ | Collocation domain | Fixed scale | Collocation points | Boundary points | Epochs |
| --- | --- | ---: | ---: | ---: | ---: |
| 0 | `[20,30] x [20,30]` | `c0 = 15` | `5e4` | `5e3` | `6000` |
| 1 | `[30,40] x [30,40] x [15,20]` | `c1 = 5` | `1e5` | `1e4` | `8000` |
| 2 | `[50,60] x [50,60] x [20,25] x [10,15]` | `c2 = 5` | `1.5e5` | `1.7e4` | `10000` |

The benchmark values of $\varepsilon$ are:

| $\ell$ | Background scan |
| --- | --- |
| 0 | $\varepsilon = \{0, -1, -2, -3, -4, +5\}$ |
| 1 | $\varepsilon = \{0, -1, -2, -3\}$ |
| 2 | $\varepsilon = \{0, -1, -4, +5\}$ |

Note that $\varepsilon$ is the twist factor defined in the power-law cosmology. Here $\varepsilon = 0$ and $\varepsilon = -1$ are corresponding to the de Sitter (dS) and flat-space backgrounds. The remaining values correspond to the radiation-dominated (RD) and matter-dominated (MD) backgrounds used for the one- and two-loop systems.

## Repository Layout

```text
.
|-- main.py                         # Main configuration-driven training entry point
|-- config.json                     # Default production/run configuration
|-- lib/
|   |-- models.py                   # PINN and transfer-PINN modules
|   |-- loss.py                     # CDE residual and boundary losses
|   `-- train.py                    # Optimizer, warmup, cosine schedule
|-- two_site_chain/                 # Phase-1 analytic targets and CDE matrices
|-- tl_two_site_bubble/             # Phase-2 one-loop transfer target
|-- tl_two_site_sunset/             # Phase-3 two-loop transfer target
|-- plot_tools/                     # Per-run plotting and post-training checks
|-- results/                        # Generated checkpoints, logs, plots, caches
```

## Running Training

Edit `config.json`, then launch:

```bash
python main.py
```

Important configuration fields:

| Key | Meaning |
| --- | --- |
| `device` | `auto`, `cpu`, or `cuda` |
| `eps_global` | fixed $\varepsilon$ value for the run |
| `enable_phase2`, `enable_phase3` | select the transfer target; only one can run in a single execution |
| `run_phase2_only`, `run_phase3_only` | skip Phase-1 training and load an existing Phase-1 checkpoint |
| `phase*_output_part` | `Re`, `Im`, or `Both`; manuscript runs use real-sector training |
| `lambda1`, `lambda2` | CDE and boundary loss weights |
| `phase*_model_load_path` | explicit checkpoint path for transfer or evaluation |

If `use_local_config` is set to `true` in `config.json`, `main.py` loads
`config_local_test.json` instead.

Generated artifacts are written under `results/` by default, or
`results_local_test/` when local-test mode is enabled. Each run stores
configuration snapshots, model checkpoints, loss histories, training logs,
diagnostic plots, and optional evaluation bundles.

## Transfer-Only Runs

To train a loop-level target from an existing Phase-1 checkpoint, set:

```json
{
  "run_phase2_only": true,
  "enable_phase2": true,
  "enable_phase3": false,
  "phase1_model_load_path": "path/to/P1_model_eps_...pt"
}
```

or use the analogous `run_phase3_only` / `enable_phase3` settings for Phase 3.
If `reuse_saved_models` is true and a matching Phase-1 checkpoint exists in the
standard output location, `main.py` can infer the checkpoint path automatically.


## Notes

The formulation implemented in this codebase originate from [arXiv:2410.17192](https://arxiv.org/abs/2410.17192), where the kinematic flow and CDEs for the relevant two-site loop-level cosmological wavefunction integrals were first analyzed and derived. We ask that papers using, discussing, or extending this formulation cite [arXiv:2410.17192](https://arxiv.org/abs/2410.17192) as the original work.
