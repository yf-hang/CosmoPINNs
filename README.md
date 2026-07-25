# CosmoPINNs

CosmoPINNs is a physics-informed neural network (PINN) framework for solving the (canonical) differential equations satisfied by cosmological wavefunction coefficients. The project focuses on the two-site (two-vertex) family of cosmological wavefunction integrals in general power-law Friedmann–Robertson-Walker (FRW) backgrounds. It also incorporates transfer learning strategy, using models trained on lower-dimensional baseline systems (tree-level graphs) to initialize and improve the training of higher-dimensional target systems (loop-level graphs). This approach allows the model to reuse previously learned structures, helping reduce training difficulty and improve the efficiency and stability of solving more complex cosmological systems.

Some of the results are collected at: <https://yf-hang.github.io/CosmoPINNs/>

## Overview

### Canonical Differential Equation
The cosmological master integrals (MIs) $\\{ I_1,I_2,\ldots \\}=\vec{I}$ satisfy following canonical differential equation (CDE)

$$
\mathrm{d}\vec{I}(\vec{z},\varepsilon) = \varepsilon [\mathrm{d} A(\vec{z})] \vec{I}(\vec{z}, \varepsilon)
$$

where $\vec{z}=\\{X_1,X_2,\ldots,Y_1,Y_2,\ldots\\}$ is the set of all independent kinematic variables 
$\varepsilon$ denotes the twistor factor, and $\mathrm{d}$ is the total differential $\mathrm{d}=\sum_i\mathrm{d} z_i\partial_{z_i}$.
Moreover, $A(\vec{z})$ represents the connection matrix which can be further decomposed as

$$
A(\vec{z}) = \sum_{i=1}^{6 \cdot 2^{\ell}-1} a_i \mathrm{log}[w_i(\vec{z})] 
$$

where $a_i$ are constant matrices and $w_i(\vec{z})$ are the symbol letters which are rational or algebraic functions of the kinematic variables. The complete set of $\{w_i\}$ is referred to as the alphabet. Here $\ell$ denotes the number of loop, such as $\ell=0$ chain (tree), $\ell=1$ 1-loop bubble, $\ell=1$ 2-loop sunset, and etc.

### Neural Networks
<table>
  <tr>
    <td align="center" width="50%">
      <img src="/figures/nn_fig.png" alt="Basic NN" width="100%">
    </td>
    <td align="center" width="50%">
      <img src="/figures/nn_transfer_fig.png" alt="Transfer Learning NN" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>Basic neural network</em>
    </td>
    <td align="center">
      <em>Transfer-learning neural network</em>
    </td>
  </tr>
</table>

The key idea is to approximate the vector of MIs by a neural network and train it directly against the canonical differential system. 
The loss combines:

- Loss of canonical differential equation (CDE) evaluated at collocation points $L_{\mathrm{CDE}}$.
- Analytic boundary and anchor data $L_{\mathrm{BC}}$.

So the loss function minimized in the training is the weighted sum of the CDE loss and the boundary loss

$$
L(\theta) = \lambda_1 L_{\mathrm{CDE}}(\theta) + \lambda_2 L_{\mathrm{BC}}(\theta)
$$

with $\lambda_1$ and $\lambda_2$ being the corresponding weights.

The code implements a three-phase hierarchy:

| Phase | Topology | Inputs ($u_k$) dim | Outputs (Re($I_j$)) dim | Training role |
| -- | --- | ---: | ---: | --- |
| P0 | chain ($\ell = 0$) | 2 | 4 | source model |
| P1 | one-loop bubble ($\ell = 1$) | 3 | 10 | transfer target |
| P2 | two-loop sunset ($\ell = 2$) | 4 | 22 | transfer target |

in dim vs. out dim: <https://yf-hang.github.io/CosmoPINNs/nn_dim.html>

For transfer learning, the P0 hidden layers are copied into the target model, frozen, 
and paired with new input and output layers matching the loop-level topologies in P1 and P2.

<table>
  <tr>
    <td align="center" width="50%">
      <img src="/figures/tl_chain_bubble.png" width="100%">
    </td>
    <td align="center" width="50%">
      <img src="/figures/tl_chain_sunset.png" width="100%">
    </td>
  </tr>
  <tr>
    <td align="center">
      <em>P1: Transfer from chain to one-loop bubble</em>
    </td>
    <td align="center">
      <em>P2: Transfer from chain to two-loop sunset</em>
    </td>
  </tr>
</table>

## Setup

The default numerical setup follows the manuscript:

| $\ell$ | Collocation domain | Fixed scale | Collocation points | Boundary points | Epochs |
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


We use the Adam optimizer with initial learning rate $\eta_0^{}=10^{-3}$. The learning rate is linearly warmed up during the first $N_{\mathrm{warm}}$ epochs and is then decreased by cosine annealing to $\eta_{\min}=10^{-8}$:

$$
\eta_t = \begin{cases}
\eta_0 \dfrac{t}{N_{\mathrm{warm}}}, & 1 \leq t \leqq N_{\mathrm{warm}}
\\
\eta_{\min} + \dfrac{\eta_0 - \eta_{\min}}{2} \left[1 + \cos\left(\pi \frac{t - N_{\mathrm{warm}}}{N_{\mathrm{epoch}} - N_{\mathrm{warm}}}\right)\right], & N_{\mathrm{warm}} < t \leqq N_{\mathrm{epoch}}
\end{cases}
$$

## Repository Layout

```text
|-- main.py                         # Main configuration-driven training entry point
|-- config.json                     # Default run configuration
|-- lib/
|   |-- models.py                   # PINN and transfer-PINN modules
|   |-- loss.py                     # CDE residual and BC losses
|   |-- train.py                    # Optimizer, warmup, cosine schedule
|-- two_site_chain/                 # Phase-0 analytic targets and CDE matrices
    |-- coll_bc_1loop.py            # Collocation and boundary data
    |-- coll_mat_1loop.py           # Generation of connection matrix
    |-- mat_data_1loop.py           # Structure of connection matrix
    |-- sol_1loop.py                # Analytic solutions
|-- tl_two_site_bubble/             # Phase-1 one-loop transfer target
|-- tl_two_site_sunset/             # Phase-2 two-loop transfer target
|-- plot_tools/                     # Per-run plotting and post-training checks
    |-- plot_loss.py                
    |-- plot_error.py
    |-- post_train_check.py
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


## Agentic Analysis
On preparation ...


## Notes
The formulation implemented in this codebase originate from [arXiv:2410.17192](https://arxiv.org/abs/2410.17192), where the kinematic flow and CDEs for the relevant two-site loop-level cosmological wavefunction integrals were first analyzed and derived. We ask that papers using, discussing, or extending this formulation cite [arXiv:2410.17192](https://arxiv.org/abs/2410.17192) as the original work.
