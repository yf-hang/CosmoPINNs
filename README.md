# CosmoPINNs

CosmoPINNs is a physics-informed neural network (PINN) framework for solving the canonical differential systems that determine the families of twisted cosmological integrals. The current implementation focuses on integral families associated with two-site (two-vertex) graphs in a conformally-coupled scalar theory with polynomial self-interactions in general power-law Friedmann–Robertson–Walker (FRW) backgrounds. Following the twisted-integral formulation, cosmological wavefunction coefficients are represented in terms of twisted integrals of their flat-space counterparts, and the associated master integrals (MIs) form a finite-dimensional basis satisfying a coupled system of first-order differential equations in kinematic space.

CosmoPINNs further incorporates a transfer-learning strategy for extending solutions from simpler to more complex integral families. Models trained on lower-dimensional baseline systems, such as tree-level graphs, are used to initialize higher-dimensional target systems associated with loop-level graphs. By transferring structures learned from simpler systems, this approach reduces the difficulty of PINN optimization and improves convergence, computational efficiency, and training stability for more complex cosmological integral systems.

Some of the results are collected at: <https://yf-hang.github.io/CosmoPINNs/>

A manuscript presenting this work has been completed and will be made available on arXiv soon.

## Overview

### Canonical Differential Equation (CDE)
The cosmological master integrals (MIs) $\\{ I_1,I_2,\ldots \\}=𝑰$ satisfy following canonical differential equation (CDE):

$$
\mathrm{d}𝑰(𝒛,\varepsilon) = \varepsilon [\mathrm{d} A(𝒛)] 𝑰(𝒛, \varepsilon)
$$

where $𝒛=\\{X_1,X_2,\ldots,Y_1,Y_2,\ldots\\}$ is the set of all independent kinematic variables 
$\varepsilon$ denotes the twistor factor, and $\mathrm{d}$ is the total differential $\mathrm{d}=\sum_i\mathrm{d} z_i\partial_{z_i}$.
Moreover, $A(𝒛)$ represents the connection matrix which can be further decomposed as

$$
A(𝒛) = \sum_{i=1}^{6 \cdot 2^{\ell}-1} a_i \ \mathrm{log}\big[w_i(𝒛)\big] 
$$

where $a_i$ are constant matrices and $w_i(𝒛)$ are the symbol letters which are rational or algebraic functions of the kinematic variables. The complete set of $\\{w_i\\}$ is referred to as the alphabet. Here $\ell$ denotes the number of loop, such as $\ell=0$: chain (tree), $\ell=1$: 1-loop bubble, $\ell=2$: 2-loop sunset, and etc. For example, when $\ell=0$, connection matrix $A(𝒛)$ takes the following form:

$$
A =
\begin{pmatrix}
l_1+l_2 & l_3-l_1 & l_4-l_2 & 0 \\
0 & l_2+l_3 & 0 & l_5-l_2 \\
0 & 0 & l_1+l_4 & l_5-l_1 \\
0 & 0 & 0 & 2l_5
\end{pmatrix}
$$

where $l_i \equiv \mathrm{log}(w_i)$ and the letters are

$$
w_1 = X_1+Y_1,\quad\ w_2 = X_2+Y_1, \quad\ w_3 = X_1-Y_1,\quad\ w_4 = X_2-Y_1, \quad\ w_5 = X_1+X_2
$$

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

The key idea is to approximate the integration family of MIs by a neural network and train it 
directly against the canonical differential system. 
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

$d_{\mathrm{in}}$ vs. $d_{\mathrm{out}}$: <https://yf-hang.github.io/CosmoPINNs/nn_dim.html>

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

Collocation points:

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

After training, the network prediction is compared with the analytic solutions at the evaluation points.
For each evaluation point $\vec{u}_i$, we define

$$ 
\mathcal{L}_1 = \frac{\lVert\hat{\vec{I}}(\vec{u}_i,\varepsilon;\theta)-\vec{I}(\vec{u}_i,\varepsilon)\rVert_1}{\max(\lVert \vec{I}(\vec{u}_i,\varepsilon)\rVert_1,\delta)}, \qquad 
\mathcal{L}_2 = \frac{\lVert \hat{\vec{I}}(\vec{u}_i,\varepsilon;\theta)-\vec{I}(\vec{u}_i,\varepsilon)\rVert_2}{\max(\lVert \vec{I}(\vec{u}_i,\varepsilon) \rVert_2,\delta)}, \qquad 
\mathcal{C} = \frac{|\hat{\vec{I}}(\vec{u}_i,\varepsilon;\theta)\cdot \vec{I}(\vec{u}_i,\varepsilon)|}{\max(\lVert\hat{\vec{I}}(\vec{u}_i,\varepsilon;\theta)\rVert_2\lVert\vec{I}(\vec{u}_i,\varepsilon)\rVert_2,\delta)} 
$$

Here all norms are evaluated at a single point $\vec{u}_i$ and are taken over the trained output MIs. The regulator $\delta=10^{-30}$ is used only to avoid division by zero when the analytic vector has a vanishing or numerically tiny norm.

## Repository Layout

```text
|-- main.py                         # Main configuration-driven training entry point
|-- config.json                     # Default run configuration
|-- lib/
|   |-- pinn_models.py              # PINN and transfer-PINN modules
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
|-- results/                        # Generated checkpoints, logs, plots
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
| `enable_phase1`, `enable_phase2` | select the transfer target; only one can run in a single execution |
| `run_phase1_only`, `run_phase2_only` | skip Phase-1 training and load an existing Phase-1 checkpoint |
| `phase*_output_part` | `Re` or `Im`; real and imaginary sectors are trained separately |
| `lambda1`, `lambda2` | CDE and boundary loss weights |
| `phase*_model_load_path` | explicit checkpoint path for transfer or evaluation |

If `use_local_config` is set to `true` in `config.json`, `main.py` loads
`config_local_test.json` instead.

Generated artifacts are written under `results/` by default, or
`results_local_test/` when local-test mode is enabled. Each run stores
configuration snapshots, model checkpoints, loss histories, training logs,
diagnostic plots, and optional evaluation bundles.

## Transfer-Only Runs

To train a loop-level target from an existing Phase-0 checkpoint, set:

```json
{
  "run_phase1_only": true,
  "enable_phase1": true,
  "enable_phase2": false,
  "phase0_model_load_path": "path/to/P0_model_eps_...pt"
}
```

or use the analogous `run_phase2_only` / `enable_phase2` settings for Phase 2.
If `reuse_saved_models` is true and a matching Phase-0 checkpoint exists in the
standard output location, `main.py` can infer the checkpoint path automatically.

## Phase-0 Hyperparameter Tuning (Optuna)

The Phase-0 tuner is implemented in `agent/phase0_optuna.py`. Install the Agent
requirements and launch the study with:

```bash
python -m pip install -r agent/requirements.txt
python agent/phase0_optuna.py
```

### Search setup

Optuna searches only $\lambda_1$. The Phase-0 learning rate
`learning_rate_p0` and $\lambda_2$ remain fixed by the root `config.json`.
The search range, trial count, validation sampling, and pruning settings are
configured in `agent/phase0_optuna_config.json`.

### Validation objective

For each Phase-0 output component, the fixed validation set is used to compute

$$
E_j = \frac{\lVert \hat I_j-I_j\rVert_2}{\lVert I_j\rVert_2}.
$$

Optuna minimizes the equal-weight mean

$$
S = \frac{1}{n_{\mathrm{out}}}\sum_{j=1}^{n_{\mathrm{out}}}E_j,
\qquad n_{\mathrm{out}}=4 \text{ for Phase 0}.
$$

The completed trial with the smallest $S$ is the best trial. The individual
$E_j$ values are also recorded so that the contribution of each output can be
inspected separately.

### Pruning

The tuner uses Optuna's `MedianPruner`. At each validation step it reports $S$
with `trial.report(S, step=epoch)`. With the default configuration, median
pruning starts only after 8 completed trials exist and from epoch 600 onward,
with validation/pruning checks every 100 epochs. For this minimization study, a
trial is pruned when its best reported $S$ so far is larger than the median
score of completed trials at the same step. A non-finite training loss (`NaN`
or `Inf`) is pruned immediately.

The relevant controls are `pruner_startup_trials`, `pruner_warmup_epochs`, and
`eval_every` in `agent/phase0_optuna_config.json`.

### Optuna outputs

The study is stored in SQLite under the epsilon-specific Optuna output directory.
The main result files are:

```text
best_params.json             # best trial, lambda1, objective, and metrics
trials.csv                   # all trials and per-output validation errors
best_phase0_checkpoint.pt    # checkpoint for the best score seen so far
study.sqlite3                # persistent Optuna study
```

## CosmoAgent

`agent/cosmo_agent.py` is a persistent, tool-using conversational interface for
inspecting the Optuna study and operating CosmoPINNs training runs. It uses a
local Ollama model for language interaction, while numerical statements about
trials, metrics, and parameters come from the Optuna SQLite database.

```bash
python agent/cosmo_agent.py
```

CosmoAgent resolves the Phase-0 study automatically from the current root
`config.json` and `agent/phase0_optuna_config.json`, including the epsilon tag,
study name, and SQLite output path.

### Study inspection

CosmoAgent can:

- summarize trial states and the current best trial;
- rank or compare completed trials;
- retrieve `lambda1`, objective values, per-output relative $L_2$ metrics,
  timing, and intermediate values;
- estimate Optuna parameter importance as association rather than causality;
- preserve context for follow-up questions.

For a direct database check that bypasses the language model, use:

```text
You> /study
```

The language model does not calculate the objective, prune trials, or select the
best trial; those decisions are made and stored by Optuna.

### Ollama model selection

The default model is `phi4-mini:latest`. CosmoAgent requires the selected model
to advertise Ollama `tools` capability; `thinking` capability is optional.

```bash
# Inspect installed models and compatibility
python agent/cosmo_agent.py --list-models

# Select a model at startup
python agent/cosmo_agent.py --model qwen3:1.7b

# Set a different default
export COSMO_AGENT_MODEL=gpt-oss:20b
python agent/cosmo_agent.py
```

Models can also be inspected or switched during a session while preserving the
conversation:

```text
You> /models
You> /model qwen3:1.7b
You> switch to gpt-oss:20b
```

Models installed on a local workstation are not automatically available in a
Colab runtime. Run `ollama list` in the environment where CosmoAgent itself is
running.

### Training from the best Optuna trial

When explicitly asked to start training, CosmoAgent maps the best Phase-0 trial
back to the production configuration as follows:

```text
best trial lambda1  -> lambda1
learning_rate_p0    -> unchanged from config.json
lambda2             -> unchanged from config.json
```

The final run starts from a newly initialized model and retains the production
collocation count, epoch count, network architecture, and physical settings
from `config.json`. Before launch, CosmoAgent verifies compatibility between the
study and the current Phase-0 setup, including the background, output sector,
domain, network structure, fixed learning rate, fixed `lambda2`, and objective
definition.

Example requests:

```text
You> preview the Phase-0 training configuration using the best trial
You> start Phase-0 training using the best Optuna lambda1
You> start Phase-0 and Phase-1 training using the best Phase-0 lambda1
You> what is the status of the latest training job?
```

Phase 0 is the default pipeline. `phase0_phase1` or `phase0_phase2` is enabled
only when explicitly requested. Agent-launched jobs are isolated under:

```text
agent/training_runs/<job_id>/
|-- config.json       # effective configuration snapshot
|-- job.json          # status, PID, trial, and parameter mapping
|-- training.log      # stdout and stderr
|-- results/          # checkpoints, histories, evaluations, and plots
```

Only one Agent-launched job is allowed to run at a time to avoid accidental GPU
contention. CosmoAgent never modifies existing Optuna trials or the study
database.

## Notes
The formulation implemented in this codebase originate from [arXiv:2410.17192](https://arxiv.org/abs/2410.17192), where the kinematic flow and CDEs for the relevant two-site loop-level cosmological wavefunction integrals were first analyzed and derived. We ask that papers using, discussing, or extending this formulation cite [arXiv:2410.17192](https://arxiv.org/abs/2410.17192) as the original work.