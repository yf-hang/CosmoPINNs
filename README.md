# CosmoPINNs

This repository contains a PINN framework designed to solve the **canonical differential equations (CDEs)** arising in the computation of **cosmological wavefunction coefficients** under a general power-law FRW universe. We begin with the two-site chain case and apply the transfer learning to study the two-site one-loop and two-loop cases.

# FinalResults Script Usage

This README documents how to use the main plotting/post-processing scripts in `FinalResults/`:

- `merge_loss.py`
- `merge_L12.py`
- `merge_log.py`
- `merge_cos.py`

Run the commands from the project root:

```bash
python FinalResults/merge_loss.py ...
python FinalResults/merge_L12.py ...
```

## Directory Layout

By default, both scripts expect the following structure under `FinalResults/`:

- `Phase1/`
- `Phase2/`
- `Phase3/`

Each phase directory typically contains:

- model files in `0_models/` or `0_model/`
- output figures in script-specific subdirectories

## `merge_loss.py`

### Purpose

`merge_loss.py` reads loss history files and generates:

- per-epsilon loss plots for `total`, `cde`, and `bc`
- per-epsilon `total` loss plots
- merged `total` loss comparison plots across selected epsilon tags

### Default Input/Output

Default root:

- `FinalResults/`

Within each phase:

- input history directory: `0_models/` or `0_model/`
- output directory: `1_losses/` or `1_loss/`

### Main Arguments

```bash
python FinalResults/merge_loss.py [options]
```

- `--root PATH`
  Root directory containing `Phase1/Phase2/Phase3`.
  Default: `FinalResults/`
- `--phases 1 2 3`
  Phase list to process.
  Allowed values: `1 2 3`
  Default: `1 2 3`
- `--eps-tags 0 m1 m2 m3 m4 p5`
  Subset of epsilon tags to process.
- `--out-name-template "P{phase}_total_loss_eps_all.png"`
  Output filename template for merged total-loss plots.
  Supports the `{phase}` placeholder.
- `--skip-single`
  Skip per-epsilon plots.
- `--skip-merge`
  Skip merged total-loss plots.

### Examples

Process everything:

```bash
python FinalResults/merge_loss.py
```

Process only Phase 1:

```bash
python FinalResults/merge_loss.py --phases 1
```

Process Phase 2 and Phase 3:

```bash
python FinalResults/merge_loss.py --phases 2 3
```

Process only selected epsilon tags:

```bash
python FinalResults/merge_loss.py --phases 1 --eps-tags 0 m1 m4 p5
```

Generate merged plots only:

```bash
python FinalResults/merge_loss.py --phases 1 2 3 --skip-single
```

Generate single-epsilon plots only:

```bash
python FinalResults/merge_loss.py --phases 1 --skip-merge
```

Use a custom merged output filename:

```bash
python FinalResults/merge_loss.py --phases 2 --out-name-template "P{phase}_merged_total_loss.png"
```

### Output Files

Single-epsilon examples:

- `P1_loss_all_eps_0.png`
- `P1_loss_total_eps_0.png`

Merged examples:

- `P1_total_loss_eps_0_m1_m2_m3.png`
- `P1_total_loss_eps_0_m1_m4_p5.png`

### Notes

- Phases are numeric: `1`, `2`, `3`. Do not pass `Phase1`, `Phase2`, or `Phase3`.
- Supported epsilon tags are: `0`, `m1`, `m2`, `m3`, `m4`, `p5`.
- Merged plots are only generated for the built-in comparison groups:
  - `0 m1 m2 m3`
  - `0 m1 m4 p5`
- Missing history files are skipped without stopping the whole run.

## `merge_L12.py`

### Purpose

`merge_L12.py` computes and plots relative error histograms for each phase and epsilon tag, including:

- single-epsilon relative L2 histograms
- single-epsilon relative L1 histograms
- merged relative L2/L1 histograms across selected epsilon tags
- tail-quantile plots

It also caches per-epsilon `rel_l1` and `rel_l2` arrays in `.npz` files and reuses them on later runs when valid.

### Default Input/Output

Defaults:

- results root: `FinalResults/`
- config file: `config.json`

Within each phase:

- model directory: `0_models/` or `0_model/`
- output directory: `2_L1L2_error/`
- cache directory: `2_L1L2_error/cache/`

### Main Arguments

```bash
python FinalResults/merge_L12.py [options]
```

- `--results-root PATH`
  Root directory containing `Phase1/Phase2/Phase3`.
  Default: `FinalResults/`
- `--config PATH`
  Project config path.
  Default: `config.json`
- `--phase 2`
  Process a single phase.
- `-p 1 3` or `--phases 1 3`
  Process multiple phases.
- `--eps-tags 0 m1 m2 m3 m4 p5`
  Subset of epsilon tags to process.
- `--device auto`
  Device selection: `auto`, `cpu`, or `cuda`.
- `--denominator true`
  Relative error denominator.
  Allowed values: `true`, `pred`
- `--num-workers 8`
  Override the post-processing worker count.
  Use `0` or `all` to use as many CPU cores as possible.
- `--chunk-size 2000`
  Override the true-target chunk size.
- `--parallel-min-points 5000`
  Override the minimum point count for parallel target computation.
- `--single-hist-color tab:blue`
  Color for single-epsilon histograms.
- `--style-only`
  Redraw plots from cache only. Skip model inference and target recomputation.
- `--output-subdir 2_L1L2_error`
  Output subdirectory under each phase.
- `--denominator-floor 1e-30`
  Lower bound for the relative-error denominator.
- `--rel-floor 1e-30`
  Lower bound applied to plotted relative-error values.

Additional note:

- `--reuse-rel-l2` still exists for compatibility, but cache reuse is already the default behavior.
- Do not use `--phase` and `--phases` together.

### Examples

Process everything:

```bash
python FinalResults/merge_L12.py
```

Process only Phase 2:

```bash
python FinalResults/merge_L12.py --phase 2
```

Process multiple phases:

```bash
python FinalResults/merge_L12.py -p 1 3
```

Process only selected epsilon tags:

```bash
python FinalResults/merge_L12.py --phase 1 --eps-tags 0 m1 m4 p5
```

Use `pred` as the relative-error denominator:

```bash
python FinalResults/merge_L12.py --phase 2 --denominator pred
```

Redraw from cache only:

```bash
python FinalResults/merge_L12.py --phase 3 --eps-tags 0 m1 --style-only
```

Write outputs to a different subdirectory:

```bash
python FinalResults/merge_L12.py --phase 1 --output-subdir 2_L1L2_error_pred
```

Override CPU computation settings:

```bash
python FinalResults/merge_L12.py --phase 2 --num-workers all --chunk-size 4000 --parallel-min-points 10000
```

### Output Files

Single-epsilon examples:

- `P1_RelL2_eps_0.png`
- `P1_RelL1_eps_0.png`

Merged examples:

- `P1_RelL2_eps_0_m1_m2_m3.png`
- `P1_RelL1_eps_0_m1_m4_p5.png`

Tail-quantile examples:

- `P1_RelL2_tail_quantiles_eps_0_m1_m2_m3.png`
- `P1_RelL1_tail_quantiles_eps_0_m1_m4_p5.png`

Cache examples:

- `P1_RelL2_data_eps_0.npz`
- `P1_RelL1_data_eps_0.npz`
- `P1_RelL2_tail_quantiles_eps_0_m1_m2_m3.npz`
- `P1_RelL1_tail_quantiles_eps_0_m1_m4_p5.npz`

### Cache Behavior

Cache reuse is enabled by default.

For each `phase + eps` pair, the script first tries to load:

- `Phase*/<output_subdir>/cache/P{phase}_RelL2_data_eps_{eps}.npz`
- `Phase*/<output_subdir>/cache/P{phase}_RelL1_data_eps_{eps}.npz`

If those files are missing, it also checks legacy cache locations:

- `Phase*/P{phase}_RelL2_data_eps_{eps}.npz`
- `Phase*/P{phase}_RelL1_data_eps_{eps}.npz`

If a valid cache file is found, the script reuses it and skips recomputation for that metric.

If only one cache is missing:

- it recomputes only the missing metric
- it keeps reusing the other one from cache

If `--style-only` is provided:

- the script only redraws from cache
- it does not run model inference or recompute targets

Common reasons why an old cache will not be reused:

- `--denominator` changed from `true` to `pred`, or the reverse
- `--denominator-floor` changed
- `--rel-floor` changed
- the internal cache version changed
- the `.npz` file is incomplete, empty, or corrupted

### Notes

- Phases are numeric: `1`, `2`, `3`.
- Supported epsilon tags are: `0`, `m1`, `m2`, `m3`, `m4`, `p5`.
- Merged histogram groups follow the same built-in epsilon group definitions as `merge_loss.py`.
- Missing model files, eval bundles, or caches are skipped and reported.

## `merge_log.py`

### Purpose

`merge_log.py` computes and plots histograms of the mean log-ratio

`mean_k log10(|pred_Ik| / |true_Ik|)`

for each phase and epsilon tag. It generates:

- single-epsilon log-ratio histograms
- merged log-ratio histograms across selected epsilon tags

It also caches the per-sample log-ratio series in `.npz` files and reuses them on later runs when available.

### Default Input/Output

Defaults:

- results root: `FinalResults/`
- config file: `config.json`

Within each phase:

- model directory: `0_models/` or `0_model/`
- output directory: `3_log_ratio/`
- cache directory: `3_log_ratio/cache/`

### Main Arguments

```bash
python FinalResults/merge_log.py [options]
```

- `--results-root PATH`
  Root directory containing `Phase1/Phase2/Phase3`.
  Default: `FinalResults/`
- `--config PATH`
  Project config path.
  Default: `config.json`
- `--phase 2`
  Process a single phase.
- `-p 1 3` or `--phases 1 3`
  Process multiple phases.
- `--eps-tags 0 m1 m2 m3 m4 p5`
  Subset of epsilon tags to process.
- `--device auto`
  Device selection: `auto`, `cpu`, or `cuda`.
- `--num-workers 8`
  Override the post-processing worker count.
- `--chunk-size 2000`
  Override the true-target chunk size.
- `--parallel-min-points 5000`
  Override the minimum point count for parallel target computation.
- `--single-hist-color tab:blue`
  Color for single-epsilon histograms.
- `--style-only`
  Redraw plots from cache only. Skip model inference and target recomputation.
- `--output-subdir 3_log_ratio`
  Output subdirectory under each phase.
- `--true-floor 1e-30`
  Lower bound for `|true_Ik|` in the ratio denominator.

Additional note:

- `--reuse-log-ratio` still exists for compatibility, but cache reuse is already the default behavior.
- Do not use `--phase` and `--phases` together.

### Examples

Process everything:

```bash
python FinalResults/merge_log.py
```

Process only Phase 2:

```bash
python FinalResults/merge_log.py --phase 2
```

Process selected epsilon tags:

```bash
python FinalResults/merge_log.py --phase 1 --eps-tags 0 m1 m4 p5
```

Redraw from cache only:

```bash
python FinalResults/merge_log.py --phase 3 --eps-tags 0 m1 --style-only
```

Use a different output subdirectory:

```bash
python FinalResults/merge_log.py --phase 1 --output-subdir 3_log_ratio_alt
```

### Output Files

Single-epsilon examples:

- `P1_LogRatio_eps_0.png`
- `P1_LogRatio_eps_m1.png`

Merged examples:

- `P1_LogRatio_eps_0_m1_m2_m3.png`
- `P1_LogRatio_eps_0_m1_m4_p5.png`

Cache examples:

- `P1_LogRatio_data_eps_0.npz`
- `P1_LogRatio_data_eps_m1.npz`

### Cache Behavior

Cache reuse is enabled by default.

For each `phase + eps` pair, the script first tries to load:

- `Phase*/<output_subdir>/cache/P{phase}_LogRatio_data_eps_{eps}.npz`

If that file is missing, it also checks the legacy cache location:

- `Phase*/P{phase}_LogRatio_data_eps_{eps}.npz`

If `--style-only` is provided:

- the script only redraws from cache
- it does not run model inference or recompute targets

### Notes

- Phases are numeric: `1`, `2`, `3`.
- Supported epsilon tags are: `0`, `m1`, `m2`, `m3`, `m4`, `p5`.
- Merged histogram groups follow the same built-in epsilon group definitions as `merge_loss.py`.
- Missing model files, eval bundles, or caches are skipped and reported.

## `merge_cos.py`

### Purpose

`merge_cos.py` computes and plots histograms of cosine similarity, using

`sum_k |true_k * pred_k| / (||true||_2 * ||pred||_2)`

for each sample. It generates:

- single-epsilon cosine-similarity histograms
- merged cosine-similarity histograms across selected epsilon tags

It also caches the per-sample cosine-similarity series in `.npz` files and reuses them on later runs when valid.

### Default Input/Output

Defaults:

- results root: `FinalResults/`
- config file: `config.json`

Within each phase:

- model directory: `0_models/` or `0_model/`
- output directory: `4_cos_sim/`
- cache directory: `4_cos_sim/cache/`

### Main Arguments

```bash
python FinalResults/merge_cos.py [options]
```

- `--results-root PATH`
  Root directory containing `Phase1/Phase2/Phase3`.
  Default: `FinalResults/`
- `--config PATH`
  Project config path.
  Default: `config.json`
- `--phase 2`
  Process a single phase.
- `-p 1 3` or `--phases 1 3`
  Process multiple phases.
- `--eps-tags 0 m1 m2 m3 m4 p5`
  Subset of epsilon tags to process.
- `--device auto`
  Device selection: `auto`, `cpu`, or `cuda`.
- `--num-workers 8`
  Override the post-processing worker count.
- `--chunk-size 2000`
  Override the true-target chunk size.
- `--parallel-min-points 5000`
  Override the minimum point count for parallel target computation.
- `--single-hist-color tab:blue`
  Color for single-epsilon histograms.
- `--style-only`
  Redraw plots from cache only. Skip model inference and target recomputation.
- `--output-subdir 4_cos_sim`
  Output subdirectory under each phase.
- `--denom-floor 1e-30`
  Lower bound for the cosine denominator `||true|| * ||pred||`.
- `--x-mode cos`
  Plot either raw cosine (`cos`) or transformed gap values (`gap`).
- `--gap-scale 1e7`
  Scaling factor used when `--x-mode gap`.
- `--cos-left-window 1e-2`
  When `--x-mode cos`, zoom the x-axis to `[1 - cos_left_window, 1]`.
- `--cos-right-pad -1`
  Right-side x-axis padding for cosine plots.

Additional note:

- `--reuse-cos-sim` still exists for compatibility, but cache reuse is already the default behavior.
- Do not use `--phase` and `--phases` together.

### Examples

Process everything:

```bash
python FinalResults/merge_cos.py
```

Process only Phase 2:

```bash
python FinalResults/merge_cos.py --phase 2
```

Process selected epsilon tags:

```bash
python FinalResults/merge_cos.py --phase 1 --eps-tags 0 m1 m4 p5
```

Redraw from cache only:

```bash
python FinalResults/merge_cos.py --phase 3 --eps-tags 0 m1 --style-only
```

Plot the transformed cosine gap instead of raw cosine:

```bash
python FinalResults/merge_cos.py --phase 2 --x-mode gap --gap-scale 1e7
```

Show the full cosine range instead of zooming near 1:

```bash
python FinalResults/merge_cos.py --phase 2 --x-mode cos --cos-left-window 0
```

### Output Files

Single-epsilon examples:

- `P1_CosSim_eps_0.png`
- `P1_CosSim_eps_m1.png`

Merged examples:

- `P1_CosSim_eps_0_m1_m2_m3.png`
- `P1_CosSim_eps_0_m1_m4_p5.png`

Cache examples:

- `P1_CosSim_data_eps_0.npz`
- `P1_CosSim_data_eps_m1.npz`

### Cache Behavior

Cache reuse is enabled by default.

For each `phase + eps` pair, the script first tries to load:

- `Phase*/<output_subdir>/cache/P{phase}_CosSim_data_eps_{eps}.npz`

If that file is missing, it also checks the legacy cache location:

- `Phase*/P{phase}_CosSim_data_eps_{eps}.npz`

If `--style-only` is provided:

- the script only redraws from cache
- it does not run model inference or recompute targets

### Notes

- Phases are numeric: `1`, `2`, `3`.
- Supported epsilon tags are: `0`, `m1`, `m2`, `m3`, `m4`, `p5`.
- Merged histogram groups follow the same built-in epsilon group definitions as `merge_loss.py`.
- Missing model files, eval bundles, or caches are skipped and reported.

## Quick Commands

```bash
# merge_loss.py: default run
python FinalResults/merge_loss.py

# merge_loss.py: Phase 1, selected epsilon tags
python FinalResults/merge_loss.py --phases 1 --eps-tags 0 m1 m4 p5

# merge_L12.py: Phase 2 only
python FinalResults/merge_L12.py --phase 2

# merge_L12.py: redraw selected Phase 3 epsilon tags from cache only
python FinalResults/merge_L12.py --phase 3 --eps-tags 0 m1 --style-only

# merge_log.py: Phase 2 only
python FinalResults/merge_log.py --phase 2

# merge_cos.py: Phase 2 with cosine-gap x-axis
python FinalResults/merge_cos.py --phase 2 --x-mode gap --gap-scale 1e7
```
