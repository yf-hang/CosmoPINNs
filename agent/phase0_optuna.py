#!/usr/bin/env python3
"""Optuna tuning for Phase 0 (tree-level two-site chain).

The repository phase convention is:

    Phase 0 = tree level
    Phase 1 = transfer to 1 loop
    Phase 2 = transfer to 2 loops
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lib.loss import boundary_loss, cde_residual_loss_fixed_eps  # noqa: E402
from lib.pinn_models import PinnModel  # noqa: E402
from two_site_chain.coll_bc import (  # noqa: E402
    build_inputs_and_boundary,
    compute_function_target_from_xcoll,
)
from two_site_chain.conn_mat import ConnectionAMatricesFixedWithEps0  # noqa: E402
from two_site_chain.mat_data import (  # noqa: E402
    a1,
    a1_eps0,
    a2,
    a2_eps0,
    a3,
    a3_eps0,
    a4,
    a4_eps0,
    a5,
    a5_eps0,
)


PHASE_MAP = {
    0: "tree-level two-site chain",
    1: "transfer to one-loop bubble",
    2: "transfer to two-loop sunset",
}


@dataclass
class Phase0Data:
    x_coll: torch.Tensor
    x_boundary: torch.Tensor
    boundary_target_scaled: torch.Tensor
    x_validation: torch.Tensor
    validation_target_scaled: torch.Tensor
    solution_scale: float


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def _normalise_output_part(value: Any) -> str:
    text = str(value if value is not None else "both").strip().lower()
    if text in {"both", "all", "reim", "complex"}:
        return "both"
    if text in {"re", "real"}:
        return "re"
    if text in {"im", "imag", "imaginary"}:
        return "im"
    raise ValueError(f"Unsupported phase0 output part: {value!r}")


def _eps_tag(value: Any) -> str:
    eps = float(value)
    if abs(eps) < 1e-12:
        return "0"
    magnitude = f"{abs(eps):.6g}".replace(".", "_")
    return f"m{magnitude}" if eps < 0 else f"p{magnitude}"


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(requested)


def _solution_scale(base: dict[str, Any], boundary_target: torch.Tensor) -> float:
    if not _as_bool(base.get("normalized_bc"), True):
        return 1.0

    mode = str(base.get("solution_scale_mode", "auto")).strip().lower()
    if mode == "manual":
        return float(base.get("solution_scale_p0", 1.0))
    if mode != "auto":
        raise ValueError("solution_scale_mode must be 'auto' or 'manual'.")

    mean_abs = float(boundary_target.abs().mean().item())
    if not math.isfinite(mean_abs) or mean_abs <= 0.0:
        raise ValueError(f"Invalid boundary target mean magnitude: {mean_abs}")
    reference = float(base.get("solution_scale_ref_mean", 0.1))
    scale_min = float(base.get("solution_scale_min_p0", base.get("solution_scale_min", 1e-30)))
    scale_max = float(base.get("solution_scale_max_p0", base.get("solution_scale_max", 1e12)))
    return max(scale_min, min(scale_max, reference / mean_abs))


def _build_phase0_data(
    base: dict[str, Any],
    tune: dict[str, Any],
    device: torch.device,
    output_part: str,
) -> Phase0Data:
    bc_config = base.get("coll_bc", {})
    if not isinstance(bc_config, dict):
        bc_config = {}

    _seed_everything(int(tune["train_data_seed"]))
    x_coll, x_boundary, boundary_target, _ = build_inputs_and_boundary(
        int(tune["n_coll"]),
        float(base["x1_min"]),
        float(base["x1_max"]),
        float(base["x2_min"]),
        float(base["x2_max"]),
        float(base["cy"]),
        float(base["eps_global"]),
        device,
        compute_function_target=False,
        output_part=output_part,
        n_bc_edge=int(tune.get("n_bc_edge", bc_config.get("n_bc_edge", 15))),
        n_corner_each=int(tune.get("n_corner_each", bc_config.get("n_corner_each", 15))),
        target_total=int(tune["n_boundary"]),
    )

    scale = _solution_scale(base, boundary_target)
    boundary_target_scaled = boundary_target * scale

    validation_rng = np.random.default_rng(int(tune["validation_seed"]))
    n_validation = int(tune["n_validation"])
    validation_np = np.column_stack(
        [
            validation_rng.uniform(base["x1_min"], base["x1_max"], n_validation),
            validation_rng.uniform(base["x2_min"], base["x2_max"], n_validation),
        ]
    ).astype(np.float32)
    x_validation = torch.as_tensor(validation_np, device=device)
    validation_target = compute_function_target_from_xcoll(
        x_validation,
        cy_val=float(base["cy"]),
        eps_val=float(base["eps_global"]),
        output_part=output_part,
        num_workers=int(tune.get("validation_workers", 1)),
        chunk_size=int(tune.get("validation_chunk_size", 2000)),
        parallel_min_points=int(tune.get("validation_parallel_min_points", 5000)),
    )

    return Phase0Data(
        x_coll=x_coll,
        x_boundary=x_boundary,
        boundary_target_scaled=boundary_target_scaled,
        x_validation=x_validation,
        validation_target_scaled=validation_target * scale,
        solution_scale=scale,
    )


def _build_matrix_module(device: torch.device, cy: float) -> ConnectionAMatricesFixedWithEps0:
    regular = [matrix.to(device) for matrix in (a1, a2, a3, a4, a5)]
    eps_zero = [matrix.to(device) for matrix in (a1_eps0, a2_eps0, a3_eps0, a4_eps0, a5_eps0)]
    return ConnectionAMatricesFixedWithEps0(
        ak_list=regular,
        ak_list_eps0=eps_zero,
        cy_val=cy,
    ).to(device)


def _validation_metrics(
    model: torch.nn.Module,
    x_validation: torch.Tensor,
    target_scaled: torch.Tensor,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        prediction = model(x_validation)
        difference_norm = torch.linalg.vector_norm(prediction - target_scaled, dim=1)
        target_norm = torch.linalg.vector_norm(target_scaled, dim=1)
        denominator_floor = max(float(torch.median(target_norm).item()) * 1e-6, 1e-12)
        relative_l2 = difference_norm / target_norm.clamp_min(denominator_floor)
        median = float(torch.quantile(relative_l2, 0.50).item())
        p90 = float(torch.quantile(relative_l2, 0.90).item())
        mean = float(relative_l2.mean().item())
        # Median measures the typical point; p90 prevents a good median from
        # hiding a broad error tail.
        score = median + 0.25 * p90
    model.train()
    return {
        "score": score,
        "relative_l2_median": median,
        "relative_l2_p90": p90,
        "relative_l2_mean": mean,
    }


class Phase0Objective:
    def __init__(
        self,
        *,
        optuna_module: Any,
        base: dict[str, Any],
        tune: dict[str, Any],
        data: Phase0Data,
        device: torch.device,
        output_dir: Path,
        incumbent_score: float,
    ) -> None:
        self.optuna = optuna_module
        self.base = base
        self.tune = tune
        self.data = data
        self.device = device
        self.output_dir = output_dir
        self.output_part = _normalise_output_part(base.get("phase0_output_part", "both"))
        self.matrix_module = _build_matrix_module(device, float(base["cy"]))
        self.incumbent_score = incumbent_score

    def _boundary_loss(self, model: torch.nn.Module) -> torch.Tensor:
        use_normalized = _as_bool(
            self.base.get("bc_loss_use_normalized"),
            _as_bool(self.base.get("normalized_bc"), True),
        )
        return boundary_loss(
            model,
            self.data.x_boundary,
            self.data.boundary_target_scaled,
            use_normalized=use_normalized,
            scale_floor=float(self.base.get("bc_loss_scale_floor", 1e-4)),
            min_scale_ratio=float(self.base.get("bc_loss_min_scale_ratio", 1.0)),
            abs_mse_weight=float(self.base.get("bc_loss_abs_mse_weight", 0.05)),
            output_part=self.output_part,
        )

    def __call__(self, trial: Any) -> float:
        learning_rate = trial.suggest_float(
            "learning_rate",
            float(self.tune["learning_rate_min"]),
            float(self.tune["learning_rate_max"]),
            log=True,
        )
        cde_bc_ratio = trial.suggest_float(
            "cde_bc_ratio",
            float(self.tune["cde_bc_ratio_min"]),
            float(self.tune["cde_bc_ratio_max"]),
            log=True,
        )

        # Keeping the same initialisation for every trial makes comparisons
        # reflect hyperparameters rather than a lucky random seed.
        _seed_everything(int(self.tune["model_seed"]))
        config = SimpleNamespace(**self.base)
        model = PinnModel(config, in_dim=2, output_part=self.output_part).to(self.device)
        optimizer = Adam(model.parameters(), lr=learning_rate)

        epochs = int(self.tune["epochs"])
        warmup_epochs = min(int(self.tune["warmup_epochs"]), epochs)
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(epochs - warmup_epochs, 1),
            eta_min=float(self.tune.get("cosine_min_lr", self.base.get("cosine_min_lr", 0.0))),
        )
        eval_every = max(int(self.tune["eval_every"]), 1)
        last_losses: dict[str, float] = {}
        started = time.perf_counter()

        try:
            for epoch in range(1, epochs + 1):
                optimizer.zero_grad(set_to_none=True)
                cde_loss, _ = cde_residual_loss_fixed_eps(
                    model,
                    self.matrix_module,
                    self.data.x_coll,
                    int(self.base["n_basis"]),
                    eps_val=float(self.base["eps_global"]),
                    output_part=self.output_part,
                )
                bc_loss_value = self._boundary_loss(model)
                total_loss = cde_bc_ratio * cde_loss + bc_loss_value
                if not bool(torch.isfinite(total_loss).item()):
                    raise self.optuna.TrialPruned("Non-finite training loss.")

                total_loss.backward()
                optimizer.step()

                # Match the legacy lib/train.py warmup/cosine update order so
                # selected parameters transfer directly to the main trainer.
                if epoch <= warmup_epochs and warmup_epochs > 0:
                    scale = epoch / float(warmup_epochs)
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = learning_rate * scale
                else:
                    scheduler.step()

                if epoch == 1 or epoch % eval_every == 0 or epoch == epochs:
                    metrics = _validation_metrics(
                        model,
                        self.data.x_validation,
                        self.data.validation_target_scaled,
                    )
                    trial.report(metrics["score"], step=epoch)
                    last_losses = {
                        "train_total_loss": float(total_loss.detach().item()),
                        "train_cde_loss": float(cde_loss.detach().item()),
                        "train_bc_loss": float(bc_loss_value.detach().item()),
                        **metrics,
                    }
                    if trial.should_prune():
                        raise self.optuna.TrialPruned(f"Pruned at epoch {epoch}.")

            elapsed = time.perf_counter() - started
            for name, value in last_losses.items():
                trial.set_user_attr(name, value)
            trial.set_user_attr("elapsed_seconds", elapsed)
            trial.set_user_attr("bc_weight", 1.0)
            trial.set_user_attr("cde_weight", cde_bc_ratio)
            trial.set_user_attr("solution_scale", self.data.solution_scale)

            score = float(last_losses["score"])
            if _as_bool(self.tune.get("save_best_checkpoint"), True) and score < self.incumbent_score:
                self.incumbent_score = score
                checkpoint = {
                    "phase": 0,
                    "phase_name": PHASE_MAP[0],
                    # Use main.py's checkpoint key so this file can be loaded
                    # through phase0_model_load_path if desired.
                    "model_state_dict": model.state_dict(),
                    "input_dim": 2,
                    "n_basis": int(self.base["n_basis"]),
                    "output_part": self.output_part,
                    "pred_scale": self.data.solution_scale,
                    "learning_rate": learning_rate,
                    "cde_weight": cde_bc_ratio,
                    "bc_weight": 1.0,
                    "validation_metrics": last_losses,
                    "trial_number": int(trial.number),
                }
                torch.save(checkpoint, self.output_dir / "best_phase0_checkpoint.pt")
            return score
        finally:
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def _write_trials_csv(study: Any, path: Path) -> None:
    fieldnames = [
        "number",
        "state",
        "value",
        "learning_rate",
        "cde_bc_ratio",
        "relative_l2_median",
        "relative_l2_p90",
        "relative_l2_mean",
        "train_cde_loss",
        "train_bc_loss",
        "elapsed_seconds",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in study.trials:
            writer.writerow(
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "learning_rate": trial.params.get("learning_rate"),
                    "cde_bc_ratio": trial.params.get("cde_bc_ratio"),
                    "relative_l2_median": trial.user_attrs.get("relative_l2_median"),
                    "relative_l2_p90": trial.user_attrs.get("relative_l2_p90"),
                    "relative_l2_mean": trial.user_attrs.get("relative_l2_mean"),
                    "train_cde_loss": trial.user_attrs.get("train_cde_loss"),
                    "train_bc_loss": trial.user_attrs.get("train_bc_loss"),
                    "elapsed_seconds": trial.user_attrs.get("elapsed_seconds"),
                }
            )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "config.json")
    parser.add_argument(
        "--tune-config",
        type=Path,
        default=REPO_ROOT / "agent" / "phase0_optuna_config.json",
    )
    parser.add_argument("--n-trials", type=int, help="Override n_trials from the tune config.")
    parser.add_argument("--timeout", type=float, help="Override timeout_seconds.")
    parser.add_argument("--device", help="Override device (auto/cpu/cuda).")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use a tiny one-trial run to verify plumbing; not meaningful tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Run: python -m pip install -r agent/requirements.txt"
        ) from exc

    base = _read_json(args.base_config.resolve())
    tune = _read_json(args.tune_config.resolve())

    if args.n_trials is not None:
        tune["n_trials"] = args.n_trials
    if args.timeout is not None:
        tune["timeout_seconds"] = args.timeout
    if args.device is not None:
        tune["device"] = args.device
    if args.smoke_test:
        tune.update(
            {
                "n_trials": 1,
                "n_coll": min(int(tune["n_coll"]), 32),
                "n_boundary": min(int(tune["n_boundary"]), 24),
                "n_validation": min(int(tune["n_validation"]), 16),
                "epochs": 2,
                "warmup_epochs": 1,
                "eval_every": 1,
                "n_bc_edge": 3,
                "n_corner_each": 3,
                "study_name": f'{tune["study_name"]}_smoke',
                "output_dir": f'{tune.get("output_dir", "agent/phase0_optuna_results")}_smoke',
            }
        )

    format_values = {
        "eps": float(base["eps_global"]),
        "eps_tag": _eps_tag(base["eps_global"]),
        "output_part": _normalise_output_part(base.get("phase0_output_part", "both")),
    }
    tune["study_name"] = str(tune["study_name"]).format(**format_values)
    output_dir_text = str(tune.get("output_dir", "agent/phase0_optuna_results")).format(
        **format_values
    )
    output_dir_config = Path(output_dir_text)
    output_dir = output_dir_config if output_dir_config.is_absolute() else REPO_ROOT / output_dir_config
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = tune.get("storage")
    if not storage:
        storage = f"sqlite:///{(output_dir / 'study.sqlite3').as_posix()}"

    device = _resolve_device(str(tune.get("device", base.get("device", "auto"))))
    output_part = _normalise_output_part(base.get("phase0_output_part", "both"))
    print(f"Phase mapping: {PHASE_MAP}")
    print(f"Preparing fixed Phase 0 data on {device} (output_part={output_part})...")
    data = _build_phase0_data(base, tune, device, output_part)
    print(
        f"Data ready: coll={len(data.x_coll)}, boundary={len(data.x_boundary)}, "
        f"validation={len(data.x_validation)}, solution_scale={data.solution_scale:.6g}"
    )

    sampler = optuna.samplers.TPESampler(seed=int(tune["sampler_seed"]))
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(tune["pruner_startup_trials"]),
        n_warmup_steps=int(tune["pruner_warmup_epochs"]),
        interval_steps=max(int(tune["eval_every"]), 1),
    )
    study = optuna.create_study(
        study_name=str(tune["study_name"]),
        storage=storage,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )
    signature = {
        "eps_global": float(base["eps_global"]),
        "output_part": output_part,
        "domain": [
            float(base["x1_min"]),
            float(base["x1_max"]),
            float(base["x2_min"]),
            float(base["x2_max"]),
        ],
        "hidden_size": int(base["hidden_size"]),
        "n_hidden_layers": int(base["n_hidden_layers"]),
        "n_coll": int(tune["n_coll"]),
        "n_boundary": int(tune["n_boundary"]),
        "n_validation": int(tune["n_validation"]),
        "epochs": int(tune["epochs"]),
        "warmup_epochs": int(tune["warmup_epochs"]),
        "train_data_seed": int(tune["train_data_seed"]),
        "validation_seed": int(tune["validation_seed"]),
        "model_seed": int(tune["model_seed"]),
    }
    previous_signature = study.user_attrs.get("phase0_signature")
    if previous_signature is not None and previous_signature != signature:
        raise RuntimeError(
            "The existing Optuna study was created with a different Phase 0 setup. "
            "Use a new study_name/output directory, or restore the original config."
        )
    study.set_user_attr("phase0_signature", signature)
    completed = [
        trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    incumbent = min((float(trial.value) for trial in completed), default=math.inf)
    objective = Phase0Objective(
        optuna_module=optuna,
        base=base,
        tune=tune,
        data=data,
        device=device,
        output_dir=output_dir,
        incumbent_score=incumbent,
    )

    timeout = tune.get("timeout_seconds")
    timeout = None if timeout is None else float(timeout)
    study.optimize(
        objective,
        n_trials=int(tune["n_trials"]),
        timeout=timeout,
        gc_after_trial=True,
        show_progress_bar=_as_bool(tune.get("show_progress_bar"), True),
    )

    best = study.best_trial
    summary = {
        "phase": 0,
        "phase_name": PHASE_MAP[0],
        "study_name": study.study_name,
        "storage": storage,
        "best_trial": int(best.number),
        "objective": "median(relative_L2) + 0.25 * p90(relative_L2)",
        "best_value": float(best.value),
        "best_params": {
            "learning_rate": float(best.params["learning_rate"]),
            "cde_weight": float(best.params["cde_bc_ratio"]),
            "bc_weight": 1.0,
            "cde_bc_ratio": float(best.params["cde_bc_ratio"]),
        },
        "best_metrics": dict(best.user_attrs),
        "solution_scale": data.solution_scale,
        "base_config": str(args.base_config.resolve()),
        "tune_config": str(args.tune_config.resolve()),
        "n_trials_total": len(study.trials),
    }
    _write_json(output_dir / "best_params.json", summary)
    _write_trials_csv(study, output_dir / "trials.csv")

    print("\nBest Phase 0 parameters")
    print(json.dumps(summary["best_params"], indent=2))
    print(f"Validation objective: {best.value:.6e}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
