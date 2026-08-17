#!/usr/bin/env python3
"""Independent Phase-0 test of adaptive CDE/BC loss weighting.

The training loss is

    L = L_CDE + lambda_bc * L_BC

The adaptive BC weight is estimated from gradient magnitudes:

    lambda_bc_hat = max(|grad L_CDE|) / mean(|grad L_BC|)
    lambda_bc <- (1 - alpha) * lambda_bc + alpha * lambda_bc_hat

For comparison with the repository convention

    L = lambda1 * L_CDE + lambda2 * L_BC,

the script also reports

    lambda1_ref = lambda2 / lambda_bc.

The reported lambda1_ref is only a relative-weight reference. The optimizer
uses the adaptive loss shown above.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
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

from agent.phase0_optuna import (  # noqa: E402
    PHASE_MAP,
    _as_bool,
    _build_matrix_module,
    _build_phase0_data,
    _eps_tag,
    _normalise_output_part,
    _read_json,
    _resolve_device,
    _seed_everything,
    _validation_metrics,
    _write_json,
)
from lib.loss import boundary_loss, cde_residual_loss_fixed_eps  # noqa: E402
from lib.pinn_models import PinnModel  # noqa: E402


def _gradient_statistics(
    cde_loss: torch.Tensor,
    bc_loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> tuple[float, float]:
    cde_grads = torch.autograd.grad(
        cde_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    bc_grads = torch.autograd.grad(
        bc_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )

    cde_max = 0.0
    bc_abs_sum = 0.0
    parameter_count = 0

    for parameter, cde_grad, bc_grad in zip(parameters, cde_grads, bc_grads):
        parameter_count += parameter.numel()
        if cde_grad is not None:
            cde_max = max(cde_max, float(cde_grad.detach().abs().max().item()))
        if bc_grad is not None:
            bc_abs_sum += float(bc_grad.detach().abs().sum().item())

    if parameter_count <= 0:
        raise RuntimeError("Model has no trainable parameters.")

    bc_mean = bc_abs_sum / float(parameter_count)
    if not math.isfinite(cde_max) or not math.isfinite(bc_mean):
        raise RuntimeError(
            f"Non-finite gradient statistics: cde_max={cde_max}, bc_mean={bc_mean}"
        )
    if bc_mean <= 0.0:
        raise RuntimeError(
            "Mean absolute BC gradient is zero, so the adaptive weight is undefined."
        )
    return cde_max, bc_mean


def _write_history_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "epoch",
        "learning_rate",
        "lambda_bc",
        "lambda1_ref",
        "lambda_bc_hat",
        "grad_cde_max_abs",
        "grad_bc_mean_abs",
        "train_total_loss",
        "train_cde_loss",
        "train_bc_loss",
        "relative_l2_mean_outputs",
        "relative_l2_output_0",
        "relative_l2_output_1",
        "relative_l2_output_2",
        "relative_l2_output_3",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=REPO_ROOT / "config.json")
    parser.add_argument(
        "--test-config",
        type=Path,
        default=REPO_ROOT / "testWeight" / "adaptive_lambda1_config.json",
    )
    parser.add_argument("--device", help="Override device (auto/cpu/cuda).")
    parser.add_argument("--epochs", type=int, help="Override test-config epochs.")
    parser.add_argument(
        "--adaptive-every",
        type=int,
        help="Override how often the adaptive weight is updated.",
    )
    parser.add_argument("--alpha", type=float, help="Override moving-average alpha.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny plumbing test; results are not scientifically meaningful.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    base = _read_json(args.base_config.resolve())
    test_cfg = _read_json(args.test_config.resolve())

    if args.device is not None:
        test_cfg["device"] = args.device
    if args.epochs is not None:
        test_cfg["epochs"] = args.epochs
    if args.adaptive_every is not None:
        test_cfg["adaptive_every"] = args.adaptive_every
    if args.alpha is not None:
        test_cfg["adaptive_alpha"] = args.alpha

    if args.smoke_test:
        test_cfg.update(
            {
                "n_coll": min(int(test_cfg["n_coll"]), 32),
                "n_boundary": min(int(test_cfg["n_boundary"]), 24),
                "n_validation": min(int(test_cfg["n_validation"]), 16),
                "epochs": 2,
                "warmup_epochs": 1,
                "eval_every": 1,
                "adaptive_every": 1,
                "n_bc_edge": 3,
                "n_corner_each": 3,
                "output_dir": f'{test_cfg["output_dir"]}_smoke',
            }
        )

    alpha = float(test_cfg.get("adaptive_alpha", 0.9))
    if not (0.0 < alpha <= 1.0):
        raise ValueError(f"adaptive_alpha must be in (0, 1], got {alpha}")
    adaptive_every = int(test_cfg.get("adaptive_every", 10))
    if adaptive_every <= 0:
        raise ValueError(f"adaptive_every must be positive, got {adaptive_every}")
    lambda_bc = float(test_cfg.get("lambda_bc_initial", 1.0))
    if not math.isfinite(lambda_bc) or lambda_bc <= 0.0:
        raise ValueError(f"lambda_bc_initial must be finite and positive, got {lambda_bc}")

    output_part = _normalise_output_part(base.get("phase0_output_part", "re"))
    format_values = {
        "eps": float(base["eps_global"]),
        "eps_tag": _eps_tag(base["eps_global"]),
        "output_part": output_part,
    }
    output_dir_text = str(test_cfg.get("output_dir", "testWeight/results"))
    output_dir_text = output_dir_text.format(**format_values)
    output_dir_config = Path(output_dir_text)
    output_dir = output_dir_config if output_dir_config.is_absolute() else REPO_ROOT / output_dir_config
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(str(test_cfg.get("device", base.get("device", "auto"))))
    use_solution_scale = _as_bool(base.get("use_solution_scale"), False)
    use_bc_channel_normalization = _as_bool(
        base.get("use_bc_channel_normalization"), False
    )
    lambda2_reference = float(base.get("lambda2", 1.0))
    if not math.isfinite(lambda2_reference) or lambda2_reference <= 0.0:
        raise ValueError(
            "lambda2 must be finite and positive to define lambda1_ref=lambda2/lambda_bc."
        )

    print(f"Phase mapping: {PHASE_MAP}")
    print(f"Adaptive Phase 0 weight test on {device} (output_part={output_part})")
    print(
        "Normalization switches: "
        f"use_solution_scale={use_solution_scale}, "
        f"use_bc_channel_normalization={use_bc_channel_normalization}"
    )
    print(
        "Adaptive update: "
        f"alpha={alpha:g}, every={adaptive_every} epoch(s), "
        f"lambda_bc_initial={lambda_bc:g}"
    )
    print(
        "Current-convention reference: "
        f"lambda1_ref=lambda2/lambda_bc with lambda2={lambda2_reference:g}"
    )

    data = _build_phase0_data(base, test_cfg, device, output_part)
    print(
        f"Data ready: coll={len(data.x_coll)}, boundary={len(data.x_boundary)}, "
        f"validation={len(data.x_validation)}, solution_scale={data.solution_scale:.6g}"
    )

    _seed_everything(int(test_cfg["model_seed"]))
    model = PinnModel(SimpleNamespace(**base), in_dim=2, output_part=output_part).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    matrix_module = _build_matrix_module(device, float(base["cy"]))

    learning_rate = float(base["learning_rate_p0"])
    optimizer = Adam(model.parameters(), lr=learning_rate)
    epochs = int(test_cfg["epochs"])
    warmup_epochs = min(int(test_cfg["warmup_epochs"]), epochs)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(epochs - warmup_epochs, 1),
        eta_min=float(test_cfg.get("cosine_min_lr", base.get("cosine_min_lr", 0.0))),
    )
    eval_every = max(int(test_cfg.get("eval_every", 100)), 1)

    history: list[dict[str, Any]] = []
    lambda1_values: list[float] = []
    best_score = math.inf
    best_epoch = -1
    best_metrics: dict[str, float] = {}
    best_lambda_bc = lambda_bc
    best_lambda1_ref = lambda2_reference / lambda_bc
    latest_grad_cde_max: float | None = None
    latest_grad_bc_mean: float | None = None
    latest_lambda_bc_hat: float | None = None
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            warmup_scale = epoch / float(warmup_epochs)
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate * warmup_scale

        current_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.zero_grad(set_to_none=True)

        cde_loss, _ = cde_residual_loss_fixed_eps(
            model,
            matrix_module,
            data.x_coll,
            int(base["n_basis"]),
            eps_val=float(base["eps_global"]),
            output_part=output_part,
        )
        bc_loss_value = boundary_loss(
            model,
            data.x_boundary,
            data.boundary_target_scaled,
            use_normalized=use_bc_channel_normalization,
            scale_floor=float(base.get("bc_loss_scale_floor", 1e-4)),
            min_scale_ratio=float(base.get("bc_loss_min_scale_ratio", 1.0)),
            abs_mse_weight=float(base.get("bc_loss_abs_mse_weight", 0.05)),
            output_part=output_part,
        )

        should_adapt = epoch == 1 or epoch % adaptive_every == 0
        lambda_bc_hat: float | None = None
        grad_cde_max: float | None = None
        grad_bc_mean: float | None = None
        if should_adapt:
            grad_cde_max, grad_bc_mean = _gradient_statistics(
                cde_loss,
                bc_loss_value,
                parameters,
            )
            lambda_bc_hat = grad_cde_max / grad_bc_mean
            if not math.isfinite(lambda_bc_hat) or lambda_bc_hat < 0.0:
                raise RuntimeError(f"Invalid adaptive lambda_bc_hat={lambda_bc_hat}")
            lambda_bc = (1.0 - alpha) * lambda_bc + alpha * lambda_bc_hat
            if not math.isfinite(lambda_bc) or lambda_bc <= 0.0:
                raise RuntimeError(f"Invalid adaptive lambda_bc={lambda_bc}")
            latest_grad_cde_max = grad_cde_max
            latest_grad_bc_mean = grad_bc_mean
            latest_lambda_bc_hat = lambda_bc_hat

        lambda1_ref = lambda2_reference / lambda_bc
        lambda1_values.append(lambda1_ref)

        total_loss = cde_loss + lambda_bc * bc_loss_value
        if not bool(torch.isfinite(total_loss).item()):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
        total_loss.backward()
        optimizer.step()

        if epoch > warmup_epochs:
            scheduler.step()

        metrics: dict[str, float] = {}
        should_eval = epoch == 1 or epoch % eval_every == 0 or epoch == epochs
        if should_eval:
            metrics = _validation_metrics(
                model,
                data.x_validation,
                data.validation_target_scaled,
            )
            score = float(metrics["score"])
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_metrics = dict(metrics)
                best_lambda_bc = lambda_bc
                best_lambda1_ref = lambda1_ref
                if _as_bool(test_cfg.get("save_best_checkpoint"), True):
                    torch.save(
                        {
                            "phase": 0,
                            "phase_name": PHASE_MAP[0],
                            "method": "adaptive_loss_weight",
                            "model_state_dict": model.state_dict(),
                            "input_dim": 2,
                            "n_basis": int(base["n_basis"]),
                            "output_part": output_part,
                            "pred_scale": data.solution_scale,
                            "learning_rate": learning_rate,
                            "lambda_bc": lambda_bc,
                            "lambda1_ref": lambda1_ref,
                            "lambda2_reference": lambda2_reference,
                            "adaptive_alpha": alpha,
                            "adaptive_every": adaptive_every,
                            "epoch": epoch,
                            "validation_metrics": metrics,
                            "use_solution_scale": use_solution_scale,
                            "use_bc_channel_normalization": use_bc_channel_normalization,
                        },
                        output_dir / "best_phase0_adaptive_checkpoint.pt",
                    )

            print(
                f"epoch={epoch:5d} "
                f"lambda_bc={lambda_bc:.6g} "
                f"lambda1_ref={lambda1_ref:.6g} "
                f"CDE={float(cde_loss.detach().item()):.6e} "
                f"BC={float(bc_loss_value.detach().item()):.6e} "
                f"val={score:.6e}"
            )

        row: dict[str, Any] = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "lambda_bc": lambda_bc,
            "lambda1_ref": lambda1_ref,
            "lambda_bc_hat": lambda_bc_hat,
            "grad_cde_max_abs": grad_cde_max,
            "grad_bc_mean_abs": grad_bc_mean,
            "train_total_loss": float(total_loss.detach().item()),
            "train_cde_loss": float(cde_loss.detach().item()),
            "train_bc_loss": float(bc_loss_value.detach().item()),
            "relative_l2_mean_outputs": metrics.get("relative_l2_mean_outputs"),
        }
        for index in range(4):
            row[f"relative_l2_output_{index}"] = metrics.get(
                f"relative_l2_output_{index}"
            )
        history.append(row)

    elapsed = time.perf_counter() - started
    _write_history_csv(output_dir / "history.csv", history)

    final_lambda1_ref = lambda2_reference / lambda_bc
    lambda1_np = np.asarray(lambda1_values, dtype=float)
    post_warmup = lambda1_np[warmup_epochs:] if warmup_epochs < len(lambda1_np) else lambda1_np
    summary = {
        "phase": 0,
        "phase_name": PHASE_MAP[0],
        "method": "adaptive_loss_weight",
        "training_loss": "L_CDE + lambda_bc * L_BC",
        "lambda1_reference_definition": "lambda1_ref = lambda2 / lambda_bc",
        "eps_global": float(base["eps_global"]),
        "output_part": output_part,
        "learning_rate_p0": learning_rate,
        "lambda2_reference": lambda2_reference,
        "adaptive_alpha": alpha,
        "adaptive_every": adaptive_every,
        "epochs": epochs,
        "warmup_epochs": warmup_epochs,
        "solution_scale": data.solution_scale,
        "use_solution_scale": use_solution_scale,
        "use_bc_channel_normalization": use_bc_channel_normalization,
        "final_lambda_bc": lambda_bc,
        "final_lambda1_ref": final_lambda1_ref,
        "post_warmup_lambda1_ref_median": float(np.median(post_warmup)),
        "post_warmup_lambda1_ref_min": float(np.min(post_warmup)),
        "post_warmup_lambda1_ref_max": float(np.max(post_warmup)),
        "best_validation_epoch": best_epoch,
        "best_validation_score": best_score,
        "best_validation_metrics": best_metrics,
        "best_validation_lambda_bc": best_lambda_bc,
        "best_validation_lambda1_ref": best_lambda1_ref,
        "latest_grad_cde_max_abs": latest_grad_cde_max,
        "latest_grad_bc_mean_abs": latest_grad_bc_mean,
        "latest_lambda_bc_hat": latest_lambda_bc_hat,
        "elapsed_seconds": elapsed,
        "base_config": str(args.base_config.resolve()),
        "test_config": str(args.test_config.resolve()),
        "history_csv": str((output_dir / "history.csv").resolve()),
    }
    _write_json(output_dir / "summary.json", summary)

    torch.save(
        {
            "phase": 0,
            "phase_name": PHASE_MAP[0],
            "method": "adaptive_loss_weight",
            "model_state_dict": model.state_dict(),
            "input_dim": 2,
            "n_basis": int(base["n_basis"]),
            "output_part": output_part,
            "pred_scale": data.solution_scale,
            "lambda_bc": lambda_bc,
            "lambda1_ref": final_lambda1_ref,
            "lambda2_reference": lambda2_reference,
            "adaptive_alpha": alpha,
            "adaptive_every": adaptive_every,
            "validation_metrics": best_metrics,
        },
        output_dir / "final_phase0_adaptive_checkpoint.pt",
    )

    print("\nAdaptive Phase 0 weight summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
