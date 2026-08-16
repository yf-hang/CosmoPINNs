import os
import tempfile
import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


def _ensure_torch_compile_debug_dir():
    env_name = "TORCH_COMPILE_DEBUG_DIR"
    current = os.environ.get(env_name, "").strip()
    if current:
        return current, False

    debug_root = os.path.join(tempfile.gettempdir(), "cosmopinns_torch_debug")
    os.makedirs(debug_root, exist_ok=True)
    os.environ[env_name] = debug_root
    return debug_root, True


def _loss_grad_norm_l2(loss, params):
    """Return the L2 norm of d(loss)/d(params) without changing .grad buffers."""
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += g.detach().pow(2).sum().item()
    return sq ** 0.5


def train_model_fixed_eps(
    model,
    a_builder,
    x_coll,
    x_b_tensor,
    bc_target,
    *,
    cde_loss_fixed_fn,
    bc_loss_fn,
    n_basis,
    eps_val,
    lr_init,
    warmup_len,
    total_epochs,
    lam1,
    lam2,
    cosine_min_lr=0.0,
    print_every=100,
    phase_name="P0",
    log_fn=None,
    # Kept only for compatibility with existing main.py calls. The old
    # total/global gradient-norm probe and clipping are no longer used.
    use_grad_norm_probe=False,
    grad_clip_max_norm=10.0,
    adaptive_lam1=True,
    lam1_update_every=50,
    lam1_ema=0.9,
    lam1_min=1e-3,
    lam1_max=1e3,
    lam1_grad_eps=1e-12,
):
    def _emit(msg: str):
        print(msg)
        if log_fn is not None:
            log_fn(msg)

    torch_debug_dir, torch_debug_dir_set = _ensure_torch_compile_debug_dir()
    if torch_debug_dir_set:
        _emit(f"[{phase_name}] set TORCH_COMPILE_DEBUG_DIR={torch_debug_dir}")

    optimizer = optim.Adam(model.parameters(), lr=lr_init)

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(total_epochs - warmup_len, 1),
        eta_min=cosine_min_lr,
    )

    lam1 = float(lam1)
    lam1_initial = float(lam1)
    lam2 = float(lam2)
    lam2_initial = float(lam2)

    lam1_update_every = max(int(lam1_update_every), 1)
    lam1_ema = min(max(float(lam1_ema), 0.0), 1.0)
    lam1_min = float(lam1_min)
    lam1_max = float(lam1_max)
    lam1_grad_eps = float(lam1_grad_eps)
    if lam1_min <= 0.0 or lam1_max < lam1_min:
        raise ValueError(
            f"Invalid adaptive lambda1 bounds: min={lam1_min}, max={lam1_max}"
        )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise ValueError("Model has no trainable parameters.")

    loss_tot_hist = []
    loss_cde_hist = []
    loss_bc_hist = []

    last_g_cde = float("nan")
    last_g_bc = float("nan")
    last_lam1_target = float("nan")
    lam1_has_adapted = False

    for step in range(1, total_epochs + 1):
        optimizer.zero_grad()

        loss_cde, Nc = cde_loss_fixed_fn(
            model, a_builder, x_coll, n_basis, eps_val=eps_val
        )
        loss_bc = bc_loss_fn(model, x_b_tensor, bc_target)

        # Keep lambda2 fixed and adapt only lambda1. The target value is chosen
        # so that the weighted CDE and BC gradient norms are approximately equal:
        #
        #   lambda1 * ||grad L_CDE|| ~= lambda2 * ||grad L_BC||.
        #
        # Gradient norms are measured only every lam1_update_every epochs to
        # avoid the cost of two extra autograd passes at every training step.
        do_lam1_update = adaptive_lam1 and (
            step == 1 or step % lam1_update_every == 0
        )
        if do_lam1_update:
            g_cde = _loss_grad_norm_l2(loss_cde, trainable_params)
            g_bc = _loss_grad_norm_l2(loss_bc, trainable_params)

            lam1_target = lam2 * g_bc / max(g_cde, lam1_grad_eps)
            lam1_target = min(max(lam1_target, lam1_min), lam1_max)

            # The first measured target does not depend on a manually chosen
            # initial lambda1. Subsequent updates use EMA smoothing.
            if not lam1_has_adapted:
                lam1 = lam1_target
                lam1_has_adapted = True
            else:
                lam1 = lam1_ema * lam1 + (1.0 - lam1_ema) * lam1_target
                lam1 = min(max(lam1, lam1_min), lam1_max)

            last_g_cde = float(g_cde)
            last_g_bc = float(g_bc)
            last_lam1_target = float(lam1_target)

        loss_total = lam1 * loss_cde + lam2 * loss_bc
        loss_total.backward()

        # Apply warm-up before optimizer.step() so the parameter update at
        # epoch t actually uses eta_t = lr_init * t / warmup_len.
        if step <= warmup_len:
            scale = step / float(warmup_len)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_init * scale

        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()

        # After warm-up, cosine annealing sets the learning rate for the next
        # parameter update.
        if step > warmup_len:
            cosine_scheduler.step()

        loss_tot_hist.append(loss_total.item())
        loss_cde_hist.append(loss_cde.item())
        loss_bc_hist.append(loss_bc.item())

        if step % print_every == 0 or step == 1 or step == total_epochs:
            tot_val = float(loss_total.item())
            cde_val = float(loss_cde.item())
            bc_val = float(loss_bc.item())
            cde_weighted_val = float(lam1) * cde_val
            bc_weighted_val = float(lam2) * bc_val

            if abs(tot_val) > 1e-30:
                cde_over_tot_pct = 100.0 * cde_weighted_val / tot_val
                bc_over_tot_pct = 100.0 * bc_weighted_val / tot_val
            else:
                cde_over_tot_pct = float("nan")
                bc_over_tot_pct = float("nan")

            msg = (
                f"[{step:04d}] "
                f"tot={tot_val:.3e} | "
                f"cde={cde_val:.3e} | "
                f"bc={bc_val:.3e} | "
                f"lambda1={float(lam1):.3e} | "
                f"lambda2={float(lam2):.3e} | "
                f"wCDE/tot={cde_over_tot_pct:.2f}% | "
                f"wBC/tot={bc_over_tot_pct:.2f}% | "
                f"g_cde={last_g_cde:.2e} | "
                f"g_bc={last_g_bc:.2e} | "
                f"lambda1_target={last_lam1_target:.2e} | "
                f"lr={float(lr_used):.2e}"
            )
            if step == 1:
                nb = int(x_b_tensor.shape[0])
                msg = (
                    f"Nc={int(Nc.item())}, Nb={nb}, "
                    f"adaptive_lambda1={bool(adaptive_lam1)}, "
                    f"lambda1_update_every={lam1_update_every}, "
                    f"lambda1_ema={lam1_ema:g}, "
                    f"lambda2_fixed={float(lam2):g}\n"
                    + msg
                )
            _emit(msg)

    final_tot = float(loss_tot_hist[-1]) if loss_tot_hist else float("nan")
    final_bc = float(loss_bc_hist[-1]) if loss_bc_hist else float("nan")
    final_bc_weighted = float(lam2) * final_bc
    if math.isfinite(final_tot) and abs(final_tot) > 1e-30:
        final_bc_tot_pct = 100.0 * final_bc_weighted / final_tot
    else:
        final_bc_tot_pct = float("nan")

    _emit(
        f"[{phase_name}] final loss weights: "
        f"lambda1_initial={float(lam1_initial):g}, "
        f"lambda1_final={float(lam1):g}, "
        f"lambda2_fixed={float(lam2):g}, "
        f"lam2*bc/tot={final_bc_tot_pct:.2f}%"
    )
    _emit(f" ------ {phase_name} fixed-eps training complete ------\n")

    return (
        model,
        loss_tot_hist,
        loss_cde_hist,
        loss_bc_hist,
        {
            "lambda1_initial": float(lam1_initial),
            "lambda2_initial": float(lam2_initial),
            "lambda1_final": float(lam1),
            "lambda2_final": float(lam2),
            "lambda1_adaptive": bool(adaptive_lam1),
            "lambda1_update_every": int(lam1_update_every),
            "lambda1_ema": float(lam1_ema),
            "lambda1_min": float(lam1_min),
            "lambda1_max": float(lam1_max),
            "loss_weight_final_bc_tot_pct": float(final_bc_tot_pct),
        },
    )