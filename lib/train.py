import os
import tempfile
import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as nn_utils


def _ensure_torch_compile_debug_dir():
    env_name = "TORCH_COMPILE_DEBUG_DIR"
    current = os.environ.get(env_name, "").strip()
    if current:
        return current, False

    debug_root = os.path.join(tempfile.gettempdir(), "cosmopinns_torch_debug")
    os.makedirs(debug_root, exist_ok=True)
    os.environ[env_name] = debug_root
    return debug_root, True


def _grad_norm_l2(model):
    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.detach().pow(2).sum().item()
    return sq ** 0.5


def _loss_parameter_grad_norm(loss, parameters):
    """L2 norm over all trainable parameters, without touching parameter .grad."""
    if not parameters or not loss.requires_grad:
        return 0.0
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, create_graph=False, allow_unused=True
    )
    squares = [g.detach().to(dtype=torch.float64).square().sum()
               for g in gradients if g is not None]
    norm = float(torch.stack(squares).sum().sqrt().item()) if squares else 0.0
    if not math.isfinite(norm):
        raise FloatingPointError("Non-finite loss gradient norm during weight update.")
    return norm


def _gradient_balanced_weights(loss_cde, loss_bc, parameters, lam1, lam2, ema):
    """Update both weights from unweighted losses; returned scalars are detached."""
    trainable = tuple(p for p in parameters if p.requires_grad)
    g_cde = _loss_parameter_grad_norm(loss_cde, trainable)
    g_bc = _loss_parameter_grad_norm(loss_bc, trainable)
    total_norm = g_cde + g_bc
    # A zero gradient leaves that term's old weight unchanged; no division floor.
    if g_cde > 0.0:
        lam1 = ema * lam1 + (1.0 - ema) * (total_norm / g_cde)
    if g_bc > 0.0:
        lam2 = ema * lam2 + (1.0 - ema) * (total_norm / g_bc)
    if not math.isfinite(lam1) or not math.isfinite(lam2):
        raise FloatingPointError("Non-finite gradient-balanced loss weight.")
    return float(lam1), float(lam2), g_cde, g_bc


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
    use_grad_norm_probe=False,
    grad_clip_max_norm=10.0,
    loss_weighting="fixed",
    loss_weight_update_every=1000,
    loss_weight_ema=0.9,
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
    lam2 = float(lam2)
    lam1_initial = float(lam1)
    lam2_initial = float(lam2)
    if not all(math.isfinite(w) and w > 0.0 for w in (lam1, lam2)):
        raise ValueError("Initial loss weights must be finite and positive.")
    loss_weighting = str(loss_weighting).strip().lower()
    if loss_weighting not in {"fixed", "grad_norm"}:
        raise ValueError("loss_weighting must be 'fixed' or 'grad_norm'.")
    loss_weight_update_every = int(loss_weight_update_every)
    loss_weight_ema = float(loss_weight_ema)
    if loss_weight_update_every <= 0:
        raise ValueError("loss_weight_update_every must be positive.")
    if not 0.0 <= loss_weight_ema < 1.0:
        raise ValueError("loss_weight_ema must be in [0, 1).")
    trainable = tuple(p for p in model.parameters() if p.requires_grad)
    weight_updates = []
    lambda1_history = []
    lambda2_history = []

    loss_tot_hist = []
    loss_cde_hist = []
    loss_bc_hist = []

    for step in range(1, total_epochs + 1):
        optimizer.zero_grad()

        loss_cde, Nc = cde_loss_fixed_fn(
            model, a_builder, x_coll, n_basis, eps_val=eps_val
        )
        loss_bc = bc_loss_fn(model, x_b_tensor, bc_target)

        # Measure unweighted loss gradients before backward/clipping. Each phase
        # starts from the configured initial weights; first update is at step K.
        if loss_weighting == "grad_norm" and step % loss_weight_update_every == 0:
            lam1, lam2, g_cde, g_bc = _gradient_balanced_weights(
                loss_cde, loss_bc, trainable, lam1, lam2, loss_weight_ema
            )
            weight_updates.append(dict(
                step=step, grad_cde_l2=g_cde, grad_bc_l2=g_bc,
                lambda1=lam1, lambda2=lam2,
            ))
            _emit(
                f"[{phase_name}] weight update step={step}: "
                f"g_cde={g_cde:.6e}, g_bc={g_bc:.6e}, "
                f"lambda1={lam1:.6g}, lambda2={lam2:.6g}"
            )

        loss_total = lam1 * loss_cde + lam2 * loss_bc
        loss_total.backward()

        if use_grad_norm_probe:
            # grad-norm probe + optional clipping
            gn = float(nn_utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float(grad_clip_max_norm),
            ))

            # compute post-clip norm (L2)
            gn_post = _grad_norm_l2(model)
            gn_post_text = f" | gn_post={float(gn_post):.2e}"
        else:
            # raw L2 grad norm without clipping
            gn = _grad_norm_l2(model)
            gn_post_text = ""

        # Apply the warm-up learning rate before the parameter update so that
        # the first optimizer step starts at lr_init / warmup_len instead of
        # using the full initial learning rate.
        if step <= warmup_len:
            scale = step / float(warmup_len)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_init * scale

        # Record the learning rate actually used for this parameter update.
        lr_used = optimizer.param_groups[0]["lr"]
        optimizer.step()

        # After warm-up, decay the learning rate for subsequent updates using
        # cosine annealing down to cosine_min_lr.
        if step > warmup_len:
            cosine_scheduler.step()

        loss_tot_hist.append(loss_total.item())
        loss_cde_hist.append(loss_cde.item())
        loss_bc_hist.append(loss_bc.item())
        lambda1_history.append(lam1)
        lambda2_history.append(lam2)

        if step % print_every == 0 or step == 1 or step == total_epochs:
            lr = lr_used
            tot_val = float(loss_total.item())
            cde_val = float(loss_cde.item())
            bc_val = float(loss_bc.item())
            bc_weighted_val = float(lam2) * bc_val
            if abs(tot_val) > 1e-30:
                bc_over_tot_pct = 100.0 * bc_weighted_val / tot_val
            else:
                bc_over_tot_pct = float("nan")
            msg = (
                f"[{step:04d}] "
                f"tot={tot_val:.3e} | "
                f"cde={cde_val:.3e} | "
                f"bc={bc_val:.3e} | "
                f"lambda1={lam1:.6g}, lambda2={lam2:.6g} | "
                f"lam2*bc/tot={bc_over_tot_pct:.2f}% | "
                f"lr={lr:.2e} | "
                f"gn={float(gn):.2e}"
                f"{gn_post_text}"
            )
            if step == 1:
                nb = int(x_b_tensor.shape[0])
                msg = (
                    f"Nc={int(Nc.item())}, Nb={nb}, "
                    f"lambda1={float(lam1):g}, lambda2={float(lam2):g}\n"
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
        f"lambda1={float(lam1):g}, lambda2={float(lam2):g}, "
        f"lam2*bc/tot={final_bc_tot_pct:.2f}%"
    )
    _emit(f" ------ {phase_name} fixed-eps training complete ------\n")

    return (
        model,
        loss_tot_hist,
        loss_cde_hist,
        loss_bc_hist,
        {
            "loss_weighting": loss_weighting,
            "loss_weight_update_every": loss_weight_update_every,
            "loss_weight_ema": loss_weight_ema,
            "loss_weight_updates": weight_updates,
            "lambda1_history": lambda1_history,
            "lambda2_history": lambda2_history,
            "lambda1_initial": float(lam1_initial),
            "lambda2_initial": float(lam2_initial),
            "lambda1_final": float(lam1),
            "lambda2_final": float(lam2),
            "loss_weight_final_bc_tot_pct": float(final_bc_tot_pct),
        },
    )
