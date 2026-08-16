import torch
import torch.nn as nn

# -------------------------------------------------
#  Compute gradients for complex outputs
# -------------------------------------------------
def _normalize_output_part(value, default="both"):
    if value is None:
        value = default
    s = str(value).strip().lower()
    if s in {"both", "all", "reim", "complex"}:
        return "both"
    if s in {"re", "real"}:
        return "re"
    if s in {"im", "imag", "imaginary"}:
        return "im"
    raise ValueError(f"Unsupported output part: {value!r}. Expected one of Re/Im/Both.")


def _split_model_outputs(outputs: torch.Tensor, n_basis: int, output_part="both"):
    part = _normalize_output_part(output_part)
    if outputs.ndim != 2:
        raise ValueError(f"model outputs must be 2D, got shape {tuple(outputs.shape)}")

    if part == "both":
        expected = int(2 * n_basis)
        if outputs.shape[1] != expected:
            raise ValueError(
                f"Expected model output dim={expected} for output_part='both', got {tuple(outputs.shape)}"
            )
        return outputs[:, :n_basis], outputs[:, n_basis:]

    expected = int(n_basis)
    if outputs.shape[1] != expected:
        raise ValueError(
            f"Expected model output dim={expected} for output_part='{part}', got {tuple(outputs.shape)}"
        )
    if part == "re":
        return outputs, None
    return None, outputs


def _compute_channel_grads(channel_out: torch.Tensor, x_req: torch.Tensor, n_coords: int):
    grads_list = []
    n_basis_local = int(channel_out.shape[1])
    for idx in range(n_basis_local):
        g = torch.autograd.grad(channel_out[:, idx].sum(), x_req, create_graph=True)[0]
        grads_list.append(g[:, :n_coords].unsqueeze(2))
    return torch.cat(grads_list, dim=2)


def _compute_gradients_complex_coords(
    model: nn.Module,
    x_input: torch.Tensor,
    n_basis: int,
    n_coords: int,
    output_part="both",
):
    x_req = x_input.clone().detach().requires_grad_(True)

    outputs = model(x_req)
    re_out, im_out = _split_model_outputs(outputs, n_basis, output_part=output_part)

    grads_re = _compute_channel_grads(re_out, x_req, n_coords) if re_out is not None else None
    grads_im = _compute_channel_grads(im_out, x_req, n_coords) if im_out is not None else None

    return re_out, im_out, grads_re, grads_im


def compute_gradients_complex(model: nn.Module, x_input: torch.Tensor, n_basis: int):
    re_out, im_out, grads_re, grads_im = _compute_gradients_complex_coords(
        model=model,
        x_input=x_input,
        n_basis=n_basis,
        n_coords=2,
    )

    d_re_dx1 = grads_re[:, 0, :]
    d_re_dx2 = grads_re[:, 1, :]
    d_im_dx1 = grads_im[:, 0, :]
    d_im_dx2 = grads_im[:, 1, :]

    return re_out, im_out, d_re_dx1, d_re_dx2, d_im_dx1, d_im_dx2


def compute_gradients_complex_1loop(model: nn.Module, x_input: torch.Tensor, n_basis: int):
    re_out, im_out, grads_re, grads_im = _compute_gradients_complex_coords(
        model=model,
        x_input=x_input,
        n_basis=n_basis,
        n_coords=3,
    )

    d_re_dx1 = grads_re[:, 0, :]
    d_re_dx2 = grads_re[:, 1, :]
    d_re_dy1 = grads_re[:, 2, :]
    d_im_dx1 = grads_im[:, 0, :]
    d_im_dx2 = grads_im[:, 1, :]
    d_im_dy1 = grads_im[:, 2, :]

    return re_out, im_out, d_re_dx1, d_re_dx2, d_re_dy1, d_im_dx1, d_im_dx2, d_im_dy1


def _batched_complex_matmul(a_mat, re, im):
    out_re = torch.einsum("bij,bj->bi", a_mat, re)
    out_im = torch.einsum("bij,bj->bi", a_mat, im)
    return out_re, out_im


def _batched_matmul(a_mat, vec):
    return torch.einsum("bij,bj->bi", a_mat, vec)


def _mean_sq(x):
    return x.pow(2).mean()


def cde_residual_loss_fixed_eps(
    model,
    a_builder,
    x_batch,
    n_basis,
    eps_val,
    eps0_tol=1e-12,
    output_part="both",
):
    """
    CDE loss for fixed-eps training.
    x_batch is (N,2) = (x1,x2), eps is a global scalar.
    """
    if x_batch.shape[1] != 2:
        raise ValueError(
            f"x_batch must have 2 columns (x1,x2) for fixed-eps training, got {tuple(x_batch.shape)}"
        )

    output_part = _normalize_output_part(output_part)
    re_out, im_out, grads_re, grads_im = _compute_gradients_complex_coords(
        model=model,
        x_input=x_batch,
        n_basis=n_basis,
        n_coords=2,
        output_part=output_part,
    )
    d_re_dx1 = grads_re[:, 0, :] if grads_re is not None else None
    d_re_dx2 = grads_re[:, 1, :] if grads_re is not None else None
    d_im_dx1 = grads_im[:, 0, :] if grads_im is not None else None
    d_im_dx2 = grads_im[:, 1, :] if grads_im is not None else None

    a1, a2, a1_hat, a2_hat = a_builder(x_batch[:, :2])

    terms = []
    eps_scalar = float(eps_val)
    if abs(eps_scalar) < eps0_tol:
        if re_out is not None:
            rhs1_re = _batched_matmul(a1_hat, re_out)
            rhs2_re = _batched_matmul(a2_hat, re_out)
            res1_re = d_re_dx1 - rhs1_re
            res2_re = d_re_dx2 - rhs2_re
            terms.append(_mean_sq(res1_re))
            terms.append(_mean_sq(res2_re))
        if im_out is not None:
            rhs1_im = _batched_matmul(a1_hat, im_out)
            rhs2_im = _batched_matmul(a2_hat, im_out)
            res1_im = d_im_dx1 - rhs1_im
            res2_im = d_im_dx2 - rhs2_im
            terms.append(_mean_sq(res1_im))
            terms.append(_mean_sq(res2_im))
    else:
        eps_t = x_batch.new_tensor(eps_scalar)
        if re_out is not None:
            rhs1_re = _batched_matmul(a1, re_out)
            rhs2_re = _batched_matmul(a2, re_out)
            res1_re = d_re_dx1 - eps_t * rhs1_re
            res2_re = d_re_dx2 - eps_t * rhs2_re
            terms.append(_mean_sq(res1_re))
            terms.append(_mean_sq(res2_re))
        if im_out is not None:
            rhs1_im = _batched_matmul(a1, im_out)
            rhs2_im = _batched_matmul(a2, im_out)
            res1_im = d_im_dx1 - eps_t * rhs1_im
            res2_im = d_im_dx2 - eps_t * rhs2_im
            terms.append(_mean_sq(res1_im))
            terms.append(_mean_sq(res2_im))

    if not terms:
        raise RuntimeError("No residual terms computed. Check output_part and model output shape.")
    # Average the per-coordinate CDE residuals so the loss scale is
    # independent of the number of input variables (x1, x2, ...).
    loss = 0.5 * torch.stack(terms).sum() / x_batch.shape[1]
    Nc = x_batch.new_tensor(float(x_batch.shape[0]))
    return loss, Nc


def cde_residual_loss_fixed_eps_1loop(
    model,
    a_builder,
    x_batch,
    n_basis,
    eps_val,
    eps0_tol=1e-12,
    output_part="both",
):
    """
    CDE loss for fixed-eps 1-loop training.
    x_batch is (N,3) = (x1,x2,y1), eps is a global scalar.
    """
    if x_batch.shape[1] != 3:
        raise ValueError(
            f"x_batch must have 3 columns (x1,x2,y1) for 1-loop fixed-eps training, got {tuple(x_batch.shape)}"
        )

    output_part = _normalize_output_part(output_part)
    re_out, im_out, grads_re, grads_im = _compute_gradients_complex_coords(
        model=model,
        x_input=x_batch,
        n_basis=n_basis,
        n_coords=3,
        output_part=output_part,
    )
    d_re_dx1 = grads_re[:, 0, :] if grads_re is not None else None
    d_re_dx2 = grads_re[:, 1, :] if grads_re is not None else None
    d_re_dy1 = grads_re[:, 2, :] if grads_re is not None else None
    d_im_dx1 = grads_im[:, 0, :] if grads_im is not None else None
    d_im_dx2 = grads_im[:, 1, :] if grads_im is not None else None
    d_im_dy1 = grads_im[:, 2, :] if grads_im is not None else None

    mats = a_builder(x_batch[:, :3])
    if len(mats) != 6:
        raise ValueError(
            f"a_builder for 1-loop fixed-eps training must return 6 tensors "
            f"(a_x1,a_x2,a_y1,a0_x1,a0_x2,a0_y1), got {len(mats)}."
        )
    a1, a2, a3, a1_hat, a2_hat, a3_hat = mats

    terms = []
    eps_scalar = float(eps_val)
    if abs(eps_scalar) < eps0_tol:
        if re_out is not None:
            rhs1_re = _batched_matmul(a1_hat, re_out)
            rhs2_re = _batched_matmul(a2_hat, re_out)
            rhs3_re = _batched_matmul(a3_hat, re_out)
            res1_re = d_re_dx1 - rhs1_re
            res2_re = d_re_dx2 - rhs2_re
            res3_re = d_re_dy1 - rhs3_re
            terms.extend([_mean_sq(res1_re), _mean_sq(res2_re), _mean_sq(res3_re)])
        if im_out is not None:
            rhs1_im = _batched_matmul(a1_hat, im_out)
            rhs2_im = _batched_matmul(a2_hat, im_out)
            rhs3_im = _batched_matmul(a3_hat, im_out)
            res1_im = d_im_dx1 - rhs1_im
            res2_im = d_im_dx2 - rhs2_im
            res3_im = d_im_dy1 - rhs3_im
            terms.extend([_mean_sq(res1_im), _mean_sq(res2_im), _mean_sq(res3_im)])
    else:
        eps_t = x_batch.new_tensor(eps_scalar)
        if re_out is not None:
            rhs1_re = _batched_matmul(a1, re_out)
            rhs2_re = _batched_matmul(a2, re_out)
            rhs3_re = _batched_matmul(a3, re_out)
            res1_re = d_re_dx1 - eps_t * rhs1_re
            res2_re = d_re_dx2 - eps_t * rhs2_re
            res3_re = d_re_dy1 - eps_t * rhs3_re
            terms.extend([_mean_sq(res1_re), _mean_sq(res2_re), _mean_sq(res3_re)])
        if im_out is not None:
            rhs1_im = _batched_matmul(a1, im_out)
            rhs2_im = _batched_matmul(a2, im_out)
            rhs3_im = _batched_matmul(a3, im_out)
            res1_im = d_im_dx1 - eps_t * rhs1_im
            res2_im = d_im_dx2 - eps_t * rhs2_im
            res3_im = d_im_dy1 - eps_t * rhs3_im
            terms.extend([_mean_sq(res1_im), _mean_sq(res2_im), _mean_sq(res3_im)])

    if not terms:
        raise RuntimeError("No residual terms computed. Check output_part and model output shape.")
    # Average the per-coordinate CDE residuals so the loss scale is
    # independent of the number of input variables (x1, x2, ...).
    loss = 0.5 * torch.stack(terms).sum() / x_batch.shape[1]
    Nc = x_batch.new_tensor(float(x_batch.shape[0]))
    return loss, Nc


def cde_residual_loss_fixed_eps_2loop(
    model,
    a_builder,
    x_batch,
    n_basis,
    eps_val,
    eps0_tol=1e-12,
    output_part="both",
):
    """
    CDE loss for fixed-eps 2-loop sunset training.
    x_batch is (N,4) = (x1,x2,y1,y2), eps is a global scalar.
    """
    if x_batch.shape[1] != 4:
        raise ValueError(
            f"x_batch must have 4 columns (x1,x2,y1,y2) for 2-loop fixed-eps training, got {tuple(x_batch.shape)}"
        )

    output_part = _normalize_output_part(output_part)
    re_out, im_out, grads_re, grads_im = _compute_gradients_complex_coords(
        model=model,
        x_input=x_batch,
        n_basis=n_basis,
        n_coords=4,
        output_part=output_part,
    )
    d_re_dx1 = grads_re[:, 0, :] if grads_re is not None else None
    d_re_dx2 = grads_re[:, 1, :] if grads_re is not None else None
    d_re_dy1 = grads_re[:, 2, :] if grads_re is not None else None
    d_re_dy2 = grads_re[:, 3, :] if grads_re is not None else None
    d_im_dx1 = grads_im[:, 0, :] if grads_im is not None else None
    d_im_dx2 = grads_im[:, 1, :] if grads_im is not None else None
    d_im_dy1 = grads_im[:, 2, :] if grads_im is not None else None
    d_im_dy2 = grads_im[:, 3, :] if grads_im is not None else None

    mats = a_builder(x_batch[:, :4])
    if len(mats) != 8:
        raise ValueError(
            "a_builder for 2-loop fixed-eps training must return 8 tensors "
            "(a_x1,a_x2,a_y1,a_y2,a0_x1,a0_x2,a0_y1,a0_y2), "
            f"got {len(mats)}."
        )
    a1, a2, a3, a4, a1_hat, a2_hat, a3_hat, a4_hat = mats

    terms = []
    eps_scalar = float(eps_val)
    if abs(eps_scalar) < eps0_tol:
        if re_out is not None:
            rhs1_re = _batched_matmul(a1_hat, re_out)
            rhs2_re = _batched_matmul(a2_hat, re_out)
            rhs3_re = _batched_matmul(a3_hat, re_out)
            rhs4_re = _batched_matmul(a4_hat, re_out)
            res1_re = d_re_dx1 - rhs1_re
            res2_re = d_re_dx2 - rhs2_re
            res3_re = d_re_dy1 - rhs3_re
            res4_re = d_re_dy2 - rhs4_re
            terms.extend(
                [_mean_sq(res1_re), _mean_sq(res2_re), _mean_sq(res3_re), _mean_sq(res4_re)]
            )
        if im_out is not None:
            rhs1_im = _batched_matmul(a1_hat, im_out)
            rhs2_im = _batched_matmul(a2_hat, im_out)
            rhs3_im = _batched_matmul(a3_hat, im_out)
            rhs4_im = _batched_matmul(a4_hat, im_out)
            res1_im = d_im_dx1 - rhs1_im
            res2_im = d_im_dx2 - rhs2_im
            res3_im = d_im_dy1 - rhs3_im
            res4_im = d_im_dy2 - rhs4_im
            terms.extend(
                [_mean_sq(res1_im), _mean_sq(res2_im), _mean_sq(res3_im), _mean_sq(res4_im)]
            )
    else:
        eps_t = x_batch.new_tensor(eps_scalar)
        if re_out is not None:
            rhs1_re = _batched_matmul(a1, re_out)
            rhs2_re = _batched_matmul(a2, re_out)
            rhs3_re = _batched_matmul(a3, re_out)
            rhs4_re = _batched_matmul(a4, re_out)
            res1_re = d_re_dx1 - eps_t * rhs1_re
            res2_re = d_re_dx2 - eps_t * rhs2_re
            res3_re = d_re_dy1 - eps_t * rhs3_re
            res4_re = d_re_dy2 - eps_t * rhs4_re
            terms.extend(
                [_mean_sq(res1_re), _mean_sq(res2_re), _mean_sq(res3_re), _mean_sq(res4_re)]
            )
        if im_out is not None:
            rhs1_im = _batched_matmul(a1, im_out)
            rhs2_im = _batched_matmul(a2, im_out)
            rhs3_im = _batched_matmul(a3, im_out)
            rhs4_im = _batched_matmul(a4, im_out)
            res1_im = d_im_dx1 - eps_t * rhs1_im
            res2_im = d_im_dx2 - eps_t * rhs2_im
            res3_im = d_im_dy1 - eps_t * rhs3_im
            res4_im = d_im_dy2 - eps_t * rhs4_im
            terms.extend(
                [_mean_sq(res1_im), _mean_sq(res2_im), _mean_sq(res3_im), _mean_sq(res4_im)]
            )

    if not terms:
        raise RuntimeError("No residual terms computed. Check output_part and model output shape.")
    # Average the per-coordinate CDE residuals so the loss scale is
    # independent of the number of input variables (x1, x2, ...).
    loss = 0.5 * torch.stack(terms).sum() / x_batch.shape[1]
    Nc = x_batch.new_tensor(float(x_batch.shape[0]))
    return loss, Nc

# --------------------------
#  Boundary loss
# --------------------------
def _build_bc_channel_scale(
    bc_target: torch.Tensor,
    *,
    scale_floor: float = 1e-4,
    min_scale_ratio: float = 1.0,
    output_part="both",
):
    """
    Build stable per-channel normalization scale for BC loss.
    For complex concatenated targets [Re..., Im...], Im-channel scale is
    lower-bounded by the corresponding Re-channel scale.
    """
    if bc_target.ndim != 2:
        raise ValueError(f"bc_target must be 2D, got shape {tuple(bc_target.shape)}")

    ch_rms = torch.sqrt(torch.mean(bc_target.pow(2), dim=0))
    global_rms = torch.sqrt(torch.mean(bc_target.pow(2)))

    output_part = _normalize_output_part(output_part)
    d = int(ch_rms.shape[0])
    if (output_part == "both") and (d % 2 == 0) and d > 0:
        n_basis = d // 2
        re_rms = ch_rms[:n_basis]
        im_rms = ch_rms[n_basis:]
        # In many fixed-eps settings Im targets can be near-zero; avoid over-weighting.
        im_rms = torch.maximum(im_rms, re_rms)
        ch_rms = torch.cat([re_rms, im_rms], dim=0)

    min_scale = torch.clamp(
        global_rms * float(min_scale_ratio),
        min=float(scale_floor),
    )
    scale = torch.clamp(ch_rms, min=min_scale)
    return scale.detach()


def boundary_loss(
    model,
    x_b_tensor,
    bc_target,
    *,
    use_normalized=True,
    scale_floor=1e-4,
    min_scale_ratio=1.0,
    abs_mse_weight=0.05,
    output_part="both",
):
    pred = model(x_b_tensor)
    diff = pred - bc_target
    abs_mse = diff.pow(2).mean()

    if not use_normalized:
        return abs_mse * 2.0

    bc_scale = _build_bc_channel_scale(
        bc_target,
        scale_floor=scale_floor,
        min_scale_ratio=min_scale_ratio,
        output_part=output_part,
    )
    norm_diff = diff / bc_scale.unsqueeze(0)
    norm_mse = norm_diff.pow(2).mean()

    w_abs = float(abs_mse_weight)
    if w_abs < 0.0 or w_abs > 1.0:
        raise ValueError(f"abs_mse_weight must be in [0,1], got {w_abs}")

    mixed = (1.0 - w_abs) * norm_mse + w_abs * abs_mse
    return mixed * 2.0
