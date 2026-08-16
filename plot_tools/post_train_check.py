import torch

def post_train_check(
    *,
    model,
    x_coll,
    x_b_tensor,
    bc_target,
    cy_val,
    eps_global,
    compute_function_target_from_xcoll,
    precomputed_true=None,
    phase_name="P0",
    pred_scale=1.0,
    log_fn=None,
    output_part="re",
):
    """
    Post-train diagnostics for fixed-eps single-model training.
    """
    def _normalize_output_part(value, default="re"):
        if value is None:
            value = default
        s = str(value).strip().lower()
        if s in {"re", "real"}:
            return "re"
        if s in {"im", "imag", "imaginary"}:
            return "im"
        raise ValueError(f"Unsupported output part: {value!r}. Expected one of Re or Im.")

    def _emit(msg: str):
        print(msg)
        if log_fn is not None:
            log_fn(msg)

    def _to_metric_dtype(x: torch.Tensor) -> torch.Tensor:
        return x.to(dtype=torch.float64)

    def _quantile_max_summary(vals: torch.Tensor) -> str:
        flat = vals.reshape(-1)
        finite = flat[torch.isfinite(flat)]
        if finite.numel() == 0:
            return "median, p90, p99, p99.9 = None, max= None"

        q = torch.tensor([0.5, 0.9, 0.99, 0.999], device=finite.device, dtype=finite.dtype)
        qv = torch.quantile(finite, q)
        vmax = finite.max()
        return (
            f"median, p90, p99, p99.9= "
            f"{qv[0].item():.6e}, {qv[1].item():.6e}, {qv[2].item():.6e}, {qv[3].item():.6e}; "
            f"max= {vmax.item():.6e}"
        )

    def _emit_component_quantiles(tag: str, true_part: torch.Tensor, pred_part: torch.Tensor, part_label: str):
        if true_part.ndim != 2 or pred_part.ndim != 2 or true_part.shape != pred_part.shape:
            _emit(f"\n[{phase_name} | {tag} {part_label} quantiles]")
            _emit(f"skip: incompatible {part_label} tensor shapes")
            return

        _emit(f"\n[{phase_name} | {tag} {part_label} quantiles]")
        n_basis_local = true_part.shape[1]
        for j in range(n_basis_local):
            _emit(f"I{j + 1} {part_label} true  {_quantile_max_summary(true_part[:, j])}")
            _emit(f"I{j + 1} {part_label} pred  {_quantile_max_summary(pred_part[:, j])}")
        if str(tag).strip().upper() == "BC":
            _emit("")

    if precomputed_true is None:
        true = compute_function_target_from_xcoll(
            x_coll,
            cy_val=cy_val,
            eps_val=eps_global,
        )
    else:
        true = precomputed_true.to(device=x_coll.device)
    true = _to_metric_dtype(true)

    model.eval()
    with torch.no_grad():
        pred = _to_metric_dtype(model(x_coll) / float(pred_scale))

    output_part = _normalize_output_part(output_part)

    if output_part in {"re", "im"}:
        part_label = "Re" if output_part == "re" else "Im"

        if pred.ndim != 2 or true.ndim != 2 or pred.shape != true.shape:
            _emit(f"\n[{phase_name} | collocation {part_label} check]")
            _emit("skip: incompatible tensor shapes")
            return

        diff = pred - true
        true_mean = true.abs().mean()
        pred_mean = pred.abs().mean()
        abs_mean_err = diff.abs().mean()
        rmse = torch.sqrt(diff.pow(2).mean())
        scale_ratio = pred_mean / (true_mean + 1e-15)
        diff_percent = ((true_mean - pred_mean).abs() / (true_mean + 1e-15) * 100)

        abs_l1 = diff.abs().sum()
        rel_l1 = abs_l1 / (true.abs().sum() + 1e-12)
        abs_l2 = diff.norm()
        rel_l2 = abs_l2 / (true.norm() + 1e-12)

        _emit(f"\n[{phase_name} | collocation {part_label} check]")
        _emit(f"{part_label} true.mean abs: {true_mean.item()}")
        _emit(f"{part_label} pred.mean abs: {pred_mean.item()}")
        _emit(f"{part_label} pred true diff: {diff_percent.item():.2f}%")
        _emit(f"{part_label} diff MAE: {abs_mean_err.item()}")
        _emit(f"{part_label} diff RMSE: {rmse.item()}")
        _emit(f"{part_label} diff absL1: {abs_l1.item()}")
        _emit(f"{part_label} diff relL1: {rel_l1.item()}")
        _emit(f"{part_label} diff absL2: {abs_l2.item()}")
        _emit(f"{part_label} diff relL2: {rel_l2.item()}")
        _emit(f"{part_label} pred/true scale ratio: {scale_ratio.item()}")
        _emit_component_quantiles("collocation", true, pred, part_label)
        if true_mean.item() < 1e-5:
            _emit("note: true.mean abs is tiny; relative % can be inflated by denominator scale.")

        with torch.no_grad():
            pred_b = _to_metric_dtype(model(x_b_tensor) / float(pred_scale))
            true_b = _to_metric_dtype(bc_target)
            if pred_b.ndim != 2 or true_b.ndim != 2 or pred_b.shape != true_b.shape:
                _emit(f"\n[{phase_name} | BC {part_label} check]")
                _emit("skip: incompatible BC tensor shapes")
                return
            diff_b = pred_b - true_b

        abs_l1_b = diff_b.abs().sum()
        rel_l1_b = abs_l1_b / (true_b.abs().sum() + 1e-12)
        abs_l2_b = diff_b.norm()
        rel_l2_b = abs_l2_b / (true_b.norm() + 1e-12)
        mse_b = diff_b.pow(2).mean()
        rmse_b = torch.sqrt(mse_b)
        abs_mean_b = diff_b.abs().mean()
        bc_true_mean = true_b.abs().mean()
        bc_pred_mean = pred_b.abs().mean()
        bc_scale_ratio = bc_pred_mean / (bc_true_mean + 1e-15)
        bc_diff_percent = ((bc_true_mean - bc_pred_mean).abs() / (bc_true_mean + 1e-15) * 100)

        _emit(f"\n[{phase_name} | BC {part_label} check]")
        _emit(f"BC {part_label} true.mean abs: {bc_true_mean.item()}")
        _emit(f"BC {part_label} pred.mean abs: {bc_pred_mean.item()}")
        _emit(f"BC {part_label} pred true diff: {bc_diff_percent.item():.2f}%")
        _emit(f"BC {part_label} diff MAE: {abs_mean_b.item()}")
        _emit(f"BC {part_label} diff RMSE: {rmse_b.item()}")
        _emit(f"BC {part_label} diff absL1: {abs_l1_b.item()}")
        _emit(f"BC {part_label} diff relL1: {rel_l1_b.item()}")
        _emit(f"BC {part_label} diff absL2: {abs_l2_b.item()}")
        _emit(f"BC {part_label} diff relL2: {rel_l2_b.item()}")
        _emit(f"BC {part_label} pred/true scale ratio: {bc_scale_ratio.item()}")
        _emit_component_quantiles("BC", true_b, pred_b, part_label)
        if bc_true_mean.item() < 1e-5:
            _emit("note: BC true.mean abs is tiny; relative metrics can look extreme.")
        return

    diff = pred - true
    true_mean = true.abs().mean()
    pred_mean = pred.abs().mean()
    abs_mean_err = diff.abs().mean()
    rmse = torch.sqrt(diff.pow(2).mean())
    scale_ratio = pred_mean / (true_mean + 1e-15)
    diff_percent = ((true_mean - pred_mean).abs() / (true_mean + 1e-15) * 100)

    abs_l1 = diff.abs().sum()
    rel_l1 = abs_l1 / (true.abs().sum() + 1e-12)

    abs_l2 = diff.norm()
    rel_l2 = abs_l2 / (true.norm() + 1e-12)

    _emit(f"\n[{phase_name} | collocation scale check]")
    _emit(f"true.mean abs: {true_mean.item()}")
    _emit(f"pred.mean abs: {pred_mean.item()}")
    _emit(f"pred true diff: {diff_percent.item():.2f}%")
    _emit(f"diff MAE: {abs_mean_err.item()}")
    _emit(f"diff RMSE: {rmse.item()}")
    _emit(f"diff absL1 (Re+Im): {abs_l1.item()}")
    _emit(f"diff relL1 (Re+Im): {rel_l1.item()}")
    _emit(f"diff absL2 (Re+Im): {abs_l2.item()}")
    _emit(f"diff relL2 (Re+Im): {rel_l2.item()}")
    _emit(f"pred/true scale ratio: {scale_ratio.item()}")
    if pred.ndim == 2 and true.ndim == 2 and pred.shape == true.shape and pred.shape[1] % 2 == 0:
        n_basis = pred.shape[1] // 2
        pred_re, pred_im = pred[:, :n_basis], pred[:, n_basis:]
        true_re, true_im = true[:, :n_basis], true[:, n_basis:]
        diff_re = pred_re - true_re
        diff_im = pred_im - true_im

        true_re_abs_sum = true_re.abs().sum()
        true_im_abs_sum = true_im.abs().sum()
        abs_l1_re = diff_re.abs().sum()
        rel_l1_re = abs_l1_re / (true_re_abs_sum + 1e-12)
        abs_l1_im = diff_im.abs().sum()
        rel_l1_im_2 = abs_l1_im / (true_re_abs_sum + 1e-12)

        abs_l2_re = diff_re.norm()
        rel_l2_re = abs_l2_re / (true_re.norm() + 1e-12)
        abs_l2_im = diff_im.norm()
        rel_l2_im_2 = abs_l2_im / (true_re.norm() + 1e-12)

        _emit(f"Re absL1: {abs_l1_re.item()}")
        _emit(f"Re relL1: {rel_l1_re.item()}")
        _emit(f"Im absL1: {abs_l1_im.item()}")
        if true_im_abs_sum.item() < 1e-12:
            _emit("Im relL1: None")
        else:
            rel_l1_im = abs_l1_im / true_im_abs_sum
            _emit(f"Im relL1: {rel_l1_im.item()}")
        _emit(f"Im relL1 (vs Re true): {rel_l1_im_2.item()}")
        _emit(f"Re absL2: {abs_l2_re.item()}")
        _emit(f"Re relL2: {rel_l2_re.item()}")
        _emit(f"Im absL2: {abs_l2_im.item()}")

        true_im_norm = true_im.norm()
        if true_im_norm.item() < 1e-12:
            _emit("Im relL2: None")
        else:
            rel_l2_im = abs_l2_im / true_im_norm
            _emit(f"Im relL2: {rel_l2_im.item()}")
        _emit(f"Im relL2 (vs Re true): {rel_l2_im_2.item()}")
        _emit_component_quantiles("collocation", true_im, pred_im, "Im")
    if true_mean.item() < 1e-5:
        _emit("note: true.mean abs is tiny; relative % can be inflated by denominator scale.")

    with torch.no_grad():
        pred_b = _to_metric_dtype(model(x_b_tensor) / float(pred_scale))
        true_b = _to_metric_dtype(bc_target)
        diff_b = pred_b - true_b

    abs_l1_b = diff_b.abs().sum()
    rel_l1_b = abs_l1_b / (true_b.abs().sum() + 1e-12)
    abs_l2_b = diff_b.norm()
    rel_l2_b = abs_l2_b / (true_b.norm() + 1e-12)
    mse_b = diff_b.pow(2).mean()
    rmse_b = torch.sqrt(mse_b)
    abs_mean_b = diff_b.abs().mean()
    bc_true_mean = true_b.abs().mean()
    bc_pred_mean = pred_b.abs().mean()
    bc_scale_ratio = bc_pred_mean / (bc_true_mean + 1e-15)
    bc_diff_percent = ((bc_true_mean - bc_pred_mean).abs() / (bc_true_mean + 1e-15) * 100)

    _emit(f"\n[{phase_name} | BC check]")
    _emit(f"BC true.mean abs: {bc_true_mean.item()}")
    _emit(f"BC pred.mean abs: {bc_pred_mean.item()}")
    _emit(f"BC pred true diff: {bc_diff_percent.item():.2f}%")
    _emit(f"BC diff MAE: {abs_mean_b.item()}")
    _emit(f"BC diff RMSE: {rmse_b.item()}")
    _emit(f"BC diff absL1 (Re+Im): {abs_l1_b.item()}")
    _emit(f"BC diff relL1 (Re+Im): {rel_l1_b.item()}")
    _emit(f"BC diff absL2 (Re+Im): {abs_l2_b.item()}")
    _emit(f"BC diff relL2 (Re+Im): {rel_l2_b.item()}")
    _emit(f"BC pred/true scale ratio: {bc_scale_ratio.item()}")
    if pred_b.ndim == 2 and true_b.ndim == 2 and pred_b.shape == true_b.shape and pred_b.shape[1] % 2 == 0:
        n_basis_b = pred_b.shape[1] // 2
        pred_re_b, pred_im_b = pred_b[:, :n_basis_b], pred_b[:, n_basis_b:]
        true_re_b, true_im_b = true_b[:, :n_basis_b], true_b[:, n_basis_b:]
        diff_re_b = pred_re_b - true_re_b
        diff_im_b = pred_im_b - true_im_b

        true_re_b_abs_sum = true_re_b.abs().sum()
        true_im_b_abs_sum = true_im_b.abs().sum()
        abs_l1_re_b = diff_re_b.abs().sum()
        rel_l1_re_b = abs_l1_re_b / (true_re_b_abs_sum + 1e-12)
        abs_l1_im_b = diff_im_b.abs().sum()
        rel_l1_im_2_b = abs_l1_im_b / (true_re_b_abs_sum + 1e-12)

        abs_l2_re_b = diff_re_b.norm()
        rel_l2_re_b = abs_l2_re_b / (true_re_b.norm() + 1e-12)
        abs_l2_im_b = diff_im_b.norm()
        rel_l2_im_2_b = abs_l2_im_b / (true_re_b.norm() + 1e-12)

        _emit(f"BC Re absL1: {abs_l1_re_b.item()}")
        _emit(f"BC Re relL1: {rel_l1_re_b.item()}")
        _emit(f"BC Im absL1: {abs_l1_im_b.item()}")
        if true_im_b_abs_sum.item() < 1e-12:
            _emit("BC Im relL1: None")
        else:
            rel_l1_im_b = abs_l1_im_b / true_im_b_abs_sum
            _emit(f"BC Im relL1: {rel_l1_im_b.item()}")
        _emit(f"BC Im relL1 (vs Re true): {rel_l1_im_2_b.item()}")
        _emit(f"BC Re absL2: {abs_l2_re_b.item()}")
        _emit(f"BC Re relL2: {rel_l2_re_b.item()}")
        _emit(f"BC Im absL2: {abs_l2_im_b.item()}")

        true_im_norm_b = true_im_b.norm()
        if true_im_norm_b.item() < 1e-12:
            _emit("BC Im relL2: None")
        else:
            rel_l2_im_b = abs_l2_im_b / true_im_norm_b
            _emit(f"BC Im relL2: {rel_l2_im_b.item()}")
        _emit(f"BC Im relL2 (vs Re true): {rel_l2_im_2_b.item()}")
        _emit_component_quantiles("BC", true_im_b, pred_im_b, "Im")
    #_emit(f"BC MSE: {mse_b.item()}\n")
    if bc_true_mean.item() < 1e-5:
        _emit("note: BC true.mean abs is tiny; relative metrics can look extreme.")
