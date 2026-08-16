import json
import os
import time
import pickle
import inspect
import numpy as np
import torch
import warnings

from lib.loss import (
    boundary_loss,
    cde_residual_loss_fixed_eps,
    cde_residual_loss_fixed_eps_1loop,
    cde_residual_loss_fixed_eps_2loop,
)
from lib.pinn_models import PinnModel, TransferPinnModel
from lib.train import train_model_fixed_eps
from plot_tools.plot_error import plot_error_dis
from plot_tools.plot_losses import (
    plot_losses,
    get_nested_save_dir,
    set_results_root_name,
    get_results_root_name,
)
from plot_tools.post_train_check import post_train_check
from two_site_chain.sol_chain import eps_to_n_pos_int
from two_site_chain.coll_bc import (
    build_inputs_and_boundary,
    compute_function_target_from_xcoll,
)
from tl_two_site_bubble.coll_bc_1loop import (
    build_inputs_and_boundary_1loop,
    compute_function_target_from_xcoll_1loop,
)
from tl_two_site_sunset.coll_bc_2loop import (
    build_inputs_and_boundary_2loop,
    compute_function_target_from_xcoll_2loop,
)


class Config:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        self.__dict__.update(data)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __repr__(self):
        return json.dumps(self.__dict__, indent=2)


def _classify_eps_global(eps, eps0_tol=1e-12, tol=1e-6):
    epsf = float(eps)
    half_map = {-0.5: 1, -1.5: 2, -2.5: 3, -3.5: 4}

    if abs(epsf) < eps0_tol:
        return "eps0", "I*_eps0", "a_hat (eps=0 special CDE)"

    n_pos_int = eps_to_n_pos_int(epsf)
    if n_pos_int is not None:
        return "pos-int", f"I*_eps_pos_int (n={n_pos_int})", "eps * A"

    n_int = int(round(-epsf))
    if n_int >= 1 and abs(epsf + n_int) < tol:
        return "neg-int", f"I*_eps_int (n={n_int})", "eps * A"

    for v, n in half_map.items():
        if abs(epsf - v) < tol:
            return "neg-half-int", f"I*_eps_half (n={n})", "eps * A"

    return "general-cont", "I*_eps (general)", "eps * A"


def _make_eps_tag(eps) -> str:
    epsf = float(eps)
    if abs(epsf) < 1e-12:
        return "0"
    mag = f"{abs(epsf):.6g}".replace(".", "_")
    return f"m{mag}" if epsf < 0 else f"p{mag}"


def _to_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _normalize_output_part(value, default="re"):
    if value is None:
        value = default
    s = str(value).strip().lower()
    if s in {"re", "real"}:
        return "re"
    if s in {"im", "imag", "imaginary"}:
        return "im"
    raise ValueError(f"Unsupported output part: {value!r}. Expected one of Re or Im.")


def _output_part_tag(output_part: str):
    part = _normalize_output_part(output_part)
    if part == "re":
        return "Re"
    if part == "im":
        return "Im"
    raise ValueError(f"Unsupported output part: {output_part!r}. Expected Re or Im.")


def _slice_phase0_target_by_part(
    tensor: torch.Tensor,
    *,
    n_basis: int,
    output_part: str,
    tensor_name: str,
):
    if tensor.ndim != 2:
        raise ValueError(f"{tensor_name} must be 2D, got shape {tuple(tensor.shape)}")

    part = _normalize_output_part(output_part)
    dim = int(tensor.shape[1])
    nb = int(n_basis)

    if dim == nb:
        return tensor
    if dim == 2 * nb:
        return tensor[:, :nb] if part == "re" else tensor[:, nb:]
    raise ValueError(
        f"{tensor_name} has incompatible dim={dim} for output_part='{part}' and n_basis={nb}"
    )

def _get_nested_save_dir_compat(
    save_dir: str,
    cy: float,
    *,
    phase: int,
    phase_tag: str = None,
    create_dir: bool = True,
):
    sig = inspect.signature(get_nested_save_dir)
    if "create_dir" in sig.parameters:
        return get_nested_save_dir(
            save_dir,
            cy,
            phase=phase,
            phase_tag=phase_tag,
            create_dir=create_dir,
        )

    abs_dir, short_dir = get_nested_save_dir(
        save_dir,
        cy,
        phase=phase,
        phase_tag=phase_tag,
    )
    if create_dir:
        os.makedirs(abs_dir, exist_ok=True)
    return abs_dir, short_dir


def _open_phase_log_writer(
    phase: int,
    cy: float,
    eps_global,
    phase_tag: str = None,
    output_part_tag: str = None,
    log_suffix: str = None,
):
    losses_dir_abs, losses_dir_short = _get_nested_save_dir_compat(
        "1_losses",
        cy,
        phase=phase,
        phase_tag=phase_tag,
        create_dir=True,
    )
    phase_root_abs = os.path.dirname(losses_dir_abs)
    phase_root_short = os.path.dirname(losses_dir_short)

    eps_tag = _make_eps_tag(eps_global)
    suffix_parts = []
    if output_part_tag is not None:
        s_part = str(output_part_tag).strip()
        if s_part:
            suffix_parts.append(s_part)
    if log_suffix is not None:
        s_log = str(log_suffix).strip()
        if s_log:
            suffix_parts.append(s_log)
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    log_name = f"P{phase}_train_log_eps_{eps_tag}{suffix}.txt"
    log_path_abs = os.path.join(phase_root_abs, log_name)
    log_path_short = os.path.join(phase_root_short, log_name)

    f = open(log_path_abs, "w", encoding="utf-8")

    def _write(msg: str):
        text = str(msg)
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")
        f.flush()

    return f, log_path_short, _write


def _auto_solution_scale_from_bc(
    bc_target: torch.Tensor,
    *,
    ref_mean_abs: float,
    max_scale: float,
    min_scale: float = 1.0,
):
    if int(bc_target.numel()) == 0:
        raise ValueError("auto solution scale received an empty bc_target tensor.")

    bc_mean_abs = float(torch.mean(torch.abs(bc_target)).item())
    if not np.isfinite(bc_mean_abs):
        raise ValueError(
            "auto solution scale received a non-finite BC mean abs. "
            "Check boundary target construction and filtering."
        )
    if bc_mean_abs <= 0.0:
        return float(min_scale), bc_mean_abs, 0.0, False, True

    raw_scale = float(ref_mean_abs) / bc_mean_abs
    if not np.isfinite(raw_scale):
        raise ValueError(
            "auto solution scale computed a non-finite raw scale. "
            "Check BC target magnitude and solution_scale settings."
        )
    used = max(float(min_scale), min(float(max_scale), raw_scale))
    capped = bool(raw_scale > float(max_scale))
    floored = bool(raw_scale < float(min_scale))
    return float(used), bc_mean_abs, float(raw_scale), capped, floored


def _format_elapsed(elapsed_sec: float) -> str:
    sec = max(float(elapsed_sec), 0.0)
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60.0
    if h > 0:
        return f"{h:d}h {m:02d}m {s:05.2f}s"
    return f"{m:d}m {s:05.2f}s"

def _resolve_postcalc_workers(value):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return 1
    if v == 0:
        cpu = os.cpu_count() or 1
        return max(cpu - 1, 1)
    return max(v, 1)


def _resolve_optional_path(value):
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return os.path.abspath(os.path.expanduser(s))


def _guess_history_path_from_model_path(model_path: str) -> str:
    if "_model_" in model_path:
        guessed = model_path.replace("_model_", "_loss_history_")
        if guessed.endswith(".pt"):
            guessed = guessed[:-3] + ".npz"
        return guessed
    base, ext = os.path.splitext(model_path)
    if ext == ".pt":
        return f"{base}_history.npz"
    return f"{model_path}_loss_history.npz"


def _phase_artifact_paths(
    phase: int,
    cy: float,
    eps_global,
    phase_tag: str = None,
    output_part_tag: str = None,
    create_dirs: bool = False,
):
    save_abs, save_short = _get_nested_save_dir_compat(
        "0_models",
        cy,
        phase=phase,
        phase_tag=phase_tag,
        create_dir=bool(create_dirs),
    )
    eps_tag = _make_eps_tag(eps_global)
    suffix = ""
    if output_part_tag is not None:
        s = str(output_part_tag).strip()
        if s:
            suffix = f"_{s}"
    model_name = f"P{phase}_model_eps_{eps_tag}{suffix}.pt"
    history_name = f"P{phase}_loss_history_eps_{eps_tag}{suffix}.npz"
    return {
        "model_abs": os.path.join(save_abs, model_name),
        "model_short": os.path.join(save_short, model_name),
        "history_abs": os.path.join(save_abs, history_name),
        "history_short": os.path.join(save_short, history_name),
    }


def _phase_eval_bundle_paths(
    phase: int,
    cy: float,
    eps_global,
    phase_tag: str = None,
    output_part_tag: str = None,
    create_dirs: bool = False,
):
    save_abs, save_short = _get_nested_save_dir_compat(
        "0_models",
        cy,
        phase=phase,
        phase_tag=phase_tag,
        create_dir=bool(create_dirs),
    )
    eps_tag = _make_eps_tag(eps_global)
    suffix = ""
    if output_part_tag is not None:
        s = str(output_part_tag).strip()
        if s:
            suffix = f"_{s}"
    bundle_name = f"P{phase}_eval_bundle_eps_{eps_tag}{suffix}.pt"
    return {
        "bundle_abs": os.path.join(save_abs, bundle_name),
        "bundle_short": os.path.join(save_short, bundle_name),
    }


def _torch_load_compat(path, *, map_location):
    """
    Prefer safe loading on new PyTorch versions, with compatibility fallback.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        # Older PyTorch without `weights_only` argument.
        return torch.load(path, map_location=map_location)
    except (RuntimeError, pickle.UnpicklingError, ValueError, EOFError):
        # Fallback for legacy trusted checkpoints containing non-weights objects.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="You are using `torch.load` with `weights_only=False`",
                category=FutureWarning,
            )
            return torch.load(path, map_location=map_location, weights_only=False)


def _load_model_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    ckpt = _torch_load_compat(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        meta = ckpt
    elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        state_dict = ckpt
        meta = {}
    else:
        raise ValueError(f"Unsupported checkpoint format at: {ckpt_path}")
    model.load_state_dict(state_dict)
    return meta


def _infer_phase0_in_dim_from_checkpoint(ckpt_path: str):
    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and all(torch.is_tensor(v) for v in ckpt.values()):
        state_dict = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format at: {ckpt_path}")

    w0 = state_dict.get("net.0.weight", None)
    if isinstance(w0, torch.Tensor) and w0.ndim == 2:
        in_dim = int(w0.shape[1])
    else:
        in_dim = None
        for k, v in state_dict.items():
            if torch.is_tensor(v) and v.ndim == 2 and (k.endswith("net.0.weight") or k.endswith("0.weight")):
                in_dim = int(v.shape[1])
                break
        if in_dim is None:
            raise ValueError(f"Cannot infer phase-0 input dim from checkpoint: {ckpt_path}")

    if in_dim not in (2, 3):
        raise ValueError(f"Unsupported inferred phase-0 input dim={in_dim} from checkpoint: {ckpt_path}")
    return in_dim


def _load_loss_history(history_path: str):
    if history_path is None or (not os.path.isfile(history_path)):
        return None
    with np.load(history_path) as arr:
        if ("total" not in arr) or ("cde" not in arr) or ("bc" not in arr):
            raise ValueError(f"Invalid loss-history file (required keys: total/cde/bc): {history_path}")
        return arr["total"].tolist(), arr["cde"].tolist(), arr["bc"].tolist()


def _load_eval_bundle(bundle_path: str, device: torch.device):
    obj = _torch_load_compat(bundle_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported eval-bundle format at: {bundle_path}")
    required = ("x_coll", "x_b_tensor", "bc_target")
    for k in required:
        if k not in obj:
            raise ValueError(f"Missing '{k}' in eval-bundle: {bundle_path}")
    x_coll = obj["x_coll"].to(device=device, dtype=torch.float32)
    x_b_tensor = obj["x_b_tensor"].to(device=device, dtype=torch.float32)
    bc_target = obj["bc_target"].to(device=device, dtype=torch.float32)
    meta = {k: v for k, v in obj.items() if k not in {"x_coll", "x_b_tensor", "bc_target"}}
    return x_coll, x_b_tensor, bc_target, meta


def _save_phase_artifacts(
    *,
    model: torch.nn.Module,
    hist_tot,
    hist_cde,
    hist_bc,
    phase: int,
    cy: float,
    eps_global,
    pred_scale: float,
    extra_meta=None,
    phase_tag: str = None,
    output_part_tag: str = None,
):
    paths = _phase_artifact_paths(
        phase=phase,
        cy=cy,
        eps_global=eps_global,
        phase_tag=phase_tag,
        output_part_tag=output_part_tag,
        create_dirs=True,
    )
    ckpt = {
        "phase": int(phase),
        "cy": float(cy),
        "eps_global": float(eps_global),
        "pred_scale": float(pred_scale),
        "model_state_dict": model.state_dict(),
    }
    if isinstance(extra_meta, dict):
        ckpt.update(extra_meta)
    torch.save(ckpt, paths["model_abs"])
    np.savez(
        paths["history_abs"],
        total=np.asarray(hist_tot, dtype=float),
        cde=np.asarray(hist_cde, dtype=float),
        bc=np.asarray(hist_bc, dtype=float),
    )
    return paths


def _save_eval_bundle(
    *,
    phase: int,
    cy: float,
    eps_global,
    x_coll: torch.Tensor,
    x_b_tensor: torch.Tensor,
    bc_target: torch.Tensor,
    pred_scale: float,
    extra_meta=None,
    phase_tag: str = None,
    output_part_tag: str = None,
):
    paths = _phase_eval_bundle_paths(
        phase=phase,
        cy=cy,
        eps_global=eps_global,
        phase_tag=phase_tag,
        output_part_tag=output_part_tag,
        create_dirs=True,
    )
    payload = {
        "phase": int(phase),
        "cy": float(cy),
        "eps_global": float(eps_global),
        "pred_scale": float(pred_scale),
        "x_coll": x_coll.detach().cpu().float(),
        "x_b_tensor": x_b_tensor.detach().cpu().float(),
        "bc_target": bc_target.detach().cpu().float(),
    }
    if isinstance(extra_meta, dict):
        payload.update(extra_meta)
    torch.save(payload, paths["bundle_abs"])
    return paths


def _save_config_snapshot(config_path: str, eps_global):
    cfg_path_abs = os.path.abspath(config_path)
    with open(cfg_path_abs, "r", encoding="utf-8-sig") as f:
        cfg_data = json.load(f)

    here = os.path.dirname(os.path.abspath(__file__))
    results_root_name = get_results_root_name()
    results_root = os.path.join(here, results_root_name)
    cfg_dir_abs = os.path.join(results_root, "configs")
    os.makedirs(cfg_dir_abs, exist_ok=True)

    eps_tag = _make_eps_tag(eps_global)
    cfg_name = f"config_used_eps_{eps_tag}.json"
    cfg_out_abs = os.path.join(cfg_dir_abs, cfg_name)
    with open(cfg_out_abs, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    cfg_out_short = os.path.join(results_root_name, "configs", cfg_name)
    return cfg_out_abs, cfg_out_short


def main():
    bootstrap_cfg_path = "config.json"
    if not os.path.isfile(bootstrap_cfg_path):
        raise FileNotFoundError(f"Config bootstrap file not found: {bootstrap_cfg_path}")

    with open(bootstrap_cfg_path, "r", encoding="utf-8-sig") as f:
        bootstrap_cfg = json.load(f)

    use_local_cfg = _to_bool(bootstrap_cfg.get("use_local_config", False), default=False)
    cfg_path = "config_local_test.json" if use_local_cfg else bootstrap_cfg_path
    if use_local_cfg and (not os.path.isfile(cfg_path)):
        raise FileNotFoundError(
            f"use_local_config=True but local config file not found: {cfg_path}"
        )

    with open(cfg_path, "r", encoding="utf-8-sig") as f:
        loaded_cfg_dict = json.load(f)
    print("Loaded configuration:")
    print(json.dumps(loaded_cfg_dict, indent=2, ensure_ascii=False))

    cfg = Config(cfg_path)
    device = torch.device(cfg.device)
    eps_tag = _make_eps_tag(cfg.eps_global)
    enable_phase1 = _to_bool(getattr(cfg, "enable_phase1", True), default=True)
    enable_phase2 = _to_bool(getattr(cfg, "enable_phase2", False), default=False)

    normalized_bc = _to_bool(getattr(cfg, "normalized_bc", True), default=True)
    train_two_phase_only = _to_bool(getattr(cfg, "train_two_phase_only", False), default=False)
    run_phase1_only = _to_bool(getattr(cfg, "run_phase1_only", False), default=False)
    run_phase2_only = _to_bool(getattr(cfg, "run_phase2_only", False), default=False)
    eps_pos_int_n = eps_to_n_pos_int(cfg.eps_global)
    if eps_pos_int_n is not None:
        if run_phase1_only or enable_phase1:
            print(
                f"[Warn] eps_global={cfg.eps_global} is a positive integer (n={eps_pos_int_n}); "
                "Phase1 analytic targets are not implemented yet. Disable Phase1 and keep Phase0/Phase2 available."
            )
        enable_phase1 = False
        run_phase1_only = False

    solution_scale_mode = str(getattr(cfg, "solution_scale_mode", "auto")).strip().lower()
    if solution_scale_mode not in {"auto", "manual"}:
        print(f"[Warn] Unknown solution_scale_mode={solution_scale_mode}, fallback to auto")
        solution_scale_mode = "auto"
    p0_solution_scale = float(getattr(cfg, "solution_scale_p0", 1.0))
    p1_solution_scale = float(getattr(cfg, "solution_scale_p1", p0_solution_scale))
    p2_solution_scale = float(getattr(cfg, "solution_scale_p2", p1_solution_scale))
    solution_scale_ref_mean = float(getattr(cfg, "solution_scale_ref_mean", 1e-1))
    solution_scale_max = float(getattr(cfg, "solution_scale_max", 1e12))
    solution_scale_min = float(getattr(cfg, "solution_scale_min", 1e-30))
    p0_solution_scale_max = float(getattr(cfg, "solution_scale_max_p0", solution_scale_max))
    p1_solution_scale_max = float(getattr(cfg, "solution_scale_max_p1", solution_scale_max))
    p2_solution_scale_max = float(getattr(cfg, "solution_scale_max_p2", solution_scale_max))
    p0_solution_scale_min = float(getattr(cfg, "solution_scale_min_p0", solution_scale_min))
    p1_solution_scale_min = float(getattr(cfg, "solution_scale_min_p1", solution_scale_min))
    p2_solution_scale_min = float(getattr(cfg, "solution_scale_min_p2", solution_scale_min))
    if p0_solution_scale_min <= 0.0:
        p0_solution_scale_min = 1e-30
    if p1_solution_scale_min <= 0.0:
        p1_solution_scale_min = 1e-30
    if p2_solution_scale_min <= 0.0:
        p2_solution_scale_min = 1e-30
    bc_loss_use_normalized = _to_bool(getattr(cfg, "bc_loss_use_normalized", True), default=True)
    bc_loss_scale_floor = float(getattr(cfg, "bc_loss_scale_floor", 1e-4))
    bc_loss_min_scale_ratio = float(getattr(cfg, "bc_loss_min_scale_ratio", 1.0))
    bc_loss_abs_mse_weight = float(getattr(cfg, "bc_loss_abs_mse_weight", 0.05))
    plot_vector_l2_hist = _to_bool(getattr(cfg, "plot_vector_l2_hist", True), default=True)
    use_clip_gn_phase1 = _to_bool(getattr(cfg, "use_clip_gn_phase1", True), default=True)
    grad_clip_max_norm_phase1 = float(getattr(cfg, "grad_clip_max_norm_phase1", 10.0))
    if grad_clip_max_norm_phase1 <= 0.0:
        print(
            f"[Warn] grad_clip_max_norm_phase1={grad_clip_max_norm_phase1} <= 0, "
            "fallback to 10.0"
        )
        grad_clip_max_norm_phase1 = 10.0
    p1_loss_name_suffix = (
        f"_clip_{_make_eps_tag(grad_clip_max_norm_phase1)}" if use_clip_gn_phase1 else ""
    )
    use_clip_gn_phase2 = _to_bool(getattr(cfg, "use_clip_gn_phase2", True), default=True)
    grad_clip_max_norm_phase2 = float(
        getattr(cfg, "grad_clip_max_norm_phase2", grad_clip_max_norm_phase1)
    )
    if grad_clip_max_norm_phase2 <= 0.0:
        print(
            f"[Warn] grad_clip_max_norm_phase2={grad_clip_max_norm_phase2} <= 0, "
            f"fallback to {grad_clip_max_norm_phase1:g}"
        )
        grad_clip_max_norm_phase2 = grad_clip_max_norm_phase1
    p2_loss_name_suffix = (
        f"_clip_{_make_eps_tag(grad_clip_max_norm_phase2)}" if use_clip_gn_phase2 else ""
    )
    n_basis_2loop = int(getattr(cfg, "n_basis_2loop", 22))
    if n_basis_2loop <= 0:
        raise ValueError(f"n_basis_2loop must be positive, got {n_basis_2loop}")
    n_coll_2loop = int(getattr(cfg, "n_coll_2loop", getattr(cfg, "n_coll_1loop", cfg.n_coll)))
    cy_2loop = float(getattr(cfg, "cy_2loop", getattr(cfg, "cy_1loop", cfg.cy)))
    x1_min_2loop = float(getattr(cfg, "x1_min_2loop", getattr(cfg, "x1_min_1loop", cfg.x1_min)))
    x1_max_2loop = float(getattr(cfg, "x1_max_2loop", getattr(cfg, "x1_max_1loop", cfg.x1_max)))
    x2_min_2loop = float(getattr(cfg, "x2_min_2loop", getattr(cfg, "x2_min_1loop", cfg.x2_min)))
    x2_max_2loop = float(getattr(cfg, "x2_max_2loop", getattr(cfg, "x2_max_1loop", cfg.x2_max)))
    y1_min_2loop = float(getattr(cfg, "y1_min_2loop", getattr(cfg, "y1_min_1loop", 0.0)))
    y1_max_2loop = float(getattr(cfg, "y1_max_2loop", getattr(cfg, "y1_max_1loop", 1.0)))
    y2_min_2loop = float(getattr(cfg, "y2_min_2loop", y1_min_2loop))
    y2_max_2loop = float(getattr(cfg, "y2_max_2loop", y1_max_2loop))
    p0_output_part = _normalize_output_part(getattr(cfg, "phase0_output_part", "re"))
    p0_output_part_tag = _output_part_tag(p0_output_part)
    p0_output_part_label = "Re" if p0_output_part == "re" else "Im"
    p1_output_part = _normalize_output_part(getattr(cfg, "phase1_output_part", "re"))
    p1_output_part_tag = _output_part_tag(p1_output_part)
    p1_output_part_label = "Re" if p1_output_part == "re" else "Im"
    p2_output_part = _normalize_output_part(getattr(cfg, "phase2_output_part", "re"))
    p2_output_part_tag = _output_part_tag(p2_output_part)
    p2_output_part_label = "Re" if p2_output_part == "re" else "Im"
    p0_phase_tag = "P0"
    p1_phase_tag = "P1_gnclip" if use_clip_gn_phase1 else "P1"
    p2_phase_tag = "P2_gnclip" if use_clip_gn_phase2 else "P2"
    postcalc_num_workers = _resolve_postcalc_workers(getattr(cfg, "postcalc_num_workers", 1))
    postcalc_chunk_size = max(int(getattr(cfg, "postcalc_chunk_size", 2000)), 1)
    postcalc_parallel_min_points = max(int(getattr(cfg, "postcalc_parallel_min_points", 5000)), 1)
    use_results_gpu_models = _to_bool(getattr(cfg, "use_results_gpu_models", False), default=False)
    save_phase_artifacts = _to_bool(getattr(cfg, "save_phase_artifacts", True), default=True)
    reuse_saved_models = _to_bool(getattr(cfg, "reuse_saved_models", False), default=False)
    save_eval_bundle = _to_bool(getattr(cfg, "save_eval_bundle", True), default=True)
    reuse_eval_bundle = _to_bool(getattr(cfg, "reuse_eval_bundle", False), default=False)
    p0_model_load_path = _resolve_optional_path(getattr(cfg, "phase0_model_load_path", None))
    p0_history_load_path = _resolve_optional_path(getattr(cfg, "phase0_history_load_path", None))
    p1_model_load_path = _resolve_optional_path(getattr(cfg, "phase1_model_load_path", None))
    p1_history_load_path = _resolve_optional_path(getattr(cfg, "phase1_history_load_path", None))
    p2_model_load_path = _resolve_optional_path(getattr(cfg, "phase2_model_load_path", None))
    p2_history_load_path = _resolve_optional_path(getattr(cfg, "phase2_history_load_path", None))
    p0_eval_bundle_load_path = _resolve_optional_path(getattr(cfg, "phase0_eval_bundle_load_path", None))
    p1_eval_bundle_load_path = _resolve_optional_path(getattr(cfg, "phase1_eval_bundle_load_path", None))
    p2_eval_bundle_load_path = _resolve_optional_path(getattr(cfg, "phase2_eval_bundle_load_path", None))

    if use_results_gpu_models:
        set_results_root_name("results_gpu_local_test" if use_local_cfg else "results_gpu")
        reuse_saved_models = True
        reuse_eval_bundle = True
    else:
        set_results_root_name("results_local_test" if use_local_cfg else "results")

    if run_phase1_only and run_phase2_only:
        raise ValueError("run_phase1_only and run_phase2_only cannot both be True.")
    if enable_phase1 and enable_phase2:
        raise ValueError("Phase1 and Phase2 cannot run together in one execution.")
    if run_phase1_only and enable_phase2:
        raise ValueError("run_phase1_only=True requires enable_phase2=False.")
    if run_phase2_only and enable_phase1:
        raise ValueError("run_phase2_only=True requires enable_phase1=False.")
    if run_phase1_only and (not enable_phase1):
        print("[Warn] run_phase1_only=True overrides enable_phase1=False -> enable_phase1=True")
        enable_phase1 = True
    if run_phase2_only and (not enable_phase2):
        print("[Warn] run_phase2_only=True overrides enable_phase2=False -> enable_phase2=True")
        enable_phase2 = True
    run_transfer_only = bool(run_phase1_only or run_phase2_only)

    if train_two_phase_only:
        reuse_saved_models = False
        reuse_eval_bundle = False
        p1_model_load_path = None
        p1_history_load_path = None
        p1_eval_bundle_load_path = None
        p2_model_load_path = None
        p2_history_load_path = None
        p2_eval_bundle_load_path = None
        if not run_transfer_only:
            p0_model_load_path = None
            p0_history_load_path = None
            p0_eval_bundle_load_path = None

    if not normalized_bc:
        solution_scale_mode = "manual"
        p0_solution_scale = 1.0
        p1_solution_scale = 1.0
        p2_solution_scale = 1.0
        bc_loss_use_normalized = False

    p0_bc_cfg = getattr(cfg, "coll_bc", {})
    if not isinstance(p0_bc_cfg, dict):
        p0_bc_cfg = {}
    p0_n_bc_edge = int(p0_bc_cfg.get("n_bc_edge", getattr(cfg, "n_bc_edge", 5)))
    p0_n_corner_each = int(p0_bc_cfg.get("n_corner_each", getattr(cfg, "n_corner_each", 5)))
    p0_target_total = int(p0_bc_cfg.get("target_total", getattr(cfg, "target_total", 200)))

    p1_bc_cfg = getattr(cfg, "coll_bc_1loop", {})
    if not isinstance(p1_bc_cfg, dict):
        p1_bc_cfg = {}
    p1_target_total_bc = int(p1_bc_cfg.get("target_total_bc", getattr(cfg, "target_total_bc", 500)))
    p1_n_bc_edge = int(p1_bc_cfg.get("n_bc_edge", getattr(cfg, "n_bc_edge_1loop", 6)))
    p1_n_face_pts = int(p1_bc_cfg.get("n_face_pts", getattr(cfg, "n_face_pts", 40)))
    p1_n_corner_extra = int(p1_bc_cfg.get("n_corner_extra", getattr(cfg, "n_corner_extra", 5)))
    p1_bc_abs_cap = float(p1_bc_cfg.get("bc_abs_cap", getattr(cfg, "bc_abs_cap", 1e8)))

    p2_bc_cfg = getattr(cfg, "coll_bc_2loop", {})
    if not isinstance(p2_bc_cfg, dict):
        p2_bc_cfg = {}
    p2_target_total_bc = int(p2_bc_cfg.get("target_total_bc", getattr(cfg, "target_total_bc_2loop", 500)))
    p2_n_bc_edge = int(p2_bc_cfg.get("n_bc_edge", getattr(cfg, "n_bc_edge_2loop", 6)))
    p2_n_face_pts = int(p2_bc_cfg.get("n_face_pts", getattr(cfg, "n_face_pts_2loop", 40)))
    p2_n_cell_pts = int(p2_bc_cfg.get("n_cell_pts", getattr(cfg, "n_cell_pts_2loop", 40)))
    p2_n_corner_extra = int(p2_bc_cfg.get("n_corner_extra", getattr(cfg, "n_corner_extra_2loop", 5)))
    p2_bc_abs_cap = float(p2_bc_cfg.get("bc_abs_cap", getattr(cfg, "bc_abs_cap_2loop", 1e8)))

    p0_artifacts = _phase_artifact_paths(
        phase=0,
        cy=cfg.cy,
        eps_global=cfg.eps_global,
        phase_tag=p0_phase_tag,
        output_part_tag=p0_output_part_tag,
    )
    p1_artifacts = _phase_artifact_paths(
        phase=1,
        cy=cfg.cy_1loop,
        eps_global=cfg.eps_global,
        phase_tag=p1_phase_tag,
        output_part_tag=p1_output_part_tag,
    )
    p2_artifacts = _phase_artifact_paths(
        phase=2,
        cy=cy_2loop,
        eps_global=cfg.eps_global,
        phase_tag=p2_phase_tag,
        output_part_tag=p2_output_part_tag,
    )
    p0_bundle_artifacts = _phase_eval_bundle_paths(
        phase=0,
        cy=cfg.cy,
        eps_global=cfg.eps_global,
        phase_tag=p0_phase_tag,
        output_part_tag=p0_output_part_tag,
    )
    p1_bundle_artifacts = _phase_eval_bundle_paths(
        phase=1,
        cy=cfg.cy_1loop,
        eps_global=cfg.eps_global,
        phase_tag=p1_phase_tag,
        output_part_tag=p1_output_part_tag,
    )
    p2_bundle_artifacts = _phase_eval_bundle_paths(
        phase=2,
        cy=cy_2loop,
        eps_global=cfg.eps_global,
        phase_tag=p2_phase_tag,
        output_part_tag=p2_output_part_tag,
    )

    if p0_model_load_path is None and run_transfer_only and os.path.isfile(p0_artifacts["model_abs"]):
        p0_model_load_path = p0_artifacts["model_abs"]

    if p0_model_load_path is None and reuse_saved_models and os.path.isfile(p0_artifacts["model_abs"]):
        p0_model_load_path = p0_artifacts["model_abs"]
    if p1_model_load_path is None and reuse_saved_models and os.path.isfile(p1_artifacts["model_abs"]):
        p1_model_load_path = p1_artifacts["model_abs"]
    if p2_model_load_path is None and reuse_saved_models and os.path.isfile(p2_artifacts["model_abs"]):
        p2_model_load_path = p2_artifacts["model_abs"]
    if p0_eval_bundle_load_path is None and reuse_eval_bundle and os.path.isfile(p0_bundle_artifacts["bundle_abs"]):
        p0_eval_bundle_load_path = p0_bundle_artifacts["bundle_abs"]
    if p1_eval_bundle_load_path is None and reuse_eval_bundle and os.path.isfile(p1_bundle_artifacts["bundle_abs"]):
        p1_eval_bundle_load_path = p1_bundle_artifacts["bundle_abs"]
    if p2_eval_bundle_load_path is None and reuse_eval_bundle and os.path.isfile(p2_bundle_artifacts["bundle_abs"]):
        p2_eval_bundle_load_path = p2_bundle_artifacts["bundle_abs"]

    if p0_history_load_path is None and p0_model_load_path is not None:
        if os.path.abspath(p0_model_load_path) == os.path.abspath(p0_artifacts["model_abs"]):
            if os.path.isfile(p0_artifacts["history_abs"]):
                p0_history_load_path = p0_artifacts["history_abs"]
        else:
            guessed = _guess_history_path_from_model_path(p0_model_load_path)
            if os.path.isfile(guessed):
                p0_history_load_path = guessed

    if p1_history_load_path is None and p1_model_load_path is not None:
        if os.path.abspath(p1_model_load_path) == os.path.abspath(p1_artifacts["model_abs"]):
            if os.path.isfile(p1_artifacts["history_abs"]):
                p1_history_load_path = p1_artifacts["history_abs"]
        else:
            guessed = _guess_history_path_from_model_path(p1_model_load_path)
            if os.path.isfile(guessed):
                p1_history_load_path = guessed

    if p2_history_load_path is None and p2_model_load_path is not None:
        if os.path.abspath(p2_model_load_path) == os.path.abspath(p2_artifacts["model_abs"]):
            if os.path.isfile(p2_artifacts["history_abs"]):
                p2_history_load_path = p2_artifacts["history_abs"]
        else:
            guessed = _guess_history_path_from_model_path(p2_model_load_path)
            if os.path.isfile(guessed):
                p2_history_load_path = guessed

    if run_transfer_only and p0_model_load_path is None:
        transfer_target = "Phase-1" if run_phase1_only else "Phase-2"
        raise FileNotFoundError(
            f"{transfer_target} transfer-only mode requires a Phase-0 checkpoint, "
            "but none was found. Set `phase0_model_load_path`, or enable "
            "`reuse_saved_models` with an existing Phase-0 checkpoint."
        )

    if p0_model_load_path is not None and (not os.path.isfile(p0_model_load_path)):
        raise FileNotFoundError(f"P0 checkpoint not found: {p0_model_load_path}")
    if enable_phase1 and p1_model_load_path is not None and (not os.path.isfile(p1_model_load_path)):
        raise FileNotFoundError(f"P1 checkpoint not found: {p1_model_load_path}")
    if enable_phase2 and p2_model_load_path is not None and (not os.path.isfile(p2_model_load_path)):
        raise FileNotFoundError(f"P2 checkpoint not found: {p2_model_load_path}")
    if p0_eval_bundle_load_path is not None and (not os.path.isfile(p0_eval_bundle_load_path)):
        raise FileNotFoundError(f"P0 eval-bundle not found: {p0_eval_bundle_load_path}")
    if enable_phase1 and p1_eval_bundle_load_path is not None and (not os.path.isfile(p1_eval_bundle_load_path)):
        raise FileNotFoundError(f"P1 eval-bundle not found: {p1_eval_bundle_load_path}")
    if enable_phase2 and p2_eval_bundle_load_path is not None and (not os.path.isfile(p2_eval_bundle_load_path)):
        raise FileNotFoundError(f"P2 eval-bundle not found: {p2_eval_bundle_load_path}")

    _cfg_abs, cfg_short = _save_config_snapshot(cfg_path, cfg.eps_global)

    print(f"[Using device]: {device}")
    print(f"[Mode] config source: {cfg_path} (use_local_config={use_local_cfg})")
    print(f"[saved] config snapshot to [{cfg_short}]")

    eps_kind, sol_branch, cde_form = _classify_eps_global(cfg.eps_global)
    print(f"[Mode] phase0 fixed eps_global reference: {cfg.eps_global}")
    print(f"[Mode] eps category: {eps_kind}")
    print(f"[Mode] analytic branch: {sol_branch}")
    print(f"[Mode] CDE form: {cde_form}")
    if eps_pos_int_n is not None:
        print(f"[Mode] positive-integer eps support: Phase0/Phase2 enabled, Phase1 disabled (n={eps_pos_int_n})")
    print(f"[Mode] phase0 output part: {p0_output_part_label}")
    print(f"[Mode] phase1 output part: {p1_output_part_label}")
    print(f"[Mode] phase2 output part: {p2_output_part_label}")
    print(f"[Mode] normalized BC mode: {normalized_bc}")
    print(f"[Mode] run phase1 only: {run_phase1_only}")
    print(f"[Mode] run phase2 only: {run_phase2_only}")

    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def make_boundary_loss_cfg(output_part: str):
        def _loss(model, xb, bct):
            return boundary_loss(
                model,
                xb,
                bct,
                use_normalized=bc_loss_use_normalized,
                scale_floor=bc_loss_scale_floor,
                min_scale_ratio=bc_loss_min_scale_ratio,
                abs_mse_weight=bc_loss_abs_mse_weight,
                output_part=output_part,
            )
        return _loss

    boundary_loss_cfg_p0 = make_boundary_loss_cfg(p0_output_part)
    boundary_loss_cfg_p1 = make_boundary_loss_cfg(p1_output_part)
    boundary_loss_cfg_p2 = make_boundary_loss_cfg(p2_output_part)

    print(
        "[Mode] BC loss: "
        f"{'normalized' if bc_loss_use_normalized else 'absolute-mse'}, "
        f"scale_floor={bc_loss_scale_floor:g}, "
        f"min_scale_ratio={bc_loss_min_scale_ratio:g}, "
        f"abs_mse_weight={bc_loss_abs_mse_weight:g}"
    )
    print(
        "[Mode] loss weights: "
        f"manual, lambda1={float(cfg.lambda1):g}, lambda2={float(cfg.lambda2):g}"
    )
    print(
        "[Mode] solution scaling: "
        f"mode={solution_scale_mode}, "
        f"P0={p0_solution_scale:g}, "
        f"P1={p1_solution_scale:g}, "
        f"P2={p2_solution_scale:g}, "
        f"ref_mean={solution_scale_ref_mean:g}, "
        f"min(P0/P1/P2)=({p0_solution_scale_min:g}/{p1_solution_scale_min:g}/{p2_solution_scale_min:g}), "
        f"max(P0/P1/P2)=({p0_solution_scale_max:g}/{p1_solution_scale_max:g}/{p2_solution_scale_max:g})"
    )
    print(f"[Mode] plot vector-L2 hist: {plot_vector_l2_hist}")
    print(
        "[Mode] grad-norm probe: "
        f"P0=False (fixed), "
        f"P1={use_clip_gn_phase1}(clip={grad_clip_max_norm_phase1:g}), "
        f"P2={use_clip_gn_phase2}(clip={grad_clip_max_norm_phase2:g})"
    )
    print(
        "[Mode] post-calc CPU parallel: "
        f"workers={postcalc_num_workers}, "
        f"chunk_size={postcalc_chunk_size}, "
        f"min_points={postcalc_parallel_min_points}"
    )
    print(f"[Mode] train_two_phase_only (skip plots/checks): {train_two_phase_only}")
    print(f"[Mode] use results_gpu models: {use_results_gpu_models}")
    print(f"[Mode] active results root: {get_results_root_name()}")
    print(f"[Mode] save phase artifacts: {save_phase_artifacts}")
    print(f"[Mode] reuse saved models: {reuse_saved_models}")
    print(f"[Mode] save eval bundle: {save_eval_bundle}")
    print(f"[Mode] reuse eval bundle: {reuse_eval_bundle}")
    p0_ckpt_mode_text = p0_model_load_path if p0_model_load_path is not None else f"train then save -> {p0_artifacts['model_short']}"
    p1_ckpt_mode_text = p1_model_load_path if p1_model_load_path is not None else f"train then save -> {p1_artifacts['model_short']}"
    p2_ckpt_mode_text = p2_model_load_path if p2_model_load_path is not None else f"train then save -> {p2_artifacts['model_short']}"
    p0_bundle_mode_text = (
        p0_eval_bundle_load_path
        if p0_eval_bundle_load_path is not None
        else f"sample then save -> {p0_bundle_artifacts['bundle_short']}"
    )
    p1_bundle_mode_text = (
        p1_eval_bundle_load_path
        if p1_eval_bundle_load_path is not None
        else f"sample then save -> {p1_bundle_artifacts['bundle_short']}"
    )
    p2_bundle_mode_text = (
        p2_eval_bundle_load_path
        if p2_eval_bundle_load_path is not None
        else f"sample then save -> {p2_bundle_artifacts['bundle_short']}"
    )
    print(f"[Mode] P0 checkpoint: {p0_ckpt_mode_text}")
    print(f"[Mode] P1 checkpoint: {p1_ckpt_mode_text}")
    print(f"[Mode] P2 checkpoint: {p2_ckpt_mode_text}")
    print(f"[Mode] P0 eval-bundle: {p0_bundle_mode_text}")
    print(f"[Mode] P1 eval-bundle: {p1_bundle_mode_text}")
    print(f"[Mode] P2 eval-bundle: {p2_bundle_mode_text}")
    if run_phase1_only:
        print(f"[Mode] run_phase1_only source P0 checkpoint: {p0_model_load_path}")
    if run_phase2_only:
        print(f"[Mode] run_phase2_only source P0 checkpoint: {p0_model_load_path}")
    phase0_t0 = time.perf_counter()
    p0_bundle_meta = {}
    x_coll = None
    x_b_tensor = None
    bc_target = None
    bc_target_train = None
    p0_transfer_log_messages = []
    if run_transfer_only:
        transfer_phase_name = "Phase 1" if run_phase1_only else "Phase 2"
        print(
            f"\n[Mode] transfer-only ({transfer_phase_name}) -> skip Phase 0 sampling/training/plots/checks; "
            "load Phase 0 checkpoint for transfer."
        )
        print(f"[Mode] transfer-only using P0 checkpoint: {p0_model_load_path}")
        p0_in_dim = _infer_phase0_in_dim_from_checkpoint(p0_model_load_path)
        p0_uses_eps_input = bool(p0_in_dim == 3)
        model_base = PinnModel(cfg, in_dim=p0_in_dim, output_part=p0_output_part).to(device)
    else:
        if p0_eval_bundle_load_path is not None:
            x_coll, x_b_tensor, bc_target, p0_bundle_meta = _load_eval_bundle(p0_eval_bundle_load_path, device)
            print(f"[loaded] P0 eval-bundle from [{p0_eval_bundle_load_path}]")
            b_part_norm = _normalize_output_part(
                p0_bundle_meta.get("output_part", p0_output_part),
                default=p0_output_part,
            )
            if b_part_norm != p0_output_part:
                print(
                    f"[Warn] loaded P0 eval-bundle output_part={b_part_norm} "
                    f"!= config phase0_output_part={p0_output_part}; applying config selection."
                )
        else:
            x_coll, x_b_tensor, bc_target, _ = build_inputs_and_boundary(
                cfg.n_coll,
                cfg.x1_min,
                cfg.x1_max,
                cfg.x2_min,
                cfg.x2_max,
                cfg.cy,
                cfg.eps_global,
                device,
                compute_function_target=False,
                output_part=p0_output_part,
                n_bc_edge=p0_n_bc_edge,
                n_corner_each=p0_n_corner_each,
                target_total=p0_target_total,
            )

        if x_coll.ndim != 2 or x_coll.shape[1] not in (2, 3):
            raise ValueError(f"x_coll must be shape (N,2) or (N,3), got {tuple(x_coll.shape)}")
        p0_uses_eps_input = bool(x_coll.shape[1] == 3)
        if p0_uses_eps_input:
            eps_u = torch.unique(x_coll[:, 2]).detach().cpu().numpy()
            print(
                f"[P0] eps-input collocation mode: unique eps={len(eps_u)}, "
                f"range=[{float(np.min(eps_u)):.4g},{float(np.max(eps_u)):.4g}]"
            )

        bc_target = _slice_phase0_target_by_part(
            bc_target,
            n_basis=cfg.n_basis,
            output_part=p0_output_part,
            tensor_name="P0 bc_target",
        )

        if solution_scale_mode == "auto":
            (
                p0_solution_scale,
                p0_bc_mean_abs,
                p0_raw_solution_scale,
                p0_scale_capped,
                p0_scale_floored,
            ) = _auto_solution_scale_from_bc(
                bc_target,
                ref_mean_abs=solution_scale_ref_mean,
                max_scale=p0_solution_scale_max,
                min_scale=p0_solution_scale_min,
            )
            p0_scale_flags = []
            if p0_scale_capped:
                p0_scale_flags.append(f"capped@{p0_solution_scale_max:.3e}")
            if p0_scale_floored:
                p0_scale_flags.append(f"floored@{p0_solution_scale_min:.3e}")
            p0_scale_flag_text = f" ({', '.join(p0_scale_flags)})" if p0_scale_flags else ""
            print(
                f"[P0] auto solution scale from BC mean abs={p0_bc_mean_abs:.3e} "
                f"-> raw={p0_raw_solution_scale:.3e}, used={p0_solution_scale:.3e}"
                f"{p0_scale_flag_text}"
            )
        bc_target_train = bc_target * p0_solution_scale

        model_base = PinnModel(cfg, in_dim=int(x_coll.shape[1]), output_part=p0_output_part).to(device)

    from two_site_chain.mat_data import (
        a1,
        a2,
        a3,
        a4,
        a5,
        a1_eps0,
        a2_eps0,
        a3_eps0,
        a4_eps0,
        a5_eps0,
    )

    ak_list = [a.to(device) for a in [a1, a2, a3, a4, a5]]
    ak_list_eps0 = [a.to(device) for a in [a1_eps0, a2_eps0, a3_eps0, a4_eps0, a5_eps0]]

    from two_site_chain.conn_mat import ConnectionAMatricesFixedWithEps0

    a_builder_fixed = ConnectionAMatricesFixedWithEps0(
        ak_list=ak_list,
        ak_list_eps0=ak_list_eps0,
        cy_val=cfg.cy,
    ).to(device)

    if run_transfer_only:
        p0_log_file = None
        p0_log_short = None

        def p0_log(msg: str):
            p0_transfer_log_messages.append(str(msg))

    else:
        p0_log_file, p0_log_short, p0_log = _open_phase_log_writer(
            phase=0,
            cy=cfg.cy,
            eps_global=cfg.eps_global,
            phase_tag=p0_phase_tag,
            output_part_tag=p0_output_part_tag,
        )

    if run_transfer_only:
        transfer_mode_tag = "phase1-only" if run_phase1_only else "phase2-only"
        p0_mode_text = (
            f"{transfer_mode_tag}-load-eps-input"
            if p0_uses_eps_input
            else f"{transfer_mode_tag}-load-fixed-eps"
        )
    else:
        p0_mode_text = "fixed-eps"
    p0_log(
        f"[P0] mode={p0_mode_text}, eps_global={cfg.eps_global}, "
        f"cy={cfg.cy}, output_part={p0_output_part_label}, solution_scale={p0_solution_scale:g}"
    )

    hist_tot = None
    hist_cde = None
    hist_bc = None
    p0_train_info = {}
    if p0_model_load_path is not None:
        p0_load_header = "------ Phase 0: Load 2-Site Chain checkpoint ------"
        print(f"\n{p0_load_header}")
        p0_log(p0_load_header)
        p0_meta = _load_model_checkpoint(model_base, p0_model_load_path, device)
        if isinstance(p0_meta, dict) and ("output_part" in p0_meta):
            ckpt_part = _normalize_output_part(p0_meta["output_part"])
            if ckpt_part != p0_output_part:
                warn_msg = (
                    f"[Warn] P0 checkpoint output_part={ckpt_part} "
                    f"!= config phase0_output_part={p0_output_part}."
                )
                print(warn_msg)
                p0_log(warn_msg)
        loaded_msg = (
            f"[loaded] P0 checkpoint from [{p0_model_load_path}] "
            f"(in_dim={int(model_base.net[0].in_features)})"
        )
        print(loaded_msg)
        p0_log(loaded_msg)
        p0_log(f"[P0] loaded checkpoint: {p0_model_load_path}")

        if isinstance(p0_meta, dict) and ("pred_scale" in p0_meta):
            p0_solution_scale = float(p0_meta["pred_scale"])
            msg = f"[P0] use pred_scale from checkpoint: {p0_solution_scale:g}"
            print(msg)
            p0_log(msg)
        elif "pred_scale" in p0_bundle_meta:
            p0_solution_scale = float(p0_bundle_meta["pred_scale"])
            msg = f"[P0] use pred_scale from eval-bundle: {p0_solution_scale:g}"
            print(msg)
            p0_log(msg)

        loaded_hist = _load_loss_history(p0_history_load_path)
        if loaded_hist is not None:
            hist_tot, hist_cde, hist_bc = loaded_hist
            msg = f"[P0] loaded loss history from [{p0_history_load_path}]"
            print(msg)
            p0_log(msg)
        else:
            msg = "[P0] no loss history found for loaded checkpoint."
            print(msg)
            p0_log(msg)
    else:
        if (x_coll is None) or (x_b_tensor is None) or (bc_target_train is None):
            raise RuntimeError(
                "P0 training tensors are not initialized. "
                "This indicates an unexpected control-flow issue."
            )

        print(
            f"\n------ Phase 0: Train 2-Site Chain (fixed eps={cfg.eps_global}, output={p0_output_part_label}) ------"
        )

        def cde_fixed_p0(model, a_builder, x_batch, n_basis, eps_val):
            return cde_residual_loss_fixed_eps(
                model,
                a_builder,
                x_batch,
                n_basis,
                eps_val=eps_val,
                output_part=p0_output_part,
            )

        (
            model_base,
            hist_tot,
            hist_cde,
            hist_bc,
            p0_train_info,
        ) = train_model_fixed_eps(
            model=model_base,
            a_builder=a_builder_fixed,
            x_coll=x_coll,
            x_b_tensor=x_b_tensor,
            bc_target=bc_target_train,
            cde_loss_fixed_fn=cde_fixed_p0,
            bc_loss_fn=boundary_loss_cfg_p0,
            n_basis=cfg.n_basis,
            eps_val=cfg.eps_global,
            lr_init=cfg.learning_rate_p0,
            warmup_len=cfg.warmup_epochs_p0,
            total_epochs=cfg.phase0_epochs,
            lam1=cfg.lambda1,
            lam2=cfg.lambda2,
            cosine_min_lr=cfg.cosine_min_lr,
            print_every=cfg.print_every,
            phase_name="P0",
            log_fn=p0_log,
            use_grad_norm_probe=False,
        )

        if save_phase_artifacts:
            saved_paths = _save_phase_artifacts(
                model=model_base,
                hist_tot=hist_tot,
                hist_cde=hist_cde,
                hist_bc=hist_bc,
                phase=0,
                cy=cfg.cy,
                eps_global=cfg.eps_global,
                pred_scale=p0_solution_scale,
                extra_meta={
                    "in_dim": int(x_coll.shape[1]),
                    "n_basis": int(cfg.n_basis),
                    "output_part": p0_output_part,
                    **p0_train_info,
                },
                phase_tag=p0_phase_tag,
                output_part_tag=p0_output_part_tag,
            )
            print(f"[saved] {os.path.basename(saved_paths['model_abs'])} to [{os.path.dirname(saved_paths['model_short'])}]")
            print(f"[saved] {os.path.basename(saved_paths['history_abs'])} to [{os.path.dirname(saved_paths['history_short'])}]")
            p0_log(f"[P0] saved checkpoint: {saved_paths['model_short']}")
            p0_log(f"[P0] saved loss history: {saved_paths['history_short']}")

    if save_eval_bundle and (not run_transfer_only):
        if (x_coll is None) or (x_b_tensor is None) or (bc_target is None):
            raise RuntimeError(
                "P0 eval-bundle tensors are not initialized. "
                "This indicates an unexpected control-flow issue."
            )
        saved_bundle_p0 = _save_eval_bundle(
            phase=0,
            cy=cfg.cy,
            eps_global=cfg.eps_global,
            x_coll=x_coll,
            x_b_tensor=x_b_tensor,
            bc_target=bc_target,
            pred_scale=p0_solution_scale,
            extra_meta={
                "in_dim": int(x_coll.shape[1]),
                "n_basis": int(cfg.n_basis),
                "output_part": p0_output_part,
                **p0_train_info,
            },
            phase_tag=p0_phase_tag,
            output_part_tag=p0_output_part_tag,
        )
        print(f"[saved] {os.path.basename(saved_bundle_p0['bundle_abs'])} to [{os.path.dirname(saved_bundle_p0['bundle_short'])}]")
        p0_log(f"[P0] saved eval-bundle: {saved_bundle_p0['bundle_short']}")

    if (not train_two_phase_only) and (not run_transfer_only):
        if (x_coll is None) or (x_b_tensor is None) or (bc_target is None):
            raise RuntimeError(
                "P0 plot/check tensors are not initialized. "
                "This indicates an unexpected control-flow issue."
            )
        if (hist_tot is not None) and (hist_cde is not None) and (hist_bc is not None):
            plot_losses(
                total_vals=hist_tot,
                cde_vals=hist_cde,
                bc_vals=hist_bc,
                title=rf"2-site chain, $c={cfg.cy}$, $\varepsilon={cfg.eps_global:g}$",
                cy=cfg.cy,
                save_dir="1_losses",
                fname=f"P0_loss_all_eps_{eps_tag}.png",
                fname2=f"P0_loss_total_eps_{eps_tag}.png",
                phase=0,
                phase_tag=p0_phase_tag,
            )
        else:
            msg = "[P0] skip plot_losses: loss history unavailable."
            print(msg)
            p0_log(msg)

        with torch.no_grad():
            f_target = compute_function_target_from_xcoll(
                x_coll,
                cy_val=cfg.cy,
                eps_val=cfg.eps_global,
                output_part=p0_output_part,
                num_workers=postcalc_num_workers,
                chunk_size=postcalc_chunk_size,
                parallel_min_points=postcalc_parallel_min_points,
            )

        post_train_check(
            model=model_base,
            x_coll=x_coll,
            x_b_tensor=x_b_tensor,
            bc_target=bc_target,
            cy_val=cfg.cy,
            eps_global=cfg.eps_global,
            compute_function_target_from_xcoll=compute_function_target_from_xcoll,
            precomputed_true=f_target,
            phase_name="P0",
            pred_scale=p0_solution_scale,
            log_fn=p0_log,
            output_part=p0_output_part,
        )

        msg = f"[P0] output_part={p0_output_part_label}: skip plot_error_dis."
        print(msg)
        p0_log(msg)
    elif run_transfer_only:
        transfer_phase_name = "phase1_only" if run_phase1_only else "phase2_only"
        msg = f"[P0] {transfer_phase_name}=True: skip P0 training/plots/checks."
        print(msg)
        p0_log(msg)
    else:
        msg = "[P0] train_two_phase_only=True: skip plot_losses/post_train_check/plot_error."
        print(msg)
        p0_log(msg)
    phase0_elapsed = time.perf_counter() - phase0_t0
    phase0_msg = f"[P0] elapsed: {_format_elapsed(phase0_elapsed)} ({phase0_elapsed:.2f}s)"
    print(phase0_msg)
    p0_log(phase0_msg)
    if p0_log_file is not None:
        p0_log_file.close()
        print(f"[saved] P0 log to [{p0_log_short}]")
    else:
        print("[Mode] transfer-only=True: existing P0 log is kept unchanged.")

    if (not enable_phase1) and (not enable_phase2):
        print("\n[Mode] enable_phase1=False and enable_phase2=False -> no transfer phase to run.")
        print("\n----- ALL TRAINING COMPLETE -----\n")
        return

    if enable_phase1:
        # ============================================================
        # Phase 1: Transfer learning for 2-site 1-loop bubble (3D input)
        # ============================================================
        print(
            f"\n------ Phase 1: Transfer to 1-loop Bubble "
            f"(fixed eps={cfg.eps_global}, output={p1_output_part_label}) ------"
        )
        phase1_t0 = time.perf_counter()

        p1_bundle_meta = {}
        if p1_eval_bundle_load_path is not None:
            x_coll_1loop, x_b_tensor_1loop, bc_target_1loop, p1_bundle_meta = _load_eval_bundle(
                p1_eval_bundle_load_path,
                device,
            )
            print(f"[loaded] P1 eval-bundle from [{p1_eval_bundle_load_path}]")
            b_part_norm = _normalize_output_part(
                p1_bundle_meta.get("output_part", p1_output_part),
                default=p1_output_part,
            )
            if b_part_norm != p1_output_part:
                print(
                    f"[Warn] loaded P1 eval-bundle output_part={b_part_norm} "
                    f"!= config phase1_output_part={p1_output_part}; applying config selection."
                )
        else:
            x_coll_1loop, x_b_tensor_1loop, bc_target_1loop, _ = build_inputs_and_boundary_1loop(
                cfg.n_coll_1loop,
                cfg.x1_min_1loop,
                cfg.x1_max_1loop,
                cfg.x2_min_1loop,
                cfg.x2_max_1loop,
                cfg.y1_min_1loop,
                cfg.y1_max_1loop,
                cfg.cy_1loop,
                cfg.eps_global,
                device,
                compute_function_target=False,
                output_part=p1_output_part,
                target_total_bc=p1_target_total_bc,
                n_bc_edge=p1_n_bc_edge,
                n_face_pts=p1_n_face_pts,
                n_corner_extra=p1_n_corner_extra,
                bc_abs_cap=p1_bc_abs_cap,
            )
        bc_target_1loop = _slice_phase0_target_by_part(
            bc_target_1loop,
            n_basis=cfg.n_basis_1loop,
            output_part=p1_output_part,
            tensor_name="P1 bc_target",
        )
        if solution_scale_mode == "auto":
            (
                p1_solution_scale,
                p1_bc_mean_abs,
                p1_raw_solution_scale,
                p1_scale_capped,
                p1_scale_floored,
            ) = _auto_solution_scale_from_bc(
                bc_target_1loop,
                ref_mean_abs=solution_scale_ref_mean,
                max_scale=p1_solution_scale_max,
                min_scale=p1_solution_scale_min,
            )
            p1_scale_flags = []
            if p1_scale_capped:
                p1_scale_flags.append(f"capped@{p1_solution_scale_max:.3e}")
            if p1_scale_floored:
                p1_scale_flags.append(f"floored@{p1_solution_scale_min:.3e}")
            p1_scale_flag_text = f" ({', '.join(p1_scale_flags)})" if p1_scale_flags else ""
            print(
                f"[P1] auto solution scale from BC mean abs={p1_bc_mean_abs:.3e} "
                f"-> raw={p1_raw_solution_scale:.3e}, used={p1_solution_scale:.3e}"
                f"{p1_scale_flag_text}"
            )
        bc_target_1loop_train = bc_target_1loop * p1_solution_scale
    
        if x_coll_1loop.shape[1] != 3:
            raise ValueError(
                f"x_coll_1loop must have 3 columns (x1,x2,y1). Got shape: {tuple(x_coll_1loop.shape)}"
            )
    
        model_p1 = TransferPinnModel(
            cfg,
            phase0_model=model_base,
            freeze_core=True,
            output_part=p1_output_part,
        ).to(device)
    
        from tl_two_site_bubble.mat_data_1loop import (
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a1_eps0,
            a2_eps0,
            a3_eps0,
            a4_eps0,
            a5_eps0,
            a6_eps0,
            a7_eps0,
            a8_eps0,
            a9_eps0,
            a10_eps0,
            a11_eps0,
        )
    
        ak_list_1loop = [a.to(device) for a in [a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]]
        ak_list_1loop_eps0 = [
            a.to(device)
            for a in [a1_eps0, a2_eps0, a3_eps0, a4_eps0, a5_eps0, a6_eps0, a7_eps0, a8_eps0, a9_eps0, a10_eps0, a11_eps0]
        ]
    
        from tl_two_site_bubble.conn_mat_1loop import ConnectionAMatricesFixedWithEps0_1Loop
    
        a_builder_1loop = ConnectionAMatricesFixedWithEps0_1Loop(
            ak_list=ak_list_1loop,
            ak_list_eps0=ak_list_1loop_eps0,
            cy_val=cfg.cy_1loop,
        ).to(device)
    
        p1_log_file, p1_log_short, p1_log = _open_phase_log_writer(
            phase=1,
            cy=cfg.cy_1loop,
            eps_global=cfg.eps_global,
            phase_tag=p1_phase_tag,
            output_part_tag=p1_output_part_tag,
            log_suffix=(
                f"clip_{_make_eps_tag(grad_clip_max_norm_phase1)}"
                if use_clip_gn_phase1
                else None
            ),
        )
        p1_log(
            f"[P1] eps_global={cfg.eps_global}, cy={cfg.cy_1loop}, "
            f"output_part={p1_output_part_label}, solution_scale={p1_solution_scale:g}"
        )
        if use_clip_gn_phase1:
            p1_log(
                f"[P1] grad clip enabled: clip_max_norm={grad_clip_max_norm_phase1:g}"
            )
        if run_phase1_only:
            p1_log("[P0 transfer context]")
            for msg in p0_transfer_log_messages:
                p1_log(msg)
            p1_log("[P0] run_phase1_only=True: existing P0 log file kept unchanged.")
    
        hist_tot_p1 = None
        hist_cde_p1 = None
        hist_bc_p1 = None
        p1_train_info = {}
        if p1_model_load_path is not None:
            print("\n------ Phase 1: Load 1-loop checkpoint ------")
            p1_meta = _load_model_checkpoint(model_p1, p1_model_load_path, device)
            if isinstance(p1_meta, dict) and ("output_part" in p1_meta):
                ckpt_part = _normalize_output_part(p1_meta["output_part"])
                if ckpt_part != p1_output_part:
                    warn_msg = (
                        f"[Warn] P1 checkpoint output_part={ckpt_part} "
                        f"!= config phase1_output_part={p1_output_part}."
                    )
                    print(warn_msg)
                    p1_log(warn_msg)
            print(f"[loaded] P1 checkpoint from [{p1_model_load_path}]")
            p1_log(f"[P1] loaded checkpoint: {p1_model_load_path}")
    
            if isinstance(p1_meta, dict) and ("pred_scale" in p1_meta):
                p1_solution_scale = float(p1_meta["pred_scale"])
                msg = f"[P1] use pred_scale from checkpoint: {p1_solution_scale:g}"
                print(msg)
                p1_log(msg)
            elif "pred_scale" in p1_bundle_meta:
                p1_solution_scale = float(p1_bundle_meta["pred_scale"])
                msg = f"[P1] use pred_scale from eval-bundle: {p1_solution_scale:g}"
                print(msg)
                p1_log(msg)
    
            loaded_hist_p1 = _load_loss_history(p1_history_load_path)
            if loaded_hist_p1 is not None:
                hist_tot_p1, hist_cde_p1, hist_bc_p1 = loaded_hist_p1
                msg = f"[P1] loaded loss history from [{p1_history_load_path}]"
                print(msg)
                p1_log(msg)
            else:
                msg = "[P1] no loss history found for loaded checkpoint."
                print(msg)
                p1_log(msg)
        else:
            def cde_fixed_p1(model, a_builder, x_batch, n_basis, eps_val):
                return cde_residual_loss_fixed_eps_1loop(
                    model,
                    a_builder,
                    x_batch,
                    n_basis,
                    eps_val=eps_val,
                    output_part=p1_output_part,
                )
    
            (
                model_p1,
                hist_tot_p1,
                hist_cde_p1,
                hist_bc_p1,
                p1_train_info,
            ) = train_model_fixed_eps(
                model=model_p1,
                a_builder=a_builder_1loop,
                x_coll=x_coll_1loop,
                x_b_tensor=x_b_tensor_1loop,
                bc_target=bc_target_1loop_train,
                cde_loss_fixed_fn=cde_fixed_p1,
                bc_loss_fn=boundary_loss_cfg_p1,
                n_basis=cfg.n_basis_1loop,
                eps_val=cfg.eps_global,
                lr_init=cfg.learning_rate_p1,
                warmup_len=cfg.warmup_epochs_p1,
                total_epochs=cfg.phase1_epochs,
                lam1=cfg.lambda1,
                lam2=cfg.lambda2,
                cosine_min_lr=cfg.cosine_min_lr,
                print_every=cfg.print_every,
                phase_name="P1",
                log_fn=p1_log,
                use_grad_norm_probe=use_clip_gn_phase1,
                grad_clip_max_norm=grad_clip_max_norm_phase1,
            )
    
            if save_phase_artifacts:
                saved_paths_p1 = _save_phase_artifacts(
                    model=model_p1,
                    hist_tot=hist_tot_p1,
                    hist_cde=hist_cde_p1,
                    hist_bc=hist_bc_p1,
                    phase=1,
                    cy=cfg.cy_1loop,
                    eps_global=cfg.eps_global,
                    pred_scale=p1_solution_scale,
                    extra_meta={
                        "in_dim": 3,
                        "n_basis": int(cfg.n_basis_1loop),
                        "output_part": p1_output_part,
                        "phase0_model_ref": p0_artifacts["model_short"],
                        **p1_train_info,
                    },
                    phase_tag=p1_phase_tag,
                    output_part_tag=p1_output_part_tag,
                )
                print(f"[saved] {os.path.basename(saved_paths_p1['model_abs'])} to [{os.path.dirname(saved_paths_p1['model_short'])}]")
                print(f"[saved] {os.path.basename(saved_paths_p1['history_abs'])} to [{os.path.dirname(saved_paths_p1['history_short'])}]")
                p1_log(f"[P1] saved checkpoint: {saved_paths_p1['model_short']}")
                p1_log(f"[P1] saved loss history: {saved_paths_p1['history_short']}")
    
        if save_eval_bundle:
            saved_bundle_p1 = _save_eval_bundle(
                phase=1,
                cy=cfg.cy_1loop,
                eps_global=cfg.eps_global,
                x_coll=x_coll_1loop,
                x_b_tensor=x_b_tensor_1loop,
                bc_target=bc_target_1loop,
                pred_scale=p1_solution_scale,
                extra_meta={
                    "in_dim": int(x_coll_1loop.shape[1]),
                    "n_basis": int(cfg.n_basis_1loop),
                    "output_part": p1_output_part,
                    "phase0_model_ref": p0_artifacts["model_short"],
                    **p1_train_info,
                },
                phase_tag=p1_phase_tag,
                output_part_tag=p1_output_part_tag,
            )
            print(f"[saved] {os.path.basename(saved_bundle_p1['bundle_abs'])} to [{os.path.dirname(saved_bundle_p1['bundle_short'])}]")
            p1_log(f"[P1] saved eval-bundle: {saved_bundle_p1['bundle_short']}")
    
        if not train_two_phase_only:
            if (hist_tot_p1 is not None) and (hist_cde_p1 is not None) and (hist_bc_p1 is not None):
                plot_losses(
                    total_vals=hist_tot_p1,
                    cde_vals=hist_cde_p1,
                    bc_vals=hist_bc_p1,
                    title=rf"2-site 1-loop bubble, $c={cfg.cy_1loop}$, $\varepsilon={cfg.eps_global:g}$",
                    cy=cfg.cy_1loop,
                    save_dir="1_losses",
                    fname=f"P1_loss_all_eps_{eps_tag}{p1_loss_name_suffix}.png",
                    fname2=f"P1_loss_total_eps_{eps_tag}{p1_loss_name_suffix}.png",
                    phase=1,
                    phase_tag=p1_phase_tag,
                )
            else:
                msg = "[P1] skip plot_losses: loss history unavailable."
                print(msg)
                p1_log(msg)
    
            with torch.no_grad():
                f_target_1loop = compute_function_target_from_xcoll_1loop(
                    x_coll_1loop,
                    cy_val=cfg.cy_1loop,
                    eps_val=cfg.eps_global,
                    output_part=p1_output_part,
                    num_workers=postcalc_num_workers,
                    chunk_size=postcalc_chunk_size,
                    parallel_min_points=postcalc_parallel_min_points,
                )
    
            post_train_check(
                model=model_p1,
                x_coll=x_coll_1loop,
                x_b_tensor=x_b_tensor_1loop,
                bc_target=bc_target_1loop,
                cy_val=cfg.cy_1loop,
                eps_global=cfg.eps_global,
                compute_function_target_from_xcoll=compute_function_target_from_xcoll_1loop,
                precomputed_true=f_target_1loop,
                phase_name="P1",
                pred_scale=p1_solution_scale,
                log_fn=p1_log,
                output_part=p1_output_part,
            )
    
            msg = f"[P1] output_part={p1_output_part_label}: skip plot_error_dis."
            print(msg)
            p1_log(msg)
        else:
            msg = "[P1] train_two_phase_only=True: skip plot_losses/post_train_check/plot_error."
            print(msg)
            p1_log(msg)
        phase1_elapsed = time.perf_counter() - phase1_t0
        phase1_msg = f"[P1] elapsed: {_format_elapsed(phase1_elapsed)} ({phase1_elapsed:.2f}s)"
        print(phase1_msg)
        p1_log(phase1_msg)
        p1_log_file.close()
        print(f"[saved] P1 log to [{p1_log_short}]")

    elif enable_phase2:
        # ============================================================
        # Phase 2: Transfer learning for 2-site 2-loop sunset (4D input)
        # ============================================================
        print(
            f"\n------ Phase 2: Transfer to 2-loop Sunset "
            f"(fixed eps={cfg.eps_global}, output={p2_output_part_label}) ------"
        )
        phase2_t0 = time.perf_counter()

        phase2_epochs = int(
            getattr(cfg, "phase2_epochs", getattr(cfg, "phase1_epochs", 2000))
        )
        phase2_warmup_epochs = int(
            getattr(cfg, "warmup_epochs_p2", getattr(cfg, "warmup_epochs_p1", 200))
        )
        phase2_learning_rate = float(
            getattr(cfg, "learning_rate_p2", getattr(cfg, "learning_rate_p1", 1e-3))
        )

        p2_bundle_meta = {}
        if p2_eval_bundle_load_path is not None:
            x_coll_2loop, x_b_tensor_2loop, bc_target_2loop, p2_bundle_meta = _load_eval_bundle(
                p2_eval_bundle_load_path,
                device,
            )
            print(f"[loaded] P2 eval-bundle from [{p2_eval_bundle_load_path}]")
            b_part_norm = _normalize_output_part(
                p2_bundle_meta.get("output_part", p2_output_part),
                default=p2_output_part,
            )
            if b_part_norm != p2_output_part:
                print(
                    f"[Warn] loaded P2 eval-bundle output_part={b_part_norm} "
                    f"!= config phase2_output_part={p2_output_part}; applying config selection."
                )
        else:
            x_coll_2loop, x_b_tensor_2loop, bc_target_2loop, _ = build_inputs_and_boundary_2loop(
                n_coll_2loop,
                x1_min_2loop,
                x1_max_2loop,
                x2_min_2loop,
                x2_max_2loop,
                y1_min_2loop,
                y1_max_2loop,
                y2_min_2loop,
                y2_max_2loop,
                cy_2loop,
                cfg.eps_global,
                device,
                compute_function_target=False,
                output_part=p2_output_part,
                target_total_bc=p2_target_total_bc,
                n_bc_edge=p2_n_bc_edge,
                n_face_pts=p2_n_face_pts,
                n_cell_pts=p2_n_cell_pts,
                n_corner_extra=p2_n_corner_extra,
                bc_abs_cap=p2_bc_abs_cap,
                n_basis=n_basis_2loop,
            )

        bc_target_2loop = _slice_phase0_target_by_part(
            bc_target_2loop,
            n_basis=n_basis_2loop,
            output_part=p2_output_part,
            tensor_name="P2 bc_target",
        )
        if solution_scale_mode == "auto":
            (
                p2_solution_scale,
                p2_bc_mean_abs,
                p2_raw_solution_scale,
                p2_scale_capped,
                p2_scale_floored,
            ) = _auto_solution_scale_from_bc(
                bc_target_2loop,
                ref_mean_abs=solution_scale_ref_mean,
                max_scale=p2_solution_scale_max,
                min_scale=p2_solution_scale_min,
            )
            p2_scale_flags = []
            if p2_scale_capped:
                p2_scale_flags.append(f"capped@{p2_solution_scale_max:.3e}")
            if p2_scale_floored:
                p2_scale_flags.append(f"floored@{p2_solution_scale_min:.3e}")
            p2_scale_flag_text = f" ({', '.join(p2_scale_flags)})" if p2_scale_flags else ""
            print(
                f"[P2] auto solution scale from BC mean abs={p2_bc_mean_abs:.3e} "
                f"-> raw={p2_raw_solution_scale:.3e}, used={p2_solution_scale:.3e}"
                f"{p2_scale_flag_text}"
            )
        bc_target_2loop_train = bc_target_2loop * p2_solution_scale

        if x_coll_2loop.shape[1] != 4:
            raise ValueError(
                f"x_coll_2loop must have 4 columns (x1,x2,y1,y2). "
                f"Got shape: {tuple(x_coll_2loop.shape)}"
            )

        model_p2 = TransferPinnModel(
            cfg,
            phase0_model=model_base,
            freeze_core=True,
            output_part=p2_output_part,
            target_in_dim=4,
            target_n_basis=n_basis_2loop,
        ).to(device)

        from tl_two_site_sunset.mat_data_2loop import (
            a1,
            a2,
            a3,
            a4,
            a5,
            a6,
            a7,
            a8,
            a9,
            a10,
            a11,
            a12,
            a13,
            a14,
            a15,
            a16,
            a17,
            a18,
            a19,
            a20,
            a21,
            a22,
            a23,
            a1_eps0,
            a2_eps0,
            a3_eps0,
            a4_eps0,
            a5_eps0,
            a6_eps0,
            a7_eps0,
            a8_eps0,
            a9_eps0,
            a10_eps0,
            a11_eps0,
            a12_eps0,
            a13_eps0,
            a14_eps0,
            a15_eps0,
            a16_eps0,
            a17_eps0,
            a18_eps0,
            a19_eps0,
            a20_eps0,
            a21_eps0,
            a22_eps0,
            a23_eps0,
        )

        ak_list_2loop = [
            a.to(device)
            for a in [
                a1,
                a2,
                a3,
                a4,
                a5,
                a6,
                a7,
                a8,
                a9,
                a10,
                a11,
                a12,
                a13,
                a14,
                a15,
                a16,
                a17,
                a18,
                a19,
                a20,
                a21,
                a22,
                a23,
            ]
        ]
        ak_list_2loop_eps0 = [
            a.to(device)
            for a in [
                a1_eps0,
                a2_eps0,
                a3_eps0,
                a4_eps0,
                a5_eps0,
                a6_eps0,
                a7_eps0,
                a8_eps0,
                a9_eps0,
                a10_eps0,
                a11_eps0,
                a12_eps0,
                a13_eps0,
                a14_eps0,
                a15_eps0,
                a16_eps0,
                a17_eps0,
                a18_eps0,
                a19_eps0,
                a20_eps0,
                a21_eps0,
                a22_eps0,
                a23_eps0,
            ]
        ]

        from tl_two_site_sunset.conn_mat_2loop import ConnectionAMatricesFixedWithEps0_2Loop

        a_builder_2loop = ConnectionAMatricesFixedWithEps0_2Loop(
            n_basis_local=n_basis_2loop,
            n_letters=23,
            ak_list=ak_list_2loop,
            ak_list_eps0=ak_list_2loop_eps0,
            cy_val=cy_2loop,
        ).to(device)

        p2_log_file, p2_log_short, p2_log = _open_phase_log_writer(
            phase=2,
            cy=cy_2loop,
            eps_global=cfg.eps_global,
            phase_tag=p2_phase_tag,
            output_part_tag=p2_output_part_tag,
            log_suffix=(
                f"clip_{_make_eps_tag(grad_clip_max_norm_phase2)}"
                if use_clip_gn_phase2
                else None
            ),
        )
        p2_log(
            f"[P2] eps_global={cfg.eps_global}, cy={cy_2loop}, "
            f"output_part={p2_output_part_label}, solution_scale={p2_solution_scale:g}"
        )
        if use_clip_gn_phase2:
            p2_log(f"[P2] grad clip enabled: clip_max_norm={grad_clip_max_norm_phase2:g}")
        if run_phase2_only:
            p2_log("[P0 transfer context]")
            for msg in p0_transfer_log_messages:
                p2_log(msg)
            p2_log("[P0] run_phase2_only=True: existing P0 log file kept unchanged.")

        hist_tot_p2 = None
        hist_cde_p2 = None
        hist_bc_p2 = None
        p2_train_info = {}
        if p2_model_load_path is not None:
            print("\n------ Phase 2: Load 2-loop checkpoint ------")
            p2_meta = _load_model_checkpoint(model_p2, p2_model_load_path, device)
            if isinstance(p2_meta, dict) and ("output_part" in p2_meta):
                ckpt_part = _normalize_output_part(p2_meta["output_part"])
                if ckpt_part != p2_output_part:
                    warn_msg = (
                        f"[Warn] P2 checkpoint output_part={ckpt_part} "
                        f"!= config phase2_output_part={p2_output_part}."
                    )
                    print(warn_msg)
                    p2_log(warn_msg)
            print(f"[loaded] P2 checkpoint from [{p2_model_load_path}]")
            p2_log(f"[P2] loaded checkpoint: {p2_model_load_path}")

            if isinstance(p2_meta, dict) and ("pred_scale" in p2_meta):
                p2_solution_scale = float(p2_meta["pred_scale"])
                msg = f"[P2] use pred_scale from checkpoint: {p2_solution_scale:g}"
                print(msg)
                p2_log(msg)
            elif "pred_scale" in p2_bundle_meta:
                p2_solution_scale = float(p2_bundle_meta["pred_scale"])
                msg = f"[P2] use pred_scale from eval-bundle: {p2_solution_scale:g}"
                print(msg)
                p2_log(msg)

            loaded_hist_p2 = _load_loss_history(p2_history_load_path)
            if loaded_hist_p2 is not None:
                hist_tot_p2, hist_cde_p2, hist_bc_p2 = loaded_hist_p2
                msg = f"[P2] loaded loss history from [{p2_history_load_path}]"
                print(msg)
                p2_log(msg)
            else:
                msg = "[P2] no loss history found for loaded checkpoint."
                print(msg)
                p2_log(msg)
        else:
            def cde_fixed_p2(model, a_builder, x_batch, n_basis, eps_val):
                return cde_residual_loss_fixed_eps_2loop(
                    model,
                    a_builder,
                    x_batch,
                    n_basis,
                    eps_val=eps_val,
                    output_part=p2_output_part,
                )

            (
                model_p2,
                hist_tot_p2,
                hist_cde_p2,
                hist_bc_p2,
                p2_train_info,
            ) = train_model_fixed_eps(
                model=model_p2,
                a_builder=a_builder_2loop,
                x_coll=x_coll_2loop,
                x_b_tensor=x_b_tensor_2loop,
                bc_target=bc_target_2loop_train,
                cde_loss_fixed_fn=cde_fixed_p2,
                bc_loss_fn=boundary_loss_cfg_p2,
                n_basis=n_basis_2loop,
                eps_val=cfg.eps_global,
                lr_init=phase2_learning_rate,
                warmup_len=phase2_warmup_epochs,
                total_epochs=phase2_epochs,
                lam1=cfg.lambda1,
                lam2=cfg.lambda2,
                cosine_min_lr=cfg.cosine_min_lr,
                print_every=cfg.print_every,
                phase_name="P2",
                log_fn=p2_log,
                use_grad_norm_probe=use_clip_gn_phase2,
                grad_clip_max_norm=grad_clip_max_norm_phase2,
            )

            if save_phase_artifacts:
                saved_paths_p2 = _save_phase_artifacts(
                    model=model_p2,
                    hist_tot=hist_tot_p2,
                    hist_cde=hist_cde_p2,
                    hist_bc=hist_bc_p2,
                    phase=2,
                    cy=cy_2loop,
                    eps_global=cfg.eps_global,
                    pred_scale=p2_solution_scale,
                    extra_meta={
                        "in_dim": 4,
                        "n_basis": int(n_basis_2loop),
                        "output_part": p2_output_part,
                        "phase0_model_ref": p0_artifacts["model_short"],
                        **p2_train_info,
                    },
                    phase_tag=p2_phase_tag,
                    output_part_tag=p2_output_part_tag,
                )
                print(
                    f"[saved] {os.path.basename(saved_paths_p2['model_abs'])} "
                    f"to [{os.path.dirname(saved_paths_p2['model_short'])}]"
                )
                print(
                    f"[saved] {os.path.basename(saved_paths_p2['history_abs'])} "
                    f"to [{os.path.dirname(saved_paths_p2['history_short'])}]"
                )
                p2_log(f"[P2] saved checkpoint: {saved_paths_p2['model_short']}")
                p2_log(f"[P2] saved loss history: {saved_paths_p2['history_short']}")

        if save_eval_bundle:
            saved_bundle_p2 = _save_eval_bundle(
                phase=2,
                cy=cy_2loop,
                eps_global=cfg.eps_global,
                x_coll=x_coll_2loop,
                x_b_tensor=x_b_tensor_2loop,
                bc_target=bc_target_2loop,
                pred_scale=p2_solution_scale,
                extra_meta={
                    "in_dim": int(x_coll_2loop.shape[1]),
                    "n_basis": int(n_basis_2loop),
                    "output_part": p2_output_part,
                    "phase0_model_ref": p0_artifacts["model_short"],
                    **p2_train_info,
                },
                phase_tag=p2_phase_tag,
                output_part_tag=p2_output_part_tag,
            )
            print(
                f"[saved] {os.path.basename(saved_bundle_p2['bundle_abs'])} "
                f"to [{os.path.dirname(saved_bundle_p2['bundle_short'])}]"
            )
            p2_log(f"[P2] saved eval-bundle: {saved_bundle_p2['bundle_short']}")

        if not train_two_phase_only:
            if (hist_tot_p2 is not None) and (hist_cde_p2 is not None) and (hist_bc_p2 is not None):
                plot_losses(
                    total_vals=hist_tot_p2,
                    cde_vals=hist_cde_p2,
                    bc_vals=hist_bc_p2,
                    title=rf"2-site 2-loop sunset, $c={cy_2loop}$, $\varepsilon={cfg.eps_global:g}$",
                    cy=cy_2loop,
                    save_dir="1_losses",
                    fname=f"P2_loss_all_eps_{eps_tag}{p2_loss_name_suffix}.png",
                    fname2=f"P2_loss_total_eps_{eps_tag}{p2_loss_name_suffix}.png",
                    phase=2,
                    phase_tag=p2_phase_tag,
                )
            else:
                msg = "[P2] skip plot_losses: loss history unavailable."
                print(msg)
                p2_log(msg)

            with torch.no_grad():
                f_target_2loop = compute_function_target_from_xcoll_2loop(
                    x_coll_2loop,
                    cy_val=cy_2loop,
                    eps_val=cfg.eps_global,
                    output_part=p2_output_part,
                    num_workers=postcalc_num_workers,
                    chunk_size=postcalc_chunk_size,
                    parallel_min_points=postcalc_parallel_min_points,
                    n_basis=n_basis_2loop,
                )

            post_train_check(
                model=model_p2,
                x_coll=x_coll_2loop,
                x_b_tensor=x_b_tensor_2loop,
                bc_target=bc_target_2loop,
                cy_val=cy_2loop,
                eps_global=cfg.eps_global,
                compute_function_target_from_xcoll=compute_function_target_from_xcoll_2loop,
                precomputed_true=f_target_2loop,
                phase_name="P2",
                pred_scale=p2_solution_scale,
                log_fn=p2_log,
                output_part=p2_output_part,
            )

            msg = f"[P2] output_part={p2_output_part_label}: skip plot_error_dis."
            print(msg)
            p2_log(msg)
        else:
            msg = "[P2] train_two_phase_only=True: skip plot_losses/post_train_check/plot_error."
            print(msg)
            p2_log(msg)

        phase2_elapsed = time.perf_counter() - phase2_t0
        phase2_msg = f"[P2] elapsed: {_format_elapsed(phase2_elapsed)} ({phase2_elapsed:.2f}s)"
        print(phase2_msg)
        p2_log(phase2_msg)
        p2_log_file.close()
        print(f"[saved] P2 log to [{p2_log_short}]")

    print("\n----- ALL TRAINING COMPLETE -----\n")


if __name__ == "__main__":
    main()
