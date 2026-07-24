import os
#import json
import numpy as np
import matplotlib.pyplot as plt

_RESULTS_ROOT_NAME = os.environ.get("COSMOPINNS_RESULTS_ROOT", "results").strip() or "results"

def _cy_to_folder_name(cy) -> str:
    try:
        cyf = float(cy)
    except (TypeError, ValueError):
        return "c_misc"

    s = f"{cyf:.6g}"
    if s.startswith("-"):
        s = "m" + s[1:]
    else:
        s = "p" + s
    s = s.replace(".", "_")
    return "c_" + s


def set_results_root_name(root_name: str):
    global _RESULTS_ROOT_NAME
    s = str(root_name).strip()
    if not s:
        raise ValueError("results root name cannot be empty")
    _RESULTS_ROOT_NAME = s


def get_results_root_name() -> str:
    return _RESULTS_ROOT_NAME


def get_nested_save_dir(
    save_dir: str,
    cy: float,
    phase: int = 0,
    phase_tag: str = None,
    create_dir: bool = True,
):
    here = os.path.dirname(os.path.abspath(__file__))   # .../plot_tools
    project_root = os.path.dirname(here)                # .../
    results_root_abs = os.path.join(project_root, _RESULTS_ROOT_NAME)

    cy_folder = _cy_to_folder_name(cy)
    if phase_tag is None:
        phase_prefix = f"P{int(phase)}"
    else:
        phase_prefix = str(phase_tag).strip()
        if not phase_prefix:
            raise ValueError("phase_tag cannot be empty")
    phase_folder = f"{phase_prefix}_{cy_folder}"

    abs_dir = os.path.join(results_root_abs, phase_folder, save_dir)
    if create_dir:
        os.makedirs(abs_dir, exist_ok=True)

    short = os.path.join(_RESULTS_ROOT_NAME, phase_folder, save_dir)

    return abs_dir, short


# Matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    "axes.linewidth": 1.2,
    "font.size": 12,
    "legend.frameon": False,
    "legend.handlelength": 2.8,
    "legend.fontsize": 11,
})
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"


def _finite_positive(arr):
    a = np.asarray(arr, dtype=float).reshape(-1)
    return a[np.isfinite(a) & (a > 0.0)]


def _safe_log_ylim(*series, default_low=1e-12, default_high=1.0):
    vals = np.concatenate([_finite_positive(s) for s in series if np.size(s) > 0]) if series else np.array([])
    if vals.size == 0:
        return default_low, default_high

    low = float(np.min(vals))
    high = float(np.max(vals))

    low = max(low * 0.8, default_low)
    high = max(high * 1.2, low * 10.0)

    if (not np.isfinite(low)) or (not np.isfinite(high)) or (high <= low):
        return default_low, default_high
    return low, high


def plot_losses(
    total_vals,
    cde_vals,
    bc_vals,
    *,
    title,
    cy,
    save_dir="1_losses",
    fname="P0_loss_all.png",
    fname2="P0_loss_total.png",
    phase=0,
    phase_tag=None,
):

    save_dir_abs, save_dir_short = get_nested_save_dir(save_dir, cy, phase, phase_tag=phase_tag)

    # ----------------------------------------------------
    #                 Plot: Total + CDE + BC
    # ----------------------------------------------------
    plt.figure(figsize=(6, 4))
    x = np.arange(1, len(total_vals) + 1)

    colors = {
        "total": "tab:blue",
        "cde": "tab:orange",
        "bc": "tab:green"
    }

    plt.semilogy(x, total_vals, "-", color=colors["total"], label="Total")
    plt.semilogy(x, cde_vals, "-", color=colors["cde"], alpha=0.8, label="CDE")
    plt.semilogy(x, bc_vals, "-", color=colors["bc"], alpha=0.7, label="BC")

    plt.legend(fontsize=8, frameon=True, facecolor="white",
               edgecolor="black", framealpha=1.0)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)

    y_low, y_high = _safe_log_ylim(total_vals, cde_vals, bc_vals)
    plt.ylim(y_low, y_high)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_abs, fname), dpi=200, bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------
    #                 Plot: Total only
    # ----------------------------------------------------
    plt.figure(figsize=(6, 4))

    plt.semilogy(x, total_vals, "-", color=colors["total"], label="Total")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)

    y_low, y_high = _safe_log_ylim(total_vals)
    plt.ylim(y_low, y_high)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir_abs, fname2), dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[saved] {fname} to [{save_dir_short}]")
    print(f"[saved] {fname2} to [{save_dir_short}]")
