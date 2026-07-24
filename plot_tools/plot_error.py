import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator, FuncFormatter

from plot_tools.plot_losses import get_nested_save_dir


def plot_error_dis(
    model,
    x_coll,
    function_target,
    phase_name="P0",
    eps_value=None,
    cy=None,
    cy_loop=None,
    pred_scale=1.0,
    plot_vector_l2_hist=True,
    phase_tag=None,
):
    """
    Draw:
      (1) Re/Im scatter per basis
      (2) Abs diff hist per basis
      (3a) Rel diff hist per basis
      (3b) Mean rel diff hist across basis
      (4) Optional vector-L2 hist across all basis
    """
    phase_name_s = str(phase_name).strip()
    phase_name_u = phase_name_s.upper()
    if phase_name_u.startswith("P") and phase_name_u[1:].isdigit():
        phase = int(phase_name_u[1:])
    else:
        raise ValueError(
            f"Unsupported phase_name={phase_name!r}. Expected pattern like 'P0', 'P1', 'P2'."
        )

    phase_label = f"P{phase}"
    cy_for_folder = cy if phase == 0 else cy_loop
    if cy_for_folder is None:
        raise ValueError(
            f"Please provide cy (for {phase_label}) or cy_loop (for transfer phases) for folder naming."
        )

    save_abs, save_short = get_nested_save_dir(
        "2_true_pred",
        cy_for_folder,
        phase=phase,
        phase_tag=phase_tag,
    )
    os.makedirs(save_abs, exist_ok=True)

    model.eval()
    with torch.no_grad():
        pred = model(x_coll) / float(pred_scale)
        true = function_target

    n_basis = pred.shape[1] // 2
    pred_re_np = pred[:, :n_basis].detach().cpu().numpy()
    pred_im_np = pred[:, n_basis:].detach().cpu().numpy()
    true_re_np = true[:, :n_basis].detach().cpu().numpy()
    true_im_np = true[:, n_basis:].detach().cpu().numpy()

    x_np = x_coll.detach().cpu().numpy()
    if eps_value is not None:
        eps_val = float(eps_value)
        eps_desc = f"{eps_val:.6g}"
        if np.isclose(eps_val, 0.0, rtol=0.0, atol=1e-12):
            s = "0"
        else:
            mag = f"{abs(eps_val):.6g}".replace(".", "_")
            s = f"m{mag}" if eps_val < 0 else f"p{mag}"
        suffix = f"eps_{s}"
    elif x_np.shape[1] >= 3:
        eps_vals = x_np[:, -1]
        eps_min = float(eps_vals.min())
        eps_max = float(eps_vals.max())
        if np.isclose(eps_min, eps_max, rtol=0.0, atol=1e-12):
            eps_desc = f"{eps_min:.6g}"
        else:
            eps_desc = f"[{eps_min:.3g}, {eps_max:.3g}]"
        suffix = "eps_data"
    else:
        raise ValueError("eps_value must be provided when x_coll has no eps column.")

    func_labels = [f"I{i + 1}" for i in range(n_basis)]
    func_labels_tex = [rf"$\mathcal{{I}}_{{{i + 1}}}$" for i in range(n_basis)]

    def _finite_1d(vals):
        arr = np.asarray(vals, dtype=float).reshape(-1)
        return arr[np.isfinite(arr)]

    def _format_log_tick_pow10(x, _pos=None):
        if not np.isfinite(x) or x <= 0.0:
            return ""
        exp = int(np.round(np.log10(x)))
        if not np.isclose(x, 10.0 ** exp, rtol=1e-10, atol=0.0):
            return ""
        return rf"$10^{{{exp}}}$"

    def _auto_log_xlim(
        *arrays,
        q_lo=0.001,
        q_hi=0.999,
        pad_decades=1.0,
    ):
        pos_parts = []
        for vals in arrays:
            clean = _finite_1d(vals)
            pos = clean[clean > 0.0]
            if pos.size > 0:
                pos_parts.append(pos)

        if not pos_parts:
            return None

        all_pos = np.concatenate(pos_parts)
        lo = float(np.nanquantile(all_pos, q_lo))
        hi = float(np.nanquantile(all_pos, q_hi))

        if (not np.isfinite(lo)) or lo <= 0.0:
            lo = float(np.nanmin(all_pos))
        if (not np.isfinite(hi)) or hi <= 0.0:
            hi = float(np.nanmax(all_pos))

        lo = max(lo, np.finfo(float).tiny)
        if hi <= lo:
            hi = lo * 10.0

        # Keep at least one full decade margin on both sides.
        scale = 10.0 ** float(pad_decades)
        lo_pad = lo / scale
        hi_pad = hi * scale

        # Snap to decade boundaries for clean, complete tick labels.
        lo_exp = int(np.floor(np.log10(max(lo_pad, np.finfo(float).tiny))))
        hi_exp = int(np.ceil(np.log10(max(hi_pad, np.finfo(float).tiny))))
        if hi_exp <= lo_exp:
            hi_exp = lo_exp + 1

        return float(10.0 ** lo_exp), float(10.0 ** hi_exp)

    def _hist_percent(ax, vals, bins, color, label, alpha=1.0):
        clean = _finite_1d(vals)
        if clean.size == 0:
            return False
        w = np.ones_like(clean) * (100.0 / max(len(clean), 1))
        ax.hist(clean, bins=bins, weights=w, histtype="step", lw=1.8, color=color, alpha=alpha, label=label)
        return True

    def _build_adaptive_log_bins(bins, *arrays, default_n_edges=60):
        pos_parts = []
        for vals in arrays:
            clean = _finite_1d(vals)
            pos = clean[clean > 0.0]
            if pos.size > 0:
                pos_parts.append(pos)

        if not pos_parts:
            return bins

        all_pos = np.concatenate(pos_parts)
        d_lo = float(np.nanmin(all_pos))
        d_hi = float(np.nanmax(all_pos))
        if (not np.isfinite(d_lo)) or (not np.isfinite(d_hi)) or d_lo <= 0.0:
            return bins

        n_edges = int(default_n_edges)
        b_lo = None
        b_hi = None
        if np.ndim(bins) == 1:
            bins_arr = np.asarray(bins, dtype=float).reshape(-1)
            if bins_arr.size >= 3 and np.all(np.isfinite(bins_arr)):
                n_edges = int(bins_arr.size)
                b_lo = float(bins_arr[0])
                b_hi = float(bins_arr[-1])
                if (d_lo >= b_lo) and (d_hi <= b_hi):
                    return bins_arr

        xlim = _auto_log_xlim(*arrays)
        if xlim is None:
            return bins
        xlo, xhi = xlim
        xlo = max(float(xlo), np.finfo(float).tiny)
        xhi = max(float(xhi), xlo * 10.0)
        return np.logspace(np.log10(xlo), np.log10(xhi), max(n_edges, 3))

    def _row_mean_ignore_nonfinite(arr2d):
        arr = np.asarray(arr2d, dtype=float)
        finite = np.isfinite(arr)
        counts = finite.sum(axis=1)
        sums = np.where(finite, arr, 0.0).sum(axis=1)
        out = np.full(arr.shape[0], np.nan, dtype=float)
        valid = counts > 0
        out[valid] = sums[valid] / counts[valid]
        return out

    def draw_hist(re_vals, im_vals, bins, xlabel, ylabel, title_j, filename):
        fig, ax = plt.subplots(figsize=(6, 4))
        bins_eff = _build_adaptive_log_bins(bins, re_vals, im_vals)
        has_re = _hist_percent(ax, re_vals, bins_eff, "tab:blue", "Re", alpha=1.0)
        has_im = _hist_percent(ax, im_vals, bins_eff, "tab:red", "Im", alpha=0.8)

        if not has_re and not has_im:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        else:
            legend_handles = [
                Line2D([0], [0], color="tab:blue", lw=1.8, label="Re"),
                Line2D([0], [0], color="tab:red", lw=1.8, label="Im"),
            ]
            ax.legend(
                handles=legend_handles,
                fontsize=9,
                frameon=True,
                facecolor="white",
                edgecolor="black",
                framealpha=1.0,
            )

        ax.set_xscale("log")
        xlim = _auto_log_xlim(re_vals, im_vals)
        if xlim is not None:
            xlo, xhi = xlim
            ax.set_xlim(xlo / 1.35, xhi)
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            ax.xaxis.set_major_formatter(FuncFormatter(_format_log_tick_pow10))
        if has_re or has_im:
            ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="x", which="major", pad=7)
        ax.tick_params(axis="y", which="major", pad=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title_j)
        fig.tight_layout()
        fig.savefig(filename, dpi=200)
        plt.close(fig)

    def draw_hist_single(vals, bins, xlabel, ylabel, title_j, filename, color, label):
        fig, ax = plt.subplots(figsize=(6, 4))
        bins_eff = _build_adaptive_log_bins(bins, vals)
        has_data = _hist_percent(ax, vals, bins_eff, color, label, alpha=1.0)

        if not has_data:
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        else:
            legend_handles = [Line2D([0], [0], color=color, lw=1.8, label=label)]
            ax.legend(
                handles=legend_handles,
                fontsize=9,
                frameon=True,
                facecolor="white",
                edgecolor="black",
                framealpha=1.0,
            )

        ax.set_xscale("log")
        xlim = _auto_log_xlim(vals)
        if xlim is not None:
            xlo, xhi = xlim
            ax.set_xlim(xlo / 1.35, xhi)
            ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,)))
            ax.xaxis.set_major_formatter(FuncFormatter(_format_log_tick_pow10))
        if has_data:
            ax.set_ylim(bottom=0.0)
        ax.tick_params(axis="x", which="major", pad=7)
        ax.tick_params(axis="y", which="major", pad=7)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title_j)
        fig.tight_layout()
        fig.savefig(filename, dpi=200)
        plt.close(fig)

    # (1) Re/Im scatter
    for j in range(n_basis):
        fig, ax = plt.subplots(figsize=(6, 4))

        m_true = np.isfinite(true_re_np[:, j]) & np.isfinite(true_im_np[:, j])
        m_pred = np.isfinite(pred_re_np[:, j]) & np.isfinite(pred_im_np[:, j])

        if m_true.any():
            ax.scatter(
                true_re_np[m_true, j],
                true_im_np[m_true, j],
                s=10,
                c="tab:blue",
                label="True",
                marker="o",
                zorder=3,
            )
        if m_pred.any():
            ax.scatter(
                pred_re_np[m_pred, j],
                pred_im_np[m_pred, j],
                s=10,
                c="tab:red",
                label="Pred",
                marker="x",
                alpha=0.65,
                linewidths=0.8,
                zorder=2,
            )
        if (not m_true.any()) and (not m_pred.any()):
            ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)

        title_j = rf"{func_labels_tex[j]}($\vec{{x}}$, {eps_desc})"

        ax.set_xlabel("Re")
        ax.set_ylabel("Im")
        ax.set_title(title_j)

        ims = []
        if m_true.any():
            ims.append(true_im_np[m_true, j])
        if m_pred.any():
            ims.append(pred_im_np[m_pred, j])
        if ims:
            im_all = np.concatenate(ims)
            base = np.nanpercentile(np.abs(im_all), 99)
            base = max(base, 1e-12)
            widen = 20.0
            ax.set_ylim(-widen * base, widen * base)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        yfmt = ScalarFormatter(useMathText=True)
        yfmt.set_scientific(True)
        yfmt.set_powerlimits((0, 0))
        yfmt.set_useOffset(True)
        ax.yaxis.set_major_formatter(yfmt)
        ax.yaxis.get_offset_text().set_fontsize(9)

        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        xfmt = ScalarFormatter(useMathText=True)
        xfmt.set_scientific(True)
        xfmt.set_powerlimits((0, 0))
        xfmt.set_useOffset(True)
        ax.xaxis.set_major_formatter(xfmt)
        ax.xaxis.get_offset_text().set_fontsize(9)

        ax.axhline(0, color="k", lw=1, alpha=0.2, zorder=1)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("auto")
        ax.legend(
            loc="upper right",
            fontsize=9,
            frameon=True,
            facecolor="white",
            edgecolor="black",
            framealpha=1.0,
        )

        fig.tight_layout()
        outname = os.path.join(save_abs, f"{phase_name}_{suffix}_ReIm_{func_labels[j]}.png")
        fig.savefig(outname, dpi=200)
        plt.close(fig)
        print(f"[saved] {os.path.basename(outname)} to [{save_short}]")

    # (2) absolute difference hist
    bins_abs = np.logspace(-8, 0, 60)
    for j in range(n_basis):
        abs_re = np.abs(pred_re_np[:, j] - true_re_np[:, j])
        abs_im = np.abs(pred_im_np[:, j] - true_im_np[:, j])

        title_j = rf"{phase_label}: {func_labels_tex[j]} ({eps_desc}, $c={cy_for_folder}$)"

        outname = os.path.join(save_abs, f"{phase_name}_{suffix}_AbsDiff_{func_labels[j]}.png")
        draw_hist(
            abs_re,
            abs_im,
            bins_abs,
            xlabel=rf"$\Delta_{{{j + 1}}}$",
            ylabel=r"$\%$",
            title_j=title_j,
            filename=outname,
        )
        print(f"[saved] {os.path.basename(outname)} to [{save_short}]")

    # (3a) relative difference hist
    bins_rel = np.logspace(-8, 1, 60)
    rel_tol = 1e-8
    rel_diff_re = np.abs(pred_re_np - true_re_np)
    rel_diff_im = np.abs(pred_im_np - true_im_np)
    rel_den_re = np.maximum(np.abs(true_re_np), rel_tol)
    rel_den_im = np.maximum(np.abs(true_im_np), rel_tol)
    rel_re_mat = np.maximum(rel_diff_re / rel_den_re, 1e-15)
    rel_im_mat = np.maximum(rel_diff_im / rel_den_im, 1e-15)

    for j in range(n_basis):
        rel_re = rel_re_mat[:, j]
        rel_im = rel_im_mat[:, j]

        title_j = rf"{phase_label}: {func_labels_tex[j]} ({eps_desc}, $c={cy_for_folder}$)"

        outname = os.path.join(save_abs, f"{phase_name}_{suffix}_RelDiff_{func_labels[j]}.png")
        draw_hist(
            rel_re,
            rel_im,
            bins_rel,
            xlabel=rf"$\Delta^{{rel}}_{{{j + 1}}}$",
            ylabel=r"$\%$",
            title_j=title_j,
            filename=outname,
        )
        print(f"[saved] {os.path.basename(outname)} to [{save_short}]")

    # (3b) relative difference hist average
    rel_re_avg = _row_mean_ignore_nonfinite(rel_re_mat)
    rel_im_avg = _row_mean_ignore_nonfinite(rel_im_mat)

    title_avg = rf"{phase_label}: $\overline{{\Delta^{{rel}}}}$ ({eps_desc}, $c={cy_for_folder}$)"

    outname = os.path.join(save_abs, f"{phase_name}_{suffix}_RelDiff_Ave.png")
    draw_hist(
        rel_re_avg,
        rel_im_avg,
        bins_rel,
        xlabel=r"$\overline{\Delta^{rel}}$",
        ylabel=r"$\%$",
        title_j=title_avg,
        filename=outname,
    )
    print(f"[saved] {os.path.basename(outname)} to [{save_short}]")

    # (4) vector L1 & L2 hist across all basis
    # Keep the existing flag name for backward compatibility.
    if plot_vector_l2_hist:
        save_abs_vec, save_short_vec = get_nested_save_dir(
            "3_vecL1_L2",
            cy_for_folder,
            phase=phase,
            phase_tag=phase_tag,
        )
        os.makedirs(save_abs_vec, exist_ok=True)

        diff_re = pred_re_np - true_re_np
        diff_im = pred_im_np - true_im_np
        eps_vec = 1e-12

        bins_norm_abs = np.logspace(-8, 1, 60)
        bins_norm_rel = np.logspace(-8, 1, 60)

        title_l1 = rf"P{phase}: $\vec{{I}}$ L1 error"
        title_l2 = rf"P{phase}: $\vec{{I}}$ L2 error"

        # L1 norms across basis, per sample
        l1_abs_re = np.abs(diff_re).sum(axis=1)
        l1_abs_im = np.abs(diff_im).sum(axis=1)
        den_l1 = np.maximum(np.abs(true_re_np).sum(axis=1), eps_vec)
        l1_rel_re = np.maximum(l1_abs_re / den_l1, 1e-15)
        l1_rel_im = np.maximum(l1_abs_im / den_l1, 1e-15)

        out_abs_l1 = os.path.join(save_abs_vec, f"{phase_name}_{suffix}_AbsL1.png")
        draw_hist(
            l1_abs_re,
            l1_abs_im,
            bins_norm_abs,
            xlabel=r"$\|\Delta\|_1$",
            ylabel=r"$\%$",
            title_j=title_l1 + r" | abs",
            filename=out_abs_l1,
        )
        print(f"[saved] {os.path.basename(out_abs_l1)} to [{save_short_vec}]")
        draw_hist_single(
            l1_abs_re,
            bins_norm_abs,
            xlabel=r"$\|\Delta\|_1$",
            ylabel=r"$\%$",
            title_j=title_l1 + r" | abs (Re)",
            filename=os.path.join(save_abs_vec, f"{phase_name}_{suffix}_AbsL1_Re.png"),
            color="tab:blue",
            label="Re",
        )
        draw_hist_single(
            l1_abs_im,
            bins_norm_abs,
            xlabel=r"$\|\Delta\|_1$",
            ylabel=r"$\%$",
            title_j=title_l1 + r" | abs (Im)",
            filename=os.path.join(save_abs_vec, f"{phase_name}_{suffix}_AbsL1_Im.png"),
            color="tab:red",
            label="Im",
        )

        out_rel_l1 = os.path.join(save_abs_vec, f"{phase_name}_{suffix}_RelL1.png")
        draw_hist(
            l1_rel_re,
            l1_rel_im,
            bins_norm_rel,
            xlabel=r"$\|\Delta\|_1 / \|\mathrm{Re}(I)\|_1$",
            ylabel=r"$\%$",
            title_j=title_l1 + r" | rel",
            filename=out_rel_l1,
        )
        print(f"[saved] {os.path.basename(out_rel_l1)} to [{save_short_vec}]")
        draw_hist_single(
            l1_rel_re,
            bins_norm_rel,
            xlabel=r"$\|\Delta\|_1 / \|\mathrm{Re}(I)\|_1$",
            ylabel=r"$\%$",
            title_j=title_l1 + r" | rel (Re)",
            filename=os.path.join(save_abs_vec, f"{phase_name}_{suffix}_RelL1_Re.png"),
            color="tab:blue",
            label="Re",
        )
        draw_hist_single(
            l1_rel_im,
            bins_norm_rel,
            xlabel=r"$\|\Delta\|_1 / \|\mathrm{Re}(I)\|_1$",
            ylabel=r"$\%$",
            title_j=title_l1 + r" | rel (Im)",
            filename=os.path.join(save_abs_vec, f"{phase_name}_{suffix}_RelL1_Im.png"),
            color="tab:red",
            label="Im",
        )

        # L2 norms across basis, per sample
        l2_abs_re = np.linalg.norm(diff_re, axis=1)
        l2_abs_im = np.linalg.norm(diff_im, axis=1)
        den_l2 = np.maximum(np.linalg.norm(true_re_np, axis=1), eps_vec)
        l2_rel_re = np.maximum(l2_abs_re / den_l2, 1e-15)
        l2_rel_im = np.maximum(l2_abs_im / den_l2, 1e-15)

        out_abs_l2 = os.path.join(save_abs_vec, f"{phase_name}_{suffix}_AbsL2.png")
        draw_hist(
            l2_abs_re,
            l2_abs_im,
            bins_norm_abs,
            xlabel=r"$\|\Delta\|_2$",
            ylabel=r"$\%$",
            title_j=title_l2 + r" | abs",
            filename=out_abs_l2,
        )
        print(f"[saved] {os.path.basename(out_abs_l2)} to [{save_short_vec}]")
        draw_hist_single(
            l2_abs_re,
            bins_norm_abs,
            xlabel=r"$\|\Delta\|_2$",
            ylabel=r"$\%$",
            title_j=title_l2 + r" | abs (Re)",
            filename=os.path.join(save_abs_vec, f"{phase_name}_{suffix}_AbsL2_Re.png"),
            color="tab:blue",
            label="Re",
        )
        draw_hist_single(
            l2_abs_im,
            bins_norm_abs,
            xlabel=r"$\|\Delta\|_2$",
            ylabel=r"$\%$",
            title_j=title_l2 + r" | abs (Im)",
            filename=os.path.join(save_abs_vec, f"{phase_name}_{suffix}_AbsL2_Im.png"),
            color="tab:red",
            label="Im",
        )

        out_rel_l2 = os.path.join(save_abs_vec, f"{phase_name}_{suffix}_RelL2.png")
        draw_hist(
            l2_rel_re,
            l2_rel_im,
            bins_norm_rel,
            xlabel=r"$\|\Delta\|_2 / \|\mathrm{Re}(I)\|_2$",
            ylabel=r"$\%$",
            title_j=title_l2 + r" | rel",
            filename=out_rel_l2,
        )
        print(f"[saved] {os.path.basename(out_rel_l2)} to [{save_short_vec}]")
        draw_hist_single(
            l2_rel_re,
            bins_norm_rel,
            xlabel=r"$\|\Delta\|_2 / \|\mathrm{Re}(I)\|_2$",
            ylabel=r"$\%$",
            title_j=title_l2 + r" | rel (Re)",
            filename=os.path.join(save_abs_vec, f"{phase_name}_{suffix}_RelL2_Re.png"),
            color="tab:blue",
            label="Re",
        )
        draw_hist_single(
            l2_rel_im,
            bins_norm_rel,
            xlabel=r"$\|\Delta\|_2 / \|\mathrm{Re}(I)\|_2$",
            ylabel=r"$\%$",
            title_j=title_l2 + r" | rel (Im)",
            filename=os.path.join(save_abs_vec, f"{phase_name}_{suffix}_RelL2_Im.png"),
            color="tab:red",
            label="Im",
        )
