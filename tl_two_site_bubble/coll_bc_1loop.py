import math
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

try:
    from scipy.special import spence as _scipy_spence
except Exception:
    _scipy_spence = None

from two_site_chain.sol_chain import eps_to_n_int

from tl_two_site_bubble.sol_1loop import (
    I1_fin,
    I2_fin,
    I3_fin,
    I4_fin,
    I5_fin,
    I6_fin,
    I7_fin,
    I8_fin,
    I9_fin,
    I10_fin,
)


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


def _complex_to_output_channels(function_complex: np.ndarray, output_part="both") -> np.ndarray:
    part = _normalize_output_part(output_part)
    if part == "both":
        return np.concatenate([np.real(function_complex), np.imag(function_complex)], axis=1)
    if part == "re":
        return np.real(function_complex)
    return np.imag(function_complex)


_EPS_TOL = 1e-12


def _build_ws_tensor(x_coll: torch.Tensor, *, cy_val: float):
    if x_coll.ndim != 2:
        raise ValueError(f"x_coll must be 2D, got shape {tuple(x_coll.shape)}")
    if int(x_coll.shape[1]) not in (3, 4):
        raise ValueError(f"x_coll must have 3 or 4 columns, got shape {tuple(x_coll.shape)}")

    x_core = x_coll[:, :3].to(dtype=torch.float64)
    x1, x2, y1 = x_core.unbind(dim=1)
    c = torch.tensor(float(cy_val), dtype=torch.float64, device=x_coll.device)

    w1 = x1 + y1 + c
    w2 = x2 + y1 + c
    w3 = x1 + y1 - c
    w4 = x2 + y1 - c
    w5 = x1 - y1 + c
    w6 = x2 - y1 + c
    w7 = x1 - y1 - c
    w8 = x2 - y1 - c
    w9 = x1 + x2 + 2.0 * y1
    w10 = x1 + x2 + 2.0 * c
    w11 = x1 + x2
    return w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11


def _polylog2_complex_np(z):
    if _scipy_spence is None:
        raise RuntimeError("scipy.special.spence is unavailable.")
    z_arr = np.asarray(z, dtype=np.complex128)
    return _scipy_spence(1.0 - z_arr)


def _torch_re_to_output_channels(re_vals: torch.Tensor, output_part="both") -> torch.Tensor:
    part = _normalize_output_part(output_part)
    if part == "both":
        return torch.cat([re_vals, torch.zeros_like(re_vals)], dim=1)
    if part == "re":
        return re_vals
    return torch.zeros_like(re_vals)


def _fast_p1_target_eps0(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    output_part: str,
) -> torch.Tensor:
    if _scipy_spence is None:
        raise RuntimeError("SciPy is required for the eps=0 1-loop fast path.")

    x_np = x_coll[:, :3].detach().cpu().numpy().astype(np.float64, copy=False)
    x1 = x_np[:, 0]
    x2 = x_np[:, 1]
    y1 = x_np[:, 2]
    c = float(cy_val)

    w1 = x1 + y1 + c
    w2 = x2 + y1 + c
    w3 = x1 + y1 - c
    w4 = x2 + y1 - c
    w5 = x1 - y1 + c
    w6 = x2 - y1 + c
    w7 = x1 - y1 - c
    w8 = x2 - y1 - c
    w9 = x1 + x2 + 2.0 * y1
    w10 = x1 + x2 + 2.0 * c
    w11 = x1 + x2

    w1c = np.asarray(w1, dtype=np.complex128)
    w2c = np.asarray(w2, dtype=np.complex128)
    w3c = np.asarray(w3, dtype=np.complex128)
    w4c = np.asarray(w4, dtype=np.complex128)
    w5c = np.asarray(w5, dtype=np.complex128)
    w6c = np.asarray(w6, dtype=np.complex128)
    w7c = np.asarray(w7, dtype=np.complex128)
    w8c = np.asarray(w8, dtype=np.complex128)
    w9c = np.asarray(w9, dtype=np.complex128)
    w10c = np.asarray(w10, dtype=np.complex128)
    w11c = np.asarray(w11, dtype=np.complex128)

    i1 = (
        -(np.pi ** 2) / 6.0
        + _polylog2_complex_np(w3c / w1c)
        + _polylog2_complex_np(w4c / w2c)
        - _polylog2_complex_np((w3c * w4c) / (w1c * w2c))
        + _polylog2_complex_np(w5c / w1c)
        + _polylog2_complex_np(w6c / w2c)
        - _polylog2_complex_np((w5c * w6c) / (w1c * w2c))
        - _polylog2_complex_np(w7c / w1c)
        - _polylog2_complex_np(w8c / w2c)
        + _polylog2_complex_np((w7c * w8c) / (w1c * w2c))
    )
    i2 = np.log(w9c / w2c)
    i3 = np.log(w10c / w2c)
    i4 = -np.log(w11c / w2c)
    i5 = np.log(w9c / w1c)
    i6 = np.log(w10c / w1c)
    i7 = -np.log(w11c / w1c)
    ones = np.ones_like(w1c)
    i8 = ones
    i9 = ones
    i10 = -ones

    function_complex = np.stack((i1, i2, i3, i4, i5, i6, i7, i8, i9, i10), axis=1)
    function_concat = _complex_to_output_channels(function_complex, output_part=output_part)
    return torch.tensor(function_concat, dtype=torch.float32, device=x_coll.device)


def _fast_p1_target_negative_int(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    eps_val: float,
    output_part: str,
) -> torch.Tensor:
    n = eps_to_n_int(float(eps_val))
    if n is None:
        raise ValueError(f"Fast 1-loop negative-integer target path expects eps=-n, got eps={eps_val}.")

    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11 = _build_ws_tensor(x_coll, cy_val=cy_val)
    comb_2n_n = float(math.comb(2 * n, n))

    def _int_series_sum(w_main, w_main2):
        acc = torch.zeros_like(w_main)
        for k in range(n):
            acc = acc + float(math.comb(n + k, k)) * torch.pow(w_main, k) * torch.pow(w_main2, n - k - 1)
        return acc

    s29 = _int_series_sum(w2, w9)
    s19 = _int_series_sum(w1, w9)
    s210 = _int_series_sum(w2, w10)
    s110 = _int_series_sum(w1, w10)
    s211 = _int_series_sum(w2, w11)
    s111 = _int_series_sum(w1, w11)

    i2 = -(w3 / (torch.pow(w2, n) * torch.pow(w9, 2 * n))) * s29
    i5 = -(w4 / (torch.pow(w1, n) * torch.pow(w9, 2 * n))) * s19
    i3 = -(w5 / (torch.pow(w2, n) * torch.pow(w10, 2 * n))) * s210
    i6 = -(w6 / (torch.pow(w1, n) * torch.pow(w10, 2 * n))) * s110
    i4 = (w7 / (torch.pow(w2, n) * torch.pow(w11, 2 * n))) * s211
    i7 = (w8 / (torch.pow(w1, n) * torch.pow(w11, 2 * n))) * s111
    i8 = comb_2n_n / torch.pow(w9, 2 * n)
    i9 = comb_2n_n / torch.pow(w10, 2 * n)
    i10 = -comb_2n_n / torch.pow(w11, 2 * n)
    i1 = (
        1.0 / (torch.pow(w1, n) * torch.pow(w2, n))
        + (i2 + i5 - i8)
        + (i3 + i6 - i9)
        + (i4 + i7 - i10)
    )

    re_vals = torch.stack((i1, i2, i3, i4, i5, i6, i7, i8, i9, i10), dim=1)
    return _torch_re_to_output_channels(re_vals, output_part=output_part).to(
        dtype=torch.float32,
        device=x_coll.device,
    )


def _eval_1loop_chunk(x_chunk, cy_val):
    x_arr = np.asarray(x_chunk, dtype=float)
    out = np.empty((x_arr.shape[0], 10), dtype=complex)
    for i, (x1, x2, y1, eps) in enumerate(x_arr):
        x1f, x2f, y1f, epsf = float(x1), float(x2), float(y1), float(eps)
        out[i, 0] = I1_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 1] = I2_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 2] = I3_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 3] = I4_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 4] = I5_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 5] = I6_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 6] = I7_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 7] = I8_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 8] = I9_fin(x1f, x2f, y1f, epsf, cy_val)
        out[i, 9] = I10_fin(x1f, x2f, y1f, epsf, cy_val)
    return out


def compute_boundary_values_rescaled_1loop(
    x_quadruplet,
    cy_val,
    *,
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
):
    """
    Compute analytic values for multiple (x1, x2, y1, eps) points.
    Input shape: (N,4)
    Output: complex array (N,10) for [I1, ..., I10]
    """
    x_all = np.asarray(x_quadruplet, dtype=float)
    if x_all.ndim == 1:
        if x_all.shape[0] != 4:
            raise ValueError("single point must be shape (4,) = (x1,x2,y1,eps)")
        x_all = x_all.reshape(1, 4)

    if x_all.shape[1] != 4:
        raise ValueError(f"x_quadruplet must have 4 columns (x1,x2,y1,eps), got shape {x_all.shape}")

    n_pts = int(x_all.shape[0])
    nw = max(int(num_workers), 1)
    cs = max(int(chunk_size), 1)
    nmin = max(int(parallel_min_points), 1)

    if (nw == 1) or (n_pts < nmin):
        return _eval_1loop_chunk(x_all, cy_val)

    chunks = [x_all[i:i + cs] for i in range(0, n_pts, cs)]
    cy_vals = [float(cy_val)] * len(chunks)
    try:
        with ProcessPoolExecutor(max_workers=nw) as pool:
            parts = list(pool.map(_eval_1loop_chunk, chunks, cy_vals))
        return np.concatenate(parts, axis=0)
    except (PermissionError, OSError):
        print("[warn] 1-loop target parallel disabled by runtime; fallback to single-process.")
        return _eval_1loop_chunk(x_all, cy_val)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def compute_function_target_from_xcoll_1loop(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    eps_val: float,
    output_part="both",
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
) -> torch.Tensor:
    output_part = _normalize_output_part(output_part)
    if isinstance(x_coll, torch.Tensor):
        x_tensor = x_coll
    else:
        x_tensor = torch.as_tensor(x_coll, dtype=torch.float32)

    if abs(float(eps_val)) < _EPS_TOL and _scipy_spence is not None:
        return _fast_p1_target_eps0(
            x_tensor,
            cy_val=float(cy_val),
            output_part=output_part,
        )

    if eps_to_n_int(float(eps_val)) is not None:
        return _fast_p1_target_negative_int(
            x_tensor,
            cy_val=float(cy_val),
            eps_val=float(eps_val),
            output_part=output_part,
        )

    x_np = to_numpy(x_tensor)

    if x_np.ndim != 2:
        raise ValueError(f"x_coll must be 2D tensor/array, got shape {x_np.shape}")

    if x_np.shape[1] == 3:
        quad_np = np.concatenate(
            [x_np[:, :3], np.full((x_np.shape[0], 1), float(eps_val))],
            axis=1,
        )
    elif x_np.shape[1] == 4:
        quad_np = x_np[:, :4]
    else:
        raise ValueError(f"x_coll must have 3 or 4 columns, got {x_np.shape}")

    function_complex = compute_boundary_values_rescaled_1loop(
        quad_np,
        cy_val,
        num_workers=num_workers,
        chunk_size=chunk_size,
        parallel_min_points=parallel_min_points,
    )
    function_concat = _complex_to_output_channels(function_complex, output_part=output_part)

    return torch.tensor(function_concat, dtype=torch.float32, device=x_tensor.device)


def build_inputs_and_boundary_1loop(
    n_coll_pts,
    x1_lo,
    x1_hi,
    x2_lo,
    x2_hi,
    y1_lo,
    y1_hi,
    cy_val,
    eps_val,
    device,
    compute_function_target=False,
    output_part="both",
    target_total_bc=500,
    n_bc_edge=6,
    n_face_pts=40,
    n_corner_extra=5,
    bc_abs_cap=1e8,
):
    # ---------- collocation points ----------
    n = int(n_coll_pts)
    x1_all = np.random.uniform(x1_lo, x1_hi, size=n).astype(np.float64)
    x2_all = np.random.uniform(x2_lo, x2_hi, size=n).astype(np.float64)
    y1_all = np.random.uniform(y1_lo, y1_hi, size=n).astype(np.float64)

    x_coll = torch.tensor(
        np.stack((x1_all, x2_all, y1_all), axis=1),
        dtype=torch.float32,
        device=device,
    )

    # ---------- boundary points ----------
    vertices = np.array(
        [
            [x1_lo, x2_lo, y1_lo],
            [x1_hi, x2_lo, y1_lo],
            [x1_lo, x2_hi, y1_lo],
            [x1_hi, x2_hi, y1_lo],
            [x1_lo, x2_lo, y1_hi],
            [x1_hi, x2_lo, y1_hi],
            [x1_lo, x2_hi, y1_hi],
            [x1_hi, x2_hi, y1_hi],
        ],
        dtype=np.float64,
    )

    def lin_edge(start, end, n_pts):
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        ts = np.linspace(0.0, 1.0, int(n_pts))
        return start[None, :] + (end - start)[None, :] * ts[:, None]

    edges = np.concatenate(
        [
            # x1 direction
            lin_edge([x1_lo, x2_lo, y1_lo], [x1_hi, x2_lo, y1_lo], n_bc_edge),
            lin_edge([x1_lo, x2_hi, y1_lo], [x1_hi, x2_hi, y1_lo], n_bc_edge),
            lin_edge([x1_lo, x2_lo, y1_hi], [x1_hi, x2_lo, y1_hi], n_bc_edge),
            lin_edge([x1_lo, x2_hi, y1_hi], [x1_hi, x2_hi, y1_hi], n_bc_edge),
            # x2 direction
            lin_edge([x1_lo, x2_lo, y1_lo], [x1_lo, x2_hi, y1_lo], n_bc_edge),
            lin_edge([x1_hi, x2_lo, y1_lo], [x1_hi, x2_hi, y1_lo], n_bc_edge),
            lin_edge([x1_lo, x2_lo, y1_hi], [x1_lo, x2_hi, y1_hi], n_bc_edge),
            lin_edge([x1_hi, x2_lo, y1_hi], [x1_hi, x2_hi, y1_hi], n_bc_edge),
            # y1 direction
            lin_edge([x1_lo, x2_lo, y1_lo], [x1_lo, x2_lo, y1_hi], n_bc_edge),
            lin_edge([x1_hi, x2_lo, y1_lo], [x1_hi, x2_lo, y1_hi], n_bc_edge),
            lin_edge([x1_lo, x2_hi, y1_lo], [x1_lo, x2_hi, y1_hi], n_bc_edge),
            lin_edge([x1_hi, x2_hi, y1_lo], [x1_hi, x2_hi, y1_hi], n_bc_edge),
        ],
        axis=0,
    )

    def sample_face_x1(x1_fixed, n_pts):
        xs = np.full(n_pts, x1_fixed)
        x2s = np.random.uniform(x2_lo, x2_hi, size=n_pts)
        y1s = np.random.uniform(y1_lo, y1_hi, size=n_pts)
        return np.stack([xs, x2s, y1s], axis=1)

    def sample_face_x2(x2_fixed, n_pts):
        x1s = np.random.uniform(x1_lo, x1_hi, size=n_pts)
        x2s = np.full(n_pts, x2_fixed)
        y1s = np.random.uniform(y1_lo, y1_hi, size=n_pts)
        return np.stack([x1s, x2s, y1s], axis=1)

    def sample_face_y1(y1_fixed, n_pts):
        x1s = np.random.uniform(x1_lo, x1_hi, size=n_pts)
        x2s = np.random.uniform(x2_lo, x2_hi, size=n_pts)
        y1s = np.full(n_pts, y1_fixed)
        return np.stack([x1s, x2s, y1s], axis=1)

    faces = np.concatenate(
        [
            sample_face_x1(x1_lo, n_face_pts),
            sample_face_x1(x1_hi, n_face_pts),
            sample_face_x2(x2_lo, n_face_pts),
            sample_face_x2(x2_hi, n_face_pts),
            sample_face_y1(y1_lo, n_face_pts),
            sample_face_y1(y1_hi, n_face_pts),
        ],
        axis=0,
    )

    dx = 0.1 * max(x1_hi - x1_lo, 1e-12)
    dy = 0.1 * max(x2_hi - x2_lo, 1e-12)
    dz = 0.1 * max(y1_hi - y1_lo, 1e-12)

    corner_extra = []
    for vx1, vx2, vy1 in vertices:
        xl, xh = (vx1, min(vx1 + dx, x1_hi)) if vx1 == x1_lo else (max(vx1 - dx, x1_lo), vx1)
        yl, yh = (vx2, min(vx2 + dy, x2_hi)) if vx2 == x2_lo else (max(vx2 - dy, x2_lo), vx2)
        zl, zh = (vy1, min(vy1 + dz, y1_hi)) if vy1 == y1_lo else (max(vy1 - dz, y1_lo), vy1)
        pts = np.random.uniform([xl, yl, zl], [xh, yh, zh], size=(n_corner_extra, 3))
        corner_extra.append(pts)
    corner_extra = np.concatenate(corner_extra, axis=0)

    n_fixed = vertices.shape[0] + edges.shape[0] + faces.shape[0] + corner_extra.shape[0]
    n_inner = max(int(target_total_bc) - n_fixed, 0)
    inner_points = np.random.uniform(
        [x1_lo, x2_lo, y1_lo],
        [x1_hi, x2_hi, y1_hi],
        size=(n_inner, 3),
    )

    x_b_all = np.concatenate([vertices, edges, faces, corner_extra, inner_points], axis=0)
    x_b_tensor = torch.tensor(x_b_all, dtype=torch.float32, device=device)

    function_target = None
    if compute_function_target:
        function_target = compute_function_target_from_xcoll_1loop(
            x_coll,
            cy_val=cy_val,
            eps_val=eps_val,
            output_part=output_part,
        )

    quad_b = np.concatenate(
        [x_b_all, np.full((x_b_all.shape[0], 1), float(eps_val), dtype=float)],
        axis=1,
    )
    bc_complex = compute_boundary_values_rescaled_1loop(quad_b, cy_val)

    # Filter out boundary points where analytic values are non-finite
    # (e.g. points on singular letters such as w7=0 or w8=0).
    re_bc = np.real(bc_complex)
    im_bc = np.imag(bc_complex)
    finite_mask = np.isfinite(re_bc).all(axis=1) & np.isfinite(im_bc).all(axis=1)
    bounded_mask = (np.abs(re_bc) <= float(bc_abs_cap)).all(axis=1) & (
        np.abs(im_bc) <= float(bc_abs_cap)
    ).all(axis=1)
    keep_mask = finite_mask & bounded_mask

    n_bad_nonfinite = int((~finite_mask).sum())
    n_bad_large = int((finite_mask & (~bounded_mask)).sum())
    n_bad_total = int((~keep_mask).sum())
    if n_bad_total > 0:
        print(
            "[warn] build_inputs_and_boundary_1loop: "
            f"filtered {n_bad_total} / {bc_complex.shape[0]} boundary points "
            f"(non-finite={n_bad_nonfinite}, |value|>{bc_abs_cap:g}={n_bad_large})."
        )
        if (not np.any(keep_mask)) and np.any(finite_mask):
            print(
                "[warn] build_inputs_and_boundary_1loop: "
                "all finite boundary points exceeded bc_abs_cap; "
                "keeping finite points and bypassing the magnitude cap for this batch."
            )
            keep_mask = finite_mask

        if not np.any(keep_mask):
            raise ValueError(
                "build_inputs_and_boundary_1loop: no valid boundary points remain after filtering. "
                "All sampled boundary targets are non-finite. Check the domain, eps, or analytic target builder."
            )

        x_b_all = x_b_all[keep_mask]
        bc_complex = bc_complex[keep_mask]

    bc_concat = _complex_to_output_channels(bc_complex, output_part=output_part)
    bc_target = torch.tensor(bc_concat, dtype=torch.float32, device=device)
    x_b_tensor = torch.tensor(x_b_all, dtype=torch.float32, device=device)

    return x_coll, x_b_tensor, bc_target, function_target
