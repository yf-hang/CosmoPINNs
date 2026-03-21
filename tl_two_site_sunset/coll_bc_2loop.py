import itertools
import math
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

try:
    from scipy.special import spence as _scipy_spence
except Exception:
    _scipy_spence = None

from two_site_chain.sol_chain import eps_to_n_int, eps_to_n_pos_int

from tl_two_site_sunset.sol_2loop import (
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
    I11_fin,
    I12_fin,
    I13_fin,
    I14_fin,
    I15_fin,
    I16_fin,
    I17_fin,
    I18_fin,
    I19_fin,
    I20_fin,
    I21_fin,
    I22_fin,
)

_ANALYTIC_FUNCS = (
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
    I11_fin,
    I12_fin,
    I13_fin,
    I14_fin,
    I15_fin,
    I16_fin,
    I17_fin,
    I18_fin,
    I19_fin,
    I20_fin,
    I21_fin,
    I22_fin,
)

_EPS_TOL = 1e-12


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


def _resolve_n_basis(n_basis):
    if n_basis is None:
        return len(_ANALYTIC_FUNCS)
    n = int(n_basis)
    if n <= 0 or n > len(_ANALYTIC_FUNCS):
        raise ValueError(
            f"n_basis must be in [1, {len(_ANALYTIC_FUNCS)}], got {n_basis}."
        )
    return n


def _build_ws_tensor(x_coll: torch.Tensor, *, cy_val: float):
    if x_coll.ndim != 2:
        raise ValueError(f"x_coll must be 2D, got shape {tuple(x_coll.shape)}")
    if int(x_coll.shape[1]) not in (4, 5):
        raise ValueError(f"x_coll must have 4 or 5 columns, got shape {tuple(x_coll.shape)}")

    x_core = x_coll[:, :4].to(dtype=torch.float64)
    x1, x2, y1, y2 = x_core.unbind(dim=1)
    c = torch.tensor(float(cy_val), dtype=torch.float64, device=x_coll.device)

    w1 = x1 + y1 + y2 + c
    w2 = x2 + y1 + y2 + c
    w9 = x1 + y1 + y2 - c
    w10 = x2 + y1 + y2 - c
    w11 = x1 + y1 - y2 + c
    w12 = x2 + y1 - y2 + c
    w13 = x1 - y1 + y2 + c
    w14 = x2 - y1 + y2 + c
    w3 = x1 + y1 - y2 - c
    w4 = x2 + y1 - y2 - c
    w5 = x1 - y1 + y2 - c
    w6 = x2 - y1 + y2 - c
    w7 = x1 - y1 - y2 + c
    w8 = x2 - y1 - y2 + c
    w15 = x1 - y1 - y2 - c
    w16 = x2 - y1 - y2 - c
    w20 = x1 + x2 + 2.0 * y1 + 2.0 * y2
    w21 = x1 + x2 + 2.0 * y1 + 2.0 * c
    w22 = x1 + x2 + 2.0 * y2 + 2.0 * c
    w17 = x1 + x2 + 2.0 * y1
    w18 = x1 + x2 + 2.0 * y2
    w19 = x1 + x2 + 2.0 * c
    w23 = x1 + x2

    return (
        w1, w2, w3, w4, w5, w6, w7, w8,
        w9, w10, w11, w12, w13, w14, w15, w16,
        w17, w18, w19, w20, w21, w22, w23,
    )


def _polylog2_complex_np(z):
    if _scipy_spence is None:
        raise RuntimeError("scipy.special.spence is unavailable.")
    z_arr = np.asarray(z, dtype=np.complex128)
    return _scipy_spence(1.0 - z_arr)


def _fast_p3_target_eps0(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    n_basis: int,
    output_part: str,
):
    if _scipy_spence is None:
        raise RuntimeError("SciPy is required for the eps=0 2-loop fast path.")

    x_np = x_coll[:, :4].detach().cpu().numpy().astype(np.float64, copy=False)
    x1 = x_np[:, 0]
    x2 = x_np[:, 1]
    y1 = x_np[:, 2]
    y2 = x_np[:, 3]
    c = float(cy_val)

    w1 = x1 + y1 + y2 + c
    w2 = x2 + y1 + y2 + c
    w9 = x1 + y1 + y2 - c
    w10 = x2 + y1 + y2 - c
    w11 = x1 + y1 - y2 + c
    w12 = x2 + y1 - y2 + c
    w13 = x1 - y1 + y2 + c
    w14 = x2 - y1 + y2 + c
    w3 = x1 + y1 - y2 - c
    w4 = x2 + y1 - y2 - c
    w5 = x1 - y1 + y2 - c
    w6 = x2 - y1 + y2 - c
    w7 = x1 - y1 - y2 + c
    w8 = x2 - y1 - y2 + c
    w15 = x1 - y1 - y2 - c
    w16 = x2 - y1 - y2 - c
    w20 = x1 + x2 + 2.0 * y1 + 2.0 * y2
    w21 = x1 + x2 + 2.0 * y1 + 2.0 * c
    w22 = x1 + x2 + 2.0 * y2 + 2.0 * c
    w17 = x1 + x2 + 2.0 * y1
    w18 = x1 + x2 + 2.0 * y2
    w19 = x1 + x2 + 2.0 * c
    w23 = x1 + x2

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
    w12c = np.asarray(w12, dtype=np.complex128)
    w13c = np.asarray(w13, dtype=np.complex128)
    w14c = np.asarray(w14, dtype=np.complex128)
    w15c = np.asarray(w15, dtype=np.complex128)
    w16c = np.asarray(w16, dtype=np.complex128)
    w17c = np.asarray(w17, dtype=np.complex128)
    w18c = np.asarray(w18, dtype=np.complex128)
    w19c = np.asarray(w19, dtype=np.complex128)
    w20c = np.asarray(w20, dtype=np.complex128)
    w21c = np.asarray(w21, dtype=np.complex128)
    w22c = np.asarray(w22, dtype=np.complex128)
    w23c = np.asarray(w23, dtype=np.complex128)

    i1 = (
        _polylog2_complex_np(w3c / w1c)
        + _polylog2_complex_np(w4c / w2c)
        - _polylog2_complex_np((w3c * w4c) / (w1c * w2c))
        + _polylog2_complex_np(w5c / w1c)
        + _polylog2_complex_np(w6c / w2c)
        - _polylog2_complex_np((w5c * w6c) / (w1c * w2c))
        + _polylog2_complex_np(w7c / w1c)
        + _polylog2_complex_np(w8c / w2c)
        - _polylog2_complex_np((w7c * w8c) / (w1c * w2c))
        - _polylog2_complex_np(w9c / w1c)
        - _polylog2_complex_np(w10c / w2c)
        + _polylog2_complex_np((w9c * w10c) / (w1c * w2c))
        - _polylog2_complex_np(w11c / w1c)
        - _polylog2_complex_np(w12c / w2c)
        + _polylog2_complex_np((w11c * w12c) / (w1c * w2c))
        - _polylog2_complex_np(w13c / w1c)
        - _polylog2_complex_np(w14c / w2c)
        + _polylog2_complex_np((w13c * w14c) / (w1c * w2c))
        - _polylog2_complex_np(w15c / w1c)
        - _polylog2_complex_np(w16c / w2c)
        + _polylog2_complex_np((w15c * w16c) / (w1c * w2c))
        + (np.pi ** 2) / 6.0
    )
    i2 = np.log(w17c / w2c)
    i3 = np.log(w18c / w2c)
    i4 = np.log(w19c / w2c)
    i5 = np.log(w2c / w20c)
    i6 = np.log(w2c / w21c)
    i7 = np.log(w2c / w22c)
    i8 = np.log(w2c / w23c)
    i9 = np.log(w17c / w1c)
    i10 = np.log(w18c / w1c)
    i11 = np.log(w19c / w1c)
    i12 = np.log(w1c / w20c)
    i13 = np.log(w1c / w21c)
    i14 = np.log(w1c / w22c)
    i15 = np.log(w1c / w23c)
    ones = np.ones_like(w1c)
    i16 = ones
    i17 = ones
    i18 = ones
    i19 = -ones
    i20 = -ones
    i21 = -ones
    i22 = -ones

    function_complex = np.stack(
        (
            i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
            i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22,
        ),
        axis=1,
    )[:, : int(n_basis)]
    function_concat = _complex_to_output_channels(function_complex, output_part=output_part)
    return torch.tensor(function_concat, dtype=torch.float64, device=x_coll.device)


def _fast_p3_target_positive_int_re(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    eps_val: float,
    n_basis: int,
):
    n = eps_to_n_pos_int(float(eps_val))
    if n is None:
        raise ValueError(f"Fast positive-integer target path expects eps=+n, got eps={eps_val}.")

    (
        w1, w2, w3, w4, w5, w6, w7, w8,
        w9, w10, w11, w12, w13, w14, w15, w16,
        w17, w18, w19, w20, w21, w22, w23,
    ) = _build_ws_tensor(x_coll, cy_val=cy_val)

    coeff = float(math.comb(2 * n, n))

    def _branch(w_main, w_alt, *, sign=1.0):
        series = torch.zeros_like(w_main)
        for k in range(1, n + 1):
            c1 = float(math.comb(2 * n - 1, n + k - 1))
            c2 = float(math.comb(2 * n - 1, n - k - 1)) if (n - k - 1) >= 0 else 0.0
            series = series + (
                c1 * torch.pow(w_alt, n + k) * torch.pow(w_main, n - k)
                - c2 * torch.pow(w_alt, n - k) * torch.pow(w_main, n + k)
            ) / float(k)

        prefactor = ((-1.0) ** (n + 1)) * 0.5 * torch.pow(w_main, n) * torch.pow(w_alt, n)
        result = prefactor
        result = result + 0.5 * torch.pow(w_main, n) * torch.pow(w_alt, n) * torch.log(w_alt / w_main)
        result = result + series / coeff
        return float(sign) * result

    def _const_branch(w_main, *, sign=1.0):
        return float(sign) * torch.pow(w_main, 2 * n) / (float(n) * coeff)

    i2 = _branch(w2, w3)
    i3 = _branch(w2, w5)
    i4 = _branch(w2, w7)
    i5 = _branch(w2, w9, sign=-1.0)
    i6 = _branch(w2, w11, sign=-1.0)
    i7 = _branch(w2, w13, sign=-1.0)
    i8 = _branch(w2, w15, sign=-1.0)
    i9 = _branch(w1, w4)
    i10 = _branch(w1, w6)
    i11 = _branch(w1, w8)
    i12 = _branch(w1, w10, sign=-1.0)
    i13 = _branch(w1, w12, sign=-1.0)
    i14 = _branch(w1, w14, sign=-1.0)
    i15 = _branch(w1, w16, sign=-1.0)
    i16 = _const_branch(w17)
    i17 = _const_branch(w18)
    i18 = _const_branch(w19)
    i19 = _const_branch(w20, sign=-1.0)
    i20 = _const_branch(w21, sign=-1.0)
    i21 = _const_branch(w22, sign=-1.0)
    i22 = _const_branch(w23, sign=-1.0)

    i1 = (
        -torch.pow(w1, n) * torch.pow(w2, n)
        + (i2 + i9 - i16)
        + (i3 + i10 - i17)
        + (i4 + i11 - i18)
        + (i5 + i12 - i19)
        + (i6 + i13 - i20)
        + (i7 + i14 - i21)
        + (i8 + i15 - i22)
    )

    out = torch.stack(
        (
            i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
            i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22,
        ),
        dim=1,
    )
    return out[:, : int(n_basis)]


def _fast_p3_target_negative_int_re(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    eps_val: float,
    n_basis: int,
):
    n = eps_to_n_int(float(eps_val))
    if n is None:
        raise ValueError(f"Fast negative-integer target path expects eps=-n, got eps={eps_val}.")

    (
        w1, w2, w3, w4, w5, w6, w7, w8,
        w9, w10, w11, w12, w13, w14, w15, w16,
        w17, w18, w19, w20, w21, w22, w23,
    ) = _build_ws_tensor(x_coll, cy_val=cy_val)

    comb_2n_n = float(math.comb(2 * n, n))

    def _int_series_sum(w_main, w_main2):
        acc = torch.zeros_like(w_main)
        for k in range(n):
            acc = acc + float(math.comb(n + k, k)) * torch.pow(w_main, k) * torch.pow(w_main2, n - k - 1)
        return acc

    def _simple_branch(w_main, w_alt, wn, *, sign):
        series = _int_series_sum(w_main, wn)
        return float(sign) * (w_alt / (torch.pow(w_main, n) * torch.pow(wn, 2 * n))) * series

    i2 = _simple_branch(w2, w3, w17, sign=-1.0)
    i3 = _simple_branch(w2, w5, w18, sign=-1.0)
    i4 = _simple_branch(w2, w7, w19, sign=-1.0)
    i5 = _simple_branch(w2, w9, w20, sign=1.0)
    i6 = _simple_branch(w2, w11, w21, sign=1.0)
    i7 = _simple_branch(w2, w13, w22, sign=1.0)
    i8 = _simple_branch(w2, w15, w23, sign=1.0)
    i9 = _simple_branch(w1, w4, w17, sign=-1.0)
    i10 = _simple_branch(w1, w6, w18, sign=-1.0)
    i11 = _simple_branch(w1, w8, w19, sign=-1.0)
    i12 = _simple_branch(w1, w10, w20, sign=1.0)
    i13 = _simple_branch(w1, w12, w21, sign=1.0)
    i14 = _simple_branch(w1, w14, w22, sign=1.0)
    i15 = _simple_branch(w1, w16, w23, sign=1.0)
    i16 = comb_2n_n / torch.pow(w17, 2 * n)
    i17 = comb_2n_n / torch.pow(w18, 2 * n)
    i18 = comb_2n_n / torch.pow(w19, 2 * n)
    i19 = -comb_2n_n / torch.pow(w20, 2 * n)
    i20 = -comb_2n_n / torch.pow(w21, 2 * n)
    i21 = -comb_2n_n / torch.pow(w22, 2 * n)
    i22 = -comb_2n_n / torch.pow(w23, 2 * n)
    i1 = (
        -1.0 / (torch.pow(w1, n) * torch.pow(w2, n))
        + (i2 + i9 - i16)
        + (i3 + i10 - i17)
        + (i4 + i11 - i18)
        + (i5 + i12 - i19)
        + (i6 + i13 - i20)
        + (i7 + i14 - i21)
        + (i8 + i15 - i22)
    )

    out = torch.stack(
        (
            i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11,
            i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22,
        ),
        dim=1,
    )
    return out[:, : int(n_basis)]


def _eval_2loop_chunk(x_chunk, cy_val, n_basis):
    x_arr = np.asarray(x_chunk, dtype=float)
    n_basis = _resolve_n_basis(n_basis)
    out = np.empty((x_arr.shape[0], n_basis), dtype=complex)
    funcs = _ANALYTIC_FUNCS[:n_basis]

    for i, (x1, x2, y1, y2, eps) in enumerate(x_arr):
        x1f = float(x1)
        x2f = float(x2)
        y1f = float(y1)
        y2f = float(y2)
        epsf = float(eps)
        for j, fn in enumerate(funcs):
            out[i, j] = fn(x1f, x2f, y1f, y2f, epsf, cy_val)
    return out


def compute_boundary_values_rescaled_2loop(
    x_quintuplet,
    cy_val,
    *,
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
    n_basis=22,
):
    """
    Compute analytic values for multiple (x1, x2, y1, y2, eps) points.
    Input shape: (N,5)
    Output: complex array (N, n_basis) for [I1, ..., I_n_basis].
    """
    x_all = np.asarray(x_quintuplet, dtype=float)
    n_basis = _resolve_n_basis(n_basis)

    if x_all.ndim == 1:
        if x_all.shape[0] != 5:
            raise ValueError("single point must be shape (5,) = (x1,x2,y1,y2,eps)")
        x_all = x_all.reshape(1, 5)

    if x_all.shape[1] != 5:
        raise ValueError(
            f"x_quintuplet must have 5 columns (x1,x2,y1,y2,eps), got shape {x_all.shape}"
        )

    n_pts = int(x_all.shape[0])
    nw = max(int(num_workers), 1)
    cs = max(int(chunk_size), 1)
    nmin = max(int(parallel_min_points), 1)

    if (nw == 1) or (n_pts < nmin):
        return _eval_2loop_chunk(x_all, cy_val, n_basis)

    chunks = [x_all[i : i + cs] for i in range(0, n_pts, cs)]
    cy_vals = [float(cy_val)] * len(chunks)
    n_basis_vals = [int(n_basis)] * len(chunks)
    try:
        with ProcessPoolExecutor(max_workers=nw) as pool:
            parts = list(pool.map(_eval_2loop_chunk, chunks, cy_vals, n_basis_vals))
        return np.concatenate(parts, axis=0)
    except (PermissionError, OSError):
        print("[warn] 2-loop target parallel disabled by runtime; fallback to single-process.")
        return _eval_2loop_chunk(x_all, cy_val, n_basis)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def compute_function_target_from_xcoll_2loop(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    eps_val: float,
    output_part="both",
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
    n_basis=22,
) -> torch.Tensor:
    output_part = _normalize_output_part(output_part)
    n_basis = _resolve_n_basis(n_basis)

    if isinstance(x_coll, torch.Tensor):
        x_tensor = x_coll
    else:
        x_tensor = torch.as_tensor(x_coll, dtype=torch.float32)

    if abs(float(eps_val)) < _EPS_TOL and _scipy_spence is not None:
        return _fast_p3_target_eps0(
            x_tensor,
            cy_val=float(cy_val),
            n_basis=n_basis,
            output_part=output_part,
        )

    if output_part == "re":
        n_pos = eps_to_n_pos_int(float(eps_val))
        if n_pos is not None:
            fast = _fast_p3_target_positive_int_re(
                x_tensor,
                cy_val=float(cy_val),
                eps_val=float(eps_val),
                n_basis=n_basis,
            )
            return fast.to(dtype=torch.float64, device=x_tensor.device)

        n_neg = eps_to_n_int(float(eps_val))
        if n_neg is not None:
            fast = _fast_p3_target_negative_int_re(
                x_tensor,
                cy_val=float(cy_val),
                eps_val=float(eps_val),
                n_basis=n_basis,
            )
            return fast.to(dtype=torch.float64, device=x_tensor.device)

    x_np = to_numpy(x_tensor)

    if x_np.ndim != 2:
        raise ValueError(f"x_coll must be 2D tensor/array, got shape {x_np.shape}")

    if x_np.shape[1] == 4:
        quint_np = np.concatenate(
            [x_np[:, :4], np.full((x_np.shape[0], 1), float(eps_val))],
            axis=1,
        )
    elif x_np.shape[1] == 5:
        quint_np = x_np[:, :5]
    else:
        raise ValueError(f"x_coll must have 4 or 5 columns, got {x_np.shape}")

    function_complex = compute_boundary_values_rescaled_2loop(
        quint_np,
        cy_val,
        num_workers=num_workers,
        chunk_size=chunk_size,
        parallel_min_points=parallel_min_points,
        n_basis=n_basis,
    )
    function_concat = _complex_to_output_channels(function_complex, output_part=output_part)

    return torch.tensor(function_concat, dtype=torch.float64, device=x_tensor.device)


def _lin_edge(start, end, n_pts):
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    ts = np.linspace(0.0, 1.0, int(n_pts))
    return start[None, :] + (end - start)[None, :] * ts[:, None]


def _sample_boundary_4d(
    *,
    lo,
    hi,
    target_total_bc,
    n_bc_edge,
    n_face_pts,
    n_cell_pts,
    n_corner_extra,
):
    lo = np.asarray(lo, dtype=np.float64).reshape(4)
    hi = np.asarray(hi, dtype=np.float64).reshape(4)

    # 16 vertices.
    vertices = np.array(
        list(itertools.product([lo[0], hi[0]], [lo[1], hi[1]], [lo[2], hi[2]], [lo[3], hi[3]])),
        dtype=np.float64,
    )

    # 32 edges (4 directions * 2^3 fixed combinations).
    edges_all = []
    for axis in range(4):
        fixed_axes = [a for a in range(4) if a != axis]
        for bits in itertools.product([0, 1], repeat=3):
            start = np.empty(4, dtype=np.float64)
            end = np.empty(4, dtype=np.float64)
            for i, ax in enumerate(fixed_axes):
                val = lo[ax] if bits[i] == 0 else hi[ax]
                start[ax] = val
                end[ax] = val
            start[axis] = lo[axis]
            end[axis] = hi[axis]
            edges_all.append(_lin_edge(start, end, n_bc_edge))
    edges = np.concatenate(edges_all, axis=0)

    # 24 square faces (choose 2 varying dims, 2 fixed dims on lo/hi).
    faces_all = []
    for varying in itertools.combinations(range(4), 2):
        fixed = [a for a in range(4) if a not in varying]
        for bits in itertools.product([0, 1], repeat=2):
            pts = np.empty((int(n_face_pts), 4), dtype=np.float64)
            for ax in varying:
                pts[:, ax] = np.random.uniform(lo[ax], hi[ax], size=int(n_face_pts))
            for i, ax in enumerate(fixed):
                pts[:, ax] = lo[ax] if bits[i] == 0 else hi[ax]
            faces_all.append(pts)
    faces = np.concatenate(faces_all, axis=0)

    # 8 cubic cells on the boundary (fix one dim to lo/hi, vary other 3 dims).
    cells_all = []
    for fixed_axis in range(4):
        varying = [a for a in range(4) if a != fixed_axis]
        for fixed_val in (lo[fixed_axis], hi[fixed_axis]):
            pts = np.empty((int(n_cell_pts), 4), dtype=np.float64)
            for ax in varying:
                pts[:, ax] = np.random.uniform(lo[ax], hi[ax], size=int(n_cell_pts))
            pts[:, fixed_axis] = fixed_val
            cells_all.append(pts)
    cells = np.concatenate(cells_all, axis=0)

    # Extra points around each corner.
    delta = 0.1 * np.maximum(hi - lo, 1e-12)
    corner_extra_list = []
    for v in vertices:
        low_local = np.empty(4, dtype=np.float64)
        high_local = np.empty(4, dtype=np.float64)
        for d in range(4):
            if np.isclose(v[d], lo[d]):
                low_local[d] = v[d]
                high_local[d] = min(v[d] + delta[d], hi[d])
            else:
                low_local[d] = max(v[d] - delta[d], lo[d])
                high_local[d] = v[d]
        pts = np.random.uniform(low_local, high_local, size=(int(n_corner_extra), 4))
        corner_extra_list.append(pts)
    corner_extra = np.concatenate(corner_extra_list, axis=0)

    n_fixed = (
        vertices.shape[0]
        + edges.shape[0]
        + faces.shape[0]
        + cells.shape[0]
        + corner_extra.shape[0]
    )
    n_inner = max(int(target_total_bc) - int(n_fixed), 0)
    inner_points = np.random.uniform(lo, hi, size=(n_inner, 4))

    x_b_all = np.concatenate(
        [vertices, edges, faces, cells, corner_extra, inner_points],
        axis=0,
    )
    return x_b_all


def build_inputs_and_boundary_2loop(
    n_coll_pts,
    x1_lo,
    x1_hi,
    x2_lo,
    x2_hi,
    y1_lo,
    y1_hi,
    y2_lo,
    y2_hi,
    cy_val,
    eps_val,
    device,
    compute_function_target=False,
    output_part="both",
    target_total_bc=500,
    n_bc_edge=6,
    n_face_pts=40,
    n_cell_pts=40,
    n_corner_extra=5,
    bc_abs_cap=1e8,
    n_basis=22,
):
    # ---------- collocation points ----------
    n = int(n_coll_pts)
    x1_all = np.random.uniform(x1_lo, x1_hi, size=n).astype(np.float64)
    x2_all = np.random.uniform(x2_lo, x2_hi, size=n).astype(np.float64)
    y1_all = np.random.uniform(y1_lo, y1_hi, size=n).astype(np.float64)
    y2_all = np.random.uniform(y2_lo, y2_hi, size=n).astype(np.float64)

    x_coll = torch.tensor(
        np.stack((x1_all, x2_all, y1_all, y2_all), axis=1),
        dtype=torch.float32,
        device=device,
    )

    # ---------- boundary points ----------
    x_b_all = _sample_boundary_4d(
        lo=[x1_lo, x2_lo, y1_lo, y2_lo],
        hi=[x1_hi, x2_hi, y1_hi, y2_hi],
        target_total_bc=target_total_bc,
        n_bc_edge=n_bc_edge,
        n_face_pts=n_face_pts,
        n_cell_pts=n_cell_pts,
        n_corner_extra=n_corner_extra,
    )

    function_target = None
    if compute_function_target:
        function_target = compute_function_target_from_xcoll_2loop(
            x_coll,
            cy_val=cy_val,
            eps_val=eps_val,
            output_part=output_part,
            n_basis=n_basis,
        )

    quint_b = np.concatenate(
        [x_b_all, np.full((x_b_all.shape[0], 1), float(eps_val), dtype=float)],
        axis=1,
    )
    bc_complex = compute_boundary_values_rescaled_2loop(
        quint_b,
        cy_val,
        n_basis=n_basis,
    )

    # Filter out boundary points with non-finite or extremely large analytic values.
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
            "[warn] build_inputs_and_boundary_2loop: "
            f"filtered {n_bad_total} / {bc_complex.shape[0]} boundary points "
            f"(non-finite={n_bad_nonfinite}, |value|>{bc_abs_cap:g}={n_bad_large})."
        )
        if (not np.any(keep_mask)) and np.any(finite_mask):
            print(
                "[warn] build_inputs_and_boundary_2loop: "
                "all finite boundary points exceeded bc_abs_cap; "
                "keeping finite points and bypassing the magnitude cap for this batch."
            )
            keep_mask = finite_mask

        if not np.any(keep_mask):
            raise ValueError(
                "build_inputs_and_boundary_2loop: no valid boundary points remain after filtering. "
                "All sampled boundary targets are non-finite. Check the domain, eps, or analytic target builder."
            )

        x_b_all = x_b_all[keep_mask]
        bc_complex = bc_complex[keep_mask]

    bc_concat = _complex_to_output_channels(bc_complex, output_part=output_part)
    bc_target = torch.tensor(bc_concat, dtype=torch.float32, device=device)
    x_b_tensor = torch.tensor(x_b_all, dtype=torch.float32, device=device)

    return x_coll, x_b_tensor, bc_target, function_target
