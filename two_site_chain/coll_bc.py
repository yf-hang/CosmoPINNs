import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor

from two_site_chain.sol_chain import I1_fin, I2_fin, I3_fin, I4_fin


def _eval_chain_chunk(x_chunk, cy_val):
    x_arr = np.asarray(x_chunk, dtype=float)
    out = np.empty((x_arr.shape[0], 4), dtype=complex)
    for i, (x1, x2, eps) in enumerate(x_arr):
        x1f, x2f, epsf = float(x1), float(x2), float(eps)
        out[i, 0] = I1_fin(x1f, x2f, epsf, cy_val)
        out[i, 1] = I2_fin(x1f, x2f, epsf, cy_val)
        out[i, 2] = I3_fin(x1f, x2f, epsf, cy_val)
        out[i, 3] = I4_fin(x1f, x2f, epsf, cy_val)
    return out


def _normalize_output_part(value, default="re"):
    if value is None:
        value = default
    s = str(value).strip().lower()
    if s in {"re", "real"}:
        return "re"
    if s in {"im", "imag", "imaginary"}:
        return "im"
    raise ValueError(f"Unsupported output part: {value!r}. Expected one of Re or Im.")


def _complex_to_output_channels(function_complex: np.ndarray, output_part="re") -> np.ndarray:
    part = _normalize_output_part(output_part)
    if part == "re":
        return np.real(function_complex)
    return np.imag(function_complex)


def compute_boundary_values_rescaled(
    x_triplet,
    cy_val,
    *,
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
):
    """
    Compute analytic values for multiple (x1, x2, eps) triplets.
    Input shape: (N,3)
    Output: complex array (N,4) for [I1, I2, I3, I4]
    """
    x_all = np.asarray(x_triplet, dtype=float)
    if x_all.ndim == 1:
        if x_all.shape[0] != 3:
            raise ValueError("single point must be shape (3,) = (x1,x2,eps)")
        x_all = x_all.reshape(1, 3)

    if x_all.shape[1] != 3:
        raise ValueError(f"x_triplet must have 3 columns (x1,x2,eps), got shape {x_all.shape}")

    n_pts = int(x_all.shape[0])
    nw = max(int(num_workers), 1)
    cs = max(int(chunk_size), 1)
    nmin = max(int(parallel_min_points), 1)

    if (nw == 1) or (n_pts < nmin):
        return _eval_chain_chunk(x_all, cy_val)

    chunks = [x_all[i:i + cs] for i in range(0, n_pts, cs)]
    cy_vals = [float(cy_val)] * len(chunks)
    try:
        with ProcessPoolExecutor(max_workers=nw) as pool:
            parts = list(pool.map(_eval_chain_chunk, chunks, cy_vals))
        return np.concatenate(parts, axis=0)
    except (PermissionError, OSError):
        print("[warn] chain target parallel disabled by runtime; fallback to single-process.")
        return _eval_chain_chunk(x_all, cy_val)


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def compute_function_target_from_xcoll(
    x_coll: torch.Tensor,
    *,
    cy_val: float,
    eps_val: float,
    output_part="re",
    num_workers=1,
    chunk_size=2000,
    parallel_min_points=5000,
) -> torch.Tensor:
    x_np = to_numpy(x_coll)

    if x_np.ndim != 2:
        raise ValueError(f"x_coll must be 2D tensor/array, got shape {x_np.shape}")

    if x_np.shape[1] == 2:
        trip_np = np.concatenate(
            [x_np[:, :2], np.full((x_np.shape[0], 1), float(eps_val))],
            axis=1,
        )
    elif x_np.shape[1] == 3:
        trip_np = x_np[:, :3]
    else:
        raise ValueError(f"x_coll must have 2 or 3 columns, got {x_np.shape}")

    function_complex = compute_boundary_values_rescaled(
        trip_np,
        cy_val,
        num_workers=num_workers,
        chunk_size=chunk_size,
        parallel_min_points=parallel_min_points,
    )
    function_concat = _complex_to_output_channels(function_complex, output_part=output_part)

    return torch.tensor(function_concat, dtype=torch.float32, device=x_coll.device)


def build_inputs_and_boundary(
    n_coll_pts,
    x1_lo,
    x1_hi,
    x2_lo,
    x2_hi,
    cy_val,
    eps_val,
    device,
    compute_function_target=False,
    output_part="re",
    n_bc_edge=5,
    n_corner_each=5,
    target_total=200,
):
    # ---------- collocation points: mixture sampling in x ----------
    n = int(n_coll_pts)

    p_uniform = 1.0#0.7
    p_diag = 0.0#0.2

    p_outer = 0.0#0.05
    outer_band = 1.0

    n_o = int(n * p_outer)
    n_u = int(n * p_uniform) - n_o
    n_d = int(n * p_diag)
    n_c = n - n_u - n_d - n_o

    if n_u < 0:
        raise ValueError("p_outer too large: makes n_u < 0. Reduce p_outer.")

    x1_u = np.random.uniform(x1_lo, x1_hi, size=n_u).astype(np.float64)
    x2_u = np.random.uniform(x2_lo, x2_hi, size=n_u).astype(np.float64)

    diag_halfwidth = 0.5
    x1_d = np.random.uniform(x1_lo, x1_hi, size=n_d).astype(np.float64)
    noise = np.random.uniform(-diag_halfwidth, diag_halfwidth, size=n_d).astype(np.float64)
    x2_d = np.clip(x1_d + noise, x2_lo, x2_hi).astype(np.float64)

    corner_band = 1.0
    corners = np.random.randint(0, 4, size=n_c)

    x1_c = np.empty(n_c, dtype=np.float64)
    x2_c = np.empty(n_c, dtype=np.float64)

    m = corners == 0
    x1_c[m] = np.random.uniform(x1_lo, min(x1_lo + corner_band, x1_hi), size=m.sum())
    x2_c[m] = np.random.uniform(x2_lo, min(x2_lo + corner_band, x2_hi), size=m.sum())

    m = corners == 1
    x1_c[m] = np.random.uniform(x1_lo, min(x1_lo + corner_band, x1_hi), size=m.sum())
    x2_c[m] = np.random.uniform(max(x2_hi - corner_band, x2_lo), x2_hi, size=m.sum())

    m = corners == 2
    x1_c[m] = np.random.uniform(max(x1_hi - corner_band, x1_lo), x1_hi, size=m.sum())
    x2_c[m] = np.random.uniform(x2_lo, min(x2_lo + corner_band, x2_hi), size=m.sum())

    m = corners == 3
    x1_c[m] = np.random.uniform(max(x1_hi - corner_band, x1_lo), x1_hi, size=m.sum())
    x2_c[m] = np.random.uniform(max(x2_hi - corner_band, x2_lo), x2_hi, size=m.sum())

    if n_o > 0:
        n_r = n_o // 2
        n_t = n_o - n_r

        x1_r = np.random.uniform(max(x1_hi - outer_band, x1_lo), x1_hi, size=n_r).astype(np.float64)
        x2_r = np.random.uniform(x2_lo, x2_hi, size=n_r).astype(np.float64)

        x1_t = np.random.uniform(x1_lo, x1_hi, size=n_t).astype(np.float64)
        x2_t = np.random.uniform(max(x2_hi - outer_band, x2_lo), x2_hi, size=n_t).astype(np.float64)

        x1_o = np.concatenate([x1_r, x1_t], axis=0)
        x2_o = np.concatenate([x2_r, x2_t], axis=0)
    else:
        x1_o = np.empty(0, dtype=np.float64)
        x2_o = np.empty(0, dtype=np.float64)

    x1_all = np.concatenate((x1_u, x1_d, x1_c, x1_o), axis=0)[:n]
    x2_all = np.concatenate((x2_u, x2_d, x2_c, x2_o), axis=0)[:n]

    idx = np.random.permutation(n)
    x1_all, x2_all = x1_all[idx], x2_all[idx]

    x_coll = torch.tensor(np.stack((x1_all, x2_all), axis=1), dtype=torch.float32, device=device)

    # ---------- boundary points ----------
    rng = np.random.default_rng(seed=0)

    x1_grid = np.linspace(x1_lo, x1_hi, n_bc_edge)
    x2_grid = np.linspace(x2_lo, x2_hi, n_bc_edge)

    bc_bottom = np.stack([x1_grid, np.full_like(x1_grid, x2_lo)], axis=1)
    bc_top = np.stack([x1_grid, np.full_like(x1_grid, x2_hi)], axis=1)
    bc_left = np.stack([np.full_like(x2_grid, x1_lo), x2_grid], axis=1)
    bc_right = np.stack([np.full_like(x2_grid, x1_hi), x2_grid], axis=1)

    x_b_all_edges = np.concatenate([bc_bottom, bc_top, bc_left, bc_right], axis=0)

    corner_width = 0.1 * (x1_hi - x1_lo)

    corner_points = np.concatenate([
        rng.uniform([x1_lo, x2_lo], [x1_lo + corner_width, x2_lo + corner_width], size=(n_corner_each, 2)),
        rng.uniform([x1_hi - corner_width, x2_lo], [x1_hi, x2_lo + corner_width], size=(n_corner_each, 2)),
        rng.uniform([x1_lo, x2_hi - corner_width], [x1_lo + corner_width, x2_hi], size=(n_corner_each, 2)),
        rng.uniform([x1_hi - corner_width, x2_hi - corner_width], [x1_hi, x2_hi], size=(n_corner_each, 2)),
    ], axis=0)

    n_boundary = x_b_all_edges.shape[0]
    n_corners = corner_points.shape[0]
    n_remaining = max(target_total - n_boundary - n_corners, 0)

    remaining_points = rng.uniform([x1_lo, x2_lo], [x1_hi, x2_hi], size=(n_remaining, 2))
    x_b_all = np.concatenate([x_b_all_edges, corner_points, remaining_points], axis=0)

    function_target = None
    if compute_function_target:
        function_target = compute_function_target_from_xcoll(
            x_coll,
            cy_val=cy_val,
            eps_val=eps_val,
            output_part=output_part,
        )

    # ---------- boundary target (N_b, 8), fixed eps ----------
    trip_b = np.concatenate(
        [x_b_all, np.full((x_b_all.shape[0], 1), float(eps_val), dtype=float)],
        axis=1,
    )
    bc_complex = compute_boundary_values_rescaled(trip_b, cy_val)
    bc_concat = _complex_to_output_channels(bc_complex, output_part=output_part)
    bc_target = torch.tensor(bc_concat, dtype=torch.float32, device=device)

    x_b_tensor = torch.tensor(x_b_all, dtype=torch.float32, device=device)

    return x_coll, x_b_tensor, bc_target, function_target
