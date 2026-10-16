# Empty-subset-first convention: S0={}, S1={y1}, S2={c}.
# a_k multiplies dlog(w_k) in the new I basis (also for a_k_eps0).
# The simultaneous basis/letter permutation leaves these sparse entries unchanged.
import torch

_DTYPE = torch.float32
_SHAPE = (10, 10)


def _dense_from_entries(entries):
    mat = torch.zeros(_SHAPE, dtype=_DTYPE)
    if not entries:
        return mat
    rows, cols, vals = zip(*entries)
    mat[list(rows), list(cols)] = torch.tensor(vals, dtype=_DTYPE)
    return mat


a1 = _dense_from_entries([
    (0, 0, 1.0), (0, 1, -1.0), (0, 2, -1.0), (0, 3, -1.0),
    (4, 4, 1.0), (4, 7, -1.0),
    (5, 5, 1.0), (5, 8, -1.0),
    (6, 6, 1.0), (6, 9, -1.0),
])
a2 = _dense_from_entries([
    (0, 0, 1.0), (0, 4, -1.0), (0, 5, -1.0), (0, 6, -1.0),
    (1, 1, 1.0), (1, 7, -1.0),
    (2, 2, 1.0), (2, 8, -1.0),
    (3, 3, 1.0), (3, 9, -1.0),
])

a3 = _dense_from_entries([(0, 1, 1.0), (1, 1, 1.0)])
a4 = _dense_from_entries([(0, 4, 1.0), (4, 4, 1.0)])

a5 = _dense_from_entries([(0, 2, 1.0), (2, 2, 1.0)])
a6 = _dense_from_entries([(0, 5, 1.0), (5, 5, 1.0)])

a7 = _dense_from_entries([(0, 3, 1.0), (3, 3, 1.0)])
a8 = _dense_from_entries([(0, 6, 1.0), (6, 6, 1.0)])

a9 = _dense_from_entries([(1, 7, 1.0), (4, 7, 1.0), (7, 7, 2.0)])
a10 = _dense_from_entries([(2, 8, 1.0), (5, 8, 1.0), (8, 8, 2.0)])
a11 = _dense_from_entries([(3, 9, 1.0), (6, 9, 1.0), (9, 9, 2.0)])


# -------- eps = 0 --------
a1_eps0 = _dense_from_entries([
    (0, 1, -1.0), (0, 2, -1.0), (0, 3, -1.0),
    (4, 7, -1.0),
    (5, 8, -1.0),
    (6, 9, -1.0),
])
a2_eps0 = _dense_from_entries([
    (0, 4, -1.0), (0, 5, -1.0), (0, 6, -1.0),
    (1, 7, -1.0),
    (2, 8, -1.0),
    (3, 9, -1.0),
])

a3_eps0 = _dense_from_entries([(0, 1, 1.0)])
a4_eps0 = _dense_from_entries([(0, 4, 1.0)])

a5_eps0 = _dense_from_entries([(0, 2, 1.0)])
a6_eps0 = _dense_from_entries([(0, 5, 1.0)])

a7_eps0 = _dense_from_entries([(0, 3, 1.0)])
a8_eps0 = _dense_from_entries([(0, 6, 1.0)])

a9_eps0 = _dense_from_entries([(1, 7, 1.0), (4, 7, 1.0)])
a10_eps0 = _dense_from_entries([(2, 8, 1.0), (5, 8, 1.0)])
a11_eps0 = _dense_from_entries([(3, 9, 1.0), (6, 9, 1.0)])


__all__ = [
    "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11",
    "a1_eps0", "a2_eps0", "a3_eps0", "a4_eps0", "a5_eps0", "a6_eps0",
    "a7_eps0", "a8_eps0", "a9_eps0", "a10_eps0", "a11_eps0",
]
