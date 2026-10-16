# Empty-subset-first convention: S0={}, S1={y1}, S2={y2}, S3={c}, S4={y1,y2}, S5={y1,c}, S6={y2,c}.
# a_k multiplies dlog(w_k) in the new I basis (also for a_k_eps0).
# The simultaneous basis/letter permutation leaves these sparse entries unchanged.
import torch

_DTYPE = torch.float32
_SHAPE = (22, 22)


def _dense_from_entries(entries):
    mat = torch.zeros(_SHAPE, dtype=_DTYPE)
    if not entries:
        return mat
    rows, cols, vals = zip(*entries)
    mat[list(rows), list(cols)] = torch.tensor(vals, dtype=_DTYPE)
    return mat


a1 = _dense_from_entries([
    (0, 0, 1.0), (0, 1, -1.0), (0, 2, -1.0), (0, 3, -1.0),
    (0, 4, -1.0), (0, 5, -1.0), (0, 6, -1.0), (0, 7, -1.0),
    (8, 8, 1.0), (8, 15, -1.0),
    (9, 9, 1.0), (9, 16, -1.0),
    (10, 10, 1.0), (10, 17, -1.0),
    (11, 11, 1.0), (11, 18, -1.0),
    (12, 12, 1.0), (12, 19, -1.0),
    (13, 13, 1.0), (13, 20, -1.0),
    (14, 14, 1.0), (14, 21, -1.0),
])
a2 = _dense_from_entries([
    (0, 0, 1.0), (0, 8, -1.0), (0, 9, -1.0), (0, 10, -1.0),
    (0, 11, -1.0), (0, 12, -1.0), (0, 13, -1.0), (0, 14, -1.0),
    (1, 1, 1.0), (1, 15, -1.0),
    (2, 2, 1.0), (2, 16, -1.0),
    (3, 3, 1.0), (3, 17, -1.0),
    (4, 4, 1.0), (4, 18, -1.0),
    (5, 5, 1.0), (5, 19, -1.0),
    (6, 6, 1.0), (6, 20, -1.0),
    (7, 7, 1.0), (7, 21, -1.0),
])

a3 = _dense_from_entries([(0, 1, 1.0), (1, 1, 1.0)])
a4 = _dense_from_entries([(0, 8, 1.0), (8, 8, 1.0)])

a5 = _dense_from_entries([(0, 2, 1.0), (2, 2, 1.0)])
a6 = _dense_from_entries([(0, 9, 1.0), (9, 9, 1.0)])

a7 = _dense_from_entries([(0, 3, 1.0), (3, 3, 1.0)])
a8 = _dense_from_entries([(0, 10, 1.0), (10, 10, 1.0)])

a9 = _dense_from_entries([(0, 4, 1.0), (4, 4, 1.0)])
a10 = _dense_from_entries([(0, 11, 1.0), (11, 11, 1.0)])

a11 = _dense_from_entries([(0, 5, 1.0), (5, 5, 1.0)])
a12 = _dense_from_entries([(0, 12, 1.0), (12, 12, 1.0)])

a13 = _dense_from_entries([(0, 6, 1.0), (6, 6, 1.0)])
a14 = _dense_from_entries([(0, 13, 1.0), (13, 13, 1.0)])

a15 = _dense_from_entries([(0, 7, 1.0), (7, 7, 1.0)])
a16 = _dense_from_entries([(0, 14, 1.0), (14, 14, 1.0)])

a17 = _dense_from_entries([(1, 15, 1.0), (8, 15, 1.0), (15, 15, 2.0)])
a18 = _dense_from_entries([(2, 16, 1.0), (9, 16, 1.0), (16, 16, 2.0)])
a19 = _dense_from_entries([(3, 17, 1.0), (10, 17, 1.0), (17, 17, 2.0)])
a20 = _dense_from_entries([(4, 18, 1.0), (11, 18, 1.0), (18, 18, 2.0)])
a21 = _dense_from_entries([(5, 19, 1.0), (12, 19, 1.0), (19, 19, 2.0)])
a22 = _dense_from_entries([(6, 20, 1.0), (13, 20, 1.0), (20, 20, 2.0)])
a23 = _dense_from_entries([(7, 21, 1.0), (14, 21, 1.0), (21, 21, 2.0)])


# -------- eps = 0 --------
a1_eps0 = _dense_from_entries([
    (0, 1, -1.0), (0, 2, -1.0), (0, 3, -1.0),
    (0, 4, -1.0), (0, 5, -1.0), (0, 6, -1.0), (0, 7, -1.0),
    (8, 15, -1.0),
    (9, 16, -1.0),
    (10, 17, -1.0),
    (11, 18, -1.0),
    (12, 19, -1.0),
    (13, 20, -1.0),
    (14, 21, -1.0),
])
a2_eps0 = _dense_from_entries([
    (0, 8, -1.0), (0, 9, -1.0), (0, 10, -1.0),
    (0, 11, -1.0), (0, 12, -1.0), (0, 13, -1.0), (0, 14, -1.0),
    (1, 15, -1.0),
    (2, 16, -1.0),
    (3, 17, -1.0),
    (4, 18, -1.0),
    (5, 19, -1.0),
    (6, 20, -1.0),
    (7, 21, -1.0),
])

a3_eps0 = _dense_from_entries([(0, 1, 1.0)])
a4_eps0 = _dense_from_entries([(0, 8, 1.0)])

a5_eps0 = _dense_from_entries([(0, 2, 1.0)])
a6_eps0 = _dense_from_entries([(0, 9, 1.0)])

a7_eps0 = _dense_from_entries([(0, 3, 1.0)])
a8_eps0 = _dense_from_entries([(0, 10, 1.0)])

a9_eps0 = _dense_from_entries([(0, 4, 1.0)])
a10_eps0 = _dense_from_entries([(0, 11, 1.0)])

a11_eps0 = _dense_from_entries([(0, 5, 1.0)])
a12_eps0 = _dense_from_entries([(0, 12, 1.0)])

a13_eps0 = _dense_from_entries([(0, 6, 1.0)])
a14_eps0 = _dense_from_entries([(0, 13, 1.0)])

a15_eps0 = _dense_from_entries([(0, 7, 1.0)])
a16_eps0 = _dense_from_entries([(0, 14, 1.0)])

a17_eps0 = _dense_from_entries([(1, 15, 1.0), (8, 15, 1.0)])
a18_eps0 = _dense_from_entries([(2, 16, 1.0), (9, 16, 1.0)])
a19_eps0 = _dense_from_entries([(3, 17, 1.0), (10, 17, 1.0)])
a20_eps0 = _dense_from_entries([(4, 18, 1.0), (11, 18, 1.0)])
a21_eps0 = _dense_from_entries([(5, 19, 1.0), (12, 19, 1.0)])
a22_eps0 = _dense_from_entries([(6, 20, 1.0), (13, 20, 1.0)])
a23_eps0 = _dense_from_entries([(7, 21, 1.0), (14, 21, 1.0)])


__all__ = [
    "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9", "a10", "a11",
    "a12", "a13", "a14", "a15", "a16", "a17", "a18", "a19", "a20", "a21", "a22", "a23",
    "a1_eps0", "a2_eps0", "a3_eps0", "a4_eps0", "a5_eps0", "a6_eps0",
    "a7_eps0", "a8_eps0", "a9_eps0", "a10_eps0", "a11_eps0",
    "a12_eps0", "a13_eps0", "a14_eps0", "a15_eps0", "a16_eps0", "a17_eps0",
    "a18_eps0", "a19_eps0", "a20_eps0", "a21_eps0", "a22_eps0", "a23_eps0",
]
