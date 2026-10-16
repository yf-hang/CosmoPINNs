# Empty-subset-first convention: S0={}, S1={y1}, S2={c}.
import mpmath as mp
from two_site_chain.sol_chain import eps_to_n_int

# ---------------- 1-loop letters ----------------
def _ws(x1: float, x2: float, y1: float, c: float):
    x1m, x2m, y1m, cm = mp.mpf(x1), mp.mpf(x2), mp.mpf(y1), mp.mpf(c)

    w1 = x1m + y1m + cm
    w2 = x2m + y1m + cm
    w3 = x1m - y1m - cm
    w4 = x2m - y1m - cm
    w5 = x1m + y1m - cm
    w6 = x2m + y1m - cm
    w7 = x1m - y1m + cm
    w8 = x2m - y1m + cm
    w9 = x1m + x2m
    w10 = x1m + x2m + 2 * y1m
    w11 = x1m + x2m + 2 * cm
    return (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11)

def _w_select(x1: float, x2: float, y1: float, c: float, *idx):
    ws = _ws(x1, x2, y1, c)
    return tuple(ws[i - 1] for i in idx)

# -----------------------------------------
# ---------------- eps = 0 ----------------
# -----------------------------------------
def I1_eps0(x1: float, x2: float, y1: float, c: float):
    (w1, w2, w3, w4, w5, w6, w7, w8, _, _, _) = _ws(x1, x2, y1, c)

    return (-(mp.pi**2) / 6
            + mp.polylog(2, w5 / w1)
            + mp.polylog(2, w6 / w2)
            - mp.polylog(2, (w5 * w6) / (w1 * w2))
            + mp.polylog(2, w7 / w1)
            + mp.polylog(2, w8 / w2)
            - mp.polylog(2, (w7 * w8) / (w1 * w2))
            - mp.polylog(2, w3 / w1)
            - mp.polylog(2, w4 / w2)
            + mp.polylog(2, (w3 * w4) / (w1 * w2)))

def I2_eps0(x1: float, x2: float, y1: float, c: float):
    w2, w9 = _w_select(x1, x2, y1, c, 2, 9)
    return -mp.log(w9 / w2)

def I3_eps0(x1: float, x2: float, y1: float, c: float):
    w2, w10 = _w_select(x1, x2, y1, c, 2, 10)
    return mp.log(w10 / w2)

def I4_eps0(x1: float, x2: float, y1: float, c: float):
    w2, w11 = _w_select(x1, x2, y1, c, 2, 11)
    return mp.log(w11 / w2)

def I5_eps0(x1: float, x2: float, y1: float, c: float):
    w1, w9 = _w_select(x1, x2, y1, c, 1, 9)
    return -mp.log(w9 / w1)

def I6_eps0(x1: float, x2: float, y1: float, c: float):
    w1, w10 = _w_select(x1, x2, y1, c, 1, 10)
    return mp.log(w10 / w1)

def I7_eps0(x1: float, x2: float, y1: float, c: float):
    w1, w11 = _w_select(x1, x2, y1, c, 1, 11)
    return mp.log(w11 / w1)

def I8_eps0(_x1: float, _x2: float, _y1: float, _c: float):
    return mp.mpf("-1")

def I9_eps0(_x1: float, _x2: float, _y1: float, _c: float):
    return mp.mpf("1")

def I10_eps0(_x1: float, _x2: float, _y1: float, _c: float):
    return mp.mpf("1")

# ------------------------------------------
# ---------------- eps = -n ----------------
# ------------------------------------------
def _jacobi_P_int(n: int, x):
    z = (1 - x) / 2
    return mp.binomial(2*n - 1, n - 1) * mp.hyper([1 - n, n], [n + 1], z)

def _int_branch_series_sum(w_main, w_main2, n: int):
    # Sum_{k=0}^{n-1} C(n+k,k) * w_main^k * w_main2^(n-k-1)
    return mp.fsum(
        mp.binomial(n + k, k) * (w_main ** k) * (w_main2 ** (n - k - 1))
        for k in range(n)
    )

def I1_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11) = _ws(x1, x2, y1, c)

    term0 = 1 / (w1**n * w2**n)
    term10 = (
        _jacobi_P_int(n, 1 - 2 * w5 / w10) / (w10**n * w2**n)
        + _jacobi_P_int(n, 1 - 2 * w6 / w10) / (w10**n * w1**n)
    )
    term11 = (
        _jacobi_P_int(n, 1 - 2 * w7 / w11) / (w11**n * w2**n)
        + _jacobi_P_int(n, 1 - 2 * w8 / w11) / (w11**n * w1**n)
    )
    term9 = (
        _jacobi_P_int(n, 1 - 2 * w3 / w9) / (w9**n * w2**n)
        + _jacobi_P_int(n, 1 - 2 * w4 / w9) / (w9**n * w1**n)
    )

    return term0 - term10 - term11 + term9

def I2_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w2, w3, w9 = _w_select(x1, x2, y1, c, 2, 3, 9)
    s29 = _int_branch_series_sum(w2, w9, n)
    return (w3 / (w2**n * w9**(2*n))) * s29

def I3_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w2, w5, w10 = _w_select(x1, x2, y1, c, 2, 5, 10)
    s210 = _int_branch_series_sum(w2, w10, n)
    return -(w5 / (w2**n * w10**(2*n))) * s210

def I4_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w2, w7, w11 = _w_select(x1, x2, y1, c, 2, 7, 11)
    s211 = _int_branch_series_sum(w2, w11, n)
    return -(w7 / (w2**n * w11**(2*n))) * s211

def I5_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w4, w9 = _w_select(x1, x2, y1, c, 1, 4, 9)
    s19 = _int_branch_series_sum(w1, w9, n)
    return (w4 / (w1**n * w9**(2*n))) * s19

def I6_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w6, w10 = _w_select(x1, x2, y1, c, 1, 6, 10)
    s110 = _int_branch_series_sum(w1, w10, n)
    return -(w6 / (w1**n * w10**(2*n))) * s110

def I7_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    w1, w8, w11 = _w_select(x1, x2, y1, c, 1, 8, 11)
    s111 = _int_branch_series_sum(w1, w11, n)
    return -(w8 / (w1**n * w11**(2*n))) * s111

def I8_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    (w9,) = _w_select(x1, x2, y1, c, 9)
    return -mp.binomial(2*n, n) / (w9 ** (2*n))

def I9_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    (w10,) = _w_select(x1, x2, y1, c, 10)
    return mp.binomial(2*n, n) / (w10 ** (2*n))

def I10_eps_int(x1: float, x2: float, y1: float, n: int, c: float):
    (w11,) = _w_select(x1, x2, y1, c, 11)
    return mp.binomial(2*n, n) / (w11 ** (2*n))

# ---------------- final solutions ----------------
EPS_TOL = mp.mpf("1e-12")


def _raise_unsupported_eps(eps):
    raise ValueError(
        "sol_1loop.py currently supports only "
        "eps=0 or eps=-n; "
        f"got eps={eps}."
    )

def I1_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp0=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I1_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I1_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)

def I2_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I2_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I2_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)

def I3_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I3_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I3_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)

def I4_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I4_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I4_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)

def I5_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I5_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I5_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)

def I6_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I6_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I6_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)

def I7_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp1=None, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I7_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I7_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)

def I8_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I8_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I8_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)

def I9_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I9_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I9_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)

def I10_fin(x1: float, x2: float, y1: float, eps: float, c: float, *, _alp2=None):
    eps = mp.mpf(eps)
    if mp.fabs(eps) < EPS_TOL:
        return I10_eps0(x1, x2, y1, c)

    n_int = eps_to_n_int(eps)
    if n_int is not None:
        return I10_eps_int(x1, x2, y1, n_int, c)

    _raise_unsupported_eps(eps)
