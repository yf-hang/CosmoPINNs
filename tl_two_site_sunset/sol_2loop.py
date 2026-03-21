import mpmath as mp
from two_site_chain.sol_chain import eps_to_n_int, eps_to_n_pos_int

# ---------------- 2-loop letters ----------------
def _ws(x1: float, x2: float, y1: float, y2: float, c: float):
    x1m, x2m, y1m, y2m, cm = mp.mpf(x1), mp.mpf(x2), mp.mpf(y1), mp.mpf(y2), mp.mpf(c)

    w1 = x1m + y1m + y2m + cm
    w2 = x2m + y1m + y2m + cm

    w9 = x1m + y1m + y2m - cm
    w10 = x2m + y1m + y2m - cm

    w11 = x1m + y1m - y2m + cm
    w12 = x2m + y1m - y2m + cm

    w13 = x1m - y1m + y2m + cm
    w14 = x2m - y1m + y2m + cm

    w3 = x1m + y1m - y2m - cm
    w4 = x2m + y1m - y2m - cm

    w5 = x1m - y1m + y2m - cm
    w6 = x2m - y1m + y2m - cm

    w7 = x1m - y1m - y2m + cm
    w8 = x2m - y1m - y2m + cm

    w15 = x1m - y1m - y2m - cm
    w16 = x2m - y1m - y2m - cm

    w20 = x1m + x2m + 2*y1m + 2*y2m
    w21 = x1m + x2m + 2*y1m + 2*cm
    w22 = x1m + x2m + 2*y2m + 2*cm

    w17 = x1m + x2m + 2*y1m
    w18 = x1m + x2m + 2*y2m
    w19 = x1m + x2m + 2*cm

    w23 = x1m + x2m

    return (w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18,
           w19, w20, w21, w22, w23)

def _w_select(x1: float, x2: float, y1: float, y2: float, c: float, *idx):
    ws = _ws(x1, x2, y1, y2, c)
    return tuple(ws[i - 1] for i in idx)

# -----------------------------------------
# ---------------- eps = 0 ----------------
# -----------------------------------------
def I1_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    (w1, w2, w3, w4, w5, w6, w7, w8,
     w9, w10, w11, w12, w13, w14, w15, w16) = _w_select(
        x1, x2, y1, y2, c,
        1, 2, 3, 4, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 15, 16
    )

    return (
        mp.polylog(2, w3 / w1)
        + mp.polylog(2, w4 / w2)
        - mp.polylog(2, (w3 * w4) / (w1 * w2))
        + mp.polylog(2, w5 / w1)
        + mp.polylog(2, w6 / w2)
        - mp.polylog(2, (w5 * w6) / (w1 * w2))
        + mp.polylog(2, w7 / w1)
        + mp.polylog(2, w8 / w2)
        - mp.polylog(2, (w7 * w8) / (w1 * w2))
        - mp.polylog(2, w9 / w1)
        - mp.polylog(2, w10 / w2)
        + mp.polylog(2, (w9 * w10) / (w1 * w2))
        - mp.polylog(2, w11 / w1)
        - mp.polylog(2, w12 / w2)
        + mp.polylog(2, (w11 * w12) / (w1 * w2))
        - mp.polylog(2, w13 / w1)
        - mp.polylog(2, w14 / w2)
        + mp.polylog(2, (w13 * w14) / (w1 * w2))
        - mp.polylog(2, w15 / w1)
        - mp.polylog(2, w16 / w2)
        + mp.polylog(2, (w15 * w16) / (w1 * w2))
        + (mp.pi**2) / 6
    )

def I2_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w17, w2 = _w_select(x1, x2, y1, y2, c, 17, 2)
    return mp.log(w17 / w2)

def I3_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w18, w2 = _w_select(x1, x2, y1, y2, c, 18, 2)
    return mp.log(w18 / w2)

def I4_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w19, w2 = _w_select(x1, x2, y1, y2, c, 19, 2)
    return mp.log(w19 / w2)

def I5_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w2, w20 = _w_select(x1, x2, y1, y2, c, 2, 20)
    return mp.log(w2 / w20)

def I6_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w2, w21 = _w_select(x1, x2, y1, y2, c, 2, 21)
    return mp.log(w2 / w21)

def I7_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w2, w22 = _w_select(x1, x2, y1, y2, c, 2, 22)
    return mp.log(w2 / w22)

def I8_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w2, w23 = _w_select(x1, x2, y1, y2, c, 2, 23)
    return mp.log(w2 / w23)

def I9_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w17, w1 = _w_select(x1, x2, y1, y2, c, 17, 1)
    return mp.log(w17 / w1)

def I10_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w18, w1 = _w_select(x1, x2, y1, y2, c, 18, 1)
    return mp.log(w18 / w1)

def I11_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w19, w1 = _w_select(x1, x2, y1, y2, c, 19, 1)
    return mp.log(w19 / w1)

def I12_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w1, w20 = _w_select(x1, x2, y1, y2, c, 1, 20)
    return mp.log(w1 / w20)

def I13_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w1, w21 = _w_select(x1, x2, y1, y2, c, 1, 21)
    return mp.log(w1 / w21)

def I14_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w1, w22 = _w_select(x1, x2, y1, y2, c, 1, 22)
    return mp.log(w1 / w22)

def I15_eps0(x1: float, x2: float, y1: float, y2: float, c: float):
    w1, w23 = _w_select(x1, x2, y1, y2, c, 1, 23)
    return mp.log(w1 / w23)

def I16_eps0(_x1: float, _x2: float, _y1: float, _y2: float, _c: float):
    return mp.mpf("1")

def I17_eps0(_x1: float, _x2: float, _y1: float, _y2: float, _c: float):
    return mp.mpf("1")

def I18_eps0(_x1: float, _x2: float, _y1: float, _y2: float, _c: float):
    return mp.mpf("1")

def I19_eps0(_x1: float, _x2: float, _y1: float, _y2: float, _c: float):
    return mp.mpf("-1")

def I20_eps0(_x1: float, _x2: float, _y1: float, _y2: float, _c: float):
    return mp.mpf("-1")

def I21_eps0(_x1: float, _x2: float, _y1: float, _y2: float, _c: float):
    return mp.mpf("-1")

def I22_eps0(_x1: float, _x2: float, _y1: float, _y2: float, _c: float):
    return mp.mpf("-1")

# ------------------------------------------
# ---------------- eps = +n ----------------
# ------------------------------------------
def _pos_int_branch_sum(w_main, w_alt, n: int):
    coeff = mp.binomial(2 * n, n)
    series = mp.fsum(
        (
            mp.binomial(2 * n - 1, n + k - 1) * (w_alt ** (n + k)) * (w_main ** (n - k))
            - (mp.binomial(2 * n - 1, n - k - 1) if (n - k - 1) >= 0 else mp.mpf("0"))
            * (w_alt ** (n - k)) * (w_main ** (n + k))
        ) / k
        for k in range(1, n + 1)
    )
    prefactor = ((-1) ** (n + 1)) * mp.mpf("0.5") * (w_main ** n) * (w_alt ** n)
    log_term = mp.mpf("0.5") * (w_main ** n) * (w_alt ** n) * mp.log(w_alt / w_main)
    return prefactor + log_term + series / coeff

def _pos_int_branch(w_main, w_alt, n: int, *, sign=1):
    return mp.mpf(sign) * _pos_int_branch_sum(w_main, w_alt, n)

def _pos_int_const_branch(w_main, n: int, *, sign=1):
    return mp.mpf(sign) * (w_main ** (2 * n)) / (n * mp.binomial(2 * n, n))

def I1_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w2 = _w_select(x1, x2, y1, y2, c, 1, 2)
    return ( - (w1**n * w2**n)
        + (I2_eps_pos_int(x1, x2, y1, y2, n, c) + I9_eps_pos_int(x1, x2, y1, y2, n, c) - I16_eps_pos_int(x1, x2, y1, y2, n, c))
        + (I3_eps_pos_int(x1, x2, y1, y2, n, c) + I10_eps_pos_int(x1, x2, y1, y2, n, c) - I17_eps_pos_int(x1, x2, y1, y2, n, c))
        + (I4_eps_pos_int(x1, x2, y1, y2, n, c) + I11_eps_pos_int(x1, x2, y1, y2, n, c) - I18_eps_pos_int(x1, x2, y1, y2, n, c))
        + (I5_eps_pos_int(x1, x2, y1, y2, n, c) + I12_eps_pos_int(x1, x2, y1, y2, n, c) - I19_eps_pos_int(x1, x2, y1, y2, n, c))
        + (I6_eps_pos_int(x1, x2, y1, y2, n, c) + I13_eps_pos_int(x1, x2, y1, y2, n, c) - I20_eps_pos_int(x1, x2, y1, y2, n, c))
        + (I7_eps_pos_int(x1, x2, y1, y2, n, c) + I14_eps_pos_int(x1, x2, y1, y2, n, c) - I21_eps_pos_int(x1, x2, y1, y2, n, c))
        + (I8_eps_pos_int(x1, x2, y1, y2, n, c) + I15_eps_pos_int(x1, x2, y1, y2, n, c) - I22_eps_pos_int(x1, x2, y1, y2, n, c))
    )

def I2_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w3 = _w_select(x1, x2, y1, y2, c, 2, 3)
    return _pos_int_branch(w2, w3, n)

def I3_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w5 = _w_select(x1, x2, y1, y2, c, 2, 5)
    return _pos_int_branch(w2, w5, n)

def I4_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w7 = _w_select(x1, x2, y1, y2, c, 2, 7)
    return _pos_int_branch(w2, w7, n)

def I5_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w9 = _w_select(x1, x2, y1, y2, c, 2, 9)
    return _pos_int_branch(w2, w9, n, sign=-1)

def I6_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w11 = _w_select(x1, x2, y1, y2, c, 2, 11)
    return _pos_int_branch(w2, w11, n, sign=-1)

def I7_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w13 = _w_select(x1, x2, y1, y2, c, 2, 13)
    return _pos_int_branch(w2, w13, n, sign=-1)

def I8_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w15 = _w_select(x1, x2, y1, y2, c, 2, 15)
    return _pos_int_branch(w2, w15, n, sign=-1)

def I9_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w4 = _w_select(x1, x2, y1, y2, c, 1, 4)
    return _pos_int_branch(w1, w4, n)

def I10_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w6 = _w_select(x1, x2, y1, y2, c, 1, 6)
    return _pos_int_branch(w1, w6, n)

def I11_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w8 = _w_select(x1, x2, y1, y2, c, 1, 8)
    return _pos_int_branch(w1, w8, n)

def I12_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w10 = _w_select(x1, x2, y1, y2, c, 1, 10)
    return _pos_int_branch(w1, w10, n, sign=-1)

def I13_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w12 = _w_select(x1, x2, y1, y2, c, 1, 12)
    return _pos_int_branch(w1, w12, n, sign=-1)

def I14_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w14 = _w_select(x1, x2, y1, y2, c, 1, 14)
    return _pos_int_branch(w1, w14, n, sign=-1)

def I15_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w16 = _w_select(x1, x2, y1, y2, c, 1, 16)
    return _pos_int_branch(w1, w16, n, sign=-1)

def I16_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w17,) = _w_select(x1, x2, y1, y2, c, 17)
    return _pos_int_const_branch(w17, n)

def I17_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w18,) = _w_select(x1, x2, y1, y2, c, 18)
    return _pos_int_const_branch(w18, n)

def I18_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w19,) = _w_select(x1, x2, y1, y2, c, 19)
    return _pos_int_const_branch(w19, n)

def I19_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w20,) = _w_select(x1, x2, y1, y2, c, 20)
    return _pos_int_const_branch(w20, n, sign=-1)

def I20_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w21,) = _w_select(x1, x2, y1, y2, c, 21)
    return _pos_int_const_branch(w21, n, sign=-1)

def I21_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w22,) = _w_select(x1, x2, y1, y2, c, 22)
    return _pos_int_const_branch(w22, n, sign=-1)

def I22_eps_pos_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w23,) = _w_select(x1, x2, y1, y2, c, 23)
    return _pos_int_const_branch(w23, n, sign=-1)

# ------------------------------------------
# ---------------- eps = -n ----------------
# ------------------------------------------
def _jacobi_P_int(n: int, x):
    # P_{n-1}^{(n,-n)}(x)
    z = (1 - x) / 2
    return mp.binomial(2 * n - 1, n - 1) * mp.hyper([1 - n, n], [n + 1], z)


def _int_branch_series_sum(w_main, w_main2, n: int):
    # Sum_{k=0}^{n-1} C(n+k,k) * w_main^k * w_main2^(n-k-1)
    return mp.fsum(
        mp.binomial(n + k, k) * (w_main ** k) * (w_main2 ** (n - k - 1))
        for k in range(n)
    )


def I1_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w2 = _w_select(x1, x2, y1, y2, c, 1, 2)
    return (
        -1 / (w1**n * w2**n)
        + (I2_eps_int(x1, x2, y1, y2, n, c) + I9_eps_int(x1, x2, y1, y2, n, c) - I16_eps_int(x1, x2, y1, y2, n, c))
        + (I3_eps_int(x1, x2, y1, y2, n, c) + I10_eps_int(x1, x2, y1, y2, n, c) - I17_eps_int(x1, x2, y1, y2, n, c))
        + (I4_eps_int(x1, x2, y1, y2, n, c) + I11_eps_int(x1, x2, y1, y2, n, c) - I18_eps_int(x1, x2, y1, y2, n, c))
        + (I5_eps_int(x1, x2, y1, y2, n, c) + I12_eps_int(x1, x2, y1, y2, n, c) - I19_eps_int(x1, x2, y1, y2, n, c))
        + (I6_eps_int(x1, x2, y1, y2, n, c) + I13_eps_int(x1, x2, y1, y2, n, c) - I20_eps_int(x1, x2, y1, y2, n, c))
        + (I7_eps_int(x1, x2, y1, y2, n, c) + I14_eps_int(x1, x2, y1, y2, n, c) - I21_eps_int(x1, x2, y1, y2, n, c))
        + (I8_eps_int(x1, x2, y1, y2, n, c) + I15_eps_int(x1, x2, y1, y2, n, c) - I22_eps_int(x1, x2, y1, y2, n, c))
    )


def I2_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w3, w17 = _w_select(x1, x2, y1, y2, c, 2, 3, 17)
    s = _int_branch_series_sum(w2, w17, n)
    return -(w3 / (w2**n * w17**(2 * n))) * s

def I3_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w5, w18 = _w_select(x1, x2, y1, y2, c, 2, 5, 18)
    s = _int_branch_series_sum(w2, w18, n)
    return -(w5 / (w2**n * w18**(2 * n))) * s

def I4_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w7, w19 = _w_select(x1, x2, y1, y2, c, 2, 7, 19)
    s = _int_branch_series_sum(w2, w19, n)
    return -(w7 / (w2**n * w19**(2 * n))) * s

def I5_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w9, w20 = _w_select(x1, x2, y1, y2, c, 2, 9, 20)
    s = _int_branch_series_sum(w2, w20, n)
    return (w9 / (w2**n * w20**(2 * n))) * s

def I6_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w11, w21 = _w_select(x1, x2, y1, y2, c, 2, 11, 21)
    s = _int_branch_series_sum(w2, w21, n)
    return (w11 / (w2**n * w21**(2 * n))) * s

def I7_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w13, w22 = _w_select(x1, x2, y1, y2, c, 2, 13, 22)
    s = _int_branch_series_sum(w2, w22, n)
    return (w13 / (w2**n * w22**(2 * n))) * s

def I8_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w2, w15, w23 = _w_select(x1, x2, y1, y2, c, 2, 15, 23)
    s = _int_branch_series_sum(w2, w23, n)
    return (w15 / (w2**n * w23**(2 * n))) * s


def I9_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w4, w17 = _w_select(x1, x2, y1, y2, c, 1, 4, 17)
    s = _int_branch_series_sum(w1, w17, n)
    return -(w4 / (w1**n * w17**(2 * n))) * s

def I10_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w6, w18 = _w_select(x1, x2, y1, y2, c, 1, 6, 18)
    s = _int_branch_series_sum(w1, w18, n)
    return -(w6 / (w1**n * w18**(2 * n))) * s

def I11_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w8, w19 = _w_select(x1, x2, y1, y2, c, 1, 8, 19)
    s = _int_branch_series_sum(w1, w19, n)
    return -(w8 / (w1**n * w19**(2 * n))) * s

def I12_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w10, w20 = _w_select(x1, x2, y1, y2, c, 1, 10, 20)
    s = _int_branch_series_sum(w1, w20, n)
    return (w10 / (w1**n * w20**(2 * n))) * s

def I13_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w12, w21 = _w_select(x1, x2, y1, y2, c, 1, 12, 21)
    s = _int_branch_series_sum(w1, w21, n)
    return (w12 / (w1**n * w21**(2 * n))) * s

def I14_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w14, w22 = _w_select(x1, x2, y1, y2, c, 1, 14, 22)
    s = _int_branch_series_sum(w1, w22, n)
    return (w14 / (w1**n * w22**(2 * n))) * s

def I15_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    w1, w16, w23 = _w_select(x1, x2, y1, y2, c, 1, 16, 23)
    s = _int_branch_series_sum(w1, w23, n)
    return (w16 / (w1**n * w23**(2 * n))) * s


def I16_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w17,) = _w_select(x1, x2, y1, y2, c, 17)
    return mp.binomial(2*n, n) / (w17 ** (2*n))

def I17_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w18,) = _w_select(x1, x2, y1, y2, c, 18)
    return mp.binomial(2*n, n) / (w18 ** (2*n))

def I18_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w19,) = _w_select(x1, x2, y1, y2, c, 19)
    return mp.binomial(2*n, n) / (w19 ** (2*n))

def I19_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w20,) = _w_select(x1, x2, y1, y2, c, 20)
    return -mp.binomial(2*n, n) / (w20 ** (2*n))

def I20_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w21,) = _w_select(x1, x2, y1, y2, c, 21)
    return -mp.binomial(2*n, n) / (w21 ** (2*n))

def I21_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w22,) = _w_select(x1, x2, y1, y2, c, 22)
    return -mp.binomial(2*n, n) / (w22 ** (2*n))

def I22_eps_int(x1: float, x2: float, y1: float, y2: float, n: int, c: float):
    (w23,) = _w_select(x1, x2, y1, y2, c, 23)
    return -mp.binomial(2*n, n) / (w23 ** (2*n))

# ---------------- final solutions ----------------
EPS_TOL = mp.mpf("1e-12")

def _dispatch_eps_case(eps):
    eps_mpf = mp.mpf(eps)
    if mp.fabs(eps_mpf) < EPS_TOL:
        return "eps0", None
    n_pos_int = eps_to_n_pos_int(eps_mpf)
    if n_pos_int is not None:
        return "pos-int", int(n_pos_int)
    n_int = eps_to_n_int(eps_mpf)
    if n_int is not None:
        return "int", int(n_int)
    raise ValueError(
        f"sol_2loop currently supports only eps=0, eps=+n (positive integer), or eps=-n (negative integer), got eps={eps}."
    )


def _eval_fin(fn_eps0, fn_eps_pos_int, fn_eps_int, x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    case, n = _dispatch_eps_case(eps)
    if case == "eps0":
        return fn_eps0(x1, x2, y1, y2, c)
    if case == "pos-int":
        return fn_eps_pos_int(x1, x2, y1, y2, n, c)
    return fn_eps_int(x1, x2, y1, y2, n, c)


def I1_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I1_eps0, I1_eps_pos_int, I1_eps_int, x1, x2, y1, y2, eps, c)


def I2_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I2_eps0, I2_eps_pos_int, I2_eps_int, x1, x2, y1, y2, eps, c)


def I3_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I3_eps0, I3_eps_pos_int, I3_eps_int, x1, x2, y1, y2, eps, c)


def I4_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I4_eps0, I4_eps_pos_int, I4_eps_int, x1, x2, y1, y2, eps, c)


def I5_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I5_eps0, I5_eps_pos_int, I5_eps_int, x1, x2, y1, y2, eps, c)


def I6_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I6_eps0, I6_eps_pos_int, I6_eps_int, x1, x2, y1, y2, eps, c)


def I7_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I7_eps0, I7_eps_pos_int, I7_eps_int, x1, x2, y1, y2, eps, c)


def I8_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I8_eps0, I8_eps_pos_int, I8_eps_int, x1, x2, y1, y2, eps, c)


def I9_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I9_eps0, I9_eps_pos_int, I9_eps_int, x1, x2, y1, y2, eps, c)


def I10_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I10_eps0, I10_eps_pos_int, I10_eps_int, x1, x2, y1, y2, eps, c)


def I11_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I11_eps0, I11_eps_pos_int, I11_eps_int, x1, x2, y1, y2, eps, c)


def I12_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I12_eps0, I12_eps_pos_int, I12_eps_int, x1, x2, y1, y2, eps, c)


def I13_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I13_eps0, I13_eps_pos_int, I13_eps_int, x1, x2, y1, y2, eps, c)


def I14_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I14_eps0, I14_eps_pos_int, I14_eps_int, x1, x2, y1, y2, eps, c)


def I15_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I15_eps0, I15_eps_pos_int, I15_eps_int, x1, x2, y1, y2, eps, c)


def I16_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I16_eps0, I16_eps_pos_int, I16_eps_int, x1, x2, y1, y2, eps, c)


def I17_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I17_eps0, I17_eps_pos_int, I17_eps_int, x1, x2, y1, y2, eps, c)


def I18_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I18_eps0, I18_eps_pos_int, I18_eps_int, x1, x2, y1, y2, eps, c)


def I19_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I19_eps0, I19_eps_pos_int, I19_eps_int, x1, x2, y1, y2, eps, c)


def I20_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I20_eps0, I20_eps_pos_int, I20_eps_int, x1, x2, y1, y2, eps, c)


def I21_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I21_eps0, I21_eps_pos_int, I21_eps_int, x1, x2, y1, y2, eps, c)


def I22_fin(x1: float, x2: float, y1: float, y2: float, eps: float, c: float):
    return _eval_fin(I22_eps0, I22_eps_pos_int, I22_eps_int, x1, x2, y1, y2, eps, c)
