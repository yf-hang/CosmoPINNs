"""Independent checks against the manuscript's closed forms at epsilon=0,+/-n.

References: eq:sol-chain-dS, eq:sol-chain-pn, eq:sol-chain-nn,
eq:dS-loop-sol, eq:+n-loop-sol, eq:-n-loop-sol, eq:letters-loop.
The reference below uses the manuscript's original m-sums, not the paired
k-sums/Jacobi representation used by the implementation.
"""

import importlib
import itertools
import unittest

import mpmath as mp
import numpy as np
import torch


def modules(ell):
    package, suffix = (
        ('two_site_chain', ''),
        ('tl_two_site_bubble', '_1loop'),
        ('tl_two_site_sunset', '_2loop'),
    )[ell]
    sol = importlib.import_module(f'{package}.sol_{"chain" if ell == 0 else str(ell)+"loop"}')
    bc = importlib.import_module(f'{package}.coll_bc{suffix}')
    return sol, bc, suffix


def subset_data(ell, coords, c):
    energies = list(coords[2:]) + [c]
    total = mp.fsum(energies)
    subsets = [s for size in range(ell + 1)
               for s in itertools.combinations(range(ell + 1), size)]
    shifts = [mp.fsum(energies[k] for k in subset) for subset in subsets]
    signs = [1 if ell == 0 else (-1) ** (len(s) + 1) for s in subsets]
    return total, shifts, signs


def manuscript_solution(ell, coords, c, eps):
    """Direct evaluation of the printed formulas, including I1's constant."""
    total, shifts, signs = subset_data(ell, coords, c)
    x1, x2 = coords[:2]
    w1, w2 = x1 + total, x2 + total
    left, right, bottom, leading = [], [], [], []
    n = abs(eps)
    for shift, sign in zip(shifts, signs):
        a = x1 - total + 2 * shift
        b = x2 - total + 2 * shift
        d = x1 + x2 + 2 * shift
        if eps == 0:
            left.append(sign * mp.log(d / w2))
            right.append(sign * mp.log(d / w1))
            bottom.append(mp.mpf(sign))
            leading.append(sign * (mp.polylog(2, a / w1) + mp.polylog(2, b / w2)
                                   - mp.polylog(2, a * b / (w1 * w2)) - mp.zeta(2)))
        elif eps > 0:
            def branch(u, v):
                return sign * (
                    u**n * v**n / 2 * ((-1)**(n + 1) + mp.log(v / u))
                    + mp.fsum(mp.binomial(2*n - 1, m) * v**(m+1) * u**(2*n-m-1)
                              / (m-n+1) for m in range(2*n) if m != n-1)
                    / mp.binomial(2*n, n)
                )

            left.append(branch(w2, a))
            right.append(branch(w1, b))
            bottom.append(sign * d**(2*n) / (n * mp.binomial(2*n, n)))
        else:
            def branch(u, v):
                return -sign * v * mp.fsum(
                    mp.binomial(n+m, m) * u**(m-n) / d**(m+n+1)
                    for m in range(n)
                )

            left.append(branch(w2, a))
            right.append(branch(w1, b))
            bottom.append(sign * mp.binomial(2*n, n) / d**(2*n))
    if eps == 0:
        i1 = mp.fsum(leading)
    else:
        prefactor = 1 if ell == 0 else (-1)**(ell+1)
        i1 = prefactor * (w1*w2)**eps + mp.fsum(
            f + g - h for f, g, h in zip(left, right, bottom)
        )
    return [i1, *left, *right, *bottom]


def supported_eps(ell):
    return [0, *range(-1, -9, -1)] + (list(range(1, 9)) if ell != 1 else [])


def physical_samples(ell):
    # Asymmetric points in each manuscript benchmark domain, plus a point near
    # the physical-region boundary and scaled copies for homogeneity coverage.
    rows = (
        [('21.25', '28.5', '15'), ('27.2', '22.7', '15')],
        [('31.25', '38.5', '17.2', '5'), ('37.2', '32.7', '19.4', '5')],
        [('51.25', '58.5', '22.2', '12.3', '5'), ('57.2', '52.7', '24.4', '11.1', '5')],
    )[ell]
    values = [tuple(map(mp.mpf, row)) for row in rows]
    energies = [mp.mpf('0.7'), mp.mpf('1.1'), mp.mpf('1.3')][:ell+1]
    total = mp.fsum(energies)
    values.append(tuple([total + mp.mpf('0.003'), total + mp.mpf('0.027'), *energies]))
    values.extend(tuple(v * scale for v in values[0]) for scale in (mp.mpf('.01'), mp.mpf('100')))
    return values


class IntegerSolutionTests(unittest.TestCase):
    def test_all_components_against_manuscript(self):
        with mp.workdps(70):
            for ell in range(3):
                sol, _, _ = modules(ell)
                for args in physical_samples(ell):
                    coords, c = args[:-1], args[-1]
                    for eps in supported_eps(ell):
                        expected = manuscript_solution(ell, coords, c, eps)
                        for k, ref in enumerate(expected, 1):
                            with self.subTest(ell=ell, eps=eps, I=k, point=args):
                                got = getattr(sol, f'I{k}_fin')(*coords, eps, c)
                                scale = abs(ref) if ref else mp.mpf(1)
                                self.assertLess(abs(got-ref) / scale, mp.mpf('1e-55'))

    def test_signed_shift_construction(self):
        chain, _, _ = modules(0)
        with mp.workdps(70):
            for ell in (1, 2):
                args = physical_samples(ell)[0]
                coords, c = args[:-1], args[-1]
                total, shifts, signs = subset_data(ell, coords, c)
                # Also check the manuscript bubble formula at positive n;
                # bubble's public Python dispatcher deliberately rejects it.
                for eps in [0, -1, -2, -3, -4, -5, -8, 1, 2, 3, 4, 5, 8]:
                    blocks = [[sign * getattr(chain, f'I{k}_fin')(
                        coords[0]+shift, coords[1]+shift, eps, total-shift
                    ) for k in range(1, 5)] for shift, sign in zip(shifts, signs)]
                    expected = [mp.fsum(row[0] for row in blocks)]
                    expected.extend(row[k] for k in range(1, 4) for row in blocks)
                    actual = manuscript_solution(ell, coords, c, eps)
                    for k, (ref, got) in enumerate(zip(expected, actual), 1):
                        with self.subTest(ell=ell, eps=eps, I=k):
                            self.assertLess(abs(got-ref) / abs(ref), mp.mpf('1e-55'))

    def test_integer_i1_combination_and_homogeneity(self):
        with mp.workdps(70):
            for ell in range(3):
                sol, _, _ = modules(ell)
                args = physical_samples(ell)[0]
                coords, c = args[:-1], args[-1]
                total, shifts, signs = subset_data(ell, coords, c)
                h = len(shifts)
                prefactor = 1 if ell == 0 else (-1)**(ell+1)
                self.assertEqual(sum(signs), prefactor)
                for eps in supported_eps(ell):
                    vals = [getattr(sol, f'I{k}_fin')(*coords, eps, c) for k in range(1, 3*h+2)]
                    if eps != 0:
                        ref = prefactor * ((coords[0]+total)*(coords[1]+total))**eps
                        ref += mp.fsum(vals[1+a]+vals[1+h+a]-vals[1+2*h+a] for a in range(h))
                        self.assertLess(abs(vals[0]-ref) / abs(ref), mp.mpf('1e-55'))
                    scale = mp.mpf('3.7')
                    for k, ref in enumerate(vals, 1):
                        got = getattr(sol, f'I{k}_fin')(*(v*scale for v in coords), eps, c*scale)
                        self.assertLess(abs(got/scale**(2*eps)-ref) / abs(ref), mp.mpf('1e-55'))

    def test_real_and_imaginary_boundary_targets(self):
        with mp.workdps(70):
            for ell in range(3):
                _, bc, suffix = modules(ell)
                args = physical_samples(ell)[:2]
                c = float(args[0][-1])
                x = torch.tensor([[float(v) for v in a[:-1]] for a in args], dtype=torch.float64)
                for eps in supported_eps(ell):
                    ref = np.array([[complex(v) for v in manuscript_solution(
                        ell, list(map(mp.mpf, row)), mp.mpf(c), eps
                    )] for row in x.tolist()])
                    for part in ('re', 'im'):
                        with self.subTest(ell=ell, eps=eps, part=part):
                            got = getattr(bc, f'compute_function_target_from_xcoll{suffix}')(
                                x, cy_val=c, eps_val=eps, output_part=part
                            ).numpy()
                            np.testing.assert_allclose(
                                got, ref.real if part == 're' else ref.imag,
                                rtol=2e-6 if ell < 2 else 2e-10, atol=0,
                            )

    def test_full_cde_with_high_precision_derivatives(self):
        # Construct A from the actual coefficient matrices, with high-precision
        # dlog derivatives, then differentiate every implemented scalar solution.
        with mp.workdps(70):
            for ell in range(3):
                sol, _, suffix = modules(ell)
                data = importlib.import_module(sol.__package__ + '.mat_data' + suffix)
                args = physical_samples(ell)[0]
                coords, c = list(args[:-1]), args[-1]
                letters = sol._ws(*coords, c)
                n_basis = len(letters) - 1
                dlogs = []
                for coordinate in range(len(coords)):
                    def varied_letters(t):
                        varied = coords.copy()
                        varied[coordinate] = t
                        return sol._ws(*varied, c)

                    dlogs.append([mp.diff(lambda t: varied_letters(t)[k], coords[coordinate]) / w
                                  for k, w in enumerate(letters)])
                coefficients = {}
                for ending in ('', '_eps0'):
                    entries = []
                    for k in range(1, len(letters)+1):
                        matrix = getattr(data, f'a{k}{ending}')
                        entries.extend((k-1, i, j, int(matrix[i, j]))
                                       for i, j in matrix.nonzero().tolist())
                    coefficients[ending] = entries
                for eps in supported_eps(ell):
                    vals = [getattr(sol, f'I{k}_fin')(*coords, eps, c)
                            for k in range(1, n_basis+1)]
                    for coordinate, partials in enumerate(dlogs):
                        rhs_terms = [[] for _ in range(n_basis)]
                        for k, i, j, coefficient in coefficients['_eps0' if eps == 0 else '']:
                            rhs_terms[i].append(coefficient * partials[k] * vals[j])
                        for i in range(n_basis):
                            fn = getattr(sol, f'I{i+1}_fin')

                            def varied_solution(t):
                                varied = coords.copy()
                                varied[coordinate] = t
                                return fn(*varied, eps, c)

                            lhs = mp.diff(varied_solution, coords[coordinate])
                            rhs = (1 if eps == 0 else eps) * mp.fsum(rhs_terms[i])
                            with self.subTest(ell=ell, eps=eps, I=i+1, coordinate=coordinate):
                                scale = max(abs(lhs), abs(rhs), mp.mpf('1e-80'))
                                self.assertLess(abs(lhs-rhs) / scale, mp.mpf('1e-50'))

    def test_bubble_positive_integer_is_explicitly_unsupported(self):
        sol, bc, _ = modules(1)
        for n in (1, 2, 5, 8):
            for k in range(1, 11):
                with self.assertRaisesRegex(ValueError, 'supports only'):
                    getattr(sol, f'I{k}_fin')(31, 39, 17, n, 5)
            with self.assertRaisesRegex(ValueError, 'supports only'):
                bc.compute_function_target_from_xcoll_1loop(
                    torch.tensor([[31., 39., 17.]]), cy_val=5, eps_val=n
                )


if __name__ == '__main__':
    unittest.main()
