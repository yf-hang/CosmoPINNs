"""Check the manuscript convention, analytic CDEs, and saved-run compatibility."""

import copy
import importlib
import itertools
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mpmath as mp
import numpy as np
import torch

import main_core
from lib.basis_order import BASIS_ORDER, BASIS_PERMUTATIONS, needs_basis_migration
from lib.pinn_models import PinnModel, TransferPinnModel


def system(phase):
    package = 'tl_two_site_bubble' if phase == 1 else 'tl_two_site_sunset'
    modules = [importlib.import_module(f'{package}.{stem}_{phase}loop')
               for stem in ('sol', 'coll_bc', 'conn_mat', 'mat_data')]
    sol, bc, conn, data = modules
    n_basis = len(BASIS_PERMUTATIONS[phase])
    n_letters = n_basis + 1
    builder = getattr(conn, f'ConnectionAMatricesFixedWithEps0_{phase}Loop')(
        ak_list=[getattr(data, f'a{k}') for k in range(1, n_letters + 1)],
        ak_list_eps0=[getattr(data, f'a{k}_eps0') for k in range(1, n_letters + 1)],
        cy_val=0.4,
    ).double()
    return sol, bc, builder


def points(phase):
    rows = [[6.25, 7.5, 0.8], [4.5, 5.75, 1.1]]
    if phase == 2:
        rows = [row + [0.6] for row in rows]
    return torch.tensor(rows, dtype=torch.float64)


class AnalyticOrderTests(unittest.TestCase):
    def test_letters_and_dlog_partials_match_subsets(self):
        for phase in (1, 2):
            with self.subTest(phase=phase):
                sol, bc, builder = system(phase)
                x = points(phase).requires_grad_(True)
                energies = [x[:, k] for k in range(2, phase + 2)]
                energies.append(torch.full_like(x[:, 0], 0.4))
                total = sum(energies)
                # Proper subsets: empty, singletons, then pairs (sunset).
                subsets = [subset for size in range(len(energies))
                           for subset in itertools.combinations(range(len(energies)), size)]
                shifts = [2 * sum((energies[k] for k in subset), torch.zeros_like(total))
                          for subset in subsets]
                letters = [x[:, 0] + total, x[:, 1] + total]
                for shift in shifts:
                    letters.extend((x[:, 0] - total + shift, x[:, 1] - total + shift))
                letters.extend(x[:, 0] + x[:, 1] + shift for shift in shifts)
                expected = torch.stack(letters, dim=1)
                actual = torch.stack(bc._build_ws_tensor(x, cy_val=0.4), dim=1)
                torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
                np.testing.assert_allclose(
                    [float(w) for w in sol._ws(*x[0].detach().tolist(), 0.4)],
                    expected[0].detach().numpy(), rtol=1e-14, atol=1e-14,
                )
                partials = builder.dlog_partials(
                    *(x[:, k:k+1] for k in range(x.shape[1])), 0.4
                )
                for k, letter in enumerate(letters):
                    grad = torch.autograd.grad(letter.sum(), x, retain_graph=True)[0]
                    dlog = grad / letter[:, None]
                    for coordinate, partial in enumerate(partials):
                        torch.testing.assert_close(
                            partial[:, k, 0], dlog[:, coordinate], rtol=1e-14, atol=1e-14
                        )

    def test_scalar_solutions_satisfy_cde(self):
        with mp.workdps(40):
            for phase in (1, 2):
                sol, _, builder = system(phase)
                x = points(phase)[:1]
                coords = [mp.mpf(v) for v in x[0].tolist()]
                c = mp.mpf(0.4)
                n_basis = len(BASIS_PERMUTATIONS[phase])
                n_coords = len(coords)
                matrices = builder(x)
                for eps in [0, -1, -3] + ([1, 2] if phase == 2 else []):
                    values = [getattr(sol, f'I{k}_fin')(*coords, eps, c)
                              for k in range(1, n_basis + 1)]
                    for coordinate in range(n_coords):
                        matrix = matrices[coordinate + (n_coords if eps == 0 else 0)][0]
                        factor = 1 if eps == 0 else eps
                        for k in range(n_basis):
                            with self.subTest(phase=phase, eps=eps, coordinate=coordinate, I=k+1):
                                fn = getattr(sol, f'I{k+1}_fin')

                                def varied(t):
                                    args = coords.copy()
                                    args[coordinate] = t
                                    return fn(*args, eps, c)

                                lhs = mp.diff(varied, coords[coordinate])
                                rhs = factor * mp.fsum(
                                    mp.mpf(float(matrix[k, j])) * values[j] for j in range(n_basis)
                                )
                                self.assertLess(abs(lhs - rhs) / (1 + abs(lhs)), mp.mpf('1e-11'))

    def test_fast_targets_match_scalar_solutions(self):
        for phase in (1, 2):
            _, bc, _ = system(phase)
            x = points(phase)
            for eps in [0, -1, -3] + ([1, 2] if phase == 2 else []):
                full_x = np.column_stack((x.numpy(), np.full(len(x), eps)))
                scalar = getattr(bc, f'compute_boundary_values_rescaled_{phase}loop')(full_x, 0.4)
                for part in ('re', 'im'):
                    with self.subTest(phase=phase, eps=eps, part=part):
                        fast = getattr(bc, f'compute_function_target_from_xcoll_{phase}loop')(
                            x, cy_val=0.4, eps_val=eps, output_part=part
                        )
                        expected = scalar.real if part == 're' else scalar.imag
                        np.testing.assert_allclose(fast.numpy(), expected, rtol=2e-6, atol=1e-10)
                        if phase == 2:
                            partial = bc.compute_function_target_from_xcoll_2loop(
                                x, cy_val=0.4, eps_val=eps, output_part=part, n_basis=8
                            )
                            torch.testing.assert_close(partial, fast[:, :8], rtol=0, atol=0)


class ArtifactOrderTests(unittest.TestCase):
    def test_legacy_load_and_current_save_reload(self):
        config = SimpleNamespace(hidden_size=8, n_hidden_layers=2, n_basis=4,
                                 n_basis_1loop=10, activation_f='tanh')
        torch.manual_seed(42)
        base = PinnModel(config)
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            model_path, bundle_path = temp / 'model.pt', temp / 'bundle.pt'
            for phase in (0, 1, 2):
                p = list(BASIS_PERMUTATIONS[phase])
                model = (base if phase == 0 else TransferPinnModel(
                    config, base, target_in_dim=phase+2, target_n_basis=len(p)
                ))
                x = torch.ones(2, phase+2)
                old_prediction = model(x).detach()
                loaded = copy.deepcopy(model)
                for raw in (False, True):
                    with self.subTest(phase=phase, raw=raw):
                        payload = model.state_dict() if raw else {
                            'phase': phase, 'model_state_dict': model.state_dict()
                        }
                        torch.save(payload, model_path)
                        main_core._load_model_checkpoint(loaded, str(model_path), 'cpu', phase=phase)
                        torch.testing.assert_close(loaded(x), old_prediction[:, p])
                paths = {'model_abs': str(model_path), 'history_abs': str(temp / 'history.npz')}
                with patch.object(main_core, '_phase_artifact_paths', return_value=paths):
                    main_core._save_phase_artifacts(
                        model=loaded, hist_tot=[1], hist_cde=[0.5], hist_bc=[0.5],
                        phase=phase, cy=0.4, eps_global=0, pred_scale=1,
                    )
                saved = torch.load(model_path, weights_only=True)
                self.assertEqual(saved['basis_order'], BASIS_ORDER)
                prediction = loaded(x).detach().clone()
                main_core._load_model_checkpoint(loaded, str(model_path), 'cpu', phase=phase)
                torch.testing.assert_close(loaded(x), prediction, rtol=0, atol=0)
                for blocks in (1, 2):
                    target = torch.arange(2 * len(p) * blocks).reshape(2, -1).float()
                    torch.save(dict(phase=phase, x_coll=x, x_b_tensor=x, bc_target=target), bundle_path)
                    _, _, new_target, meta = main_core._load_eval_bundle(str(bundle_path), 'cpu', phase=phase)
                    indices = p if blocks == 1 else p + [k + len(p) for k in p]
                    torch.testing.assert_close(new_target, target[:, indices], rtol=0, atol=0)
                    with patch.object(main_core, '_phase_eval_bundle_paths', return_value={'bundle_abs': str(bundle_path)}):
                        main_core._save_eval_bundle(
                            phase=phase, cy=0.4, eps_global=0, x_coll=x, x_b_tensor=x,
                            bc_target=new_target, pred_scale=1, extra_meta=meta,
                        )
                    _, _, reloaded, _ = main_core._load_eval_bundle(str(bundle_path), 'cpu', phase=phase)
                    torch.testing.assert_close(reloaded, new_target, rtol=0, atol=0)

    def test_reject_incompatible_artifacts(self):
        with self.assertRaisesRegex(ValueError, 'basis_order'):
            needs_basis_migration({'basis_order': 'unknown'}, phase=1)
        with self.assertRaisesRegex(ValueError, 'does not match'):
            needs_basis_migration({'phase': 2}, phase=1)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'partial.pt'
            torch.save(dict(x_coll=torch.ones(1, 4), x_b_tensor=torch.ones(1, 4),
                            bc_target=torch.ones(1, 8)), path)
            with self.assertRaisesRegex(ValueError, 'partial legacy basis'):
                main_core._load_eval_bundle(str(path), 'cpu', phase=2)


if __name__ == '__main__':
    unittest.main()
