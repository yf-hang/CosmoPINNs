"""Regression checks for raw MSE, clean configuration, and revised domains."""

import importlib
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import main
import main_core
from lib.loss import (
    boundary_loss,
    cde_residual_loss_fixed_eps,
    cde_residual_loss_fixed_eps_1loop,
    cde_residual_loss_fixed_eps_2loop,
)


ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = (
    ("two_site_chain", "", 4, cde_residual_loss_fixed_eps),
    ("tl_two_site_bubble", "_1loop", 10, cde_residual_loss_fixed_eps_1loop),
    ("tl_two_site_sunset", "_2loop", 22, cde_residual_loss_fixed_eps_2loop),
)


class RawMSETests(unittest.TestCase):
    def test_cde_value_and_parameter_gradient_match_affine_oracle(self):
        # An affine model has the exact Jacobian W, independent of autograd.
        for d, (_, _, n_basis, loss_fn) in enumerate(SYSTEMS, start=2):
            x = torch.arange(3*d, dtype=torch.float64).reshape(3, d) / 7 + 1
            model = torch.nn.Linear(d, n_basis, dtype=torch.float64)
            with torch.no_grad():
                model.weight.copy_(
                    torch.arange(1, n_basis*d+1, dtype=torch.float64).reshape(n_basis, d) / 13
                )
                model.bias.copy_(torch.linspace(-1, 1, n_basis, dtype=torch.float64))
            regular = [(k+1) * torch.eye(n_basis, dtype=torch.float64) / 5 for k in range(d)]
            limiting = [a + 0.2 * torch.ones_like(a) for a in regular]

            def builder(points):
                return tuple(a.expand(len(points), -1, -1) for a in regular + limiting)

            for eps, part in itertools.product((0.0, -1.0, -2.0), ("re", "im")):
                with self.subTest(d=d, eps=eps, part=part):
                    actual, count = loss_fn(model, builder, x, n_basis, eps, output_part=part)
                    coefficient = limiting if eps == 0 else [eps*a for a in regular]
                    prediction = x @ model.weight.T + model.bias
                    exact_residual = torch.stack([
                        model.weight[:, k] - prediction @ a.T
                        for k, a in enumerate(coefficient)
                    ], dim=1)
                    expected = exact_residual.square().sum() / (len(x)*d*n_basis)
                    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
                    self.assertEqual(count.item(), len(x))
                    actual_grads = torch.autograd.grad(actual, tuple(model.parameters()))
                    expected_grads = torch.autograd.grad(expected, tuple(model.parameters()))
                    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)

    def test_boundary_default_is_unscaled_componentwise_mse(self):
        x = torch.tensor([[1., 10., 1000.], [-1., 20., 2000.]], requires_grad=True)
        target = torch.tensor([[0., 8., 999.], [0., 18., 1999.]])
        loss = boundary_loss(torch.nn.Identity(), x, target)
        self.assertEqual(loss.item(), 2.0)  # (1 + 4 + 1 + 1 + 4 + 1) / 6
        loss.backward()
        torch.testing.assert_close(x.grad, (x.detach() - target) / 3)


class CleanConfigTests(unittest.TestCase):
    def test_omitted_normalization_is_disabled_in_every_entry_point(self):
        raw = json.loads((ROOT / "config2.json").read_text())
        self.assertFalse(any("normaliz" in key or "solution_scale" in key for key in raw))
        self.assertNotIn("scheduler_type", raw)
        self.assertFalse(any(key.endswith("_load_path") for key in raw))
        self.assertIs(main.Config, main_core.Config)
        cfg = main_core.Config(ROOT / "config2.json")
        self.assertFalse(cfg.use_solution_scale)
        self.assertFalse(cfg.use_bc_channel_normalization)
        self.assertFalse(cfg.bc_loss_use_normalized)
        self.assertEqual(cfg.solution_scale_mode, "manual")
        for phase in range(3):
            self.assertEqual(getattr(cfg, f"solution_scale_p{phase}"), 1.0)

    def test_explicit_legacy_scaling_switches_remain_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for scale, normalize in itertools.product((False, True), repeat=2):
                with self.subTest(scale=scale, normalize=normalize):
                    path.write_text(json.dumps(dict(
                        device="cpu", use_solution_scale=scale,
                        use_bc_channel_normalization=normalize,
                        solution_scale_mode="manual", solution_scale_p0=7,
                    )))
                    cfg = main_core.Config(path)
                    self.assertEqual(cfg.bc_loss_use_normalized, normalize)
                    self.assertEqual(cfg.solution_scale_p0, 7 if scale else 1)

    def test_cli_default_and_explicit_config_are_forwarded(self):
        self.assertEqual(main._parse_args([]).config, ROOT / "config2.json")
        custom = main._parse_args(["--config", "config.json"]).config
        with patch.object(main_core, "main") as run:
            main.main(custom)
            run.assert_called_once_with(config_path=Path("config.json"))

    def test_new_domain_corners_have_finite_connections_and_real_targets(self):
        cfg = json.loads((ROOT / "config2.json").read_text())
        expected_bounds = (
            [(60, 70), (60, 70)],
            [(14, 16), (14, 16), (4, 6)],
            [(9, 11), (9, 11), (4, 6), (4, 6)],
        )
        for phase, (package, suffix, n_basis, _) in enumerate(SYSTEMS):
            coords = ("x1", "x2", "y1", "y2")[:phase+2]
            bounds = [(cfg[f"{c}_min{suffix}"], cfg[f"{c}_max{suffix}"]) for c in coords]
            self.assertEqual(bounds, expected_bounds[phase])
            x = torch.tensor(list(itertools.product(*bounds)), dtype=torch.float64)
            c = cfg[f"cy{suffix}"]
            self.assertEqual(c, 15 if phase == 0 else 5)
            bc = importlib.import_module(f"{package}.coll_bc{suffix}")
            conn = importlib.import_module(f"{package}.conn_mat{suffix}")
            matrices = importlib.import_module(f"{package}.mat_data{suffix}")
            builder_name = "ConnectionAMatricesFixedWithEps0" + (f"_{phase}Loop" if phase else "")
            builder = getattr(conn, builder_name)(
                ak_list=[getattr(matrices, f"a{k}") for k in range(1, n_basis+2)],
                ak_list_eps0=[getattr(matrices, f"a{k}_eps0") for k in range(1, n_basis+2)],
                cy_val=c,
            ).double()
            for matrix in builder(x):
                self.assertTrue(torch.isfinite(matrix).all())
            # Letters are affine, so corner extrema bound the entire box.
            if phase:
                for letter in bc._build_ws_tensor(x, cy_val=c):
                    self.assertTrue(bool((letter > 0).all() or (letter < 0).all()))
            for eps in (0, -1, cfg["eps_global"]):
                with self.subTest(phase=phase, eps=eps):
                    target_fn = getattr(bc, f"compute_function_target_from_xcoll{suffix}")
                    real = target_fn(x, cy_val=c, eps_val=eps, output_part="re")
                    imag = target_fn(x, cy_val=c, eps_val=eps, output_part="im")
                    self.assertEqual(tuple(real.shape), (len(x), n_basis))
                    self.assertTrue(torch.isfinite(real).all())
                    torch.testing.assert_close(imag, torch.zeros_like(imag), atol=1e-10, rtol=0)
                    scalar_fn = getattr(bc, f"compute_boundary_values_rescaled{suffix}")
                    reference = scalar_fn(np.column_stack((x[:2].numpy(), [eps, eps])), c)
                    np.testing.assert_allclose(real[:2].numpy(), reference.real, rtol=3e-6, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
