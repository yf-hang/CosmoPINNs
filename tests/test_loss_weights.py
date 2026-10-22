"""Check the agreed two-term parameter-gradient L2 balancing rule."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import main_core
from lib.train import _gradient_balanced_weights, train_model_fixed_eps


ROOT = Path(__file__).resolve().parents[1]


class GradientWeightTests(unittest.TestCase):
    def test_unweighted_global_l2_ema_and_stop_gradient(self):
        p = torch.nn.Parameter(torch.tensor([1., 2.], dtype=torch.float64))
        unused = torch.nn.Parameter(torch.tensor(8., dtype=torch.float64))
        frozen = torch.nn.Parameter(torch.tensor(3., dtype=torch.float64), requires_grad=False)
        cde = 3*p[0] + 4*p[1] + 1000*frozen  # trainable gradient norm = 5
        bc = 12*p[1]                         # trainable gradient norm = 12
        p.grad = torch.full_like(p, 99)
        w1, w2, gc, gb = _gradient_balanced_weights(
            cde, bc, (p, unused, frozen), 2., 3., 0.9
        )
        self.assertEqual((gc, gb), (5., 12.))
        self.assertAlmostEqual(w1, 0.9*2 + 0.1*17/5)
        self.assertAlmostEqual(w2, 0.9*3 + 0.1*17/12)
        self.assertIsInstance(w1, float)
        self.assertIsInstance(w2, float)
        torch.testing.assert_close(p.grad, torch.full_like(p, 99))
        p.grad = None
        (w1*cde + w2*bc).backward()
        torch.testing.assert_close(p.grad, torch.tensor([3*w1, 4*w1+12*w2], dtype=p.dtype))
        self.assertIsNone(unused.grad)
        self.assertIsNone(frozen.grad)

    def test_zero_gradient_keeps_corresponding_old_weight(self):
        p = torch.nn.Parameter(torch.tensor(1.))
        for cde, bc, expected in (
            (p*0, p*2, (7., 3.7)),
            (p*2, p*0, (6.4, 4.)),
            (p*0, p*0, (7., 4.)),
            (torch.tensor(2.), torch.tensor(3.), (7., 4.)),
        ):
            with self.subTest(expected=expected):
                w1, w2, _, _ = _gradient_balanced_weights(cde, bc, (p,), 7., 4., 0.9)
                self.assertAlmostEqual(w1, expected[0])
                self.assertAlmostEqual(w2, expected[1])

    def _train(self, *, mode="grad_norm", interval=2, epochs=4):
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(1.)
        x = torch.ones(1, 1)

        def cde(net, builder, batch, n_basis, eps_val):
            return net.weight.square().sum(), torch.tensor(float(len(batch)))

        def bc(net, batch, target):
            return 4*net.weight.square().sum()

        with contextlib.redirect_stdout(io.StringIO()):
            result = train_model_fixed_eps(
                model, None, x, x, x, cde_loss_fixed_fn=cde, bc_loss_fn=bc,
                n_basis=1, eps_val=0, lr_init=0., warmup_len=0,
                total_epochs=epochs, lam1=1., lam2=1., print_every=epochs,
                loss_weighting=mode, loss_weight_update_every=interval,
                loss_weight_ema=0.9,
            )
        return result

    def test_update_schedule_and_weighted_training_history(self):
        model, total, cde, bc, info = self._train()
        # g_B/g_C = 4. EMA targets are (5, 1.25), first used on step 2.
        np.testing.assert_allclose(info["lambda1_history"], [1, 1.4, 1.4, 1.76])
        np.testing.assert_allclose(info["lambda2_history"], [1, 1.025, 1.025, 1.0475])
        np.testing.assert_allclose(total, [5, 5.5, 5.5, 5.95], rtol=1e-6)
        self.assertEqual([item["step"] for item in info["loss_weight_updates"]], [2, 4])
        self.assertEqual(info["lambda1_initial"], 1.)
        self.assertEqual(info["lambda2_initial"], 1.)
        self.assertAlmostEqual(info["lambda1_final"], 1.76)
        self.assertAlmostEqual(info["lambda2_final"], 1.0475)
        # Checkpoints and history files must retain the weights actually used.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = dict(model_abs=str(root/'model.pt'), history_abs=str(root/'history.npz'))
            with patch.object(main_core, '_phase_artifact_paths', return_value=paths):
                main_core._save_phase_artifacts(
                    model=model, hist_tot=total, hist_cde=cde, hist_bc=bc,
                    phase=0, cy=15, eps_global=0, pred_scale=1, extra_meta=info,
                )
            saved = torch.load(paths['model_abs'], weights_only=True)
            self.assertEqual(saved['loss_weight_updates'], info['loss_weight_updates'])
            with np.load(paths['history_abs']) as history:
                np.testing.assert_allclose(
                    history['total'],
                    history['lambda1_history']*history['cde'] + history['lambda2_history']*history['bc'],
                    rtol=1e-6,
                )

    def test_production_schedule_first_updates_on_step_1000(self):
        _, _, _, _, info = self._train(interval=1000, epochs=1001)
        self.assertEqual([item['step'] for item in info['loss_weight_updates']], [1000])
        self.assertEqual(info['lambda1_history'][:999], [1.]*999)
        np.testing.assert_allclose(info['lambda1_history'][999:], [1.4, 1.4])

    def test_fixed_compatibility_and_new_default_config(self):
        _, total, _, _, info = self._train(mode="fixed")
        self.assertEqual(total, [5.]*4)
        self.assertEqual(info['loss_weight_updates'], [])
        cfg = json.loads((ROOT/'config2.json').read_text())
        self.assertEqual(cfg['loss_weighting'], 'grad_norm')
        self.assertEqual((cfg['lambda1'], cfg['lambda2']), (1, 1))
        self.assertEqual(cfg['loss_weight_update_every'], 1000)
        self.assertEqual(cfg['loss_weight_ema'], 0.9)


if __name__ == '__main__':
    unittest.main()
