import torch
import torch.nn as nn


class ConnectionAMatricesBase2Loop(nn.Module):
    """
    2-loop sunset A-matrix builder on 4D inputs (x1, x2, y1, y2).
    A_i(x) = sum_k A_k * d_{x_i} log(w_k), k=1..23.
    """

    def __init__(self, n_basis=22, n_letters=23, cy_val=None):
        super().__init__()
        self.n_basis = int(n_basis)
        self.n_letters = int(n_letters)
        self.cy_val = cy_val

    @staticmethod
    def dlog_partials(x1, x2, y1, y2, cy_val):
        zeros = torch.zeros_like(x1)

        # Keep the letter ordering consistent with tl_two_site_sunset/sol_2loop.py::_ws.
        w1 = x1 + y1 + y2 + cy_val
        w2 = x2 + y1 + y2 + cy_val
        w9 = x1 + y1 + y2 - cy_val
        w10 = x2 + y1 + y2 - cy_val
        w11 = x1 + y1 - y2 + cy_val
        w12 = x2 + y1 - y2 + cy_val
        w13 = x1 - y1 + y2 + cy_val
        w14 = x2 - y1 + y2 + cy_val
        w3 = x1 + y1 - y2 - cy_val
        w4 = x2 + y1 - y2 - cy_val
        w5 = x1 - y1 + y2 - cy_val
        w6 = x2 - y1 + y2 - cy_val
        w7 = x1 - y1 - y2 + cy_val
        w8 = x2 - y1 - y2 + cy_val
        w15 = x1 - y1 - y2 - cy_val
        w16 = x2 - y1 - y2 - cy_val
        w20 = x1 + x2 + 2.0 * y1 + 2.0 * y2
        w21 = x1 + x2 + 2.0 * y1 + 2.0 * cy_val
        w22 = x1 + x2 + 2.0 * y2 + 2.0 * cy_val
        w17 = x1 + x2 + 2.0 * y1
        w18 = x1 + x2 + 2.0 * y2
        w19 = x1 + x2 + 2.0 * cy_val
        w23 = x1 + x2

        dlog_dx1 = torch.stack(
            [
                1 / w1,
                zeros,
                1 / w3,
                zeros,
                1 / w5,
                zeros,
                1 / w7,
                zeros,
                1 / w9,
                zeros,
                1 / w11,
                zeros,
                1 / w13,
                zeros,
                1 / w15,
                zeros,
                1 / w17,
                1 / w18,
                1 / w19,
                1 / w20,
                1 / w21,
                1 / w22,
                1 / w23,
            ],
            dim=1,
        )

        dlog_dx2 = torch.stack(
            [
                zeros,
                1 / w2,
                zeros,
                1 / w4,
                zeros,
                1 / w6,
                zeros,
                1 / w8,
                zeros,
                1 / w10,
                zeros,
                1 / w12,
                zeros,
                1 / w14,
                zeros,
                1 / w16,
                1 / w17,
                1 / w18,
                1 / w19,
                1 / w20,
                1 / w21,
                1 / w22,
                1 / w23,
            ],
            dim=1,
        )

        dlog_dy1 = torch.stack(
            [
                1 / w1,
                1 / w2,
                1 / w3,
                1 / w4,
                -1 / w5,
                -1 / w6,
                -1 / w7,
                -1 / w8,
                1 / w9,
                1 / w10,
                1 / w11,
                1 / w12,
                -1 / w13,
                -1 / w14,
                -1 / w15,
                -1 / w16,
                2 / w17,
                zeros,
                zeros,
                2 / w20,
                2 / w21,
                zeros,
                zeros,
            ],
            dim=1,
        )

        dlog_dy2 = torch.stack(
            [
                1 / w1,
                1 / w2,
                -1 / w3,
                -1 / w4,
                1 / w5,
                1 / w6,
                -1 / w7,
                -1 / w8,
                1 / w9,
                1 / w10,
                -1 / w11,
                -1 / w12,
                1 / w13,
                1 / w14,
                -1 / w15,
                -1 / w16,
                zeros,
                2 / w18,
                zeros,
                2 / w20,
                zeros,
                2 / w22,
                zeros,
            ],
            dim=1,
        )

        return dlog_dx1, dlog_dx2, dlog_dy1, dlog_dy2

    def build_matrices(self, x_in, ak_list, cy_val=None):
        cy_used = self.cy_val if cy_val is None else cy_val
        if cy_used is None:
            raise ValueError("cy_val must be set.")
        if x_in.shape[1] != 4:
            raise ValueError(
                f"x_in must have 4 columns (x1,x2,y1,y2), got {tuple(x_in.shape)}"
            )
        if len(ak_list) != self.n_letters:
            raise ValueError(f"ak_list must contain exactly {self.n_letters} matrices.")

        x1 = x_in[:, 0:1]
        x2 = x_in[:, 1:2]
        y1 = x_in[:, 2:3]
        y2 = x_in[:, 3:4]

        dlog_dx1, dlog_dx2, dlog_dy1, dlog_dy2 = self.dlog_partials(
            x1, x2, y1, y2, cy_used
        )

        device, dtype = x_in.device, x_in.dtype
        a_x1 = torch.zeros(
            x_in.shape[0], self.n_basis, self.n_basis, device=device, dtype=dtype
        )
        a_x2 = torch.zeros_like(a_x1)
        a_y1 = torch.zeros_like(a_x1)
        a_y2 = torch.zeros_like(a_x1)

        for k, ak in enumerate(ak_list):
            coef_x1 = dlog_dx1[:, k : k + 1]
            coef_x2 = dlog_dx2[:, k : k + 1]
            coef_y1 = dlog_dy1[:, k : k + 1]
            coef_y2 = dlog_dy2[:, k : k + 1]

            ak_batched = ak.unsqueeze(0)
            a_x1 += coef_x1 * ak_batched
            a_x2 += coef_x2 * ak_batched
            a_y1 += coef_y1 * ak_batched
            a_y2 += coef_y2 * ak_batched

        return a_x1, a_x2, a_y1, a_y2


class ConnectionAMatricesFixed2Loop(ConnectionAMatricesBase2Loop):
    def __init__(self, n_basis_local=22, n_letters=23, ak_list=None, cy_val=None):
        super().__init__(n_basis_local, n_letters, cy_val)

        if (not ak_list) or len(ak_list) != n_letters:
            raise ValueError(f"ak_list must contain exactly {n_letters} matrices.")

        for k, ak in enumerate(ak_list):
            self.register_buffer(f"a_{k}", ak.detach().clone())

    def forward(self, x_in, cy_val=None):
        ak_list = [getattr(self, f"a_{k}") for k in range(self.n_letters)]
        return self.build_matrices(x_in, ak_list, cy_val)


class ConnectionAMatricesFixedWithEps0_2Loop(ConnectionAMatricesBase2Loop):
    def __init__(
        self,
        n_basis_local=22,
        n_letters=23,
        ak_list=None,
        ak_list_eps0=None,
        cy_val=None,
    ):
        super().__init__(n_basis_local, n_letters, cy_val)

        if (not ak_list) or len(ak_list) != n_letters:
            raise ValueError(f"ak_list must contain exactly {n_letters} matrices.")
        if (not ak_list_eps0) or len(ak_list_eps0) != n_letters:
            raise ValueError(f"ak_list_eps0 must contain exactly {n_letters} matrices.")

        for k, ak in enumerate(ak_list):
            self.register_buffer(f"a_{k}", ak.detach().clone())
        for k, ak in enumerate(ak_list_eps0):
            self.register_buffer(f"a_eps0_{k}", ak.detach().clone())

    def forward(self, x_in, cy_val=None):
        ak_list = [getattr(self, f"a_{k}") for k in range(self.n_letters)]
        ak0_list = [getattr(self, f"a_eps0_{k}") for k in range(self.n_letters)]

        a_x1, a_x2, a_y1, a_y2 = self.build_matrices(x_in, ak_list, cy_val)
        a0_x1, a0_x2, a0_y1, a0_y2 = self.build_matrices(x_in, ak0_list, cy_val)
        return a_x1, a_x2, a_y1, a_y2, a0_x1, a0_x2, a0_y1, a0_y2
