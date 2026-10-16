# Empty-subset-first convention: S0={}, S1={y1}, S2={c}.
import torch
import torch.nn as nn

class ConnectionAMatricesBase1Loop(nn.Module):
    """
    1-loop A-matrix
    A_i(x) = sum_k A_k * ∂_{x_i} log w_k
    w1,w2: x1,x2 + y1 + cy_val
    w3,w4: x1,x2 - y1 - cy_val (S0 = empty)
    w5,w6: x1,x2 + y1 - cy_val (S1 = {y1})
    w7,w8: x1,x2 - y1 + cy_val (S2 = {cy_val})
    w9 = x1 + x2; w10 = w9 + 2*y1; w11 = w9 + 2*cy_val.
    """
    def __init__(self, n_basis=10, n_letters=11, cy_val=None):
        super().__init__()
        self.n_basis = n_basis
        self.n_letters = n_letters
        self.cy_val = cy_val

    @staticmethod
    def dlog_partials(x1, x2, y1, cy_val):
        # x1,x2,y1: (N,1)
        zeros = torch.zeros_like(x1)

        w1 = x1 + y1 + cy_val
        w2 = x2 + y1 + cy_val
        w3 = x1 - y1 - cy_val
        w4 = x2 - y1 - cy_val
        w5 = x1 + y1 - cy_val
        w6 = x2 + y1 - cy_val
        w7 = x1 - y1 + cy_val
        w8 = x2 - y1 + cy_val
        w9 = x1 + x2
        w10  = x1 + x2 + 2.0 * y1
        w11 = x1 + x2 + 2.0 * cy_val

        dlog_dx1 = torch.stack(
            [
                1 / w1, zeros, 1 / w3, zeros,
                1 / w5, zeros, 1 / w7, zeros,
                1 / w9, 1 / w10, 1 / w11,
            ],
            dim=1,
        )

        dlog_dx2 = torch.stack(
            [
                zeros, 1 / w2, zeros, 1 / w4,
                zeros, 1 / w6, zeros, 1 / w8,
                1 / w9, 1 / w10, 1 / w11,
            ],
            dim=1,
        )

        dlog_dy1 = torch.stack(
            [
                1 / w1, 1 / w2, -1 / w3, -1 / w4,
                1 / w5, 1 / w6, -1 / w7, -1 / w8,
                zeros, 2 / w10, zeros,
            ],
            dim=1,
        )

        # each return tensor has shape (N, n_letters, 1)
        return dlog_dx1, dlog_dx2, dlog_dy1

    def build_matrices(self, x_in, ak_list, cy_val=None):
        cy_used = self.cy_val if cy_val is None else cy_val
        if cy_used is None:
            raise ValueError("cy_val must be set.")

        x1, x2, y1= (x_in[:, 0:1],
                     x_in[:, 1:2],
                     x_in[:, 2:3])

        dlog_dx1, dlog_dx2, dlog_dy1 = self.dlog_partials(x1, x2, y1, cy_used)

        device, dtype = x_in.device, x_in.dtype
        a_x1 = torch.zeros(x_in.shape[0], self.n_basis, self.n_basis, device=device, dtype=dtype)
        a_x2 = torch.zeros_like(a_x1)
        a_y1 = torch.zeros_like(a_x1)

        for k, ak in enumerate(ak_list):
            coef_a1 = dlog_dx1[:, k:k + 1]  # (N,1,1)
            coef_a2 = dlog_dx2[:, k:k + 1]  # (N,1,1)
            coef_ay = dlog_dy1[:, k:k + 1]  # (N,1,1)

            a_x1 += coef_a1 * ak.unsqueeze(0)  # (1,n,n) -> (N,n,n)
            a_x2 += coef_a2 * ak.unsqueeze(0)
            a_y1 += coef_ay * ak.unsqueeze(0)

        return a_x1, a_x2, a_y1


class ConnectionAMatricesFixed1Loop(ConnectionAMatricesBase1Loop):
    def __init__(self, n_basis_local=10, n_letters=11, ak_list=None, cy_val=None):
        super().__init__(n_basis_local, n_letters, cy_val)

        if not ak_list or len(ak_list) != n_letters:
            raise ValueError(f"ak_list must contain exactly {n_letters} matrices.")

        for k, ak in enumerate(ak_list):
            self.register_buffer(f"a_{k}", ak.detach().clone())

    def forward(self, x_in, cy_val=None):
        ak_list = [getattr(self, f"a_{k}") for k in range(self.n_letters)]
        return self.build_matrices(x_in, ak_list, cy_val)


class ConnectionAMatricesFixedWithEps0_1Loop(ConnectionAMatricesBase1Loop):

    def __init__(
        self,
        n_basis_local=10,
        n_letters=11,
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

        a_x1, a_x2, a_y1 = self.build_matrices(x_in, ak_list, cy_val)
        a0_x1, a0_x2, a0_y1 = self.build_matrices(x_in, ak0_list, cy_val)

        return a_x1, a_x2, a_y1, a0_x1, a0_x2, a0_y1
