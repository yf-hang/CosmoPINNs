import copy
import torch
import torch.nn as nn


def _normalize_output_part(value, default="both"):
    if value is None:
        value = default
    s = str(value).strip().lower()
    if s in {"both", "all", "reim", "complex"}:
        return "both"
    if s in {"re", "real"}:
        return "re"
    if s in {"im", "imag", "imaginary"}:
        return "im"
    raise ValueError(f"Unsupported output part: {value!r}. Expected one of Re/Im/Both.")


class PinnModel(nn.Module):
    def __init__(self, config, *, in_dim=2, output_part=None):
        super().__init__()

        hidden_size = config.hidden_size
        n_hidden = config.n_hidden_layers
        n_basis = config.n_basis
        activation_name = config.activation_f
        if output_part is None:
            output_part = getattr(config, "phase0_output_part", "both")
        output_part = _normalize_output_part(output_part)

        activation = self.get_activation(activation_name)

        layers = [nn.Linear(in_dim, hidden_size), activation]
        for _ in range(n_hidden - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation)
        out_dim = (2 * n_basis) if output_part == "both" else n_basis
        layers.append(nn.Linear(hidden_size, out_dim))

        self.net = nn.Sequential(*layers)
        self.output_part = output_part
        self.out_dim = int(out_dim)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def get_activation(name):
        name = name.lower()
        if name == "tanh":
            return nn.Tanh()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation function: {name}")

# -------------------------------------------------------------
#  Transfer learning PINN model for Phase 1/2 loop targets
# -------------------------------------------------------------
class TransferPinnModel(nn.Module):

    def __init__(
        self,
        config,
        phase0_model,
        freeze_core=True,
        output_part=None,
        target_in_dim=3,
        target_n_basis=None,
    ):
        super().__init__()

        if not hasattr(phase0_model, "net"):
            raise ValueError("phase0_model must provide a '.net' Sequential for transfer learning.")

        phase0_layers = list(phase0_model.net.children())
        if len(phase0_layers) < 3:
            raise ValueError("phase0_model.net must have at least [input, core, output] layers.")

        phase0_in = phase0_layers[0]
        if not isinstance(phase0_in, nn.Linear):
            raise ValueError("Expected the first phase0 layer to be nn.Linear.")
        if phase0_in.in_features not in (2, 3):
            raise ValueError(
                f"Expected phase0 input dim in {{2,3}}, got {phase0_in.in_features}."
            )

        hidden = phase0_in.out_features

        target_in_dim = int(target_in_dim)
        if target_in_dim <= 0:
            raise ValueError(f"target_in_dim must be positive, got {target_in_dim}.")

        # New input layer for transfer target coordinates.
        # No activation here because transferred core starts with original first activation.
        self.input_layer = nn.Linear(target_in_dim, hidden)

        # Copy all middle modules from phase 0 (drop first and last), then freeze.
        self.core = nn.Sequential(*[copy.deepcopy(m) for m in phase0_layers[1:-1]])
        if freeze_core:
            for p in self.core.parameters():
                p.requires_grad = False

        if output_part is None:
            output_part = getattr(config, "phase1_output_part", "both")
        output_part = _normalize_output_part(output_part)

        if target_n_basis is None:
            target_n_basis = int(config.n_basis_1loop)
        target_n_basis = int(target_n_basis)
        if target_n_basis <= 0:
            raise ValueError(f"target_n_basis must be positive, got {target_n_basis}.")

        out_dim = (2 * target_n_basis) if output_part == "both" else target_n_basis
        self.output_layer = nn.Linear(hidden, out_dim)
        self.output_part = output_part
        self.out_dim = int(out_dim)

        # Transfer initialization: reuse phase-0 input weights for x1/x2;
        # initialize extra dims with mean(x1,x2) weights.
        with torch.no_grad():
            self.input_layer.weight.zero_()
            copy_cols = min(2, phase0_in.in_features, target_in_dim)
            if copy_cols > 0:
                self.input_layer.weight[:, :copy_cols].copy_(phase0_in.weight[:, :copy_cols])
            if target_in_dim > copy_cols:
                if copy_cols > 0:
                    fill_w = phase0_in.weight[:, :copy_cols].mean(dim=1)
                else:
                    fill_w = torch.zeros_like(self.input_layer.bias)
                for j in range(copy_cols, target_in_dim):
                    self.input_layer.weight[:, j].copy_(fill_w)
            self.input_layer.bias.copy_(phase0_in.bias)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.core(x)
        x = self.output_layer(x)
        return x
