"""Empty-subset-first ordering and compatibility with previously saved runs.

Permutations are zero-based: new[..., j] = old[..., permutation[j]].
The chain (phase 0) is unchanged. Normalizations and signs are unchanged.
"""

import copy

import torch


BASIS_ORDER = "empty_first_v1"
LEGACY_BASIS_ORDER = "empty_last_v1"

BASIS_PERMUTATIONS = {
    0: (0, 1, 2, 3),
    1: (0, 3, 1, 2, 6, 4, 5, 9, 7, 8),
    2: (0, 7, 1, 2, 3, 4, 5, 6, 14, 8, 9, 10, 11, 12, 13,
        21, 15, 16, 17, 18, 19, 20),
}
LETTER_PERMUTATIONS = {
    0: (0, 1, 2, 3, 4),
    1: (0, 1, 6, 7, 2, 3, 4, 5, 10, 8, 9),
    2: (0, 1, 14, 15, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
        22, 16, 17, 18, 19, 20, 21),
}


def needs_basis_migration(metadata, *, phase):
    """Untagged artifacts predate this convention and use the legacy order."""
    if phase not in BASIS_PERMUTATIONS:
        raise ValueError(f"Unsupported phase: {phase}")
    if "phase" in metadata and int(metadata["phase"]) != phase:
        raise ValueError(f"Artifact phase={metadata['phase']} does not match phase={phase}.")
    order = metadata.get("basis_order", LEGACY_BASIS_ORDER)
    if order not in (BASIS_ORDER, LEGACY_BASIS_ORDER):
        raise ValueError(f"Unsupported basis_order: {order!r}")
    return phase != 0 and order == LEGACY_BASIS_ORDER


def permute_legacy_channels(values, *, phase, dim=-1):
    """Reorder a full I vector, or both blocks of a [Re I, Im I] vector."""
    permutation = BASIS_PERMUTATIONS[phase]
    n_basis = len(permutation)
    n_channels = values.shape[dim]
    if n_channels == n_basis:
        indices = permutation
    elif n_channels == 2 * n_basis:
        indices = permutation + tuple(i + n_basis for i in permutation)
    else:
        raise ValueError(
            f"Cannot migrate phase-{phase} artifact with {n_channels} channels; "
            f"expected {n_basis} or {2 * n_basis}. A partial legacy basis may "
            "omit components needed in the new order."
        )
    return values.index_select(dim, torch.tensor(indices, device=values.device))


def migrate_legacy_model_state(state_dict, *, phase):
    """Permute the loop output head without changing the source checkpoint."""
    converted = copy.copy(state_dict)
    for key in ("output_layer.weight", "output_layer.bias"):
        if key not in converted:
            raise ValueError(f"Legacy phase-{phase} checkpoint is missing {key!r}.")
        converted[key] = permute_legacy_channels(converted[key], phase=phase, dim=0)
    return converted
