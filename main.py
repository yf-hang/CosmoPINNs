"""CosmoPINNs training entry point.

The training implementation lives in ``main_core.py``. This thin entry layer
maps the public configuration switches

    use_solution_scale
    use_bc_channel_normalization

to the legacy internal fields used by the training core. The two public
switches are intentionally independent: disabling solution scaling does not
change the BC-loss normalization setting, and vice versa.
"""

from __future__ import annotations

import builtins

import main_core as _core


_OriginalConfig = _core.Config
_ORIGINAL_PRINT = builtins.print
_ACTIVE_SWITCHES = {
    "use_solution_scale": False,
    "use_bc_channel_normalization": False,
}


class Config(_OriginalConfig):
    """Load the public config and adapt the two independent normalization flags."""

    def __init__(self, json_path):
        super().__init__(json_path)

        use_solution_scale = _core._to_bool(
            getattr(self, "use_solution_scale", False),
            default=False,
        )
        use_bc_channel_normalization = _core._to_bool(
            getattr(self, "use_bc_channel_normalization", False),
            default=False,
        )

        self.use_solution_scale = use_solution_scale
        self.use_bc_channel_normalization = use_bc_channel_normalization
        _ACTIVE_SWITCHES["use_solution_scale"] = use_solution_scale
        _ACTIVE_SWITCHES["use_bc_channel_normalization"] = (
            use_bc_channel_normalization
        )

        # main_core.py historically used ``normalized_bc`` both as the switch
        # for global solution scaling and as a gate that could force BC channel
        # normalization off. Keep that legacy gate enabled so it cannot couple
        # the two new public switches.
        self.normalized_bc = True
        self.bc_loss_use_normalized = use_bc_channel_normalization

        # Global solution scaling is controlled only by use_solution_scale.
        # When disabled, force a unit scale without touching the BC-loss flag.
        if not use_solution_scale:
            self.solution_scale_mode = "manual"
            self.solution_scale_p0 = 1.0
            self.solution_scale_p1 = 1.0
            self.solution_scale_p2 = 1.0


# Make the training core instantiate the adapted configuration above.
_core.Config = Config


def _entry_print(*args, **kwargs):
    """Replace the legacy normalization status line with the public switches."""
    if len(args) == 1 and isinstance(args[0], str):
        text = args[0]
        if text.startswith("[Mode] normalized BC mode:"):
            _ORIGINAL_PRINT(
                "[Mode] solution scaling enabled: "
                f"{_ACTIVE_SWITCHES['use_solution_scale']}",
                **kwargs,
            )
            _ORIGINAL_PRINT(
                "[Mode] BC channel normalization: "
                f"{_ACTIVE_SWITCHES['use_bc_channel_normalization']}",
                **kwargs,
            )
            return
        if text.startswith("[Mode] BC loss: normalized,"):
            text = text.replace(
                "[Mode] BC loss: normalized,",
                "[Mode] BC loss: channel-normalized,",
                1,
            )
            args = (text,)
    _ORIGINAL_PRINT(*args, **kwargs)


_core.print = _entry_print


def main():
    return _core.main()


if __name__ == "__main__":
    main()
