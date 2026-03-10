"""
Minimal base class for control-affine dynamical systems.

This is a local stub replacing the neural_clbf.systems.ControlAffineSystem
to make the ISS-ICNN bundle self-contained.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


class ControlAffineSystem(nn.Module):
    """
    Base class for control-affine dynamical systems of the form:
        dx/dt = f(x) + g(x) * u

    Subclasses must implement:
        - control_affine_dynamics(x, params) -> (f, g)
        - n_dims property
        - n_controls property
    """

    def __init__(
        self,
        nominal_params: Dict = None,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        use_linearized_controller: bool = False,
        scenarios: Optional[Dict] = None,
    ):
        super().__init__()
        self.nominal_params = nominal_params or {}
        self.dt = dt
        self.controller_dt = controller_dt if controller_dt is not None else dt
        self.use_linearized_controller = use_linearized_controller
        self.scenarios = scenarios or {}

        # Newton solver settings (used by DAE subclasses)
        self.newton_iterations = 5
        self.newton_reuse_jacobian = True
        self.newton_adaptive = True
        self.newton_warn_threshold = 1e-3
        self.newton_warn_on_nonconvergence = True
        self.newton_damping = 1.0
        self.newton_step_resid_factor = 10.0

        # These are set by subclasses
        self._goal_point = None
        self._u_eq = None
        self._control_limits = None

    @property
    def n_dims(self) -> int:
        raise NotImplementedError("Subclass must define n_dims")

    @property
    def n_controls(self) -> int:
        raise NotImplementedError("Subclass must define n_controls")

    @property
    def goal_point(self) -> torch.Tensor:
        return self._goal_point

    @goal_point.setter
    def goal_point(self, value):
        self._goal_point = value

    @property
    def u_eq(self) -> torch.Tensor:
        return self._u_eq

    @u_eq.setter
    def u_eq(self, value):
        self._u_eq = value

    @property
    def control_limits(self):
        return self._control_limits

    @control_limits.setter
    def control_limits(self, value):
        self._control_limits = value

    def control_affine_dynamics(self, x: torch.Tensor, params=None):
        raise NotImplementedError("Subclass must implement control_affine_dynamics")


def repair_unpickled_module(model):
    """Re-initialise missing ``nn.Module`` internals on a pickled model.

    Models saved with ``torch.save(model, path)`` under one PyTorch version
    may lack private dicts (``_buffers``, ``_modules``, ``_parameters``) when
    loaded under a newer version.  This helper patches those attributes so
    that the model can participate in ``load_state_dict`` calls (e.g. when
    stored as a submodule of another ``nn.Module``).

    Args:
        model: An ``nn.Module`` instance loaded via ``torch.load``.

    Returns:
        The same *model*, mutated in-place with repaired internals.
    """
    from collections import OrderedDict
    d = object.__getattribute__(model, "__dict__")
    for attr in (
        "_parameters",
        "_buffers",
        "_modules",
        "_backward_hooks",
        "_forward_hooks",
    ):
        if attr not in d:
            d[attr] = OrderedDict()
    if "_non_persistent_buffers_set" not in d:
        d["_non_persistent_buffers_set"] = set()
    # Ensure training flag
    if "training" not in d:
        d["training"] = False
    return model
