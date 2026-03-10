#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IEEE39ControlAffineDAE_GFM — 6-state Grid-Forming Inverter Model with Adaptive Virtual Inertia

This implements the GFM/VSG (Virtual Synchronous Generator) model from the paper,
with time-varying inertia and a voltage-source-behind-impedance (VSBI) interface.

6 states per inverter:
1. theta_v    - Virtual rotor angle (voltage source angle, electrical radians)
2. omega_v    - Virtual rotor speed (per-unit, 1.0 = synchronous)
3. M_inv      - Virtual inertia (AVI state, seconds — analogous to 2H)
4. V_meas     - Filtered voltage measurement (per-unit)
5. P_filt     - Filtered active power measurement (per-unit)
6. Q_filt     - Filtered reactive power measurement (per-unit)

Key difference from GFL6:
  GFL: Inverter current I_inv = (P_ord + j*Q_ord)/V_meas * exp(j*phi)
       -> Does NOT depend on Newton unknowns (V, theta)
  GFM: Inverter current I_inv = (E*exp(j*theta_v) - V*exp(j*theta))/Z_v
       -> DOES depend on Newton unknowns, adds Y_v = 1/Z_v to Jacobian

GFM Dynamics:
  Swing equation:  M_i(t) * d_omega_v/dt = P_set - P_e - D_v*(omega_v - 1) - u_i
  Angle:           d_theta_v/dt = (omega_v - 1) * omega_s
  AVI:             dM_i/dt = (M_cmd - M_i) / tau_m
                   M_cmd = M_0 + k_m * |omega_v - 1|
  Voltage filter:  dV_meas/dt = (V_bus - V_meas) / T_v
  Power filters:   dP_filt/dt = (P_e - P_filt) / T_p
                   dQ_filt/dt = (Q_e - Q_filt) / T_q

Voltage-source model:
  V_inv = E * exp(j * theta_v)    (internal EMF behind virtual impedance)
  I_inv = (V_inv - V_bus) / Z_v   (current into bus, depends on Newton unknowns)
  E = E_0 + K_QV * (V_ref - V_meas)  (optional voltage droop)

State layout (129 states with 10 inverters, two-axis machine model):
[delta_rel(9) | omega(10) | Eqp(10) | Edp(10) | Efd(10) | Pm(10) | Pvalve(10) |
 theta_v(10) | omega_v(10) | M_inv(10) | V_meas(10) | P_filt(10) | Q_filt(10)]

Control inputs: u = [u_i(n_inv)]  (active power modulation from tri-branch controller)
"""

from typing import Optional, Tuple, List, Dict
import copy
import math
import numpy as np
import torch
from torch import Tensor
from torch.func import vmap, jacrev

try:
    from .control_affine_system import ControlAffineSystem
except ImportError:
    from control_affine_system import ControlAffineSystem


def _to_torch_complex(a: np.ndarray, dtype=torch.complex64) -> Tensor:
    """Convert numpy array to torch complex tensor."""
    if np.iscomplexobj(a):
        real = torch.tensor(a.real, dtype=torch.float32)
        imag = torch.tensor(a.imag, dtype=torch.float32)
        return torch.complex(real, imag).to(dtype)
    else:
        return torch.tensor(a, dtype=torch.float32).to(dtype)


class KCLNewtonBatchedGFM(torch.autograd.Function):
    """
    Batched implicit KCL solver for 6-state GFM model.

    Forward:
        Solves F(z; inputs) = 0 for z = [theta_rel(1..n-1), V(0..n-1)]
        using Newton or "modified Newton" (Jacobian reuse / chord method).

    Backward:
        Uses an IFT-style VJP:
            J(z*)^T v = dL/dz
            dL/dinputs = - (dF/dinputs)^T v

    Key difference from GFL6:
        GFM inverter current I_inv = (E*exp(j*theta_v) - V*exp(j*theta))/Z_v
        depends on Newton unknowns (V, theta), so:
        - Residual includes V/theta-dependent GFM current
        - Jacobian gains Y_v = 1/Z_v diagonal terms at inverter buses
        - Backward VJP takes theta_v and E_inv as inputs (not P_ord, Q_ord, phi)
    """

    # ----------------------------
    # Helper: residual (batched)
    # ----------------------------
    @staticmethod
    def _kcl_residual_masked_batch(
        *,
        z_batch: Tensor,
        delta_batch: Tensor,
        Eqp_batch: Tensor,
        Edp_batch: Tensor,
        theta_v_batch: Tensor,
        E_inv_batch: Tensor,
        sP_batch: Tensor,
        sQ_batch: Tensor,
        Y: Tensor,
        Xd_prime: Tensor,
        Xq_prime: Tensor,
        sg_mask_float: Tensor,
        sg_Imax: Optional[Tensor],
        PL_base: Tensor,
        QL_base: Tensor,
        inv_bus_map: Tensor,
        gen_bus_idx: Tensor,
        Z_v: Tensor,
        kZ_P: float, kI_P: float, kP_P: float,
        kZ_Q: float, kI_Q: float, kP_Q: float,
        mask: Optional[Tensor],
        eps_v: float = 1e-4,
        solve_full_theta: bool = False,
    ) -> Tensor:
        """Compute KCL residual F(z) for all batch elements.

        GFM inverter current: I_inv = (E*exp(j*theta_v) - V*exp(j*theta)) / Z_v
        This depends on Newton unknowns (V, theta) unlike GFL.
        """
        device = z_batch.device
        dtype = z_batch.dtype
        B = z_batch.shape[0]
        n_bus = Y.shape[0]
        n_gen = Xd_prime.numel()
        cdtype = Y.dtype

        if solve_full_theta:
            theta = z_batch[:, :n_bus]
            V = z_batch[:, n_bus:]
        else:
            theta_rel = z_batch[:, :n_bus - 1]
            V = z_batch[:, n_bus - 1:]
            theta = torch.cat(
                [torch.zeros(B, 1, device=device, dtype=dtype), theta_rel],
                dim=1
            )

        two_pi = 2.0 * math.pi
        theta = torch.remainder(theta + math.pi, two_pi) - math.pi
        delta_wrapped = torch.remainder(delta_batch + math.pi, two_pi) - math.pi

        ejth = torch.exp(1j * theta.to(cdtype))
        Vc = V.to(cdtype) * ejth

        # Two-axis generator current injection (same as GFL6)
        theta_gen = theta[:, gen_bus_idx]
        V_gen = V[:, gen_bus_idx]
        tau = delta_wrapped - theta_gen
        Vq = V_gen * torch.cos(tau)
        Vd = V_gen * torch.sin(tau)
        Id = (Eqp_batch - Vq) / Xd_prime.unsqueeze(0)
        Iq = (Vd - Edp_batch) / Xq_prime.unsqueeze(0)
        if sg_Imax is not None:
            i_lim = sg_Imax.to(device=device, dtype=Id.dtype).unsqueeze(0)
            i_mag = torch.sqrt(torch.clamp(Id.square() + Iq.square(), min=1e-12))
            i_scale = torch.minimum(torch.ones_like(i_mag), i_lim / i_mag.clamp(min=1e-9))
            Id = Id * i_scale
            Iq = Iq * i_scale
        ejd = torch.exp(1j * delta_wrapped.to(cdtype))
        I_sg_per = ((Iq - 1j * Id).to(cdtype) * ejd) * sg_mask_float.to(cdtype).unsqueeze(0)

        I_sg = torch.zeros(B, n_bus, device=device, dtype=cdtype)
        I_sg = I_sg.index_add(1, gen_bus_idx, I_sg_per)

        # Network current
        I_net = Vc @ Y.T

        # ZIP load currents (same as GFL6)
        V_safe = torch.clamp(V, min=eps_v)
        Pz = sP_batch * kZ_P * PL_base.unsqueeze(0) * (V ** 2)
        Pi = sP_batch * kI_P * PL_base.unsqueeze(0) * V
        Pp = sP_batch * kP_P * PL_base.unsqueeze(0)

        Qz = sQ_batch * kZ_Q * QL_base.unsqueeze(0) * (V ** 2)
        Qi = sQ_batch * kI_Q * QL_base.unsqueeze(0) * V
        Qp = sQ_batch * kP_Q * QL_base.unsqueeze(0)

        P = Pz + Pi + Pp
        Q = Qz + Qi + Qp
        S_over_V = (P - 1j * Q).to(cdtype) / V_safe.to(cdtype)
        I_load = S_over_V * ejth

        # GFM inverter current injection: I_inv = (E*exp(j*theta_v) - V*exp(j*theta)) / Z_v
        # This DEPENDS on Newton unknowns (V, theta) — key difference from GFL
        theta_inv = theta[:, inv_bus_map]        # (B, n_inv)
        V_inv_bus = V[:, inv_bus_map]            # (B, n_inv)
        ejth_inv = torch.exp(1j * theta_inv.to(cdtype))
        V_bus_c = V_inv_bus.to(cdtype) * ejth_inv  # V_bus complex at inverter buses

        theta_v_wrapped = torch.remainder(theta_v_batch + math.pi, two_pi) - math.pi
        V_inv_internal = E_inv_batch.to(cdtype) * torch.exp(1j * theta_v_wrapped.to(cdtype))

        Z_v_c = Z_v.to(dtype=cdtype, device=device)
        if Z_v_c.dim() == 1:
            Z_v_c = Z_v_c.unsqueeze(0)  # (1, n_inv) for broadcasting

        I_inv_per = (V_inv_internal - V_bus_c) / Z_v_c  # (B, n_inv)

        I_inv = torch.zeros(B, n_bus, device=device, dtype=cdtype)
        I_inv = I_inv.index_add(1, inv_bus_map, I_inv_per)

        # KCL residual (complex)
        k = I_net + I_load - I_sg - I_inv

        F_full = torch.cat([k.real.to(dtype), k.imag.to(dtype)], dim=1)
        if mask is None:
            return F_full
        return F_full[:, mask]

    # ----------------------------
    # Helper: analytic Jacobian dF/dz (batched)
    # ----------------------------
    @staticmethod
    def _kcl_jacobian_z_masked_batch(
        *,
        z_batch: Tensor,
        delta_batch: Tensor,
        Eqp_batch: Tensor,
        Edp_batch: Tensor,
        theta_v_batch: Tensor,
        E_inv_batch: Tensor,
        sP_batch: Tensor,
        sQ_batch: Tensor,
        Y: Tensor,
        Xd_prime: Tensor,
        Xq_prime: Tensor,
        sg_mask_float: Tensor,
        sg_Imax: Optional[Tensor],
        PL_base: Tensor,
        QL_base: Tensor,
        inv_bus_map: Tensor,
        gen_bus_idx: Tensor,
        Z_v: Tensor,
        kZ_P: float, kI_P: float, kP_P: float,
        kZ_Q: float, kI_Q: float, kP_Q: float,
        mask: Optional[Tensor],
        eps_v: float = 1e-4,
        solve_full_theta: bool = False,
    ) -> Tensor:
        """Analytic Jacobian of KCL residual F(z) w.r.t z.

        GFM adds admittance-like diagonal terms at inverter buses:
            d(-I_inv)/d_theta_k = j * Vc_k / Z_v  (at inverter bus k)
            d(-I_inv)/dV_k = exp(j*theta_k) / Z_v  (at inverter bus k)
        These are equivalent to adding Y_v = 1/Z_v to the network admittance diagonal.
        """
        device = z_batch.device
        dtype = z_batch.dtype
        B = z_batch.shape[0]
        n_bus = Y.shape[0]
        n_gen = Xd_prime.numel()
        cdtype = Y.dtype

        if solve_full_theta:
            theta = z_batch[:, :n_bus]
            V = z_batch[:, n_bus:]
        else:
            theta_rel = z_batch[:, :n_bus - 1]
            V = z_batch[:, n_bus - 1:]
            theta = torch.cat(
                [torch.zeros(B, 1, device=device, dtype=dtype), theta_rel],
                dim=1
            )

        two_pi = 2.0 * math.pi
        theta = torch.remainder(theta + math.pi, two_pi) - math.pi
        delta_wrapped = torch.remainder(delta_batch + math.pi, two_pi) - math.pi

        ejth = torch.exp(1j * theta.to(cdtype))
        Vc = V.to(cdtype) * ejth

        # --- 2-axis SG derivatives (same as GFL6) ---
        theta_gen = theta[:, gen_bus_idx]
        V_gen = V[:, gen_bus_idx]
        tau = delta_wrapped - theta_gen
        cos_tau = torch.cos(tau)
        sin_tau = torch.sin(tau)
        Vq = V_gen * cos_tau
        Vd = V_gen * sin_tau
        ejd = torch.exp(1j * delta_wrapped.to(cdtype))
        sg_c = sg_mask_float.to(cdtype).unsqueeze(0)
        Xd_c = Xd_prime.to(cdtype).unsqueeze(0)
        Xq_c = Xq_prime.to(cdtype).unsqueeze(0)

        if sg_Imax is not None:
            Id_raw = (Eqp_batch - Vq) / Xd_prime.unsqueeze(0)
            Iq_raw = (Vd - Edp_batch) / Xq_prime.unsqueeze(0)
            i_lim = sg_Imax.to(device=device, dtype=Id_raw.dtype).unsqueeze(0)
            i_mag = torch.sqrt(torch.clamp(Id_raw.square() + Iq_raw.square(), min=1e-12))
            i_scale = torch.minimum(torch.ones_like(i_mag), i_lim / i_mag.clamp(min=1e-9)).to(cdtype)
        else:
            i_scale = 1.0

        neg_dIsg_dtheta_per = i_scale * ((Vq.to(cdtype) / Xq_c - 1j * Vd.to(cdtype) / Xd_c) * ejd) * sg_c
        neg_dIsg_dV_per = i_scale * (-(sin_tau.to(cdtype) / Xq_c + 1j * cos_tau.to(cdtype) / Xd_c) * ejd) * sg_c

        neg_dIsg_dtheta = torch.zeros(B, n_bus, device=device, dtype=cdtype)
        neg_dIsg_dtheta = neg_dIsg_dtheta.index_add(1, gen_bus_idx, neg_dIsg_dtheta_per)
        neg_dIsg_dV = torch.zeros(B, n_bus, device=device, dtype=cdtype)
        neg_dIsg_dV = neg_dIsg_dV.index_add(1, gen_bus_idx, neg_dIsg_dV_per)

        # --- Load current derivatives (same as GFL6) ---
        V_safe = torch.clamp(V, min=eps_v)
        dVsafe_dV = (V > eps_v).to(dtype)

        Pz = sP_batch * kZ_P * PL_base.unsqueeze(0) * (V ** 2)
        Pi = sP_batch * kI_P * PL_base.unsqueeze(0) * V
        Pp = sP_batch * kP_P * PL_base.unsqueeze(0)
        Qz = sQ_batch * kZ_Q * QL_base.unsqueeze(0) * (V ** 2)
        Qi = sQ_batch * kI_Q * QL_base.unsqueeze(0) * V
        Qp = sQ_batch * kP_Q * QL_base.unsqueeze(0)

        P = Pz + Pi + Pp
        Q = Qz + Qi + Qp
        S = (P - 1j * Q).to(cdtype)
        u = S / V_safe.to(cdtype)
        I_load = u * ejth

        dP_dV = sP_batch * (2.0 * kZ_P * PL_base.unsqueeze(0) * V + kI_P * PL_base.unsqueeze(0))
        dQ_dV = sQ_batch * (2.0 * kZ_Q * QL_base.unsqueeze(0) * V + kI_Q * QL_base.unsqueeze(0))
        dS_dV = (dP_dV - 1j * dQ_dV).to(cdtype)

        Vsafe_c = V_safe.to(cdtype)
        dVsafe_c = dVsafe_dV.to(cdtype)
        du_dV = (dS_dV * Vsafe_c - S * dVsafe_c) / (Vsafe_c * Vsafe_c)

        # --- Complex Jacobians for k (complex residual) ---
        dVc_dtheta = 1j * Vc
        dVc_dV = ejth

        # Network contribution (full matrix)
        J_theta_net = Y.unsqueeze(0) * dVc_dtheta.unsqueeze(1)
        J_V_net = Y.unsqueeze(0) * dVc_dV.unsqueeze(1)

        # SG contribution (diagonal)
        J_theta_sg = torch.diag_embed(neg_dIsg_dtheta)
        J_V_sg = torch.diag_embed(neg_dIsg_dV)

        # Load contributions (diagonal)
        J_theta_load = torch.diag_embed(1j * I_load)
        J_V_load = torch.diag_embed(du_dV * ejth)

        # GFM inverter contribution (diagonal at inverter buses)
        # I_inv = (E*exp(j*theta_v) - V*exp(j*theta)) / Z_v
        # d(-I_inv)/d_theta = j * V*exp(j*theta) / Z_v  (at inverter buses)
        # d(-I_inv)/dV = exp(j*theta) / Z_v  (at inverter buses)
        Z_v_c = Z_v.to(dtype=cdtype, device=device)
        if Z_v_c.dim() == 1:
            Z_v_c = Z_v_c.unsqueeze(0)  # (1, n_inv)
        Y_v = 1.0 / Z_v_c  # (1, n_inv) or (B, n_inv) virtual admittance

        theta_inv = theta[:, inv_bus_map]
        V_inv_bus = V[:, inv_bus_map]
        ejth_inv = torch.exp(1j * theta_inv.to(cdtype))

        # d(-I_inv)/d_theta_k = j * V_k * ejth_k / Z_v = j * Vc_k * Y_v
        neg_dI_inv_dtheta_per = 1j * V_inv_bus.to(cdtype) * ejth_inv * Y_v  # (B, n_inv)
        neg_dI_inv_dtheta_full = torch.zeros(B, n_bus, device=device, dtype=cdtype)
        neg_dI_inv_dtheta_full = neg_dI_inv_dtheta_full.index_add(1, inv_bus_map, neg_dI_inv_dtheta_per)
        J_theta_inv = torch.diag_embed(neg_dI_inv_dtheta_full)

        # d(-I_inv)/dV_k = ejth_k / Z_v = ejth_k * Y_v
        neg_dI_inv_dV_per = ejth_inv * Y_v  # (B, n_inv)
        neg_dI_inv_dV_full = torch.zeros(B, n_bus, device=device, dtype=cdtype)
        neg_dI_inv_dV_full = neg_dI_inv_dV_full.index_add(1, inv_bus_map, neg_dI_inv_dV_per)
        J_V_inv = torch.diag_embed(neg_dI_inv_dV_full)

        # Total Jacobians
        J_theta_full = J_theta_net + J_theta_sg + J_theta_load + J_theta_inv
        J_V_full = J_V_net + J_V_sg + J_V_load + J_V_inv

        if solve_full_theta:
            Jz_c = torch.cat([J_theta_full, J_V_full], dim=2)
        else:
            J_theta_rel = J_theta_full[:, :, 1:]
            Jz_c = torch.cat([J_theta_rel, J_V_full], dim=2)

        J_real_full = torch.cat([Jz_c.real.to(dtype), Jz_c.imag.to(dtype)], dim=1)

        if mask is None:
            return J_real_full
        return J_real_full[:, mask, :]

    # ----------------------------
    # Forward (Newton / chord)
    # ----------------------------
    @staticmethod
    def forward(
        ctx,
        sys,
        delta_batch,
        Eqp_batch,
        Edp_batch,
        theta_v_batch,
        E_inv_batch,
        sP_batch=None,
        sQ_batch=None,
    ):
        n_bus = sys.n_bus
        n_gen = sys.n_gen
        n_inv = sys.n_inv
        B = delta_batch.shape[0]
        device = delta_batch.device
        solve_full_theta = bool(getattr(sys, "newton_solve_full_theta", True))

        # Warm start
        last_V_batch = getattr(sys, "_last_V_batch", None)
        last_theta_batch = getattr(sys, "_last_theta_batch", None)

        if last_V_batch is not None and last_V_batch.shape[0] == B:
            V0_batch = last_V_batch.to(device=device, dtype=torch.float32)
            th0_batch = last_theta_batch.to(device=device, dtype=torch.float32)
            if solve_full_theta:
                z_batch = torch.cat([th0_batch, V0_batch], dim=1)
            else:
                z_batch = torch.cat([th0_batch[:, 1:], V0_batch], dim=1)
        else:
            V0 = getattr(sys, "_last_V", sys.Vset).to(device=device, dtype=torch.float32)
            th0 = getattr(sys, "_last_theta", torch.zeros(n_bus, device=device)).to(torch.float32)
            if solve_full_theta:
                z_init = torch.cat([th0, V0], dim=0)
            else:
                z_init = torch.cat([th0[1:], V0], dim=0)
            z_batch = z_init.unsqueeze(0).expand(B, -1).clone()

        n_iterations = int(getattr(sys, "newton_iterations", 5))

        if solve_full_theta:
            mask = None
        else:
            drop_kind, drop_bus = getattr(sys, "kcl_row_drop", ("imag", 0))
            drop_idx = (drop_bus if drop_kind == "real" else n_bus + drop_bus)
            mask = torch.ones(2 * n_bus, dtype=torch.bool, device=device)
            mask[drop_idx] = False

        delta32 = delta_batch.to(torch.float32)
        Eqp32 = Eqp_batch.to(torch.float32)
        Edp32 = Edp_batch.to(torch.float32)
        theta_v32 = theta_v_batch.to(torch.float32)
        E_inv32 = E_inv_batch.to(torch.float32)

        Y = sys.Y.to(device=device, dtype=torch.complex64)
        Xd_prime = sys.Xd_prime.to(device=device, dtype=torch.float32)
        Xq_prime = sys.Xq_prime.to(device=device, dtype=torch.float32)
        sg_mask_float = (sys.sg_ratio > 0).to(torch.float32).to(device)
        sys._ensure_generator_capability()
        sg_Imax = sys.sg_Imax.to(device=device, dtype=torch.float32)
        PL_base = sys.PL_base.to(device=device, dtype=torch.float32)
        QL_base = sys.QL_base.to(device=device, dtype=torch.float32)
        inv_bus_map = sys.inv_bus_indices.to(device)
        gen_bus_idx = sys.gen_bus_idx.to(device)
        Z_v = sys.Z_v.to(device=device, dtype=torch.complex64)

        if sP_batch is not None and sQ_batch is not None:
            sP32 = sP_batch.to(device=device, dtype=torch.float32)
            sQ32 = sQ_batch.to(device=device, dtype=torch.float32)
        else:
            sP_global = sys.sP.to(device=device, dtype=torch.float32)
            sQ_global = sys.sQ.to(device=device, dtype=torch.float32)
            sP32 = sP_global.unsqueeze(0).expand(B, -1)
            sQ32 = sQ_global.unsqueeze(0).expand(B, -1)

        kZ_P, kI_P, kP_P = sys.kZ_P, sys.kI_P, sys.kP_P
        kZ_Q, kI_Q, kP_Q = sys.kZ_Q, sys.kI_Q, sys.kP_Q

        use_analytic = bool(getattr(sys, "newton_use_analytic_jacobian", True))
        reuse_J = bool(getattr(sys, "newton_reuse_jacobian", True))
        use_lu_reuse = bool(getattr(sys, "newton_use_lu_reuse", True))
        v_lo = float(getattr(sys, "newton_v_clamp_low", 0.35))

        m = (2 * n_bus) if solve_full_theta else (2 * n_bus - 1)
        reg = 1e-8 * torch.eye(m, device=device, dtype=torch.float32)

        def residual_only(z):
            return KCLNewtonBatchedGFM._kcl_residual_masked_batch(
                z_batch=z,
                delta_batch=delta32, Eqp_batch=Eqp32, Edp_batch=Edp32,
                theta_v_batch=theta_v32, E_inv_batch=E_inv32,
                sP_batch=sP32, sQ_batch=sQ32,
                Y=Y, Xd_prime=Xd_prime, Xq_prime=Xq_prime,
                sg_mask_float=sg_mask_float, sg_Imax=sg_Imax,
                PL_base=PL_base, QL_base=QL_base,
                inv_bus_map=inv_bus_map, gen_bus_idx=gen_bus_idx,
                Z_v=Z_v,
                kZ_P=kZ_P, kI_P=kI_P, kP_P=kP_P,
                kZ_Q=kZ_Q, kI_Q=kI_Q, kP_Q=kP_Q,
                mask=mask, solve_full_theta=solve_full_theta,
            )

        def jacobian_z(z):
            return KCLNewtonBatchedGFM._kcl_jacobian_z_masked_batch(
                z_batch=z,
                delta_batch=delta32, Eqp_batch=Eqp32, Edp_batch=Edp32,
                theta_v_batch=theta_v32, E_inv_batch=E_inv32,
                sP_batch=sP32, sQ_batch=sQ32,
                Y=Y, Xd_prime=Xd_prime, Xq_prime=Xq_prime,
                sg_mask_float=sg_mask_float, sg_Imax=sg_Imax,
                PL_base=PL_base, QL_base=QL_base,
                inv_bus_map=inv_bus_map, gen_bus_idx=gen_bus_idx,
                Z_v=Z_v,
                kZ_P=kZ_P, kI_P=kI_P, kP_P=kP_P,
                kZ_Q=kZ_Q, kI_Q=kI_Q, kP_Q=kP_Q,
                mask=mask, solve_full_theta=solve_full_theta,
            )

        def jacobian_z_autograd(z):
            """Fallback: jacrev-based Jacobian (expensive, for debugging)."""
            def residual_single(z1, d, e, ed, tv, ei, sP_vec, sQ_vec):
                F = KCLNewtonBatchedGFM._kcl_residual_masked_batch(
                    z_batch=z1.unsqueeze(0),
                    delta_batch=d.unsqueeze(0), Eqp_batch=e.unsqueeze(0),
                    Edp_batch=ed.unsqueeze(0),
                    theta_v_batch=tv.unsqueeze(0), E_inv_batch=ei.unsqueeze(0),
                    sP_batch=sP_vec.unsqueeze(0), sQ_batch=sQ_vec.unsqueeze(0),
                    Y=Y, Xd_prime=Xd_prime, Xq_prime=Xq_prime,
                    sg_mask_float=sg_mask_float, sg_Imax=sg_Imax,
                    PL_base=PL_base, QL_base=QL_base,
                    inv_bus_map=inv_bus_map, gen_bus_idx=gen_bus_idx,
                    Z_v=Z_v,
                    kZ_P=kZ_P, kI_P=kI_P, kP_P=kP_P,
                    kZ_Q=kZ_Q, kI_Q=kI_Q, kP_Q=kP_Q,
                    mask=mask, solve_full_theta=solve_full_theta,
                )
                return F.squeeze(0)

            def jac_single(z1, d, e, ed, tv, ei, sP_vec, sQ_vec):
                return jacrev(lambda zz: residual_single(zz, d, e, ed, tv, ei, sP_vec, sQ_vec))(z1)

            return vmap(jac_single)(z, delta32, Eqp32, Edp32, theta_v32, E_inv32, sP32, sQ32)

        jac_eval = jacobian_z if use_analytic else jacobian_z_autograd
        theta_dim = n_bus if solve_full_theta else (n_bus - 1)

        tol_F = float(getattr(sys, "newton_tol_F", 1e-3))
        tol_z = float(getattr(sys, "newton_tol_z", 1e-3))
        step_resid_factor = float(getattr(sys, "newton_step_resid_factor", 10.0))
        adaptive = bool(getattr(sys, "newton_adaptive", True))

        actual_iters = 0
        if reuse_J:
            J = jac_eval(z_batch)
            J_reg = J + reg.unsqueeze(0)
            if use_lu_reuse:
                lu, pivots = torch.linalg.lu_factor(J_reg)

            alpha = float(getattr(sys, "newton_damping", 1.0))
            for it in range(n_iterations):
                F = residual_only(z_batch)
                if use_lu_reuse:
                    dz = torch.linalg.lu_solve(lu, pivots, (-F).unsqueeze(-1)).squeeze(-1)
                else:
                    dz = torch.linalg.solve(J_reg, -F)

                if alpha != 1.0:
                    dz = alpha * dz
                z_batch = z_batch + dz
                z_batch = torch.cat(
                    [z_batch[:, :theta_dim], torch.clamp(z_batch[:, theta_dim:], v_lo, 1.60)],
                    dim=1
                )
                actual_iters += 1

                if adaptive and it < n_iterations - 1:
                    F_norm = F.abs().max(dim=1)[0]
                    dz_norm = dz.abs().max(dim=1)[0]
                    z_norm = z_batch.abs().max(dim=1)[0]
                    resid_small = F_norm < tol_F
                    step_small = dz_norm < tol_z * (1.0 + z_norm)
                    step_resid_ok = F_norm < (step_resid_factor * tol_F)
                    converged = resid_small | (step_small & step_resid_ok)
                    if converged.all():
                        break
        else:
            alpha = float(getattr(sys, "newton_damping", 1.0))
            for it in range(n_iterations):
                F = residual_only(z_batch)
                J = jac_eval(z_batch)
                dz = torch.linalg.solve(J + reg.unsqueeze(0), -F)
                if alpha != 1.0:
                    dz = alpha * dz
                z_batch = z_batch + dz
                z_batch = torch.cat(
                    [z_batch[:, :theta_dim], torch.clamp(z_batch[:, theta_dim:], v_lo, 1.60)],
                    dim=1
                )
                actual_iters += 1

                if adaptive and it < n_iterations - 1:
                    F_norm = F.abs().max(dim=1)[0]
                    dz_norm = dz.abs().max(dim=1)[0]
                    z_norm = z_batch.abs().max(dim=1)[0]
                    resid_small = F_norm < tol_F
                    step_small = dz_norm < tol_z * (1.0 + z_norm)
                    step_resid_ok = F_norm < (step_resid_factor * tol_F)
                    converged = resid_small | (step_small & step_resid_ok)
                    if converged.all():
                        break

        F_final = residual_only(z_batch)
        residual_norms = torch.norm(F_final, dim=1)

        sys._last_newton_residuals = residual_norms.detach().cpu()
        sys._last_newton_iterations = actual_iters

        tol = float(getattr(sys, "newton_tol", 1e-5))
        warn_threshold = float(getattr(sys, "newton_warn_threshold", 1e-3))
        poorly_converged = residual_norms > warn_threshold

        if poorly_converged.any() and getattr(sys, "newton_warn_on_nonconvergence", True):
            n_poor = int(poorly_converged.sum().item())
            max_res = float(residual_norms.max().item())
            import warnings
            warnings.warn(
                f"KCL Newton (GFM): {n_poor}/{B} batch elements have residual > {warn_threshold:.1e} "
                f"(max={max_res:.2e}) after {actual_iters} iterations.",
                RuntimeWarning
            )

        if solve_full_theta:
            theta_batch = z_batch[:, :n_bus]
            V_batch = torch.clamp(z_batch[:, n_bus:], v_lo, 1.60)
        else:
            theta_rel_batch = z_batch[:, :n_bus - 1]
            V_batch = torch.clamp(z_batch[:, n_bus - 1:], v_lo, 1.60)
            theta_batch = torch.cat(
                [torch.zeros(B, 1, device=device, dtype=torch.float32), theta_rel_batch],
                dim=1
            )

        sys._last_V_batch = V_batch.detach().clone()
        sys._last_theta_batch = theta_batch.detach().clone()
        sys._last_V = V_batch.mean(dim=0).detach().clone()
        sys._last_theta = theta_batch.mean(dim=0).detach().clone()

        ctx.sys = sys
        ctx.n_bus = n_bus
        ctx.n_gen = n_gen
        ctx.n_inv = n_inv
        ctx.B = B
        ctx.mask = mask
        ctx.solve_full_theta = solve_full_theta
        ctx.z_star = z_batch.detach().clone()
        ctx.delta_batch = delta_batch.detach().clone()
        ctx.Eqp_batch = Eqp_batch.detach().clone()
        ctx.Edp_batch = Edp_batch.detach().clone()
        ctx.theta_v_batch = theta_v_batch.detach().clone()
        ctx.E_inv_batch = E_inv_batch.detach().clone()
        ctx.sP_batch = sP32.detach().clone()
        ctx.sQ_batch = sQ32.detach().clone()

        return V_batch, theta_batch

    # ----------------------------
    # Backward (IFT with analytic dF/dz)
    # ----------------------------
    @staticmethod
    def backward(ctx, grad_V_batch, grad_theta_batch):
        from torch.func import vjp

        sys = ctx.sys
        n_bus, B = ctx.n_bus, ctx.B
        n_inv = ctx.n_inv
        mask = ctx.mask
        solve_full_theta = bool(getattr(ctx, "solve_full_theta", False))
        z_star = ctx.z_star
        device = z_star.device

        z64 = z_star.to(torch.float64)
        delta64 = ctx.delta_batch.to(torch.float64)
        Eqp64 = ctx.Eqp_batch.to(torch.float64)
        Edp64 = ctx.Edp_batch.to(torch.float64)
        theta_v64 = ctx.theta_v_batch.to(torch.float64)
        E_inv64 = ctx.E_inv_batch.to(torch.float64)
        sP64 = ctx.sP_batch.to(torch.float64)
        sQ64 = ctx.sQ_batch.to(torch.float64)

        Y = sys.Y.to(device=device, dtype=torch.complex128)
        Xd_prime = sys.Xd_prime.to(device=device, dtype=torch.float64)
        Xq_prime = sys.Xq_prime.to(device=device, dtype=torch.float64)
        sg_mask_float = (sys.sg_ratio > 0).to(torch.float64).to(device)
        sys._ensure_generator_capability()
        sg_Imax = sys.sg_Imax.to(device=device, dtype=torch.float64)
        PL_base = sys.PL_base.to(device=device, dtype=torch.float64)
        QL_base = sys.QL_base.to(device=device, dtype=torch.float64)
        inv_bus_map = sys.inv_bus_indices.to(device)
        gen_bus_idx = sys.gen_bus_idx.to(device)
        Z_v = sys.Z_v.to(device=device, dtype=torch.complex128)

        kZ_P, kI_P, kP_P = sys.kZ_P, sys.kI_P, sys.kP_P
        kZ_Q, kI_Q, kP_Q = sys.kZ_Q, sys.kI_Q, sys.kP_Q

        # Analytic Jacobian wrt z (float64)
        J = KCLNewtonBatchedGFM._kcl_jacobian_z_masked_batch(
            z_batch=z64,
            delta_batch=delta64, Eqp_batch=Eqp64, Edp_batch=Edp64,
            theta_v_batch=theta_v64, E_inv_batch=E_inv64,
            sP_batch=sP64, sQ_batch=sQ64,
            Y=Y, Xd_prime=Xd_prime, Xq_prime=Xq_prime,
            sg_mask_float=sg_mask_float, sg_Imax=sg_Imax,
            PL_base=PL_base, QL_base=QL_base,
            inv_bus_map=inv_bus_map, gen_bus_idx=gen_bus_idx,
            Z_v=Z_v,
            kZ_P=kZ_P, kI_P=kI_P, kP_P=kP_P,
            kZ_Q=kZ_Q, kI_Q=kI_Q, kP_Q=kP_Q,
            mask=mask, solve_full_theta=solve_full_theta,
        )
        m = (2 * n_bus) if solve_full_theta else (2 * n_bus - 1)

        if solve_full_theta:
            grad_z = torch.cat(
                [grad_theta_batch.to(torch.float64), grad_V_batch.to(torch.float64)],
                dim=1
            )
        else:
            grad_z = torch.cat(
                [grad_theta_batch[:, 1:].to(torch.float64), grad_V_batch.to(torch.float64)],
                dim=1
            )

        J_T = J.transpose(-2, -1)
        reg = 1e-8 * torch.eye(m, device=device, dtype=torch.float64)
        J_T_reg = J_T + reg.unsqueeze(0)

        lu, piv = torch.linalg.lu_factor(J_T_reg)
        v = torch.linalg.lu_solve(lu, piv, grad_z.unsqueeze(-1)).squeeze(-1)

        # VJP for input gradients: dF/dinputs for (delta, Eqp, Edp, theta_v, E_inv)
        def residual_masked_single(z, delta, Eqp, Edp, theta_v, E_inv, sP_vec, sQ_vec):
            F = KCLNewtonBatchedGFM._kcl_residual_masked_batch(
                z_batch=z.unsqueeze(0),
                delta_batch=delta.unsqueeze(0),
                Eqp_batch=Eqp.unsqueeze(0),
                Edp_batch=Edp.unsqueeze(0),
                theta_v_batch=theta_v.unsqueeze(0),
                E_inv_batch=E_inv.unsqueeze(0),
                sP_batch=sP_vec.unsqueeze(0),
                sQ_batch=sQ_vec.unsqueeze(0),
                Y=Y, Xd_prime=Xd_prime, Xq_prime=Xq_prime,
                sg_mask_float=sg_mask_float, sg_Imax=sg_Imax,
                PL_base=PL_base, QL_base=QL_base,
                inv_bus_map=inv_bus_map, gen_bus_idx=gen_bus_idx,
                Z_v=Z_v,
                kZ_P=kZ_P, kI_P=kI_P, kP_P=kP_P,
                kZ_Q=kZ_Q, kI_Q=kI_Q, kP_Q=kP_Q,
                mask=mask, solve_full_theta=solve_full_theta,
            )
            return F.squeeze(0)

        def input_grads_single(v1, z1, d1, eq1, ed1, tv1, ei1, sP1, sQ1):
            def F_of_inputs(delta, Eqp, Edp, theta_v, E_inv):
                return residual_masked_single(z1, delta, Eqp, Edp, theta_v, E_inv, sP1, sQ1)

            _, vjp_fn = vjp(F_of_inputs, d1, eq1, ed1, tv1, ei1)
            g_delta, g_Eqp, g_Edp, g_tv, g_ei = vjp_fn(v1)
            return (-g_delta, -g_Eqp, -g_Edp, -g_tv, -g_ei)

        grad_delta, grad_Eqp, grad_Edp, grad_theta_v, grad_E_inv = vmap(input_grads_single)(
            v, z64, delta64, Eqp64, Edp64, theta_v64, E_inv64, sP64, sQ64
        )

        return (
            None,                                # sys
            grad_delta.to(torch.float32),        # delta_batch
            grad_Eqp.to(torch.float32),          # Eqp_batch
            grad_Edp.to(torch.float32),          # Edp_batch
            grad_theta_v.to(torch.float32),      # theta_v_batch
            grad_E_inv.to(torch.float32),        # E_inv_batch
            None, None,                          # sP_batch, sQ_batch
        )


class IEEE39ControlAffineDAE_GFM(ControlAffineSystem):
    """
    IEEE-39 with 6-state Grid-Forming inverters and Adaptive Virtual Inertia.

    6 states per inverter:
    | Idx | Symbol  | Description              | Dynamics |
    |-----|---------|--------------------------|----------|
    | 0   | theta_v | Virtual rotor angle      | d_theta_v/dt = (omega_v - 1) * omega_s |
    | 1   | omega_v | Virtual rotor frequency  | M_i * d_omega_v = P_set - P_e - D_v*(omega_v-1) - u_i |
    | 2   | M_inv   | Virtual inertia (AVI)    | dM/dt = (M_cmd - M) / tau_m |
    | 3   | V_meas  | Filtered voltage         | dV/dt = (V_bus - V_meas) / T_v |
    | 4   | P_filt  | Filtered active power    | dP/dt = (P_e - P_filt) / T_p |
    | 5   | Q_filt  | Filtered reactive power  | dQ/dt = (Q_e - Q_filt) / T_q |

    Voltage source model: V_inv = E*exp(j*theta_v), I = (V_inv - V_bus)/Z_v
    AVI: M_cmd = M_0 + k_m*|omega_v - 1|, dM/dt = (M_cmd - M)/tau_m

    Control inputs: u = [u_i(n_inv)]  (active power modulation)
    """

    # State indices within the 6-state block
    IDX_THETA_V = 0
    IDX_OMEGA_V = 1
    IDX_M_INV = 2
    IDX_V_MEAS = 3
    IDX_P_FILT = 4
    IDX_Q_FILT = 5

    GFM_STATE_DIM = 6

    def __init__(
        self,
        nominal_params: Dict = None,
        dt: float = 0.01,
        controller_dt: Optional[float] = None,
        scenarios: Optional[Dict] = None,
        n_inverters: int = 10,
        inverter_bus_indices: Optional[List[int]] = None,
        pv_ratio: Optional[np.ndarray] = None,
    ) -> None:

        if nominal_params is None:
            nominal_params = {}

        super().__init__(
            nominal_params=nominal_params,
            dt=dt,
            controller_dt=controller_dt,
            use_linearized_controller=False,
            scenarios=scenarios,
        )

        self.n_gen = 10
        self.n_inv = n_inverters

        # 27-bus system: 10 gen buses + 17 load-only buses
        self.n_bus = 27
        self.gen_bus_idx = torch.arange(self.n_gen, dtype=torch.long)

        if inverter_bus_indices is None:
            inverter_bus_indices = list(range(self.n_gen))
        self.inv_bus_indices = torch.tensor(inverter_bus_indices, dtype=torch.long)

        # Build reduced network (same as GFL6)
        load_equiv = nominal_params.get("reduced_load_equiv", "ward_shunt")
        (Yred_np, G_np, B_np, PL_base_np, QL_base_np, Pg_target_np, Vset_np) = (
            self._build_reduced_network_and_loads(load_equiv=load_equiv)
        )
        self.Y: Tensor = _to_torch_complex(Yred_np)
        self.G: Tensor = torch.tensor(G_np, dtype=torch.float32)
        self.B: Tensor = torch.tensor(B_np, dtype=torch.float32)
        self.PL_base: Tensor = torch.tensor(PL_base_np, dtype=torch.float32)
        self.QL_base: Tensor = torch.tensor(QL_base_np, dtype=torch.float32)
        self.PL_base_nominal: Tensor = self.PL_base.clone()
        self.QL_base_nominal: Tensor = self.QL_base.clone()
        self.Pg_target_total: Tensor = torch.from_numpy(Pg_target_np).float()
        self.Vset: Tensor = torch.from_numpy(Vset_np).float()

        # Generator parameters (identical to GFL6)
        H = np.array([15.15, 21.0, 17.9, 14.3, 13.0, 17.4, 13.2, 12.15, 17.25, 250.0])
        D = np.array([3.46, 2.36, 3.46, 3.46, 3.46, 3.46, 3.46, 3.46, 3.644, 3.644])
        Xd = np.array([0.1000, 0.2950, 0.2495, 0.2620, 0.6700, 0.2540, 0.2950, 0.2900, 0.2106, 0.0200])
        Xd_prime = np.array([0.0697, 0.0310, 0.0531, 0.0436, 0.1320, 0.0500, 0.0490, 0.0570, 0.0570, 0.0060])
        Xq = np.array([0.0690, 0.2820, 0.2370, 0.2580, 0.6200, 0.2410, 0.2920, 0.2800, 0.2050, 0.0190])
        Xq_prime = Xd_prime * Xq / Xd
        Td0_prime = np.array([6.56, 10.2, 5.70, 5.69, 5.40, 7.30, 5.66, 6.70, 4.79, 7.00])
        Tq0_prime = 0.5 * Td0_prime
        self.H_base = torch.from_numpy(H).float()
        self.D_base = torch.from_numpy(D).float()
        self.Xd = torch.from_numpy(Xd).float()
        self.Xd_prime = torch.from_numpy(Xd_prime).float()
        self.Xq = torch.from_numpy(Xq).float()
        self.Xq_prime = torch.from_numpy(Xq_prime).float()
        self.Td0_prime = torch.from_numpy(Td0_prime).float()
        self.Tq0_prime = torch.from_numpy(Tq0_prime).float()

        self.Ka = torch.full((self.n_gen,), 50.0, dtype=torch.float32)
        self.Ta = torch.full((self.n_gen,), 0.05, dtype=torch.float32)
        self.R = torch.full((self.n_gen,), 0.05, dtype=torch.float32)
        self.Tg = torch.full((self.n_gen,), 0.05, dtype=torch.float32)
        self.Tt = torch.full((self.n_gen,), 0.50, dtype=torch.float32)

        if pv_ratio is None:
            pv_ratio = np.zeros(self.n_gen)
        pv_ratio = np.asarray(pv_ratio, dtype=float)
        self.pv_ratio = torch.from_numpy(pv_ratio).float().clamp(0.0, 1.0)
        self.sg_ratio = (1.0 - self.pv_ratio).float()

        # SG capability settings
        self.sg_capability_mode = str(nominal_params.get("sg_capability_mode", "current_limit"))
        self.sg_imax_mult = float(nominal_params.get("sg_imax_mult", 1.15))
        self.sg_imax_min = float(nominal_params.get("sg_imax_min", 1e-3))
        self.sg_Imax: Optional[Tensor] = None

        self.base_freq = 60.0
        self.omega_s = 2.0 * math.pi * self.base_freq

        eps = 1e-6
        self.H = self.H_base * self.sg_ratio.clamp(min=eps)
        self.D = self.D_base * self.sg_ratio

        self.kZ_P, self.kI_P, self.kP_P = 0.80, 0.10, 0.10
        self.kZ_Q, self.kI_Q, self.kP_Q = 0.80, 0.10, 0.10

        # ========== GFM Inverter Parameters ==========
        # Virtual impedance Z_v = R_v + jX_v (per inverter)
        R_v_val = float(nominal_params.get("R_v", 0.01))
        X_v_val = float(nominal_params.get("X_v", 0.10))
        self.Z_v = torch.complex(
            torch.full((self.n_inv,), R_v_val, dtype=torch.float32),
            torch.full((self.n_inv,), X_v_val, dtype=torch.float32),
        )

        # AVI parameters
        self.M_0 = torch.full((self.n_inv,), float(nominal_params.get("M_0", 5.0)), dtype=torch.float32)
        self.k_m = torch.full((self.n_inv,), float(nominal_params.get("k_m", 10.0)), dtype=torch.float32)
        self.tau_m = torch.full((self.n_inv,), float(nominal_params.get("tau_m", 0.10)), dtype=torch.float32)
        self.M_max = torch.full((self.n_inv,), float(nominal_params.get("M_max", 15.0)), dtype=torch.float32)  # Cap at 3*M_0

        # Virtual damping
        self.D_v = torch.full((self.n_inv,), float(nominal_params.get("D_v", 20.0)), dtype=torch.float32)

        # Voltage droop: E = E_0 + K_QV * (V_ref - V_meas)
        self.K_QV = torch.full((self.n_inv,), float(nominal_params.get("K_QV", 0.05)), dtype=torch.float32)

        # Measurement filter time constants
        self.T_v = torch.full((self.n_inv,), float(nominal_params.get("T_v", 0.02)), dtype=torch.float32)
        self.T_p = torch.full((self.n_inv,), float(nominal_params.get("T_p", 0.02)), dtype=torch.float32)
        self.T_q = torch.full((self.n_inv,), float(nominal_params.get("T_q", 0.02)), dtype=torch.float32)

        # Control saturation
        self.u_max = float(nominal_params.get("u_max", 1.0))
        self.u_min = float(nominal_params.get("u_min", -1.0))

        # Setpoints (set at equilibrium)
        self.P_set = torch.zeros(self.n_inv, dtype=torch.float32)
        self.V_ref_0 = self.Vset[self.inv_bus_indices].clone()
        self.E_0 = torch.ones(self.n_inv, dtype=torch.float32)  # Set at equilibrium

        # Solve equilibrium
        (
            V_eq, theta_eq, Pg_eq, Qg_eq,
            P_sg_eq, Q_sg_eq, P_inv_eq, Q_inv_eq,
            delta_eq, Eqp_eq, Edp_eq, Efd_eq, Pm_eq, Pvalve_eq,
            inv_state_eq
        ) = self._solve_equilibrium()

        self._u_eq = torch.zeros(1, self.n_inv, dtype=torch.float32)

        self._goal_point = self._pack_state(
            delta_rel=delta_eq[1:] - delta_eq[0],
            omega=torch.ones(self.n_gen),
            Eqp=Eqp_eq,
            Edp=Edp_eq,
            Efd=Efd_eq,
            Pm=Pm_eq,
            Pvalve=Pvalve_eq,
            inv_states=inv_state_eq,
        ).unsqueeze(0)

        # Dimensions: 69 gen states + 60 inverter states = 129
        n_gen_states = (self.n_gen - 1) + 6 * self.n_gen
        n_inv_states = self.n_inv * self.GFM_STATE_DIM
        self._n_dims = n_gen_states + n_inv_states
        self._n_controls = self.n_inv  # u = [u_i(n_inv)]

        self.kcl_row_drop = ("imag", 0)
        self.newton_solve_full_theta = True

        self._setup_state_limits(Efd_eq, Pm_eq)

        self.newton_iterations = 7  # GFM needs slightly more (voltage source coupling)
        self.newton_use_analytic_jacobian = True
        self.newton_reuse_jacobian = True
        self.newton_use_lu_reuse = True
        self.newton_tol = 1e-5
        self.newton_warn_threshold = 1e-3
        self.newton_warn_on_nonconvergence = True
        self._last_newton_residuals = None
        self._last_newton_iterations = None

        self._repair_equilibrium()
        self._compute_potential_energy_matrix()
        self._equilibrium_cache: Dict[Tuple[int, ...], Dict[str, object]] = {}

    def _compute_potential_energy_matrix(self):
        """Kron-reduce susceptance to machine buses and cache equilibrium angles for W_tilde_p.

        The incremental potential energy from Eq. (V_inc) in the paper is:
          W_tilde_p(theta) = sum_{i<j} C_ij * [cos(theta_ij*) - cos(theta_ij)
                                                - sin(theta_ij*)(theta_ij - theta_ij*)]
        where C_ij = V_i*V_j*B_kron[i,j] and B_kron is the Kron-reduced susceptance
        connecting the n_gen machine buses (load buses eliminated).
        """
        n_m = self.n_gen
        n_l = self.n_bus - n_m
        B = self.B

        B_mm = B[:n_m, :n_m]
        B_ml = B[:n_m, n_m:]
        B_ll = B[n_m:, n_m:]

        # Kron-reduce load buses: B_kron = B_mm - B_ml @ B_ll^{-1} @ B_ml^T
        try:
            B_ll_reg = B_ll + 1e-8 * torch.eye(n_l, dtype=B.dtype)
            B_kron = B_mm - B_ml @ torch.linalg.solve(B_ll_reg, B_ml.T)
        except RuntimeError:
            B_kron = B_mm

        # Voltage-weighted coupling: C_ij = V_eq_i * V_eq_j * B_kron_ij
        V_eq_m = self.Vset[:n_m]
        self._Wp_C = (V_eq_m.unsqueeze(1) * V_eq_m.unsqueeze(0)) * B_kron
        self._Wp_triu = torch.triu(torch.ones(n_m, n_m, dtype=torch.float32), diagonal=1)

        # Equilibrium machine angles
        gp = self._goal_point
        if gp.dim() == 1:
            gp = gp.unsqueeze(0)
        dr_star, _, _, _, _, _, _, inv_states_star = self._unpack_state(gp)
        delta_star = self._angle_reconstruct_batched(dr_star).squeeze(0)  # (n_gen,)
        inv_star = self._unpack_inverter_states(inv_states_star)
        theta_v_star = inv_star['theta_v'].squeeze(0)  # (n_inv,)

        # Effective angle at each machine bus (weighted by SG/GFM ratio)
        sg_r = self.sg_ratio[:n_m]
        pv_r = self.pv_ratio[:n_m]
        theta_eq = (sg_r * delta_star).clone()
        inv_bus = self.inv_bus_indices
        theta_eq.index_add_(0, inv_bus, pv_r[inv_bus] * theta_v_star)
        self._theta_eq_machine = theta_eq

    def _setup_state_limits(self, Efd_eq: Tensor, Pm_eq: Tensor):
        """Set state limits for safe/unsafe mask computation."""
        n = self.n_gen
        n_inv = self.n_inv

        gen_lo = torch.cat([
            -math.pi * torch.ones(n - 1),  # delta_rel
            0.9 * torch.ones(n),            # omega
            0.0 * torch.ones(n),            # Eqp
            -2.0 * torch.ones(n),           # Edp
            0.0 * torch.ones(n),            # Efd
            0.0 * torch.ones(n),            # Pm
            0.0 * torch.ones(n),            # Pvalve
        ])
        gen_hi = torch.cat([
            math.pi * torch.ones(n - 1),
            1.1 * torch.ones(n),
            5.0 * torch.ones(n),
            2.0 * torch.ones(n),
            5.0 * torch.ones(n),
            5.0 * torch.ones(n),
            5.0 * torch.ones(n),
        ])

        # 6 GFM inverter states
        inv_lo = torch.cat([
            -math.pi * torch.ones(n_inv),   # theta_v
            0.9 * torch.ones(n_inv),         # omega_v
            0.5 * torch.ones(n_inv),         # M_inv (min inertia)
            0.5 * torch.ones(n_inv),         # V_meas
            -2.0 * torch.ones(n_inv),        # P_filt
            -2.0 * torch.ones(n_inv),        # Q_filt
        ])
        inv_hi = torch.cat([
            math.pi * torch.ones(n_inv),
            1.1 * torch.ones(n_inv),
            50.0 * torch.ones(n_inv),        # M_inv (max inertia, generous)
            1.5 * torch.ones(n_inv),
            2.0 * torch.ones(n_inv),
            2.0 * torch.ones(n_inv),
        ])

        self._x_lo = torch.cat([gen_lo, inv_lo])
        self._x_hi = torch.cat([gen_hi, inv_hi])

        self.beta_limit = 20.0
        Efd_eq = Efd_eq.detach()
        Pm_eq = Pm_eq.detach()
        sg = self.sg_ratio.detach().to(Efd_eq.device, Efd_eq.dtype).clamp(min=0.05)

        vr_span = torch.clamp(0.60 * sg, min=0.20)
        efd_dn_span = torch.clamp(0.90 * sg, min=0.30)
        efd_up_span = torch.clamp(1.60 * sg, min=0.55)

        self.Vr_min = Efd_eq - vr_span
        self.Vr_max = Efd_eq + vr_span
        self.Efd_min = (Efd_eq - efd_dn_span).clamp(min=0.0)
        self.Efd_max = Efd_eq + efd_up_span

        scale = 1.5
        self.Pref_min = torch.zeros_like(Pm_eq)
        self.Pref_max = scale * Pm_eq
        self.Pvalve_min = torch.zeros_like(Pm_eq)
        self.Pvalve_max = scale * Pm_eq
        self.Pm_min = torch.zeros_like(Pm_eq)
        self.Pm_max = scale * Pm_eq

        self.enable_vr_soft_clip = True
        self.enable_efd_box = True
        self.enable_efd_rate = True
        self.max_dEfd_up = torch.clamp(6.0 * sg, min=1.2)
        self.max_dEfd_dn = torch.clamp(6.0 * sg, min=1.2)
        self.enable_pref_box = True
        self.enable_pvalve_box = True
        self.enable_pm_box = True
        self.enable_pvalve_rate = True
        self.max_dPvalve_up = torch.full_like(Pm_eq, 2.0)
        self.max_dPvalve_dn = torch.full_like(Pm_eq, 2.0)
        self.enable_pm_rate = False
        self.max_dPm_up = None
        self.max_dPm_dn = None

    # ---- Properties ----

    @property
    def n_dims(self) -> int:
        return self._n_dims

    @property
    def n_controls(self) -> int:
        return self._n_controls

    @property
    def goal_point(self) -> torch.Tensor:
        gp = self._goal_point
        return gp if gp.dim() == 2 else gp.unsqueeze(0)

    @property
    def state_limits(self) -> Tuple[Tensor, Tensor]:
        return self._x_hi, self._x_lo

    @property
    def u_eq(self) -> torch.Tensor:
        ue = self._u_eq
        return ue if ue.dim() == 2 else ue.unsqueeze(0)

    @property
    def control_limits(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Control limits for u_i (active power modulation)."""
        n_inv = self.n_inv
        device = self.P_set.device
        upper = torch.full((n_inv,), self.u_max, dtype=torch.float32, device=device)
        lower = torch.full((n_inv,), self.u_min, dtype=torch.float32, device=device)
        return upper, lower

    @property
    def angle_dims(self) -> List[int]:
        return list(range(0, self.n_gen - 1))

    def safe_mask(self, x: torch.Tensor) -> torch.Tensor:
        upper, lower = self.state_limits
        safe_lower = lower + 0.05 * (upper - lower)
        safe_upper = upper - 0.05 * (upper - lower)
        return torch.all(x >= safe_lower, dim=1) & torch.all(x <= safe_upper, dim=1)

    def unsafe_mask(self, x: torch.Tensor) -> torch.Tensor:
        upper, lower = self.state_limits
        return torch.any(x < lower, dim=1) | torch.any(x > upper, dim=1)

    def validate_params(self, params: Dict) -> bool:
        return True

    # ---- Capability models (SG only, GFM doesn't need inverter capability) ----

    def _refresh_generator_capability(self, i_sg_eq: Optional[Tensor] = None):
        """Recompute per-generator stator-current capability."""
        imin = max(float(getattr(self, "sg_imax_min", 1e-3)), 1e-6)
        imult = max(float(getattr(self, "sg_imax_mult", 1.15)), 1.0)

        if i_sg_eq is None:
            p0 = getattr(self, "Pref0", torch.zeros(self.n_gen)).detach()
            v0 = getattr(self, "Vset", torch.ones(self.n_bus))[self.gen_bus_idx].detach()
            i_base = (p0.abs() / v0.clamp(min=1e-3)).to(dtype=torch.float32)
        else:
            i_base = i_sg_eq.detach().abs().to(dtype=torch.float32)

        sg = self.sg_ratio.detach().to(i_base.device, i_base.dtype)
        i_base = i_base * (sg > 0).to(i_base.dtype)
        self.sg_Imax = torch.clamp(imult * i_base, min=imin).to(dtype=i_base.dtype, device=i_base.device)

    def _ensure_generator_capability(self):
        if not hasattr(self, "sg_capability_mode"):
            self.sg_capability_mode = "current_limit"
        if not hasattr(self, "sg_imax_mult"):
            self.sg_imax_mult = 1.15
        if not hasattr(self, "sg_imax_min"):
            self.sg_imax_min = 1e-3
        sg_imax = getattr(self, "sg_Imax", None)
        if (sg_imax is None) or (not isinstance(sg_imax, torch.Tensor)) or (sg_imax.numel() != self.n_gen):
            self._refresh_generator_capability()

    def _apply_generator_current_capability(self, id_raw: Tensor, iq_raw: Tensor) -> Tuple[Tensor, Tensor]:
        """Enforce synchronous-generator stator current magnitude capability."""
        self._ensure_generator_capability()
        imax = self.sg_Imax.to(device=id_raw.device, dtype=id_raw.dtype)
        if id_raw.dim() == 2:
            imax = imax.unsqueeze(0)
            sg_mask = (self.sg_ratio > 0).to(device=id_raw.device, dtype=id_raw.dtype).unsqueeze(0)
        else:
            sg_mask = (self.sg_ratio > 0).to(device=id_raw.device, dtype=id_raw.dtype)
        i_mag = torch.sqrt(torch.clamp(id_raw.square() + iq_raw.square(), min=1e-12))
        scale = torch.minimum(torch.ones_like(i_mag), imax / i_mag.clamp(min=1e-9))
        scale = scale * sg_mask
        return id_raw * scale, iq_raw * scale

    # ---- Packing/Unpacking ----

    def _pack_state(self, *, delta_rel, omega, Eqp, Edp, Efd, Pm, Pvalve, inv_states):
        return torch.cat([delta_rel, omega, Eqp, Edp, Efd, Pm, Pvalve, inv_states], dim=-1)

    def _unpack_state(self, x: Tensor):
        n = self.n_gen
        n_inv = self.n_inv

        i0 = 0
        delta_rel = x[..., i0:i0 + n - 1]; i0 += (n - 1)
        omega = x[..., i0:i0 + n]; i0 += n
        Eqp = x[..., i0:i0 + n]; i0 += n
        Edp = x[..., i0:i0 + n]; i0 += n
        Efd = x[..., i0:i0 + n]; i0 += n
        Pm = x[..., i0:i0 + n]; i0 += n
        Pvalve = x[..., i0:i0 + n]; i0 += n

        inv_flat = x[..., i0:]
        if x.dim() == 2:
            inv_states = inv_flat.view(-1, n_inv, self.GFM_STATE_DIM)
        else:
            inv_states = inv_flat.view(n_inv, self.GFM_STATE_DIM)

        return delta_rel, omega, Eqp, Edp, Efd, Pm, Pvalve, inv_states

    def _unpack_inverter_states(self, inv_states: Tensor) -> Dict[str, Tensor]:
        return {
            'theta_v': inv_states[..., self.IDX_THETA_V],
            'omega_v': inv_states[..., self.IDX_OMEGA_V],
            'M_inv': inv_states[..., self.IDX_M_INV],
            'V_meas': inv_states[..., self.IDX_V_MEAS],
            'P_filt': inv_states[..., self.IDX_P_FILT],
            'Q_filt': inv_states[..., self.IDX_Q_FILT],
        }

    def _angle_reconstruct_batched(self, delta_rel: torch.Tensor) -> torch.Tensor:
        d0 = getattr(self, "delta0_goal", torch.tensor(0.0))
        if d0.dim() == 0:
            d0 = d0.view(1)
        B = delta_rel.shape[0]
        d0_batch = d0.expand(B, 1).to(delta_rel.device)
        return torch.cat([d0_batch, d0_batch + delta_rel], dim=-1)

    def _sg_mask(self) -> Tensor:
        return (self.sg_ratio > 0.0)

    def _compute_E_inv(self, V_meas: Tensor) -> Tensor:
        """Compute GFM internal EMF with optional voltage droop.

        E = E_0 + K_QV * (V_ref - V_meas)
        """
        device = V_meas.device
        E_0 = self.E_0.to(device=device, dtype=V_meas.dtype)
        K_QV = self.K_QV.to(device=device, dtype=V_meas.dtype)
        V_ref = self.V_ref_0.to(device=device, dtype=V_meas.dtype)

        if V_meas.dim() == 2:
            E_0 = E_0.unsqueeze(0)
            K_QV = K_QV.unsqueeze(0)
            V_ref = V_ref.unsqueeze(0)

        return E_0 + K_QV * (V_ref - V_meas)

    def _compute_gfm_power(self, theta_v: Tensor, E_inv: Tensor,
                            V_bus: Tensor, theta_bus: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute GFM active and reactive power injection into the bus.

        P_e + jQ_e = V_bus * conj(I_inv)
        where I_inv = (E*exp(j*theta_v) - V_bus*exp(j*theta_bus)) / Z_v
        """
        device = theta_v.device
        Z_v = self.Z_v.to(device=device, dtype=torch.complex64)
        if Z_v.dim() == 1:
            Z_v = Z_v.unsqueeze(0)

        V_inv = E_inv.to(torch.complex64) * torch.exp(1j * theta_v.to(torch.complex64))
        V_bus_c = V_bus.to(torch.complex64) * torch.exp(1j * theta_bus.to(torch.complex64))

        I_inv = (V_inv - V_bus_c) / Z_v
        S_e = V_bus_c * I_inv.conj()

        P_e = S_e.real.to(theta_v.dtype)
        Q_e = S_e.imag.to(theta_v.dtype)
        return P_e, Q_e

    # ---- Dynamics ----

    def _f(self, x: torch.Tensor, params: dict,
           sP_batch: Optional[torch.Tensor] = None,
           sQ_batch: Optional[torch.Tensor] = None,
           V_solved: Optional[torch.Tensor] = None,
           theta_solved: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute drift dynamics f(x) for the control-affine system dx/dt = f(x) + g(x)*u."""
        import torch.nn.functional as F

        def _soft_clamp(v, lo, hi, beta):
            return v + F.softplus(lo - v, beta=beta) - F.softplus(v - hi, beta=beta)

        def _maybe_box(v, lo, hi, enable, beta):
            return _soft_clamp(v, lo, hi, beta) if enable else v

        device = x.device
        n = self.n_gen
        n_inv = self.n_inv
        B = x.shape[0]

        delta_rel, omega, Eqp, Edp, Efd, Pm, Pvalve, inv_states = self._unpack_state(x)
        inv = self._unpack_inverter_states(inv_states)

        theta_v = inv['theta_v']      # (B, n_inv)
        omega_v = inv['omega_v']      # (B, n_inv)
        M_inv = inv['M_inv']          # (B, n_inv)
        V_meas = inv['V_meas']        # (B, n_inv)

        # Compute internal EMF with voltage droop
        E_inv = self._compute_E_inv(V_meas)  # (B, n_inv)

        delta_batch = self._angle_reconstruct_batched(delta_rel)

        if V_solved is not None and theta_solved is not None:
            V_batch = V_solved
            theta_batch = theta_solved
        else:
            V_batch, theta_batch = self._solve_kcl_newton_batched(
                delta_batch, Eqp, Edp, theta_v, E_inv,
                sP_batch=sP_batch, sQ_batch=sQ_batch
            )

        beta = getattr(self, "beta_limit", 20.0)

        # ========== Generator Dynamics (Two-Axis Model — same as GFL6) ==========
        gen_bus_idx = self.gen_bus_idx.to(device)
        V_gen = V_batch[:, gen_bus_idx]
        theta_gen = theta_batch[:, gen_bus_idx]

        Xd_p = self.Xd_prime.to(device).unsqueeze(0)
        Xq_p = self.Xq_prime.to(device).unsqueeze(0)
        Vq_batch = V_gen * torch.cos(delta_batch - theta_gen)
        Vd_batch = V_gen * torch.sin(delta_batch - theta_gen)
        Id_batch = (Eqp - Vq_batch) / Xd_p
        Iq_batch = (Vd_batch - Edp) / Xq_p
        Id_batch, Iq_batch = self._apply_generator_current_capability(Id_batch, Iq_batch)
        Pe_batch = Vd_batch * Id_batch + Vq_batch * Iq_batch

        Vref = self.Vref.to(device).unsqueeze(0)
        Ka = self.Ka.to(device).unsqueeze(0)
        Vr_raw = Ka * (Vref - V_gen)

        Vr_min = self.Vr_min.unsqueeze(0).to(device)
        Vr_max = self.Vr_max.unsqueeze(0).to(device)
        Vr_cmd = _maybe_box(Vr_raw, Vr_min, Vr_max, getattr(self, "enable_vr_soft_clip", False), beta)

        Efd_min = self.Efd_min.unsqueeze(0).to(device)
        Efd_max = self.Efd_max.unsqueeze(0).to(device)
        Efd_tgt = _maybe_box(Vr_cmd, Efd_min, Efd_max, getattr(self, "enable_efd_box", False), beta)

        Ta = self.Ta.to(device).unsqueeze(0)
        dEfd = (Efd_tgt - Efd) / Ta
        if getattr(self, "enable_efd_rate", False):
            up = getattr(self, "max_dEfd_up", None)
            dn = getattr(self, "max_dEfd_dn", None)
            if up is not None and dn is not None:
                up = torch.as_tensor(up, device=device, dtype=dEfd.dtype).unsqueeze(0)
                dn = torch.as_tensor(dn, device=device, dtype=dEfd.dtype).unsqueeze(0)
                dEfd = torch.clamp(dEfd, min=-dn, max=up)

        H_sg = self.H.to(device).unsqueeze(0)
        D_sg = self.D.to(device).unsqueeze(0)
        domega = (1.0 / (2.0 * H_sg)) * (Pm - Pe_batch - D_sg * self.omega_s * (omega - 1.0))
        ddelta = self.omega_s * (omega - omega[..., :1])

        Td0_prime = self.Td0_prime.to(device).unsqueeze(0)
        Tq0_prime = self.Tq0_prime.to(device).unsqueeze(0)
        Xd_full = self.Xd.to(device).unsqueeze(0)
        Xq_full = self.Xq.to(device).unsqueeze(0)
        dEqp = (Efd - Eqp - (Xd_full - Xd_p) * Id_batch) / Td0_prime
        dEdp = (-Edp + (Xq_full - Xq_p) * Iq_batch) / Tq0_prime

        Pref0 = self.Pref0.to(device).unsqueeze(0)
        R = self.R.to(device).unsqueeze(0)
        Pref_raw = Pref0 + self.omega_s * (1.0 - omega) / R

        Pref_min = self.Pref_min.to(device).unsqueeze(0)
        Pref_max = self.Pref_max.to(device).unsqueeze(0)
        Pref_cmd = _maybe_box(Pref_raw, Pref_min, Pref_max,
                              getattr(self, "enable_pref_box", False), beta)

        Pvalve_min = self.Pvalve_min.to(device).unsqueeze(0)
        Pvalve_max = self.Pvalve_max.to(device).unsqueeze(0)
        Pvalve_tgt = _maybe_box(Pref_cmd, Pvalve_min, Pvalve_max,
                                getattr(self, "enable_pvalve_box", False), beta)

        Tg = self.Tg.to(device).unsqueeze(0)
        dPvalve = (Pvalve_tgt - Pvalve) / Tg

        if getattr(self, "enable_pvalve_rate", False):
            up = getattr(self, "max_dPvalve_up", None)
            dn = getattr(self, "max_dPvalve_dn", None)
            if up is not None and dn is not None:
                up = torch.as_tensor(up, device=device, dtype=dPvalve.dtype).unsqueeze(0)
                dn = torch.as_tensor(dn, device=device, dtype=dPvalve.dtype).unsqueeze(0)
                dPvalve = torch.clamp(dPvalve, min=-dn, max=up)

        Pm_min = self.Pm_min.to(device).unsqueeze(0)
        Pm_max = self.Pm_max.to(device).unsqueeze(0)
        Pm_tgt = _maybe_box(Pvalve, Pm_min, Pm_max,
                            getattr(self, "enable_pm_box", False), beta)

        Tt = self.Tt.to(device).unsqueeze(0)
        dPm = (Pm_tgt - Pm) / Tt

        if getattr(self, "enable_pm_rate", False):
            up = getattr(self, "max_dPm_up", None)
            dn = getattr(self, "max_dPm_dn", None)
            if up is not None and dn is not None:
                up = torch.as_tensor(up, device=device, dtype=dPm.dtype).unsqueeze(0)
                dn = torch.as_tensor(dn, device=device, dtype=dPm.dtype).unsqueeze(0)
                dPm = torch.clamp(dPm, min=-dn, max=up)

        # ========== GFM Inverter Dynamics (6 states) ==========
        inv_bus_idx = self.inv_bus_indices.to(device)
        V_inv_bus = V_batch[:, inv_bus_idx]           # (B, n_inv)
        theta_inv_bus = theta_batch[:, inv_bus_idx]   # (B, n_inv)

        # Compute electrical power injection P_e, Q_e
        P_e_inv, Q_e_inv = self._compute_gfm_power(
            theta_v, E_inv, V_inv_bus, theta_inv_bus
        )

        # Load parameters
        P_set = self.P_set.to(device).unsqueeze(0)       # (1, n_inv)
        D_v = self.D_v.to(device).unsqueeze(0)            # (1, n_inv)
        M_0 = self.M_0.to(device).unsqueeze(0)            # (1, n_inv)
        k_m = self.k_m.to(device).unsqueeze(0)            # (1, n_inv)
        tau_m = self.tau_m.to(device).unsqueeze(0)         # (1, n_inv)
        T_v = self.T_v.to(device).unsqueeze(0)
        T_p = self.T_p.to(device).unsqueeze(0)
        T_q = self.T_q.to(device).unsqueeze(0)

        # 1. Virtual angle: d_theta_v/dt = (omega_v - 1) * omega_s
        d_theta_v = (omega_v - 1.0) * self.omega_s

        # 2. Swing equation: M_i * d_omega_v = P_set - P_e - D_v*(omega_v - 1) - u_i
        #    Drift (u=0): d_omega_v = (P_set - P_e - D_v*(omega_v - 1)) / M_i
        M_safe = torch.clamp(M_inv, min=0.1)  # Prevent division by zero
        d_omega_v = (P_set - P_e_inv - D_v * (omega_v - 1.0)) / M_safe

        # 3. AVI dynamics: dM/dt = (M_cmd - M) / tau_m
        #    M_cmd = M_0 + k_m * |omega_v - 1|, capped at M_max
        omega_dev = omega_v - 1.0
        M_max = self.M_max.to(device).unsqueeze(0)
        M_cmd = torch.clamp(M_0 + k_m * torch.abs(omega_dev), max=M_max)
        d_M_inv = (M_cmd - M_inv) / tau_m

        # 4. Voltage filter: dV_meas/dt = (V_bus - V_meas) / T_v
        d_V_meas = (V_inv_bus - V_meas) / T_v

        # 5. Power filters
        d_P_filt = (P_e_inv - inv['P_filt']) / T_p
        d_Q_filt = (Q_e_inv - inv['Q_filt']) / T_q

        # Pack inverter derivatives: (B, n_inv, 6)
        d_inv = torch.stack([
            d_theta_v,
            d_omega_v,
            d_M_inv,
            d_V_meas,
            d_P_filt,
            d_Q_filt,
        ], dim=-1)

        d_inv_flat = d_inv.view(B, -1)

        f = torch.cat([
            ddelta[..., 1:],
            domega,
            dEqp,
            dEdp,
            dEfd,
            dPm,
            dPvalve,
            d_inv_flat,
        ], dim=-1)

        return f

    def _g(self, x: torch.Tensor, params: dict) -> torch.Tensor:
        """Control matrix for u = [u_i(n_inv)].

        The control input u_i enters the swing equation:
            M_i * d_omega_v = P_set - P_e - D_v*(omega_v-1) - u_i
        So g at (omega_v row, u_i col) = -1/M_i
        """
        B = x.shape[0]
        n = self.n_gen
        n_inv = self.n_inv
        device = x.device

        G = x.new_zeros(B, self.n_dims, self.n_controls)

        gen_state_end = (n - 1) + 6 * n  # delta_rel(9) + omega,Eqp,Edp,Efd,Pm,Pvalve

        # Extract M_inv from state to compute state-dependent g
        _, _, _, _, _, _, _, inv_states = self._unpack_state(x)
        inv = self._unpack_inverter_states(inv_states)
        M_inv = inv['M_inv']  # (B, n_inv)
        M_safe = torch.clamp(M_inv, min=0.1)

        for j in range(n_inv):
            inv_start = gen_state_end + j * self.GFM_STATE_DIM

            # u_i -> omega_v: d_omega_v/dt += (-1/M_i) * u_i
            row_omega_v = inv_start + self.IDX_OMEGA_V
            col_u = j
            G[:, row_omega_v, col_u] = -1.0 / M_safe[:, j]

        return G

    def _solve_kcl_newton_batched(
        self, delta_batch, Eqp_batch, Edp_batch, theta_v_batch, E_inv_batch,
        sP_batch=None, sQ_batch=None,
    ):
        """Batched Newton solver for GFM voltage-source model."""
        return KCLNewtonBatchedGFM.apply(
            self, delta_batch, Eqp_batch, Edp_batch,
            theta_v_batch, E_inv_batch,
            sP_batch, sQ_batch
        )

    def control_affine_dynamics(self, x, params=None, sP_batch=None, sQ_batch=None,
                                 V_solved=None, theta_solved=None):
        f = self._f(x, params or {}, sP_batch=sP_batch, sQ_batch=sQ_batch,
                    V_solved=V_solved, theta_solved=theta_solved)
        g = self._g(x, params or {})
        return f, g

    # ---- Lyapunov observables ----

    def lyapunov_observables(self, x: torch.Tensor) -> Dict[str, Tensor]:
        """Extract Lyapunov-relevant observables from state vector.

        Returns dict with:
          omega_dev:   (B, n_inv) frequency deviation omega_v - 1
          M_inv:       (B, n_inv) current virtual inertia
          M_dot:       (B, n_inv) dM/dt = (M_cmd - M)/tau_m
          M_cmd:       (B, n_inv) inertia command (clamped)
          D_v:         (B, n_inv) virtual damping
          A_i:         (B, n_inv) non-passive coefficient = 0.5*M_dot - D_v
        """
        device = x.device
        if x.dim() == 1:
            x = x.unsqueeze(0)
        _, _, _, _, _, _, _, inv_states = self._unpack_state(x)
        inv = self._unpack_inverter_states(inv_states)

        omega_dev = inv['omega_v'] - 1.0
        M_inv = inv['M_inv']

        M_0 = self.M_0.to(device).unsqueeze(0)
        k_m = self.k_m.to(device).unsqueeze(0)
        tau_m = self.tau_m.to(device).unsqueeze(0)
        M_max = self.M_max.to(device).unsqueeze(0)
        D_v = self.D_v.to(device).unsqueeze(0)

        M_cmd = torch.clamp(M_0 + k_m * torch.abs(omega_dev), max=M_max)
        M_dot = (M_cmd - M_inv) / tau_m
        A_i = 0.5 * M_dot - D_v

        return {
            'omega_dev': omega_dev,
            'M_inv': M_inv,
            'M_dot': M_dot,
            'M_cmd': M_cmd,
            'D_v': D_v.expand_as(M_inv),
            'A_i': A_i,
        }

    def incremental_energy(self, x: torch.Tensor,
                           theta_star: Optional[Tensor] = None) -> Tensor:
        """Compute incremental energy from Eq. (V_inc):

          V(omega, theta, t) = 0.5*sum_i M_i*omega_i^2 + V_sg_kinetic + W_tilde_p(theta)

        where W_tilde_p is the Bregman-type incremental potential energy:
          W_tilde_p = sum_{i<j} C_ij * [cos(theta_ij*) - cos(theta_ij)
                                        - sin(theta_ij*)(theta_ij - theta_ij*)]
        using the Kron-reduced susceptance matrix connecting machine buses.

        For loss computation only. Returns scalar per batch element: (B,).
        """
        device = x.device
        if x.dim() == 1:
            x = x.unsqueeze(0)
        B_size = x.shape[0]

        obs = self.lyapunov_observables(x)
        V_kinetic = 0.5 * (obs['M_inv'] * obs['omega_dev'] ** 2).sum(dim=-1)

        # SG kinetic energy: 0.5 * 2H_j * (omega_j - 1)^2
        delta_rel, omega, _, _, _, _, _, inv_states = self._unpack_state(x)
        H_sg = self.H.to(device).unsqueeze(0)
        omega_sg_dev = omega - 1.0
        V_sg_kinetic = (H_sg * omega_sg_dev ** 2).sum(dim=-1)

        # --- W_tilde_p(theta): incremental potential energy (Eq. V_inc / Wp_inc) ---
        n_m = self.n_gen
        delta = self._angle_reconstruct_batched(delta_rel)  # (B, n_gen)
        inv = self._unpack_inverter_states(inv_states)
        theta_v = inv['theta_v']  # (B, n_inv)

        # Effective machine angle at each bus: sg_ratio*delta + pv_ratio*theta_v
        sg_r = self.sg_ratio[:n_m].to(device).unsqueeze(0)  # (1, n_m)
        pv_r = self.pv_ratio[:n_m].to(device).unsqueeze(0)  # (1, n_m)
        inv_bus = self.inv_bus_indices.to(device)

        theta_machine = sg_r * delta  # (B, n_m)
        gfm_contrib = torch.zeros(B_size, n_m, device=device, dtype=x.dtype)
        gfm_contrib.index_add_(1, inv_bus, pv_r[:, inv_bus].expand(B_size, -1) * theta_v)
        theta_machine = theta_machine + gfm_contrib  # (B, n_m)

        # Equilibrium angles
        if theta_star is not None:
            theta_eq = theta_star.to(device)
        else:
            theta_eq = self._theta_eq_machine.to(device)  # (n_m,)

        # Angle differences: (B, n_m, n_m) and (n_m, n_m)
        theta_diff = theta_machine.unsqueeze(-1) - theta_machine.unsqueeze(-2)
        theta_eq_diff = theta_eq.unsqueeze(-1) - theta_eq.unsqueeze(-2)

        # Bregman divergence of -cos at equilibrium:
        # D_{-cos}(x || x*) = cos(x*) - cos(x) - sin(x*)(x - x*)
        bregman = (torch.cos(theta_eq_diff).unsqueeze(0)
                   - torch.cos(theta_diff)
                   - torch.sin(theta_eq_diff).unsqueeze(0) * (theta_diff - theta_eq_diff.unsqueeze(0)))

        # Weight by C_ij and sum upper triangle (i < j)
        C = self._Wp_C.to(device)      # (n_m, n_m)
        triu = self._Wp_triu.to(device) # (n_m, n_m) upper triangle mask
        W_tilde_p = (C.unsqueeze(0) * bregman * triu.unsqueeze(0)).sum(dim=(-2, -1))

        return V_kinetic + V_sg_kinetic + W_tilde_p

    # ---- Equilibrium ----

    def _solve_equilibrium(self):
        """Solve equilibrium for GFM inverters using voltage-behind-impedance initialization.

        At equilibrium:
            - omega_v = 1 (synchronous)
            - theta_v = angle(V_bus + Z_v * I_inv) (behind-impedance angle)
            - M_inv = M_0 (since |omega_v - 1| = 0)
            - V_meas = V_bus
            - P_filt = P_e, Q_filt = Q_e
        """
        device = self.Vset.device
        n = self.n_gen
        n_bus = self.n_bus
        n_inv = self.n_inv

        self.sP, self.sQ = self._calibrate_zip(self.Vset, self.PL_base, self.QL_base)

        theta_eq = torch.zeros(n_bus, dtype=torch.float32, device=device)
        V_eq = self.Vset.clone()

        Vc_eq = V_eq * torch.exp(1j * theta_eq)
        I_net = self.Y.to(torch.complex64) @ Vc_eq.to(torch.complex64)
        S_net = Vc_eq.to(torch.complex64) * I_net.conj()

        Pz = self.sP * self.kZ_P * self.PL_base * (V_eq ** 2)
        Pi = self.sP * self.kI_P * self.PL_base * V_eq
        Pp = self.sP * self.kP_P * self.PL_base
        Qz = self.sQ * self.kZ_Q * self.QL_base * (V_eq ** 2)
        Qi = self.sQ * self.kI_Q * self.QL_base * V_eq
        Qp = self.sQ * self.kP_Q * self.QL_base
        S_load = (Pz + Pi + Pp) + 1j * (Qz + Qi + Qp)

        S_gen_full = S_net + S_load.to(torch.complex64)

        gen_bus_idx = self.gen_bus_idx
        S_gen = S_gen_full[gen_bus_idx]

        sg = self.sg_ratio.to(torch.float32)
        pv = self.pv_ratio.to(torch.float32)

        P_gen_target = self.Pg_target_total.to(device=device, dtype=torch.float32)
        P_sg_eq = sg * P_gen_target
        P_inv_gen_eq = pv * P_gen_target

        Q_sg_eq = sg * S_gen.imag.to(torch.float32)
        Q_inv_gen_eq = pv * S_gen.imag.to(torch.float32)

        S_sg_eq = P_sg_eq.to(torch.complex64) + 1j * Q_sg_eq.to(torch.complex64)

        P_inv_bus_eq = torch.zeros(n_bus, dtype=torch.float32, device=device)
        Q_inv_bus_eq = torch.zeros(n_bus, dtype=torch.float32, device=device)
        P_inv_bus_eq[gen_bus_idx] = P_inv_gen_eq
        Q_inv_bus_eq[gen_bus_idx] = Q_inv_gen_eq
        P_inv_eq = P_inv_bus_eq[self.inv_bus_indices]
        Q_inv_eq = Q_inv_bus_eq[self.inv_bus_indices]

        # Generator initialization (same as GFL6)
        Vc_gen = Vc_eq[gen_bus_idx]
        V_gen = V_eq[gen_bus_idx]
        theta_gen = theta_eq[gen_bus_idx]

        I_sg_eq = torch.zeros(n, dtype=torch.complex64, device=device)
        mask = self.sg_ratio > 0
        I_sg_eq[mask] = (S_sg_eq[mask] / Vc_gen[mask]).conj()

        E_behind_Xq = Vc_gen.to(torch.complex64).clone()
        E_behind_Xq[mask] = Vc_gen[mask] + 1j * self.Xq[mask].to(torch.complex64) * I_sg_eq[mask]
        delta_eq = torch.angle(E_behind_Xq).float()

        Vq_eq = V_gen * torch.cos(delta_eq - theta_gen)
        Vd_eq = V_gen * torch.sin(delta_eq - theta_gen)

        I_dq = I_sg_eq * torch.exp(-1j * delta_eq.to(torch.complex64))
        Iq_eq = I_dq.real.float()
        Id_eq = -I_dq.imag.float()

        Eqp_eq = Vq_eq + self.Xd_prime * Id_eq
        Edp_eq = Vd_eq - self.Xq_prime * Iq_eq
        Efd_eq = Eqp_eq + (self.Xd - self.Xd_prime) * Id_eq
        self.Vref = V_gen + Efd_eq / self.Ka
        Pm_eq = P_sg_eq.clone()
        Pvalve_eq = Pm_eq.clone()
        self.Pref0 = Pm_eq.clone()

        self.delta0_goal = delta_eq[0].detach().clone()

        # ========== GFM Inverter Equilibrium (voltage-behind-impedance) ==========
        V_inv_eq = V_eq[self.inv_bus_indices]
        theta_inv_eq = theta_eq[self.inv_bus_indices]
        Vc_inv_eq = V_inv_eq.to(torch.complex64) * torch.exp(1j * theta_inv_eq.to(torch.complex64))

        # Current from power injection: I = conj(S/V)
        S_inv_eq = P_inv_eq.to(torch.complex64) + 1j * Q_inv_eq.to(torch.complex64)
        I_inv_eq = (S_inv_eq / Vc_inv_eq).conj()

        # Internal voltage: V_inv = V_bus + Z_v * I
        Z_v = self.Z_v.to(device=device, dtype=torch.complex64)
        V_internal_eq = Vc_inv_eq + Z_v * I_inv_eq

        E_0_eq = V_internal_eq.abs().float()
        theta_v_eq = torch.angle(V_internal_eq).float()

        # Store equilibrium EMF
        self.E_0 = E_0_eq.clone()
        self.P_set = P_inv_eq.clone()
        self.V_ref_0 = V_inv_eq.clone()

        # GFM inverter state equilibrium
        omega_v_eq = torch.ones(n_inv, dtype=torch.float32, device=device)
        M_inv_eq = self.M_0.clone().to(device)  # M_cmd = M_0 at omega_v = 1
        V_meas_eq = V_inv_eq.clone()
        P_filt_eq = P_inv_eq.clone()
        Q_filt_eq = Q_inv_eq.clone()

        self._refresh_generator_capability(I_sg_eq)

        inv_state_eq = torch.stack([
            theta_v_eq, omega_v_eq, M_inv_eq, V_meas_eq, P_filt_eq, Q_filt_eq
        ], dim=-1).flatten()

        Pg_eq = P_sg_eq + P_inv_eq.sum()
        Qg_eq = Q_sg_eq + Q_inv_eq.sum()

        return (V_eq, theta_eq, Pg_eq, Qg_eq,
                P_sg_eq, Q_sg_eq, P_inv_eq, Q_inv_eq,
                delta_eq, Eqp_eq, Edp_eq, Efd_eq, Pm_eq, Pvalve_eq,
                inv_state_eq)

    def _calibrate_zip(self, V_ref, P_ref, Q_ref):
        V = V_ref.detach()
        denomP = (self.kZ_P * V ** 2 + self.kI_P * V + self.kP_P) * torch.clamp(self.PL_base, min=1e-8)
        denomQ = (self.kZ_Q * V ** 2 + self.kI_Q * V + self.kP_Q) * torch.clamp(self.QL_base.abs(), min=1e-8) * torch.sign(self.QL_base).clamp(min=1.0)
        sP = (P_ref / denomP).clamp(-10.0, 10.0)
        sQ = (Q_ref / denomQ).clamp(-10.0, 10.0)
        return sP, sQ

    def _repair_equilibrium(self):
        """Repair equilibrium to ensure consistency with GFM voltage-source model."""
        gp = self.goal_point[0]
        delta_rel, omega, Eqp, Edp, Efd, Pm, Pvalve, inv_states = self._unpack_state(gp)
        inv = self._unpack_inverter_states(inv_states)

        V_init = self.Vset.clone()
        theta_init = torch.zeros_like(V_init)

        self._last_V = V_init.detach().clone().to(torch.float32)
        self._last_theta = theta_init.detach().clone().to(torch.float32)

        saved_iters = self.newton_iterations
        saved_reuse = self.newton_reuse_jacobian
        saved_warn = self.newton_warn_on_nonconvergence
        saved_sg_imax = getattr(self, "sg_Imax", None)
        self.newton_iterations = 15
        self.newton_reuse_jacobian = False
        self.newton_warn_on_nonconvergence = False
        if saved_sg_imax is not None:
            self.sg_Imax = torch.full_like(saved_sg_imax, 1e6)

        delta_batch = self._angle_reconstruct_batched(delta_rel.unsqueeze(0))
        delta_full = delta_batch.squeeze(0)
        Edp_cur = Edp.clone()

        gen_bus_idx = self.gen_bus_idx

        # Compute E_inv for equilibrium
        E_inv_eq = self._compute_E_inv(inv['V_meas'])

        for _repair_iter in range(50):
            with torch.no_grad():
                V_eq, theta_eq = self._solve_kcl_newton_batched(
                    delta_batch,
                    Eqp.unsqueeze(0),
                    Edp_cur.unsqueeze(0),
                    inv['theta_v'].unsqueeze(0),
                    E_inv_eq.unsqueeze(0),
                )
            V_eq = V_eq.squeeze(0)
            theta_eq = theta_eq.squeeze(0)

            V_gen = V_eq[gen_bus_idx]
            theta_gen = theta_eq[gen_bus_idx]

            Vd_eq = V_gen * torch.sin(delta_full - theta_gen)
            Edp_new = (self.Xq - self.Xq_prime) * Vd_eq / self.Xq
            if (Edp_new - Edp_cur).abs().max() < 1e-6:
                Edp_cur = Edp_new
                break
            Edp_cur = Edp_new

        V_gen = V_eq[gen_bus_idx]
        theta_gen = theta_eq[gen_bus_idx]
        Vq_eq = V_gen * torch.cos(delta_full - theta_gen)
        Vd_eq = V_gen * torch.sin(delta_full - theta_gen)
        Id_raw = (Eqp - Vq_eq) / self.Xd_prime
        Iq_raw = (Vd_eq - Edp_cur) / self.Xq_prime
        i_sg_raw = torch.sqrt(torch.clamp(Id_raw.square() + Iq_raw.square(), min=1e-12))
        self._refresh_generator_capability(i_sg_raw)
        Id_eq, Iq_eq = self._apply_generator_current_capability(Id_raw, Iq_raw)
        Pe_eq = Vd_eq * Id_eq + Vq_eq * Iq_eq

        Pm_new = Pe_eq.detach()
        Pvalve_new = Pm_new.clone()
        Efd_new = (Eqp + (self.Xd - self.Xd_prime) * Id_eq).detach()
        Edp_final = Edp_cur.detach()

        # Update GFM inverter equilibrium from converged network
        V_inv_eq = V_eq[self.inv_bus_indices]
        theta_inv_eq = theta_eq[self.inv_bus_indices]
        Vc_inv = V_inv_eq.to(torch.complex64) * torch.exp(1j * theta_inv_eq.to(torch.complex64))

        # Recompute behind-impedance voltage with converged network
        Z_v = self.Z_v.to(dtype=torch.complex64)
        S_inv_eq = self.P_set.to(torch.complex64) + 1j * (
            torch.zeros_like(self.P_set).to(torch.complex64)  # Q at equilibrium from power flow
        )
        # Use actual power flow to find Q_inv_eq
        theta_v_cur = inv['theta_v']
        E_inv_cur = E_inv_eq
        V_inv_c = E_inv_cur.to(torch.complex64) * torch.exp(1j * theta_v_cur.to(torch.complex64))
        I_inv_c = (V_inv_c - Vc_inv) / Z_v
        S_meas = Vc_inv * I_inv_c.conj()
        P_e_eq = S_meas.real.float()
        Q_e_eq = S_meas.imag.float()

        n = self.n_gen
        n_inv = self.n_inv
        gen_state_end = (n - 1) + 6 * n

        new_gp = gp.clone()
        edp_start = (n - 1) + 2 * n
        efd_start = (n - 1) + 3 * n
        pm_start = (n - 1) + 4 * n
        pv_start = (n - 1) + 5 * n
        new_gp[edp_start:edp_start + n] = Edp_final
        new_gp[efd_start:efd_start + n] = Efd_new
        new_gp[pm_start:pm_start + n] = Pm_new
        new_gp[pv_start:pv_start + n] = Pvalve_new

        # Update GFM inverter states in goal point
        for j in range(n_inv):
            inv_start = gen_state_end + j * self.GFM_STATE_DIM
            # theta_v stays (already computed from behind-impedance)
            new_gp[inv_start + self.IDX_V_MEAS] = V_inv_eq[j]
            new_gp[inv_start + self.IDX_P_FILT] = P_e_eq[j]
            new_gp[inv_start + self.IDX_Q_FILT] = Q_e_eq[j]

        self._goal_point = new_gp.unsqueeze(0)

        # Update setpoints with converged values
        self.P_set = P_e_eq.clone()
        self.V_ref_0 = V_inv_eq.clone()

        self.Vref = (V_gen + Efd_new / self.Ka.to(V_gen.dtype)).to(Efd_new.dtype)
        self.Pref0 = Pm_new.clone()
        i_sg_eq = torch.sqrt(torch.clamp(Id_raw.square() + Iq_raw.square(), min=1e-12))
        self._refresh_generator_capability(i_sg_eq)
        self._last_V = V_eq.detach().clone().to(torch.float32)
        self._last_theta = theta_eq.detach().clone().to(torch.float32)

        self._setup_state_limits(Efd_new, Pm_new)

        self.newton_iterations = saved_iters
        self.newton_reuse_jacobian = saved_reuse
        self.newton_warn_on_nonconvergence = saved_warn

    def _rebuild_equilibrium_from_current_load(self):
        """Recompute goal-point equilibrium for current PL_base/QL_base."""
        (
            V_eq, theta_eq, _, _,
            _, _, _, _,
            delta_eq, Eqp_eq, Edp_eq, Efd_eq, Pm_eq, Pvalve_eq,
            inv_state_eq
        ) = self._solve_equilibrium()

        self._u_eq = torch.zeros(
            1, self.n_inv, dtype=torch.float32, device=self.Vset.device
        )
        self._goal_point = self._pack_state(
            delta_rel=delta_eq[1:] - delta_eq[0],
            omega=torch.ones(self.n_gen, dtype=torch.float32, device=self.Vset.device),
            Eqp=Eqp_eq,
            Edp=Edp_eq,
            Efd=Efd_eq,
            Pm=Pm_eq,
            Pvalve=Pvalve_eq,
            inv_states=inv_state_eq,
        ).unsqueeze(0)
        self._setup_state_limits(Efd_eq, Pm_eq)
        self._repair_equilibrium()

    def solve_operating_point(
        self,
        load_scale: float = 1.0,
        load_profile: Optional[Tensor] = None,
        use_cache: bool = True,
    ) -> Dict[str, object]:
        """Solve and return operating-point data for a load condition."""
        device = self.PL_base.device

        if load_profile is None:
            profile = torch.full(
                (self.n_bus,), float(load_scale), dtype=torch.float32, device=device
            )
        else:
            profile = torch.as_tensor(
                load_profile, dtype=torch.float32, device=device
            ).view(-1)
            if profile.numel() != self.n_bus:
                raise ValueError(
                    f"load_profile must have length {self.n_bus}, got {profile.numel()}"
                )

        key = tuple(torch.round(profile.detach().cpu() * 1e6).to(torch.int64).tolist())
        if not hasattr(self, "_equilibrium_cache"):
            self._equilibrium_cache = {}

        def _clone_payload(payload: Dict[str, object], tgt_device: torch.device) -> Dict[str, object]:
            out: Dict[str, object] = {}
            for k, v in payload.items():
                if isinstance(v, torch.Tensor):
                    out[k] = v.to(device=tgt_device).clone()
                else:
                    out[k] = v
            return out

        if use_cache and key in self._equilibrium_cache:
            return _clone_payload(self._equilibrium_cache[key], device)

        cache_backup = self._equilibrium_cache
        self._equilibrium_cache = {}
        try:
            worker = copy.deepcopy(self).to(device)
        finally:
            self._equilibrium_cache = cache_backup
        worker._equilibrium_cache = {}
        worker.PL_base = worker.PL_base_nominal.to(device=device, dtype=torch.float32) * profile
        worker.QL_base = worker.QL_base_nominal.to(device=device, dtype=torch.float32) * profile

        saved_rebuild_iters = int(getattr(worker, "newton_iterations", 5))
        saved_rebuild_reuse = bool(getattr(worker, "newton_reuse_jacobian", True))
        worker.newton_iterations = 12
        worker.newton_reuse_jacobian = False
        worker._rebuild_equilibrium_from_current_load()
        worker.newton_iterations = saved_rebuild_iters
        worker.newton_reuse_jacobian = saved_rebuild_reuse

        x_star_nom = worker.goal_point.squeeze(0).detach().clone()
        delta_rel, omega_nom, Eqp, Edp, Efd, Pm_nom, Pvalve_nom, inv_states = worker._unpack_state(x_star_nom)
        inv = worker._unpack_inverter_states(inv_states)
        d0 = worker.delta0_goal.view(1).to(device=x_star_nom.device)
        delta = torch.cat([d0, d0 + delta_rel], dim=0)

        V_star = getattr(worker, "_last_V", worker.Vset).detach().clone()
        theta_star = getattr(worker, "_last_theta", torch.zeros_like(worker.Vset)).detach().clone()

        w = worker.PL_base_nominal.abs().to(device=device)
        denom = float(w.sum().clamp(min=1e-8).item())
        load_scale_equiv = float((profile * w).sum().item() / denom)

        result: Dict[str, object] = {
            "load_profile": profile.detach().clone(),
            "load_scale_equiv": load_scale_equiv,
            "x_star": x_star_nom,
            "V_star": V_star,
            "theta_star": theta_star,
            "omega_star": omega_nom.detach().clone(),
            "delta_star": delta.detach().clone(),
            "Eqp_star": Eqp.detach().clone(),
            "Edp_star": Edp.detach().clone(),
            "Pm_star": Pm_nom.detach().clone(),
            "Pvalve_star": Pvalve_nom.detach().clone(),
            "theta_v_star": inv["theta_v"].detach().clone(),
            "omega_v_star": inv["omega_v"].detach().clone(),
            "M_inv_star": inv["M_inv"].detach().clone(),
            "V_meas_star": inv["V_meas"].detach().clone(),
            "P_filt_star": inv["P_filt"].detach().clone(),
            "Q_filt_star": inv["Q_filt"].detach().clone(),
            "P_load": worker.PL_base.detach().clone(),
            "Q_load": worker.QL_base.detach().clone(),
            "V_ref": worker.Vset.detach().clone(),
        }

        if use_cache:
            self._equilibrium_cache[key] = _clone_payload(result, torch.device("cpu"))
        return result

    def solve_operating_points_batch(
        self,
        load_profiles: Tensor,
        use_cache: bool = True,
    ) -> List[Dict[str, object]]:
        """Solve one operating point per batch element."""
        device = self.PL_base.device
        profiles = torch.as_tensor(load_profiles, dtype=torch.float32, device=device).view(-1, self.n_bus)
        B = int(profiles.shape[0])

        key_to_indices: Dict[Tuple[int, ...], List[int]] = {}
        key_to_profile: Dict[Tuple[int, ...], Tensor] = {}
        for b in range(B):
            p = profiles[b]
            key = tuple(torch.round(p.detach().cpu() * 1e6).to(torch.int64).tolist())
            if key not in key_to_indices:
                key_to_indices[key] = []
                key_to_profile[key] = p.detach().clone()
            key_to_indices[key].append(b)

        key_to_result: Dict[Tuple[int, ...], Dict[str, object]] = {}
        for key, p in key_to_profile.items():
            key_to_result[key] = self.solve_operating_point(
                load_profile=p,
                use_cache=use_cache,
            )

        outputs: List[Dict[str, object]] = [None] * B
        for key, idxs in key_to_indices.items():
            payload = key_to_result[key]
            for b in idxs:
                out_b: Dict[str, object] = {}
                for k, v in payload.items():
                    if isinstance(v, torch.Tensor):
                        out_b[k] = v.to(device=device).clone()
                    else:
                        out_b[k] = v
                outputs[b] = out_b

        return outputs

    # ---- Network builder (identical to GFL6) ----

    def _build_reduced_network_and_loads(self, load_equiv: str = "ward_shunt"):
        """Build reduced 27-bus network (10 gen + 17 load-only buses).

        Identical to GFL6 — the network model is independent of inverter type.
        """
        base_MVA = 100.0
        nbus = 39

        branches = [
            (1, 2, 0.0035, 0.0411, 0.6987, 0.0), (1, 39, 0.001, 0.025, 0.75, 0.0),
            (2, 3, 0.0013, 0.0151, 0.2572, 0.0), (2, 25, 0.007, 0.0086, 0.146, 0.0),
            (3, 4, 0.0013, 0.0213, 0.2214, 0.0), (3, 18, 0.0011, 0.0133, 0.2138, 0.0),
            (4, 5, 0.0008, 0.0128, 0.1342, 0.0), (4, 14, 0.0008, 0.0129, 0.1382, 0.0),
            (5, 6, 0.0002, 0.0026, 0.0434, 0.0), (5, 8, 0.0008, 0.0112, 0.1476, 0.0),
            (6, 7, 0.0006, 0.0092, 0.113, 0.0), (6, 11, 0.0007, 0.0082, 0.1389, 0.0),
            (7, 8, 0.0004, 0.0046, 0.078, 0.0), (8, 9, 0.0023, 0.0363, 0.3804, 0.0),
            (9, 39, 0.001, 0.025, 1.2, 0.0), (10, 11, 0.0004, 0.0043, 0.0729, 0.0),
            (10, 13, 0.0004, 0.0043, 0.0729, 0.0), (13, 14, 0.0009, 0.0101, 0.1723, 0.0),
            (14, 15, 0.0018, 0.0217, 0.366, 0.0), (15, 16, 0.0009, 0.0094, 0.171, 0.0),
            (16, 17, 0.0007, 0.0089, 0.1342, 0.0), (16, 19, 0.0016, 0.0195, 0.304, 0.0),
            (16, 21, 0.0008, 0.0135, 0.2548, 0.0), (16, 24, 0.0003, 0.0059, 0.068, 0.0),
            (17, 18, 0.0007, 0.0082, 0.1319, 0.0), (17, 27, 0.0013, 0.0173, 0.3216, 0.0),
            (21, 22, 0.0008, 0.014, 0.2565, 0.0), (22, 23, 0.0006, 0.0096, 0.1846, 0.0),
            (23, 24, 0.0022, 0.035, 0.361, 0.0), (25, 26, 0.0032, 0.0323, 0.513, 0.0),
            (26, 27, 0.0014, 0.0147, 0.2396, 0.0), (26, 28, 0.0043, 0.0474, 0.7802, 0.0),
            (26, 29, 0.0057, 0.0625, 1.029, 0.0), (28, 29, 0.0014, 0.0151, 0.249, 0.0),
            (12, 11, 0.0016, 0.0435, 0.0, 1.006), (12, 13, 0.0016, 0.0435, 0.0, 1.006),
            (6, 31, 0.0, 0.025, 0.0, 1.07), (10, 32, 0.0, 0.02, 0.0, 1.07),
            (19, 33, 0.0007, 0.0142, 0.0, 1.07), (20, 34, 0.0009, 0.018, 0.0, 1.009),
            (22, 35, 0.0, 0.0143, 0.0, 1.025), (23, 36, 0.0005, 0.0272, 0.0, 1.0),
            (25, 37, 0.0006, 0.0232, 0.0, 1.025), (2, 30, 0.0, 0.0181, 0.0, 1.025),
            (29, 38, 0.0008, 0.0156, 0.0, 1.025), (19, 20, 0.0007, 0.0138, 0.0, 1.06),
        ]

        Y = np.zeros((nbus, nbus), dtype=complex)
        for f, t, r, x, b, tap in branches:
            f -= 1; t -= 1
            y = 0.0 if (r == 0.0 and x == 0.0) else 1.0 / complex(r, x)
            bsh = 1j * (b / 2.0)
            a = 1.0 if tap == 0.0 else float(tap)
            Y[f, f] += (y + bsh) / (a * a)
            Y[t, t] += y + bsh
            Y[f, t] -= y / a
            Y[t, f] -= y / a

        gen_bus_order = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        load_bus_order = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29]
        bus_order_27 = gen_bus_order + load_bus_order
        keep = [b - 1 for b in bus_order_27]
        elim = sorted([i for i in range(nbus) if i not in keep])

        Yaa, Ybb = Y[np.ix_(keep, keep)], Y[np.ix_(elim, elim)]
        Yab, Yba = Y[np.ix_(keep, elim)], Y[np.ix_(elim, keep)]

        P_MW = {1: 97.6, 3: 322.0, 4: 500.0, 7: 233.8, 8: 522.0, 9: 6.5, 12: 8.53,
                15: 320.0, 16: 329.0, 18: 158.0, 20: 680.0, 21: 274.0, 23: 247.5,
                24: 308.6, 25: 224.0, 26: 139.0, 27: 281.0, 28: 206.0, 29: 283.5,
                31: 9.2, 39: 1104.0}
        Q_MVAr = {1: 44.2, 3: 2.4, 4: 184.0, 7: 84.0, 8: 176.6, 9: -66.6, 12: 88.0,
                  15: 153.0, 16: 32.3, 18: 30.0, 20: 103.0, 21: 115.0, 23: 84.6,
                  24: -92.2, 25: 47.2, 26: 17.0, 27: 75.5, 28: 27.6, 29: 26.9,
                  31: 4.6, 39: 250.0}
        P, Q = np.zeros(nbus), np.zeros(nbus)
        for k, v in P_MW.items(): P[k - 1] = v / base_MVA
        for k, v in Q_MVAr.items(): Q[k - 1] = v / base_MVA

        Vset_gen = {30: 1.0499, 31: 0.9820, 32: 0.9841, 33: 0.9972, 34: 1.0123,
                    35: 1.0494, 36: 1.0636, 37: 1.0275, 38: 1.0265, 39: 1.0300}

        S_dir = P[keep] + 1j * Q[keep]

        load_equiv = str(load_equiv).lower().strip()

        def _estimate_Vb0_constant_power(
            Ybb_, Yba_, Va_, S_elim_, max_iter=50, damp=0.6, tol=1e-10,
        ):
            Vb = -np.linalg.solve(Ybb_, Yba_ @ Va_)
            Vb = np.where(np.abs(Vb) < 0.2, 1.0 + 0j, Vb)
            for _ in range(max_iter):
                Vb_safe = np.where(np.abs(Vb) < 1e-4, 1e-4 + 0j, Vb)
                I_load = np.conj(S_elim_) / np.conj(Vb_safe)
                Vb_new = -np.linalg.solve(Ybb_, (Yba_ @ Va_) + I_load)
                Vb_next = damp * Vb_new + (1.0 - damp) * Vb
                if np.max(np.abs(Vb_next - Vb)) < tol:
                    Vb = Vb_next
                    break
                Vb = Vb_next
            return Vb

        Va0_gen = np.array([Vset_gen[b] for b in gen_bus_order], dtype=complex)
        Va0_load = np.ones(len(load_bus_order), dtype=complex)
        Va0 = np.concatenate([Va0_gen, Va0_load])

        S_elim = P[elim] + 1j * Q[elim]
        Vb0 = _estimate_Vb0_constant_power(Ybb, Yba, Va0, S_elim)

        V_full_est = np.ones(nbus, dtype=complex)
        for i, bus_0idx in enumerate(keep):
            V_full_est[bus_0idx] = Va0[i]
        for i, bus_0idx in enumerate(elim):
            V_full_est[bus_0idx] = Vb0[i]

        load_bus_0idx = [b - 1 for b in load_bus_order]
        for _iter in range(20):
            for k, bus_0idx in enumerate(load_bus_0idx):
                S_bus = P[bus_0idx] + 1j * Q[bus_0idx]
                I_inj = np.conj(S_bus) / np.conj(V_full_est[bus_0idx]) if abs(V_full_est[bus_0idx]) > 1e-6 else 0.0
                rhs = -I_inj
                for j in range(nbus):
                    if j != bus_0idx:
                        rhs -= Y[bus_0idx, j] * V_full_est[j]
                Y_diag = Y[bus_0idx, bus_0idx]
                if abs(Y_diag) > 1e-10:
                    V_full_est[bus_0idx] = rhs / Y_diag

        Vset = np.abs(np.array([V_full_est[b] for b in keep]))

        if load_equiv in ("current_fold", "legacy", "old"):
            Yred = Yaa - Yab @ np.linalg.solve(Ybb, Yba)
            I_b = P[elim] - 1j * Q[elim]
            I_eq = -Yab @ np.linalg.solve(Ybb, I_b)
            S_eq = Vset.astype(complex) * np.conj(I_eq)
            S_tot = S_eq + S_dir
            PL, QL = S_tot.real, S_tot.imag

        elif load_equiv in ("ward_shunt", "shunt", "ward"):
            Va0_c = Vset.astype(complex)
            Vb0_ws = _estimate_Vb0_constant_power(Ybb, Yba, Va0_c, S_elim)
            Vb_mag2 = np.maximum(np.abs(Vb0_ws) ** 2, 1e-6)
            Yload_elim = np.conj(S_elim) / Vb_mag2
            Ybb_eff = Ybb + np.diag(Yload_elim)
            Yred = Yaa - Yab @ np.linalg.solve(Ybb_eff, Yba)
            PL, QL = S_dir.real, S_dir.imag

        else:
            raise ValueError(
                f"Unknown load_equiv='{load_equiv}'. "
                "Use 'ward_shunt' (recommended) or 'current_fold' (legacy)."
            )

        gen_P_MW = {30: 250.0, 31: 677.871, 32: 650.0, 33: 632.0, 34: 508.0,
                    35: 650.0, 36: 560.0, 37: 540.0, 38: 830.0, 39: 1000.0}
        Pg_pu = np.zeros(10)
        for idx, bus in enumerate(gen_bus_order):
            if bus in gen_P_MW:
                Pg_pu[idx] = gen_P_MW[bus] / base_MVA

        return Yred, Yred.real, Yred.imag, PL, QL, Pg_pu, Vset

    # ---- Convergence monitoring ----

    def get_newton_convergence_info(self) -> dict:
        if self._last_newton_residuals is None:
            return {'residuals': None, 'iterations': None, 'converged': None,
                    'max_residual': None, 'n_not_converged': None}
        residuals = self._last_newton_residuals
        converged = residuals < self.newton_tol
        return {
            'residuals': residuals,
            'iterations': self._last_newton_iterations,
            'converged': converged,
            'max_residual': residuals.max().item(),
            'n_not_converged': (~converged).sum().item(),
        }

    def check_newton_convergence(self, raise_on_failure: bool = False) -> bool:
        info = self.get_newton_convergence_info()
        if info['residuals'] is None:
            return True
        all_converged = info['n_not_converged'] == 0
        if not all_converged and raise_on_failure:
            raise RuntimeError(
                f"Newton solver (GFM) did not converge for {info['n_not_converged']} batch elements. "
                f"Max residual: {info['max_residual']:.2e}, tolerance: {self.newton_tol:.2e}."
            )
        return all_converged

    def auto_tune_newton_iterations(self, x_sample: torch.Tensor, target_tol: float = 1e-6,
                                     max_iterations: int = 20, verbose: bool = True) -> int:
        if x_sample.dim() == 1:
            x_sample = x_sample.unsqueeze(0)
        original_iters = self.newton_iterations
        original_warn = self.newton_warn_on_nonconvergence
        self.newton_warn_on_nonconvergence = False
        try:
            for n_iter in range(1, max_iterations + 1):
                self.newton_iterations = n_iter
                with torch.no_grad():
                    _, _ = self.control_affine_dynamics(x_sample, params=None)
                info = self.get_newton_convergence_info()
                residual = info['max_residual']
                if verbose:
                    print(f"  iter={n_iter:2d}: residual={residual:.2e}")
                if residual < target_tol:
                    if verbose:
                        print(f"Converged at {n_iter} iterations (residual={residual:.2e} < tol={target_tol:.2e})")
                    return n_iter
            if verbose:
                print(f"Did not reach target tolerance after {max_iterations} iterations "
                      f"(residual={residual:.2e} > tol={target_tol:.2e})")
            return max_iterations
        finally:
            self.newton_iterations = original_iters
            self.newton_warn_on_nonconvergence = original_warn

    # ---- Device management ----

    def to(self, device):
        """Move tensors to device."""
        device = torch.device(device) if isinstance(device, str) else device

        tensor_attrs = [
            'Y', 'G', 'B', 'PL_base', 'QL_base', 'PL_base_nominal', 'QL_base_nominal',
            'Pg_target_total', 'Vset',
            'H_base', 'D_base', 'Xd', 'Xd_prime', 'Td0_prime', 'Xq', 'Xq_prime', 'Tq0_prime',
            'Ka', 'Ta', 'R', 'Tg', 'Tt',
            'pv_ratio', 'sg_ratio', 'H', 'D',
            'sP', 'sQ',
            '_x_lo', '_x_hi',
            'Vr_min', 'Vr_max', 'Efd_min', 'Efd_max',
            'max_dEfd_up', 'max_dEfd_dn',
            'Pref_min', 'Pref_max', 'Pvalve_min', 'Pvalve_max',
            'Pm_min', 'Pm_max',
            'max_dPvalve_up', 'max_dPvalve_dn', 'max_dPm_up', 'max_dPm_dn',
            '_goal_point', '_u_eq',
            'Vref', 'Pref0',
            '_last_V', '_last_theta',
            'delta0_goal',
            'gen_bus_idx',
            'inv_bus_indices',
            # GFM-specific
            'Z_v', 'M_0', 'k_m', 'tau_m', 'M_max', 'D_v', 'K_QV',
            'T_v', 'T_p', 'T_q',
            'E_0', 'P_set', 'V_ref_0',
            'sg_Imax',
        ]

        for attr_name in tensor_attrs:
            if hasattr(self, attr_name):
                attr = getattr(self, attr_name)
                if attr is not None and isinstance(attr, torch.Tensor):
                    setattr(self, attr_name, attr.to(device))

        return self
