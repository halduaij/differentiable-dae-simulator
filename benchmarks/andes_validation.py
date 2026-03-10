#!/usr/bin/env python3
"""Cross-validate PyTorch IEEE39 DAE model against ANDES.

Generator-only comparison (no inverters):
  - Load IEEE 39-bus in ANDES with GENROU+IEEEX1+TGOV1N
  - Apply 50% load step at Bus 18 at t=0.1s
  - Run TDS, extract generator frequencies
  - Run PyTorch model (open-loop, no inverter control)
  - Compare trajectories

Incremental mode (--incremental):
  Progressively matches ANDES parameters to PyTorch, running a comparison
  after each step. Steps are cumulative:
    0. Baseline (original ANDES — PSS on, D=0, 6th-order, const-P, Tt=2.1s)
    1. Disable PSS (PyTorch has no PSS)
    2. Add passive damping D (PyTorch uses D * omega_s in swing equation)
    3. Freeze subtransient dynamics (PyTorch uses 4th-order model)
    4. ZIP load model 80/10/10 (match PyTorch load model)
    5. Match governor turbine time constant T3=0.5s

Usage:
    python tools/andes_validation.py --device cpu
    python tools/andes_validation.py --device cpu --incremental --tf 10
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
CODE_DIR = THIS_DIR.parent
sys.path.insert(0, str(CODE_DIR))

ASSET_DIR = (CODE_DIR / ".." / "assets").resolve()
SYS_MODEL = str(ASSET_DIR / "gfl6_27bus_system_pv40_hetero055 (1).pt")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OMEGA_S = 2.0 * np.pi * 60.0  # 376.99 rad/s
SB = 100.0  # System base MVA

# PyTorch generator parameters (read-only reference values)
H_BASE = np.array([15.15, 21.0, 17.9, 14.3, 13.0, 17.4, 13.2, 12.15, 17.25, 250.0])
D_BASE = np.array([3.46, 2.36, 3.46, 3.46, 3.46, 3.46, 3.46, 3.46, 3.644, 3.644])
SG_RATIO = np.array([0.4279, 0.5564, 0.6, 0.6, 0.3662, 0.3314, 0.4385, 0.3955, 0.4606, 0.3236])
PV_RATIO = 1.0 - SG_RATIO

# PyTorch generator electrical parameters (Pai 1989 / Athay et al.) — SYSTEM BASE (Sb=100)
XD_PY    = np.array([0.1000, 0.2950, 0.2495, 0.2620, 0.6700, 0.2540, 0.2950, 0.2900, 0.2106, 0.0200])
XD1_PY   = np.array([0.0697, 0.0310, 0.0531, 0.0436, 0.1320, 0.0500, 0.0490, 0.0570, 0.0570, 0.0060])
XQ_PY    = np.array([0.0690, 0.2820, 0.2370, 0.2580, 0.6200, 0.2410, 0.2920, 0.2800, 0.2050, 0.0190])
XQ1_PY   = XD1_PY * XQ_PY / XD_PY  # round-rotor approx: X'q = X'd * Xq/Xd
TD0_PY   = np.array([6.56, 10.2, 5.70, 5.69, 5.40, 7.30, 5.66, 6.70, 4.79, 7.00])
TQ0_PY   = 0.5 * TD0_PY  # round-rotor approx: T'q0 = 0.5 * T'd0

# PyTorch governor / exciter constants
KA_PY = 50.0      # AVR gain
TA_PY = 0.05      # AVR time constant [s]
R_PY = 0.05       # Droop coefficient (5%)
TG_PY = 0.05      # Governor valve time constant [s]
TT_PY = 0.50      # Governor turbine time constant [s]

# PyTorch GFL6 inverter constants
KP_PLL = 60.0 / OMEGA_S    # ≈ 0.1592 (PLL proportional gain)
KI_PLL = 900.0 / OMEGA_S   # ≈ 2.3873 (PLL integral gain)
T_F = 0.02      # Frequency measurement filter [s]
T_V = 0.02      # Voltage measurement filter [s]
T_PORD = 0.02   # Active power order filter [s]
T_QORD = 0.02   # Reactive power order / converter lag [s]
K_QV = 0.1      # Q-V droop gain

# 27-bus reduced-system bus order used by IEEE39ControlAffineDAE_GFL6.
REDUCED_GEN_BUSES = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
REDUCED_LOAD_BUSES = [3, 4, 7, 8, 12, 15, 16, 18, 20, 21, 23, 24, 25, 26, 27, 28, 29]
REDUCED_BUS_ORDER = REDUCED_GEN_BUSES + REDUCED_LOAD_BUSES

# Step definitions for incremental mode
STEP_CONFIGS = [
    ("baseline",        "Original ANDES (PSS on, D=0, 6th-order, const-P, Tt=2.1s)"),
    ("no_pss",          "+ Disable PSS"),
    ("add_D",           "+ Add passive damping D (from PyTorch model)"),
    ("no_subtransient", "+ Freeze subtransient (Td20,Tq20 -> 999)"),
    ("zip_load",        "+ ZIP load 80Z/10I/10P"),
    ("match_governor",  "+ Governor T3 -> 0.5s"),
]


def compute_D_andes(D_pytorch_effective: np.ndarray, Sn_andes: np.ndarray) -> np.ndarray:
    """Convert PyTorch effective D values to ANDES system-base convention.

    PyTorch swing eq:  2H * dw/dt = Pm - Pe - D_eff * ws * (w-1)   [system base]
    ANDES GENROU:      M  * dw/dt = Tm - Te - D.v * (w-1)          [D.v is system base after setup]

    Matching damping:  D.v = D_eff * ws  (no Sn/Sb — we set D.v directly)

    D_pytorch_effective already includes sg_ratio scaling (D_base * sg_ratio).
    Sn_andes is kept in signature for backward compat but unused.
    """
    return D_pytorch_effective * OMEGA_S


def get_overrides_for_step(step_idx: int, D_andes: np.ndarray) -> dict:
    """Build cumulative override dict for a given step index."""
    ov = {}
    if step_idx >= 1:
        ov["disable_pss"] = True
    if step_idx >= 2:
        ov["damping_D"] = D_andes
    if step_idx >= 3:
        ov["freeze_subtransient"] = True
    if step_idx >= 4:
        ov["zip_load"] = (0.8, 0.1, 0.1)  # p2z, p2i, p2p
    if step_idx >= 5:
        ov["governor_T3"] = 0.5
    return ov


def _map_physical_bus_to_pytorch_index(load_bus: int, n_bus: int) -> int:
    """Map IEEE physical bus number to reduced 27-bus PyTorch index."""
    load_bus = int(load_bus)
    if load_bus in REDUCED_BUS_ORDER:
        idx = REDUCED_BUS_ORDER.index(load_bus)
        if idx >= int(n_bus):
            raise ValueError(
                f"Physical bus {load_bus} maps to reduced index {idx}, "
                f"but PyTorch model has only {int(n_bus)} buses."
            )
        return idx
    raise ValueError(
        f"Physical bus {load_bus} is not retained in reduced order "
        f"{REDUCED_BUS_ORDER}."
    )


def _map_reduced_index_to_physical_bus(load_bus: int, n_bus: int) -> int:
    """Map reduced-model bus index to retained IEEE physical bus number."""
    load_bus = int(load_bus)
    n_max = min(int(n_bus), len(REDUCED_BUS_ORDER))
    if 0 <= load_bus < n_max:
        return int(REDUCED_BUS_ORDER[load_bus])
    raise ValueError(
        f"Reduced bus index {load_bus} out of range [0,{n_max - 1}] "
        f"for current model."
    )


def _resolve_disturbance_buses(load_bus: int, n_bus: int, load_bus_mode: str) -> tuple[int, int]:
    """Resolve disturbance bus to (physical_bus, reduced_index)."""
    mode = str(load_bus_mode).strip().lower()
    if mode == "physical":
        physical = int(load_bus)
        reduced = _map_physical_bus_to_pytorch_index(physical, n_bus)
        return physical, reduced
    if mode in {"reduced-index", "reduced_index", "reduced"}:
        reduced = int(load_bus)
        physical = _map_reduced_index_to_physical_bus(reduced, n_bus)
        return physical, reduced
    raise ValueError(
        f"Unsupported load-bus mode '{load_bus_mode}'. "
        "Use 'physical' or 'reduced-index'."
    )


# ---------------------------------------------------------------------------
# ANDES simulation
# ---------------------------------------------------------------------------

def run_andes_sim(
    load_bus: int = 18,
    load_mult_p: float = 1.5,
    load_mult_q: float = 1.5,
    load_t: float = 0.1,
    tf: float = 2.0,
    dt: float = 0.005,
    overrides: dict | None = None,
) -> dict:
    """Run ANDES IEEE39 TDS with a load step, return results dict.

    Parameters
    ----------
    overrides : dict, optional
        Parameter overrides to apply before simulation:
        - "disable_pss": bool — set IEEEST.u = 0
        - "damping_D": ndarray — GENROU.D values (machine base)
        - "freeze_subtransient": bool — set Td20, Tq20 = 999
        - "zip_load": (p2z, p2i, p2p) — ZIP load coefficients
        - "governor_T3": float — TGOV1N.T3 value
    """
    import andes

    if overrides is None:
        overrides = {}

    tag = ""
    if overrides:
        parts = []
        if overrides.get("disable_pss"):
            parts.append("PSS-off")
        if "damping_D" in overrides:
            parts.append("D-matched")
        if overrides.get("freeze_subtransient"):
            parts.append("no-subtrans")
        if "zip_load" in overrides:
            parts.append("ZIP")
        if "governor_T3" in overrides:
            parts.append(f"T3={overrides['governor_T3']}")
        tag = " [" + ", ".join(parts) + "]"

    print(f"\n[andes]{tag} Loading IEEE 39-bus full case...")
    ss = andes.load(
        andes.get_case("ieee39/ieee39_full.xlsx"),
        setup=False,
        default_config=True,
        no_output=True,
    )

    ss.setup()

    # --- Apply overrides ---

    # Step 1: Disable PSS
    if overrides.get("disable_pss"):
        ss.IEEEST.u.v[:] = 0
        print(f"[andes]{tag} PSS disabled (IEEEST.u = 0)")

    # Step 2: Set damping D
    if "damping_D" in overrides:
        D_vals = overrides["damping_D"]
        ss.GENROU.D.v[:] = D_vals
        print(f"[andes]{tag} GENROU.D set: {np.array2string(D_vals, precision=2)}")

    # Step 3: Freeze subtransient dynamics
    if overrides.get("freeze_subtransient"):
        ss.GENROU.Td20.v[:] = 999.0
        ss.GENROU.Tq20.v[:] = 999.0
        print(f"[andes]{tag} Subtransient frozen (Td20=Tq20=999)")

    # Step 4: ZIP load model
    if "zip_load" in overrides:
        p2z, p2i, p2p = overrides["zip_load"]
        ss.PQ.config.p2z = p2z
        ss.PQ.config.p2i = p2i
        ss.PQ.config.p2p = p2p
        ss.PQ.config.q2z = p2z
        ss.PQ.config.q2i = p2i
        ss.PQ.config.q2q = p2p
        print(f"[andes]{tag} ZIP load: p2z={p2z}, p2i={p2i}, p2p={p2p}")
    else:
        # Default: 100% constant power
        ss.PQ.config.p2p = 1
        ss.PQ.config.q2q = 1
        ss.PQ.config.p2z = 0
        ss.PQ.config.q2z = 0

    # Step 5: Governor turbine time constant
    if "governor_T3" in overrides:
        T3_val = overrides["governor_T3"]
        ss.TGOV1N.T3.v[:] = T3_val
        print(f"[andes]{tag} TGOV1N.T3 = {T3_val}")

    # --- Power flow ---
    print(f"[andes]{tag} Running power flow...")
    ss.PFlow.run()
    if not ss.PFlow.converged:
        raise RuntimeError("ANDES power flow did not converge")

    # Find PQ load at the target bus
    pq_df = ss.PQ.as_df()
    pq_at_bus = pq_df[pq_df["bus"] == load_bus]
    if pq_at_bus.empty:
        raise ValueError(f"No PQ load at bus {load_bus}")

    pq_idx = pq_at_bus["idx"].iloc[0]
    pq_int_idx = list(ss.PQ.idx.v).index(pq_idx)

    # Read current Ppf/Qpf after power flow
    ppf_orig = float(ss.PQ.Ppf.v[pq_int_idx])
    qpf_orig = float(ss.PQ.Qpf.v[pq_int_idx])
    ppf_new = ppf_orig * load_mult_p
    qpf_new = qpf_orig * load_mult_q

    print(f"[andes]{tag} Bus {load_bus}: Ppf {ppf_orig:.4f} -> {ppf_new:.4f}, Qpf {qpf_orig:.4f} -> {qpf_new:.4f}")

    # Read Sn for reference (useful for D conversion)
    Sn = np.array(ss.GENROU.Sn.v)

    # --- Phase 1: Steady state ---
    print(f"[andes]{tag} TDS phase 1 (0 -> {load_t}s)...")
    ss.TDS.config.tf = load_t
    ss.TDS.config.tstep = dt
    t0 = time.perf_counter()
    ss.TDS.run()

    # --- Phase 2: Load step ---
    print(f"[andes]{tag} Applying load step at t={load_t}s...")
    ss.PQ.Ppf.v[pq_int_idx] = ppf_new
    ss.PQ.Qpf.v[pq_int_idx] = qpf_new

    # For ZIP load, also scale constant-current and constant-impedance components
    if "zip_load" in overrides:
        _scale_zip_components(ss, pq_int_idx, load_mult_p, load_mult_q, tag)

    print(f"[andes]{tag} TDS phase 2 ({load_t} -> {tf}s)...")
    ss.TDS.config.tf = tf
    ss.TDS.run()
    elapsed = time.perf_counter() - t0
    print(f"[andes]{tag} TDS done in {elapsed:.1f}s")

    # --- Extract results ---
    t_arr = np.array(ss.dae.ts.t)
    omega_idx = ss.GENROU.omega.a
    omega_ts = np.array(ss.dae.ts.x[:, omega_idx])  # (T, 10)

    v_idx = ss.Bus.v.a
    v_ts = np.array(ss.dae.ts.y[:, v_idx])

    a_idx = ss.Bus.a.a
    a_ts = np.array(ss.dae.ts.y[:, a_idx])

    bus_ids = list(ss.Bus.idx.v)

    return {
        "t": t_arr,
        "omega": omega_ts,
        "v_bus": v_ts,
        "a_bus": a_ts,
        "bus_ids": bus_ids,
        "gen_buses": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        "Sn": Sn,
    }


def _capture_pq_component_bases(ss, pq_int_idx: int) -> dict[str, float]:
    """Capture baseline PQ components for one load row."""
    base = {
        "Ppf": float(ss.PQ.Ppf.v[pq_int_idx]),
        "Qpf": float(ss.PQ.Qpf.v[pq_int_idx]),
    }
    for attr in ("Ipeq", "Iqeq", "Req", "Xeq"):
        try:
            obj = getattr(ss.PQ, attr, None)
            if obj is not None and hasattr(obj, "v"):
                base[attr] = float(obj.v[pq_int_idx])
        except (AttributeError, IndexError, TypeError):
            continue
    return base


def _apply_pq_component_scale(
    ss,
    pq_int_idx: int,
    base: dict[str, float],
    mult_p: float,
    mult_q: float,
    tag: str,
    verbose: bool = True,
):
    """Apply active/reactive scaling to captured PQ components."""
    ss.PQ.Ppf.v[pq_int_idx] = float(base["Ppf"]) * float(mult_p)
    ss.PQ.Qpf.v[pq_int_idx] = float(base["Qpf"]) * float(mult_q)

    scaled_any = False
    for attr_p, attr_q in (("Ipeq", "Iqeq"), ("Req", "Xeq")):
        if attr_p in base:
            val_p = float(base[attr_p]) * float(mult_p)
            getattr(ss.PQ, attr_p).v[pq_int_idx] = val_p
            scaled_any = True
            if verbose:
                print(f"[andes]{tag}   {attr_p}: {float(base[attr_p]):.4f} -> {val_p:.4f}")
        if attr_q in base:
            val_q = float(base[attr_q]) * float(mult_q)
            getattr(ss.PQ, attr_q).v[pq_int_idx] = val_q
            scaled_any = True
            if verbose:
                print(f"[andes]{tag}   {attr_q}: {float(base[attr_q]):.4f} -> {val_q:.4f}")

    if (not scaled_any) and verbose:
        print(f"[andes]{tag}   [warn] Could not modify ZIP I/Z components — only Ppf scaled")
        print(f"[andes]{tag}   Effective load change may be smaller than intended")


def _scale_zip_components(ss, pq_int_idx: int, mult_p: float, mult_q: float, tag: str):
    """Backward-compatible ZIP scaler based on current component values."""
    base = _capture_pq_component_bases(ss, pq_int_idx)
    _apply_pq_component_scale(
        ss,
        pq_int_idx,
        base=base,
        mult_p=mult_p,
        mult_q=mult_q,
        tag=tag,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Full-match helpers: generator parameter matching + inverter addition
# ---------------------------------------------------------------------------

def _as_np_1d(value, *, name: str, expected_len: int | None = None) -> np.ndarray:
    """Convert tensors/lists to 1-D float numpy arrays."""
    if value is None:
        raise ValueError(f"Missing required PyTorch field: {name}")
    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if expected_len is not None and arr.size != expected_len:
        raise ValueError(f"{name} length mismatch: expected {expected_len}, got {arr.size}")
    return arr


def _extract_pytorch_operating_targets(sys_model) -> dict:
    """Extract equilibrium dispatch/shares from the loaded PyTorch system."""
    n_gen = int(getattr(sys_model, "n_gen"))
    n_inv = int(getattr(sys_model, "n_inv"))

    gen_bus_idx = _as_np_1d(getattr(sys_model, "gen_bus_idx", None), name="gen_bus_idx", expected_len=n_gen).astype(int)
    inv_bus_idx = _as_np_1d(getattr(sys_model, "inv_bus_indices", None), name="inv_bus_indices", expected_len=n_inv).astype(int)

    pref0 = _as_np_1d(getattr(sys_model, "Pref0", None), name="Pref0", expected_len=n_gen)
    p_ref_0 = _as_np_1d(getattr(sys_model, "P_ref_0", None), name="P_ref_0", expected_len=n_inv)
    q_ref_0 = _as_np_1d(getattr(sys_model, "Q_0", None), name="Q_0", expected_len=n_inv)
    H_eff = _as_np_1d(getattr(sys_model, "H", None), name="H", expected_len=n_gen)
    D_eff = _as_np_1d(getattr(sys_model, "D", None), name="D", expected_len=n_gen)

    # Map inverter setpoints to generator order through shared bus indices.
    inv_by_bus = {int(inv_bus_idx[j]): j for j in range(n_inv)}
    p_inv_gen = np.zeros(n_gen, dtype=float)
    q_inv_gen = np.zeros(n_gen, dtype=float)
    for gi, gbus in enumerate(gen_bus_idx):
        jj = inv_by_bus.get(int(gbus))
        if jj is not None:
            p_inv_gen[gi] = float(p_ref_0[jj])
            q_inv_gen[gi] = float(q_ref_0[jj])

    p_total = pref0 + p_inv_gen
    eps = 1e-9
    sg_share = np.where(np.abs(p_total) > eps, pref0 / p_total, 1.0)
    pv_share = np.where(np.abs(p_total) > eps, p_inv_gen / p_total, 0.0)
    sg_share = np.clip(sg_share, 0.0, 1.0)
    pv_share = np.clip(pv_share, 0.0, 1.0)
    share_sum = sg_share + pv_share
    nz = share_sum > eps
    sg_share[nz] /= share_sum[nz]
    pv_share[nz] /= share_sum[nz]

    print(f"[full-match] PyTorch Pref0 (SG): {np.array2string(pref0, precision=4)}")
    print(f"[full-match] PyTorch P_ref_0 (INV@gen): {np.array2string(p_inv_gen, precision=4)}")
    print(f"[full-match] PyTorch P_total target: {np.array2string(p_total, precision=4)}")
    print(f"[full-match] PyTorch SG share from equilibrium: {np.array2string(sg_share, precision=4)}")
    print(f"[full-match] PyTorch PV share from equilibrium: {np.array2string(pv_share, precision=4)}")

    return {
        "pref0": pref0,
        "p_inv_gen": p_inv_gen,
        "q_inv_gen": q_inv_gen,
        "p_total": p_total,
        "sg_share": sg_share,
        "pv_share": pv_share,
        "H_eff": H_eff,
        "D_eff": D_eff,
    }


def _extract_pytorch_reduced_load_targets(sys_model) -> tuple[dict[int, float], dict[int, float]]:
    """Extract reduced-network base PQ loads from PyTorch model definition."""
    n_bus = int(getattr(sys_model, "n_bus"))
    if n_bus > len(REDUCED_BUS_ORDER):
        raise ValueError(
            f"PyTorch reduced bus count {n_bus} exceeds known bus ordering length "
            f"{len(REDUCED_BUS_ORDER)}."
        )

    p_field = getattr(sys_model, "PL_base", None)
    if p_field is None:
        p_field = getattr(sys_model, "PL", None)
    q_field = getattr(sys_model, "QL_base", None)
    if q_field is None:
        q_field = getattr(sys_model, "QL", None)

    p_load = _as_np_1d(p_field, name="PL_base", expected_len=n_bus)
    q_load = _as_np_1d(q_field, name="QL_base", expected_len=n_bus)
    bus_ids = REDUCED_BUS_ORDER[:n_bus]

    p_by_bus = {int(bus_ids[i]): float(p_load[i]) for i in range(n_bus)}
    q_by_bus = {int(bus_ids[i]): float(q_load[i]) for i in range(n_bus)}

    load_buses = REDUCED_LOAD_BUSES[: max(0, min(len(REDUCED_LOAD_BUSES), n_bus - len(REDUCED_GEN_BUSES)))]
    p_vec = np.array([p_by_bus.get(int(b), 0.0) for b in load_buses], dtype=float)
    q_vec = np.array([q_by_bus.get(int(b), 0.0) for b in load_buses], dtype=float)
    print(f"[full-match] PyTorch reduced-load P targets (buses {load_buses}): {np.array2string(p_vec, precision=4)}")
    print(f"[full-match] PyTorch reduced-load Q targets (buses {load_buses}): {np.array2string(q_vec, precision=4)}")
    print(f"[full-match] PyTorch reduced-load totals: P={float(p_vec.sum()):.4f}, Q={float(q_vec.sum()):.4f}")

    return p_by_bus, q_by_bus


def _apply_andes_load_targets(
    ss,
    p_load_by_bus: dict[int, float],
    q_load_by_bus: dict[int, float],
):
    """Apply PyTorch reduced-network base loads to matching ANDES PQ buses."""
    updates = 0
    total_p_before = 0.0
    total_q_before = 0.0
    total_p_after = 0.0
    total_q_after = 0.0

    for i, bus in enumerate(list(ss.PQ.bus.v)):
        bus_i = int(bus)
        if bus_i in p_load_by_bus and bus_i in q_load_by_bus:
            p_old = float(ss.PQ.p0.v[i])
            q_old = float(ss.PQ.q0.v[i])
            p_new = float(p_load_by_bus[bus_i])
            q_new = float(q_load_by_bus[bus_i])
            ss.PQ.p0.v[i] = p_new
            ss.PQ.q0.v[i] = q_new
            updates += 1
            total_p_before += p_old
            total_q_before += q_old
            total_p_after += p_new
            total_q_after += q_new

    print(
        "[full-match] ANDES PQ load targets applied: "
        f"updated={updates}, P_sum {total_p_before:.4f}->{total_p_after:.4f}, "
        f"Q_sum {total_q_before:.4f}->{total_q_after:.4f}"
    )


def _apply_andes_dispatch_targets(ss, p_total_target: np.ndarray):
    """Set ANDES PV/Slack/StaticGen active dispatch to PyTorch equilibrium targets."""
    gen_buses = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    p_total_target = np.asarray(p_total_target, dtype=float).reshape(-1)
    if p_total_target.size != len(gen_buses):
        raise ValueError(
            f"p_total_target length mismatch: expected {len(gen_buses)}, got {p_total_target.size}"
        )
    p_by_bus = {bus: float(p_total_target[i]) for i, bus in enumerate(gen_buses)}

    pv_updates = 0
    for i, bus in enumerate(list(ss.PV.bus.v)):
        bus_i = int(bus)
        if bus_i in p_by_bus:
            ss.PV.p0.v[i] = p_by_bus[bus_i]
            pv_updates += 1

    slack_updates = 0
    for i, bus in enumerate(list(ss.Slack.bus.v)):
        bus_i = int(bus)
        if bus_i in p_by_bus:
            ss.Slack.p0.v[i] = p_by_bus[bus_i]
            slack_updates += 1

    static_updates = 0
    if hasattr(ss, "StaticGen") and hasattr(ss.StaticGen, "bus") and hasattr(ss.StaticGen, "p0"):
        for i, bus in enumerate(list(ss.StaticGen.bus.v)):
            bus_i = int(bus)
            if bus_i in p_by_bus:
                ss.StaticGen.p0.v[i] = p_by_bus[bus_i]
                static_updates += 1

    print(f"[full-match] ANDES dispatch targets applied: PV={pv_updates}, Slack={slack_updates}, StaticGen={static_updates}")
    print(f"[full-match] ANDES target P by bus 30-39: {np.array2string(p_total_target, precision=4)}")


def _extract_andes_total_gen_power_from_pflow(ss) -> np.ndarray:
    """Estimate total active power (SG + inverter) by generator bus from solved PFlow values."""
    gen_buses = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    bus_to_idx = {b: i for i, b in enumerate(gen_buses)}
    total = np.zeros(len(gen_buses), dtype=float)

    def _accumulate(model) -> None:
        if model is None or not hasattr(model, "bus"):
            return
        try:
            bus_vec = [int(b) for b in list(model.bus.v)]
        except Exception:
            return

        p_vec = None
        for attr in ("p", "p0"):
            if hasattr(model, attr):
                obj = getattr(model, attr)
                if hasattr(obj, "v"):
                    try:
                        arr = np.asarray(obj.v, dtype=float).reshape(-1)
                    except Exception:
                        continue
                    if arr.size > 0:
                        p_vec = arr
                        break
        if p_vec is None:
            return

        n = min(len(bus_vec), p_vec.size)
        for k in range(n):
            bus = bus_vec[k]
            ii = bus_to_idx.get(bus)
            if ii is None:
                continue
            val = float(p_vec[k])
            if np.isfinite(val):
                total[ii] += val

    _accumulate(getattr(ss, "PV", None))
    _accumulate(getattr(ss, "Slack", None))
    _accumulate(getattr(ss, "StaticGen", None))
    return total


def _tds_state_is_finite(ss) -> bool:
    """Return True when initialized DAE states are finite and non-empty."""
    try:
        x = np.asarray(ss.dae.x, dtype=float).reshape(-1)
        y = np.asarray(ss.dae.y, dtype=float).reshape(-1)
    except Exception:
        return False
    if x.size == 0 or y.size == 0:
        return False
    return bool(np.all(np.isfinite(x)) and np.all(np.isfinite(y)))


def _reconcile_dispatch_non_slack(
    ss,
    p_total_target: np.ndarray,
    max_iter: int = 6,
    tol: float = 0.05,
    alpha: float = 0.8,
    max_step: float = 0.6,
    slack_weight: float = 0.0,
) -> dict:
    """Iteratively adjust non-slack PV p0 to reduce dispatch mismatch vs target totals.

    Uses solved PFlow active powers for the reconciliation signal and leaves
    dynamic initialization to the caller (single TDS.init() after reconciliation).
    """
    gen_buses = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    slack_bus = 39
    slack_idx = gen_buses.index(slack_bus)
    target = np.asarray(p_total_target, dtype=float).reshape(-1)
    if target.size != len(gen_buses):
        raise ValueError(
            f"p_total_target length mismatch: expected {len(gen_buses)}, got {target.size}"
        )

    # Slack is bus 39 in IEEE39; keep it free for PF balance.
    # Optional slack_weight (>0) redistributes part of the slack residual
    # across non-slack PV updates, reducing concentration at G10.
    pv_bus_to_row = {int(b): i for i, b in enumerate(list(ss.PV.bus.v))}
    non_slack_buses = [b for b in gen_buses if b in pv_bus_to_row and b != slack_bus]
    if not non_slack_buses:
        print("[full-match] Dispatch reconciliation skipped (no non-slack PV buses found)")
        return {
            "enabled": False,
            "reason": "no_non_slack_pv",
        }

    iter_logs = []
    ss.PFlow.run()
    if not ss.PFlow.converged:
        return {
            "enabled": False,
            "reason": "initial_pflow_failed",
            "iter_logs": iter_logs,
            "non_slack_buses": non_slack_buses,
        }

    best_p0 = np.asarray(ss.PV.p0.v, dtype=float).copy()
    reason = "max_iter"
    pflow_ok = True
    for it in range(int(max_iter)):
        current = _extract_andes_total_gen_power_from_pflow(ss)
        if not np.all(np.isfinite(current)):
            reason = "non_finite_measurement"
            print(f"[full-match] [warn] Dispatch reconcile: non-finite totals at iter {it + 1}, reverting")
            ss.PV.p0.v[:] = best_p0
            ss.PFlow.run()
            pflow_ok = bool(ss.PFlow.converged)
            break
        resid = target - current

        resid_non = np.array([resid[gen_buses.index(b)] for b in non_slack_buses], dtype=float)
        if not np.all(np.isfinite(resid_non)):
            reason = "non_finite_residual"
            print(f"[full-match] [warn] Dispatch reconcile: non-finite residual at iter {it + 1}, reverting")
            ss.PV.p0.v[:] = best_p0
            break
        slack_resid = float(resid[slack_idx])
        max_abs_non = float(np.max(np.abs(resid_non))) if resid_non.size else 0.0
        mean_abs_non = float(np.mean(np.abs(resid_non))) if resid_non.size else 0.0
        iter_logs.append(
            {
                "iter": int(it + 1),
                "max_abs_non_slack": max_abs_non,
                "mean_abs_non_slack": mean_abs_non,
                "slack_residual": slack_resid,
                "residual_non_slack": resid_non.tolist(),
            }
        )
        print(
            f"[full-match] Dispatch reconcile iter {it + 1}: "
            f"mean|res|={mean_abs_non:.4f}, max|res|={max_abs_non:.4f} (non-slack), "
            f"slack_res={slack_resid:.4f}"
        )

        if max_abs_non <= float(tol):
            reason = "converged"
            break

        # Keep last stable p0 before the next perturbation.
        best_p0 = np.asarray(ss.PV.p0.v, dtype=float).copy()
        n_non = max(1, len(non_slack_buses))
        slack_term = float(slack_weight) * slack_resid / float(n_non)
        for b in non_slack_buses:
            gi = gen_buses.index(b)
            r = float(resid[gi])
            # Positive slack_resid means slack generation is below target.
            # Reduce non-slack setpoints to shift power pickup toward slack.
            r_eff = r - slack_term
            delta = float(np.clip(float(alpha) * r_eff, -float(max_step), float(max_step)))
            row = pv_bus_to_row[b]
            new_p0 = float(ss.PV.p0.v[row] + delta)
            # Keep physically meaningful positive active setpoint.
            ss.PV.p0.v[row] = max(0.0, new_p0)
        ss.PFlow.run()
        if not ss.PFlow.converged:
            reason = "pflow_failed"
            print(f"[full-match] [warn] Dispatch reconcile: PFlow failed at iter {it + 1}, reverting")
            ss.PV.p0.v[:] = best_p0
            ss.PFlow.run()
            pflow_ok = bool(ss.PFlow.converged)
            break

    if not pflow_ok:
        return {
            "enabled": False,
            "reason": "final_pflow_failed",
            "iter_logs": iter_logs,
            "non_slack_buses": non_slack_buses,
        }

    final_total = _extract_andes_total_gen_power_from_pflow(ss)
    if not np.all(np.isfinite(final_total)):
        print("[full-match] [warn] Dispatch reconcile: final totals non-finite, restoring last stable p0")
        ss.PV.p0.v[:] = best_p0
        ss.PFlow.run()
        if not ss.PFlow.converged:
            return {
                "enabled": False,
                "reason": "final_non_finite_and_restore_failed",
                "iter_logs": iter_logs,
                "non_slack_buses": non_slack_buses,
            }
        final_total = _extract_andes_total_gen_power_from_pflow(ss)
        if not np.all(np.isfinite(final_total)):
            return {
                "enabled": False,
                "reason": "final_non_finite",
                "iter_logs": iter_logs,
                "non_slack_buses": non_slack_buses,
            }
    final_resid = target - final_total

    summary = {
        "enabled": True,
        "reason": reason,
        "target_total": target.tolist(),
        "final_total": final_total.tolist(),
        "final_residual": final_resid.tolist(),
        "slack_residual_final": float(final_resid[slack_idx]),
        "slack_weight": float(slack_weight),
        "mean_abs_final": float(np.mean(np.abs(final_resid))),
        "max_abs_final": float(np.max(np.abs(final_resid))),
        "iter_logs": iter_logs,
        "non_slack_buses": non_slack_buses,
        "tol": float(tol),
        "alpha": float(alpha),
        "max_step": float(max_step),
        "measurement_source": "pflow",
    }
    print(
        f"[full-match] Dispatch reconcile final: mean|res|={summary['mean_abs_final']:.4f}, "
        f"max|res|={summary['max_abs_final']:.4f}"
    )
    return summary


def _match_generator_params(
    ss,
    with_inverters: bool = False,
    use_sexs: bool = False,
    sg_ratio_target: np.ndarray | None = None,
    H_eff_target: np.ndarray | None = None,
    D_eff_target: np.ndarray | None = None,
    gov_r_scale: float = 1.0,
):
    """Force ALL ANDES generator parameters to match PyTorch model.

    Must be called AFTER ss.setup().

    Key conversions (PyTorch → ANDES system-base .v values):
      M:  M.v = 2 * H_eff                    (H_eff = H_BASE * SG_RATIO)
      D:  D.v = D_eff * omega_s              (D_eff = D_BASE * SG_RATIO)
      R:  R.v = R_py / omega_s               (droop, omega_s factor in PyTorch gov)

    Parameters
    ----------
    use_sexs : bool
        If True, disable IEEEX1 and use SEXS exciter (simpler, closer to
        PyTorch's 1st-order model). SEXS must be added before setup.
    """
    Sn = np.array(ss.GENROU.Sn.v)
    n_gen = len(Sn)

    if sg_ratio_target is None:
        sg_ratio_target = SG_RATIO.copy()
    sg_ratio_target = np.asarray(sg_ratio_target, dtype=float).reshape(-1)
    if sg_ratio_target.size != n_gen:
        raise ValueError(f"sg_ratio_target length mismatch: expected {n_gen}, got {sg_ratio_target.size}")

    if H_eff_target is None:
        H_eff_target = H_BASE * sg_ratio_target
    H_eff = np.asarray(H_eff_target, dtype=float).reshape(-1)
    if H_eff.size != n_gen:
        raise ValueError(f"H_eff_target length mismatch: expected {n_gen}, got {H_eff.size}")

    if D_eff_target is None:
        D_eff_target = D_BASE * sg_ratio_target
    D_eff = np.asarray(D_eff_target, dtype=float).reshape(-1)
    if D_eff.size != n_gen:
        raise ValueError(f"D_eff_target length mismatch: expected {n_gen}, got {D_eff.size}")

    # 1. Disable PSS (PyTorch has none)
    ss.IEEEST.u.v[:] = 0
    print("[full-match] PSS disabled")

    # 2. Match inertia: M.v is system-base after setup (M.v = M_input * Sn/Sb).
    #    When we SET M.v directly, we set the system-base value used in the equation.
    #    PyTorch: 2H * dw/dt = ... where H = H_BASE * SG_RATIO (system base).
    #    ANDES:   M * dw/dt = ... where M = M.v (system base after setup).
    #    => M.v = 2 * H_eff  (no Sn/Sb conversion needed!)
    M_new = 2.0 * H_eff
    ss.GENROU.M.v[:] = M_new
    print(f"[full-match] GENROU.M set (sys base): {np.array2string(M_new, precision=3)}")

    # 3. Match damping: D.v is system-base after setup (same as M).
    #    PyTorch: damping term = D_eff * omega_s * (w-1)
    #    ANDES:   damping term = D.v * (w-1)
    #    => D.v = D_eff * omega_s  (no Sn/Sb conversion needed!)
    D_new = D_eff * OMEGA_S
    ss.GENROU.D.v[:] = D_new
    print(f"[full-match] GENROU.D set (sys base): {np.array2string(D_new, precision=3)}")

    # 4. Reactances, time constants, subtransient freezing — already set in raw data
    #    before setup (see run_andes_full_match). Just confirm.
    print(f"[full-match] GENROU Xd' (machine base): {np.array2string(np.array(ss.GENROU.xd1.v), precision=4)}")
    print(f"[full-match] GENROU Td0': {np.array2string(np.array(ss.GENROU.Td10.v), precision=2)}")

    # 5. Match exciter
    if use_sexs:
        # SEXS was already added before setup; IEEEX1 disabled before setup
        # SEXS params: K=Ka, TE=Ta, TATB=1 (bypass lead-lag), TB=small
        # These are set in _add_sexs_exciters() before setup
        print(f"[full-match] Exciter: SEXS (K={KA_PY}, TE={TA_PY}) — IEEEX1 disabled")
    else:
        # Match IEEEX1 main gain/time constant AND disable extra dynamics
        ss.IEEEX1.KA.v[:] = KA_PY
        ss.IEEEX1.TA.v[:] = TA_PY
        # Disable rate feedback (PyTorch has none)
        ss.IEEEX1.KF1.v[:] = 0.0
        # Disable lead-lag compensator (TC=0 → unity gain passthrough)
        ss.IEEEX1.TC.v[:] = 0.0
        # NOTE: Do NOT touch TE, KE, E1, E2 — they are jointly initialized by
        # ANDES to satisfy the exciter steady-state equation. Overwriting breaks init.
        print(f"[full-match] Exciter: IEEEX1 (KA={KA_PY}, TA={TA_PY}, KF1=0, no lead-lag)")

    # 6. Match governor: R, T1 (valve lag), T2/T3 (lead-lag → turbine)
    #    PyTorch: Pref = Pref0 + omega_s*(1-w)/R  [gain = omega_s / R]
    #    ANDES:   pd = -(w-wref)/R.v + pref        [gain = 1 / R.v]
    #    R.v is system-base after setup (ipower=True → R.v = R_input * Sb/Sn).
    #    When we SET R.v directly: R.v = R_py / omega_s (uniform, no Sn/Sb!)
    R_new = (R_PY / OMEGA_S) * float(gov_r_scale)
    ss.TGOV1N.R.v[:] = R_new
    # TGOV1N signal flow: pd → LAG(T1) → LEADLAG(T2/T3) → pout
    # PyTorch has two cascaded lags: 1/(1+sTg) * 1/(1+sTt)
    # Match: T1=Tg (valve LAG), T2=0 (no lead), T3=Tt (turbine LAG)
    ss.TGOV1N.T1.v[:] = TG_PY    # Valve/governor lag = 0.05s
    ss.TGOV1N.T2.v[:] = 0.0      # No lead in LEADLAG numerator
    ss.TGOV1N.T3.v[:] = TT_PY    # Turbine lag = 0.50s
    # Ensure turbine damping is zero
    ss.TGOV1N.Dt.v[:] = 0.0
    print(f"[full-match] Governor: R.v={R_new:.6f} (uniform, scale={float(gov_r_scale):.4f}), "
          f"T1={TG_PY}, T2=0, T3={TT_PY}, Dt=0")

    # 6b. Governor LIMITS: PyTorch clamps Pm at 1.5 * Pm_eq, Pref at 1.5 * Pref0
    #     ANDES TGOV1N VMAX has power=True → VMAX.v = VMAX_input * Sn/Sb (system base).
    #     Native VMAX.v = 1.2*Sn/Sb ≈ 12 pu — WAY too permissive vs PyTorch ~2.6 pu.
    #     Set VMAX.v = 1.5 * tm0 (on system base) to match PyTorch's 1.5x ratio.
    tm0 = np.array(ss.GENROU.tm0.v)  # equilibrium Pm after PFlow (system base)
    # NOTE: tm0 is only available AFTER PFlow, but _match_generator_params is called
    #       BEFORE PFlow for now. We'll set these after PFlow in the runner.
    #       For now, just note the issue.
    vmax_target = 1.5 * np.abs(tm0)   # may be zero before PFlow
    if np.all(tm0 > 0.01):
        ss.TGOV1N.VMAX.v[:] = vmax_target
        ss.TGOV1N.VMIN.v[:] = 0.0
        print(f"[full-match] Governor VMAX = 1.5*tm0: {np.array2string(vmax_target, precision=3)}")
    else:
        print(f"[full-match] Governor VMAX: NOT SET (tm0 not yet available, will set after PFlow)")

    # 7. ZIP load model 80/10/10 (matches PyTorch)
    ss.PQ.config.p2z = 0.8
    ss.PQ.config.p2i = 0.1
    ss.PQ.config.p2p = 0.1
    ss.PQ.config.q2z = 0.8
    ss.PQ.config.q2i = 0.1
    ss.PQ.config.q2q = 0.1
    print("[full-match] ZIP load: 80Z/10I/10P")

    # 8. If adding inverters, reduce GENROU power share via gammap
    if with_inverters:
        ss.GENROU.gammap.v[:] = sg_ratio_target
        ss.GENROU.gammaq.v[:] = sg_ratio_target
        print(f"[full-match] GENROU gammap/q set to target SG share: {np.array2string(sg_ratio_target, precision=4)}")


def _add_sexs_exciters(ss):
    """Replace IEEEX1 with SEXS exciters for all 10 generators.

    Must be called BEFORE ss.setup() (add models), but IEEEX1 disabling
    happens after setup.

    SEXS transfer function: (1+sTA)/(1+sTB) * K/(1+sTE)
    To match PyTorch's 1st-order: dEfd/dt = (Ka*(Vref-V) - Efd)/Ta
    we set TATB=1.0 and TB=very small (so lead-lag → unity) and K=Ka, TE=Ta.
    """
    gen_indices = list(ss.GENROU.idx.v)  # e.g. ['GENROU_1', ..., 'GENROU_10']

    for i, gen_idx in enumerate(gen_indices):
        ss.add("SEXS", {
            "idx": f"SEXS_{i+1}",
            "syn": gen_idx,  # must match GENROU idx strings
            "K": KA_PY,        # Gain = Ka = 50
            "TE": TA_PY,       # Time constant = Ta = 0.05s
            "TATB": 1.0,       # TA/TB = 1 → lead-lag unity gain
            "TB": 0.001,       # Very small → lead-lag is bypassed
            "EMIN": -99.0,     # Wide limits (no saturation)
            "EMAX": 99.0,
            "u": 1,
        })

    print(f"[full-match] Added 10 SEXS exciters (K={KA_PY}, TE={TA_PY}, lead-lag bypassed)")


def _add_inverters(
    ss,
    use_reeca: bool = False,
    pv_ratio_target: np.ndarray | None = None,
    linear_control: bool = False,
    linear_kpf: float = 0.0,
    linear_kvv: float = 0.0,
    enable_linear_qdroop: bool = False,
):
    """Add 10 GFL inverters (REGCP1 + PLL2, optionally + REECA1/REECA1E) to ANDES.

    Must be called BEFORE ss.setup().  Each inverter is co-located at
    a generator bus (30-39) and shares the existing StaticGen via
    gammap = PV_RATIO (inverter's share of total bus power).

    Parameters
    ----------
    use_reeca : bool
        If True, add REECA1 electrical controller (Q-V droop, P control).
        Default False because REECA1 has initialization issues that cause
        pre-disturbance drift.  Without REECA1, REGCP1 injects constant
        P/Q at PFlow dispatch, which is the closest match to PyTorch
        open-loop (kpf=0, kvv=0).
    linear_control : bool
        If True, add REECA1E and configure linear droop gains to mirror
        PyTorch's LinearDroopTransferController (active-frequency via Kf and
        reactive-voltage via Kqv).

    Models added:
      PLL2    — Phase-locked loop (2 states: PI_xi, am)
      REGCP1  — Converter model with PLL coupling (3 states)
      REECA1 / REECA1E  — Electrical controller [optional]
    """
    gen_buses = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    # StaticGen indices in IEEE39 (integers 1-10)
    sg_indices = list(range(1, 11))
    if pv_ratio_target is None:
        pv_ratio_target = PV_RATIO
    pv_ratio_target = np.asarray(pv_ratio_target, dtype=float).reshape(-1)
    if pv_ratio_target.size != len(gen_buses):
        raise ValueError(
            f"pv_ratio_target length mismatch: expected {len(gen_buses)}, got {pv_ratio_target.size}"
        )

    for i in range(10):
        bus = gen_buses[i]
        sg_idx = sg_indices[i]
        pv = float(pv_ratio_target[i])
        suffix = i + 1

        pll_idx = f"PLL_INV_{suffix}"
        reg_idx = f"REG_INV_{suffix}"

        # PLL2: tracks bus frequency
        ss.add("PLL2", {
            "idx": pll_idx,
            "bus": bus,
            "fn": 60.0,
            "Kp": KP_PLL,
            "Ki": KI_PLL,
            "u": 1,
        })

        # REGCP1: converter model (PLL-coupled variant of REGCA1)
        ss.add("REGCP1", {
            "idx": reg_idx,
            "bus": bus,
            "gen": sg_idx,
            "pll": pll_idx,
            "Sn": SB,           # 100 MVA system base
            "Tg": T_QORD,       # Converter lag = T_qord
            "gammap": pv,        # P fraction from StaticGen
            "gammaq": pv,        # Q fraction from StaticGen
            "Lvplsw": 0,        # Disable LVPL (PyTorch doesn't have it)
            "Rrpwr": 999.0,     # Disable ramp rate limit
            "Khv": 0.0,         # Disable HV reactive current
            "u": 1,
        })

        # REECA controller (optional)
        if linear_control:
            ree_idx = f"REE_INV_{suffix}"
            qdroop_on = bool(enable_linear_qdroop and (abs(float(linear_kvv)) > 1e-12))
            ss.add("REECA1E", {
                "idx": ree_idx,
                "reg": reg_idx,
                # Optional V-Q path: keep disabled by default for robust init.
                "PFFLAG": 0,
                "VFLAG": 1 if qdroop_on else 0,
                "QFLAG": 1 if qdroop_on else 0,
                "PFLAG": 0,     # Speed dependency disabled; REECA1E adds Kf term
                "PQFLAG": 1,    # P priority for current limiting
                "Trv": T_V,     # Voltage filter
                "dbd1": 0.0 if qdroop_on else -0.1,
                "dbd2": 0.0 if qdroop_on else 0.1,
                "Kqv": float(linear_kvv) if qdroop_on else 0.0,
                "Kqp": 0.0,
                "Kqi": 0.0,
                "Kvp": 0.0,
                "Kvi": 0.0,
                "Kf": float(linear_kpf),
                "Kdf": 0.0,     # No ROCOF channel for this comparison
                "Tpord": T_PORD,
                "Tp": T_F,
                # Keep current limits effectively inactive in comparison mode.
                "Imax": 999.0,
                "Iq1": 999.0,
                "Iq2": 999.0,
                "Iq3": 999.0,
                "Iq4": 999.0,
                "Ip1": 999.0,
                "Ip2": 999.0,
                "Ip3": 999.0,
                "Ip4": 999.0,
                "Iqh1": 999.0,
                "Iql1": -999.0,
                "QMax": 999.0,
                "QMin": -999.0,
                "VMAX": 999.0,
                "VMIN": -999.0,
                "dPmax": 999.0,
                "dPmin": -999.0,
                "PMAX": 999.0,
                "PMIN": 0.0,
                "u": 1,
            })
        elif use_reeca:
            ree_idx = f"REE_INV_{suffix}"
            ss.add("REECA1", {
                "idx": ree_idx,
                "reg": reg_idx,
                "PFFLAG": 0,    # No power factor control
                "VFLAG": 1,     # Voltage regulation active
                "QFLAG": 1,     # Q from voltage error
                "PFLAG": 0,     # No freq-P droop (open-loop: kpf=0)
                "PQFLAG": 1,    # P priority for current limiting
                "Trv": T_V,     # Voltage filter
                "dbd1": -0.1,   # Wide deadband to reduce init issues
                "dbd2": 0.1,
                "Kqv": K_QV,    # Q-V droop gain
                "Tpord": T_PORD,
                "Tp": T_F,      # Pe filter ≈ T_f
                "Imax": 1.5,    # Current limit
                "Iqh1": 999.0,
                "Iql1": -999.0,
                "QMax": 999.0,
                "QMin": -999.0,
                "VMAX": 999.0,
                "VMIN": -999.0,
                "dPmax": 999.0,
                "dPmin": -999.0,
                "PMAX": 999.0,
                "PMIN": 0.0,
                "u": 1,
            })

    if linear_control:
        stack = "REGCP1+REECA1E+PLL2"
    else:
        stack = "REGCP1+REECA1+PLL2" if use_reeca else "REGCP1+PLL2"
    print(f"[full-match] Added 10 GFL inverters ({stack}) at buses {gen_buses}")
    print(f"[full-match] PV share target used by REGCP1: {np.array2string(pv_ratio_target, precision=4)}")
    if linear_control:
        print(f"[full-match] Linear inverter control enabled: kpf={float(linear_kpf):.4f}, kvv={float(linear_kvv):.4f}")
        if abs(float(linear_kvv)) > 1e-12 and not bool(enable_linear_qdroop):
            print("[full-match] [warn] ANDES REECA Q-channel droop mapping is disabled for init robustness (frequency droop still active).")


# ---------------------------------------------------------------------------
# PyTorch simulation (open-loop, no inverter control)
# ---------------------------------------------------------------------------

def _load_pytorch_system_for_andes(
    device: str,
    disable_gov_rate_limit: bool = False,
    rebuild_equilibrium: bool = False,
    strict_checkpoint_init: bool = False,
) -> tuple[torch.nn.Module, str]:
    """Load system model with train_gfl6-consistent initialization."""
    from control_affine_system import repair_unpickled_module

    if str(device).lower().startswith("cuda") and torch.cuda.is_available():
        device_obj = torch.device(device)
    else:
        device_obj = torch.device("cpu")

    print("[pytorch] Loading system model...")
    obj = torch.load(SYS_MODEL, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "sys_model" in obj:
        sys_model = obj["sys_model"]
    else:
        sys_model = obj

    repair_unpickled_module(sys_model)
    sys_model = sys_model.to(device_obj)

    if strict_checkpoint_init:
        # Use checkpoint state exactly as saved (no runtime retuning/rebuild).
        print("[pytorch] Strict checkpoint init: preserving saved runtime/equilibrium state")
    else:
        # Match train_gfl6 runtime defaults for loaded systems.
        sys_model.inv_p_headroom_up_pct = 0.20
        sys_model.inv_p_headroom_dn_pct = 0.20
        sys_model.inv_v_headroom_up_pct = 0.10
        sys_model.inv_v_headroom_dn_pct = 0.10
        sys_model.inv_capability_mode = "p_priority"
        sys_model.inv_smax_mult = 1.25
        sys_model.inv_smax_min = 1e-3
        if hasattr(sys_model, "refresh_runtime_limits_from_goal"):
            sys_model.refresh_runtime_limits_from_goal()

        # Match train_gfl6 Newton configuration defaults.
        sys_model.newton_iterations = 5
        sys_model.newton_reuse_jacobian = True
        sys_model.newton_adaptive = True
        sys_model.newton_warn_threshold = 1e-3
        sys_model.newton_warn_on_nonconvergence = True
        sys_model.newton_damping = 1.0
        sys_model.newton_step_resid_factor = 10.0

        # Optional comparison mode: force-rebuild equilibrium from current load.
        # Disabled by default to preserve checkpoint operating-point behavior.
        if rebuild_equilibrium:
            if hasattr(sys_model, "_rebuild_equilibrium_from_current_load"):
                print("[pytorch] Rebuilding equilibrium from current load (explicit)")
                with torch.no_grad():
                    sys_model._rebuild_equilibrium_from_current_load()
            elif hasattr(sys_model, "_repair_equilibrium"):
                print("[pytorch] Running class equilibrium repair (explicit)")
                with torch.no_grad():
                    sys_model._repair_equilibrium()
            if hasattr(sys_model, "refresh_runtime_limits_from_goal"):
                sys_model.refresh_runtime_limits_from_goal()

    # Optional comparison mode: disable PyTorch governor valve slew limiter
    # so it aligns better with TGOV1N (which has valve position limits but
    # no explicit dPvalve/dt saturation parameter).
    if disable_gov_rate_limit and hasattr(sys_model, "enable_pvalve_rate"):
        sys_model.enable_pvalve_rate = False
        print("[pytorch] Governor rate limiter disabled for comparison")

    if hasattr(sys_model, "_equilibrium_cache"):
        sys_model._equilibrium_cache = {}

    # Same equilibrium sanity check style as train_gfl6.
    try:
        x_eq = sys_model.goal_point
        if x_eq.dim() == 1:
            x_eq = x_eq.unsqueeze(0)
        f_eq, g_eq = sys_model.control_affine_dynamics(x_eq, params=None)
        u_eq = sys_model.u_eq
        if u_eq.dim() == 1:
            u_eq = u_eq.unsqueeze(0)
        xdot_eq = f_eq + torch.bmm(g_eq, u_eq.unsqueeze(-1)).squeeze(-1)
        print(f"[pytorch] Equilibrium check ||xdot||={float(xdot_eq.norm().item()):.2e}")
    except Exception as exc:
        print(f"[pytorch] [warn] Equilibrium check skipped: {exc}")

    return sys_model.eval(), str(device_obj)


def run_pytorch_sim(
    load_bus: int = 18,
    load_bus_mode: str = "physical",
    load_mult_p: float = 1.5,
    load_mult_q: float = 1.5,
    load_t: float = 0.1,
    tf: float = 2.0,
    dt: float = 0.001,
    device: str = "cpu",
    disable_gov_rate_limit: bool = False,
    linear_kpf: float = 0.0,
    linear_kvv: float = 0.0,
    pytorch_rebuild_equilibrium: bool = False,
    pytorch_strict_checkpoint_init: bool = False,
) -> dict:
    """Run PyTorch IEEE39+GFL6 model open-loop, return results dict.

    `load_bus_mode` controls whether `load_bus` is interpreted as:
      - `physical`: IEEE physical bus number in the retained reduced set
      - `reduced-index`: reduced-model bus index [0..n_bus-1]
    """
    from train_gfl6 import evaluate_controller
    from baselines.transfer_to_gfl6 import LinearDroopTransferController

    sys_model, device = _load_pytorch_system_for_andes(
        device,
        disable_gov_rate_limit=disable_gov_rate_limit,
        rebuild_equilibrium=pytorch_rebuild_equilibrium,
        strict_checkpoint_init=pytorch_strict_checkpoint_init,
    )

    ctrl = LinearDroopTransferController(
        sys_model, kpf=float(linear_kpf), kvv=float(linear_kvv)
    ).to(device).eval()
    print(f"[pytorch] Linear droop controller: kpf={float(linear_kpf):.4f}, kvv={float(linear_kvv):.4f}")
    load_bus_physical, load_bus_reduced = _resolve_disturbance_buses(
        load_bus, int(sys_model.n_bus), load_bus_mode
    )
    print(
        f"[pytorch] Disturbance bus mode={str(load_bus_mode)}: "
        f"physical {load_bus_physical} -> reduced index {load_bus_reduced}"
    )

    n_steps = int(tf / dt)
    print(f"[pytorch] Running TDS (tf={tf}s, dt={dt}s, steps={n_steps})...")
    t0 = time.perf_counter()
    t_pts, x_pts = evaluate_controller(
        sys_model=sys_model,
        controller=ctrl,
        T=tf,
        load_change=load_mult_p,
        load_change_q=load_mult_q,
        load_bus=load_bus_reduced,
        load_t=load_t,
        n_sample=n_steps,
        dt=dt,
        device=device,
        control_limit_mode="capacity",
    )
    elapsed = time.perf_counter() - t0
    print(f"[pytorch] TDS done in {elapsed:.1f}s")

    x = x_pts[:, 0, :] if x_pts.ndim == 3 else x_pts
    n_gen = int(sys_model.n_gen)

    # omega: indices [n_gen-1 : n_gen-1+n_gen] = [9:19]
    omega = x[:, n_gen - 1 : n_gen - 1 + n_gen].detach().cpu().numpy()

    # Extract D and H for reference
    D_sys = sys_model.D[:n_gen].detach().cpu().numpy()
    H_sys = sys_model.H[:n_gen].detach().cpu().numpy()

    return {
        "t": t_pts.detach().cpu().numpy(),
        "omega": omega,
        "D_sys": D_sys,
        "H_sys": H_sys,
        "load_t": float(load_t),
        "load_bus_input": int(load_bus),
        "load_bus_mode": str(load_bus_mode),
        "load_bus_physical": int(load_bus_physical),
        "load_bus_reduced": int(load_bus_reduced),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

GEN_BUSES = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]


def plot_comparison(andes_res: dict, pytorch_res: dict, out_dir: Path,
                    suffix: str = "", title_extra: str = ""):
    """Plot ANDES vs PyTorch generator frequencies."""
    out_dir.mkdir(parents=True, exist_ok=True)

    t_a = andes_res["t"]
    omega_a = andes_res["omega"]
    t_p = pytorch_res["t"]
    omega_p = pytorch_res["omega"]

    n_gen = omega_a.shape[1]

    hz_a = (omega_a - 1.0) * 60.0
    hz_p = (omega_p - 1.0) * 60.0

    # 1) Worst-gen frequency comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    worst_a = hz_a.min(axis=1)
    worst_p = hz_p.min(axis=1)
    ax.plot(t_a, worst_a, "b-", label=f"ANDES (nadir {worst_a.min():.4f} Hz)", linewidth=1.5)
    ax.plot(t_p, worst_p, "r--", label=f"PyTorch (nadir {worst_p.min():.4f} Hz)", linewidth=1.5)
    ax.axvline(0.1, color="gray", linestyle="--", alpha=0.5, label="Disturbance")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Worst-gen Df [Hz]")
    title = "ANDES vs PyTorch: Worst Generator Frequency"
    if title_extra:
        title += f"\n{title_extra}"
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fname = f"andes_vs_pytorch_worst_gen_hz{suffix}"
    fig.savefig(out_dir / f"{fname}.png", dpi=150)
    fig.savefig(out_dir / f"{fname}.pdf")
    plt.close(fig)

    # 2) Per-generator comparison (2x5 grid)
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True, sharey=True)
    for i in range(n_gen):
        ax = axes[i // 5, i % 5]
        ax.plot(t_a, hz_a[:, i], "b-", linewidth=1, label="ANDES")
        ax.plot(t_p, hz_p[:, i], "r--", linewidth=1, label="PyTorch")
        ax.set_title(f"Gen {i+1} (Bus {GEN_BUSES[i]})", fontsize=9)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)
    axes[1, 2].set_xlabel("Time [s]")
    axes[0, 0].set_ylabel("Df [Hz]")
    axes[1, 0].set_ylabel("Df [Hz]")
    suptitle = "ANDES vs PyTorch: Per-Generator Frequency Response"
    if title_extra:
        suptitle += f" ({title_extra})"
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fname = f"andes_vs_pytorch_per_gen_hz{suffix}"
    fig.savefig(out_dir / f"{fname}.png", dpi=150)
    fig.savefig(out_dir / f"{fname}.pdf")
    plt.close(fig)

    return _compute_stats(hz_a, hz_p)


def _compute_stats(hz_a: np.ndarray, hz_p: np.ndarray) -> dict:
    """Compute per-gen and worst-gen nadir comparison statistics."""
    n_gen = hz_a.shape[1]
    stats = {"per_gen": [], "worst_andes": float(hz_a.min()), "worst_pytorch": float(hz_p.min())}

    for i in range(n_gen):
        na = float(hz_a[:, i].min())
        np_ = float(hz_p[:, i].min())
        diff = na - np_
        ratio = na / np_ if abs(np_) > 1e-8 else float("inf")
        stats["per_gen"].append({
            "gen": i + 1, "bus": GEN_BUSES[i],
            "andes_nadir_hz": na, "pytorch_nadir_hz": np_,
            "diff_hz": diff, "ratio": ratio,
        })

    stats["worst_diff_hz"] = stats["worst_andes"] - stats["worst_pytorch"]
    stats["worst_ratio"] = (
        stats["worst_andes"] / stats["worst_pytorch"]
        if abs(stats["worst_pytorch"]) > 1e-8 else float("inf")
    )
    return stats


def print_stats(stats: dict, label: str = ""):
    """Print nadir comparison table."""
    if label:
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"{'='*60}")
    print(f"  {'Gen':>4} {'Bus':>4} {'ANDES':>10} {'PyTorch':>10} {'Diff':>10} {'Ratio':>8}")
    for g in stats["per_gen"]:
        print(f"  {g['gen']:>4d} {g['bus']:>4d} {g['andes_nadir_hz']:>10.4f} "
              f"{g['pytorch_nadir_hz']:>10.4f} {g['diff_hz']:>10.4f} {g['ratio']:>8.2f}")
    print(f"  {'':>4} {'WORST':>4} {stats['worst_andes']:>10.4f} "
          f"{stats['worst_pytorch']:>10.4f} {stats['worst_diff_hz']:>10.4f} "
          f"{stats['worst_ratio']:>8.2f}")


def plot_all_steps_overlay(andes_results: list[tuple[str, dict]],
                           pytorch_res: dict, out_dir: Path):
    """Overlay worst-gen frequency from all ANDES steps with PyTorch."""
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    # PyTorch reference
    t_p = pytorch_res["t"]
    hz_p = (pytorch_res["omega"] - 1.0) * 60.0
    worst_p = hz_p.min(axis=1)
    ax.plot(t_p, worst_p, "k-", linewidth=2.5, label="PyTorch", alpha=0.9, zorder=10)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (name, ares) in enumerate(andes_results):
        t_a = ares["t"]
        hz_a = (ares["omega"] - 1.0) * 60.0
        worst_a = hz_a.min(axis=1)
        c = colors[i % len(colors)]
        ax.plot(t_a, worst_a, color=c, linewidth=1.2, alpha=0.8,
                label=f"S{i}: {name} ({worst_a.min():.4f})")

    ax.axvline(0.1, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Worst-gen Df [Hz]")
    ax.set_title("Incremental Parameter Matching: Worst-Gen Frequency Overlay")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "incremental_all_steps_overlay.png", dpi=150)
    fig.savefig(out_dir / "incremental_all_steps_overlay.pdf")
    plt.close(fig)
    print(f"[plot] Saved all-steps overlay")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_single(args):
    """Original single-comparison mode."""
    out_dir = Path(args.output_dir)
    lin_kpf = float(getattr(args, "linear_kpf", 10.0)) if getattr(args, "linear_control", False) else 0.0
    lin_kvv = float(getattr(args, "linear_kvv", 4.0)) if getattr(args, "linear_control", False) else 0.0
    load_bus_physical, load_bus_reduced = _resolve_disturbance_buses(
        args.load_bus, len(REDUCED_BUS_ORDER), getattr(args, "load_bus_mode", "physical")
    )
    print(
        f"[bus-map] mode={args.load_bus_mode}: input={int(args.load_bus)} "
        f"-> physical={int(load_bus_physical)}, reduced={int(load_bus_reduced)}"
    )

    andes_res = run_andes_sim(
        load_bus=load_bus_physical,
        load_mult_p=args.load_mult_p,
        load_mult_q=args.load_mult_q,
        load_t=args.load_t,
        tf=args.tf,
    )

    pytorch_res = run_pytorch_sim(
        load_bus=args.load_bus,
        load_bus_mode=getattr(args, "load_bus_mode", "physical"),
        load_mult_p=args.load_mult_p,
        load_mult_q=args.load_mult_q,
        load_t=args.load_t,
        tf=args.tf,
        device=args.device,
        disable_gov_rate_limit=getattr(args, "disable_pytorch_gov_rate_limit", False),
        linear_kpf=lin_kpf,
        linear_kvv=lin_kvv,
        pytorch_rebuild_equilibrium=getattr(args, "pytorch_rebuild_equilibrium", False),
        pytorch_strict_checkpoint_init=getattr(args, "pytorch_strict_checkpoint_init", False),
    )

    stats = plot_comparison(andes_res, pytorch_res, out_dir)
    print_stats(stats)

    np.savez(
        out_dir / "andes_pytorch_comparison.npz",
        andes_t=andes_res["t"],
        andes_omega=andes_res["omega"],
        pytorch_t=pytorch_res["t"],
        pytorch_omega=pytorch_res["omega"],
    )
    print(f"\n[done] Results saved to {out_dir}")


def run_incremental(args):
    """Incremental parameter matching: run each step cumulatively."""
    out_dir = Path(args.output_dir) / "incremental"
    out_dir.mkdir(parents=True, exist_ok=True)
    lin_kpf = float(getattr(args, "linear_kpf", 10.0)) if getattr(args, "linear_control", False) else 0.0
    lin_kvv = float(getattr(args, "linear_kvv", 4.0)) if getattr(args, "linear_control", False) else 0.0
    load_bus_physical, load_bus_reduced = _resolve_disturbance_buses(
        args.load_bus, len(REDUCED_BUS_ORDER), getattr(args, "load_bus_mode", "physical")
    )
    print(
        f"[bus-map] mode={args.load_bus_mode}: input={int(args.load_bus)} "
        f"-> physical={int(load_bus_physical)}, reduced={int(load_bus_reduced)}"
    )

    # --- Run PyTorch once ---
    pytorch_res = run_pytorch_sim(
        load_bus=args.load_bus,
        load_bus_mode=getattr(args, "load_bus_mode", "physical"),
        load_mult_p=args.load_mult_p,
        load_mult_q=args.load_mult_q,
        load_t=args.load_t,
        tf=args.tf,
        device=args.device,
        disable_gov_rate_limit=getattr(args, "disable_pytorch_gov_rate_limit", False),
        linear_kpf=lin_kpf,
        linear_kvv=lin_kvv,
        pytorch_rebuild_equilibrium=getattr(args, "pytorch_rebuild_equilibrium", False),
        pytorch_strict_checkpoint_init=getattr(args, "pytorch_strict_checkpoint_init", False),
    )

    # --- Get Sn from a quick ANDES setup (needed for D conversion) ---
    import andes
    ss_tmp = andes.load(
        andes.get_case("ieee39/ieee39_full.xlsx"),
        setup=False, default_config=True, no_output=True,
    )
    ss_tmp.setup()
    Sn_andes = np.array(ss_tmp.GENROU.Sn.v)
    H_andes = np.array(ss_tmp.GENROU.M.v) / 2.0  # M = 2H on machine base
    del ss_tmp

    # Use actual D from loaded PyTorch model (includes per-gen sg_ratio scaling)
    D_py_effective = pytorch_res["D_sys"]   # D_base * sg_ratio, on system base
    H_py_effective = pytorch_res["H_sys"]   # H_base * sg_ratio, on system base
    D_andes = compute_D_andes(D_py_effective, Sn_andes)

    print(f"\n[info] ANDES Sn: {Sn_andes}")
    print(f"[info] ANDES H (machine base): {np.array2string(H_andes, precision=2)}")
    print(f"[info] PyTorch D_effective (sys base, incl sg_ratio): {np.array2string(D_py_effective, precision=3)}")
    print(f"[info] PyTorch H_effective (sys base, incl sg_ratio): {np.array2string(H_py_effective, precision=2)}")
    print(f"[info] D_andes (machine base): {np.array2string(D_andes, precision=2)}")

    # --- Run each step ---
    all_stats = []
    andes_results = []

    for step_idx, (step_name, step_desc) in enumerate(STEP_CONFIGS):
        print(f"\n{'#'*60}")
        print(f"# STEP {step_idx}: {step_name} — {step_desc}")
        print(f"{'#'*60}")

        overrides = get_overrides_for_step(step_idx, D_andes)

        andes_res = run_andes_sim(
            load_bus=load_bus_physical,
            load_mult_p=args.load_mult_p,
            load_mult_q=args.load_mult_q,
            load_t=args.load_t,
            tf=args.tf,
            overrides=overrides,
        )

        suffix = f"_step{step_idx}_{step_name}"
        stats = plot_comparison(
            andes_res, pytorch_res, out_dir,
            suffix=suffix,
            title_extra=f"Step {step_idx}: {step_desc}",
        )

        print_stats(stats, label=f"Step {step_idx}: {step_name} — {step_desc}")
        all_stats.append((step_name, step_desc, stats))
        andes_results.append((step_name, andes_res))

        # Save per-step data
        np.savez(
            out_dir / f"data{suffix}.npz",
            andes_t=andes_res["t"],
            andes_omega=andes_res["omega"],
            pytorch_t=pytorch_res["t"],
            pytorch_omega=pytorch_res["omega"],
        )

    # --- Summary ---
    plot_all_steps_overlay(andes_results, pytorch_res, out_dir)

    print(f"\n{'='*60}")
    print(f"  INCREMENTAL SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Step':>4} {'Name':<16} {'ANDES nadir':>12} {'PyTorch':>10} {'|Diff|':>10} {'Ratio':>8}")
    for i, (name, desc, stats) in enumerate(all_stats):
        diff = abs(stats["worst_diff_hz"])
        print(f"  {i:>4d} {name:<16} {stats['worst_andes']:>12.4f} "
              f"{stats['worst_pytorch']:>10.4f} {diff:>10.4f} {stats['worst_ratio']:>8.3f}")

    # Save summary JSON
    summary = {
        "steps": [],
        "D_py_effective": D_py_effective.tolist(),
        "H_py_effective": H_py_effective.tolist(),
        "D_andes": D_andes.tolist(),
        "H_andes": H_andes.tolist(),
        "Sn_andes": Sn_andes.tolist(),
    }
    for i, (name, desc, stats) in enumerate(all_stats):
        summary["steps"].append({
            "index": i,
            "name": name,
            "description": desc,
            "worst_andes_hz": stats["worst_andes"],
            "worst_pytorch_hz": stats["worst_pytorch"],
            "worst_diff_hz": stats["worst_diff_hz"],
            "worst_ratio": stats["worst_ratio"],
            "per_gen": stats["per_gen"],
        })

    with open(out_dir / "incremental_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] All results saved to {out_dir}")


# ---------------------------------------------------------------------------
# Full-match ANDES simulation (generators + inverters, all params matched)
# ---------------------------------------------------------------------------

def run_andes_full_match(
    load_bus: int = 18,
    load_mult_p: float = 1.5,
    load_mult_q: float = 1.5,
    load_t: float = 0.1,
    tf: float = 2.0,
    dt: float = 0.005,
    load_ramp_tau: float = 0.02,
    with_inverters: bool = True,
    use_reeca: bool = False,
    use_sexs: bool = False,
    match_reactances: bool = False,
    repair_equilibrium: bool = True,
    match_reduced_load_profile: bool = True,
    match_shares_from_pytorch: bool = False,
    match_dispatch_from_pytorch: bool = False,
    reconcile_dispatch: bool = True,
    reconcile_tol: float = 0.05,
    reconcile_max_iter: int = 6,
    reconcile_alpha: float = 0.8,
    reconcile_max_step: float = 0.6,
    reconcile_slack_weight: float = 0.0,
    disable_pytorch_gov_rate_limit: bool = False,
    linear_control: bool = False,
    linear_kpf: float = 0.0,
    linear_kvv: float = 0.0,
    andes_enable_linear_qdroop: bool = False,
    andes_disable_gov_limits: bool = False,
    andes_gov_r_scale: float = 1.0,
    pytorch_rebuild_equilibrium: bool = False,
    pytorch_strict_checkpoint_init: bool = False,
    andes_case: str | None = None,
    device: str = "cpu",
) -> dict:
    """Run ANDES IEEE39 with ALL parameters matched to PyTorch.

    Adds GFL inverters (if with_inverters=True), matches generator H/D/gov/exc,
    applies ZIP load model, then runs TDS with load step.

    Parameters
    ----------
    use_sexs : bool
        Replace IEEEX1 with SEXS (simplified exciter matching PyTorch 1st-order).
    repair_equilibrium : bool
        With ``match_reactances=True``, enforce a GENROU-consistent transient
        pair ``Xq' = Xd'`` (machine base) to avoid initialization drift.
    match_reduced_load_profile : bool
        If True (default), overwrite ANDES PQ loads at retained reduced buses with
        the PyTorch reduced-network base load profile.
    match_shares_from_pytorch : bool
        If True, copy SG/PV power shares from PyTorch equilibrium. Default False:
        use structural SG/PV ratios from the model definition (dynamics-first).

    Returns dict with generator omega + inverter PLL states.
    """
    import andes

    # Dynamics-first defaults: match structural physical parameters, not
    # equilibrium dispatch values copied from PyTorch.
    H_eff_target = H_BASE * SG_RATIO
    D_eff_target = D_BASE * SG_RATIO
    p_total_target = None
    pref0_target = None
    p_inv_target = None
    p_load_target_by_bus = None
    q_load_target_by_bus = None
    sys_model_match = None
    dispatch_reconcile_summary = None

    def _get_sys_model():
        nonlocal sys_model_match
        if sys_model_match is None:
            sys_model_match, _ = _load_pytorch_system_for_andes(
                device=device,
                disable_gov_rate_limit=disable_pytorch_gov_rate_limit,
                rebuild_equilibrium=pytorch_rebuild_equilibrium,
                strict_checkpoint_init=pytorch_strict_checkpoint_init,
            )
        return sys_model_match

    if match_reduced_load_profile:
        p_load_target_by_bus, q_load_target_by_bus = _extract_pytorch_reduced_load_targets(_get_sys_model())
        print("[full-match] Load profile source: PyTorch reduced-network base loads")
    else:
        print("[full-match] Load profile source: native ANDES case loads")

    if with_inverters:
        if match_shares_from_pytorch:
            pt_targets = _extract_pytorch_operating_targets(_get_sys_model())
            sg_share_target = np.asarray(pt_targets["sg_share"], dtype=float)
            pv_share_target = np.asarray(pt_targets["pv_share"], dtype=float)
            pref0_target = np.asarray(pt_targets["pref0"], dtype=float)
            p_inv_target = np.asarray(pt_targets["p_inv_gen"], dtype=float)
            p_total_target = np.asarray(pt_targets["p_total"], dtype=float)
            print("[full-match] SG/PV shares source: PyTorch equilibrium (explicit opt-in)")
        else:
            sg_share_target = SG_RATIO.copy()
            pv_share_target = PV_RATIO.copy()
            print("[full-match] SG/PV shares source: structural model ratios (default)")
    else:
        sg_share_target = np.ones(10, dtype=float)
        pv_share_target = np.zeros(10, dtype=float)
        print("[full-match] Generators-only mode: SG share=1.0, PV share=0.0")

    # Dispatch forcing needs PyTorch equilibrium targets (explicit opt-in path).
    if match_dispatch_from_pytorch:
        if p_total_target is None:
            pt_targets = _extract_pytorch_operating_targets(_get_sys_model())
            pref0_target = np.asarray(pt_targets["pref0"], dtype=float)
            p_inv_target = np.asarray(pt_targets["p_inv_gen"], dtype=float)
            p_total_target = np.asarray(pt_targets["p_total"], dtype=float)
        print("[full-match] Dispatch source: PyTorch equilibrium (explicit opt-in)")
    else:
        print("[full-match] Dispatch source: native ANDES power flow (default)")

    reconcile_dispatch = bool(reconcile_dispatch)
    do_reconcile = bool(reconcile_dispatch and with_inverters and (not match_dispatch_from_pytorch))
    if do_reconcile:
        if p_total_target is None:
            pt_targets = _extract_pytorch_operating_targets(_get_sys_model())
            p_total_target = np.asarray(pt_targets["p_total"], dtype=float)
        print(
            "[full-match] Dispatch reconciliation: enabled "
            f"(non-slack PV iterative fit, slack_weight={float(reconcile_slack_weight):.3f})"
        )
    elif reconcile_dispatch and match_dispatch_from_pytorch:
        print("[full-match] Dispatch reconciliation: skipped (dispatch already forced from PyTorch)")
    elif reconcile_dispatch and (not with_inverters):
        print("[full-match] Dispatch reconciliation: skipped (generators-only mode)")
    else:
        print("[full-match] Dispatch reconciliation: disabled")

    case_path = str(andes_case).strip() if andes_case is not None else ""
    if not case_path:
        case_path = andes.get_case("ieee39/ieee39_full.xlsx")
    print(f"\n[full-match] Loading case (setup=False): {case_path}")
    ss = andes.load(
        case_path,
        setup=False,
        default_config=True,
        no_output=True,
    )

    if match_reduced_load_profile and p_load_target_by_bus is not None and q_load_target_by_bus is not None:
        _apply_andes_load_targets(ss, p_load_target_by_bus, q_load_target_by_bus)

    # --- Optionally match GENROU reactances/time constants BEFORE setup ---
    if match_reactances:
        _Sn_list = ss.GENROU.Sn.v  # list of Sn values
        for i in range(len(_Sn_list)):
            _sn = _Sn_list[i]
            # Reactances: convert PyTorch system base → ANDES machine base
            xd1_mach = float(XD1_PY[i] * _sn / SB)
            xq1_mach = float(XQ1_PY[i] * _sn / SB)
            if repair_equilibrium:
                # GENROU round-rotor consistency repair:
                # enforce Xq' = Xd' to avoid pre-disturbance drift after mapping.
                xq1_mach = xd1_mach
            ss.GENROU.xd.v[i]  = float(XD_PY[i]  * _sn / SB)
            ss.GENROU.xd1.v[i] = xd1_mach
            ss.GENROU.xq.v[i]  = float(XQ_PY[i]  * _sn / SB)
            ss.GENROU.xq1.v[i] = xq1_mach
            # Subtransient = transient (4th order)
            ss.GENROU.xd2.v[i] = xd1_mach
            ss.GENROU.xq2.v[i] = xq1_mach
            # Leakage reactance: must be < Xd'' = Xd' (PyTorch has no xl)
            # Set to 80% of min(Xd', Xq') to satisfy GENROU constraint xl <= Xd''
            xl_new = 0.8 * min(xd1_mach, xq1_mach)
            ss.GENROU.xl.v[i] = xl_new
            # Time constants (seconds, no base conversion)
            ss.GENROU.Td10.v[i] = float(TD0_PY[i])
            ss.GENROU.Tq10.v[i] = float(TQ0_PY[i])
            ss.GENROU.Td20.v[i] = 999.0
            ss.GENROU.Tq20.v[i] = 999.0
            ss.GENROU.ra.v[i] = 0.0
        print(
            "[full-match] GENROU reactances + xl matched (Pai 1989 → machine base), "
            f"repair_equilibrium={'on' if repair_equilibrium else 'off'}"
        )
    else:
        # Only freeze subtransient (4th order) — keep ANDES native reactances
        for i in range(len(ss.GENROU.Td20.v)):
            ss.GENROU.Td20.v[i] = 999.0
            ss.GENROU.Tq20.v[i] = 999.0
            # Set Xd2=Xd1, Xq2=Xq1 (effectively 4th order)
            ss.GENROU.xd2.v[i] = ss.GENROU.xd1.v[i]
            ss.GENROU.xq2.v[i] = ss.GENROU.xq1.v[i]
        print("[full-match] Reactances: ANDES native (subtransient frozen → 4th order)")

    # Optional: force PF dispatch targets to PyTorch equilibrium.
    # Default is OFF so we match initialization mechanics without forcing
    # absolute dispatch values into the ANDES case data.
    if match_dispatch_from_pytorch:
        _apply_andes_dispatch_targets(ss, p_total_target)
    else:
        print("[full-match] Dispatch forcing disabled (using native ANDES PV/Slack p0)")

    # --- Add models BEFORE setup ---
    if with_inverters:
        _add_inverters(
            ss,
            use_reeca=use_reeca,
            pv_ratio_target=pv_share_target,
            linear_control=linear_control,
            linear_kpf=linear_kpf,
            linear_kvv=linear_kvv,
            enable_linear_qdroop=andes_enable_linear_qdroop,
        )

    if use_sexs:
        # Disable IEEEX1 in raw data BEFORE setup (it's a list at this point)
        for i in range(len(ss.IEEEX1.u.v)):
            ss.IEEEX1.u.v[i] = 0
        _add_sexs_exciters(ss)

    ss.setup()

    # --- Match ALL generator parameters AFTER setup ---
    _match_generator_params(
        ss,
        with_inverters=with_inverters,
        use_sexs=use_sexs,
        sg_ratio_target=sg_share_target,
        H_eff_target=H_eff_target,
        D_eff_target=D_eff_target,
        gov_r_scale=float(andes_gov_r_scale),
    )

    # --- Power flow ---
    print("[full-match] Running power flow...")
    ss.PFlow.run()
    if not ss.PFlow.converged:
        raise RuntimeError("ANDES power flow did not converge")

    pv_p0_before_reconcile = np.asarray(ss.PV.p0.v, dtype=float).copy()

    # Optional iterative reconciliation:
    # reduce dispatch mismatch against PyTorch equilibrium totals while
    # keeping slack bus free and preserving structural SG/PV dynamics.
    if do_reconcile and p_total_target is not None:
        dispatch_reconcile_summary = _reconcile_dispatch_non_slack(
            ss,
            p_total_target=p_total_target,
            max_iter=int(reconcile_max_iter),
            tol=float(reconcile_tol),
            alpha=float(reconcile_alpha),
            max_step=float(reconcile_max_step),
            slack_weight=float(reconcile_slack_weight),
        )

    # --- Initialize TDS to populate dynamic states (tm0) ---
    print("[full-match] Initializing TDS to populate dynamic states...")
    tds_exc = None
    try:
        ss.TDS.init()
    except Exception as exc:
        tds_exc = exc
    tds_ok = (tds_exc is None) and _tds_state_is_finite(ss)
    if not tds_ok:
        fail_msg = "exception during TDS.init" if tds_exc is not None else "non-finite DAE state after TDS.init"
        if do_reconcile:
            print(f"[full-match] [warn] TDS init failed after reconciliation ({fail_msg}); restoring pre-reconcile PV p0")
            if dispatch_reconcile_summary is None:
                dispatch_reconcile_summary = {"enabled": False, "reason": "post_reconcile_tds_init_failed"}
            else:
                dispatch_reconcile_summary = dict(dispatch_reconcile_summary)
                dispatch_reconcile_summary["enabled"] = False
                dispatch_reconcile_summary["reason"] = "post_reconcile_tds_init_failed"
            dispatch_reconcile_summary["fallback_restore_used"] = True
            dispatch_reconcile_summary["fallback_detail"] = fail_msg
            if tds_exc is not None:
                dispatch_reconcile_summary["fallback_exception"] = str(tds_exc)
            ss.PV.p0.v[:] = pv_p0_before_reconcile
            ss.PFlow.run()
            if not ss.PFlow.converged:
                raise RuntimeError("ANDES power flow failed while restoring pre-reconcile PV p0")
            ss.TDS.init()
            if not _tds_state_is_finite(ss):
                raise RuntimeError("ANDES TDS initialization remained non-finite after reconciliation rollback")
        else:
            if tds_exc is not None:
                raise RuntimeError(f"ANDES TDS initialization failed: {tds_exc}") from tds_exc
            raise RuntimeError("ANDES TDS initialization produced non-finite DAE state")

    # --- Post-PFlow: set governor VMAX/VMIN to match PyTorch ---
    tm0 = np.array(ss.GENROU.tm0.v)  # equilibrium Pm (system base, from PFlow)
    if andes_disable_gov_limits:
        vmax_target = np.full_like(tm0, 999.0, dtype=float)
        vmin_target = np.full_like(tm0, -999.0, dtype=float)
        ss.TGOV1N.VMAX.v[:] = vmax_target
        ss.TGOV1N.VMIN.v[:] = vmin_target
        print(f"[full-match] Post-TDS-init: tm0 = {np.array2string(tm0, precision=3)}")
        print("[full-match] Governor limits disabled (VMAX=+999, VMIN=-999)")
    else:
        vmax_target = 1.5 * np.abs(tm0)
        ss.TGOV1N.VMAX.v[:] = vmax_target
        ss.TGOV1N.VMIN.v[:] = 0.0
        print(f"[full-match] Post-TDS-init: tm0 = {np.array2string(tm0, precision=3)}")
        print(f"[full-match] Post-TDS-init: VMAX = 1.5*tm0 = {np.array2string(vmax_target, precision=3)}")

    # Print gen/inverter power dispatch
    Sn = np.array(ss.GENROU.Sn.v)
    print(f"[full-match] GENROU Sn: {np.array2string(Sn, precision=0)}")

    # Find PQ load at the target bus
    pq_df = ss.PQ.as_df()
    pq_at_bus = pq_df[pq_df["bus"] == load_bus]
    if pq_at_bus.empty:
        raise ValueError(f"No PQ load at bus {load_bus}")

    pq_idx = pq_at_bus["idx"].iloc[0]
    pq_int_idx = list(ss.PQ.idx.v).index(pq_idx)

    ppf_orig = float(ss.PQ.Ppf.v[pq_int_idx])
    qpf_orig = float(ss.PQ.Qpf.v[pq_int_idx])
    ppf_new = ppf_orig * load_mult_p
    qpf_new = qpf_orig * load_mult_q
    pq_base = _capture_pq_component_bases(ss, pq_int_idx)
    print(f"[full-match] Bus {load_bus}: Ppf {ppf_orig:.4f} -> {ppf_new:.4f}, "
          f"Qpf {qpf_orig:.4f} -> {qpf_new:.4f}")

    # --- Phase 1: Steady state ---
    print(f"[full-match] TDS phase 1 (0 -> {load_t}s)...")
    ss.TDS.config.tf = load_t
    ss.TDS.config.tstep = dt
    t0 = time.perf_counter()
    ss.TDS.run()

    # Check steady-state omega drift (should be ~0)
    omega_idx = ss.GENROU.omega.a
    omega_ss = np.array(ss.dae.x[omega_idx])
    drift = (omega_ss - 1.0) * 60.0
    print(f"[full-match] Pre-disturbance omega drift: max={drift.max():.6f} Hz, "
          f"min={drift.min():.6f} Hz")

    # --- Phase 2: Disturbance application ---
    ramp_tau = max(0.0, float(load_ramp_tau))
    if ramp_tau <= 1e-12:
        print(f"[full-match] Applying load step at t={load_t}s...")
        _apply_pq_component_scale(
            ss,
            pq_int_idx,
            base=pq_base,
            mult_p=load_mult_p,
            mult_q=load_mult_q,
            tag=" [full-match]",
            verbose=True,
        )
        print(f"[full-match] TDS phase 2 ({load_t} -> {tf}s)...")
        ss.TDS.config.tf = tf
        ss.TDS.run()
    else:
        # Match PyTorch AlterManager disturbance profile:
        # factor(t)=1+(mult-1)*(1-exp(-(t-load_t)/tau)).
        ramp_end = min(float(tf), float(load_t) + 6.0 * ramp_tau)
        ramp_step = max(min(float(dt), ramp_tau / 4.0), 1e-4)
        t_cursor = float(load_t)
        n_updates = 0
        print(
            f"[full-match] Applying smooth load ramp at t={load_t}s "
            f"(tau={ramp_tau:.4f}s, step={ramp_step:.4f}s, end={ramp_end:.4f}s)..."
        )
        while t_cursor < (ramp_end - 1e-12):
            t_next = min(ramp_end, t_cursor + ramp_step)
            z = max(0.0, t_next - float(load_t))
            fac_p = 1.0 + (float(load_mult_p) - 1.0) * (1.0 - np.exp(-z / ramp_tau))
            fac_q = 1.0 + (float(load_mult_q) - 1.0) * (1.0 - np.exp(-z / ramp_tau))
            _apply_pq_component_scale(
                ss,
                pq_int_idx,
                base=pq_base,
                mult_p=fac_p,
                mult_q=fac_q,
                tag=" [full-match-ramp]",
                verbose=False,
            )
            ss.TDS.config.tf = t_next
            ss.TDS.run()
            t_cursor = t_next
            n_updates += 1

        # Snap to final target and continue to tf.
        _apply_pq_component_scale(
            ss,
            pq_int_idx,
            base=pq_base,
            mult_p=load_mult_p,
            mult_q=load_mult_q,
            tag=" [full-match-ramp-final]",
            verbose=False,
        )
        print(f"[full-match] Ramp updates applied: {n_updates}")
        if ramp_end < (float(tf) - 1e-12):
            print(f"[full-match] TDS post-ramp ({ramp_end:.4f} -> {tf}s)...")
            ss.TDS.config.tf = tf
            ss.TDS.run()
    elapsed = time.perf_counter() - t0
    print(f"[full-match] TDS done in {elapsed:.1f}s")

    # --- Extract ALL results ---
    t_arr = np.asarray(ss.dae.ts.t, dtype=float)
    x_ts = np.asarray(ss.dae.ts.x, dtype=float)
    y_ts = np.asarray(ss.dae.ts.y, dtype=float)
    if x_ts.ndim != 2 or x_ts.shape[0] == 0 or x_ts.shape[1] == 0:
        raise RuntimeError(
            "ANDES TDS produced no dynamic-state trajectories (ss.dae.ts.x is empty). "
            "This usually indicates initialization instability."
        )
    omega_idx = np.asarray(omega_idx, dtype=int).reshape(-1)
    if omega_idx.size == 0 or int(np.max(omega_idx)) >= x_ts.shape[1]:
        raise RuntimeError(
            f"Invalid GENROU omega indices for trajectory extraction: max_idx={int(np.max(omega_idx)) if omega_idx.size else -1}, "
            f"state_cols={x_ts.shape[1]}"
        )
    omega_ts = np.array(x_ts[:, omega_idx])  # (T, 10)
    n_y = y_ts.shape[1] if y_ts.ndim == 2 else 0

    result = {
        "t": t_arr,
        "load_t": float(load_t),
        "andes_load_ramp_tau": float(load_ramp_tau),
        "andes_case": str(case_path),
        "load_profile_source": ("pytorch_reduced_base" if match_reduced_load_profile else "andes_native"),
        "exciter_model": "SEXS" if use_sexs else "IEEEX1",
        "linear_control": bool(linear_control),
        "linear_kpf": float(linear_kpf),
        "linear_kvv": float(linear_kvv),
        "inverter_control_model": ("REECA1E" if linear_control else ("REECA1" if use_reeca else "open_loop")),
        "omega": omega_ts,
        "gen_buses": [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
        "Sn": Sn,
        "H_eff_target": np.asarray(H_eff_target, dtype=float),
        "D_eff_target": np.asarray(D_eff_target, dtype=float),
        "sg_share_target": np.asarray(sg_share_target, dtype=float),
        "pv_share_target": np.asarray(pv_share_target, dtype=float),
        "p_total_target": (None if p_total_target is None else np.asarray(p_total_target, dtype=float)),
        "pref0_target": (None if pref0_target is None else np.asarray(pref0_target, dtype=float)),
        "p_inv_target": (None if p_inv_target is None else np.asarray(p_inv_target, dtype=float)),
        "share_target_source": ("pytorch_equilibrium" if match_shares_from_pytorch else "model_structural"),
        "dispatch_forced_from_pytorch": bool(match_dispatch_from_pytorch),
        "dispatch_reconciled": bool(dispatch_reconcile_summary is not None and dispatch_reconcile_summary.get("enabled", False)),
        "dispatch_reconcile_summary": dispatch_reconcile_summary,
        "R_andes_target": np.full(10, R_PY / OMEGA_S, dtype=float),
    }

    # --- Generator dynamic states (x) ---
    def _safe_x(model, attr):
        obj = getattr(model, attr, None)
        if obj is not None and hasattr(obj, "a") and len(obj.a) > 0:
            idx = np.asarray(obj.a, dtype=int).reshape(-1)
            if np.all(idx < x_ts.shape[1]):
                return np.array(x_ts[:, idx])
        return None

    def _safe_y(model, attr):
        obj = getattr(model, attr, None)
        if obj is not None and hasattr(obj, "a") and len(obj.a) > 0:
            idx = np.asarray(obj.a, dtype=int).reshape(-1)
            if np.all(idx < n_y):
                return np.array(y_ts[:, idx])
        return None

    # Rotor angle (absolute in ANDES — convert to relative below)
    delta_abs = _safe_x(ss.GENROU, "delta")
    if delta_abs is not None:
        # Convert to relative: delta_rel[i] = delta[i+1] - delta[0]
        # PyTorch packs delta_rel as generators 2..10 relative to generator 1.
        result["delta_abs"] = delta_abs
        result["delta_rel"] = delta_abs[:, 1:10] - delta_abs[:, 0:1]

    # Transient EMFs
    e1q = _safe_x(ss.GENROU, "e1q")
    if e1q is not None:
        result["Eqp"] = e1q  # E'q (machine base)
    e1d = _safe_x(ss.GENROU, "e1d")
    if e1d is not None:
        result["Edp"] = e1d  # E'd (machine base)

    # --- Exciter: field voltage Efd ---
    # ANDES: vf is the algebraic output (y), IEEEX1.LA_y is the exciter state
    vf = _safe_y(ss.GENROU, "vf")
    if vf is not None:
        result["Efd"] = vf  # Efd on machine base (y variable)

    # --- Governor: mechanical power, valve position ---
    # tm = mechanical torque (algebraic y, set by governor)
    tm = _safe_y(ss.GENROU, "tm")
    if tm is not None:
        result["Pm"] = tm  # Pm on machine base

    # Pvalve = governor lag output (state x in TGOV1N)
    pvalve = _safe_x(ss.TGOV1N, "LAG_y")
    if pvalve is not None:
        result["Pvalve"] = pvalve  # on machine base

    # Governor total output (algebraic)
    pout = _safe_y(ss.TGOV1N, "pout")
    if pout is not None:
        result["gov_pout"] = pout

    # --- Electrical power (algebraic) ---
    Pe = _safe_y(ss.GENROU, "Pe")
    if Pe is not None:
        result["Pe"] = Pe  # on machine base
    Qe = _safe_y(ss.GENROU, "Qe")
    if Qe is not None:
        result["Qe"] = Qe

    # --- Bus voltages ---
    v_idx = ss.Bus.v.a
    result["v_bus"] = np.array(ss.dae.ts.y[:, v_idx])
    result["bus_ids"] = list(ss.Bus.idx.v)

    # Get gen-bus voltage indices
    gen_bus_list = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39]
    bus_id_list = [int(b) for b in ss.Bus.idx.v]
    gen_bus_int_idx = [bus_id_list.index(b) for b in gen_bus_list if b in bus_id_list]
    if gen_bus_int_idx:
        result["v_gen_bus"] = np.array(ss.dae.ts.y[:, np.array(v_idx)[gen_bus_int_idx]])

    # --- Inverter PLL states (if present) ---
    if with_inverters and hasattr(ss, "PLL2") and len(ss.PLL2.idx.v) > 0:
        try:
            pll_xi = _safe_x(ss.PLL2, "PI_xi")
            pll_am = _safe_x(ss.PLL2, "am")
            if pll_xi is not None:
                result["pll_xi"] = pll_xi
            if pll_am is not None:
                result["pll_am"] = pll_am
            print(f"[full-match] Extracted PLL2 states ({pll_xi.shape[1] if pll_xi is not None else 0} inverters)")
        except Exception as e:
            print(f"[full-match] [warn] Could not extract PLL2 states: {e}")

    # REGCP1 converter states
    if with_inverters and hasattr(ss, "REGCP1") and len(ss.REGCP1.idx.v) > 0:
        try:
            for sname in ["S1_y", "S2_y"]:
                val = _safe_x(ss.REGCP1, sname)
                if val is not None:
                    result[f"regcp1_{sname}"] = val
            for sname in ["Pe", "Qe", "Ipout", "Iqout"]:
                val = _safe_y(ss.REGCP1, sname)
                if val is not None:
                    result[f"regcp1_{sname}"] = val
        except Exception as e:
            print(f"[full-match] [warn] Could not extract REGCP1 outputs: {e}")

    print(f"[full-match] Extracted {len(result)} state trajectories")
    return result


def run_pytorch_sim_full(
    load_bus: int = 18,
    load_bus_mode: str = "physical",
    load_mult_p: float = 1.5,
    load_mult_q: float = 1.5,
    load_t: float = 0.1,
    tf: float = 2.0,
    dt: float = 0.001,
    device: str = "cpu",
    disable_gov_rate_limit: bool = False,
    linear_kpf: float = 0.0,
    linear_kvv: float = 0.0,
    pytorch_rebuild_equilibrium: bool = False,
    pytorch_strict_checkpoint_init: bool = False,
) -> dict:
    """Run PyTorch IEEE39+GFL6 model open-loop, return ALL state trajectories.

    `load_bus_mode` controls whether `load_bus` is interpreted as:
      - `physical`: IEEE physical bus number in the retained reduced set
      - `reduced-index`: reduced-model bus index [0..n_bus-1]
    """
    from train_gfl6 import evaluate_controller
    from baselines.transfer_to_gfl6 import LinearDroopTransferController

    sys_model, device = _load_pytorch_system_for_andes(
        device,
        disable_gov_rate_limit=disable_gov_rate_limit,
        rebuild_equilibrium=pytorch_rebuild_equilibrium,
        strict_checkpoint_init=pytorch_strict_checkpoint_init,
    )

    ctrl = LinearDroopTransferController(
        sys_model, kpf=float(linear_kpf), kvv=float(linear_kvv)
    ).to(device).eval()
    print(f"[pytorch] Linear droop controller: kpf={float(linear_kpf):.4f}, kvv={float(linear_kvv):.4f}")
    load_bus_physical, load_bus_reduced = _resolve_disturbance_buses(
        load_bus, int(sys_model.n_bus), load_bus_mode
    )
    print(
        f"[pytorch] Disturbance bus mode={str(load_bus_mode)}: "
        f"physical {load_bus_physical} -> reduced index {load_bus_reduced}"
    )

    n_steps = int(tf / dt)
    print(f"[pytorch] Running TDS (tf={tf}s, dt={dt}s, steps={n_steps})...")
    t0 = time.perf_counter()
    t_pts, x_pts = evaluate_controller(
        sys_model=sys_model,
        controller=ctrl,
        T=tf,
        load_change=load_mult_p,
        load_change_q=load_mult_q,
        load_bus=load_bus_reduced,
        load_t=load_t,
        n_sample=n_steps,
        dt=dt,
        device=device,
        control_limit_mode="capacity",
    )
    elapsed = time.perf_counter() - t0
    print(f"[pytorch] TDS done in {elapsed:.1f}s")

    x = x_pts[:, 0, :] if x_pts.ndim == 3 else x_pts
    n_gen = int(sys_model.n_gen)
    n_inv = int(sys_model.n_inv)
    xn = x.detach().cpu().numpy()

    # State index layout (129 total):
    #   delta_rel: 0 .. n_gen-2   (9 states), generators 2..10 relative to gen 1
    #   omega:     n_gen-1 .. 2*n_gen-2  (10 states)
    #   Eqp:       2*n_gen-1 .. 3*n_gen-2
    #   Edp:       3*n_gen-1 .. 4*n_gen-2
    #   Efd:       4*n_gen-1 .. 5*n_gen-2
    #   Pm:        5*n_gen-1 .. 6*n_gen-2
    #   Pvalve:    6*n_gen-1 .. 7*n_gen-2
    #   Inverter:  7*n_gen-1 .. end  (6 per inverter)
    g = n_gen
    gov_rate_enabled = bool(getattr(sys_model, "enable_pvalve_rate", False))
    max_dpv_up = getattr(sys_model, "max_dPvalve_up", None)
    max_dpv_dn = getattr(sys_model, "max_dPvalve_dn", None)
    if max_dpv_up is not None:
        max_dpv_up = max_dpv_up[:n_gen].detach().cpu().numpy()
    if max_dpv_dn is not None:
        max_dpv_dn = max_dpv_dn[:n_gen].detach().cpu().numpy()

    result = {
        "t": t_pts.detach().cpu().numpy(),
        "load_t": float(load_t),
        # Generator states (all on system base Sb=100)
        "delta_rel": xn[:, 0: g - 1],            # (T, 9) — generators 2..10 relative to gen 1
        "omega":     xn[:, g - 1: 2*g - 1],      # (T, 10)
        "Eqp":       xn[:, 2*g - 1: 3*g - 1],    # (T, 10)
        "Edp":       xn[:, 3*g - 1: 4*g - 1],    # (T, 10)
        "Efd":       xn[:, 4*g - 1: 5*g - 1],    # (T, 10)
        "Pm":        xn[:, 5*g - 1: 6*g - 1],    # (T, 10)
        "Pvalve":    xn[:, 6*g - 1: 7*g - 1],    # (T, 10)
        # Inverter states
        "inv_phi":    xn[:, 7*g-1+0::6][:, :n_inv],   # PLL angle
        "inv_xi_pll": xn[:, 7*g-1+1::6][:, :n_inv],   # PLL integrator
        "inv_f_meas": xn[:, 7*g-1+2::6][:, :n_inv],   # Filtered frequency
        "inv_V_meas": xn[:, 7*g-1+3::6][:, :n_inv],   # Filtered voltage
        "inv_P_ord":  xn[:, 7*g-1+4::6][:, :n_inv],   # Active power order
        "inv_Q_ord":  xn[:, 7*g-1+5::6][:, :n_inv],   # Reactive power order
        # Model params for reference
        "D_sys": sys_model.D[:n_gen].detach().cpu().numpy(),
        "H_sys": sys_model.H[:n_gen].detach().cpu().numpy(),
        "pytorch_enable_pvalve_rate": gov_rate_enabled,
        "pytorch_max_dPvalve_up": max_dpv_up,
        "pytorch_max_dPvalve_dn": max_dpv_dn,
        "load_bus_input": int(load_bus),
        "load_bus_mode": str(load_bus_mode),
        "load_bus_physical": int(load_bus_physical),
        "load_bus_reduced": int(load_bus_reduced),
    }
    return result


def _plot_per_gen_grid(t_a, data_a, t_p, data_p, out_dir, fname, title,
                       ylabel, n_cols=10, convert_fn=None):
    """Generic 2x5 per-generator/inverter overlay plot.

    Parameters
    ----------
    convert_fn : callable, optional
        Applied to each column pair: (a_col, p_col) -> (a_col, p_col).
        Use for per-unit base conversion.
    """
    n = data_a.shape[1]
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    n_cols_actual = min(n, n_cols)
    fig, axes = plt.subplots(n_rows, n_cols_actual,
                             figsize=(3.2 * n_cols_actual, 3 * n_rows),
                             sharex=True, squeeze=False)
    for i in range(n):
        ax = axes[i // n_cols_actual, i % n_cols_actual]
        a_col = data_a[:, i]
        p_col = data_p[:, i] if data_p is not None and i < data_p.shape[1] else None
        if convert_fn is not None and p_col is not None:
            a_col, p_col = convert_fn(a_col, p_col, i)
        ax.plot(t_a, a_col, "b-", linewidth=0.8, label="ANDES")
        if p_col is not None:
            ax.plot(t_p, p_col, "r--", linewidth=0.8, label="PyTorch")
        bus = GEN_BUSES[i] if i < len(GEN_BUSES) else i + 1
        ax.set_title(f"Gen {i+1} (Bus {bus})", fontsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=6)
    # Hide unused subplots
    for j in range(n, n_rows * n_cols_actual):
        axes[j // n_cols_actual, j % n_cols_actual].set_visible(False)
    axes[-1, n_cols_actual // 2].set_xlabel("Time [s]")
    for r in range(n_rows):
        axes[r, 0].set_ylabel(ylabel, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{fname}.png", dpi=150)
    plt.close(fig)


def _compute_trajectory_metrics(t_a, data_a, t_p, data_p):
    """Compute per-column RMS error, peak error, and correlation.

    Interpolates PyTorch data onto ANDES time grid for fair comparison.
    Returns dict of arrays (one value per column).
    """
    n = data_a.shape[1]
    # Interpolate PyTorch onto ANDES time grid
    from scipy.interpolate import interp1d
    n_p = min(data_p.shape[1], n)
    rms = np.full(n, np.nan)
    peak = np.full(n, np.nan)
    corr = np.full(n, np.nan)
    nadir_a = np.full(n, np.nan)
    nadir_p = np.full(n, np.nan)

    for i in range(n_p):
        f = interp1d(t_p, data_p[:, i], kind="linear", fill_value="extrapolate")
        p_interp = f(t_a)
        diff = data_a[:, i] - p_interp
        rms[i] = float(np.sqrt(np.mean(diff ** 2)))
        peak[i] = float(np.max(np.abs(diff)))
        # Pearson correlation
        std_a = np.std(data_a[:, i])
        std_p = np.std(p_interp)
        if std_a > 1e-12 and std_p > 1e-12:
            corr[i] = float(np.corrcoef(data_a[:, i], p_interp)[0, 1])
        else:
            corr[i] = 1.0 if std_a < 1e-12 and std_p < 1e-12 else 0.0
        nadir_a[i] = float(data_a[:, i].min())
        nadir_p[i] = float(data_p[:, i].min())

    return {"rms": rms, "peak": peak, "corr": corr,
            "nadir_a": nadir_a, "nadir_p": nadir_p}


def _center_pre_disturbance(t: np.ndarray, data: np.ndarray, load_t: float, margin: float = 0.02) -> np.ndarray:
    """Return data centered by pre-disturbance mean per channel."""
    t_ref = max(0.0, float(load_t) - float(margin))
    mask = t < t_ref
    if mask.any():
        base = np.mean(data[mask, :], axis=0)
    else:
        base = data[0, :]
    return data - base.reshape(1, -1)


def _compute_trajectory_metrics_on_deviation(
    t_a: np.ndarray,
    data_a: np.ndarray,
    t_p: np.ndarray,
    data_p: np.ndarray,
    load_t_a: float,
    load_t_p: float,
) -> dict:
    """Compute metrics on deviation trajectories (state - pre-disturbance mean)."""
    a_dev = _center_pre_disturbance(t_a, data_a, load_t=load_t_a)
    p_dev = _center_pre_disturbance(t_p, data_p, load_t=load_t_p)
    return _compute_trajectory_metrics(t_a, a_dev, t_p, p_dev)


def plot_full_match(andes_res: dict, pytorch_res: dict, out_dir: Path):
    """Plot comprehensive full-match ANDES vs PyTorch comparison.

    Creates per-state 2x5 grid overlays for ALL available states:
      - Generator: omega (Hz), delta_rel (rad), Eqp, Edp, Efd, Pm, Pvalve
      - Bus: voltage magnitude at gen buses
      - Inverter: PLL xi, PLL am, f_meas, V_meas, P_ord, Q_ord

    Also computes per-state trajectory metrics (RMS, peak, correlation).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    t_a = andes_res["t"]
    omega_a = andes_res["omega"]
    t_p = pytorch_res["t"]
    omega_p = pytorch_res["omega"]
    load_t_a = float(andes_res.get("load_t", 0.1))
    load_t_p = float(pytorch_res.get("load_t", 0.1))

    hz_a = (omega_a - 1.0) * 60.0
    hz_p = (omega_p - 1.0) * 60.0
    n_gen = omega_a.shape[1]
    Sn = andes_res.get("Sn", np.ones(n_gen) * SB)

    all_metrics = {}

    # ===================== GENERATOR STATES =====================

    # 1) Worst-gen frequency comparison
    fig, ax = plt.subplots(figsize=(7, 4))
    worst_a = hz_a.min(axis=1)
    worst_p = hz_p.min(axis=1)
    ax.plot(t_a, worst_a, "b-", label=f"ANDES (nadir {worst_a.min():.4f} Hz)", linewidth=1.5)
    ax.plot(t_p, worst_p, "r--", label=f"PyTorch (nadir {worst_p.min():.4f} Hz)", linewidth=1.5)
    ax.axvline(0.1, color="gray", linestyle="--", alpha=0.5, label="Disturbance")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Worst-gen Df [Hz]")
    ax.set_title("Full-Match ANDES vs PyTorch: Worst Generator Frequency")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "full_match_worst_gen_hz.png", dpi=150)
    plt.close(fig)

    # 2) Per-gen omega (Hz)
    _plot_per_gen_grid(t_a, hz_a, t_p, hz_p, out_dir,
                       "full_match_per_gen_hz",
                       "ANDES vs PyTorch: Generator Frequency", "Df [Hz]")
    all_metrics["omega_hz"] = _compute_trajectory_metrics(t_a, hz_a, t_p, hz_p)
    all_metrics["omega_hz_dev"] = _compute_trajectory_metrics_on_deviation(
        t_a, hz_a, t_p, hz_p, load_t_a=load_t_a, load_t_p=load_t_p
    )

    # 3) Rotor angle delta_rel (rad)
    if "delta_rel" in andes_res and "delta_rel" in pytorch_res:
        _plot_per_gen_grid(t_a, andes_res["delta_rel"], t_p, pytorch_res["delta_rel"],
                           out_dir, "full_match_delta_rel",
                           "ANDES vs PyTorch: Rotor Angle (relative)", "delta_rel [rad]")
        all_metrics["delta_rel"] = _compute_trajectory_metrics(
            t_a, andes_res["delta_rel"], t_p, pytorch_res["delta_rel"])
        all_metrics["delta_rel_dev"] = _compute_trajectory_metrics_on_deviation(
            t_a, andes_res["delta_rel"], t_p, pytorch_res["delta_rel"],
            load_t_a=load_t_a, load_t_p=load_t_p
        )

    # ANDES GENROU stores Eqp, Edp, Efd, Pm, Pvalve in per-unit on bases that
    # are directly comparable to PyTorch system-base values:
    #   - Voltages (Eqp, Edp, Efd): per-unit voltage is base-invariant
    #   - Powers (Pm, Pvalve): ANDES tm/pout algebraic variables are on system
    #     base Sb, verified by swing equation block validation (M_machine × dω
    #     = Pm_sys − Pe_sys − D_machine × (ω−1) passes at 0.37%)
    # NO conversion needed for any of these.
    state_pairs = [
        ("Eqp",    "Eqp",    "Transient EMF E'q",         "E'q [pu]",      None),
        ("Edp",    "Edp",    "Transient EMF E'd",         "E'd [pu]",      None),
        ("Efd",    "Efd",    "Field Voltage Efd",         "Efd [pu]",      None),
        ("Pm",     "Pm",     "Mechanical Power Pm",       "Pm [pu]",       None),
        ("Pvalve", "Pvalve", "Governor Valve Pvalve",     "Pvalve [pu]",   None),
    ]

    for a_key, p_key, title, ylabel, conv_fn in state_pairs:
        if a_key in andes_res and p_key in pytorch_res:
            _plot_per_gen_grid(t_a, andes_res[a_key], t_p, pytorch_res[p_key],
                               out_dir, f"full_match_{a_key.lower()}",
                               f"ANDES vs PyTorch: {title}", ylabel,
                               convert_fn=conv_fn)
            # Compute metrics (no base conversion needed — verified empirically)
            all_metrics[a_key.lower()] = _compute_trajectory_metrics(
                t_a, andes_res[a_key], t_p, pytorch_res[p_key])
            all_metrics[f"{a_key.lower()}_dev"] = _compute_trajectory_metrics_on_deviation(
                t_a, andes_res[a_key], t_p, pytorch_res[p_key],
                load_t_a=load_t_a, load_t_p=load_t_p
            )

    # 9) Bus voltage at generator buses
    if "v_gen_bus" in andes_res and "inv_V_meas" in pytorch_res:
        _plot_per_gen_grid(t_a, andes_res["v_gen_bus"], t_p, pytorch_res["inv_V_meas"],
                           out_dir, "full_match_v_gen_bus",
                           "ANDES Bus V vs PyTorch V_meas", "|V| [pu]")
        all_metrics["v_gen_bus"] = _compute_trajectory_metrics(
            t_a, andes_res["v_gen_bus"], t_p, pytorch_res["inv_V_meas"])
        all_metrics["v_gen_bus_dev"] = _compute_trajectory_metrics_on_deviation(
            t_a, andes_res["v_gen_bus"], t_p, pytorch_res["inv_V_meas"],
            load_t_a=load_t_a, load_t_p=load_t_p
        )

    # ===================== INVERTER STATES =====================

    inv_pairs = [
        ("pll_xi",  "inv_xi_pll", "PLL Integrator xi",          "xi [pu]"),
        ("pll_am",  "inv_phi",    "PLL Angle (am vs phi)",       "angle [rad]"),
    ]
    for a_key, p_key, title, ylabel in inv_pairs:
        if a_key in andes_res and p_key in pytorch_res:
            _plot_per_gen_grid(t_a, andes_res[a_key], t_p, pytorch_res[p_key],
                               out_dir, f"full_match_{a_key}",
                               f"ANDES vs PyTorch: {title}", ylabel)
            all_metrics[a_key] = _compute_trajectory_metrics(
                t_a, andes_res[a_key], t_p, pytorch_res[p_key])
            all_metrics[f"{a_key}_dev"] = _compute_trajectory_metrics_on_deviation(
                t_a, andes_res[a_key], t_p, pytorch_res[p_key],
                load_t_a=load_t_a, load_t_p=load_t_p
            )

    # Inverter frequency: ANDES gen omega vs PyTorch f_meas
    if "inv_f_meas" in pytorch_res:
        f_meas_hz = (pytorch_res["inv_f_meas"] - 1.0) * 60.0
        _plot_per_gen_grid(t_a, hz_a, t_p, f_meas_hz, out_dir,
                           "full_match_inv_freq",
                           "Gen Omega (ANDES) vs Inverter f_meas (PyTorch)", "Df [Hz]")

    # ===================== METRICS SUMMARY =====================

    _plot_mismatch_dashboard(andes_res, pytorch_res, all_metrics, out_dir)
    _print_metrics_summary(all_metrics, out_dir)

    return _compute_stats(hz_a, hz_p), all_metrics


def _plot_mismatch_dashboard(andes_res: dict, pytorch_res: dict, all_metrics: dict, out_dir: Path):
    """Create compact visual diagnostics for current mismatch magnitude."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: state-level mismatch bars (absolute vs deviation) ---
    state_order = [
        "omega_hz",
        "delta_rel",
        "eqp",
        "edp",
        "efd",
        "pm",
        "pvalve",
        "v_gen_bus",
        "pll_am",
    ]
    labels = []
    abs_vals = []
    dev_vals = []
    for st in state_order:
        if st not in all_metrics:
            continue
        st_dev = f"{st}_dev"
        if st_dev not in all_metrics:
            continue
        labels.append(st)
        abs_vals.append(float(np.nanmean(np.asarray(all_metrics[st]["rms"], dtype=float))))
        dev_vals.append(float(np.nanmean(np.asarray(all_metrics[st_dev]["rms"], dtype=float))))

    if labels:
        x = np.arange(len(labels))
        w = 0.38
        fig, ax = plt.subplots(figsize=(max(10.0, 0.9 * len(labels) + 4.0), 5.0))
        ax.bar(x - w / 2, abs_vals, width=w, label="Absolute RMS", color="#2B6CB0")
        ax.bar(x + w / 2, dev_vals, width=w, label="Deviation RMS", color="#D97706")
        ax.set_yscale("log")
        ax.set_ylabel("Mean RMS Error (log scale)")
        ax.set_title("State Mismatch: Absolute vs Deviation")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "mismatch_state_abs_vs_dev.png", dpi=170)
        plt.close(fig)

    # --- Figure 2: per-generator deviation-RMS heatmap ---
    heat_states = [
        "omega_hz_dev",
        "eqp_dev",
        "edp_dev",
        "efd_dev",
        "pm_dev",
        "pvalve_dev",
        "v_gen_bus_dev",
        "pll_am_dev",
    ]
    rows = []
    row_labels = []
    for st in heat_states:
        if st not in all_metrics:
            continue
        arr = np.asarray(all_metrics[st]["rms"], dtype=float).reshape(-1)
        if arr.size != 10:
            continue
        rows.append(arr)
        row_labels.append(st)

    if rows:
        mat = np.vstack(rows)
        fig, ax = plt.subplots(figsize=(12, max(4.0, 0.6 * len(row_labels) + 2.0)))
        im = ax.imshow(mat, aspect="auto", cmap="magma")
        ax.set_title("Per-Generator Deviation RMS (ANDES vs PyTorch)")
        ax.set_xlabel("Generator")
        ax.set_ylabel("State")
        ax.set_xticks(np.arange(10))
        ax.set_xticklabels([f"G{i+1}" for i in range(10)])
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_yticklabels(row_labels)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("RMS error")
        fig.tight_layout()
        fig.savefig(out_dir / "mismatch_dev_heatmap.png", dpi=170)
        plt.close(fig)

    # --- Figure 3: operating-point dispatch mismatch at t=0 ---
    if (
        ("Pm" in andes_res)
        and ("Pm" in pytorch_res)
        and ("regcp1_Pe" in andes_res)
        and ("inv_P_ord" in pytorch_res)
    ):
        a_pm0 = np.asarray(andes_res["Pm"][0, :], dtype=float)
        a_inv0 = np.asarray(andes_res["regcp1_Pe"][0, :], dtype=float)
        p_pm0 = np.asarray(pytorch_res["Pm"][0, :], dtype=float)
        p_inv0 = np.asarray(pytorch_res["inv_P_ord"][0, :], dtype=float)
        a_tot0 = a_pm0 + a_inv0
        p_tot0 = p_pm0 + p_inv0
        diff_tot = a_tot0 - p_tot0

        x = np.arange(10)
        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        axes[0].plot(x, p_tot0, "o-", color="#1D4ED8", label="PyTorch total P @ t0")
        axes[0].plot(x, a_tot0, "s--", color="#B45309", label="ANDES total P @ t0")
        axes[0].set_ylabel("P [pu]")
        axes[0].set_title("Operating-Point Total Power by Generator Bus")
        axes[0].grid(True, alpha=0.25)
        axes[0].legend(fontsize=8)

        axes[1].bar(x, diff_tot, color="#9F1239")
        axes[1].axhline(0.0, color="black", linewidth=0.8)
        axes[1].set_ylabel("ANDES - PyTorch [pu]")
        axes[1].set_xlabel("Generator")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels([f"G{i+1}" for i in range(10)])
        axes[1].grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / "op_dispatch_mismatch.png", dpi=170)
        plt.close(fig)

        op_summary = {
            "pytorch_pm_t0": p_pm0.tolist(),
            "pytorch_inv_t0": p_inv0.tolist(),
            "pytorch_total_t0": p_tot0.tolist(),
            "andes_pm_t0": a_pm0.tolist(),
            "andes_inv_t0": a_inv0.tolist(),
            "andes_total_t0": a_tot0.tolist(),
            "total_mismatch_t0": diff_tot.tolist(),
            "mean_abs_total_mismatch_t0": float(np.mean(np.abs(diff_tot))),
            "max_abs_total_mismatch_t0": float(np.max(np.abs(diff_tot))),
        }
        with open(out_dir / "op_dispatch_mismatch.json", "w") as f:
            json.dump(op_summary, f, indent=2)
        print(
            "[viz] OP mismatch t0 |mean|="
            f"{op_summary['mean_abs_total_mismatch_t0']:.4f}, "
            f"max={op_summary['max_abs_total_mismatch_t0']:.4f}"
        )

    print(f"[viz] Saved mismatch dashboard figures to {out_dir}")


def _print_metrics_summary(all_metrics: dict, out_dir: Path):
    """Print and save comprehensive trajectory comparison metrics."""
    print(f"\n{'='*80}")
    print(f"  COMPREHENSIVE STATE TRAJECTORY METRICS")
    print(f"{'='*80}")
    print(f"  {'State':<14} {'Mean RMS':>10} {'Max RMS':>10} {'Mean Peak':>10} "
          f"{'Min Corr':>10} {'Mean Corr':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    summary_rows = []
    for name, m in all_metrics.items():
        rms = m["rms"]
        peak = m["peak"]
        corr = m["corr"]
        valid = ~np.isnan(rms)
        if not valid.any():
            continue
        row = {
            "state": name,
            "mean_rms": float(np.nanmean(rms)),
            "max_rms": float(np.nanmax(rms)),
            "mean_peak": float(np.nanmean(peak)),
            "min_corr": float(np.nanmin(corr)),
            "mean_corr": float(np.nanmean(corr)),
            "per_gen_rms": rms[valid].tolist(),
            "per_gen_peak": peak[valid].tolist(),
            "per_gen_corr": corr[valid].tolist(),
        }
        if "nadir_a" in m:
            row["nadir_a"] = m["nadir_a"][valid].tolist()
            row["nadir_p"] = m["nadir_p"][valid].tolist()
        summary_rows.append(row)

        print(f"  {name:<14} {row['mean_rms']:>10.6f} {row['max_rms']:>10.6f} "
              f"{row['mean_peak']:>10.6f} {row['min_corr']:>10.4f} {row['mean_corr']:>10.4f}")

    print(f"{'='*80}")

    # Per-gen detail for each state
    for row in summary_rows:
        n = len(row["per_gen_rms"])
        print(f"\n  {row['state']} — per-gen RMS: ", end="")
        for i in range(n):
            bus = GEN_BUSES[i] if i < len(GEN_BUSES) else i + 1
            print(f"G{i+1}={row['per_gen_rms'][i]:.5f}", end="  ")
        print()
        print(f"  {row['state']} — per-gen Corr: ", end="")
        for i in range(n):
            print(f"G{i+1}={row['per_gen_corr'][i]:.4f}", end="  ")
        print()

    # Save to JSON
    with open(out_dir / "trajectory_metrics.json", "w") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\n  Metrics saved to {out_dir / 'trajectory_metrics.json'}")


def _summarize_state_metrics(all_metrics: dict) -> dict:
    """Build compact per-state trajectory mismatch summary."""
    summary = {}
    for name, m in all_metrics.items():
        rms = np.asarray(m.get("rms", []), dtype=float)
        peak = np.asarray(m.get("peak", []), dtype=float)
        corr = np.asarray(m.get("corr", []), dtype=float)
        if rms.size == 0:
            continue
        valid = ~np.isnan(rms)
        if not valid.any():
            continue
        rms_v = rms[valid]
        peak_v = peak[valid]
        corr_v = corr[valid]
        local_idx = int(np.nanargmax(rms_v))
        valid_idx = np.where(valid)[0]
        summary[name] = {
            "mean_rms": float(np.nanmean(rms_v)),
            "max_rms": float(np.nanmax(rms_v)),
            "mean_peak": float(np.nanmean(peak_v)),
            "min_corr": float(np.nanmin(corr_v)),
            "mean_corr": float(np.nanmean(corr_v)),
            "worst_channel_idx": int(valid_idx[local_idx]),
        }
    return summary


def _summarize_state_metrics_excluding_channel(all_metrics: dict, excluded_idx: int) -> dict:
    """Build compact per-state summary excluding one channel index (e.g., slack gen)."""
    summary = {}
    for name, m in all_metrics.items():
        rms = np.asarray(m.get("rms", []), dtype=float).reshape(-1)
        peak = np.asarray(m.get("peak", []), dtype=float).reshape(-1)
        corr = np.asarray(m.get("corr", []), dtype=float).reshape(-1)
        if rms.size == 0:
            continue
        if 0 <= int(excluded_idx) < rms.size:
            keep = np.ones(rms.size, dtype=bool)
            keep[int(excluded_idx)] = False
            rms = rms[keep]
            peak = peak[keep]
            corr = corr[keep]
        valid = ~np.isnan(rms)
        if not valid.any():
            continue
        rms_v = rms[valid]
        peak_v = peak[valid]
        corr_v = corr[valid]
        summary[name] = {
            "mean_rms": float(np.nanmean(rms_v)),
            "max_rms": float(np.nanmax(rms_v)),
            "mean_peak": float(np.nanmean(peak_v)),
            "min_corr": float(np.nanmin(corr_v)),
            "mean_corr": float(np.nanmean(corr_v)),
        }
    return summary


# ---------------------------------------------------------------------------
# Mathematical block-by-block equation validation
# ---------------------------------------------------------------------------

def _numerical_derivative(t, x):
    """Central-difference numerical derivative, same shape as x.

    Uses central difference in interior, forward/backward at boundaries.
    """
    dx = np.zeros_like(x)
    # Interior: central difference
    for i in range(1, len(t) - 1):
        dt = t[i + 1] - t[i - 1]
        if dt > 1e-15:
            dx[i] = (x[i + 1] - x[i - 1]) / dt
    # Boundaries: forward/backward
    dt0 = t[1] - t[0]
    if dt0 > 1e-15:
        dx[0] = (x[1] - x[0]) / dt0
    dtN = t[-1] - t[-2]
    if dtN > 1e-15:
        dx[-1] = (x[-1] - x[-2]) / dtN
    return dx


def validate_equation_blocks(andes_res: dict, pytorch_res: dict, out_dir: Path):
    """Validate each dynamic block by comparing numerical derivatives to ODE RHS.

    For each equation block, compute:
      LHS = numerical derivative of state trajectory (from data)
      RHS = ODE right-hand side (from equation + matched parameters + other states)
      Residual = LHS - RHS  (should be ~0 if equations + params match)

    Blocks validated:
      1. Turbine:   dPm/dt = (Pvalve - Pm) / Tt
      2. Governor:  dPvalve/dt = (Pv_tgt - Pvalve) / Tg,
                    where Pv_tgt = Pref0 + omega_s*(1 - omega)/R  (PyTorch Convention B)
      3. Exciter:   dEfd/dt = (Ka*(Vref - V) - Efd) / Ta
      4. PLL:       d(xi)/dt = Ki_pll * (omega_bus - 1)
                    d(phi)/dt = Kp_pll * (omega_bus - 1) + xi
    """
    from scipy.interpolate import interp1d

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*80}")
    print(f"  MATHEMATICAL BLOCK-BY-BLOCK EQUATION VALIDATION")
    print(f"{'='*80}")

    summary = {}
    n_gen = 10

    # Use post-disturbance window only (skip transient at t=0.1)
    # to avoid derivative artifacts at the step discontinuity
    t_skip = 0.15

    # ========== BLOCK 1: TURBINE  dPm/dt = (Pvalve - Pm) / Tt ==========
    for src_name, res in [("PyTorch", pytorch_res), ("ANDES", andes_res)]:
        if "Pm" not in res or "Pvalve" not in res:
            print(f"  [{src_name}] Turbine block: SKIP (missing Pm or Pvalve)")
            continue

        t = res["t"]
        Pm = res["Pm"]
        Pvalve = res["Pvalve"]
        mask = t >= t_skip
        t_m = t[mask]

        # For ANDES, Pm and Pvalve are on machine base.
        # Turbine equation: Tt * dPm/dt = Pvalve - Pm
        # This holds in ANY consistent base, so no conversion needed.
        Sn = andes_res.get("Sn", np.ones(n_gen) * SB)

        residuals = []
        for gi in range(min(n_gen, Pm.shape[1])):
            dPm_num = _numerical_derivative(t, Pm[:, gi])[mask]
            rhs = (Pvalve[mask, gi] - Pm[mask, gi]) / TT_PY
            resid = dPm_num - rhs
            residuals.append(np.sqrt(np.mean(resid**2)))

        mean_r = np.mean(residuals)
        max_r = np.max(residuals)
        # Normalize by typical Pm magnitude for relative error
        pm_scale = np.mean(np.abs(Pm[mask, :])) + 1e-12
        rel_err = mean_r / pm_scale * 100
        summary[f"turbine_{src_name}"] = {
            "mean_rms_residual": mean_r, "max_rms_residual": max_r,
            "relative_pct": rel_err, "per_gen": residuals,
        }
        status = "PASS" if rel_err < 5.0 else "CHECK"
        print(f"  [{src_name}] Turbine (dPm/dt = (Pv-Pm)/Tt):  "
              f"mean_rms={mean_r:.6f}  rel={rel_err:.2f}%  [{status}]")

    # ========== BLOCK 2: GOVERNOR  dPvalve/dt = (Pv_tgt - Pvalve) / Tg ==========
    # PyTorch: Pv_tgt = Pref0 + omega_s*(1-omega)/R   [system base]
    # ANDES:   Pv_tgt = pref0 - (omega-1)/R            [machine base]
    # Both forms are equivalent if:
    #   R_andes = R_py * Sn / (omega_s * Sb)
    for src_name, res in [("PyTorch", pytorch_res), ("ANDES", andes_res)]:
        if "Pvalve" not in res or "omega" not in res:
            print(f"  [{src_name}] Governor block: SKIP (missing Pvalve or omega)")
            continue

        t = res["t"]
        Pvalve = res["Pvalve"]
        omega = res["omega"]
        mask = t >= t_skip
        t_m = t[mask]
        Sn = andes_res.get("Sn", np.ones(n_gen) * SB)

        residuals = []
        pytorch_mode_votes = {"with_rate_limit": 0, "without_rate_limit": 0}
        for gi in range(min(n_gen, Pvalve.shape[1])):
            dPv_num = _numerical_derivative(t, Pvalve[:, gi])[mask]
            # Estimate Pref0 from initial steady-state value
            ss_mask = t < 0.08
            Pref0 = np.mean(Pvalve[ss_mask, gi]) if ss_mask.any() else Pvalve[0, gi]

            if src_name == "PyTorch":
                # PyTorch Convention B: Pv_tgt = Pref0 + omega_s*(1-omega)/R
                # Then clamped to [0, 1.5*Pref0]. The valve-rate limiter may be
                # enabled or disabled at runtime in class initialization.
                Pv_tgt_raw = Pref0 + OMEGA_S * (1.0 - omega[mask, gi]) / R_PY
                Pv_max = 1.5 * Pref0
                Pv_tgt = np.clip(Pv_tgt_raw, 0.0, Pv_max)
                rhs_norate = (Pv_tgt - Pvalve[mask, gi]) / TG_PY

                # Detect/compare optional rate-limited path.
                max_dPv_up = 2.0
                max_dPv_dn = 2.0
                if "pytorch_max_dPvalve_up" in res:
                    up_arr = np.asarray(res["pytorch_max_dPvalve_up"], dtype=float).reshape(-1)
                    if gi < up_arr.size:
                        max_dPv_up = float(up_arr[gi])
                if "pytorch_max_dPvalve_dn" in res:
                    dn_arr = np.asarray(res["pytorch_max_dPvalve_dn"], dtype=float).reshape(-1)
                    if gi < dn_arr.size:
                        max_dPv_dn = float(dn_arr[gi])

                rhs_rate = np.clip(rhs_norate, -max_dPv_dn, max_dPv_up)
                resid_norate = dPv_num - rhs_norate
                resid_rate = dPv_num - rhs_rate
                rms_norate = float(np.sqrt(np.mean(resid_norate ** 2)))
                rms_rate = float(np.sqrt(np.mean(resid_rate ** 2)))
                if rms_norate <= rms_rate:
                    residuals.append(rms_norate)
                    pytorch_mode_votes["without_rate_limit"] += 1
                else:
                    residuals.append(rms_rate)
                    pytorch_mode_votes["with_rate_limit"] += 1
                continue
            else:
                # ANDES full-match path sets R.v directly on system base:
                #   R.v = R_PY / omega_s (uniform for all generators).
                # Keep legacy Sn-scaled fallback only when target data is absent.
                if "R_andes_target" in andes_res:
                    R_andes_gi = float(np.asarray(andes_res["R_andes_target"])[gi])
                else:
                    R_andes_gi = R_PY * Sn[gi] / (OMEGA_S * SB)
                Pv_tgt = Pref0 - (omega[mask, gi] - 1.0) / R_andes_gi

            rhs = (Pv_tgt - Pvalve[mask, gi]) / TG_PY
            resid = dPv_num - rhs
            residuals.append(np.sqrt(np.mean(resid**2)))

        mean_r = np.mean(residuals)
        max_r = np.max(residuals)
        pv_scale = np.mean(np.abs(Pvalve[mask, :])) + 1e-12
        rel_err = mean_r / pv_scale * 100
        summary[f"governor_{src_name}"] = {
            "mean_rms_residual": mean_r, "max_rms_residual": max_r,
            "relative_pct": rel_err, "per_gen": residuals,
        }
        if src_name == "PyTorch":
            mode = ("with_rate_limit"
                    if pytorch_mode_votes["with_rate_limit"] > pytorch_mode_votes["without_rate_limit"]
                    else "without_rate_limit")
            summary[f"governor_{src_name}"]["detected_mode"] = mode
            summary[f"governor_{src_name}"]["mode_votes"] = pytorch_mode_votes
            mode_note = f" mode={mode} votes={pytorch_mode_votes}"
        else:
            mode_note = ""
        status = "PASS" if rel_err < 5.0 else "CHECK"
        print(f"  [{src_name}] Governor (dPv/dt = (Pv_tgt-Pv)/Tg): "
              f"mean_rms={mean_r:.6f}  rel={rel_err:.2f}%  [{status}]{mode_note}")

    # ========== BLOCK 3: EXCITER  dEfd/dt = (Ka*(Vref - V) - Efd) / Ta ==========
    exc_model = str(andes_res.get("exciter_model", "IEEEX1")).upper()
    if exc_model == "SEXS":
        exc_note = " [SEXS path — should match PyTorch 1st-order form]"
    else:
        exc_note = " [NOTE: IEEEX1 has lead-lag/feedback — expect residual]"
    for src_name, res, note in [
        ("ANDES", andes_res, exc_note),
    ]:
        # Only validate ANDES — PyTorch doesn't extract V_terminal directly
        if "Efd" not in res or "v_gen_bus" not in res:
            print(f"  [{src_name}] Exciter block: SKIP (missing Efd or v_gen_bus)")
            continue

        t = res["t"]
        Efd = res["Efd"]
        V = res["v_gen_bus"]
        mask = t >= t_skip

        residuals = []
        for gi in range(min(n_gen, Efd.shape[1], V.shape[1])):
            dEfd_num = _numerical_derivative(t, Efd[:, gi])[mask]
            # Estimate Vref from initial Efd and V:
            # At steady state: Efd_eq = Ka*(Vref - V_eq) → Vref = V_eq + Efd_eq/Ka
            ss_mask = t < 0.08
            Efd_eq = np.mean(Efd[ss_mask, gi])
            V_eq = np.mean(V[ss_mask, gi])
            Vref = V_eq + Efd_eq / KA_PY

            rhs = (KA_PY * (Vref - V[mask, gi]) - Efd[mask, gi]) / TA_PY
            resid = dEfd_num - rhs
            residuals.append(np.sqrt(np.mean(resid**2)))

        mean_r = np.mean(residuals)
        max_r = np.max(residuals)
        efd_scale = np.mean(np.abs(Efd[mask, :])) + 1e-12
        rel_err = mean_r / efd_scale * 100
        summary[f"exciter_{src_name}"] = {
            "mean_rms_residual": mean_r, "max_rms_residual": max_r,
            "relative_pct": rel_err, "per_gen": residuals,
            "exciter_model": exc_model,
            "note": ("IEEEX1 has additional lead-lag/feedback dynamics" if exc_model != "SEXS"
                     else "SEXS simplified model"),
        }
        if exc_model == "SEXS":
            status = "PASS" if rel_err < 5.0 else "CHECK"
        else:
            status = "EXPECTED" if rel_err > 5.0 else "PASS"
        print(f"  [{src_name}] Exciter (dEfd/dt = (Ka(Vr-V)-Efd)/Ta): "
              f"mean_rms={mean_r:.6f}  rel={rel_err:.2f}%  [{status}]{note}")

    # ========== BLOCK 4: SWING  2H*dω/dt = Pm - Pe - D*ωs*(ω-1) ==========
    # Validate using ANDES data (has Pe), check PyTorch via implied Pe
    for src_name, res in [("ANDES", andes_res)]:
        if "omega" not in res or "Pm" not in res or "Pe" not in res:
            print(f"  [{src_name}] Swing block: SKIP (missing omega, Pm, or Pe)")
            continue

        t = res["t"]
        omega = res["omega"]
        Pm = res["Pm"]
        Pe = res["Pe"]
        mask = t >= t_skip
        Sn = andes_res.get("Sn", np.ones(n_gen) * SB)

        H_eff = np.asarray(andes_res.get("H_eff_target", H_BASE * SG_RATIO), dtype=float)
        D_eff = np.asarray(andes_res.get("D_eff_target", D_BASE * SG_RATIO), dtype=float)

        residuals = []
        for gi in range(min(n_gen, omega.shape[1])):
            dw_num = _numerical_derivative(t, omega[:, gi])[mask]

            # ANDES swing eq (machine base): 2H_a * dω/dt = Pm_a - Pe_a - D_a*(ω-1)
            M_a = 2.0 * H_eff[gi] * SB / Sn[gi]
            D_a = D_eff[gi] * OMEGA_S * SB / Sn[gi]
            lhs = M_a * dw_num
            rhs = Pm[mask, gi] - Pe[mask, gi] - D_a * (omega[mask, gi] - 1.0)
            resid = lhs - rhs
            residuals.append(np.sqrt(np.mean(resid**2)))

        mean_r = np.mean(residuals)
        max_r = np.max(residuals)
        pe_scale = np.mean(np.abs(Pe[mask, :])) + 1e-12
        rel_err = mean_r / pe_scale * 100
        summary[f"swing_{src_name}"] = {
            "mean_rms_residual": mean_r, "max_rms_residual": max_r,
            "relative_pct": rel_err, "per_gen": residuals,
        }
        status = "PASS" if rel_err < 5.0 else "CHECK"
        print(f"  [{src_name}] Swing (2H*dω/dt = Pm-Pe-D*(ω-1)):   "
              f"mean_rms={mean_r:.6f}  rel={rel_err:.2f}%  [{status}]")

    # ========== BLOCK 5: PLL  dξ/dt = Ki*(ω_bus - 1) ==========
    # ANDES PLL2: dPI_xi/dt = Ki_pll * (omega_bus - 1), Ki=2.387 (ANDES per-unit)
    # PyTorch:    dxi_pll/dt = Ki_pll_raw * (f_meas - 1) = 900/omega_s * (f_meas - 1)
    #             But f_meas is filtered frequency, not raw omega.
    # Use inv_f_meas for PyTorch (the signal the PLL integrator sees),
    # and gen omega for ANDES (bus frequency that PLL2 tracks).
    for src_name, res, xi_key, freq_key, ki in [
        ("ANDES",   andes_res,   "pll_xi",     "omega",       KI_PLL),
        ("PyTorch", pytorch_res, "inv_xi_pll", "inv_f_meas",  KI_PLL),
    ]:
        if xi_key not in res or freq_key not in res:
            print(f"  [{src_name}] PLL block: SKIP (missing {xi_key} or {freq_key})")
            continue

        t = res["t"]
        xi = res[xi_key]
        freq = res[freq_key]
        mask = t >= t_skip
        n_inv = xi.shape[1]

        residuals = []
        for ii in range(min(n_inv, freq.shape[1])):
            dxi_num = _numerical_derivative(t, xi[:, ii])[mask]
            omega_err = freq[mask, ii] - 1.0
            rhs = ki * omega_err
            resid = dxi_num - rhs
            residuals.append(np.sqrt(np.mean(resid**2)))

        mean_r = np.mean(residuals)
        max_r = np.max(residuals)
        # PLL xi is very small — use absolute threshold, not relative
        abs_ok = mean_r < 0.01  # absolute RMS residual
        xi_scale = np.mean(np.abs(xi[mask, :])) + 1e-12
        rel_err = mean_r / xi_scale * 100
        summary[f"pll_{src_name}"] = {
            "mean_rms_residual": mean_r, "max_rms_residual": max_r,
            "relative_pct": rel_err, "abs_ok": abs_ok,
            "per_gen": residuals,
        }
        status = "PASS" if abs_ok else "CHECK"
        print(f"  [{src_name}] PLL (dξ/dt = Ki*(ω-1)):              "
              f"mean_rms={mean_r:.6f}  abs_ok={abs_ok}  [{status}]")

    # ========== BLOCK 6: FLUX DECAY  Td0'*dEqp/dt = Efd - Eqp - (Xd-Xd')*Id ==========
    # We don't have Id directly, but we can compute implied (Xd-Xd')*Id as the residual term
    # to verify Td0' is correct (both ANDES and PyTorch should give same implied term)
    for src_name, res in [("ANDES", andes_res)]:
        if "Eqp" not in res or "Efd" not in res:
            print(f"  [{src_name}] Flux decay block: SKIP (missing Eqp or Efd)")
            continue

        t = res["t"]
        Eqp = res["Eqp"]
        Efd = res["Efd"]
        mask = t >= t_skip

        print(f"  [{src_name}] Flux decay: implied_term = Efd - Eqp - Td0'*dEqp/dt")
        # Just report that the equation is present — full validation needs Id
        print(f"  [{src_name}]   (requires Id for full validation — reporting Efd-Eqp range)")
        for gi in range(min(3, Eqp.shape[1])):  # first 3 gens as sample
            efd_eq_diff = Efd[mask, gi] - Eqp[mask, gi]
            print(f"    Gen {gi+1}: Efd-Eqp range [{efd_eq_diff.min():.3f}, {efd_eq_diff.max():.3f}]")

    print(f"{'='*80}")

    # ========== SUMMARY TABLE ==========
    print(f"\n  BLOCK VALIDATION SUMMARY")
    print(f"  {'Block':<30} {'Source':<10} {'Mean RMS':>10} {'Rel%':>8} {'Status':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*10}")
    for key, val in summary.items():
        parts = key.rsplit("_", 1)
        block = parts[0]
        src = parts[1]
        rel = val["relative_pct"]
        if val.get("abs_ok") is not None:
            # PLL: use absolute threshold
            status = "PASS" if val["abs_ok"] else "CHECK"
        elif val.get("exciter_model") == "SEXS":
            status = "PASS" if rel < 5.0 else "CHECK"
        elif val.get("note"):
            # Exciter IEEEX1: expected mismatch against simplified first-order formula.
            status = "EXPECTED" if rel > 5.0 else "PASS"
        else:
            status = "PASS" if rel < 5.0 else ("WARN" if rel < 20 else "CHECK")
        print(f"  {block:<30} {src:<10} {val['mean_rms_residual']:>10.6f} "
              f"{rel:>7.2f}% {status:>10}")
    print()

    # Save
    def _to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, list):
            return [_to_json(x) for x in obj]
        if isinstance(obj, dict):
            return {k2: _to_json(v2) for k2, v2 in obj.items()}
        return obj

    json_summary = _to_json(summary)

    with open(out_dir / "equation_block_validation.json", "w") as f:
        json.dump(json_summary, f, indent=2)
    print(f"  Block validation saved to {out_dir / 'equation_block_validation.json'}")

    return summary


def run_full_match(args):
    """Full parameter match: generators + inverters, all params matched."""
    out_dir = Path(args.output_dir) / "full_match"
    out_dir.mkdir(parents=True, exist_ok=True)

    with_inv = not getattr(args, "no_inverters", False)
    use_reeca = getattr(args, "use_reeca", False)
    linear_control = bool(getattr(args, "linear_control", False))
    lin_kpf = float(getattr(args, "linear_kpf", 10.0)) if linear_control else 0.0
    lin_kvv = float(getattr(args, "linear_kvv", 0.0)) if linear_control else 0.0
    # Default to SEXS (simplified 1st-order exciter) to better match PyTorch.
    use_ieeex1 = getattr(args, "use_ieeex1", False)
    use_sexs = bool(getattr(args, "use_sexs", False)) or (not use_ieeex1)
    tag = "with inverters" if with_inv else "generators only"
    if with_inv and linear_control:
        tag += f" + linear-control(kpf={lin_kpf:.2f},kvv={lin_kvv:.2f})"
        if bool(getattr(args, "andes_enable_linear_qdroop", False)) and abs(lin_kvv) > 1e-12:
            tag += " + ANDES-Qdroop"
    if bool(getattr(args, "andes_disable_gov_limits", False)):
        tag += " + no-gov-limits"
    gov_r_scale = float(getattr(args, "andes_gov_r_scale", 1.0))
    if abs(gov_r_scale - 1.0) > 1e-12:
        tag += f" + govR*{gov_r_scale:.3g}"
    elif with_inv and use_reeca:
        tag += " + REECA1"
    if use_sexs:
        tag += " + SEXS exciter"
    else:
        tag += " + IEEEX1 exciter"
    if not getattr(args, "no_match_reduced_load_profile", False):
        tag += " + reduced-load-profile"
    if getattr(args, "match_shares_from_pytorch", False):
        tag += " + shares-from-PT(eq)"
    if getattr(args, "match_dispatch_from_pytorch", False):
        tag += " + dispatch-from-PT(eq)"
    if getattr(args, "match_reactances", False):
        if getattr(args, "no_repair_equilibrium", False):
            tag += " + react-no-repair"
        else:
            tag += " + react-repair-eq"
    if with_inv and (not getattr(args, "no_reconcile_dispatch", False)) and (not getattr(args, "match_dispatch_from_pytorch", False)):
        tag += " + dispatch-reconcile"
        slack_w = float(getattr(args, "reconcile_slack_weight", 0.0))
        if abs(slack_w) > 1e-12:
            tag += f"(w={slack_w:.2f})"
    if getattr(args, "pytorch_rebuild_equilibrium", False):
        tag += " + PT-rebuild-eq"
    if getattr(args, "pytorch_strict_checkpoint_init", False):
        tag += " + PT-strict-init"
    if getattr(args, "disable_pytorch_gov_rate_limit", False):
        tag += " + PT-gov-rate-off"
    print(f"\n{'='*60}")
    print(f"  FULL PARAMETER MATCH ({tag})")
    print(f"{'='*60}")
    load_bus_physical, load_bus_reduced = _resolve_disturbance_buses(
        args.load_bus, len(REDUCED_BUS_ORDER), getattr(args, "load_bus_mode", "physical")
    )
    print(
        f"[bus-map] mode={args.load_bus_mode}: input={int(args.load_bus)} "
        f"-> physical={int(load_bus_physical)}, reduced={int(load_bus_reduced)}"
    )

    # --- Run PyTorch ---
    pytorch_res = run_pytorch_sim_full(
        load_bus=args.load_bus,
        load_bus_mode=getattr(args, "load_bus_mode", "physical"),
        load_mult_p=args.load_mult_p,
        load_mult_q=args.load_mult_q,
        load_t=args.load_t,
        tf=args.tf,
        device=args.device,
        disable_gov_rate_limit=getattr(args, "disable_pytorch_gov_rate_limit", False),
        linear_kpf=lin_kpf,
        linear_kvv=lin_kvv,
        pytorch_rebuild_equilibrium=getattr(args, "pytorch_rebuild_equilibrium", False),
        pytorch_strict_checkpoint_init=getattr(args, "pytorch_strict_checkpoint_init", False),
    )

    # --- Run ANDES with full match ---
    andes_res = run_andes_full_match(
        load_bus=load_bus_physical,
        load_mult_p=args.load_mult_p,
        load_mult_q=args.load_mult_q,
        load_t=args.load_t,
        tf=args.tf,
        dt=float(getattr(args, "dt_andes", 0.005)),
        load_ramp_tau=float(getattr(args, "andes_load_ramp_tau", 0.02)),
        with_inverters=with_inv,
        use_reeca=use_reeca,
        use_sexs=use_sexs,
        match_reactances=getattr(args, 'match_reactances', False),
        repair_equilibrium=(not getattr(args, "no_repair_equilibrium", False)),
        match_reduced_load_profile=(not getattr(args, "no_match_reduced_load_profile", False)),
        match_shares_from_pytorch=getattr(args, "match_shares_from_pytorch", False),
        match_dispatch_from_pytorch=getattr(args, "match_dispatch_from_pytorch", False),
        reconcile_dispatch=(not getattr(args, "no_reconcile_dispatch", False)),
        reconcile_tol=float(getattr(args, "reconcile_tol", 0.05)),
        reconcile_max_iter=int(getattr(args, "reconcile_max_iter", 6)),
        reconcile_alpha=float(getattr(args, "reconcile_alpha", 0.8)),
        reconcile_max_step=float(getattr(args, "reconcile_max_step", 0.6)),
        reconcile_slack_weight=float(getattr(args, "reconcile_slack_weight", 0.0)),
        disable_pytorch_gov_rate_limit=getattr(args, "disable_pytorch_gov_rate_limit", False),
        linear_control=linear_control,
        linear_kpf=lin_kpf,
        linear_kvv=lin_kvv,
        andes_enable_linear_qdroop=bool(getattr(args, "andes_enable_linear_qdroop", False)),
        andes_disable_gov_limits=bool(getattr(args, "andes_disable_gov_limits", False)),
        andes_gov_r_scale=float(getattr(args, "andes_gov_r_scale", 1.0)),
        pytorch_rebuild_equilibrium=getattr(args, "pytorch_rebuild_equilibrium", False),
        pytorch_strict_checkpoint_init=getattr(args, "pytorch_strict_checkpoint_init", False),
        andes_case=getattr(args, "andes_case", None),
        device=args.device,
    )

    # --- Compare and plot ---
    stats, all_metrics = plot_full_match(andes_res, pytorch_res, out_dir)
    print_stats(stats, label=f"Full-Match Comparison ({tag})")
    state_metric_summary = _summarize_state_metrics(all_metrics)
    gen_buses = [int(b) for b in andes_res.get("gen_buses", GEN_BUSES)]
    slack_bus = 39 if 39 in gen_buses else gen_buses[-1]
    slack_gen_idx = gen_buses.index(slack_bus)
    state_metric_non_slack = _summarize_state_metrics_excluding_channel(
        all_metrics, excluded_idx=slack_gen_idx
    )
    print("\n[state-metrics] ANDES vs PyTorch trajectory mismatch summary")
    print(f"  {'State':<14} {'Mean RMS':>10} {'Max RMS':>10} {'Mean Corr':>10}")
    for s_name, vals in sorted(
        state_metric_summary.items(),
        key=lambda kv: kv[1]["mean_rms"],
        reverse=True,
    ):
        print(
            f"  {s_name:<14} {vals['mean_rms']:>10.6f} {vals['max_rms']:>10.6f} {vals['mean_corr']:>10.4f}"
        )
    print(
        f"\n[state-metrics] Non-slack summary (excluding G{slack_gen_idx + 1}, Bus {slack_bus})"
    )
    for s_name in ["omega_hz_dev", "delta_rel_dev", "eqp_dev", "edp_dev", "efd_dev", "pm_dev", "pvalve_dev"]:
        if s_name in state_metric_non_slack:
            v = state_metric_non_slack[s_name]
            print(
                f"  {s_name:<14} mean_rms={v['mean_rms']:.6f} "
                f"max_rms={v['max_rms']:.6f} mean_corr={v['mean_corr']:.4f}"
            )

    # --- Mathematical block-by-block equation validation ---
    block_summary = validate_equation_blocks(andes_res, pytorch_res, out_dir)

    # --- Save data (all extracted states) ---
    save_dict = {}
    for key, val in andes_res.items():
        if isinstance(val, np.ndarray):
            save_dict[f"andes_{key}"] = val
    for key, val in pytorch_res.items():
        if isinstance(val, np.ndarray):
            save_dict[f"pytorch_{key}"] = val

    np.savez(out_dir / "full_match_data.npz", **save_dict)

    def _maybe_list(value):
        if value is None:
            return None
        return np.asarray(value, dtype=float).tolist()

    # Summary JSON
    summary = {
        "mode": "full_match",
        "with_inverters": with_inv,
        "repair_equilibrium": bool(not getattr(args, "no_repair_equilibrium", False)),
        "load_profile_source": str(andes_res.get("load_profile_source", "andes_native")),
        "reduced_load_profile_matched": bool(not getattr(args, "no_match_reduced_load_profile", False)),
        "share_target_source": str(andes_res.get("share_target_source", "model_structural")),
        "shares_forced_from_pytorch": bool(getattr(args, "match_shares_from_pytorch", False)),
        "dispatch_forced_from_pytorch": bool(andes_res.get("dispatch_forced_from_pytorch", False)),
        "dispatch_reconciled": bool(andes_res.get("dispatch_reconciled", False)),
        "dispatch_reconcile_enabled": bool(not getattr(args, "no_reconcile_dispatch", False)),
        "dispatch_reconcile_slack_weight": float(getattr(args, "reconcile_slack_weight", 0.0)),
        "pytorch_gov_rate_limit_disabled": bool(getattr(args, "disable_pytorch_gov_rate_limit", False)),
        "pytorch_rebuild_equilibrium": bool(getattr(args, "pytorch_rebuild_equilibrium", False)),
        "pytorch_strict_checkpoint_init": bool(getattr(args, "pytorch_strict_checkpoint_init", False)),
        "linear_control": bool(linear_control),
        "linear_kpf": float(lin_kpf),
        "linear_kvv": float(lin_kvv),
        "andes_enable_linear_qdroop": bool(getattr(args, "andes_enable_linear_qdroop", False)),
        "andes_disable_gov_limits": bool(getattr(args, "andes_disable_gov_limits", False)),
        "andes_gov_r_scale": float(getattr(args, "andes_gov_r_scale", 1.0)),
        "load_bus": args.load_bus,
        "load_bus_input": int(args.load_bus),
        "load_bus_mode": str(getattr(args, "load_bus_mode", "physical")),
        "load_bus_physical": int(pytorch_res.get("load_bus_physical", load_bus_physical)),
        "load_bus_pytorch_reduced": int(pytorch_res.get("load_bus_reduced", -1)),
        "load_mult_p": args.load_mult_p,
        "andes_load_ramp_tau": float(getattr(args, "andes_load_ramp_tau", 0.02)),
        "andes_case": str(andes_res.get("andes_case", "")),
        "tf": args.tf,
        "worst_andes_hz": stats["worst_andes"],
        "worst_pytorch_hz": stats["worst_pytorch"],
        "worst_diff_hz": stats["worst_diff_hz"],
        "worst_ratio": stats["worst_ratio"],
        "per_gen": stats["per_gen"],
        "state_metrics": state_metric_summary,
        "slack_bus": int(slack_bus),
        "slack_gen_index": int(slack_gen_idx),
        "state_metrics_non_slack": state_metric_non_slack,
        "params_matched": {
            "H_eff": np.asarray(andes_res.get("H_eff_target", H_BASE * SG_RATIO), dtype=float).tolist(),
            "D_eff": np.asarray(andes_res.get("D_eff_target", D_BASE * SG_RATIO), dtype=float).tolist(),
            "sg_share": np.asarray(andes_res.get("sg_share_target", SG_RATIO), dtype=float).tolist(),
            "pv_share": np.asarray(andes_res.get("pv_share_target", PV_RATIO), dtype=float).tolist(),
            "p_total_target": _maybe_list(andes_res.get("p_total_target")),
            "pref0_target": _maybe_list(andes_res.get("pref0_target")),
            "p_inv_target": _maybe_list(andes_res.get("p_inv_target")),
            "Ka": KA_PY, "Ta": TA_PY,
            "R": R_PY, "Tg": TG_PY, "Tt": TT_PY,
            "linear_kpf": float(lin_kpf),
            "linear_kvv": float(lin_kvv),
            "ZIP": "80/10/10",
            "PLL": {"Kp": KP_PLL, "Ki": KI_PLL},
        },
        "dispatch_reconcile_summary": andes_res.get("dispatch_reconcile_summary"),
    }
    with open(out_dir / "full_match_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[done] Full-match results saved to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("--load-bus", type=int, default=18,
                   help="Disturbance bus value (interpretation is controlled by --load-bus-mode)")
    p.add_argument("--load-bus-mode", type=str, default="physical",
                   choices=["physical", "reduced-index"],
                   help=("Interpret --load-bus as a retained IEEE physical bus "
                         "or as reduced-model index [0..26]."))
    p.add_argument("--load-mult-p", type=float, default=1.5)
    p.add_argument("--load-mult-q", type=float, default=1.5)
    p.add_argument("--load-t", type=float, default=0.1)
    p.add_argument("--dt-andes", type=float, default=0.005,
                   help="With --full-match: ANDES TDS time step [s] (PyTorch stays at 0.001s)")
    p.add_argument("--andes-load-ramp-tau", type=float, default=0.02,
                   help="With --full-match: disturbance ramp time constant [s] for ANDES load change (match PyTorch AlterManager tau)")
    p.add_argument("--tf", type=float, default=10.0)
    p.add_argument("--output-dir", default=str(CODE_DIR / ".." / "data" / "andes_validation"))
    p.add_argument("--andes-case", type=str, default="",
                   help="Optional ANDES case path (.xlsx). Default: built-in ieee39 full case")
    p.add_argument("--incremental", action="store_true",
                   help="Run incremental parameter matching (Steps 0-5)")
    p.add_argument("--full-match", action="store_true",
                   help="Full parameter match with generators + inverters")
    p.add_argument("--no-inverters", action="store_true",
                   help="With --full-match: match generators only, no inverters")
    p.add_argument("--use-reeca", action="store_true",
                   help="With --full-match: add REECA1 controller (has init issues)")
    p.add_argument("--use-sexs", action="store_true",
                   help="With --full-match: force SEXS exciter (default behavior)")
    p.add_argument("--use-ieeex1", action="store_true",
                   help="With --full-match: use native IEEEX1 exciters instead of default SEXS")
    p.add_argument("--match-reactances", action="store_true",
                   help="With --full-match: also match GENROU reactances/time constants to Pai 1989")
    p.add_argument("--no-repair-equilibrium", action="store_true",
                   help="With --full-match --match-reactances: disable GENROU transient-pair equilibrium repair (Xq'=Xd')")
    p.add_argument("--no-match-reduced-load-profile", action="store_true",
                   help="With --full-match: keep native ANDES PQ base loads (default is to match PyTorch reduced-load profile)")
    p.add_argument("--match-shares-from-pytorch", action="store_true",
                   help="With --full-match: force SG/PV shares to PyTorch equilibrium (off by default)")
    p.add_argument("--match-dispatch-from-pytorch", action="store_true",
                   help="With --full-match: force ANDES PV/Slack dispatch (p0) to PyTorch equilibrium targets")
    p.add_argument("--no-reconcile-dispatch", action="store_true",
                   help="With --full-match: disable iterative non-slack dispatch reconciliation (enabled by default)")
    p.add_argument("--reconcile-tol", type=float, default=0.05,
                   help="With --full-match: reconciliation stop tolerance on max non-slack residual [pu]")
    p.add_argument("--reconcile-max-iter", type=int, default=6,
                   help="With --full-match: max reconciliation iterations")
    p.add_argument("--reconcile-alpha", type=float, default=0.8,
                   help="With --full-match: reconciliation step gain")
    p.add_argument("--reconcile-max-step", type=float, default=0.6,
                   help="With --full-match: max PV p0 adjustment per iteration [pu]")
    p.add_argument("--reconcile-slack-weight", type=float, default=0.0,
                   help="With --full-match: distribute slack residual into non-slack PV updates (0=off)")
    p.add_argument("--linear-control", action="store_true",
                   help="Enable linear inverter droop control for BOTH PyTorch and ANDES (REECA1E vs LinearDroop)")
    p.add_argument("--linear-kpf", type=float, default=10.0,
                   help="Active-frequency droop gain for --linear-control")
    p.add_argument("--linear-kvv", type=float, default=0.0,
                   help="Reactive-voltage droop gain for --linear-control (default 0 because ANDES Q-droop mapping is disabled in this comparison path)")
    p.add_argument("--andes-enable-linear-qdroop", action="store_true",
                   help="With --full-match --linear-control: enable REECA1E V-Q path (map kvv->Kqv). Off by default for init robustness.")
    p.add_argument("--andes-disable-gov-limits", action="store_true",
                   help="With --full-match: disable TGOV1N VMAX/VMIN clamping (diagnostic).")
    p.add_argument("--andes-gov-r-scale", type=float, default=1.0,
                   help="With --full-match: multiplier on ANDES droop mapping R=(R_py/omega_s)*scale.")
    p.add_argument("--pytorch-rebuild-equilibrium", action="store_true",
                   help="Rebuild/repair PyTorch equilibrium after loading checkpoint (off by default to preserve saved behavior)")
    p.add_argument("--pytorch-strict-checkpoint-init", action="store_true",
                   help="Use PyTorch checkpoint exactly as saved (skip runtime retuning/rebuild)")
    p.add_argument("--disable-pytorch-gov-rate-limit", action="store_true",
                   help="Disable PyTorch governor valve slew-rate limiter during comparisons")
    p.add_argument("--steps", type=str, default=None,
                   help="Comma-separated step indices to run (e.g. '0,1,2'). Default: all.")
    args = p.parse_args()

    if args.pytorch_strict_checkpoint_init and args.pytorch_rebuild_equilibrium:
        print("[pytorch] Strict checkpoint init requested; ignoring --pytorch-rebuild-equilibrium.")
        args.pytorch_rebuild_equilibrium = False

    if args.full_match:
        run_full_match(args)
    elif args.incremental:
        run_incremental(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
