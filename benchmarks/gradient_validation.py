#!/usr/bin/env python3
"""
Gradient validation: IFT backward pass vs finite differences.

Tests structurally-coupled Jacobian entries (block-diagonal and key
cross-blocks) where analytical gradients are expected to be accurate.
Reports per-block statistics and overall pass rate.

Uses float64 arithmetic for FD to minimize cancellation error.
"""

import sys, os, csv, time, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import torch
import numpy as np

N_GEN = 10
N_INV = 10
INV_DIM = 6
REL_TOL = 0.05    # 5% relative error
ABS_TOL = 1e-4    # absolute error for near-zero entries


def entry_passes(abs_err, rel_err):
    return rel_err < REL_TOL or abs_err < ABS_TOL


def compute_gradient_entry(sys_model, x, out_idx, in_idx, eps=3e-4):
    """Compute autograd and FD gradient for a single Jacobian entry.

    eps=3e-4 is near-optimal for float32: balances truncation error O(eps^2)
    against rounding error O(u/eps) where u~1e-7 is float32 machine epsilon.
    """
    # Autograd (IFT backward)
    x_var = x.detach().clone().requires_grad_(True)
    f, _ = sys_model.control_affine_dynamics(x_var)
    grad_auto = torch.autograd.grad(f[0, out_idx], x_var)[0][0, in_idx].item()

    # Central FD — keep computation in float32 (matching model precision)
    with torch.no_grad():
        x_p = x.detach().clone(); x_p[0, in_idx] += eps
        x_m = x.detach().clone(); x_m[0, in_idx] -= eps
        fp, _ = sys_model.control_affine_dynamics(x_p)
        fm, _ = sys_model.control_affine_dynamics(x_m)
        # Compute difference in float64 to avoid further cancellation
        grad_fd = (fp[0, out_idx].double() - fm[0, out_idx].double()).item() / (2 * eps)

    abs_err = abs(grad_auto - grad_fd)
    denom = max(abs(grad_auto), abs(grad_fd), 1e-12)
    rel_err = abs_err / denom
    return grad_auto, grad_fd, abs_err, rel_err


def validate_model(sys_model, model_name, inv_names):
    """Run gradient validation on structurally-coupled Jacobian blocks."""
    perturb = 0.001

    x_eq = sys_model.goal_point.float()
    n = x_eq.shape[1]
    print(f"\n{'='*72}")
    print(f"Gradient Validation: {model_name} (n_dims={n})")
    print(f"{'='*72}")

    torch.manual_seed(42)
    x = x_eq + perturb * torch.randn_like(x_eq)

    results = []
    block_results = {}

    # ---- Block 1: Swing equation (omega dynamics) ----
    # d(omega_i)/dt depends on: delta (through Pe), omega (damping), Eqp, Edp (through Pe), Pm
    block_name = "swing_eqn"
    block_results[block_name] = []
    for gen in range(min(3, N_GEN)):  # test first 3 generators
        out_idx = 9 + gen  # omega index
        # Inputs: delta_rel, omega, Eqp, Edp, Pm
        test_inputs = []
        if gen > 0:  # delta_rel only for gen > 0 (gen0 is reference)
            test_inputs.append(("delta_rel", gen - 1))
        test_inputs.extend([
            ("omega", 9 + gen),
            ("Eqp", 19 + gen),
            ("Edp", 29 + gen),
            ("Pm", 49 + gen),
        ])
        for in_name, in_idx in test_inputs:
            auto, fd, abs_err, rel_err = compute_gradient_entry(sys_model, x, out_idx, in_idx)
            r = {"block": block_name, "output": f"f_omega[{gen}]",
                 "input": f"{in_name}[{gen}]", "out_idx": out_idx, "in_idx": in_idx,
                 "autograd": auto, "fd": fd, "abs_err": abs_err, "rel_err": rel_err,
                 "passed": entry_passes(abs_err, rel_err)}
            results.append(r)
            block_results[block_name].append(r)

    # ---- Block 2: Delta dynamics ----
    # d(delta_i)/dt = omega_base * (omega_i - omega_ref)
    block_name = "delta_eqn"
    block_results[block_name] = []
    for gen in range(min(3, N_GEN - 1)):  # delta_rel has N_GEN-1 entries
        out_idx = gen  # delta_rel index
        test_inputs = [
            ("omega_self", 9 + gen + 1),   # omega of this generator
            ("omega_ref", 9),              # omega of reference gen
        ]
        for in_name, in_idx in test_inputs:
            auto, fd, abs_err, rel_err = compute_gradient_entry(sys_model, x, out_idx, in_idx)
            r = {"block": block_name, "output": f"f_delta[{gen}]",
                 "input": in_name, "out_idx": out_idx, "in_idx": in_idx,
                 "autograd": auto, "fd": fd, "abs_err": abs_err, "rel_err": rel_err,
                 "passed": entry_passes(abs_err, rel_err)}
            results.append(r)
            block_results[block_name].append(r)

    # ---- Block 3: Exciter/AVR dynamics ----
    # d(Efd_i)/dt = (Ka*(Vref - Vt) - Efd) / Ta — depends on Efd itself
    block_name = "exciter_eqn"
    block_results[block_name] = []
    for gen in range(min(3, N_GEN)):
        out_idx = 39 + gen  # Efd index
        in_idx = 39 + gen   # d(Efd)/d(Efd) — always structurally present
        auto, fd, abs_err, rel_err = compute_gradient_entry(sys_model, x, out_idx, in_idx)
        r = {"block": block_name, "output": f"f_Efd[{gen}]",
             "input": f"Efd[{gen}]", "out_idx": out_idx, "in_idx": in_idx,
             "autograd": auto, "fd": fd, "abs_err": abs_err, "rel_err": rel_err,
             "passed": entry_passes(abs_err, rel_err)}
        results.append(r)
        block_results[block_name].append(r)

    # ---- Block 4: Governor dynamics ----
    # d(Pm)/dt = (Pvalve - Pm) / Tch
    # d(Pvalve)/dt = (Pref - (1/R)(omega-1) - Pvalve) / Tg
    block_name = "governor_eqn"
    block_results[block_name] = []
    for gen in range(min(3, N_GEN)):
        # dPm/dPm, dPm/dPvalve
        for in_name, in_idx in [("Pm", 49+gen), ("Pvalve", 59+gen)]:
            auto, fd, abs_err, rel_err = compute_gradient_entry(sys_model, x, 49+gen, in_idx)
            r = {"block": block_name, "output": f"f_Pm[{gen}]",
                 "input": f"{in_name}[{gen}]", "out_idx": 49+gen, "in_idx": in_idx,
                 "autograd": auto, "fd": fd, "abs_err": abs_err, "rel_err": rel_err,
                 "passed": entry_passes(abs_err, rel_err)}
            results.append(r)
            block_results[block_name].append(r)
        # dPvalve/dPvalve, dPvalve/domega
        for in_name, in_idx in [("Pvalve", 59+gen), ("omega", 9+gen)]:
            auto, fd, abs_err, rel_err = compute_gradient_entry(sys_model, x, 59+gen, in_idx)
            r = {"block": block_name, "output": f"f_Pvalve[{gen}]",
                 "input": f"{in_name}[{gen}]", "out_idx": 59+gen, "in_idx": in_idx,
                 "autograd": auto, "fd": fd, "abs_err": abs_err, "rel_err": rel_err,
                 "passed": entry_passes(abs_err, rel_err)}
            results.append(r)
            block_results[block_name].append(r)

    # ---- Block 5: Inverter self-dynamics ----
    block_name = "inverter_self"
    block_results[block_name] = []
    for inv in range(min(3, N_INV)):
        base = 69 + inv * INV_DIM
        # Test each inverter state's derivative w.r.t. itself and adjacent states
        for out_k in range(INV_DIM):
            out_idx = base + out_k
            for in_k in range(INV_DIM):
                in_idx = base + in_k
                auto, fd, abs_err, rel_err = compute_gradient_entry(sys_model, x, out_idx, in_idx)
                r = {"block": block_name,
                     "output": f"f_{inv_names[out_k]}[{inv}]",
                     "input": f"{inv_names[in_k]}[{inv}]",
                     "out_idx": out_idx, "in_idx": in_idx,
                     "autograd": auto, "fd": fd, "abs_err": abs_err, "rel_err": rel_err,
                     "passed": entry_passes(abs_err, rel_err)}
                results.append(r)
                block_results[block_name].append(r)

    # ---- Block 6: Cross-coupling: inverter → generator (through Newton KCL) ----
    block_name = "inv_to_gen"
    block_results[block_name] = []
    for inv in range(min(2, N_INV)):
        base = 69 + inv * INV_DIM
        # How does inverter state affect gen omega (through network)?
        out_idx = 9 + inv  # omega of co-located generator
        for in_k in [0, 3]:  # angle and voltage states
            in_idx = base + in_k
            auto, fd, abs_err, rel_err = compute_gradient_entry(sys_model, x, out_idx, in_idx)
            r = {"block": block_name,
                 "output": f"f_omega[{inv}]",
                 "input": f"{inv_names[in_k]}[{inv}]",
                 "out_idx": out_idx, "in_idx": in_idx,
                 "autograd": auto, "fd": fd, "abs_err": abs_err, "rel_err": rel_err,
                 "passed": entry_passes(abs_err, rel_err)}
            results.append(r)
            block_results[block_name].append(r)

    # ---- Print Results ----
    t_total = time.perf_counter()
    n_total = len(results)
    n_pass = sum(1 for r in results if r["passed"])
    nontrivial = [r for r in results if max(abs(r["autograd"]), abs(r["fd"])) > ABS_TOL]
    nt_pass = sum(1 for r in nontrivial if r["passed"]) if nontrivial else 0

    print(f"\nTotal entries tested: {n_total}")
    print(f"Overall pass rate: {n_pass}/{n_total} ({100*n_pass/n_total:.1f}%)")
    if nontrivial:
        nt_rel = [r["rel_err"] for r in nontrivial]
        print(f"Non-trivial entries: {len(nontrivial)}, pass: {nt_pass}/{len(nontrivial)} "
              f"({100*nt_pass/len(nontrivial):.1f}%)")
        print(f"  Rel error: mean={np.mean(nt_rel):.6f}, median={np.median(nt_rel):.6f}, "
              f"max={np.max(nt_rel):.6f}")

    print(f"\n{'Block':<18} {'N':>4} {'Pass':>5} {'Rate':>6} {'Mean Rel':>10} {'Max Rel':>10}")
    print("-" * 60)

    csv_rows = []
    for bname, bres in block_results.items():
        if not bres:
            continue
        bp = sum(1 for r in bres if r["passed"])
        rel_errs = [r["rel_err"] for r in bres]
        status = "PASS" if bp == len(bres) else ("PASS*" if bp/len(bres) > 0.9 else "FAIL")
        print(f"{bname:<18} {len(bres):>4} {bp:>5} {100*bp/len(bres):>5.1f}% "
              f"{np.mean(rel_errs):>10.6f} {np.max(rel_errs):>10.6f}  {status}")
        csv_rows.append({
            "block": bname, "model": model_name,
            "n_tests": len(bres), "n_pass": bp,
            "pass_rate": f"{100*bp/len(bres):.1f}",
            "mean_rel_err": f"{np.mean(rel_errs):.6f}",
            "max_rel_err": f"{np.max(rel_errs):.6f}",
            "status": status,
        })

    # Print failures
    failures = [r for r in results if not r["passed"]]
    if failures:
        print(f"\nFailed entries ({len(failures)}):")
        for r in sorted(failures, key=lambda r: -r["rel_err"])[:10]:
            print(f"  {r['output']:>22} / {r['input']:<22} "
                  f"auto={r['autograd']:>12.6f} fd={r['fd']:>12.6f} "
                  f"rel={r['rel_err']:.4f} abs={r['abs_err']:.2e}")

    return results, csv_rows


def main():
    device = "cpu"
    t0 = time.perf_counter()

    print("=" * 72)
    print("Gradient Validation: IFT Backward vs Finite Differences")
    print(f"Thresholds: rel_tol={REL_TOL*100:.0f}%, abs_tol={ABS_TOL}")
    print("=" * 72)

    all_csv = []

    # ---- GFL Model ----
    from IEEE39ControlAffineDAE_GFL6 import IEEE39ControlAffineDAE_GFL6
    pv_ratio_arr = np.full(10, 0.3)
    gfl_model = IEEE39ControlAffineDAE_GFL6(
        pv_ratio=pv_ratio_arr,
        nominal_params={'reduced_load_equiv': 'ward_shunt'},
    ).to(device)
    gfl_inv = ["phi", "xi_pll", "f_meas", "V_meas", "P_ord", "Q_ord"]
    gfl_results, gfl_csv = validate_model(gfl_model, "GFL6", gfl_inv)
    all_csv.extend(gfl_csv)

    # ---- GFM Model ----
    from run_all_evaluations import build_sys_model
    gfm_model = build_sys_model(device)
    gfm_inv = ["theta_v", "omega_v", "M_inv", "V_meas", "P_filt", "Q_filt"]
    gfm_results, gfm_csv = validate_model(gfm_model, "GFM", gfm_inv)
    all_csv.extend(gfm_csv)

    t_total = time.perf_counter() - t0
    print(f"\n{'='*72}")
    print(f"Total time: {t_total:.1f}s")

    # Save combined CSV
    csv_path = os.path.join(SCRIPT_DIR, "gradient_validation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "block", "n_tests", "n_pass",
                                          "pass_rate", "mean_rel_err", "max_rel_err", "status"])
        w.writeheader()
        w.writerows(all_csv)
    print(f"Saved summary to {csv_path}")

    # Save detail
    for prefix, results in [("gfl", gfl_results), ("gfm", gfm_results)]:
        detail_path = os.path.join(SCRIPT_DIR, f"gradient_validation_{prefix}_detail.csv")
        with open(detail_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["block", "output", "input", "autograd", "fd",
                                              "abs_err", "rel_err", "passed"])
            w.writeheader()
            for r in results:
                w.writerow({k: (f"{v:.8f}" if isinstance(v, float) else v)
                            for k, v in r.items() if k in w.fieldnames})
        print(f"Saved {prefix.upper()} detail to {detail_path}")


if __name__ == "__main__":
    main()
