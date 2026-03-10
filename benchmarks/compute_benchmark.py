#!/usr/bin/env python3
"""
Computational performance benchmark for the differentiable DAE simulator.

Measures wall-clock time for:
  - Forward pass (Newton KCL solve + dynamics)
  - Forward + backward pass (IFT gradient computation)
  - Batch scaling (1, 8, 32, 128 samples)
  - CPU vs GPU (if available)
  - Newton convergence statistics
"""

import sys, os, csv, time, warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

import torch
import numpy as np

BATCH_SIZES = [1, 8, 32, 128]
N_WARMUP = 3
N_TRIALS = 10


def benchmark_model(model_cls, model_name, model_kwargs, device, batch_sizes):
    """Benchmark forward and forward+backward for various batch sizes."""
    print(f"\n{'='*60}")
    print(f"Model: {model_name}, Device: {device}")
    print(f"{'='*60}")

    results = []

    for B in batch_sizes:
        # Build model fresh
        sys_model = model_cls(**model_kwargs).to(device)
        x_eq = sys_model.goal_point.float().to(device)
        n = x_eq.shape[1]

        # Batch the equilibrium point with small perturbation
        torch.manual_seed(42)
        x_batch = x_eq.expand(B, -1) + 0.001 * torch.randn(B, n, device=device)

        # ---- Forward only ----
        for _ in range(N_WARMUP):
            with torch.no_grad():
                f, g = sys_model.control_affine_dynamics(x_batch)
        if device != "cpu":
            torch.cuda.synchronize()

        fwd_times = []
        for _ in range(N_TRIALS):
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                f, g = sys_model.control_affine_dynamics(x_batch)
            if device != "cpu":
                torch.cuda.synchronize()
            fwd_times.append(time.perf_counter() - t0)

        fwd_mean = np.mean(fwd_times) * 1000  # ms
        fwd_std = np.std(fwd_times) * 1000

        # ---- Forward + Backward ----
        for _ in range(N_WARMUP):
            x_var = x_batch.detach().clone().requires_grad_(True)
            f, g = sys_model.control_affine_dynamics(x_var)
            loss = f.sum()
            loss.backward()

        if device != "cpu":
            torch.cuda.synchronize()

        fwd_bwd_times = []
        for _ in range(N_TRIALS):
            if device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            x_var = x_batch.detach().clone().requires_grad_(True)
            f, g = sys_model.control_affine_dynamics(x_var)
            loss = f.sum()
            loss.backward()
            if device != "cpu":
                torch.cuda.synchronize()
            fwd_bwd_times.append(time.perf_counter() - t0)

        fb_mean = np.mean(fwd_bwd_times) * 1000
        fb_std = np.std(fwd_bwd_times) * 1000

        overhead = (fb_mean / fwd_mean - 1) * 100 if fwd_mean > 0 else 0

        print(f"  B={B:>4}: fwd={fwd_mean:>7.2f}±{fwd_std:.2f}ms  "
              f"fwd+bwd={fb_mean:>7.2f}±{fb_std:.2f}ms  "
              f"overhead={overhead:>5.1f}%  "
              f"per-sample fwd={fwd_mean/B:.3f}ms")

        results.append({
            "model": model_name,
            "device": device,
            "batch_size": B,
            "fwd_ms": f"{fwd_mean:.2f}",
            "fwd_std_ms": f"{fwd_std:.2f}",
            "fwd_bwd_ms": f"{fb_mean:.2f}",
            "fwd_bwd_std_ms": f"{fb_std:.2f}",
            "bwd_overhead_pct": f"{overhead:.1f}",
            "per_sample_fwd_ms": f"{fwd_mean/B:.3f}",
        })

        del sys_model, x_batch
        if device != "cpu":
            torch.cuda.empty_cache()

    return results


def newton_convergence_stats(model_cls, model_name, model_kwargs, device):
    """Collect Newton convergence statistics over many samples."""
    print(f"\n{'='*60}")
    print(f"Newton Convergence: {model_name}")
    print(f"{'='*60}")

    sys_model = model_cls(**model_kwargs).to(device)
    x_eq = sys_model.goal_point.float().to(device)
    n = x_eq.shape[1]

    # Run 100 different perturbation levels
    all_iters = []
    all_resids = []
    perturbation_levels = [0.001, 0.005, 0.01, 0.05]

    for perturb in perturbation_levels:
        torch.manual_seed(42)
        x = x_eq.expand(32, -1) + perturb * torch.randn(32, n, device=device)
        with torch.no_grad():
            f, _ = sys_model.control_affine_dynamics(x)

        info = sys_model.get_newton_convergence_info() if hasattr(sys_model, 'get_newton_convergence_info') else None
        if info:
            iters = info.get('iterations', 0)
            resid = info.get('residual', 0)
            all_iters.append(iters)
            all_resids.append(resid)
            try:
                resid_val = float(resid)
                print(f"  perturb={perturb}: iters={iters}, resid={resid_val:.2e}")
            except (TypeError, ValueError):
                print(f"  perturb={perturb}: iters={iters}, resid={resid}")
        else:
            print(f"  perturb={perturb}: convergence info not available")

    return all_iters, all_resids


def main():
    t0_total = time.perf_counter()

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU available, benchmarking CPU only")

    all_results = []

    # GFL model
    from IEEE39ControlAffineDAE_GFL6 import IEEE39ControlAffineDAE_GFL6
    gfl_kwargs = {
        'pv_ratio': np.full(10, 0.3),
        'nominal_params': {'reduced_load_equiv': 'ward_shunt'},
    }

    # GFM model
    from IEEE39ControlAffineDAE_GFM import IEEE39ControlAffineDAE_GFM
    gfm_kwargs = {
        'pv_ratio': np.full(10, 0.3),
        'nominal_params': {
            'reduced_load_equiv': 'ward_shunt',
            'M_0': 5.0, 'k_m': 2.0, 'tau_m': 0.1,
            'D_v': 20.0, 'M_max': 15.0,
            'u_max': 0.5, 'u_min': -0.5,
        },
    }

    models = [
        (IEEE39ControlAffineDAE_GFL6, "GFL6", gfl_kwargs),
        (IEEE39ControlAffineDAE_GFM, "GFM", gfm_kwargs),
    ]

    for device in devices:
        for model_cls, model_name, model_kwargs in models:
            results = benchmark_model(model_cls, model_name, model_kwargs, device, BATCH_SIZES)
            all_results.extend(results)

    # Newton convergence stats (CPU only, GFM model)
    for model_cls, model_name, model_kwargs in models:
        newton_convergence_stats(model_cls, model_name, model_kwargs, "cpu")

    # Save results
    csv_path = os.path.join(SCRIPT_DIR, "compute_benchmark_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "device", "batch_size",
                                          "fwd_ms", "fwd_std_ms",
                                          "fwd_bwd_ms", "fwd_bwd_std_ms",
                                          "bwd_overhead_pct", "per_sample_fwd_ms"])
        w.writeheader()
        w.writerows(all_results)
    print(f"\nSaved to {csv_path}")
    print(f"Total benchmark time: {time.perf_counter() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
