#!/usr/bin/env python3
"""
Minimal example: build the differentiable DAE simulator, compute dynamics
with gradients, and run a short time-domain simulation.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np

# ---- 1. Build the system model ----

from IEEE39ControlAffineDAE_GFL6 import IEEE39ControlAffineDAE_GFL6

pv_ratio = np.full(10, 0.3)  # 30% PV penetration at each bus
sys_model = IEEE39ControlAffineDAE_GFL6(
    pv_ratio=pv_ratio,
    nominal_params={'reduced_load_equiv': 'ward_shunt'},
)

x_eq = sys_model.goal_point.float()  # equilibrium: (1, 129)
n_dims = x_eq.shape[1]
n_controls = 2 * sys_model.n_inv  # P_ref + V_ref per inverter

print(f"System: {n_dims} states, {n_controls} controls")
print(f"  Generators: {sys_model.n_gen}")
print(f"  Inverters:  {sys_model.n_inv}")
print(f"  Buses:      {sys_model.n_bus}")

# ---- 2. Compute control-affine dynamics with gradients ----

x = x_eq.clone().requires_grad_(True)
f, g = sys_model.control_affine_dynamics(x)
# f: (1, 129) -- drift dynamics dx/dt when u=0
# g: (1, 129, 20) -- control input matrix

print(f"\nDrift f(x_eq): norm = {f.norm():.6f}")
print(f"Control g(x_eq): shape = {g.shape}")

# Backpropagate a scalar loss through the DAE solver
omega = f[:, 9:19]  # generator speed derivatives
loss = omega.pow(2).sum()
loss.backward()
print(f"Gradient d(loss)/dx: norm = {x.grad.norm():.6f}")

# ---- 3. Short time-domain simulation (open-loop, no control) ----

dt = 0.001  # 1 ms timestep
T = 0.5     # simulate 0.5 seconds
n_steps = int(T / dt)

# Start with a small perturbation from equilibrium
torch.manual_seed(42)
x_sim = x_eq.detach().clone()
x_sim[0, 9] += 0.001  # perturb omega_0 by 0.1%

omega_trace = []

with torch.no_grad():
    for step in range(n_steps):
        f, g = sys_model.control_affine_dynamics(x_sim)
        u = torch.zeros(1, n_controls)  # zero control input
        dxdt = f + (g @ u.unsqueeze(-1)).squeeze(-1)
        x_sim = x_sim + dt * dxdt

        if step % 50 == 0:
            omega_hz = (x_sim[0, 9:19] - 1.0) * 60.0  # deviation in Hz
            omega_trace.append(omega_hz.numpy().copy())

omega_trace = np.array(omega_trace)
t_trace = np.arange(len(omega_trace)) * 50 * dt

print(f"\nSimulated {n_steps} steps ({T}s)")
print(f"Final max freq deviation: {np.abs(omega_trace[-1]).max():.4f} Hz")

# ---- 4. Optional: plot ----

try:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for gen in range(10):
        ax.plot(t_trace, omega_trace[:, gen] * 1000, alpha=0.7,
                label=f"G{gen+1}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency deviation (mHz)")
    ax.set_title("Open-loop frequency response (no control)")
    ax.legend(ncol=5, fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "quick_start_plot.png")
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
except ImportError:
    print("matplotlib not available, skipping plot")
