# Differentiable DAE Simulator for Mixed SG/Inverter Power Systems

A fully differentiable DAE (differential-algebraic equation) simulator for the IEEE 39-bus power system with synchronous generators, grid-following (GFL) inverters, and grid-forming (GFM) inverters. Provides analytic Jacobians and implicit function theorem (IFT) backward pass for gradient-based neural controller training.

**Paper:** H. Alduaij  "High-Fidelity Differentiable DAE Simulation of Mixed Synchronous-Generator and Inverter-Based Power Systems," arXiv preprint, 2025. To be submitted

## Features

- 129-state IEEE 39-bus system (10 generators + 10 inverters, 27-bus Kron-reduced)
- Two-axis synchronous generator model with AVR and governor-turbine
- 6-state grid-following inverter (PLL, frequency/voltage filters, P/Q order)
- 6-state grid-forming inverter (virtual synchronous generator with adaptive virtual inertia)
- Control-affine decomposition: dx/dt = f(x) + g(x)u
- Newton KCL solver with analytic Jacobian and IFT backward pass
- Batched GPU execution for parallel scenario training
- ZIP load model, Kron-reduced admittance matrix
- Incremental energy (Lyapunov) functions for physics-informed training
- Validated against ANDES transient stability simulator

## Installation

```bash
# Clone the repository
git clone https://github.com/[your-username]/differentiable-dae-simulator.git
cd differentiable-dae-simulator

# Install dependencies
pip install -r requirements.txt
```

Requires Python 3.9+ and PyTorch 2.0+. GPU (CUDA) recommended for batched training.

## Quick Start

```python
import torch
import numpy as np
from IEEE39ControlAffineDAE_GFL6 import IEEE39ControlAffineDAE_GFL6

# Build the GFL system model
pv_ratio = np.full(10, 0.3)  # 30% PV penetration at each bus
sys_model = IEEE39ControlAffineDAE_GFL6(
    pv_ratio=pv_ratio,
    nominal_params={'reduced_load_equiv': 'ward_shunt'},
)

# Get equilibrium point
x_eq = sys_model.goal_point  # shape: (1, 129)

# Compute dynamics with gradients
x = x_eq.clone().requires_grad_(True)
f, g = sys_model.control_affine_dynamics(x)
# f: drift dynamics (1, 129)
# g: control input matrix (1, 129, 20)

# Backpropagate through the DAE solver
loss = f[:, 9:19].pow(2).sum()  # frequency deviation
loss.backward()
print(f"Gradient norm: {x.grad.norm():.4f}")
```

See `examples/quick_start.py` for a complete example.

## Repository Structure

```
.
├── IEEE39ControlAffineDAE_GFL6.py   # GFL inverter model (main simulator)
├── IEEE39ControlAffineDAE_GFM.py    # GFM inverter model
├── control_affine_system.py         # Base class for control-affine systems
├── train_gfl6.py                    # GFL training pipeline
├── train_gfm.py                     # GFM training pipeline
├── tri_branch_controller.py         # Tri-branch controller architecture
├── run_all_evaluations.py           # Evaluation orchestrator
├── benchmarks/
│   ├── gradient_validation.py       # IFT vs finite-difference validation
│   ├── compute_benchmark.py         # Wall-clock timing benchmarks
│   └── andes_validation.py          # Cross-validation against ANDES
├── examples/
│   └── quick_start.py               # Minimal usage example
├── figures/                         # Paper figures
├── training_logs/                   # Saved training runs
├── arxiv_dae_paper.tex              # ArXiv paper source
└── ref_arxiv.bib                    # Bibliography
```

## State Vector Layout (129 states)

```
Index   States          Count   Description
0-8     delta_rel       9       Generator rotor angles (relative to ref)
9-18    omega           10      Generator speeds (pu)
19-28   Eqp             10      Transient q-axis voltage
29-38   Edp             10      Transient d-axis voltage
39-48   Efd             10      Field voltage (AVR output)
49-58   Pm              10      Mechanical power
59-68   Pvalve          10      Governor valve position
69-128  inv_states      60      Inverter states (6 per inverter x 10)
```

GFL inverter states (per inverter): phi, xi_pll, f_meas, V_meas, P_ord, Q_ord

GFM inverter states (per inverter): theta_v, omega_v, M_inv, V_meas, P_filt, Q_filt

## Training

```bash
# GFM controller training (GPU recommended)
python train_gfm.py --epochs 100 --batch-size 32 --device cuda

# GFL controller training
python train_gfl6.py --epochs 100 --batch-size 32 --device cuda

# Resume from checkpoint
python train_gfl6.py --resume-from training_logs/latest/checkpoints/checkpoint_latest.pt
```

## Benchmarks

```bash
# Gradient validation (IFT backward vs finite differences)
python benchmarks/gradient_validation.py

# Computational performance (CPU vs GPU, batch scaling)
python benchmarks/compute_benchmark.py

# ANDES cross-validation (requires andes package)
python benchmarks/andes_validation.py --device cpu
```

## Citation

```bibtex
@article{Alduaij2025DiffDAE,
  author  = {Hamad Alduaij,
  title   = {High-Fidelity Differentiable {DAE} Simulation of Mixed
             Synchronous-Generator and Inverter-Based Power Systems, To be submitted},
  journal = {arXiv preprint},
  year    = {2025},
}
```

## License

MIT License. See [LICENSE](LICENSE).



