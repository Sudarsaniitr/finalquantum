# Quantum Classifier with Tailored Quantum Kernel

**Course project implementing and extending:**

> Blank, Park, Rhee, Petruccione — *"Quantum classifier with tailored quantum kernel"*
> *npj Quantum Information* **6**, 41 (2020).
> Paper: <https://www.nature.com/articles/s41534-020-0272-6>
> Supplemental code: <https://github.com/carstenblank/Quantum-classifier-with-tailored-quantum-kernels---Supplemental>

This repository contains a full, paper-faithful implementation of the classifier, run on both a noise-model simulator (AerSimulator) and a real IBM Quantum computer (`ibm_kingston`, 127-qubit QPU), plus an original novelty contribution called **Virtual Copy Extrapolation (VCE)**.

---

## The Research Paper (Summary)

The paper introduces a distance-based quantum classifier that uses the **quantum swap test** to compute a kernel (similarity measure) between quantum states. The key idea is that the kernel sharpens as the number of data copies `n` increases:

```
K_n(x̃) = Σ_m (-1)^{y_m} w_m |⟨x̃|x_m⟩|^{2n}
```

- As `n → ∞` the kernel approaches a Dirac-delta localised at the training states — perfect classification.
- The paper proves this is equivalent to the Helstrom measurement (optimal quantum decision rule).
- The circuit requires only `H → CSWAP → H` on an ancilla qubit and `2n+3` qubits total.

**Compared baseline:** The Hadamard classifier (prior work) uses `Re⟨x̃|x_m⟩`. The paper's toy problem is designed so that this real part is exactly 0 for all test angles, demonstrating a case the Hadamard classifier cannot solve but the swap-test can.

---

## Novelty Contribution — Virtual Copy Extrapolation (VCE)

Running `n` copies on a NISQ device multiplies the qubit count and circuit depth by `n`, amplifying hardware errors. The novelty asks:

> *Can we estimate the high-n kernel from low-n runs, using fewer qubits?*

**Method (Richardson denoising + analytical kernel map):**

1. Run `n=1` (3 qubits) and `n=2` (5 qubits) circuits on hardware.
2. Richardson-denoise the K₁ measurement: `K₁* = 2·K₁ − K₂`
3. Recover the underlying fidelity: `p = K₁* + 0.5`
4. Map to any target `n`: `K_n = ½·(p^n − (1−p)^n)`

**Results** (vs physical `n=3` run, 1024 shots, `ibm_kingston`):

| Method | Mean abs error | RMSE |
|---|---|---|
| Physical n=3 (actual hardware run) | 0.0818 | 0.0983 |
| **Virtual n=3 via VCE (our method)** | **0.0717** | **0.0860** |

VCE improves mean error by ~12% and RMSE by ~13% on real hardware, using one fewer qubit-copy register. This validates the "software fix for a hardware problem" hypothesis in NISQ-era quantum computing.

---

## Project Structure

```text
quantum/
├── main.py                          # CLI entry point — verification + simulator results
├── requirements.txt                 # Base NumPy/matplotlib deps
├── requirements-qiskit.txt          # Qiskit + IBM runtime deps
├── .env.example                     # Template for IBM credentials
├── novelty.md                       # VCE novelty design document
│
├── core/
│   ├── kernel.py                    # Classical kernel math (Eqs. 6, 9, 16-17)
│   └── quantum_gates.py             # NumPy gate primitives (H, CSWAP, tensor products)
│
├── circuits/
│   ├── hadamard_classifier.py       # Hadamard circuit (paper baseline)
│   └── swap_test_classifier.py      # Swap-test circuit (paper main contribution)
│
├── experiments/
│   └── toy_problem.py               # Training/test states, analytical kernels
│
├── visualization/
│   └── plots.py                     # 4 plot functions — all hardware (left) | simulator (right)
│
├── qiskit_layer/
│   ├── circuits.py                  # Qiskit circuit builders (swap-test + n-copy)
│   ├── backends.py                  # AerSimulator + IBM backend helpers
│   ├── runner.py                    # Shot sweep executor (simulator & hardware)
│   ├── noise.py                     # Depolarising noise model builders
│   └── mitigation.py                # VCE estimators (Richardson + closed-form map)
│
├── scripts/
│   ├── run_no_hardware.py           # FULL suite — no IBM credentials, AerSimulator only
│   ├── hardware_suite.py            # IBM hardware runner — writes clean JSON + plots
│   └── metrics_report.py            # Print/save error metrics for all experiments
│
└── results/
    ├── hardware_results.json         — hardware (IBM Kingston) kernel values + metrics
    ├── simulator_results.json        — simulator (AerSimulator) kernel values + metrics
    ├── vce_summary.json              — VCE: physical vs virtual n=3, both backends
    │
    ├── hardware/                     — 4 plots: hardware (left) | simulator (right)
    │   ├── 01_n_copies_effect.png
    │   ├── 02_helstrom_equivalence.png
    │   ├── 03_shots_comparison.png
    │   └── 04_vce_novelty.png
    │
    └── no_hardware/                  — 4 plots: simulator only (no IBM credentials needed)
        ├── 01_n_copies_effect.png
        ├── 02_helstrom_equivalence.png
        ├── 03_shots_comparison.png
        ├── 04_vce_novelty.png
        ├── simulator_results.json
        └── vce_summary.json
```

---

## Setup

**Base (NumPy analytical layer):**

```bash
pip install -r requirements.txt
```

**Full (Qiskit + IBM Runtime):**

```bash
pip install -r requirements-qiskit.txt
```

**IBM credentials** (only needed for hardware runs — `.env.example` is provided):

```bash
cp .env.example .env
# Edit .env: fill in your QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE
# These are only required if you run: python scripts/hardware_suite.py
# All other experiments (NumPy, AerSimulator) run WITHOUT credentials
```

---

## Running Experiments

### No-hardware full suite (no IBM account needed)

```bash
python scripts/run_no_hardware.py          # runs everything, 30 theta points
python scripts/run_no_hardware.py --full   # 63 theta points
python scripts/run_no_hardware.py --shots 2048  # higher shot count
```

Runs **everything without any IBM credentials**:
1. Mathematical verification (8 checks, pure NumPy)
2. Qiskit AerSimulator n=1,2,3 kernel sweep (with realistic gate noise)
3. Shots comparison: 256 vs 1024 shots
4. VCE novelty: virtual n=3 estimated from n=1,2 AerSimulator runs
5. All 4 plots (single-panel, simulator only)

Output goes to `results/no_hardware/` — completely independent from the hardware results.

### Mathematical verification

```bash
python main.py --verify
```

Runs 8 property checks: normalization, Hadamard kernel = 0, swap kernel range,
2π periodicity, Helstrom equivalence to machine precision (~1e-16), PSD kernel
matrix, circuit-vs-analytical agreement, and 100% classification accuracy.

### Simulator results + all 4 plots

```bash
python main.py --generate-results
python main.py --generate-results --quick   # faster: 30 theta points
```

Runs all simulator experiments and writes:
- `results/simulator_results.json`
- `results/vce_summary.json` (simulator section)
- All 4 plots (simulator panels; hardware panels empty until hardware run)

### IBM hardware results

```bash
# First set up credentials (only once):
cp .env.example .env
# Edit .env: add your QISKIT_IBM_TOKEN and QISKIT_IBM_INSTANCE

# Then run:
python scripts/hardware_suite.py
python scripts/hardware_suite.py --backend ibm_kingston
```

Runs all hardware experiments and writes:
- `results/hardware_results.json`
- `results/vce_summary.json` (hardware section added/updated)
- All 4 plots regenerated with hardware data on the left panels

**Note:** This step is optional. All other experiments work without IBM credentials.

### Regenerate plots only (no new circuit runs)

```bash
python main.py --regenerate-plots
```

Reads the existing JSON files and regenerates all 4 plots.

### Error metrics report

```bash
python scripts/metrics_report.py          # print to terminal
python scripts/metrics_report.py --save   # also save to results/metrics_report.txt
```

Prints mean absolute error (MAE), RMSE, and sign agreement for every measured
curve vs its theory — covering plots 01, 03, and 04:

- **Plot 01**: n=1, n=2, n=3 measured kernel vs analytical theory (hardware and simulator)
- **Plot 03**: 256-shot and 1024-shot curves vs theory (hardware and simulator)
- **Plot 04**: physical n=3 vs virtual n=3 (VCE) — including % improvement

The same metrics are printed as annotation boxes directly on each plot.

---

## Results Summary

### Each plot is side-by-side: hardware (left panel) | simulator (right panel)

| File | What it shows |
|---|---|
| `01_n_copies_effect.png` | Kernel sharpens as n increases — solid = theory, dashed = measured |
| `02_helstrom_equivalence.png` | Swap-test kernel == Helstrom operator to ~10⁻¹⁶ (Eqs. 16-17 proof) |
| `03_shots_comparison.png` | 256 vs 1024 shots — more shots converges to theory |
| `04_vce_novelty.png` | VCE: virtual n=3 from n=1,2 measurements beats physical n=3 direct run |

Each plot has **error metrics (MAE, RMSE) printed directly on the panel** as an annotation box.
Run `python scripts/metrics_report.py` for the full numeric breakdown.

### Plot 01 — n=1,2,3 kernel vs theory (1024 shots)

| Backend | n | MAE | RMSE | Sign agreement |
|---|---|---|---|---|
| Hardware (`ibm_kingston`) | n=1 | 0.0612 | 0.0727 | 96.7% |
| Hardware (`ibm_kingston`) | n=2 | 0.0758 | 0.0885 | 100.0% |
| Hardware (`ibm_kingston`) | n=3 | 0.0818 | 0.0983 | 100.0% |
| Simulator (AerSimulator) | n=1 | 0.0439 | 0.0507 | 100.0% |
| Simulator (AerSimulator) | n=2 | 0.0635 | 0.0696 | 96.7% |
| Simulator (AerSimulator) | n=3 | 0.0690 | 0.0826 | 96.7% |

### Shots comparison (n=1, swap kernel)

| Mode | Shots | Sign agreement | Mean abs err | RMSE |
|---|---|---|---|---|
| Simulator | 256 | 96.67% | 0.0591 | 0.0706 |
| Simulator | 1024 | 100.00% | 0.0422 | 0.0529 |
| Hardware | 256 | 90.00% | 0.0664 | 0.0795 |
| Hardware (`ibm_kingston`) | 1024 | 100.00% | 0.0692 | 0.0782 |

### VCE novelty — hardware (`ibm_kingston`, 1024 shots, target n=3)

| Curve | Mean abs error | RMSE |
|---|---|---|
| Physical n=3 (pre-novelty baseline) | 0.0818 | 0.0983 |
| **Virtual n=3 via VCE (post-novelty)** | **0.0717** | **0.0860** |

VCE achieves lower mean error and RMSE than the actual n=3 hardware run,
while using one fewer copy register and shorter circuits.

---

## Implementation Notes

- **No legacy API**: uses `SamplerV2` from `qiskit-ibm-runtime`, not the removed `backend.run()`.
- **Channel migration**: automatically maps deprecated `ibm_quantum` channel to `ibm_quantum_platform`.
- **Theta grid parity**: all Qiskit runs use 30 points over [0, 2π] by default (`--quick`) so simulator and hardware grids match for cross-comparison.
- **Noise model**: simulator uses a simple per-gate depolarising model (single-qubit error `p=0.001`, two-qubit error `p=0.01`).
- **IBM hardware jobs** were executed on `ibm_kingston` (127-qubit Eagle QPU, open-plan access).
  - Job IDs are embedded in each result JSON for audit and reproduction.
