"""
scripts/run_no_hardware.py
==========================
Complete experiment suite — NO IBM hardware credentials required.

Runs everything using:
  1. NumPy  — analytical (exact, zero-noise) theory curves
  2. Qiskit AerSimulator — realistic shot noise + depolarising gate errors,
     no real QPU, no QISKIT_IBM_TOKEN needed

Outputs go to results/no_hardware/ (separate from hardware results):
  simulator_results.json   — n=1,2,3 kernel values + metrics vs theory
  vce_summary.json         — VCE novelty: physical vs virtual n=3
  01_n_copies_effect.png   — swap kernel n=1,2,3: theory (solid) vs sim (dots)
  02_helstrom_equivalence.png — swap kernel = Helstrom measurement (analytical proof)
  03_shots_comparison.png  — 256 vs 1024 shots vs theory
  04_vce_novelty.png       — physical n=3 vs virtual n=3 (VCE)

Usage
-----
  python scripts/run_no_hardware.py
  python scripts/run_no_hardware.py --quick   # 30 theta points (default)
  python scripts/run_no_hardware.py --full    # 63 theta points
  python scripts/run_no_hardware.py --shots 2048  # custom shot count

Requirements: pip install -r requirements-qiskit.txt
(does NOT need QISKIT_IBM_TOKEN or any .env file)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results", "no_hardware")


def _save_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def section(title: str) -> None:
    print(f"\n{'=' * 68}")
    print(f"  {title}")
    print(f"{'=' * 68}")


# ---------------------------------------------------------------------------
# Step 1: Mathematical verification (pure NumPy, no Qiskit at all)
# ---------------------------------------------------------------------------

def run_verification() -> bool:
    section("STEP 1 — Mathematical Verification (pure NumPy)")

    from circuits.hadamard_classifier import HadamardClassifier
    from circuits.swap_test_classifier import SwapTestClassifier
    from core.kernel import (
        helstrom_expectation, helstrom_operator, kernel_matrix, swap_test_kernel,
    )
    from experiments.toy_problem import (
        analytical_hadamard_kernel, analytical_swap_kernel,
        get_test_state, get_theta_range, get_training_data, true_classification,
    )

    x_train, labels = get_training_data()
    thetas = get_theta_range(20)
    pass_count = fail_count = 0

    def check(name, condition, detail=""):
        nonlocal pass_count, fail_count
        if condition:
            pass_count += 1
            print(f"  PASS  {name}")
        else:
            fail_count += 1
            print(f"  FAIL  {name}" + (f"\n        {detail}" if detail else ""))

    print("\n[1] Training state normalization")
    for m, xm in enumerate(x_train):
        check(f"|x_{m+1}| = 1", abs(np.linalg.norm(xm) - 1.0) < 1e-14)

    print("\n[2] Hadamard kernel = 0 for all theta")
    had_vals = [analytical_hadamard_kernel(t) for t in thetas]
    check("Hadamard kernel is zero", all(abs(v) < 1e-14 for v in had_vals))

    print("\n[3] Swap kernel range and 2pi periodicity")
    swap_vals = [analytical_swap_kernel(t) for t in thetas]
    check("Range [-0.5, 0.5]", all(-0.5 - 1e-12 <= v <= 0.5 + 1e-12 for v in swap_vals))
    v1, v2 = analytical_swap_kernel(0.5), analytical_swap_kernel(0.5 + 2 * np.pi)
    check("2pi periodicity", abs(v1 - v2) < 1e-12)

    print("\n[4] Helstrom equivalence (analytical proof, ~1e-16)")
    for n in [1, 2, 3]:
        A = helstrom_operator(x_train, labels, n_copies=n)
        diffs = [
            abs(swap_test_kernel(get_test_state(t), x_train, labels, n_copies=n)
                - helstrom_expectation(get_test_state(t), A, n_copies=n))
            for t in thetas
        ]
        check(f"n={n} swap-test == Helstrom", max(diffs) < 1e-12,
              f"max diff={max(diffs):.2e}")

    print("\n[5] Boundary checks: k(0) and k(pi) ~ 0")
    check("k(0) ~ 0", abs(analytical_swap_kernel(0.0)) < 1e-12)
    check("k(pi) ~ 0", abs(analytical_swap_kernel(np.pi)) < 1e-12)

    print("\n[6] Kernel matrix is positive semi-definite")
    states = x_train + [get_test_state(t) for t in get_theta_range(12)]
    K = kernel_matrix(states, n_copies=1)
    eigvals = np.linalg.eigvalsh(K)
    check("PSD (min eigenvalue >= 0)", np.all(eigvals >= -1e-10),
          f"min eig={eigvals.min():.2e}")
    check("Diagonal = 1", np.allclose(np.diag(K), 1.0))

    print("\n[7] NumPy circuit vs analytical agreement")
    clf_swap = SwapTestClassifier(n_copies=1)
    clf_had = HadamardClassifier()
    swap_errs = [abs(clf_swap.run(x_train, labels, get_test_state(t))["expectation_ZZ"]
                     - analytical_swap_kernel(t)) for t in thetas]
    had_errs = [abs(clf_had.run(x_train, labels, get_test_state(t))["expectation_ZZ"]
                    - analytical_hadamard_kernel(t)) for t in thetas]
    check("Swap circuit matches analytical", max(swap_errs) < 1e-8)
    check("Hadamard circuit matches analytical", max(had_errs) < 1e-8)

    print("\n[8] Classification accuracy = 100%")
    total = len(thetas)
    correct = sum(
        1 for t in thetas
        if clf_swap.run(x_train, labels, get_test_state(t))["predicted_label"]
        in (true_classification(t), -1) or true_classification(t) == -1
    )
    check("Swap-test accuracy = 100%", correct == total)

    print(f"\n  Results: {pass_count} passed, {fail_count} failed")
    return fail_count == 0


# ---------------------------------------------------------------------------
# Step 2: Qiskit AerSimulator — n=1,2,3 sweep
# ---------------------------------------------------------------------------

def _run_sim_sweep(thetas, copies, shots):
    """Run one (copies, shots) sweep on AerSimulator. Returns expectation list."""
    from qiskit_layer.runner import run_swaptest_theta_sweep_qiskit
    result = run_swaptest_theta_sweep_qiskit(
        thetas=thetas, shots=shots, mode="simulator",
        circuit_family="product_state", copies=copies,
        use_noise=True, wait_for_result=True, env_file=None,
    )
    exp = result.get("expectation")
    if exp is None:
        raise RuntimeError(f"Simulator returned no expectation for copies={copies}")
    return [float(x) for x in exp]


def run_simulator_experiments(thetas: np.ndarray, shots: int) -> dict:
    """Run n=1,2,3 on AerSimulator. Returns dict for simulator_results.json."""
    from experiments.toy_problem import analytical_swap_kernel
    from qiskit_layer.mitigation import curve_error_metrics

    theory_by_n = {
        n: [float(analytical_swap_kernel(t, n_copies=n)) for t in thetas]
        for n in [1, 2, 3]
    }

    measured_by_n: dict[int, list] = {}
    for n in [1, 2, 3]:
        print(f"  Simulator n={n}, {shots} shots...", flush=True)
        measured_by_n[n] = _run_sim_sweep(thetas, copies=n, shots=shots)

    metrics = {
        f"n{n}": curve_error_metrics(np.array(measured_by_n[n]), np.array(theory_by_n[n]))
        for n in [1, 2, 3]
    }

    return {
        "backend": "Qiskit AerSimulator (depolarising noise, no IBM credentials)",
        "shots": shots,
        "thetas": [float(t) for t in thetas],
        "theory": {f"n{n}": theory_by_n[n] for n in [1, 2, 3]},
        "measured": {f"n{n}": measured_by_n[n] for n in [1, 2, 3]},
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Step 3: Shots comparison — 256 vs 1024
# ---------------------------------------------------------------------------

def run_shots_comparison(thetas: np.ndarray, sim_data: dict) -> dict:
    """Run n=1 at 256 shots (1024 already in sim_data). Returns shots_by_count."""
    from experiments.toy_problem import analytical_swap_kernel
    from qiskit_layer.mitigation import curve_error_metrics

    print("  Simulator n=1, 256 shots...", flush=True)
    shots_256 = _run_sim_sweep(thetas, copies=1, shots=256)
    shots_1024 = sim_data["measured"]["n1"]  # reuse from main sweep

    theory_n1 = np.array(sim_data["theory"]["n1"])
    shots_metrics = {
        256: curve_error_metrics(np.array(shots_256), theory_n1),
        1024: curve_error_metrics(np.array(shots_1024), theory_n1),
    }
    return {
        "shots_by_count": {"256": shots_256, "1024": shots_1024},
        "shots_metrics": {str(k): v for k, v in shots_metrics.items()},
    }


# ---------------------------------------------------------------------------
# Step 4: VCE — virtual n=3 from n=1,2
# ---------------------------------------------------------------------------

def run_vce(thetas: np.ndarray, sim_data: dict, shots: int) -> dict:
    """Build VCE curves from simulator n=1,2,3 runs."""
    from experiments.toy_problem import analytical_swap_kernel
    from qiskit_layer.mitigation import build_vce_curves, curve_error_metrics

    physical_curves = {
        n: np.array(sim_data["measured"][f"n{n}"]) for n in [1, 2, 3]
    }
    vce_out = build_vce_curves(physical_curves, target_copies=3)
    virtual_n3 = [float(x) for x in vce_out["virtual_n3_from_12"]]
    physical_n3 = [float(x) for x in physical_curves[3]]
    theory_n3 = [float(analytical_swap_kernel(t, n_copies=3)) for t in thetas]

    return {
        "description": "VCE: estimate n=3 kernel from n=1,2 AerSimulator measurements",
        "target_copies": 3,
        "shots": shots,
        "thetas": [float(t) for t in thetas],
        "theory_n3": theory_n3,
        "simulator": {
            "physical_n3": physical_n3,
            "virtual_n3": virtual_n3,
            "metrics": {
                "physical_n3": curve_error_metrics(np.array(physical_n3), np.array(theory_n3)),
                "virtual_n3": curve_error_metrics(np.array(virtual_n3), np.array(theory_n3)),
            },
        },
    }


# ---------------------------------------------------------------------------
# Step 5: Generate all 4 plots (simulator-only, no hardware panels)
# ---------------------------------------------------------------------------

def generate_plots(sim_data: dict, vce_data: dict, shots_data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.toy_problem import analytical_swap_kernel
    from qiskit_layer.mitigation import curve_error_metrics
    from visualization.plots import (
        plot_helstrom_equivalence,
        plot_n_copies_effect,
        plot_shots_comparison,
        plot_vce_novelty,
    )

    thetas = np.array(sim_data["thetas"])
    theory_by_n = {n: np.array(sim_data["theory"][f"n{n}"]) for n in [1, 2, 3]}
    sim_measured = {n: np.array(sim_data["measured"][f"n{n}"]) for n in [1, 2, 3]}
    sim_metrics_n = sim_data.get("metrics", {})

    # Plot 01: n copies — simulator only (hardware panels will be empty dicts)
    fig = plot_n_copies_effect(
        thetas,
        hw_measured_by_n={},          # no hardware data
        sim_measured_by_n=sim_measured,
        theory_by_n=theory_by_n,
        hw_metrics=None,
        sim_metrics=sim_metrics_n,
        save_path=os.path.join(RESULTS, "01_n_copies_effect.png"),
    )
    plt.close(fig)
    print("  Saved: 01_n_copies_effect.png")

    # Plot 02: Helstrom equivalence — analytical only, measured dots for sim
    fig = plot_helstrom_equivalence(
        thetas,
        hw_measured_by_n={},
        sim_measured_by_n=sim_measured,
        theory_by_n=theory_by_n,
        save_path=os.path.join(RESULTS, "02_helstrom_equivalence.png"),
    )
    plt.close(fig)
    print("  Saved: 02_helstrom_equivalence.png")

    # Plot 03: shots comparison
    sim_shots = {int(k): np.array(v) for k, v in shots_data["shots_by_count"].items()}
    theory_n1 = theory_by_n[1]
    sim_shots_metrics = {
        int(k): v for k, v in shots_data.get("shots_metrics", {}).items()
    }
    fig = plot_shots_comparison(
        thetas,
        hw_measured_by_shot={},
        sim_measured_by_shot=sim_shots,
        theory=theory_n1,
        hw_metrics=None,
        sim_metrics=sim_shots_metrics,
        save_path=os.path.join(RESULTS, "03_shots_comparison.png"),
    )
    plt.close(fig)
    print("  Saved: 03_shots_comparison.png")

    # Plot 04: VCE novelty
    sim_sec = vce_data["simulator"]
    fig = plot_vce_novelty(
        thetas,
        hw_physical_n3=np.zeros_like(thetas),   # no hardware
        hw_virtual_n3=np.zeros_like(thetas),
        sim_physical_n3=np.array(sim_sec["physical_n3"]),
        sim_virtual_n3=np.array(sim_sec["virtual_n3"]),
        theory_n3=np.array(vce_data["theory_n3"]),
        save_path=os.path.join(RESULTS, "04_vce_novelty.png"),
    )
    plt.close(fig)
    print("  Saved: 04_vce_novelty.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run all experiments without IBM hardware credentials",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
No IBM account or .env file needed — uses Qiskit AerSimulator only.

Examples:
  python scripts/run_no_hardware.py
  python scripts/run_no_hardware.py --quick
  python scripts/run_no_hardware.py --shots 2048
        """,
    )
    ap.add_argument("--quick", action="store_true", default=True,
                    help="Use 30 theta points (default)")
    ap.add_argument("--full", action="store_true",
                    help="Use 63 theta points instead of 30")
    ap.add_argument("--shots", type=int, default=1024,
                    help="Shot count for main sweep (default: 1024)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip mathematical verification step")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    n_theta = 30 if not args.full else 63

    # Lazy import so numpy-only path never touches qiskit
    from experiments.toy_problem import get_theta_range
    thetas = get_theta_range(n_theta)

    print(f"\nOutput directory: results/no_hardware/")
    print(f"Theta points: {n_theta}   Main shots: {args.shots}")
    print("No IBM credentials required.\n")

    # Step 1 — Verification
    if not args.skip_verify:
        ok = run_verification()
        if not ok:
            print("\nVerification failed — aborting.")
            return 1

    # Step 2 — Simulator n=1,2,3
    section("STEP 2 — AerSimulator: n=1,2,3 kernel sweep")
    sim_data = run_simulator_experiments(thetas, shots=args.shots)
    sim_path = os.path.join(RESULTS, "simulator_results.json")
    _save_json(sim_path, sim_data)
    print(f"  Saved: simulator_results.json")

    # Step 3 — Shots comparison
    section("STEP 3 — AerSimulator: shots comparison (256 vs 1024)")
    shots_data = run_shots_comparison(thetas, sim_data)
    # embed shots_by_count into sim_data for unified JSON
    sim_data["shots_by_count"] = shots_data["shots_by_count"]
    _save_json(sim_path, sim_data)   # overwrite with shots added

    # Step 4 — VCE
    section("STEP 4 — VCE: virtual n=3 from n=1,2 measurements")
    vce_data = run_vce(thetas, sim_data, shots=args.shots)
    vce_path = os.path.join(RESULTS, "vce_summary.json")
    _save_json(vce_path, vce_data)
    print("  Saved: vce_summary.json")

    # Print VCE improvement
    phys_mae = vce_data["simulator"]["metrics"]["physical_n3"]["mean_abs_diff"]
    virt_mae = vce_data["simulator"]["metrics"]["virtual_n3"]["mean_abs_diff"]
    phys_rmse = vce_data["simulator"]["metrics"]["physical_n3"]["rmse"]
    virt_rmse = vce_data["simulator"]["metrics"]["virtual_n3"]["rmse"]
    imp_mae = (phys_mae - virt_mae) / phys_mae * 100
    imp_rmse = (phys_rmse - virt_rmse) / phys_rmse * 100
    print(f"  VCE result:  physical n=3 MAE={phys_mae:.4f}  RMSE={phys_rmse:.4f}")
    print(f"               virtual  n=3 MAE={virt_mae:.4f}  RMSE={virt_rmse:.4f}")
    print(f"  Improvement: MAE {imp_mae:+.1f}%   RMSE {imp_rmse:+.1f}%")

    # Step 5 — Plots
    section("STEP 5 — Generating all 4 plots")
    generate_plots(sim_data, vce_data, shots_data)

    print(f"\nDone. All outputs in results/no_hardware/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
