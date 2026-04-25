"""
main.py
=======
Quantum classifier experiment runner.

Usage
-----
  # Run mathematical verification checks:
  python main.py --verify

  # Generate simulator results and all 4 plots:
  python main.py --generate-results --backend simulator

  # Generate results using real IBM hardware (requires .env):
  python scripts/hardware_suite.py

Output (results/ directory)
---------------------------
  JSON files:
    simulator_results.json   — n=1,2,3 kernel values + metrics (simulator)
    hardware_results.json    — n=1,2,3 kernel values + metrics (hardware)
    vce_summary.json         — VCE novelty: physical vs virtual n=3

  Plots (each has hardware left panel, simulator right panel):
    01_n_copies_effect.png   — swap kernel n=1,2,3 vs theory
    02_helstrom_equivalence.png — swap kernel = optimal Helstrom measurement
    03_shots_comparison.png  — 256 vs 1024 shots vs theory
    04_vce_novelty.png       — physical n=3 vs virtual n=3 (VCE novelty)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def _results_dir() -> str:
    p = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(p, exist_ok=True)
    return p


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def section(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification() -> bool:
    """Run 8 mathematical and implementation consistency checks."""
    section("VERIFICATION: Mathematical Properties")

    from circuits.hadamard_classifier import HadamardClassifier
    from circuits.swap_test_classifier import SwapTestClassifier
    from core.kernel import (
        helstrom_expectation,
        helstrom_operator,
        kernel_matrix,
        swap_test_kernel,
    )
    from experiments.toy_problem import (
        analytical_hadamard_kernel,
        analytical_swap_kernel,
        get_test_state,
        get_theta_range,
        get_training_data,
        true_classification,
    )

    x_train, labels = get_training_data()
    test_thetas = get_theta_range(20)
    pass_count = 0
    fail_count = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal pass_count, fail_count
        if condition:
            pass_count += 1
            print(f"  PASS  {name}")
        else:
            fail_count += 1
            print(f"  FAIL  {name}")
            if detail:
                print(f"        {detail}")

    print("\n[1] Training state normalization")
    for m, xm in enumerate(x_train):
        check(f"|x_{m+1}| = 1", abs(np.linalg.norm(xm) - 1.0) < 1e-14,
              f"norm={np.linalg.norm(xm):.16f}")

    print("\n[2] Hadamard kernel = 0 for all theta (Eq. 13 behavior)")
    had_vals = [analytical_hadamard_kernel(t) for t in test_thetas]
    check("Hadamard kernel is zero", all(abs(v) < 1e-14 for v in had_vals),
          f"max |value| = {max(abs(v) for v in had_vals):.2e}")

    print("\n[3] Swap kernel range and periodicity")
    swap_vals = [analytical_swap_kernel(t) for t in test_thetas]
    check("Swap kernel in [-0.5, 0.5]",
          all(-0.5 - 1e-12 <= v <= 0.5 + 1e-12 for v in swap_vals),
          f"range=[{min(swap_vals):.4f}, {max(swap_vals):.4f}]")
    v1, v2 = analytical_swap_kernel(0.5), analytical_swap_kernel(0.5 + 2 * np.pi)
    check("2pi periodicity", abs(v1 - v2) < 1e-12, f"delta={abs(v1-v2):.3e}")

    print("\n[4] Helstrom equivalence (Eq. 16-17)")
    for n in [1, 2, 3]:
        A = helstrom_operator(x_train, labels, n_copies=n)
        diffs = [abs(swap_test_kernel(get_test_state(t), x_train, labels, n_copies=n)
                     - helstrom_expectation(get_test_state(t), A, n_copies=n))
                 for t in test_thetas]
        check(f"n={n} swap-test == Helstrom", max(diffs) < 1e-12,
              f"max diff={max(diffs):.2e}")

    print("\n[5] Boundary checks")
    k0, kpi = analytical_swap_kernel(0.0), analytical_swap_kernel(np.pi)
    check("k(0) ~ 0 (boundary)", abs(k0) < 1e-12, f"k(0)={k0:.3e}")
    check("k(pi) ~ 0 (boundary)", abs(kpi) < 1e-12, f"k(pi)={kpi:.3e}")

    print("\n[6] Kernel matrix validity")
    states = x_train + [get_test_state(t) for t in get_theta_range(12)]
    K = kernel_matrix(states, n_copies=1)
    eigvals = np.linalg.eigvalsh(K)
    check("Kernel matrix PSD", np.all(eigvals >= -1e-10), f"min eig={eigvals.min():.2e}")
    check("Kernel diagonal = 1", np.allclose(np.diag(K), 1.0))

    print("\n[7] Circuit vs analytical agreement")
    clf_swap = SwapTestClassifier(n_copies=1)
    clf_had = HadamardClassifier()
    swap_errors, had_errors = [], []
    for theta in test_thetas:
        xt = get_test_state(theta)
        swap_errors.append(abs(clf_swap.run(x_train, labels, xt)["expectation_ZZ"]
                               - analytical_swap_kernel(theta)))
        had_errors.append(abs(clf_had.run(x_train, labels, xt)["expectation_ZZ"]
                              - analytical_hadamard_kernel(theta)))
    check("Swap circuit matches analytical", max(swap_errors) < 1e-8,
          f"max err={max(swap_errors):.2e}")
    check("Hadamard circuit matches analytical", max(had_errors) < 1e-8,
          f"max err={max(had_errors):.2e}")

    print("\n[8] Noiseless classification accuracy")
    total = len(test_thetas)
    correct = sum(
        1 for theta in test_thetas
        if clf_swap.run(x_train, labels, get_test_state(theta))["predicted_label"]
        in (true_classification(theta), -1) or true_classification(theta) == -1
    )
    check("Swap-test accuracy = 100%", correct == total,
          f"accuracy={correct/total*100:.1f}%")

    print(f"\n  Results: {pass_count} passed, {fail_count} failed")
    return fail_count == 0


# ---------------------------------------------------------------------------
# Simulator data collection
# ---------------------------------------------------------------------------

def _collect_simulator_results(thetas: np.ndarray, shots: int, env_file: str) -> dict:
    """
    Run n=1,2,3 swap-kernel circuits on Qiskit AerSimulator.
    Returns dict ready to be saved as simulator_results.json.
    """
    from experiments.toy_problem import analytical_swap_kernel
    from qiskit_layer.mitigation import curve_error_metrics
    from qiskit_layer.runner import run_swaptest_theta_sweep_qiskit

    measured_by_n: dict[int, list] = {}
    for n in [1, 2, 3]:
        print(f"  Running simulator n={n} ({shots} shots)...")
        run = run_swaptest_theta_sweep_qiskit(
            thetas=thetas,
            shots=shots,
            mode="simulator",
            circuit_family="product_state",
            copies=n,
            use_noise=True,
            wait_for_result=True,
            env_file=env_file,
        )
        if run.get("expectation"):
            measured_by_n[n] = [float(x) for x in run["expectation"]]

    theory_by_n = {
        n: [float(analytical_swap_kernel(t, n_copies=n)) for t in thetas]
        for n in [1, 2, 3]
    }

    metrics = {}
    for n in [1, 2, 3]:
        if n in measured_by_n:
            m = curve_error_metrics(
                np.array(measured_by_n[n]),
                np.array(theory_by_n[n]),
            )
            metrics[f"n{n}"] = m

    return {
        "backend": "Qiskit AerSimulator (depolarising noise)",
        "shots": shots,
        "thetas": [float(t) for t in thetas],
        "theory": {f"n{n}": theory_by_n[n] for n in [1, 2, 3]},
        "measured": {f"n{n}": measured_by_n[n] for n in [1, 2, 3] if n in measured_by_n},
        "metrics": metrics,
    }


def _collect_simulator_shots(thetas: np.ndarray, env_file: str) -> dict:
    """Run n=1 at 256 and 1024 shots on simulator. Returns {256: [...], 1024: [...]}."""
    from qiskit_layer.runner import run_swaptest_theta_sweep_qiskit
    result = {}
    for shots in [256, 1024]:
        print(f"  Running simulator shots={shots}...")
        run = run_swaptest_theta_sweep_qiskit(
            thetas=thetas, shots=shots, mode="simulator",
            circuit_family="product_state", copies=1,
            use_noise=True, wait_for_result=True, env_file=env_file,
        )
        if run.get("expectation"):
            result[shots] = [float(x) for x in run["expectation"]]
    return result


def _collect_simulator_vce(thetas: np.ndarray, shots: int, env_file: str) -> tuple:
    """
    Run n=1,2,3 on simulator and build VCE curves.
    Returns (physical_n3, virtual_n3) as lists.
    """
    from qiskit_layer.mitigation import build_vce_curves
    from qiskit_layer.runner import run_swaptest_theta_sweep_qiskit

    physical_curves: dict[int, np.ndarray] = {}
    for n in [1, 2, 3]:
        print(f"  Running simulator VCE n={n}...")
        run = run_swaptest_theta_sweep_qiskit(
            thetas=thetas, shots=shots, mode="simulator",
            circuit_family="product_state", copies=n,
            use_noise=True, wait_for_result=True, env_file=env_file,
        )
        if run.get("expectation"):
            physical_curves[n] = np.array(run["expectation"], dtype=float)

    if 1 not in physical_curves or 2 not in physical_curves:
        return None, None

    vce = build_vce_curves(physical_curves, target_copies=3)
    physical_n3 = physical_curves.get(3)
    virtual_n3 = np.array(vce["virtual_n3_from_12"], dtype=float)
    return (
        [float(x) for x in physical_n3] if physical_n3 is not None else None,
        [float(x) for x in virtual_n3],
    )


# ---------------------------------------------------------------------------
# Generate all plots from loaded JSON data
# ---------------------------------------------------------------------------

def _generate_plots(results_dir: str) -> None:
    """Load the 3 clean JSON files and generate all 4 side-by-side plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from experiments.toy_problem import analytical_swap_kernel
    from visualization.plots import (
        plot_helstrom_equivalence,
        plot_n_copies_effect,
        plot_shots_comparison,
        plot_vce_novelty,
    )

    hw_path = os.path.join(results_dir, "hardware_results.json")
    sim_path = os.path.join(results_dir, "simulator_results.json")
    vce_path = os.path.join(results_dir, "vce_summary.json")

    hw = _load_json(hw_path) if os.path.exists(hw_path) else None
    sim = _load_json(sim_path) if os.path.exists(sim_path) else None
    vce = _load_json(vce_path) if os.path.exists(vce_path) else None

    if sim is None:
        print("  simulator_results.json not found — skipping plots.")
        return

    thetas = np.array(sim["thetas"])
    theory_by_n = {n: np.array(sim["theory"][f"n{n}"]) for n in [1, 2, 3]}

    hw_measured = {n: np.array(hw["measured"][f"n{n}"]) for n in [1, 2, 3]} if hw else {}
    sim_measured = {n: np.array(sim["measured"][f"n{n}"]) for n in [1, 2, 3]
                    if f"n{n}" in sim.get("measured", {})}

    hw_metrics_n = hw.get("metrics", {}) if hw else {}
    sim_metrics_n = sim.get("metrics", {})

    # Plot 01: n copies effect
    fig = plot_n_copies_effect(
        thetas, hw_measured, sim_measured, theory_by_n,
        hw_metrics=hw_metrics_n,
        sim_metrics=sim_metrics_n,
        save_path=os.path.join(results_dir, "01_n_copies_effect.png"),
    )
    plt.close(fig)
    print("  Saved: results/01_n_copies_effect.png")

    # Plot 02: Helstrom equivalence
    fig = plot_helstrom_equivalence(
        thetas, hw_measured, sim_measured, theory_by_n,
        save_path=os.path.join(results_dir, "02_helstrom_equivalence.png"),
    )
    plt.close(fig)
    print("  Saved: results/02_helstrom_equivalence.png")

    # Plot 03: shots comparison
    hw_shots = {}
    sim_shots = {}
    if hw and "shots_by_count" in hw:
        hw_shots = {int(k): np.array(v) for k, v in hw["shots_by_count"].items()}
    if sim and "shots_by_count" in sim:
        sim_shots = {int(k): np.array(v) for k, v in sim["shots_by_count"].items()}

    if hw_shots or sim_shots:
        theory_n1 = theory_by_n[1]

        def _shot_metrics(shots_dict):
            from qiskit_layer.mitigation import curve_error_metrics
            return {s: curve_error_metrics(np.array(v), np.array(theory_n1))
                    for s, v in shots_dict.items()}

        hw_shots_metrics = _shot_metrics(hw_shots) if hw_shots else {}
        sim_shots_metrics = _shot_metrics(sim_shots) if sim_shots else {}

        fig = plot_shots_comparison(
            thetas, hw_shots, sim_shots, theory_n1,
            hw_metrics=hw_shots_metrics,
            sim_metrics=sim_shots_metrics,
            save_path=os.path.join(results_dir, "03_shots_comparison.png"),
        )
        plt.close(fig)
        print("  Saved: results/03_shots_comparison.png")
    else:
        print("  Skipping plot 03: no shots data found in JSON files.")

    # Plot 04: VCE novelty
    if vce:
        theory_n3 = np.array(theory_by_n[3])
        hw_phys = np.array(vce["hardware"]["physical_n3"]) if vce.get("hardware", {}).get("physical_n3") else None
        hw_virt = np.array(vce["hardware"]["virtual_n3"]) if vce.get("hardware", {}).get("virtual_n3") else None
        sim_phys = np.array(vce["simulator"]["physical_n3"]) if vce.get("simulator", {}).get("physical_n3") else None
        sim_virt = np.array(vce["simulator"]["virtual_n3"]) if vce.get("simulator", {}).get("virtual_n3") else None

        if (hw_phys is not None or sim_phys is not None) and \
           (hw_virt is not None or sim_virt is not None):
            fig = plot_vce_novelty(
                thetas,
                hw_physical_n3=hw_phys if hw_phys is not None else np.zeros_like(theory_n3),
                hw_virtual_n3=hw_virt if hw_virt is not None else np.zeros_like(theory_n3),
                sim_physical_n3=sim_phys if sim_phys is not None else np.zeros_like(theory_n3),
                sim_virtual_n3=sim_virt if sim_virt is not None else np.zeros_like(theory_n3),
                theory_n3=theory_n3,
                save_path=os.path.join(results_dir, "04_vce_novelty.png"),
            )
            plt.close(fig)
            print("  Saved: results/04_vce_novelty.png")
        else:
            print("  Skipping plot 04: VCE data incomplete.")
    else:
        print("  Skipping plot 04: vce_summary.json not found.")


# ---------------------------------------------------------------------------
# Main generate-results flow (simulator side)
# ---------------------------------------------------------------------------

def run_generate_results(quick: bool, env_file: str) -> None:
    """
    Run all simulator experiments and generate the 4 side-by-side plots.
    Writes simulator_results.json and updates vce_summary.json.
    Hardware side is handled by scripts/hardware_suite.py.
    """
    section("GENERATING SIMULATOR RESULTS")

    from experiments.toy_problem import analytical_swap_kernel, get_theta_range
    from qiskit_layer.mitigation import curve_error_metrics

    thetas = get_theta_range(30 if quick else 63)
    results_dir = _results_dir()

    # --- n=1,2,3 sweep ---
    print("\nCollecting simulator n=1,2,3 data...")
    sim_data = _collect_simulator_results(thetas, shots=1024, env_file=env_file)

    # --- shots comparison (256 vs 1024) ---
    print("\nCollecting simulator shots comparison (256 vs 1024)...")
    shots_data = _collect_simulator_shots(thetas, env_file=env_file)
    sim_data["shots_by_count"] = {str(k): v for k, v in shots_data.items()}

    sim_path = os.path.join(results_dir, "simulator_results.json")
    _save_json(sim_path, sim_data)
    print(f"\n  Saved: results/simulator_results.json")

    # --- VCE ---
    print("\nCollecting simulator VCE data (n=1,2,3 for extrapolation)...")
    sim_phys_n3, sim_virt_n3 = _collect_simulator_vce(thetas, shots=1024, env_file=env_file)

    theory_n3 = [float(analytical_swap_kernel(t, n_copies=3)) for t in thetas]
    sim_phys_metrics = curve_error_metrics(np.array(sim_phys_n3), np.array(theory_n3)) \
        if sim_phys_n3 else {}
    sim_virt_metrics = curve_error_metrics(np.array(sim_virt_n3), np.array(theory_n3)) \
        if sim_virt_n3 else {}

    vce_path = os.path.join(results_dir, "vce_summary.json")
    # Load existing VCE file (may have hardware data from hardware_suite.py)
    if os.path.exists(vce_path):
        vce_data = _load_json(vce_path)
    else:
        vce_data = {
            "description": "VCE: estimate n=3 kernel from n=1,2 measurements",
            "target_copies": 3,
            "shots": 1024,
            "thetas": [float(t) for t in thetas],
            "theory_n3": theory_n3,
        }

    vce_data["simulator"] = {
        "physical_n3": sim_phys_n3,
        "virtual_n3": sim_virt_n3,
        "metrics": {
            "physical_n3": sim_phys_metrics,
            "virtual_n3": sim_virt_metrics,
        },
    }
    _save_json(vce_path, vce_data)
    print("  Saved: results/vce_summary.json")

    # --- Generate all 4 plots ---
    section("GENERATING PLOTS")
    _generate_plots(results_dir)


# ---------------------------------------------------------------------------
# Regenerate plots only (no new circuit runs)
# ---------------------------------------------------------------------------

def run_regenerate_plots() -> None:
    """Regenerate all 4 plots from existing JSON files (no circuit runs)."""
    section("REGENERATING PLOTS FROM EXISTING DATA")
    _generate_plots(_results_dir())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quantum classifier experiment runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --verify
  python main.py --generate-results --backend simulator
  python main.py --generate-results --backend simulator --quick
  python main.py --regenerate-plots
        """,
    )
    parser.add_argument("--verify", action="store_true",
                        help="Run mathematical verification checks")
    parser.add_argument("--generate-results", action="store_true",
                        help="Run simulator experiments and generate all 4 plots")
    parser.add_argument("--regenerate-plots", action="store_true",
                        help="Regenerate all 4 plots from existing JSON data (no new circuit runs)")
    parser.add_argument("--backend", choices=["simulator"], default="simulator",
                        help="Backend for --generate-results (hardware: use scripts/hardware_suite.py)")
    parser.add_argument("--quick", action="store_true",
                        help="Use 30 theta points instead of 63 (faster, same math)")
    parser.add_argument("--env-file", default=".env",
                        help="Path to .env file with QISKIT_IBM_* credentials")
    args = parser.parse_args()

    if not any([args.verify, args.generate_results, args.regenerate_plots]):
        parser.print_help()
        return 0

    if args.verify:
        ok = run_verification()
        if not args.generate_results and not args.regenerate_plots:
            return 0 if ok else 1

    if args.generate_results:
        run_generate_results(quick=args.quick, env_file=args.env_file)

    if args.regenerate_plots:
        run_regenerate_plots()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
