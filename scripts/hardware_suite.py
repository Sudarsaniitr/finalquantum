"""
scripts/hardware_suite.py
=========================
Run all swap-kernel experiments on IBM hardware and write clean result files.

Outputs (results/ directory)
-----------------------------
  hardware_results.json   — n=1,2,3 kernel values + metrics + shots comparison
  vce_summary.json        — VCE novelty: physical vs virtual n=3 (hardware section)

  01_n_copies_effect.png      — updated with hardware data
  02_helstrom_equivalence.png — updated with hardware data
  03_shots_comparison.png     — updated with hardware data
  04_vce_novelty.png          — updated with hardware data

Usage
-----
  python scripts/hardware_suite.py
  python scripts/hardware_suite.py --backend ibm_kingston --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from dotenv import load_dotenv  # noqa: E402
from qiskit import transpile  # noqa: E402
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler  # noqa: E402

from qiskit_layer.circuits import (  # noqa: E402
    build_product_state_n_copies_circuit,
    build_swap_test_toy_circuit,
)
from qiskit_layer.runner import (  # noqa: E402
    _extract_counts_from_sampler_pub,
    _serialize_counts,
    expectation_from_counts,
)
from experiments.toy_problem import analytical_swap_kernel, get_theta_range  # noqa: E402
from qiskit_layer.mitigation import build_vce_curves, curve_error_metrics  # noqa: E402


def _results_dir() -> str:
    p = os.path.join(ROOT, "results")
    os.makedirs(p, exist_ok=True)
    return p


def _save_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _make_service(env_file: str, max_attempts: int = 5) -> QiskitRuntimeService:
    load_dotenv(env_file)
    token = os.getenv("QISKIT_IBM_TOKEN")
    instance = os.getenv("QISKIT_IBM_INSTANCE") or None
    channel = os.getenv("QISKIT_IBM_CHANNEL", "ibm_quantum_platform")

    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[svc] attempt {attempt}/{max_attempts} ...", flush=True)
            svc = QiskitRuntimeService(channel=channel, token=token, instance=instance)
            print("[svc] service ready", flush=True)
            return svc
        except Exception as exc:
            last_err = exc
            print(f"[svc] failed: {type(exc).__name__}: {str(exc)[:220]}", flush=True)
            if attempt < max_attempts:
                delay = min(30, 2 * attempt)
                print(f"[svc] sleeping {delay}s before retry", flush=True)
                time.sleep(delay)
    assert last_err is not None
    raise last_err


def _run_sweep(
    sampler: Sampler,
    backend,
    copies: int,
    shots: int,
    thetas: np.ndarray,
) -> list[float]:
    """Run product-state circuit (n=copies) for each theta, return expectation values."""
    circuits = [
        build_product_state_n_copies_circuit(theta=float(t), copies=int(copies))
        for t in thetas
    ]
    tqc = transpile(circuits, backend=backend, optimization_level=1)
    print(f"[job] submit copies={copies} shots={shots} circuits={len(circuits)}", flush=True)
    job = sampler.run(tqc, shots=shots)
    job_id = str(job.job_id())
    print(f"[job] id={job_id} waiting...", flush=True)

    result = job.result()
    try:
        pubs = list(result)
    except Exception:
        pubs = [result[i] for i in range(len(circuits))]

    counts_list = []
    for i in range(len(circuits)):
        counts_list.append(_extract_counts_from_sampler_pub(pubs[i]) if i < len(pubs) else {})

    if all(len(c) == 0 for c in counts_list):
        raise RuntimeError("Hardware result returned no extractable counts.")

    return [float(expectation_from_counts(c)) for c in counts_list]


def main(backend_name: str = "ibm_kingston", quick: bool = True, env_file: str = ".env") -> int:
    thetas = get_theta_range(30 if quick else 63)
    results_dir = _results_dir()
    env_path = os.path.join(ROOT, env_file)

    svc = _make_service(env_path)
    print(f"[svc] resolving backend {backend_name}...", flush=True)
    backend = svc.backend(backend_name)
    print(f"[svc] backend={backend.name} resolved", flush=True)
    sampler = Sampler(mode=backend)

    # ------------------------------------------------------------------
    # Step 1: n=1,2,3 at 1024 shots  (used for n-copies, Helstrom, VCE)
    # ------------------------------------------------------------------
    physical_curves: dict[int, list[float]] = {}
    for n in [1, 2, 3]:
        print(f"\n[sweep] n={n}, shots=1024")
        physical_curves[n] = _run_sweep(sampler, backend, copies=n, shots=1024, thetas=thetas)

    theory_by_n = {
        n: [float(analytical_swap_kernel(t, n_copies=n)) for t in thetas]
        for n in [1, 2, 3]
    }
    metrics_n = {
        f"n{n}": curve_error_metrics(np.array(physical_curves[n]), np.array(theory_by_n[n]))
        for n in [1, 2, 3]
    }

    # ------------------------------------------------------------------
    # Step 2: shots comparison — n=1 at 256 and 1024
    # ------------------------------------------------------------------
    print("\n[sweep] shots comparison n=1, shots=256")
    shots_256 = _run_sweep(sampler, backend, copies=1, shots=256, thetas=thetas)
    shots_by_count = {
        "256": shots_256,
        "1024": physical_curves[1],  # reuse 1024 run from step 1
    }

    # ------------------------------------------------------------------
    # Step 3: Write hardware_results.json
    # ------------------------------------------------------------------
    hw_results = {
        "backend": f"IBM hardware ({backend_name})",
        "shots": 1024,
        "thetas": [float(t) for t in thetas],
        "theory": {f"n{n}": theory_by_n[n] for n in [1, 2, 3]},
        "measured": {f"n{n}": physical_curves[n] for n in [1, 2, 3]},
        "metrics": metrics_n,
        "shots_by_count": shots_by_count,
    }
    hw_path = os.path.join(results_dir, "hardware_results.json")
    _save_json(hw_path, hw_results)
    print(f"\n[save] results/hardware_results.json", flush=True)

    # ------------------------------------------------------------------
    # Step 4: VCE — build virtual n=3 from n=1,2 measurements
    # ------------------------------------------------------------------
    phys_np = {n: np.array(physical_curves[n]) for n in [1, 2, 3]}
    vce_curves = build_vce_curves(phys_np, target_copies=3)
    virtual_n3 = [float(x) for x in vce_curves["virtual_n3_from_12"]]
    theory_n3 = [float(analytical_swap_kernel(t, n_copies=3)) for t in thetas]

    hw_vce_metrics = {
        "physical_n3": curve_error_metrics(np.array(physical_curves[3]), np.array(theory_n3)),
        "virtual_n3": curve_error_metrics(np.array(virtual_n3), np.array(theory_n3)),
    }

    # ------------------------------------------------------------------
    # Step 5: Update vce_summary.json with hardware section
    # ------------------------------------------------------------------
    vce_path = os.path.join(results_dir, "vce_summary.json")
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

    vce_data["hardware"] = {
        "physical_n3": physical_curves[3],
        "virtual_n3": virtual_n3,
        "metrics": hw_vce_metrics,
    }
    _save_json(vce_path, vce_data)
    print("[save] results/vce_summary.json", flush=True)

    # ------------------------------------------------------------------
    # Step 6: Regenerate all 4 plots with updated hardware data
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from visualization.plots import (
            plot_n_copies_effect, plot_helstrom_equivalence,
            plot_shots_comparison, plot_vce_novelty,
        )

        t = np.array([float(x) for x in thetas])
        theory = {n: np.array(theory_by_n[n]) for n in [1, 2, 3]}
        hw_m = {n: np.array(physical_curves[n]) for n in [1, 2, 3]}

        sim_path = os.path.join(results_dir, "simulator_results.json")
        sim = _load_json(sim_path) if os.path.exists(sim_path) else {}
        sim_m = {n: np.array(sim["measured"][f"n{n}"]) for n in [1, 2, 3]} \
            if sim.get("measured") else {}
        sim_metrics_n = sim.get("metrics", {})

        # Compute shots metrics
        theory_n1 = np.array(theory_by_n[1])
        hw_shots = {int(k): np.array(v) for k, v in shots_by_count.items()}
        sim_shots = {int(k): np.array(v) for k, v in sim.get("shots_by_count", {}).items()}
        hw_shots_metrics = {s: curve_error_metrics(v, theory_n1) for s, v in hw_shots.items()}
        sim_shots_metrics = {s: curve_error_metrics(v, theory_n1) for s, v in sim_shots.items()}

        fig = plot_n_copies_effect(t, hw_m, sim_m, theory,
            hw_metrics=metrics_n, sim_metrics=sim_metrics_n,
            save_path=os.path.join(results_dir, "01_n_copies_effect.png"))
        plt.close(fig)
        print("[save] results/01_n_copies_effect.png", flush=True)

        fig = plot_helstrom_equivalence(t, hw_m, sim_m, theory,
            save_path=os.path.join(results_dir, "02_helstrom_equivalence.png"))
        plt.close(fig)
        print("[save] results/02_helstrom_equivalence.png", flush=True)

        fig = plot_shots_comparison(t, hw_shots, sim_shots, theory[1],
            hw_metrics=hw_shots_metrics, sim_metrics=sim_shots_metrics,
            save_path=os.path.join(results_dir, "03_shots_comparison.png"))
        plt.close(fig)
        print("[save] results/03_shots_comparison.png", flush=True)

        vce_reload = _load_json(vce_path)
        fig = plot_vce_novelty(
            t,
            hw_physical_n3=np.array(vce_reload["hardware"]["physical_n3"]),
            hw_virtual_n3=np.array(vce_reload["hardware"]["virtual_n3"]),
            sim_physical_n3=np.array(vce_reload["simulator"]["physical_n3"]) if vce_reload.get("simulator") else np.zeros_like(t),
            sim_virtual_n3=np.array(vce_reload["simulator"]["virtual_n3"]) if vce_reload.get("simulator") else np.zeros_like(t),
            theory_n3=np.array(vce_reload["theory_n3"]),
            save_path=os.path.join(results_dir, "04_vce_novelty.png"),
        )
        plt.close(fig)
        print("[save] results/04_vce_novelty.png", flush=True)

    except Exception as exc:
        print(f"[warn] plot generation failed: {exc}", flush=True)

    print("\n[done] hardware suite complete", flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run IBM hardware experiments")
    ap.add_argument("--backend", default="ibm_kingston",
                    help="IBM backend name (default: ibm_kingston)")
    ap.add_argument("--quick", action="store_true", default=True,
                    help="Use 30 theta points (default: True)")
    ap.add_argument("--full", action="store_true",
                    help="Use 63 theta points instead of 30")
    ap.add_argument("--env-file", default=".env",
                    help="Path to .env file with QISKIT_IBM_* credentials")
    args = ap.parse_args()
    raise SystemExit(main(args.backend, quick=not args.full, env_file=args.env_file))
