"""
scripts/metrics_report.py
==========================
Print a full error-metrics report for all experiments.

Reads the 3 clean JSON files in results/ and prints a formatted table of
mean absolute error (MAE) and RMSE for every measured curve vs its theory.

Usage
-----
  python scripts/metrics_report.py
  python scripts/metrics_report.py --save   # also writes results/metrics_report.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

RESULTS = os.path.join(ROOT, "results")


def _load(name: str) -> dict | None:
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _row(label: str, m: dict) -> str:
    mae = m.get("mean_abs_diff", float("nan"))
    rmse = m.get("rmse", float("nan"))
    sign = m.get("sign_agreement", float("nan"))
    return f"  {label:<42}  MAE={mae:.4f}  RMSE={rmse:.4f}  sign={sign*100:.1f}%"


def build_report() -> str:
    lines = []

    # ---- Plot 01 / 03 metrics: n=1,2,3 kernel vs theory ----
    hw = _load("hardware_results.json")
    sim = _load("simulator_results.json")

    lines.append("=" * 72)
    lines.append("  PLOT 01 — Swap kernel n=1,2,3 vs Theory")
    lines.append("=" * 72)

    if hw:
        lines.append(f"\n  Backend: {hw.get('backend', 'hardware')}  ({hw.get('shots', '?')} shots)")
        for n in [1, 2, 3]:
            m = hw.get("metrics", {}).get(f"n{n}", {})
            lines.append(_row(f"n={n} measured vs theory n={n}", m))
    else:
        lines.append("  hardware_results.json not found — run scripts/hardware_suite.py")

    if sim:
        lines.append(f"\n  Backend: {sim.get('backend', 'simulator')}  ({sim.get('shots', '?')} shots)")
        for n in [1, 2, 3]:
            m = sim.get("metrics", {}).get(f"n{n}", {})
            lines.append(_row(f"n={n} measured vs theory n={n}", m))
    else:
        lines.append("  simulator_results.json not found — run: python main.py --generate-results")

    # ---- Plot 03 metrics: shots comparison ----
    lines.append("")
    lines.append("=" * 72)
    lines.append("  PLOT 03 — Shot count comparison (n=1) vs Theory")
    lines.append("=" * 72)

    import numpy as np
    if hw and "shots_by_count" in hw and "theory" in hw:
        theory_n1 = np.array(hw["theory"]["n1"])
        lines.append(f"\n  Backend: {hw.get('backend', 'hardware')}")
        for shot_str, curve in sorted(hw["shots_by_count"].items()):
            arr = np.array(curve)
            mae = float(np.mean(np.abs(arr - theory_n1)))
            rmse = float(np.sqrt(np.mean((arr - theory_n1) ** 2)))
            sign = float(np.mean(np.sign(arr) == np.sign(theory_n1)))
            lines.append(_row(f"{shot_str} shots vs theory", {
                "mean_abs_diff": mae, "rmse": rmse, "sign_agreement": sign
            }))
    if sim and "shots_by_count" in sim and "theory" in sim:
        theory_n1 = np.array(sim["theory"]["n1"])
        lines.append(f"\n  Backend: {sim.get('backend', 'simulator')}")
        for shot_str, curve in sorted(sim["shots_by_count"].items()):
            arr = np.array(curve)
            mae = float(np.mean(np.abs(arr - theory_n1)))
            rmse = float(np.sqrt(np.mean((arr - theory_n1) ** 2)))
            sign = float(np.mean(np.sign(arr) == np.sign(theory_n1)))
            lines.append(_row(f"{shot_str} shots vs theory", {
                "mean_abs_diff": mae, "rmse": rmse, "sign_agreement": sign
            }))

    # ---- Plot 04 metrics: VCE novelty ----
    vce = _load("vce_summary.json")

    lines.append("")
    lines.append("=" * 72)
    lines.append("  PLOT 04 — VCE Novelty: physical n=3 vs virtual n=3")
    lines.append("=" * 72)
    lines.append("  (virtual n=3 = estimated from n=1,2 via Richardson denoising)")

    if vce:
        for section_key, label in [("hardware", "Hardware"), ("simulator", "Simulator")]:
            sec = vce.get(section_key)
            if not sec:
                continue
            lines.append(f"\n  {label} ({vce.get('shots', 1024)} shots):")
            for curve_key in ["physical_n3", "virtual_n3"]:
                m = sec.get("metrics", {}).get(curve_key, {})
                lines.append(_row(curve_key, m))
        # Improvement summary
        hw_sec = vce.get("hardware", {})
        if hw_sec.get("metrics"):
            phys_mae = hw_sec["metrics"].get("physical_n3", {}).get("mean_abs_diff", float("nan"))
            virt_mae = hw_sec["metrics"].get("virtual_n3", {}).get("mean_abs_diff", float("nan"))
            phys_rmse = hw_sec["metrics"].get("physical_n3", {}).get("rmse", float("nan"))
            virt_rmse = hw_sec["metrics"].get("virtual_n3", {}).get("rmse", float("nan"))
            if not (phys_mae != phys_mae):  # not NaN
                imp_mae = (phys_mae - virt_mae) / phys_mae * 100
                imp_rmse = (phys_rmse - virt_rmse) / phys_rmse * 100
                lines.append(f"\n  VCE improvement (hardware):")
                lines.append(f"    MAE:  {phys_mae:.4f} -> {virt_mae:.4f}  ({imp_mae:+.1f}%)")
                lines.append(f"    RMSE: {phys_rmse:.4f} -> {virt_rmse:.4f}  ({imp_rmse:+.1f}%)")
    else:
        lines.append("  vce_summary.json not found.")

    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Print error metrics for all experiments")
    ap.add_argument("--save", action="store_true",
                    help="Also save report to results/metrics_report.txt")
    args = ap.parse_args()

    report = build_report()
    print(report)

    if args.save:
        out = os.path.join(RESULTS, "metrics_report.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Saved: results/metrics_report.txt")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
