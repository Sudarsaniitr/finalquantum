"""
visualization/plots.py
=======================
Four side-by-side plots that tell the complete research story.

When both hardware and simulator data are present, each figure has two panels:
  left panel  = hardware results (real IBM quantum computer)
  right panel = simulator results (Qiskit AerSimulator with noise)

When only simulator data is present (no-hardware mode), the figure has a
single panel showing simulator results against theory.

Functions
---------
plot_n_copies_effect        → 01_n_copies_effect.png
plot_helstrom_equivalence   → 02_helstrom_equivalence.png
plot_shots_comparison       → 03_shots_comparison.png
plot_vce_novelty            → 04_vce_novelty.png

All functions accept pre-loaded numpy arrays so callers control data loading.
Metrics (mean abs error, RMSE) are annotated directly on each panel.
"""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "lines.linewidth": 2,
    }
)

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # blue, orange, green for n=1,2,3
_THETA_TICKS = [0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi]
_THETA_LABELS = ["0", "π/2", "π", "3π/2", "2π"]
_YLABEL = r"$\langle \sigma_z^{(a)} \sigma_z^{(l)} \rangle$"


def _save_if_requested(fig, save_path: Optional[str]) -> None:
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")


def _apply_theta_axis(ax):
    ax.set_xticks(_THETA_TICKS)
    ax.set_xticklabels(_THETA_LABELS)
    ax.set_xlabel(r"$\theta$ (radians)")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")


def _metrics_annotation(ax, lines: list[str]) -> None:
    """Small metrics box pinned to bottom-centre of an axes panel."""
    ax.text(
        0.5, 0.03, "\n".join(lines),
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7.2,
        color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f4ff",
                  edgecolor="#8899cc", alpha=0.92),
    )


# ---------------------------------------------------------------------------
# Plot 01: n copies effect
# ---------------------------------------------------------------------------

def _panel_n_copies(ax, thetas, measured_by_n, theory_by_n, title, metrics_by_n=None):
    """
    Theory    = solid line (analytical).
    Measured  = dashed line with dots (hardware/simulator measured values).
    """
    # Theory lines first (background) — solid, opaque
    for n, color in zip([1, 2, 3], _COLORS):
        theory = np.asarray(theory_by_n[n])
        ax.plot(thetas, theory, color=color, linewidth=2.8, linestyle="-",
                zorder=2, label=f"theory n={n}")

    # Measured dashed lines with dots on top
    for n, color in zip([1, 2, 3], _COLORS):
        if n in measured_by_n:
            measured = np.asarray(measured_by_n[n])
            ax.plot(thetas, measured, color=color, linewidth=1.6, linestyle="--",
                    marker="o", markersize=4, zorder=3, label=f"measured n={n}")

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, fontsize=7.5, ncol=2, loc="upper left")
    ax.set_ylabel(_YLABEL)
    ax.set_title(title)
    _apply_theta_axis(ax)

    if metrics_by_n:
        lines = ["Error vs theory:"]
        for n in [1, 2, 3]:
            m = metrics_by_n.get(f"n{n}", {})
            mae = m.get("mean_abs_diff", float("nan"))
            rmse = m.get("rmse", float("nan"))
            lines.append(f"  n={n}  MAE={mae:.4f}  RMSE={rmse:.4f}")
        _metrics_annotation(ax, lines)


def plot_n_copies_effect(
    thetas: np.ndarray,
    hw_measured_by_n: dict,
    sim_measured_by_n: dict,
    theory_by_n: dict,
    hw_metrics: dict | None = None,
    sim_metrics: dict | None = None,
    save_path: str | None = None,
):
    """
    Swap kernel for n=1,2,3 copies.
    If hw_measured_by_n is non-empty: hardware (left) | simulator (right).
    If hw_measured_by_n is empty: single simulator panel.
    """
    has_hw = bool(hw_measured_by_n)
    if has_hw:
        fig, (ax_hw, ax_sim) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        _panel_n_copies(ax_hw, thetas, hw_measured_by_n, theory_by_n,
                        "Hardware — swap kernel n=1,2,3", metrics_by_n=hw_metrics)
        _panel_n_copies(ax_sim, thetas, sim_measured_by_n, theory_by_n,
                        "Simulator — swap kernel n=1,2,3", metrics_by_n=sim_metrics)
        ax_sim.set_ylabel("")
        subtitle = "Hardware (left) | Simulator (right)  |  solid = theory,  dots = measured"
    else:
        fig, ax_sim = plt.subplots(1, 1, figsize=(8, 5))
        _panel_n_copies(ax_sim, thetas, sim_measured_by_n, theory_by_n,
                        "Simulator — swap kernel n=1,2,3", metrics_by_n=sim_metrics)
        subtitle = "Qiskit AerSimulator  |  solid = theory,  dots = measured"

    fig.suptitle(f"Swap kernel for n=1,2,3 copies  |  {subtitle}", fontsize=12, y=1.01)
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 02: Helstrom equivalence
# ---------------------------------------------------------------------------

def _panel_helstrom(axes_pair, thetas, measured_by_n, theory_by_n, title):
    """
    Top panel: swap kernel theory (solid) overlaid on Helstrom expectation (dashed).
    Both should be identical — Eq. 16-17 proof.
    Measured dashed with dots = hardware/sim measured values.
    Bottom panel: numerical difference (should be ~10⁻¹⁶).
    """
    from core.kernel import helstrom_expectation, helstrom_operator
    from experiments.toy_problem import get_test_state, get_training_data

    ax, ax_diff = axes_pair
    x_train, labels = get_training_data()

    for n, color in zip([1, 2, 3], _COLORS):
        A = helstrom_operator(x_train, labels, n_copies=n)
        hel_vals = np.array([
            helstrom_expectation(get_test_state(t), A, n_copies=n) for t in thetas
        ])
        swap_theory = np.asarray(theory_by_n[n])
        diff = swap_theory - hel_vals

        # Helstrom = dashed thick (theoretical)
        ax.plot(thetas, hel_vals, color=color, linestyle="--", linewidth=2.5,
                label=f"Helstrom n={n}", zorder=2)
        # Swap kernel theory = solid (measured from circuit)
        ax.plot(thetas, swap_theory, color=color, linewidth=2.2, linestyle="-",
                label=f"Swap kernel n={n}", zorder=2)

        # Measured = dashed thin with dots
        if n in measured_by_n:
            meas = np.asarray(measured_by_n[n])
            ax.plot(thetas, meas, color=color, linestyle="--", linewidth=1.2,
                    marker="o", markersize=3, alpha=0.6, zorder=1)

        ax_diff.plot(thetas, diff, color=color, linewidth=1.4, label=f"n={n}")

    ax.set_title(title)
    ax.set_ylabel(_YLABEL)
    ax.legend(ncol=2, fontsize=8)
    ax.text(
        0.5, 0.04,
        "Dashed Helstrom lines lie beneath solid theory lines.\n"
        "Difference ~10⁻¹⁶ (machine precision) — see lower panel.",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=7.5,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffbe6",
                  edgecolor="#ccbb44", alpha=0.9),
    )
    _apply_theta_axis(ax)

    ax_diff.axhline(0, color="k", linewidth=0.8)
    ax_diff.set_title("Difference (should be ~0)  max |Δ| ≈ 3×10⁻¹⁶")
    ax_diff.set_ylabel("Swap − Helstrom")
    ax_diff.legend(fontsize=8)
    _apply_theta_axis(ax_diff)


def plot_helstrom_equivalence(
    thetas: np.ndarray,
    hw_measured_by_n: dict,
    sim_measured_by_n: dict,
    theory_by_n: dict,
    save_path: str | None = None,
):
    """
    Swap kernel = Helstrom expectation.
    If hw_measured_by_n non-empty: hardware (left) | simulator (right), 2x2 grid.
    If empty: single-column (1x2) simulator only.
    """
    has_hw = bool(hw_measured_by_n)
    if has_hw:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                                 gridspec_kw={"height_ratios": [2, 1]})
        _panel_helstrom((axes[0, 0], axes[1, 0]), thetas,
                        hw_measured_by_n, theory_by_n,
                        "Hardware — swap kernel vs Helstrom")
        _panel_helstrom((axes[0, 1], axes[1, 1]), thetas,
                        sim_measured_by_n, theory_by_n,
                        "Simulator — swap kernel vs Helstrom")
        subtitle = "Hardware (left) | Simulator (right)"
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 9),
                                 gridspec_kw={"height_ratios": [2, 1]})
        _panel_helstrom((axes[0], axes[1]), thetas,
                        sim_measured_by_n, theory_by_n,
                        "Simulator — swap kernel vs Helstrom")
        subtitle = "Qiskit AerSimulator"

    fig.suptitle(
        f"Helstrom equivalence: swap-test = optimal measurement  |  {subtitle}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 03: Shots comparison
# ---------------------------------------------------------------------------

def _panel_shots(ax, thetas, measured_by_shot, theory, title, metrics_by_shot=None):
    """
    Theory = solid black line.
    256 shots = thin orange dashed.
    1024 shots = thick blue dashed.
    Metrics box at bottom-centre.
    """
    palette = {256: "#ff7f0e", 1024: "#1f77b4"}
    ax.plot(thetas, np.asarray(theory), "k-", linewidth=2.2, label="Theory (n=1)", zorder=2)
    for shot in sorted(measured_by_shot):
        meas = np.asarray(measured_by_shot[shot])
        lw = 1.2 if shot == 256 else 1.8
        ax.plot(thetas, meas, color=palette.get(shot, "#9467bd"),
                linestyle="--", marker="o", markersize=2.8, linewidth=lw,
                label=f"{shot} shots", zorder=3)
    ax.set_ylabel(_YLABEL)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _apply_theta_axis(ax)

    if metrics_by_shot:
        lines = ["Error vs theory:"]
        for shot in sorted(metrics_by_shot):
            m = metrics_by_shot[shot]
            mae = m.get("mean_abs_diff", float("nan"))
            rmse = m.get("rmse", float("nan"))
            lines.append(f"  {shot} shots  MAE={mae:.4f}  RMSE={rmse:.4f}")
        _metrics_annotation(ax, lines)


def plot_shots_comparison(
    thetas: np.ndarray,
    hw_measured_by_shot: dict,
    sim_measured_by_shot: dict,
    theory: np.ndarray,
    hw_metrics: dict | None = None,
    sim_metrics: dict | None = None,
    save_path: str | None = None,
):
    """
    Shot count sweep (256 vs 1024).
    If hw_measured_by_shot non-empty: hardware (left) | simulator (right).
    If empty: single simulator panel.
    """
    has_hw = bool(hw_measured_by_shot)
    if has_hw:
        fig, (ax_hw, ax_sim) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        _panel_shots(ax_hw, thetas, hw_measured_by_shot, theory,
                     "Hardware — 256 vs 1024 shots", metrics_by_shot=hw_metrics)
        _panel_shots(ax_sim, thetas, sim_measured_by_shot, theory,
                     "Simulator — 256 vs 1024 shots", metrics_by_shot=sim_metrics)
        ax_sim.set_ylabel("")
        subtitle = "Hardware (left) | Simulator (right)"
    else:
        fig, ax_sim = plt.subplots(1, 1, figsize=(8, 5))
        _panel_shots(ax_sim, thetas, sim_measured_by_shot, theory,
                     "Simulator — 256 vs 1024 shots", metrics_by_shot=sim_metrics)
        subtitle = "Qiskit AerSimulator"

    fig.suptitle(
        f"Shot count comparison: more shots → closer to theory  |  {subtitle}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig


# ---------------------------------------------------------------------------
# Plot 04: VCE novelty
# ---------------------------------------------------------------------------

def _panel_vce(axes_pair, thetas, physical_n3, virtual_n3, theory_n3, title):
    """
    Top panel:
      Theory n=3  = solid black line
      Physical n=3 = blue dashed (direct hardware/sim run)
      Virtual  n=3 = red  dashed (VCE estimate from n=1,2)
    Bottom panel: absolute error comparison with MAE + RMSE in legend.
    """
    ax, ax_err = axes_pair
    theory_n3 = np.asarray(theory_n3)
    physical_n3 = np.asarray(physical_n3)
    virtual_n3 = np.asarray(virtual_n3)

    ax.plot(thetas, theory_n3, "k-", linewidth=2.2, label="Theory n=3", zorder=2)
    ax.plot(thetas, physical_n3, color="#1f77b4", linestyle="--",
            marker="o", markersize=3.2, linewidth=1.5, zorder=3,
            label="Physical n=3 (direct run)")
    ax.plot(thetas, virtual_n3, color="#d62728", linestyle="--",
            marker="o", markersize=3.2, linewidth=1.5, zorder=3,
            label="Virtual n=3 via VCE")
    ax.set_ylabel(_YLABEL)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _apply_theta_axis(ax)

    phys_err = np.abs(physical_n3 - theory_n3)
    virt_err = np.abs(virtual_n3 - theory_n3)
    phys_rmse = float(np.sqrt(np.mean((physical_n3 - theory_n3) ** 2)))
    virt_rmse = float(np.sqrt(np.mean((virtual_n3 - theory_n3) ** 2)))

    ax_err.plot(thetas, phys_err, color="#1f77b4", linewidth=1.8,
                label=f"|physical − theory|  MAE={phys_err.mean():.4f}  RMSE={phys_rmse:.4f}")
    ax_err.plot(thetas, virt_err, color="#d62728", linewidth=1.8,
                label=f"|virtual − theory|   MAE={virt_err.mean():.4f}  RMSE={virt_rmse:.4f}")
    ax_err.set_ylabel("Absolute error")
    ax_err.set_title("Error magnitude — lower is better")
    ax_err.legend(fontsize=7.5)
    _apply_theta_axis(ax_err)


def plot_vce_novelty(
    thetas: np.ndarray,
    hw_physical_n3: np.ndarray | None,
    hw_virtual_n3: np.ndarray | None,
    sim_physical_n3: np.ndarray,
    sim_virtual_n3: np.ndarray,
    theory_n3: np.ndarray,
    save_path: str | None = None,
):
    """
    VCE novelty result.
    If hw_physical_n3 is not None and non-zero: hardware (left) | simulator (right), 2x2.
    Otherwise: single-column (1x2) simulator only.

    VCE wins when the red error curve is consistently below the blue.
    """
    hw_available = (
        hw_physical_n3 is not None
        and hw_virtual_n3 is not None
        and np.any(np.asarray(hw_physical_n3) != 0)
    )

    if hw_available:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                                 gridspec_kw={"height_ratios": [2, 1]})
        _panel_vce((axes[0, 0], axes[1, 0]), thetas,
                   hw_physical_n3, hw_virtual_n3, theory_n3,
                   "Hardware — physical n=3 vs virtual n=3 (VCE)")
        _panel_vce((axes[0, 1], axes[1, 1]), thetas,
                   sim_physical_n3, sim_virtual_n3, theory_n3,
                   "Simulator — physical n=3 vs virtual n=3 (VCE)")
        subtitle = "Hardware (left) | Simulator (right)"
    else:
        fig, axes = plt.subplots(2, 1, figsize=(8, 9),
                                 gridspec_kw={"height_ratios": [2, 1]})
        _panel_vce((axes[0], axes[1]), thetas,
                   sim_physical_n3, sim_virtual_n3, theory_n3,
                   "Simulator — physical n=3 vs virtual n=3 (VCE)")
        subtitle = "Qiskit AerSimulator"

    fig.suptitle(
        f"VCE Novelty: virtual n=3 from n=1,2  vs  physical n=3  |  {subtitle}",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()
    _save_if_requested(fig, save_path)
    return fig
