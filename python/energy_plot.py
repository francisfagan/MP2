"""
energy_plot.py - relative-energy-drift comparison for rk4, abm4, leapfrog.

Usage:
    py -3.12  energy_plot.py
    py -3.12  energy_plot.py --years 2 --dt 3600 --sample-every 5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from bodies import (
    Callisto, Earth, Europa, Ganymede, Io, Jupiter, Mars, Mercury, Moon,
    Neptune, Saturn, Sun, Titan, Triton, Uranus, Venus,
)
from simulation import Simulation

BODIES = [Sun, Mercury, Venus, Earth, Moon, Mars, Jupiter,
          Io, Europa, Ganymede, Callisto, Saturn, Titan,
          Uranus, Neptune, Triton]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt",            type=float, default=7200.0)
    p.add_argument("--years",         type=float, default=5.0)
    p.add_argument("--sample-every",  type=int,   default=10,
                   help="record energy every Nth step (default 10)")
    p.add_argument("--out",           type=str,   default="energy_drift.png")
    args = p.parse_args()

    T            = args.years * 365 * 24 * 3600
    SEC_PER_YEAR = 365 * 24 * 3600

    methods = ["rk4", "abm4", "leapfrog"]
    titles  = {
        "rk4":      "RK4 step\n$y_{i+1}=y_i+\\frac{1}{6}(k_1+2k_2+2k_3+k_4)$",
        "abm4":     "ABM4 step\nAB4 predict, AM4 correct",
        "leapfrog": "Leapfrog step\n"
                    r"$v_{i+1/2}=v_{i-1/2}+a_i\Delta t$"  "\n"
                    r"$x_{i+1}=x_i+v_{i+1/2}\Delta t$",
    }

    # one fresh Simulation per method so initial conditions match across runs
    results = {}
    for m in methods:
        sim = Simulation(list(BODIES))
        sim.run(args.dt, T, 0.0, m)
        E     = sim.energy_history(args.sample_every)
        t_arr = np.arange(len(E)) * args.dt * args.sample_every
        results[m] = (t_arr, E)

    # -- side-by-side energy panels (mirrors the reference layout) --
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, m in zip(axes, methods):
        t_arr, E = results[m]
        drift = (E - E[0]) / abs(E[0])
        ax.plot(t_arr / SEC_PER_YEAR, drift, lw=1.0)
        ax.axhline(0, color='k', lw=0.5, alpha=0.5)
        ax.set_title(titles[m], fontsize=10)
        ax.set_xlabel("Time (years)")
        ax.grid(alpha=0.3)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axes[0].set_ylabel(r"$(E-E_0)/|E_0|$")
    fig.suptitle(f"Relative total-energy drift "
                 f"(dt={args.dt:g} s, {args.years} yr, {len(BODIES)} bodies)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches='tight')
    print(f"saved {args.out}")

    plt.show()


if __name__ == "__main__":
    main()
