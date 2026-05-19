import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bodies import (
    Body, Sun, Earth, Jupiter, Mars, Mercury,
    Neptune, Uranus, Saturn, Venus, Ganymede, Titan,
    Callisto, Io, Moon, Europa, Triton,
)

from integrators import rk4, abm4, abm4_rk4, leapfrog

# Constants
G  = 6.674e-20            # km^3 kg^-1 s^-2
AU = 149597870.700        # km

np.set_printoptions(linewidth=100)


class Simulation:
    def __init__(self, bodies: list[Body]) -> None:
        self.bodies   = bodies
        self.N_bodies = len(bodies)
        self.masses   = np.array([b.mass for b in bodies], dtype=float)
        self.mat      = np.array([b.return_vec() for b in bodies], dtype=float)
        self.history: np.ndarray  # filled by run()

    def f(self, t, mat):
        pos  = mat[:, :3]                          # (N, 3)
        diff = pos[None, :, :] - pos[:, None, :]   # (N, N, 3), r_ij = pos_j - pos_i
        r2   = np.sum(diff**2, axis=-1)            # (N, N)
        np.fill_diagonal(r2, np.inf)               # avoid self-interaction / div by 0
        inv_r3 = r2**-1.5                          # (N, N)
        acc  = np.einsum('ij,ijk,j->ik', inv_r3, diff, self.masses) * G
        dmat = np.zeros_like(mat)
        dmat[:, :3] = mat[:, 3:]
        dmat[:, 3:] = acc
        return dmat

    # Acceleration-only version, required by the symplectic leapfrog
    # integrator (which keeps r and v separate).
    def accel(self, pos: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(pos)
        for i in range(self.N_bodies):
            for j in range(self.N_bodies):
                if i == j:
                    continue
                r_vec = pos[j] - pos[i]
                r_abs = np.linalg.norm(r_vec)
                acc[i] += G * self.masses[j] * r_vec / r_abs**3
        return acc

    def run(self, dt: float, T: float, t0: float = 0.0,
            method: str = "rk4") -> None:
        n_steps = int((T - t0) / dt)
        history = [self.mat[:, :3].copy()]
        t = t0

        if method == "rk4":
            for _ in tqdm(range(n_steps)):
                self.mat = rk4(t, self.mat, dt, self.f)
                t += dt
                history.append(self.mat[:, :3].copy())

        elif method == "abm4":
            # abm4_rk4 does the RK4 bootstrap and the first ABM4 step,
            # returning the state at t0, t0+dt, t0+2dt, t0+3dt, t0+4dt.
            y1, y2, y3, y4, y5 = abm4_rk4(t, self.mat, dt, self.f)
            for y in (y2, y3, y4, y5):
                history.append(y[:, :3].copy())
            buf = [y2, y3, y4, y5]          # rolling window of last 4 states
            t  += 4 * dt
            for _ in tqdm(range(n_steps - 4)):
                y_new = abm4(t, *buf, dt, self.f)
                buf   = [buf[1], buf[2], buf[3], y_new]
                t    += dt
                history.append(y_new[:, :3].copy())
            self.mat = buf[-1]

        elif method == "leapfrog":
            r = self.mat[:, :3].copy()
            v = self.mat[:, 3:].copy()
            for _ in tqdm(range(n_steps)):
                r, v = leapfrog(r, v, dt, self.accel)
                history.append(r.copy())
            self.mat[:, :3], self.mat[:, 3:] = r, v

        else:
            raise ValueError(
                f"Unknown method {method!r}; use 'rk4', 'abm4', or 'leapfrog'."
            )

        self.history = np.array(history)


bodies = [
    Sun,
    Mercury,
    Venus,
    Earth,
    Moon,
    Mars,
    Jupiter,
    Io,
    Europa,
    Ganymede,
    Callisto,
    Saturn,
    Titan,
    Uranus,
    Neptune,
    Triton,
]

dt = 3600                          # 1 hour
T  = 22 * 365 * 24 * 3600           # 22 years

sim = Simulation(bodies)
sim.run(dt, T, t0=0, method="rk4")   # swap to "abm4" or "leapfrog"

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

for i, body in enumerate(sim.bodies):
    x = sim.history[:, i, 0] / AU
    y = sim.history[:, i, 1] / AU
    z = sim.history[:, i, 2] / AU

    ax.plot(x, y, z, label=body.name, linewidth=1)
    ax.scatter(x[0], y[0], z[0], s=15)   # starting position

ax.set_xlabel("x [AU]")
ax.set_ylabel("y [AU]")
ax.set_zlabel("z [AU]")
ax.set_title("16-body Solar System simulation")

ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()