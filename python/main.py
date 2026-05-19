import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from bodies import (
    Body,
    Sun,
    Earth,
    Jupiter,
    Mars,
    Mercury,
    Neptune,
    Uranus,
    Venus,
    Ganymede,
    Titan,
    Callisto,
    Io,
    Moon,
    Europa,
    Triton,
)

# constants
G = 6.674e-20
Au = 149597870.700  # in kilometers

np.set_printoptions(linewidth=100) # simply allows printing matrices to the screen to be more readable


class Simulation:
    def __init__(self, bodies: list[Body]) -> None:
        self.bodies = bodies
        self.N_bodies = len(bodies)
        self.Ndim = 6
        self.masses = np.array([body.mass for body in bodies], dtype=float)
        self.mat = self.initiliase_matrix(bodies)
        self.history : np.ndarray # leave history as just array for now, uninitialised

    def initiliase_matrix(self, bodies: list[Body]) -> np.ndarray:
        return np.array([body.return_vec() for body in bodies], dtype=float)

    def print_matrix(self) -> None:
        print(self.mat)

    def dmatrix(self, mat: np.ndarray) -> np.ndarray:
        """
        This function takes in the matrix that consists of the velocity and the position, where the first three coloumns ae
        It returns a matrix of acceleration and velocity, where velocity is the first three coloumns and acceleration is the last three coloumns.
        """

        pos = mat[:, :3]  # (N, 3)
        dmat = np.zeros_like(mat)
        dmat[:, :3] = mat[:, 3:]  # dr/dt, i.e. v

        for i in range(self.N_bodies):
            acc = np.zeros(3)  # set empty acceleration vector to zero
            for j in range(self.N_bodies):
                if i == j:  # obviously not calculating gravity due to itself
                    continue
                r_vec = pos[j] - pos[i]  # distance between objects
                r_abs = np.linalg.norm(r_vec)
                acc += (G * self.masses[j] * r_vec) / (r_abs**3)
            dmat[i, 3:] = acc  # dv/dt, i.e. a

        return dmat

    def rk4(self, mat: np.ndarray, dt: int) -> np.ndarray:
        k1 = self.dmatrix(mat)
        k2 = self.dmatrix(mat + 0.5 * dt * k1)
        k3 = self.dmatrix(mat + 0.5 * dt * k2)
        k4 = self.dmatrix(mat + dt * k3)

        return mat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self, dt: int, T: int, t0: int) -> None:
        t = t0
        n_steps = int((T - t0)/dt)
        history = [self.mat[:, :3].copy()] # only need positions, forgo the velocity columns
        for _ in tqdm(range(n_steps)): # just a progress bar
            self.mat = self.rk4(self.mat, dt) # rk4 iterations
            history.append(self.mat[:, :3].copy())
        self.history = np.array(history) # return history at the end for plotting

bodies = [
    Sun,
    Earth,
    Jupiter,
    Mars,
    Mercury,
    # Neptune,
    # Uranus,
    Venus,
    # Ganymede,
    # Titan,
    # Callisto,
    # Io,
    Moon,
    # Europa,
    # Triton,
]

dt = 3600 # 1 hour time steps
T = 2 * 365 * 24 * 3600 # simulation lasts a year

mySim = Simulation(bodies)
mySim.run(dt, T, t0=0)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, body in enumerate(mySim.bodies):
    x = mySim.history[:, i, 0]
    y = mySim.history[:, i, 1]
    z = mySim.history[:, i, 2]
    ax.plot(x, y, z, label=body.name)

# ax.legend() # can uncomment for legend, blocks the view tho
plt.show()
