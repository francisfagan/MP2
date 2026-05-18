import numpy as np
import matplotlib.pyplot as plt
from bodies import Body, Sun, Earth, Jupiter, Mars, Mercury, Neptune, Uranus, Venus, Ganymede

# constants
G = 6.674e-20
Au = 149597870.700  # in kilometers

np.set_printoptions(linewidth=100)


class Simulation:
    def __init__(self, bodies) -> None:
        self.bodies = bodies
        self.N_bodies = len(bodies)
        self.Ndim = 6
        self.masses = np.array([body.mass for body in bodies], dtype=float)
        self.mat = self.initiliase_matrix(bodies)

    def initiliase_matrix(self, bodies) -> np.ndarray:
        return np.array([body.return_vec() for body in bodies], dtype=float)

    def print_matrix(self) -> None:
        print(self.mat)

    def dmatrix(self, mat) -> np.ndarray:
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

    def rk4(self, mat, dt) -> np.ndarray:
        k1 = self.dmatrix(mat)
        k2 = self.dmatrix(mat + 0.5 * dt * k1)
        k3 = self.dmatrix(mat + 0.5 * dt * k2)
        k4 = self.dmatrix(mat + dt * k3)

        return mat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self, dt, T, t0) -> None:
        t = t0
        n = 0
        while t < T:
            self.mat = self.rk4(self.mat, dt)
            print(self.mat)
            n += 1
            t += dt
        print(n)


bodies = [Sun, Earth, Jupiter, Mars, Mercury, Neptune, Uranus, Venus, Ganymede]
T = 3 * 86400  # one year in seconds
dt = 86400
mySim = Simulation(bodies)
mySim.run(dt, T, 0)
