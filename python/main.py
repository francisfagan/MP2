import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import argparse

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

# parser for cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=86400, help="timestep in seconds (default = 86400s)")
parser.add_argument("--years", type=int, default=10, help="simulation duration in years (default = 10 years)")
parser.add_argument("--visual", type=str, default="plot", choices=["plot", "ani"], help="choose plot or animation")
parser.add_argument("--interval", type=float, default=30, help="controls speed of animation. default is 30. higher is slower, lower is faster.")
args = parser.parse_args()



# constants
Au = 149597870.700  # in kilometers,
G = 6.674e-20 / Au ** 3 # km-3 * km^3

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
        return np.array([body.return_vec()/Au for body in bodies], dtype=float)

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

    def rk4(self, mat: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.dmatrix(mat)
        k2 = self.dmatrix(mat + 0.5 * dt * k1)
        k3 = self.dmatrix(mat + 0.5 * dt * k2)
        k4 = self.dmatrix(mat + dt * k3)

        return mat + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def run(self, dt: int, T: int, t0: int) -> None:
        n_steps = int((T - t0)/dt)
        history = [self.mat[:, :3].copy()] # only need positions, forgo the velocity columns
        for _ in tqdm(range(n_steps)): # just a progress bar
            self.mat = self.rk4(self.mat, dt) # rk4 iterations
            history.append(self.mat[:, :3].copy())
        self.history = np.array(history) # return history at the end for plotting

    def animate(self, n: int, legend: bool = True) -> None:
        """
        Animates the simulation results. Still in progress.

        Takes two inputs n and legend. 
        Think of n like a multiplier, the higher it is the faster the simulation will be.
        Legend takes a boolean, true or false, and just controls whether there is a legend or not.

        """
        fig = plt.figure(figsize=(20, 20), facecolor='black')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('black')
        ax.grid(False)

        trails = [ax.plot([], [], [], linewidth = 0.5)[0] for _ in self.bodies]
        dots = [ax.scatter([], [], s=10, label=body.name) for body in self.bodies]

        # set limits for animation bounds
        lim = np.abs(self.history).max()
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        # disable grey 3d plane
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('none')
        ax.yaxis.pane.set_edgecolor('none')
        ax.zaxis.pane.set_edgecolor('none')

        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white')

        def update(frame):
            time_text.set_text(f'Day {frame * n}, Year {frame * n /365:.2f}')
            trail = 2000 # controls how long the trail is behind the object
            start = max(0, frame - trail)
            for i in range(self.N_bodies):
                x = self.history[start:frame, i, 0]
                y = self.history[start:frame, i, 1]
                z = self.history[start:frame, i, 2]
                trails[i].set_data(x, y)
                trails[i].set_3d_properties(z)
                dots[i]._offsets3d = (x[-1:], y[-1:], z[-1:])
            return trails + dots + [time_text]

        frames = range(1, len(self.history), n) # goes from beginning to end in steps of n
        ani = FuncAnimation(fig, update, frames=frames, interval=args.interval, blit=False)

        if legend:
            ax.legend(labelcolor='white', facecolor='black')
        # plt.tight_layout()
        plt.show()

    def plot(self) -> None:
        """
        Simulation function for plotting the overall arc that each object has taken.
        Takes in itself as an input and returns nothing.
        Displays plot.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.grid(False) # turn off grid

        # comment out if tick labels wanted
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.zaxis.set_ticklabels([])

        for i, body in enumerate(self.bodies):
            x = mySim.history[:, i, 0]
            y = mySim.history[:, i, 1]
            z = mySim.history[:, i, 2]
            ax.plot(x, y, z, label=body.name)

            # ax.legend() # can uncomment for legend, blocks the view tho
        plt.show()


# comment out or in whichever is needed
bodies = [
    Sun,
    Earth,
    Jupiter,
    Mars,
    Mercury,
    # Neptune,
    Uranus,
    Venus,
    # Ganymede,
    # Titan,
    # Callisto,
    # Io,
    Moon,
    # Europa,
    # Triton,
]



# simulation arguments
dt = args.dt
T = args.years * 365 *24 * 3600

# overrides for T and dt
# dt = 24 * 3600 # 1 hour time steps
# n = 10
# T = n * 365 * 24 * 3600 # simulation lasts n years

mySim = Simulation(bodies)
mySim.run(dt, T, t0=0)

# plot or animation
if (args.visual == "plot"):
    mySim.plot()
elif (args.visual == "ani"):
    mySim.animate(10, legend=True)
