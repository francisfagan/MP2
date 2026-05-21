#from vpython_test import FPS
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.animation import FuncAnimation
import argparse
from vpython import button, canvas, color, label, rate, sphere, vector, wtext

from animation_funcs import *
from pathlib import Path
from integrators import rk4, abm4, abm4_rk4, leapfrog

from bodies import (
    Body,
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
)
paused = False

# parser for cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=86400, help="timestep in seconds (default = 86400s)")
parser.add_argument("--years", type=int, default=10, help="simulation duration in years (default = 10 years)")
parser.add_argument("--method", type=str, default="abm4", choices=["rk4", "abm4", "leapfrog"], help="choose simulation method")
parser.add_argument("--visual", type=str, default="plot", choices=["plot", "anim"], help="choose plot or animation")
parser.add_argument("--interval", type=float, default=30, help="controls speed of animation. default is 30. higher is slower, lower is faster.")
parser.add_argument("--fps", type=float, default=20, help="VPython animation frame rate; lower values slow the animation")
parser.add_argument("--texture_dir", type=str, default="textures", help="choose directory for body textures. Texture names must match planet names")
args = parser.parse_args()

# simulation arguments
dt = args.dt
T = args.years * 365 *24 * 3600
user_method = args.method
animation_fps = args.fps


# Constants
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
        self.spheres = []
        self.scene : canvas

    def initiliase_matrix(self, bodies: list[Body]) -> np.ndarray:
        return np.array([body.return_vec()/Au for body in bodies], dtype=float)
    


    def print_matrix(self) -> None:
        print(self.mat)

    # Calculates just the acceleration for each body at time t
    # 
    def accel_matrix(self, pos: np.ndarray) -> np.ndarray:
        diff = pos[None, :, :] - pos[:, None, :]
        r2   = np.sum(diff**2, axis=-1)
        np.fill_diagonal(r2, np.inf)
        inv_r3 = r2**-1.5
        return np.einsum('ij,ijk,j->ik', inv_r3, diff, self.masses) * G
            
    def dmatrix(self, t, mat: np.ndarray) -> np.ndarray:
        """
        This function takes in the matrix that consists of the velocity and the position, where the first three coloumns are position, last 3 are velocity
        It returns a matrix of acceleration and velocity, where velocity is the first three coloumns and acceleration is the last three coloumns.
        """

        pos = mat[:, :3]  # (N, 3)
        dmat = np.zeros_like(mat)
        dmat[:, :3] = mat[:, 3:]  # dr/dt, i.e. v

        dmat[:, 3:] = self.accel_matrix(pos)
        return dmat




    # Main method called for running sim.
    #   -Initialises animation scene
    #   -Calculates body positions one time step forward using chosen method
    #   -Updates positions of bodies in animation
    def run(self, dt: float, T: float, t0: float = 0.0,
            method: str = "abm4") -> None:
        n_steps = int((T - t0) / dt)
        history = [self.mat[:, :3].copy()]
        t = t0

        if args.visual == 'anim':
            display_skip = 10
            rate(animation_fps)
            sun_idx = self.bodies.index(Sun)
            self.scene, self.spheres, name_labels, HOST_INDEX, time_text = initialise_animation(
                self.mat, method, self.bodies, sun_idx
            )

            # Create pause button
            def _toggle_pause(b):
                global paused
                paused = not paused
                b.text = "Resume" if paused else "Pause"
                for lbl in name_labels:
                    lbl.visible = paused
            self.scene.append_to_caption("\n")
            button(text="Pause", bind=_toggle_pause)

        if method == "rk4":
            for step in tqdm(range(n_steps)):
                while paused:
                    rate(animation_fps)
                self.mat = rk4(t, self.mat, dt, self.dmatrix)
                t += dt
                history.append(self.mat[:, :3].copy())
                if args.visual == 'anim':
                    if step % display_skip == 0:
                        self.advance_anim(time_text, sun_idx, name_labels, t, method, HOST_INDEX)
                        rate(animation_fps)

        elif method == "abm4":
            # abm4_rk4 does the RK4 bootstrap and the first ABM4 step,
            # returning the state at t0, t0+dt, t0+2dt, t0+3dt, t0+4dt.
            y1, y2, y3, y4, y5 = abm4_rk4(t, self.mat, dt, self.dmatrix)
            for y in (y2, y3, y4, y5):
                history.append(y[:, :3].copy())
            buf = [y2, y3, y4, y5]          # rolling window of last 4 states
            t  += 4 * dt
            for step in tqdm(range(n_steps - 4)):
                while paused:
                    rate(animation_fps)
                y_new = abm4(t, *buf, dt, self.dmatrix)
                buf   = [buf[1], buf[2], buf[3], y_new]
                t    += dt
                history.append(y_new[:, :3].copy())
                self.mat = y_new
                if args.visual == 'anim':
                    if step % display_skip == 0:
                        self.advance_anim(time_text, sun_idx, name_labels, t, method, HOST_INDEX)
                        rate(animation_fps)
            self.mat = buf[-1]

        elif method == "leapfrog":
            r = self.mat[:, :3].copy()
            v = self.mat[:, 3:].copy()
            for step in tqdm(range(n_steps)):
                while paused:
                    rate(animation_fps)
                r, v = leapfrog(r, v, dt, self.accel_matrix)
                t += dt
                #history.append(r.copy())
                self.mat[:, :3], self.mat[:, 3:] = r, v
                if args.visual == 'anim':
                    if step % display_skip == 0:
                        self.advance_anim(time_text, sun_idx, name_labels, t, method, HOST_INDEX)
                        rate(animation_fps)


        else:
            raise ValueError(
                f"Unknown method {method!r}; use 'rk4', 'abm4', or 'leapfrog'."
            )

        self.history = np.array(history)

    def advance_anim(self, time_text, sun_idx, name_labels, t, method, HOST_INDEX) -> None:
        """
        Animates the simulation results. Still in progress.

        Takes two inputs n and legend. 
        Think of n like a multiplier, the higher it is the faster the simulation will be.
        Legend takes a boolean, true or false, and just controls whether there is a legend or not.

        """
        state = self.mat


        # # Advance physics several steps per frame so the animation is fast enough.
        # for _ in range(STEPS_PER_FRAME):
        #     state, t = advance(state, t)
        #     if t >= T_END:
        #         break

        # Recenter on the Sun and push new positions to the spheres and labels.
        origin = state[sun_idx, :3]

        for i, s in enumerate(self.spheres):
            new_pos = display_position(i, state, origin, HOST_INDEX)
            s.pos = new_pos
            name_labels[i].pos = new_pos

        days  = t / 86400.0
        years = days / 365.0
        time_text.text = f"Day {days:,.1f}   Year {years:,.2f}   method = {method}"




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




mySim = Simulation(bodies)

mySim.run(dt, T, t0=0, method=user_method)

# plot or animation
if (args.visual == "plot"):
    mySim.plot()
# elif (args.visual == "anim"):
#     mySim.animate(10, legend=True)
