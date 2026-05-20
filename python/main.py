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
paused = False

# parser for cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=86400, help="timestep in seconds (default = 86400s)")
parser.add_argument("--years", type=int, default=10, help="simulation duration in years (default = 10 years)")
parser.add_argument("--method", type=str, default="abm4", choices=["rk4", "abm4", "leapfrog"], help="choose simulation method")
parser.add_argument("--visual", type=str, default="plot", choices=["plot", "anim"], help="choose plot or animation")
parser.add_argument("--interval", type=float, default=30, help="controls speed of animation. default is 30. higher is slower, lower is faster.")
parser.add_argument("--texture_dir", type=str, default="textures", help="choose directory for body textures. Texture names must match planet names")
args = parser.parse_args()

# simulation arguments
dt = args.dt
T = args.years * 365 *24 * 3600
user_method = args.method


# Constants
Au = 149597870.700  # in kilometers,
G = 6.674e-20 / Au ** 3 # km-3 * km^3

# # --- simulation settings ----------------------------------------------------
# NEPTUNE_PERIOD_YEARS = 164.79         # Neptune's orbital period (years)
# STEPS_PER_FRAME = 24                 # advance 10 simulated days per drawn frame
# FPS         = 60                      # max frames per second
# TRAIL_LEN   = 4000                    # points kept per orbital trail
# SCENE_RANGE_AU = 35.0                 # initial zoom (AU)
# MOON_ORBIT_SCALE = 150.0              # visual-only magnification of moon-to-host offsets
# TEXTURE_DIR = "../textures"              # folder with <BodyName>.jpg|png; set to None to disable

# # Each satellite is drawn at host_pos + (sat_pos - host_pos) * MOON_ORBIT_SCALE,
# # which makes the tiny moon orbits visible without touching the physics.
# HOST_BY_SATELLITE = {
#     "Moon":     "Earth",
#     "Io":       "Jupiter",
#     "Europa":   "Jupiter",
#     "Ganymede": "Jupiter",
#     "Callisto": "Jupiter",
#     "Titan":    "Saturn",
#     "Triton":   "Neptune",
# }

# # Visual-only radius multiplier per body. Dynamics use the *real* mass/radius;
# # this only fattens the spheres so you can see them.
# RADIUS_MULT = {
#     "Sun":   35,   "Mercury": 2500, "Venus": 2000, "Earth": 2000,
#     "Moon":  4500, "Mars": 2500,    "Jupiter": 500, "Io": 5500,
#     "Europa":5500, "Ganymede":5000, "Callisto":5000, "Saturn":550,
#     "Titan": 5000, "Uranus":900,    "Neptune":900,  "Triton":5500,
# }
# COLOURS = {
#     "Sun":      color.yellow,
#     "Mercury":  vector(0.60, 0.55, 0.50),
#     "Venus":    vector(0.95, 0.75, 0.35),
#     "Earth":    color.blue,
#     "Moon":     vector(0.70, 0.70, 0.75),
#     "Mars":     color.red,
#     "Jupiter":  vector(0.90, 0.55, 0.25),
#     "Io":       vector(1.00, 0.90, 0.30),
#     "Europa":   vector(0.85, 0.85, 0.65),
#     "Ganymede": color.cyan,
#     "Callisto": vector(0.50, 0.65, 0.95),
#     "Saturn":   vector(0.95, 0.80, 0.45),
#     "Titan":    vector(0.95, 0.55, 0.20),
#     "Uranus":   vector(0.40, 1.00, 0.80),
#     "Neptune":  vector(0.45, 0.35, 1.00),
#     "Triton":   vector(0.75, 0.55, 0.80),
# }

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

    def initiliase_matrix(self, bodies: list[Body]) -> np.ndarray:
        return np.array([body.return_vec()/Au for body in bodies], dtype=float)
    







    # Setup animation canvas
    # Create pause button for animation
    # find texture files for planets if available in designated folder.
    # 
    # 
    # def initialise_animation(self, state) -> None:
    #     self.scene = canvas(
    #     title=f"16-body solar system — {user_method.upper()}",
    #     width=1400, height=850, background=color.black,
    #     )
    #     self.scene.range   = SCENE_RANGE_AU
    #     self.scene.forward = vector(-1.15, -0.75, -0.55)   # oblique view
    #     self.scene.up      = vector(0, 0, 1)

    #     # Create pause button
    #     paused = False
    #     def _toggle_pause(b):
    #         global paused
    #         paused = not paused
    #         b.text = "Resume" if paused else "Pause"
    #         for lbl in name_labels:
    #             lbl.visible = paused

    #     self.scene.append_to_caption("\n")
    #     button(text="Pause", bind=_toggle_pause)
    #     self.scene.append_to_caption("    ")

    #     time_text = wtext(text="")

    #     #Search in given texture directory for textures for all bodies modelled
    #     _TEXTURES = load_textures(self.bodies, tex)

    #     NAME_TO_INDEX = {b.name: i for i, b in enumerate(self.bodies)}
    #     HOST_INDEX = {
    #         NAME_TO_INDEX[sat]: NAME_TO_INDEX[host]
    #         for sat, host in HOST_BY_SATELLITE.items()
    #         if sat in NAME_TO_INDEX and host in NAME_TO_INDEX
    #     }
    #     # Build spheres, all positioned relative to the Sun so the Sun stays at origin.
    #     sun_idx = self.bodies.index(Sun)
    #     origin  = state[sun_idx, :3].copy()

    #     name_labels = []
    #     for i, b in enumerate(self.bodies):
    #         is_moon = b.name in HOST_BY_SATELLITE
    #         pos = display_position(i, state, origin, HOST_INDEX)
    #         tex = _TEXTURES.get(b.name)
    #         kwargs = dict(
    #             pos        = pos,
    #             radius     = visual_radius(b),
    #             color      = color.white if tex is not None else COLOURS.get(b.name, color.white),
    #             emissive   = (b.name == "Sun"),
    #             make_trail = not is_moon,                 # moons don't leave trails
    #             retain     = TRAIL_LEN,
    #             trail_color= COLOURS.get(b.name, color.white),
    #         )
    #         if tex is not None:
    #             kwargs["texture"] = tex
    #         s = sphere(**kwargs)
    #         #Rotate objects to match texture orientations. At slight angle to match approximate axial tilt of earth
    #         s.rotate(angle=5*np.pi/8, axis=vector(1, 0, 0))

    #         #Not currently working. Used to add ring to saturn
    #         # if b.name == "Saturn":
    #         #     print("Adding Ring")
    #         #     # s = ring(pos = pos,
    #         #     #                     radius = 2*visual_radius(b),
    #         #     #                     thickness = 0.01,
    #         #     #                     opacity = 0.6,
    #         #     #                     #texture='textures/saturn_ring.png'
    #         #     # )
    #         #     #s = compound([s,saturn_ring], pos=pos)
    #         #     ring_shape = shapes.circle(radius = 2*visual_radius(b),
    #         #                                thickness=2*visual_radius(b))
    #         #     s = extrusion(shape = ring_shape,
    #         #                             path = [vector(0,0,0),
    #         #                                     vector(0,0,0.01)],
    #         #                             texture='textures/saturn_ring.png')
    #         #     print("Get this far")
    #         #     #s = compound([s,saturn_ring])
    #         #     print("Ring added")

    #         self.spheres.append(s)

    #         name_labels.append(label(
    #             pos=pos, text=b.name,
    #             xoffset=8, yoffset=8, height=10,
    #             border=4, box=True, opacity=0.85,
    #             color=color.black,
    #             background=color.white,
    #             visible=False,
    #         ))




    def print_matrix(self) -> None:
        print(self.mat)

    def accel_matrix(self, pos: np.ndarray) -> np.ndarray:
        acc = np.zeros_like(pos)  # Initialise empty acceleration matrix
        for i in range(self.N_bodies):
            for j in range(self.N_bodies):
                if i == j:  # obviously not calculating gravity due to itself
                    continue
                r_vec = pos[j] - pos[i]  # distance between objects
                r_abs = np.linalg.norm(r_vec)
                acc[i] += (G * self.masses[j] * r_vec) / (r_abs**3)
        return acc
            
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



    def run(self, dt: float, T: float, t0: float = 0.0,
            method: str = "abm4") -> None:
        n_steps = int((T - t0) / dt)
        history = [self.mat[:, :3].copy()]
        t = t0
        rate(FPS)
        time_text = wtext(text="")
        sun_idx = self.bodies.index(Sun)
        scene, self.spheres, name_labels, HOST_INDEX = initialise_animation(self.mat, method, self.bodies, sun_idx)

        # Create pause button
        def _toggle_pause(b):
            global paused
            paused = not paused
            b.text = "Resume" if paused else "Pause"
            for lbl in name_labels:
                lbl.visible = paused

        scene.append_to_caption("\n")
        button(text="Pause", bind=_toggle_pause)
        scene.append_to_caption("    ")

        if method == "rk4":
            for _ in tqdm(range(n_steps)):
                if paused:
                    continue
                else:
                    self.mat = rk4(t, self.mat, dt, self.dmatrix)
                    t += dt
                    history.append(self.mat[:, :3].copy())
                    self.advance_anim(scene, time_text, sun_idx, name_labels, t, method, HOST_INDEX)

        elif method == "abm4":
            # abm4_rk4 does the RK4 bootstrap and the first ABM4 step,
            # returning the state at t0, t0+dt, t0+2dt, t0+3dt, t0+4dt.
            y1, y2, y3, y4, y5 = abm4_rk4(t, self.mat, dt, self.dmatrix)
            for y in (y2, y3, y4, y5):
                history.append(y[:, :3].copy())
            buf = [y2, y3, y4, y5]          # rolling window of last 4 states
            t  += 4 * dt
            for _ in tqdm(range(n_steps - 4)):
                y_new = abm4(t, *buf, dt, self.dmatrix)
                buf   = [buf[1], buf[2], buf[3], y_new]
                t    += dt
                history.append(y_new[:, :3].copy())
                #self.advance_anim(time_text, sun_idx, name_labels, t, method) #Needs to be fixed
            self.mat = buf[-1]

        elif method == "leapfrog":
            r = self.mat[:, :3].copy()
            v = self.mat[:, 3:].copy()
            for _ in tqdm(range(n_steps)):
                if paused:
                    continue
                else:
                    r, v = leapfrog(r, v, dt, self.accel_matrix)
                    #history.append(r.copy())
                    self.mat[:, :3], self.mat[:, 3:] = r, v
                    self.advance_anim(time_text, sun_idx, name_labels, t, method)
            # self.mat[:, :3], self.mat[:, 3:] = r, v
            # self.advance_anim(time_text, sun_idx, name_labels, t, method)

        else:
            raise ValueError(
                f"Unknown method {method!r}; use 'rk4', 'abm4', or 'leapfrog'."
            )

        self.history = np.array(history)

    def advance_anim(self, scene, time_text, sun_idx, name_labels, t, method, HOST_INDEX) -> None:
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
        s: sphere
        for i, s in enumerate(self.spheres):
            new_pos = display_position(i, state, origin, HOST_INDEX)
            s.pos = new_pos
            name_labels[i].pos = new_pos

        days  = t / 86400.0
        years = days / 365.0
        time_text.text = f"Day {days:,.1f}   Year {years:,.2f}   method = {method}"


        # fig = plt.figure(figsize=(20, 20), facecolor='black')
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_facecolor('black')
        # ax.grid(False)

        # trails = [ax.plot([], [], [], linewidth = 0.5)[0] for _ in self.bodies]
        # dots = [ax.scatter([], [], s=10, label=body.name) for body in self.bodies]

        # # set limits for animation bounds
        # lim = np.abs(self.history).max()
        # ax.set_xlim(-lim, lim)
        # ax.set_ylim(-lim, lim)
        # ax.set_zlim(-lim, lim)

        # # disable grey 3d plane
        # ax.xaxis.pane.fill = False
        # ax.yaxis.pane.fill = False
        # ax.zaxis.pane.fill = False
        # ax.xaxis.pane.set_edgecolor('none')
        # ax.yaxis.pane.set_edgecolor('none')
        # ax.zaxis.pane.set_edgecolor('none')

        # time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, color='white')

        # def update(frame):
        #     time_text.set_text(f'Day {frame}, Year {frame/365:.2f}')
        #     trail = 2000 # controls how long the trail is behind the object
        #     start = max(0, frame - trail)
        #     for i in range(self.N_bodies):
        #         x = self.history[start:frame, i, 0]
        #         y = self.history[start:frame, i, 1]
        #         z = self.history[start:frame, i, 2]
        #         trails[i].set_data(x, y)
        #         trails[i].set_3d_properties(z)
        #         dots[i]._offsets3d = (x[-1:], y[-1:], z[-1:])
        #     return trails + dots + [time_text]

        # frames = range(1, len(self.history), n) # goes from beginning to end in steps of n
        # ani = FuncAnimation(fig, update, frames=frames, interval=args.interval, blit=False)

        # if legend:
        #     ax.legend(labelcolor='white', facecolor='black')
        # # plt.tight_layout()
        # plt.show()





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
]






# overrides for T and dt
# dt = 24 * 3600 # 1 hour time steps
# n = 10
# T = n * 365 * 24 * 3600 # simulation lasts n years

mySim = Simulation(bodies)

mySim.run(dt, T, t0=0, method=user_method)

# plot or animation
# if (args.visual == "plot"):
#     mySim.plot()
# elif (args.visual == "anim"):
#     mySim.animate(10, legend=True)
