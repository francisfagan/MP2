import numpy as np
from tqdm import tqdm
from integrators import rk4, abm4, abm4_rk4, leapfrog
import matplotlib.pyplot as plt
from tqdm import tqdm
from vpython import button, canvas, color, label, rate, sphere, vector, wtext
from animation_funcs import *


from bodies import (
    Body,
    Sun,
)


paused = False
running = True


# Constants
Au = 149597870.700  # in kilometers,
G = 6.674e-20 / Au ** 3 # In Au^3 * kg^-1 * s^-2
year = 365 * 24 * 3600  # 1 year in secs

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

    # Initialise self.mat with initial positions and velocities of all bodies
    # Position in Au
    # velocity in Au/sec
    def initiliase_matrix(self, bodies: list[Body]) -> np.ndarray:
        return np.array([body.return_vec()/Au for body in bodies], dtype=float)

    

    def print_matrix(self) -> None:
        print(self.mat)

    # Calculates just the acceleration for each body at time t
    # Used by leapfrog method
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
    #   -Calculates body positions one time step forward using chosen method
    #   -Stores each iteration in history 3D array
    def run(self, dt: float, T: float, t0: float = 0.0,
            method: str = "abm4") -> None:
        n_steps = int((T - t0) / dt)
        history = [self.mat[:, :3].copy()]
        t = t0


        if method == "rk4":
            for step in tqdm(range(n_steps)):
                self.mat = rk4(t, self.mat, dt, self.dmatrix)
                t += dt
                history.append(self.mat[:, :3].copy())


        elif method == "abm4":
            # abm4_rk4 does the RK4 bootstrap and the first ABM4 step,
            # returning the state at t0, t0+dt, t0+2dt, t0+3dt, t0+4dt.
            y1, y2, y3, y4, y5 = abm4_rk4(t, self.mat, dt, self.dmatrix)
            for y in (y2, y3, y4, y5):
                history.append(y[:, :3].copy())
            buf = [y2, y3, y4, y5]          # rolling window of last 4 states
            t  += 4 * dt
            for step in tqdm(range(n_steps - 4)):
                y_new = abm4(t, *buf, dt, self.dmatrix)
                buf   = [buf[1], buf[2], buf[3], y_new]
                t    += dt
                history.append(y_new[:, :3].copy())
                self.mat = y_new


        elif method == "leapfrog":
            r = self.mat[:, :3].copy()
            v = self.mat[:, 3:].copy()
            for step in tqdm(range(n_steps)):
                r, v = leapfrog(r, v, dt, self.accel_matrix)
                t += dt
                history.append(r.copy())
                self.mat[:, :3], self.mat[:, 3:] = r, v

        else:
            raise ValueError(
                f"Unknown method {method!r}; use 'rk4', 'abm4', or 'leapfrog'."
            )

        self.history = np.array(history)


    # Creates and runs animation of N bodies orbiting with the sun as the origin
    def animate(self, dt, T, t0, method, texture_dir, fps, frame_skip) -> None:
        t = t0
        nsteps = int((T-t0)/dt)
        rate(fps)
        sun_idx = self.bodies.index(Sun)  

        #Initialise animation canvas and sphere objects
        self.scene, self.spheres, name_labels, HOST_INDEX, time_text = initialise_animation(
            self.history[0,:,:], method, self.bodies, sun_idx, texture_dir
        )

        # Create pause and close buttons
        def _toggle_pause(b):
            global paused
            paused = not paused
            b.text = "Resume" if paused else "Pause"
            for lbl in name_labels:
                lbl.visible = paused
        
        def _close_anim(b):
            global running
            running = False


        self.scene.append_to_caption("\n")
        button(text="Pause", bind=_toggle_pause)
        button(text="Close", bind=_close_anim)


        # Loop animation until user exits program
        while running:
            for n in range(0, nsteps, frame_skip):
                while paused:
                    rate(fps)

                state = self.history[n, :, :]

                origin = state[sun_idx, :3]

                # Update position of each sphere object
                for i, s in enumerate(self.spheres):
                    new_pos = display_position(i, state, origin, HOST_INDEX)
                    s.pos = new_pos
                    name_labels[i].pos = new_pos
                            
                t = t0 + n * dt
                days  = t / 86400.0
                years = days / 365.0
                time_text.text = f"Day {days:,.1f}   Year {years:,.2f}   method = {method}"

                rate(fps)

                #Loop animation until user closes it
                if not running:
                    exit()

            # Clear trails for next loop of animation 
            for i, s in enumerate(self.spheres):
                s.clear_trail()
                    
                
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
            x = self.history[:, i, 0]
            y = self.history[:, i, 1]
            z = self.history[:, i, 2]
            ax.plot(x, y, z, label=body.name)

            # ax.legend() # can uncomment for legend, blocks the view
        plt.show()

    def kepler2_verification(self, T: int, dt: int, body_name: str = "Earth", reference_name: str = "Sun") -> None:
        """
        Kepler's second law states that a line segment joining a planet and the Sun sweeps out equal areas during equal intervals of time.
        This function can also be used to verify this for satellites also, using options shown in the --help cli function.
        dA/dt = 0.5 * |r x v|
        The following formula dA/dt = 0.5 * |r_i x r_{i+1}| / dt is found through the approximation outlined in the .pdf document. 
        Essentially, velocity is approximated as the distance between two points divided by the timestep.
        
        Upon completion, a plot showing constant dA/dt will be shown along with an output for the mean and the variation of these values.
        Choice of this function from the command line exits after completion and thus NO animation will be shown afterward.
        """

        print(f"Verifying Kepler's second law for {body_name} with the reference body to {reference_name}.")

        # find body and reference body index
        ref_idx = next((i for i, b in enumerate(self.bodies) if b.name == reference_name), None)
        body_idx = next((i for i, b in enumerate(self.bodies) if b.name == body_name), None)

        # get positions from self.history after run() is called
        ref_pos  = self.history[:, ref_idx, :]
        body_pos = self.history[:, body_idx, :]

        # this gives position of body relative to its reference body at each timestep
        r = body_pos - ref_pos

        cross_product = np.cross(r[:-1], r[1:]) # cross product for next calculation
        dA_dt = 0.5 * np.linalg.norm(cross_product, axis=1) / dt # dA/dt = 1/2 |r_i x r_(i+1)|

        
        days = np.arange(len(dA_dt)) * dt / 86400.0

        variation = (dA_dt.max() - dA_dt.min()) / dA_dt.mean() * 100 # how much the constant value varied, if low the value is approximately constant

        # terminal output
        print(f"The variation of areal velocity is {variation:.5f}%.")
        print(f"The mean of areal velocity is {dA_dt.mean():5e} AU^2/s.")

        plt.plot(days, dA_dt, '.', markersize=1)
        plt.axhline(dA_dt.mean(), color='r', linestyle="--", label=f"mean={dA_dt.mean():.4e}, with variation = {variation:.3f}%")
        plt.xlabel("Time [days]")
        plt.ylabel("dA/dt [$Au^2\/s$]")
        plt.title(f"Kepler's 2nd Law for {body_name} relative to {reference_name} for {T} years.")
        plt.legend()
        plt.show()


    # Calculates the orbital period of a body based on its semi-major axis length, checking Kepler's 3rd Law
    # Compares this to the theoretical orbit of that body
    def kepler3_verification(self, T: int, dt: float, body_name: str = "Earth", reference_name: str = "Sun") -> None:

        # find indices of bodies
        ref_idx = next((i for i, b in enumerate(self.bodies) if b.name == reference_name), None)
        body_idx = next((i for i, b in enumerate(self.bodies) if b.name == body_name), None)
        if ref_idx is None or body_idx is None:
            raise ValueError("Reference or body name not found in simulation bodies.")
        ref_mass = self.bodies[ref_idx].mass
        body_period = self.bodies[body_idx].orbital_period

        #Convert T to years and check if longer than orbital period
        T = T / year
        if T < body_period:
            raise ValueError("Simulation period not long enough to guarantee semi-major axis captured.")

        # positions (Au) over time
        ref_pos = self.history[:, ref_idx, :]
        body_pos = self.history[:, body_idx, :]

        # radial distance series (Au)
        rel_pos = body_pos - ref_pos
        r = np.linalg.norm(rel_pos, axis=1)

        # Min and max r on opposite sides of orbit
        r_max = np.max(r)
        r_min = np.min(r)

        # Semi-major axis length. Half distance from r_min to r_max points
        a = (r_max + r_min) / 2

        calc_period = np.sqrt(((a**3) * (4*np.pi**2)) / (G*ref_mass)) / year
        print(f'Keplers 3rd Law Verification for body: {self.bodies[body_idx].name}')
        print("-----------------------------------------------------")
        print(f'Calculated Period: {calc_period}')
        print(f'Observed Period: {body_period}')
        print(f'Percentage Error = {((calc_period - body_period) / body_period) * 100:.5f}%\n')       