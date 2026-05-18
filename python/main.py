import numpy as np
import matplotlib.pyplot as plt
from bodies import Body, Sun, Earth, Jupiter, Mars

np.set_printoptions(linewidth=100)

# constants
G = 6.674e-20
Au =  149597870.700 # in kilometers

class Simulation:
    def __init__(self, bodies) -> None:
        self.bodies = bodies
        self.N_bodies = len(bodies)
        self.Ndim = 6
        self.mat = self.initiliase_matrix(bodies)
        self.dt = 86400 # 1 day in seconds

    def initiliase_matrix(self, bodies):
        return np.array([body.return_vec()/(Au) for body in bodies], dtype=float)

    def print_vector(self):
        print(self.mat)

    def rk4(self):
        pass




bodies = [Sun, Mars, Earth, Jupiter]
mySim = Simulation(bodies)
mySim.print_vector()
