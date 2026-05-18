import numpy as np
import matplotlib.pyplot as plt
from bodies import Body, Sun, Earth, Jupiter, Mars

# constants
G = 6.674e-20


class Simulation:
    def __init__(self, bodies) -> None:
        self.bodies = bodies
        self.N_bodies = len(bodies)
        self.Ndim = 6
        self.quant_vec = np.concatenate([n.return_vec() for n in self.bodies])

    def print_vector(self):
        print(self.quant_vec)

    def rk4(self):




bodies = [Earth, Sun, Mars, Jupiter]
mySim = Simulation(bodies)
mySim.print_vector()
