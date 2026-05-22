import argparse

from simulation import *

from bodies import (
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

# parser for cli arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dt", type=float, default=3600, help="timestep in seconds (default = 3600s = 1 hr)")
parser.add_argument("--years", type=int, default=10, help="simulation duration in years (default = 10 years)")
parser.add_argument("--method", type=str, default="leapfrog", choices=["rk4", "abm4", "leapfrog"], help="choose simulation method. Default is leapfrog method")
parser.add_argument("--visual", type=str, default="anim", choices=["plot", "anim"], help="choose plot or animation")
parser.add_argument("--frame_skip", type=float, default=10, help="controls number of frames skipped during animation. Default is 10, every 10th iteration is animated. Higher produces faster animation, but movement of bodies with short orbits becomes erratic.")
parser.add_argument("--fps", type=float, default=30, help="VPython animation frame rate; lower values slow the animation. Default is 30")
parser.add_argument("--texture_dir", type=str, default="textures", help="choose directory for body textures. Texture names must match planet names. Default is /MP2/textures when cwd is /MP2/")
args = parser.parse_args()

# simulation arguments
t0 = 0
dt = args.dt
T = args.years * 365 * 24 * 3600
user_method = args.method
animation_fps = args.fps
frame_skip = int(args.frame_skip)
texture_dir = args.texture_dir

# List of bodies to be used in simulation
# Comment out unwanted bodies
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




# Calling of simulation run and either animation or plot creation

mySim = Simulation(bodies)

mySim.run(dt, T, t0, user_method)

# plot or animation
if (args.visual == "plot"):
    mySim.plot()
elif (args.visual == "anim"):
    mySim.animate(dt, T, t0, user_method, texture_dir, animation_fps, frame_skip)
