# MP2
![cover_image](images/cover.png)
EEEN30150 Modelling and Simulation, Minor Project 2.

Numerical simulation of the solar system using Newton's law of gravitation. Fourth-order numerical integration methods used include Runge-Kutta, Adam's Bashforth. Given these are non-symplectic, and do not adhere to conservation of energy and momentum, a symplectic method, leapfrog, was implemented. 

vpython is used for the animation along with matplotlib for plotting.

# Installation

```bash
git clone https://github.com/francisfagan/MP2.git
cd MP2/
```

## Python dependencies

```bash
vpython
tqdm
matplotlib
numpy
```

# Usage
```bash
python python/main.py [--help] [--dt <time difference in seconds>] [--years <years>] [--visual <plot || anim>] [--frame_skip <frames_to_skip>] [--fps <fps>] [--texture_dir <path_to_directory>]  [--kepler2 <object>] [--kepler3 <object>]
```
| Argument | Type | Default | Description |
|---|---|---|---|
| `--dt` | int | 3600 | Timestep in seconds (default is 1 hour) |
| `--years` | int | 10 | Simulation duration in years |
|`--method`| str | leapfrog | Specifices the method used for numerical integration. Other methods are `rk4` and `abm4` |
| `--visual` | str | anim | Visualisation as plot or animation, can also choose `plot` |
| `--frame_skip` | int | 10 | Skips a specified amount of frames |
| `--fps` | int | 30 | Frames per second for animation |
| `--texture_dir` | path | textures/ | Path to textures for simulations (.jpg) |
| `--kepler2` | str | None | Choose an object to verify Kepler's second law for. A small variation value printed to the screen shows that the dA/dt is approximately constant |
| `--kepler3` | str | None | Choose an object to verify Kepler's third law for. Prints the observed period from JPL Horizons, a calculated period using position data in self.history, and the percentage error |


# Examples
Below are examples for exectuting the simulation. 
```bash
# normal plot for 10 years with default dt
python python/main.py --visual plot --years 10

# animation for 5 years with an hour timestep
python python/main.py --visual ani --years 5 --dt 7200

# higher skip_frames allows for faster animation
python python/main.py --visual ani --skip_frames 50

# animation for default length and dt using method abm4
python python/main.py --method abm4

# verify Kepler's 2nd law for Earth over a time period of 150 years
python python/main.py --kepler2 Earth --years 150

# verify Kepler's 3rd law for Earth over a time period of 10 years
python python/main.py --kepler3 Earth --years 10
```

# Sources
https://www.solarsystemscope.com/

https://www.planetaryvisions.com/Top.php?cat=4

https://ssd.jpl.nasa.gov/horizons/app.html#/