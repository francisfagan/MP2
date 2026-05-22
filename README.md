# MP2
EEEN30150 Modelling and Simulation, Minor Project 2.

Numerical simluation of the solar system using Newton's law of gravitation.

# Usage
```bash
python main.py [--dt <time difference in seconds>] [--years <years>] [--visual <plot || ani>] [--frame_skip <frames_to_skip>] [--texture_dir <path_to_directory>] [--fps <fps>]
```
| Argument | Type | Default | Description |
|---|---|---|---|
| `--dt` | int | 3600 | Timestep in seconds (1 hour) |
| `--years` | int | 10 | Simulation duration in years |
| `--visual` | str | `plot` | Visualisation as plot or animation, can also choose `ani` |
| `--frame_skip` | int | 10 | The animation frame delay in ms, higher is slower |
| `--texture_dir` | path | textures/ | The animation frame delay in ms, higher is slower |
| `--fps` | int | 30 | The animation frame delay in ms, higher is slower |


# Examples
```bash
# normal plot for 10 years with default dt
python main.py --visual plot --years 10

# animation for 5 years with an hour timestep
python main.py --visual ani --years 5 --dt 7200

# higher skip_frames allows for faster animation
python main.py --visual ani --skip_frames 50
```

# Sources
https://planet-texture-maps.fandom.com/wiki/

https://www.planetaryvisions.com/index.php

https://ssd.jpl.nasa.gov/horizons/app.html#/