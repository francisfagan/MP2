# MP2
EEEN30150 Modelling and Simulation, Minor Project 2.

Numerical simluation of the solar system using Newton's law of gravitation.

# Usage
```bash
python main.py [--dt <time difference in seconds>] [--years <YEARS>] [--visual <plot || ani>] [--interval INTERVAL]
```
| Argument | Type | Default | Description |
|---|---|---|---|
| `--dt` | float | 86400 | Timestep in seconds |
| `--years` | int | 10 | Simulation duration in years |
| `--visual` | str | `plot` | Visualisation as plot or animation, can also choose `ani` |
| `--interval` | float | 30 | The animation frame delay in ms, higher is slower |

# Examples
```bash
# normal plot for 10 years with a day timestep
python main.py --visual plot --years 10

# animation for 5 years with an hour timestep
python main.py --visual ani --years 5 --dt 3600

# lower interval allows for faster animation
python main.py --visual ani --interval 5
```
