# Author: Matthew McDowell
# Student Number: 23365206
# 
# 
# 4-step Adams-Bashford implementation:
# Usage:
#       -Inputs v1 to v4 are the previous states of the system up till 4 timesteps before the next prediction.
#        On startup the extra 3 steps needed are predicted using RK4
#       -Input h is the size of the timestep used
#
#
# Evaluates equation stored in temp_eqns.py [CHANGE THIS TO REAL EQN LOCATION]
# Uses these to predict the value of f after one timestep h based on the 4 previous iterations.
# Returns this predicted value as an array with the time value this is predicated for as a 2 element vector.

from temp_eqns import f
import numpy as np
def abm4(v1, v2, v3, v4, h):
    f1 = f(v1[0], v1[1])
    f2 = f(v2[0], v2[1])
    f3 = f(v3[0], v3[1])
    f4 = f(v4[0], v4[1])

    f5_y = v4[1] + h*((55/24)*f4 - (59/24)*f3 + (37/24)*f2 - (9/24)*f1)
    f5 = np.array([v4[0]+h, f5_y])
    return f5