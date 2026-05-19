"""
Numerical integrators for ordinary differential equations.
    rk4       -- Classical 4th-order Runge-Kutta. 
    abm4      -- Adams-Bashforth-Moulton 4th-order predictor-corrector.
    leapfrog  -- Symplectic integrator for autonomous
                 Hamiltonian systems.
"""

import numpy as np

def rk4(t, y, dt, f):
    """
    4th-order Runge-Kutta, single step.
    """
    k1 = dt * f(t,            y)
    k2 = dt * f(t + 0.5 * dt, y + 0.5 * k1)
    k3 = dt * f(t + 0.5 * dt, y + 0.5 * k2)
    k4 = dt * f(t + dt,       y + k3)
    return y + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def ab4(t4, y1, y2, y3, y4, dt, f):
    """
    Adams-Bashforth 4-step predictor.
    """
    f1 = f(t4 - 3 * dt, y1)
    f2 = f(t4 - 2 * dt, y2)
    f3 = f(t4 - dt,     y3)
    f4 = f(t4,          y4)
    return y4 + dt * ((55/24)*f4 - (59/24)*f3 + (37/24)*f2 - (9/24)*f1)


def am4(t4, y2, y3, y4, y5_pred, dt, f):
    """
    Adams-Moulton 4-step corrector.
    """
    f2 = f(t4 - 2 * dt,  y2)
    f3 = f(t4 - dt,      y3)
    f4 = f(t4,           y4)
    f5 = f(t4 + dt,      y5_pred)
    return y4 + dt * ((9/24)*f5 + (19/24)*f4 - (5/24)*f3 + (1/24)*f2)


def abm4(t4, y1, y2, y3, y4, dt, f):
    """One full ABM4 step: AB4 predict, then AM4 correct."""
    y5_pred = ab4(t4, y1, y2, y3, y4, dt, f)
    return am4(t4, y2, y3, y4, y5_pred, dt, f)


def abm4_rk4(t1, y1, dt, f):
    """
    RK4 bootstrap + one ABM4 step.
    """
    y2 = rk4(t1,           y1, dt, f)
    y3 = rk4(t1 + dt,      y2, dt, f)
    y4 = rk4(t1 + 2 * dt,  y3, dt, f)
    y5 = abm4(t1 + 3 * dt, y1, y2, y3, y4, dt, f)
    return y1, y2, y3, y4, y5

def leapfrog(r, v, dt, accel):
   
    a_now  = accel(r)
    v_half = v + 0.5 * dt * a_now

    r_new  = r + dt * v_half

    a_new  = accel(r_new)
    v_new  = v_half + 0.5 * dt * a_new

    return r_new, v_new
