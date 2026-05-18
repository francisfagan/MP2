import numpy as np


def evaluate(t, y, k=1):
    """
    Evaluates the function at time t and y=y.
    """
    v = y[1]
    a = -(k**2) * y[0]
    return np.array([v, a])


def rk4(t, dt, y, evaluate):
    """
    Implements the Runge-Kutta fourth order numerical integration technique.

    Input:
    t = time at which Runge-Kutta is evaluated

    """
    k1 = dt * evaluate(t, y)
    k2 = dt * evaluate(t + 0.5 * dt, y + 0.5 * k1)
    k3 = dt * evaluate(t + 0.5 * dt, y + 0.5 * k2)
    k4 = dt * evaluate(t + dt, y + k3)

    y_updated = y + (1 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_updated
