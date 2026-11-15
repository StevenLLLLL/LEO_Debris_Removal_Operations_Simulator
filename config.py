"""
All

Unify simulation parameters and check thresholds, such as default retention distance, data age upper limit, deviation
upper limit, satellite mass and compensation budget.
"""
# numpy: needed for the HOLD_POINT vector; config.py imports no project modules
import numpy as np

# Earth gravitational parameter (m^3/s^2), SI units
MU_EARTH = 3.986004418e14

# ---------------------------------------------------------------------
# Part B: Location of us (proximity navigation)
# ---------------------------------------------------------------------
HOLD_POINT = np.array([0.0, -10.0, 0.0])  # m, V-bar hold point: 10 m directly behind the target

DEFAULT_DT = 1.0              # s, length of each time step (one sensor reading per step)
DEFAULT_STEPS = 120           # number of steps (120 s): the longest we wait at the hold point, not a fixed fire time
MAX_DROPOUT_S = 10.0          # s, continuous dropout longer than this → abort (dead reckoning no longer trusted)
MAX_SIGMA_M = 1.0             # m, position uncertainty still above this after settling → abort (filter divergence)
SETTLE_TIME_S = 15.0          # s, filter settling time after start; no uncertainty check before this

# Threshold for "this reading is clearly wrong": chi-square, 3 DOF, 99% critical value.
# A squared Mahalanobis distance of the innovation above it → bad data
GATE_CHI2_3DOF_99 = 11.345

# ---------------------------------------------------------------------
# Part B: Firing solution and net parameters
# ---------------------------------------------------------------------
# Net mass (kg), assumed
NET_MASS_KG = 2.0
# Net radius when fully deployed (m), assumed; a capture needs the net to cover the debris
NET_RADIUS_M = 2.0
# Time for the net to open fully after launch (s), assumed; a shorter flight hits with a half-open net
NET_DEPLOY_TIME_S = 0.3
# Kinetic energy the launcher gives the net (J), assumed; together with the mass it sets the net speed
LAUNCHER_ENERGY_J = 400.0
# Debris radius (m), assumed, because CelesTrak orbit data has no object size
DEBRIS_RADIUS_M = 0.5
T_MIN, T_MAX = 1e-3, 5.0     # s, search range for the net's time of flight (no solution if it needs longer)
