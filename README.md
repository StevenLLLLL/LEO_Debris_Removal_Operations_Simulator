# LEO Debris Removal Operations Simulator

## Part B: Proximity Navigation & Firing Solution

### Files

- `relnav.py` — proximity relative navigation: Clohessy-Wiltshire (CW) state transition matrix, sensor simulation with fault injection (noise, dropout, outliers), a Kalman filter with an outlier gate, and the proximity loop with abort rules (dropout too long, uncertainty too large after settling).
- `firing.py` — firing solution: CW targeting (velocity needed for the net to reach the target), bisection on time of flight so the net speed matches the launcher speed, miss-distance verification, and a Monte Carlo helper (noise vs capture rate).

### Conventions

- Target-centred RSW frame: `x` radial (away from Earth), `y` along-track, `z` cross-track.
- Units: metres and seconds (angles in radians). Vectors are length-3 numpy arrays; times are UTC `datetime`.
- Hold point: `(0, -10, 0)` m, on the V-bar 10 m behind the target.

### Public functions

- `run_proximity(n, ...)` → `list[NavFrame]` (one frame per time step; `n` is the target's orbital rate from `TargetState.n`).
- `solve_firing(frame, n, epoch=None, net_speed=20.0)` → `FiringSolution`.

### Notes for Part C

- `NavFrame` has two extra fields beyond the interface: `true_vel` (true relative velocity, used for miss verification) and `abort_reason` (text, empty if not aborted).
- `FiringSolution` has three extra fields:
  - `net_velocity_rsw` — net velocity relative to the satellite (m/s, RSW); use it to compute recoil.
  - `miss_uncertainty` — 3σ position uncertainty (m). Go / No-Go should check **`miss_uncertainty < net radius`**, because `predicted_miss` is computed from the simulation's true state, which a real mission would not have.
  - `message` — explanation when no solution is found.

### Running the tests

```bash
pip install -r requirements.txt
python -m pytest -q
```
