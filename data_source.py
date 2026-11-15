"""
A

Gain and save all needed data for this program in this file.a
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
from scipy.spatial.transform import Rotation

@dataclass
class SpaceObject:
    norad_id:int
    name:str        #name of object
    time:datetime   #UTC time
    position:np.ndarray     #3 dimention cordinary
    velocity:np.ndarray     
    orientation:np.ndarray | None = None
    frame: str = "TEME"

    def __post_init__(self):
        # Ensure that time has a timezone
        if self.time.tzinfo is None:
            raise ValueError("Time must have a timezone")

        self.time = self.time.astimezone(timezone.utc)

        # Convert coordinates to NumPy arrays
        self.position = np.asarray(
            self.position, dtype=float
        )
        self.velocity = np.asarray(
            self.velocity, dtype=float
        )

        if self.position.shape != (3,):
            raise ValueError("Position must be 3D")

        if self.velocity.shape != (3,):
            raise ValueError("Velocity must be 3D")

        if self.orientation is not None:
            self.orientation = np.asarray(
                self.orientation, dtype=float
            )
            Rotation.from_quat(self.orientation)

satellite_record: dict[str, str] = {
    "OBJECT_NAME": "MY-SIM-SAT",
    "OBJECT_ID": "2026-001A",
    "NORAD_CAT_ID": "99999",  # local number
    "EPOCH": "2026-11-15T18:00:00.000000",  # UTC

    "MEAN_MOTION": "13.98",       # Laps/Day
    "ECCENTRICITY": "0.001",      # Near-circular orbit
    "INCLINATION": "51.6",        # degree
    "RA_OF_ASC_NODE": "0.0",      # degree
    "ARG_OF_PERICENTER": "0.0",   # degree
    "MEAN_ANOMALY": "0.0",        # degree

    #set the default orientation is [0, 0, 0, 1]
    "EPHEMERIS_TYPE": "0",
    "CLASSIFICATION_TYPE": "U",
    "ELEMENT_SET_NO": "1",
    "REV_AT_EPOCH": "1",
    "BSTAR": "0.0",
    "MEAN_MOTION_DOT": "0.0",
    "MEAN_MOTION_DDOT": "0.0",
}