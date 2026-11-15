"""
A

Gain and save all needed data for this program in this file.a
"""
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