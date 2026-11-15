import argparse
import csv
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

import numpy as np
from sgp4 import omm
from sgp4.api import SGP4_ERRORS, Satrec, jday

import config
from data_source import SpaceObject
from pipeline import PipelineResult, run_pipeline

CSV_PATH = Path(__file__).with_name("cosmos-2251-debris.csv")

def find_omm_record(csv_path: Path, norad_id: int) -> dict[str, str]:
    """Find the CSV row corresponding to the NORAD number."""
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None or "NORAD_CAT_ID" not in reader.fieldnames:
            raise ValueError("The CSV is missing the NORAD_CAT_ID column; " \
            "please check the downloaded file.")

        for record in reader:
            if record["NORAD_CAT_ID"] == str(norad_id):
                return record

    raise ValueError(f"NORAD ID {norad_id} not found in CSV.")

def calculate_state(record: dict[str, str], time_utc: datetime) -> SpaceObject:
    """Use SGP4 to calculate the geocentric TEME position
    and velocity for a specified UTC time."""
    satellite = Satrec()
    omm.initialize(satellite, record)

    jd, fraction = jday(
        time_utc.year,
        time_utc.month,
        time_utc.day,
        time_utc.hour,
        time_utc.minute,
        time_utc.second + time_utc.microsecond / 1_000_000,
    )

    error_code, position, velocity = satellite.sgp4(jd, fraction)
    if error_code != 0:
        message = SGP4_ERRORS.get(error_code, f"Error Code {error_code}")
        raise ValueError(f"SGP4 calculation failed:{message}")

    return SpaceObject(
        norad_id=int(record["NORAD_CAT_ID"]),
        name=record["OBJECT_NAME"],
        time=time_utc,
        position=position,
        velocity=velocity,
    )

RULE = "=" * 60


def _format_utc(time: datetime) -> str:
    """YYYY-MM-DD HH:MM:SS in UTC, without microseconds or time-zone suffix."""
    if time.utcoffset() is not None:
        time = time.astimezone(timezone.utc)
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _row(label: str, value: str) -> None:
    print(f"  {label:<24}{value}")


def _format_vector(vector: np.ndarray, decimals: int) -> str:
    return "[" + ", ".join(f"{component:.{decimals}f}" for component in vector) + "]"


def _destination_km(debris_at_arrival: SpaceObject) -> np.ndarray:
    """Express the configured simulated hold point in geocentric TEME."""
    position = debris_at_arrival.position
    orbit_normal = np.cross(position, debris_at_arrival.velocity)
    radius = float(np.linalg.norm(position))
    normal_length = float(np.linalg.norm(orbit_normal))
    if not isfinite(radius) or radius <= 0 or not isfinite(normal_length) or normal_length <= 0:
        raise ValueError("The debris state does not define an orbital frame.")

    radial = position / radius
    normal = orbit_normal / normal_length
    along_track = np.cross(normal, radial)
    rsw_to_teme = np.column_stack((radial, along_track, normal))
    return position + rsw_to_teme @ (config.HOLD_POINT / 1000.0)


def print_report(result: PipelineResult) -> None:
    """Print the simulation result as a readable report."""
    print(RULE)
    print("  LEO DEBRIS CAPTURE - SIMULATION REPORT")
    print(RULE)

    report = result.postfire_report
    if report is None:
        print("  RESULT: NO-GO")
        print(f"  Reason: {result.reason}")
        print(RULE)
        return

    print("  RESULT: GO")
    debris = result.debris_at_start
    if debris is not None:
        _row("Debris", f"{debris.name} ({debris.norad_id})")
    solution = result.part_b.solution if result.part_b else None
    if solution is not None:
        _row("Capture margin", f"{solution.capture_margin:.2f} m")

    if debris is not None:
        print()
        print("[DESTINATION] SIMULATED HOLD POINT")
        _row("Arrival time (UTC)", _format_utc(debris.time))
        _row("Hold offset (RSW, m)", _format_vector(config.HOLD_POINT, 1))
        _row("Destination (TEME, km)", _format_vector(_destination_km(debris), 3))

    print()
    print("[AFTER RELEASE] VELOCITY CORRECTION")
    if result.debris_at_fire is not None:
        _row("Release time (UTC)", _format_utc(result.debris_at_fire.time))
    _row("Correction (TEME, m/s)", _format_vector(report.compensation.correction_delta_v_mps, 6))
    _row("Required delta-v", f"{report.compensation.required_delta_v_mps:.3f} m/s")
    print(RULE)


def _parse_arrival_time(value: str) -> datetime:
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        arrival_time = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use an ISO-8601 timestamp with a time zone.") from error
    if arrival_time.utcoffset() is None:
        raise argparse.ArgumentTypeError("Arrival time must include a time zone.")
    return arrival_time.astimezone(timezone.utc)


def main():
    parser = argparse.ArgumentParser(description="Run the debris capture simulation.")
    parser.add_argument(
        "--arrival-utc", type=_parse_arrival_time,
        help="Simulated hold-point arrival time. Defaults to the current time.",
    )
    args = parser.parse_args()

    user_input = input("Please enter NORAD_CAT_ID for the debris:").strip()
    if not user_input.isascii() or not user_input.isdecimal():
        print("NO-GO: Please enter the NORAD number using only digits.")
        return

    try:
        record = find_omm_record(CSV_PATH, int(user_input))
    except OSError as error:
        print(f"NO-GO: Unable to read CSV: {error}")
        return
    except ValueError as error:
        print(f"NO-GO: {error}")
        return

    mass_input = input("Servicer mass after net release (kg): ").strip()
    try:
        servicer_mass_after_release_kg = float(mass_input)
    except ValueError:
        print("NO-GO: Enter the servicer mass as a positive number in kg.")
        return
    if not isfinite(servicer_mass_after_release_kg) or servicer_mass_after_release_kg <= 0:
        print("NO-GO: Servicer mass must be finite and greater than 0 kg.")
        return

    result = run_pipeline(
        record,
        args.arrival_utc or datetime.now(timezone.utc),
        calculate_state=calculate_state,
        servicer_mass_after_release_kg=servicer_mass_after_release_kg,
    )

    print_report(result)


if __name__ == "__main__":
    main()
