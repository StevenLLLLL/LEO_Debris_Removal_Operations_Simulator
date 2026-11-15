import csv
from datetime import datetime, timezone
from math import degrees, isfinite
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

    raise ValueError(f"NORAD ID not found in CSV {norad_id}。")

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

EARTH_RADIUS_KM = 6371.0
RULE = "=" * 60


def _format_utc(time: datetime) -> str:
    """YYYY-MM-DD HH:MM:SS in UTC, without microseconds or time-zone suffix."""
    if time.utcoffset() is not None:
        time = time.astimezone(timezone.utc)
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _row(label: str, value: str) -> None:
    print(f"  {label:<20}{value}")


def print_report(result: PipelineResult) -> None:
    """Print the simulation result as a readable report."""
    print(RULE)
    print("  LEO DEBRIS CAPTURE - SIMULATION REPORT")
    print(RULE)

    debris = result.debris_at_start
    if debris is not None:
        print()
        print("[A] TARGET DEBRIS")
        _row("Debris", f"{debris.name} ({debris.norad_id})")
        _row("State time (UTC)", _format_utc(debris.time))
        altitude_km = float(np.linalg.norm(debris.position)) - EARTH_RADIUS_KM
        _row("Altitude", f"{altitude_km:.1f} km")
        _row("Speed", f"{float(np.linalg.norm(debris.velocity)):.2f} km/s")
        _row("Frame", debris.frame)

    part_b = result.part_b
    if part_b is not None:
        print()
        print("[B] PROXIMITY NAVIGATION AND FIRING")
        hold_m = float(np.linalg.norm(config.HOLD_POINT))
        _row("Hold point", f"{hold_m:g} m behind the debris (V-bar)")
        _row("Observation", f"{config.SETTLE_TIME_S:g} s before firing (filter settled)")

        solution = part_b.solution
        if solution is not None and part_b.fire_index is not None:
            fire_t = part_b.frames[part_b.fire_index].t
            fire_time = _format_utc(solution.fire_time) if solution.fire_time else "n/a"
            _row("Fire time (UTC)", f"{fire_time} (T+{fire_t:g} s)")

            az_deg = degrees(solution.azimuth)
            el_deg = degrees(solution.elevation)
            aim = f"azimuth {az_deg:+.2f} deg, elevation {el_deg:+.2f} deg"
            if abs(az_deg) < 1 and abs(el_deg) < 1:
                aim += " (straight ahead)"
            _row("Aim", aim)
            _row(
                "Net",
                f"{solution.net_speed:.1f} m/s, flight time {solution.time_of_flight:.2f} s",
            )
            _row("Position 3-sigma", f"{solution.miss_uncertainty:.2f} m")
            margin = f"{solution.capture_margin:.2f} m"
            if solution.capture_margin > 0:
                margin += " -> net covers the debris"
            _row("Capture margin", margin)
            _row("Net fully open", "yes" if solution.deploy_ok else "no")

    report = result.postfire_report
    if report is not None:
        print()
        print("[C] AFTER RELEASE")
        recoil = np.asarray(report.recoil.recoil_delta_v_mps, dtype=float)
        recoil_text = f"{float(np.linalg.norm(recoil)):.3f} m/s"
        if result.debris_at_fire is not None:
            flight = np.asarray(result.debris_at_fire.velocity, dtype=float)
            denom = float(np.linalg.norm(recoil) * np.linalg.norm(flight))
            if denom > 0:
                cos_angle = float(np.clip(np.dot(recoil, flight) / denom, -1.0, 1.0))
                angle = degrees(np.arccos(cos_angle))
                if angle > 150:
                    recoil_text += ", opposite to the flight direction"
                else:
                    recoil_text += f", angle {angle:.1f} deg to the flight direction"
        _row("Recoil", recoil_text)

        orbit = report.orbit_change
        _row(
            "Perigee altitude",
            f"{orbit.before.perigee_altitude_m / 1000:.2f} km -> "
            f"{orbit.after.perigee_altitude_m / 1000:.2f} km  "
            f"({orbit.perigee_change_m / 1000:+.2f} km)",
        )
        _row(
            "Apogee altitude",
            f"{orbit.before.apogee_altitude_m / 1000:.2f} km -> "
            f"{orbit.after.apogee_altitude_m / 1000:.2f} km  "
            f"({orbit.apogee_change_m / 1000:+.2f} km)",
        )
        _row(
            "Compensation burn",
            f"{report.compensation.required_delta_v_mps:.3f} m/s",
        )

    print()
    print(RULE)
    if report is not None:
        print("  RESULT: GO")
    else:
        if result.debris_at_start is None:
            stage = "Part A (data)"
        elif part_b is None or part_b.fire_index is None:
            stage = "Part B (proximity navigation)"
        else:
            stage = "Part C (after release)"
        print("  RESULT: NO-GO")
        print(f"  Stopped at: {stage}")
        print(f"  Reason: {result.reason}")
    print(RULE)


def main():
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
        datetime.now(timezone.utc),
        calculate_state=calculate_state,
        servicer_mass_after_release_kg=servicer_mass_after_release_kg,
    )

    print_report(result)


if __name__ == "__main__":#only run main when running this file
    main()
