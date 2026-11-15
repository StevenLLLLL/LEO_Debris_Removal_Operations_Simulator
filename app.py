import csv
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path

from sgp4 import omm
from sgp4.api import SGP4_ERRORS, Satrec, jday

from data_source import SpaceObject
from pipeline import run_pipeline

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

    debris = result.debris_at_start
    if debris is not None:
        print(f"Debris: {debris.name} ({debris.norad_id})")
        print(f"Calculation time: {debris.time.isoformat()}")
        print(f"Reference frame: {debris.frame}")
        print(f"Position (x, y, z), km: {debris.position}")
        print(f"Velocity (vx, vy, vz), km/s: {debris.velocity}")

    report = result.postfire_report
    if report is None:
        print(f"NO-GO: {result.reason}")
        return

    print("Simulation assessment: GO")
    if result.debris_at_fire is not None:
        print(f"Firing time (UTC): {result.debris_at_fire.time.isoformat()}")
    print(f"Recoil delta-v (TEME, m/s): {report.recoil.recoil_delta_v_mps}")
    print(f"Servicer velocity after release (TEME, m/s): {report.recoil.servicer_velocity_after_mps}")

    orbit = report.orbit_change
    print(
        f"Perigee altitude (m): {orbit.before.perigee_altitude_m:,.2f} to "
        f"{orbit.after.perigee_altitude_m:,.2f} ({orbit.perigee_change_m:+,.2f} m)"
    )
    print(
        f"Apogee altitude (m): {orbit.before.apogee_altitude_m:,.2f} to "
        f"{orbit.after.apogee_altitude_m:,.2f} ({orbit.apogee_change_m:+,.2f} m)"
    )
    print(f"Compensation delta-v (TEME, m/s): {report.compensation.correction_delta_v_mps}")
    print(f"Required compensation (m/s): {report.compensation.required_delta_v_mps:.6f}")


if __name__ == "__main__":#only run main when running this file
    main()
