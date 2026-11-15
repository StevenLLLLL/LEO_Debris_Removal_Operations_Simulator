import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from data_source import SpaceObject

from sgp4 import omm
from sgp4.api import SGP4_ERRORS, Satrec, jday

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
    user_input = input("Please enter NORAD_CAT_ID for the debris.:").strip()
    if not user_input.isascii() or not user_input.isdecimal():
        print("Please enter the NORAD number using only digits.")
        return

    try:
        record = find_omm_record(CSV_PATH, int(user_input))
        debris = calculate_state(record, datetime.now(timezone.utc))
    except OSError as error:
        print(f"Unable to read CSV:{error}")
        return
    except (KeyError, TypeError, ValueError) as error:
        print(f"Unable to retrieve wreckage status:{error}")
        return

    print(f"debris{debris.name} ({debris.norad_id})")
    print(f"Calculation time:{debris.time.isoformat()}")
    print(f"Reference frame:{debris.frame}")
    print(f"Position (x, y, z), km:{debris.position}")
    print(f"Velocity (vx, vy, vz), km/s:{debris.velocity}")


if __name__ == "__main__":
    main()