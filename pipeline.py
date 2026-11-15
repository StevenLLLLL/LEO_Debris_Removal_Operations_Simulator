"""Run one simulated debris-capture evaluation from an OMM record."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from firing import run_part_b
from postfire import evaluate_postfire_from_part_b

if TYPE_CHECKING:
    from data_source import SpaceObject
    from firing import PartBResult, SensorFaults
    from postfire import PostFireReport


@dataclass(frozen=True)
class PipelineResult:
    """Results available for display and the reason if the run stops early."""

    debris_at_start: SpaceObject | None = None
    part_b: PartBResult | None = None
    debris_at_fire: SpaceObject | None = None
    postfire_report: PostFireReport | None = None
    reason: str = ""

    @property
    def go(self) -> bool:
        """A selected release was evaluated successfully in the simulation."""
        return self.postfire_report is not None


def run_pipeline(
    record: dict[str, str],
    start_time_utc: datetime,
    *,
    calculate_state: Callable[[dict[str, str], datetime], SpaceObject],
    servicer_mass_after_release_kg: float,
    faults: SensorFaults | None = None,
) -> PipelineResult:
    """Use A's state calculator, B's release selection, and C's report.

    The state calculator returns TEME positions in km and velocities in km/s.
    A's state is evaluated at the start and again at B's selected release time.
    """
    if start_time_utc.utcoffset() is None:
        return PipelineResult(reason="Start time must include a time zone.")

    try:
        debris_at_start = calculate_state(record, start_time_utc.astimezone(timezone.utc))
    except (KeyError, TypeError, ValueError) as error:
        return PipelineResult(reason=f"Initial debris state unavailable: {error}")

    try:
        part_b = run_part_b(debris_at_start, faults=faults)
    except ValueError as error:
        return PipelineResult(debris_at_start=debris_at_start, reason=f"Part B: {error}")

    solution = part_b.solution
    if solution is None or part_b.fire_index is None:
        return PipelineResult(
            debris_at_start=debris_at_start,
            part_b=part_b,
            reason=part_b.abort_reason or "Part B found no firing opportunity.",
        )

    fire_time = solution.fire_time
    if fire_time is None or fire_time.utcoffset() is None:
        return PipelineResult(
            debris_at_start=debris_at_start,
            part_b=part_b,
            reason="Part B did not provide a firing time with a time zone.",
        )

    try:
        debris_at_fire = calculate_state(record, fire_time.astimezone(timezone.utc))
    except (KeyError, TypeError, ValueError) as error:
        return PipelineResult(
            debris_at_start=debris_at_start,
            part_b=part_b,
            reason=f"Debris state unavailable at the firing time: {error}",
        )

    try:
        report = evaluate_postfire_from_part_b(
            part_b,
            debris_at_fire,
            servicer_mass_after_release_kg=servicer_mass_after_release_kg,
        )
    except ValueError as error:
        return PipelineResult(
            debris_at_start=debris_at_start,
            part_b=part_b,
            debris_at_fire=debris_at_fire,
            reason=f"Part C: {error}",
        )

    return PipelineResult(
        debris_at_start=debris_at_start,
        part_b=part_b,
        debris_at_fire=debris_at_fire,
        postfire_report=report,
    )
