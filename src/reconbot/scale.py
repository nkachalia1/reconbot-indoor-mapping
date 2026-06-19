"""Metric scale calibration and dimensional accuracy evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class DimensionCheck:
    name: str
    actual_m: float
    reconstructed_units: float

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dimension name must not be empty")
        if self.actual_m <= 0 or self.reconstructed_units <= 0:
            raise ValueError("dimension measurements must be positive")


@dataclass(frozen=True)
class DimensionResult:
    name: str
    actual_m: float
    estimated_m: float
    absolute_error_m: float
    percent_error: float


@dataclass(frozen=True)
class MetricScaleReport:
    anchor_name: str
    anchor_distance_m: float
    anchor_reconstructed_units: float
    meters_per_reconstruction_unit: float
    checks: tuple[DimensionResult, ...]
    mean_absolute_error_m: float | None
    mean_percent_error: float | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["checks"] = [asdict(item) for item in self.checks]
        return payload


def build_metric_scale_report(
    *,
    anchor_name: str,
    anchor_distance_m: float,
    anchor_reconstructed_units: float,
    checks: tuple[DimensionCheck, ...] = (),
) -> MetricScaleReport:
    if not anchor_name.strip():
        raise ValueError("anchor_name must not be empty")
    if anchor_distance_m <= 0 or anchor_reconstructed_units <= 0:
        raise ValueError("anchor measurements must be positive")

    scale = anchor_distance_m / anchor_reconstructed_units
    results = tuple(
        DimensionResult(
            name=check.name,
            actual_m=check.actual_m,
            estimated_m=check.reconstructed_units * scale,
            absolute_error_m=abs(check.reconstructed_units * scale - check.actual_m),
            percent_error=(
                abs(check.reconstructed_units * scale - check.actual_m) / check.actual_m * 100
            ),
        )
        for check in checks
    )
    return MetricScaleReport(
        anchor_name=anchor_name,
        anchor_distance_m=anchor_distance_m,
        anchor_reconstructed_units=anchor_reconstructed_units,
        meters_per_reconstruction_unit=scale,
        checks=results,
        mean_absolute_error_m=(
            sum(item.absolute_error_m for item in results) / len(results) if results else None
        ),
        mean_percent_error=(
            sum(item.percent_error for item in results) / len(results) if results else None
        ),
    )
