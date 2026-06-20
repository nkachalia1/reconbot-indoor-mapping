"""Targeted rematching utilities for weak temporal intervals."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

PAIR_ID_MAX = 2_147_483_647


@dataclass(frozen=True)
class TargetedRematchReport:
    start_frame: int
    end_frame: int
    overlap: int
    selected_images: int
    generated_pairs: int
    existing_match_rows_removed: int
    existing_geometry_rows_removed: int
    database_integrity: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CrossIntervalRematchReport:
    first_start_frame: int
    first_end_frame: int
    second_start_frame: int
    second_end_frame: int
    first_images: int
    second_images: int
    generated_pairs: int
    existing_match_rows_removed: int
    existing_geometry_rows_removed: int
    database_integrity: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def image_pair_id(first: int, second: int) -> int:
    if first <= 0 or second <= 0 or first == second:
        raise ValueError("image ids must be distinct positive integers")
    smaller, larger = sorted((first, second))
    return PAIR_ID_MAX * smaller + larger


def _frame_index(name: str) -> int | None:
    match = re.search(r"(?:^|[/\\])frame_(\d+)(?:_|\.)", name)
    return int(match.group(1)) if match else None


def prepare_targeted_rematch(
    database_path: Path,
    pair_list_path: Path,
    report_path: Path,
    *,
    start_frame: int,
    end_frame: int,
    overlap: int,
) -> TargetedRematchReport:
    if start_frame < 0 or end_frame < start_frame:
        raise ValueError("invalid frame interval")
    if overlap <= 0:
        raise ValueError("overlap must be positive")

    database = Path(database_path)
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    connection = sqlite3.connect(database)
    try:
        image_rows = connection.execute("SELECT image_id, name FROM images").fetchall()
        selected = sorted(
            (
                (index, int(image_id), str(name))
                for image_id, name in image_rows
                if (index := _frame_index(str(name))) is not None
                and start_frame <= index <= end_frame
            ),
            key=lambda item: item[0],
        )
        pairs: list[tuple[int, str, str]] = []
        for position, (first_index, first_id, first_name) in enumerate(selected):
            for second_index, second_id, second_name in selected[position + 1 :]:
                if second_index - first_index > overlap:
                    break
                pairs.append((image_pair_id(first_id, second_id), first_name, second_name))

        pair_ids = [(pair_id,) for pair_id, _, _ in pairs]
        removed_matches = sum(
            connection.execute("SELECT COUNT(*) FROM matches WHERE pair_id = ?", item).fetchone()[0]
            for item in pair_ids
        )
        removed_geometries = sum(
            connection.execute(
                "SELECT COUNT(*) FROM two_view_geometries WHERE pair_id = ?", item
            ).fetchone()[0]
            for item in pair_ids
        )
        with connection:
            connection.executemany("DELETE FROM matches WHERE pair_id = ?", pair_ids)
            connection.executemany(
                "DELETE FROM two_view_geometries WHERE pair_id = ?", pair_ids
            )
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        connection.close()

    pair_destination = Path(pair_list_path)
    pair_destination.parent.mkdir(parents=True, exist_ok=True)
    pair_destination.write_text(
        "".join(f"{first} {second}\n" for _, first, second in pairs),
        encoding="utf-8",
    )
    report = TargetedRematchReport(
        start_frame=start_frame,
        end_frame=end_frame,
        overlap=overlap,
        selected_images=len(selected),
        generated_pairs=len(pairs),
        existing_match_rows_removed=removed_matches,
        existing_geometry_rows_removed=removed_geometries,
        database_integrity=integrity,
    )
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def prepare_cross_interval_rematch(
    database_path: Path,
    pair_list_path: Path,
    report_path: Path,
    *,
    first_start_frame: int,
    first_end_frame: int,
    second_start_frame: int,
    second_end_frame: int,
) -> CrossIntervalRematchReport:
    """Generate all image pairs between two disjoint revisit intervals."""
    if first_start_frame < 0 or first_end_frame < first_start_frame:
        raise ValueError("invalid first frame interval")
    if second_start_frame < 0 or second_end_frame < second_start_frame:
        raise ValueError("invalid second frame interval")
    if first_end_frame >= second_start_frame:
        raise ValueError("cross-rematch intervals must be ordered and disjoint")

    database = Path(database_path)
    if not database.is_file():
        raise FileNotFoundError(f"database not found: {database}")
    connection = sqlite3.connect(database)
    try:
        image_rows = connection.execute("SELECT image_id, name FROM images").fetchall()
        indexed = [
            (index, int(image_id), str(name))
            for image_id, name in image_rows
            if (index := _frame_index(str(name))) is not None
        ]
        first = sorted(
            (
                item
                for item in indexed
                if first_start_frame <= item[0] <= first_end_frame
            ),
            key=lambda item: item[0],
        )
        second = sorted(
            (
                item
                for item in indexed
                if second_start_frame <= item[0] <= second_end_frame
            ),
            key=lambda item: item[0],
        )
        if not first or not second:
            raise ValueError("both cross-rematch intervals must contain images")

        pairs = [
            (image_pair_id(first_id, second_id), first_name, second_name)
            for _, first_id, first_name in first
            for _, second_id, second_name in second
        ]
        pair_ids = [(pair_id,) for pair_id, _, _ in pairs]
        removed_matches = sum(
            connection.execute("SELECT COUNT(*) FROM matches WHERE pair_id = ?", item).fetchone()[0]
            for item in pair_ids
        )
        removed_geometries = sum(
            connection.execute(
                "SELECT COUNT(*) FROM two_view_geometries WHERE pair_id = ?", item
            ).fetchone()[0]
            for item in pair_ids
        )
        with connection:
            connection.executemany("DELETE FROM matches WHERE pair_id = ?", pair_ids)
            connection.executemany(
                "DELETE FROM two_view_geometries WHERE pair_id = ?", pair_ids
            )
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
    finally:
        connection.close()

    pair_destination = Path(pair_list_path)
    pair_destination.parent.mkdir(parents=True, exist_ok=True)
    pair_destination.write_text(
        "".join(f"{first_name} {second_name}\n" for _, first_name, second_name in pairs),
        encoding="utf-8",
    )
    report = CrossIntervalRematchReport(
        first_start_frame=first_start_frame,
        first_end_frame=first_end_frame,
        second_start_frame=second_start_frame,
        second_end_frame=second_end_frame,
        first_images=len(first),
        second_images=len(second),
        generated_pairs=len(pairs),
        existing_match_rows_removed=removed_matches,
        existing_geometry_rows_removed=removed_geometries,
        database_integrity=integrity,
    )
    report_destination = Path(report_path)
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    report_destination.write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report
