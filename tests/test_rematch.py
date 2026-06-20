from __future__ import annotations

import sqlite3

from reconbot.rematch import (
    image_pair_id,
    prepare_cross_interval_rematch,
    prepare_targeted_rematch,
)


def test_pair_id_is_order_independent():
    assert image_pair_id(2, 7) == image_pair_id(7, 2)


def test_prepare_targeted_rematch_invalidates_only_selected_pairs(tmp_path):
    database = tmp_path / "database.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE matches(pair_id INTEGER PRIMARY KEY);
        CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY);
        """
    )
    images = [
        (1, "frame_00010_t00002.000.jpg"),
        (2, "frame_00011_t00002.200.jpg"),
        (3, "frame_00013_t00002.600.jpg"),
        (4, "frame_00020_t00004.000.jpg"),
    ]
    connection.executemany("INSERT INTO images VALUES (?, ?)", images)
    selected_pair = image_pair_id(1, 2)
    untouched_pair = image_pair_id(3, 4)
    connection.executemany("INSERT INTO matches VALUES (?)", [(selected_pair,), (untouched_pair,)])
    connection.executemany(
        "INSERT INTO two_view_geometries VALUES (?)", [(selected_pair,), (untouched_pair,)]
    )
    connection.commit()
    connection.close()

    pairs = tmp_path / "pairs.txt"
    report = prepare_targeted_rematch(
        database,
        pairs,
        tmp_path / "report.json",
        start_frame=10,
        end_frame=13,
        overlap=2,
    )

    assert report.generated_pairs == 2
    assert report.existing_match_rows_removed == 1
    assert report.database_integrity == "ok"
    assert pairs.read_text(encoding="utf-8").splitlines() == [
        "frame_00010_t00002.000.jpg frame_00011_t00002.200.jpg",
        "frame_00011_t00002.200.jpg frame_00013_t00002.600.jpg",
    ]
    connection = sqlite3.connect(database)
    assert connection.execute("SELECT pair_id FROM matches").fetchall() == [(untouched_pair,)]
    connection.close()


def test_prepare_cross_interval_rematch_generates_cartesian_pairs(tmp_path):
    database = tmp_path / "database.db"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE images(image_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE matches(pair_id INTEGER PRIMARY KEY);
        CREATE TABLE two_view_geometries(pair_id INTEGER PRIMARY KEY);
        """
    )
    images = [
        (1, "frame_00010_t00002.000.jpg"),
        (2, "frame_00011_t00002.200.jpg"),
        (3, "frame_00020_t00004.000.jpg"),
        (4, "frame_00021_t00004.200.jpg"),
    ]
    connection.executemany("INSERT INTO images VALUES (?, ?)", images)
    existing_pair = image_pair_id(1, 3)
    connection.execute("INSERT INTO matches VALUES (?)", (existing_pair,))
    connection.execute("INSERT INTO two_view_geometries VALUES (?)", (existing_pair,))
    connection.commit()
    connection.close()

    pairs = tmp_path / "pairs.txt"
    report = prepare_cross_interval_rematch(
        database,
        pairs,
        tmp_path / "report.json",
        first_start_frame=10,
        first_end_frame=11,
        second_start_frame=20,
        second_end_frame=21,
    )

    assert report.first_images == 2
    assert report.second_images == 2
    assert report.generated_pairs == 4
    assert report.existing_match_rows_removed == 1
    assert report.existing_geometry_rows_removed == 1
    assert report.database_integrity == "ok"
    assert pairs.read_text(encoding="utf-8").splitlines() == [
        "frame_00010_t00002.000.jpg frame_00020_t00004.000.jpg",
        "frame_00010_t00002.000.jpg frame_00021_t00004.200.jpg",
        "frame_00011_t00002.200.jpg frame_00020_t00004.000.jpg",
        "frame_00011_t00002.200.jpg frame_00021_t00004.200.jpg",
    ]
