from __future__ import annotations

from pathlib import Path

import pytest

from reconbot.colmap import (
    MAPPER_PROFILES,
    ColmapSparseConfig,
    build_sparse_commands,
    parse_model_analyzer,
    windows_to_wsl_path,
    write_loop_pairs,
)


def test_windows_path_conversion():
    converted = windows_to_wsl_path(Path(r"C:\Users\Neel\frames"))
    assert converted == "/mnt/c/Users/Neel/frames"


def test_loop_pairs_cross_start_and_end_windows(tmp_path):
    images = [tmp_path / f"frame_{index:03d}.jpg" for index in range(20)]
    destination = tmp_path / "pairs.txt"

    count = write_loop_pairs(images, destination, window=3)

    lines = destination.read_text().splitlines()
    assert count == 9
    assert lines[0] == "frame_000.jpg frame_017.jpg"
    assert lines[-1] == "frame_002.jpg frame_019.jpg"


def test_sparse_commands_pin_cpu_and_overlap():
    config = ColmapSparseConfig(sequential_overlap=40, use_gpu=False)
    commands = build_sparse_commands(
        images_path="/images",
        database_path="/run/database.db",
        sparse_path="/run/sparse",
        loop_pairs_path="/run/loop_pairs.txt",
        config=config,
    )

    assert commands[0][0:2] == ("colmap", "feature_extractor")
    assert "--SiftExtraction.use_gpu" in commands[0]
    assert commands[1][commands[1].index("--SequentialMatching.overlap") + 1] == "40"
    assert commands[2][0:2] == ("colmap", "matches_importer")
    assert commands[3][0:2] == ("colmap", "mapper")
    assert commands[3][commands[3].index("--Mapper.ba_global_max_num_iterations") + 1] == "50"


def test_fast_mapper_profile_reduces_bundle_adjustment_work():
    config = ColmapSparseConfig(mapper_profile="fast")
    commands = build_sparse_commands(
        images_path="/images",
        database_path="/run/database.db",
        sparse_path="/run/sparse",
        loop_pairs_path="/run/loop_pairs.txt",
        config=config,
    )
    mapper = commands[3]

    assert MAPPER_PROFILES["fast"]["--Mapper.ba_local_max_num_iterations"] == "10"
    assert mapper[mapper.index("--Mapper.ba_global_images_ratio") + 1] == "1.5"
    assert mapper[mapper.index("--Mapper.ba_global_max_refinements") + 1] == "2"


def test_unknown_mapper_profile_is_rejected():
    with pytest.raises(ValueError, match="mapper_profile"):
        ColmapSparseConfig(mapper_profile="turbo")


def test_model_analyzer_parser():
    metrics = parse_model_analyzer(
        """
        Registered images: 812
        Points: 153204
        Observations: 706000
        Mean track length: 4.6082
        Mean observations per image: 869.45
        Mean reprojection error: 0.842px
        """
    )

    assert metrics["registered_images"] == 812
    assert metrics["points"] == 153204
    assert metrics["mean_reprojection_error_px"] == pytest.approx(0.842)
