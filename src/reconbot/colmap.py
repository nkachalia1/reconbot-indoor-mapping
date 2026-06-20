"""Reproducible COLMAP sparse reconstruction orchestration through WSL."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath

MAPPER_PROFILES: dict[str, dict[str, str]] = {
    "baseline": {
        "--Mapper.ba_local_max_num_iterations": "25",
        "--Mapper.ba_local_max_refinements": "2",
        "--Mapper.ba_global_images_ratio": "1.1",
        "--Mapper.ba_global_points_ratio": "1.1",
        "--Mapper.ba_global_max_num_iterations": "50",
        "--Mapper.ba_global_max_refinements": "5",
    },
    "fast": {
        "--Mapper.ba_local_max_num_iterations": "10",
        "--Mapper.ba_local_max_refinements": "1",
        "--Mapper.ba_global_images_ratio": "1.5",
        "--Mapper.ba_global_points_ratio": "1.5",
        "--Mapper.ba_global_max_num_iterations": "20",
        "--Mapper.ba_global_max_refinements": "2",
    },
}


@dataclass(frozen=True)
class ColmapSparseConfig:
    camera_model: str = "SIMPLE_RADIAL"
    max_num_features: int = 4096
    sequential_overlap: int = 40
    loop_anchor_window: int = 10
    min_num_matches: int = 15
    mapper_profile: str = "baseline"
    use_gpu: bool = False
    wsl_distribution: str = "Ubuntu-22.04"
    wsl_run_root: str = "reconbot_runs"

    def __post_init__(self) -> None:
        if not self.camera_model.strip():
            raise ValueError("camera_model must not be empty")
        if self.max_num_features <= 0:
            raise ValueError("max_num_features must be positive")
        if self.sequential_overlap <= 0:
            raise ValueError("sequential_overlap must be positive")
        if self.loop_anchor_window <= 0:
            raise ValueError("loop_anchor_window must be positive")
        if self.min_num_matches <= 0:
            raise ValueError("min_num_matches must be positive")
        if self.mapper_profile not in MAPPER_PROFILES:
            choices = ", ".join(sorted(MAPPER_PROFILES))
            raise ValueError(f"mapper_profile must be one of: {choices}")
        if not self.wsl_run_root.strip():
            raise ValueError("wsl_run_root must not be empty")


@dataclass(frozen=True)
class ColmapRunReport:
    images: int
    loop_pairs: int
    selected_model: str
    registered_images: int | None
    points: int | None
    observations: int | None
    mean_track_length: float | None
    mean_observations_per_image: float | None
    mean_reprojection_error_px: float | None
    wsl_workspace: str
    reused_matched_database: bool
    matched_database_source: str | None
    config: dict[str, object]
    analyzer_output: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def windows_to_wsl_path(path: Path) -> str:
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"path has no Windows drive: {resolved}")
    relative = resolved.as_posix().split(":", maxsplit=1)[1].lstrip("/")
    return str(PurePosixPath("/mnt") / drive / relative)


def write_loop_pairs(image_paths: list[Path], destination: Path, window: int) -> int:
    if window <= 0:
        raise ValueError("window must be positive")
    if len(image_paths) < 2:
        raise ValueError("at least two images are required")
    selected = min(window, len(image_paths) // 2)
    starts = image_paths[:selected]
    ends = image_paths[-selected:]
    pairs = [(start.name, end.name) for start in starts for end in ends]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(f"{start} {end}\n" for start, end in pairs),
        encoding="utf-8",
    )
    return len(pairs)


def build_sparse_commands(
    *,
    images_path: str,
    database_path: str,
    sparse_path: str,
    loop_pairs_path: str,
    config: ColmapSparseConfig,
) -> tuple[tuple[str, ...], ...]:
    gpu = "1" if config.use_gpu else "0"
    mapper_options = tuple(
        item
        for option, value in MAPPER_PROFILES[config.mapper_profile].items()
        for item in (option, value)
    )
    return (
        (
            "colmap",
            "feature_extractor",
            "--database_path",
            database_path,
            "--image_path",
            images_path,
            "--ImageReader.camera_model",
            config.camera_model,
            "--ImageReader.single_camera",
            "1",
            "--SiftExtraction.use_gpu",
            gpu,
            "--SiftExtraction.max_num_features",
            str(config.max_num_features),
        ),
        (
            "colmap",
            "sequential_matcher",
            "--database_path",
            database_path,
            "--SiftMatching.use_gpu",
            gpu,
            "--SequentialMatching.overlap",
            str(config.sequential_overlap),
        ),
        (
            "colmap",
            "matches_importer",
            "--database_path",
            database_path,
            "--match_list_path",
            loop_pairs_path,
            "--match_type",
            "pairs",
            "--SiftMatching.use_gpu",
            gpu,
        ),
        (
            "colmap",
            "mapper",
            "--database_path",
            database_path,
            "--image_path",
            images_path,
            "--output_path",
            sparse_path,
            "--Mapper.min_num_matches",
            str(config.min_num_matches),
            *mapper_options,
        ),
    )


def _parse_optional_number(pattern: str, text: str, cast: type[int] | type[float]):
    match = re.search(pattern, text)
    return cast(match.group(1)) if match else None


def parse_model_analyzer(text: str) -> dict[str, int | float | None]:
    return {
        "registered_images": _parse_optional_number(r"Registered images:\s+(\d+)", text, int),
        "points": _parse_optional_number(r"Points:\s+(\d+)", text, int),
        "observations": _parse_optional_number(r"Observations:\s+(\d+)", text, int),
        "mean_track_length": _parse_optional_number(
            r"Mean track length:\s+([0-9.eE+-]+)", text, float
        ),
        "mean_observations_per_image": _parse_optional_number(
            r"Mean observations per image:\s+([0-9.eE+-]+)", text, float
        ),
        "mean_reprojection_error_px": _parse_optional_number(
            r"Mean reprojection error:\s+([0-9.eE+-]+)px", text, float
        ),
    }


def _wsl_command(distribution: str, args: tuple[str, ...]) -> list[str]:
    return ["wsl.exe", "-d", distribution, "--", *args]


def _run_wsl(
    args: tuple[str, ...],
    *,
    distribution: str,
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            _wsl_command(distribution, args),
            check=False,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            f"WSL stage failed with exit code {result.returncode}; see {log_path}"
        )


def _run_wsl_capture(
    args: tuple[str, ...],
    *,
    distribution: str,
    log_path: Path,
) -> str:
    result = subprocess.run(
        _wsl_command(distribution, args),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(
            f"WSL command failed with exit code {result.returncode}; see {log_path}"
        )
    return result.stdout


def _wsl_home(distribution: str, log_path: Path) -> str:
    return _run_wsl_capture(
        ("bash", "-lc", 'printf %s "$HOME"'),
        distribution=distribution,
        log_path=log_path,
    ).strip()


def run_sparse_colmap_wsl(
    images_dir: Path,
    workspace_dir: Path,
    config: ColmapSparseConfig | None = None,
    matched_database: Path | None = None,
) -> ColmapRunReport:
    selected = config or ColmapSparseConfig()
    images = sorted(Path(images_dir).glob("*.jpg"))
    if len(images) < 2:
        raise ValueError(f"expected at least two JPG images in {images_dir}")
    cached_database = Path(matched_database).resolve() if matched_database else None
    if cached_database is not None and not cached_database.is_file():
        raise FileNotFoundError(f"matched database not found: {cached_database}")

    workspace = Path(workspace_dir)
    logs = workspace / "logs"
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"output workspace is not empty: {workspace}")
    workspace.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    pairs_path = workspace / "loop_pairs.txt"
    loop_pairs = write_loop_pairs(images, pairs_path, selected.loop_anchor_window)
    home = _wsl_home(selected.wsl_distribution, logs / "wsl_home.log")
    wsl_workspace = str(
        PurePosixPath(home) / selected.wsl_run_root / workspace.name
    )
    existence = subprocess.run(
        _wsl_command(selected.wsl_distribution, ("test", "-e", wsl_workspace)),
        check=False,
    )
    if existence.returncode == 0:
        raise FileExistsError(f"WSL workspace already exists: {wsl_workspace}")

    wsl_images = str(PurePosixPath(wsl_workspace) / "images")
    wsl_database = str(PurePosixPath(wsl_workspace) / "database.db")
    wsl_sparse = str(PurePosixPath(wsl_workspace) / "sparse")
    wsl_pairs = str(PurePosixPath(wsl_workspace) / "loop_pairs.txt")
    _run_wsl(
        ("mkdir", "-p", wsl_images, wsl_sparse),
        distribution=selected.wsl_distribution,
        log_path=logs / "prepare_workspace.log",
    )
    _run_wsl(
        ("cp", "-a", f"{windows_to_wsl_path(Path(images_dir))}/.", f"{wsl_images}/"),
        distribution=selected.wsl_distribution,
        log_path=logs / "stage_images.log",
    )
    _run_wsl(
        ("cp", windows_to_wsl_path(pairs_path), wsl_pairs),
        distribution=selected.wsl_distribution,
        log_path=logs / "stage_loop_pairs.log",
    )

    commands = build_sparse_commands(
        images_path=wsl_images,
        database_path=wsl_database,
        sparse_path=wsl_sparse,
        loop_pairs_path=wsl_pairs,
        config=selected,
    )
    if cached_database is None:
        stage_names = ("feature_extractor", "sequential_matcher", "loop_pairs")
        for stage_name, command in zip(stage_names, commands[:3], strict=True):
            print(f"Running COLMAP stage: {stage_name}", flush=True)
            _run_wsl(
                command,
                distribution=selected.wsl_distribution,
                log_path=logs / f"{stage_name}.log",
            )
        print("Caching completed COLMAP feature/match database", flush=True)
        _run_wsl(
            (
                "cp",
                wsl_database,
                windows_to_wsl_path(workspace / "matched_database.db"),
            ),
            distribution=selected.wsl_distribution,
            log_path=logs / "export_matched_database.log",
        )
    else:
        print(f"Reusing matched COLMAP database: {cached_database}", flush=True)
        _run_wsl(
            ("cp", windows_to_wsl_path(cached_database), wsl_database),
            distribution=selected.wsl_distribution,
            log_path=logs / "stage_matched_database.log",
        )
        database_images = int(
            _run_wsl_capture(
                (
                    "python3",
                    "-c",
                    (
                        "import sqlite3,sys; "
                        "c=sqlite3.connect(sys.argv[1]); "
                        "print(c.execute('select count(*) from images').fetchone()[0]); "
                        "c.close()"
                    ),
                    wsl_database,
                ),
                distribution=selected.wsl_distribution,
                log_path=logs / "validate_matched_database.log",
            ).strip()
        )
        if database_images != len(images):
            raise ValueError(
                "matched database image count does not match frame set: "
                f"{database_images} != {len(images)}"
            )

    print(f"Running COLMAP stage: mapper ({selected.mapper_profile} profile)", flush=True)
    _run_wsl(
        commands[3],
        distribution=selected.wsl_distribution,
        log_path=logs / "mapper.log",
    )

    model_names = _run_wsl_capture(
        (
            "find",
            wsl_sparse,
            "-mindepth",
            "1",
            "-maxdepth",
            "1",
            "-type",
            "d",
            "-print0",
        ),
        distribution=selected.wsl_distribution,
        log_path=logs / "list_models.log",
    )
    models = sorted(
        PurePosixPath(path).name for path in model_names.split("\0") if path.strip()
    )
    if not models:
        raise RuntimeError("COLMAP mapper produced no sparse models")

    analyzed: list[
        tuple[int, str, str, dict[str, int | float | None]]
    ] = []
    for model_name in models:
        model_path = str(PurePosixPath(wsl_sparse) / model_name)
        analyzer_output = _run_wsl_capture(
            ("colmap", "model_analyzer", "--path", model_path),
            distribution=selected.wsl_distribution,
            log_path=logs / f"model_analyzer_{model_name}.log",
        )
        metrics = parse_model_analyzer(analyzer_output)
        analyzed.append(
            (int(metrics["registered_images"] or 0), model_name, analyzer_output, metrics)
        )
    _, best_model_name, analyzer_output, metrics = max(
        analyzed, key=lambda item: item[0]
    )
    best_model = str(PurePosixPath(wsl_sparse) / best_model_name)

    wsl_text_model = str(PurePosixPath(wsl_workspace) / "sparse_text")
    _run_wsl(
        ("mkdir", "-p", wsl_text_model),
        distribution=selected.wsl_distribution,
        log_path=logs / "prepare_text_model.log",
    )
    _run_wsl(
        (
            "colmap",
            "model_converter",
            "--input_path",
            best_model,
            "--output_path",
            wsl_text_model,
            "--output_type",
            "TXT",
        ),
        distribution=selected.wsl_distribution,
        log_path=logs / "model_converter.log",
    )

    _run_wsl(
        ("cp", wsl_database, windows_to_wsl_path(workspace / "database.db")),
        distribution=selected.wsl_distribution,
        log_path=logs / "export_database.log",
    )
    _run_wsl(
        ("cp", "-a", best_model, windows_to_wsl_path(workspace / "sparse_model")),
        distribution=selected.wsl_distribution,
        log_path=logs / "export_sparse_model.log",
    )
    _run_wsl(
        ("cp", "-a", wsl_text_model, windows_to_wsl_path(workspace / "sparse_text")),
        distribution=selected.wsl_distribution,
        log_path=logs / "export_text_model.log",
    )

    report = ColmapRunReport(
        images=len(images),
        loop_pairs=loop_pairs,
        selected_model=best_model_name,
        registered_images=metrics["registered_images"],  # type: ignore[arg-type]
        points=metrics["points"],  # type: ignore[arg-type]
        observations=metrics["observations"],  # type: ignore[arg-type]
        mean_track_length=metrics["mean_track_length"],  # type: ignore[arg-type]
        mean_observations_per_image=metrics["mean_observations_per_image"],  # type: ignore[arg-type]
        mean_reprojection_error_px=metrics["mean_reprojection_error_px"],  # type: ignore[arg-type]
        wsl_workspace=wsl_workspace,
        reused_matched_database=cached_database is not None,
        matched_database_source=(str(cached_database) if cached_database else None),
        config={
            **asdict(selected),
            "resolved_mapper_options": MAPPER_PROFILES[selected.mapper_profile],
        },
        analyzer_output=analyzer_output,
    )
    (workspace / "sparse_report.json").write_text(
        json.dumps(report.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report
