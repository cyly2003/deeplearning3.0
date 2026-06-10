from __future__ import annotations

from pathlib import Path

from qsar_tl.gui.job_model import JobStatus, TrainingJob
from qsar_tl.gui.process_runner import build_training_command


def test_training_job_tracks_status_timestamps() -> None:
    job = TrainingJob(config_path=Path("configs/experiment.example.yaml"), execution_mode="remote")

    assert job.status == JobStatus.QUEUED
    assert job.started_at is None
    assert job.finished_at is None

    job.set_status(JobStatus.RUNNING)
    assert job.started_at is not None
    assert job.finished_at is None

    job.set_status(JobStatus.SUCCEEDED)
    assert job.finished_at is not None
    assert job.is_terminal


def test_build_training_command_supports_local_and_remote() -> None:
    config_path = Path("configs/experiment.example.yaml")

    local_job = TrainingJob(config_path=config_path, execution_mode="local")
    assert build_training_command(local_job, local_python="python") == [
        "python",
        "-m",
        "qsar_tl.training.train",
        "--config",
        str(config_path),
    ]

    remote_job = TrainingJob(config_path=config_path, execution_mode="remote")
    command = build_training_command(
        remote_job,
        remote_host="gpu.example.org",
        remote_user="researcher",
        remote_project_dir="/srv/ecotox",
        remote_python="/opt/conda/envs/qsar/bin/python",
    )
    assert command[0] == "ssh"
    assert command[1] == "researcher@gpu.example.org"
    assert "qsar_tl.training.train" in command[2]
    assert "/srv/ecotox/configs/experiment.example.yaml" in command[2]

    explicit_remote_config = TrainingJob(
        config_path=config_path,
        execution_mode="remote",
        remote_config_path="/scratch/run_configs/custom.yaml",
    )
    explicit_command = build_training_command(
        explicit_remote_config,
        remote_host="gpu.example.org",
        remote_user="researcher",
        remote_project_dir="/srv/ecotox",
        remote_python="/opt/conda/envs/qsar/bin/python",
    )
    assert "/scratch/run_configs/custom.yaml" in explicit_command[2]
