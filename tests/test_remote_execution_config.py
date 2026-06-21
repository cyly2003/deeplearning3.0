from __future__ import annotations

from pathlib import Path

from qsar_tl.config import parse_execution_config
from qsar_tl.training.runner import RemoteRunner, build_runner


def test_parse_remote_execution_port() -> None:
    config = {
        "execution": {
            "mode": "remote",
            "remote": {
                "host": "gpu.example.org",
                "port": "32136",
                "user": "researcher",
                "project_dir": "/srv/ecotox",
                "python": "/opt/conda/bin/python",
            },
        }
    }

    execution = parse_execution_config(config)

    assert execution.remote_port == 32136


def test_build_runner_carries_remote_port() -> None:
    runner = build_runner(
        {
            "execution": {
                "mode": "remote",
                "remote": {
                    "host": "gpu.example.org",
                    "port": 32136,
                    "user": "researcher",
                    "project_dir": "/srv/ecotox",
                    "python": "/opt/conda/bin/python",
                },
            }
        }
    )

    assert isinstance(runner, RemoteRunner)
    assert runner.port == 32136


def test_remote_runner_dry_run_includes_ssh_port(capsys) -> None:
    runner = RemoteRunner(
        host="gpu.example.org",
        port=32136,
        user="researcher",
        project_dir="/srv/ecotox",
        remote_python="/opt/conda/bin/python",
    )

    runner.run(Path("configs/experiment.example.yaml"), dry_run=True)

    output = capsys.readouterr().out
    assert "ssh -p 32136 researcher@gpu.example.org" in output
    assert "/srv/ecotox/configs/experiment.example.yaml" in output
