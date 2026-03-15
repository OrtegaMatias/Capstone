from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.framework_service import framework_service


@pytest.fixture
def framework_repo_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(Path(__file__).resolve().parents[2] / "seed", repo_root / "seed")
    shutil.copytree(Path(__file__).resolve().parents[2] / "framework", repo_root / "framework")
    return repo_root


@pytest.fixture
def framework_client(framework_repo_root: Path):
    framework_service.reconfigure(
        repo_root=framework_repo_root,
        seed_dir=framework_repo_root / "seed",
        workspace_dir=framework_repo_root / "workspace",
        manifest_path=framework_repo_root / "framework/manifest.json",
    )
    with TestClient(app) as client:
        yield client
    framework_service.reconfigure()
