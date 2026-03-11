from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.v1.endpoints import datasets, framework, notes, pivot, supervised, upload
from app.main import app
from app.services.framework_service import framework_service
from app.storage.file_store import DatasetFileStore
from app.storage.memory_cache import DatasetMemoryCache


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    test_store = DatasetFileStore(tmp_path / "datasets")

    for service in [upload.service, datasets.service, supervised.service, notes.dataset_service, pivot.service]:
        service.file_store = test_store
        service.cache = DatasetMemoryCache(max_items=10)

    notes.notes_service.file_store = test_store

    return TestClient(app)


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
