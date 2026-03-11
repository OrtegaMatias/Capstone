import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


def _discover_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "framework/manifest.json").exists() and (parent / "seed").exists():
            return parent
    return current.parents[3]


REPO_ROOT = _discover_repo_root()


class Settings(BaseModel):
    app_name: str = "Capstone ETL EDA API"
    api_v1_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"])
    cors_origin_regex: str = (
        r"^https?://("
        r"localhost|127\.0\.0\.1|0\.0\.0\.0|"
        r"192\.168\.\d{1,3}\.\d{1,3}|"
        r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
        r"172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}"
        r")(:\d+)?$"
    )
    repo_root: Path = REPO_ROOT
    datasets_dir: Path = REPO_ROOT / "backend/data/datasets"
    seed_dir: Path = REPO_ROOT / "seed"
    workspace_dir: Path = REPO_ROOT / "workspace"
    framework_manifest_path: Path = REPO_ROOT / "framework/manifest.json"
    cache_max_items: int = 10


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    app_name = os.getenv("APP_NAME", "Capstone ETL EDA API")
    api_v1_prefix = os.getenv("API_V1_PREFIX", "/api/v1")
    repo_root = Path(os.getenv("REPO_ROOT", str(REPO_ROOT))).resolve()
    datasets_dir = Path(os.getenv("DATASETS_DIR", str(repo_root / "backend/data/datasets"))).resolve()
    seed_dir = Path(os.getenv("SEED_DIR", str(repo_root / "seed"))).resolve()
    workspace_dir = Path(os.getenv("WORKSPACE_DIR", str(repo_root / "workspace"))).resolve()
    framework_manifest_path = Path(
        os.getenv("FRAMEWORK_MANIFEST_PATH", str(repo_root / "framework/manifest.json"))
    ).resolve()
    cache_max_items = int(os.getenv("CACHE_MAX_ITEMS", "10"))
    cors_origins_env = os.getenv("CORS_ORIGINS", "")
    cors_origins = [item.strip() for item in cors_origins_env.split(",") if item.strip()]
    if not cors_origins:
        cors_origins = ["http://localhost:5173", "http://127.0.0.1:5173"]
    cors_origin_regex = os.getenv(
        "CORS_ORIGIN_REGEX",
        (
            r"^https?://("
            r"localhost|127\.0\.0\.1|0\.0\.0\.0|"
            r"192\.168\.\d{1,3}\.\d{1,3}|"
            r"10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"172\.(1[6-9]|2[0-9]|3[0-1])\.\d{1,3}\.\d{1,3}"
            r")(:\d+)?$"
        ),
    )

    return Settings(
        app_name=app_name,
        api_v1_prefix=api_v1_prefix,
        cors_origins=cors_origins,
        cors_origin_regex=cors_origin_regex,
        repo_root=repo_root,
        datasets_dir=datasets_dir,
        seed_dir=seed_dir,
        workspace_dir=workspace_dir,
        framework_manifest_path=framework_manifest_path,
        cache_max_items=cache_max_items,
    )
