from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Application settings for the ITK Snap ADPKD plugin."""

    model_config = SettingsConfigDict(env_file=".env")
    adpkd_url: str = "http://adpkd:9000"
    adpkd_shared_input_dir: Path = Path("/data/adpkd/inputs/itksnap")
    adpkd_shared_output_dir: Path = Path("/data/adpkd/outputs")
    adpkd_run_small: bool = False
    adpkd_run_cpu: bool = False
    adpkd_poll_interval: float = 2.0
    adpkd_timeout: float = 3600.0


@lru_cache
def get_plugin_settings() -> AppSettings:
    """Return a cached :class:`AppSettings` instance."""
    return AppSettings()
