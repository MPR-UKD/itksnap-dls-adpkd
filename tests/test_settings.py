"""Tests for the ADPKD plugin settings."""

from pathlib import Path

import pytest

from itksnap_dls.settings import AppSettings, get_plugin_settings

pytestmark = pytest.mark.unit

ENV_KEYS = [
    "ADPKD_URL",
    "ADPKD_SHARED_INPUT_DIR",
    "ADPKD_SHARED_OUTPUT_DIR",
    "ADPKD_RUN_SMALL",
    "ADPKD_RUN_CPU",
    "ADPKD_POLL_INTERVAL",
    "ADPKD_TIMEOUT",
]


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Remove ADPKD env vars and run from a dir without a .env file."""
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_plugin_settings.cache_clear()
    yield
    get_plugin_settings.cache_clear()


class TestAppSettings:
    """Tests for defaults, overrides and caching."""

    def test_defaults(self, clean_env):
        """Without env vars the documented defaults are used."""
        settings = AppSettings()

        assert settings.adpkd_url == "http://adpkd:9000"
        assert settings.adpkd_shared_input_dir == Path("/data/adpkd/inputs/itksnap")
        assert settings.adpkd_shared_output_dir == Path("/data/adpkd/outputs")
        assert settings.adpkd_run_small is False
        assert settings.adpkd_run_cpu is False
        assert settings.adpkd_poll_interval == 2.0
        assert settings.adpkd_timeout == 3600.0

    def test_env_overrides(self, clean_env, monkeypatch):
        """Env vars override the defaults and are parsed to the field types."""
        monkeypatch.setenv("ADPKD_URL", "http://other:1234")
        monkeypatch.setenv("ADPKD_RUN_CPU", "true")
        monkeypatch.setenv("ADPKD_TIMEOUT", "60")

        settings = AppSettings()

        assert settings.adpkd_url == "http://other:1234"
        assert settings.adpkd_run_cpu is True
        assert settings.adpkd_timeout == 60.0

    def test_env_file_is_read(self, clean_env, tmp_path):
        """A .env file in the working directory is read."""
        (tmp_path / ".env").write_text("ADPKD_POLL_INTERVAL=0.25\n")
        assert AppSettings().adpkd_poll_interval == 0.25

    def test_get_plugin_settings_is_cached(self, clean_env):
        """get_plugin_settings returns the same instance on repeated calls."""
        assert get_plugin_settings() is get_plugin_settings()
