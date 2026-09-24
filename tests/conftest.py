"""Shared fixtures for the itksnap_dls test suite."""

import pytest
from fastapi.testclient import TestClient

from itksnap_dls.server import app
from itksnap_dls.session import session_manager
from itksnap_dls.settings import get_plugin_settings

from .test_toolbox import FakeAdpkdClient, make_image


@pytest.fixture
def anyio_backend():
    """Run ``@pytest.mark.anyio`` tests on asyncio only (the server's backend)."""
    return "asyncio"


@pytest.fixture
def adpkd_settings(tmp_path, monkeypatch):
    """Point the ADPKD settings at temporary shared dirs with fast polling."""
    monkeypatch.setenv("ADPKD_URL", "http://adpkd-test:9000")
    monkeypatch.setenv("ADPKD_SHARED_INPUT_DIR", str(tmp_path / "inputs"))
    monkeypatch.setenv("ADPKD_SHARED_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("ADPKD_POLL_INTERVAL", "0")
    monkeypatch.setenv("ADPKD_TIMEOUT", "5")
    get_plugin_settings.cache_clear()
    yield get_plugin_settings()
    get_plugin_settings.cache_clear()


@pytest.fixture
def sample_image():
    """Small float32 3D image with non-default spacing and origin."""
    return make_image()


@pytest.fixture
def fake_client(adpkd_settings, mocker):
    """Replace the adpkd-net HTTP client with an in-memory fake container."""
    client = FakeAdpkdClient(adpkd_settings.adpkd_shared_output_dir)
    mocker.patch("itksnap_dls.adpkd_seg.AdpkdClient", return_value=client)
    return client


@pytest.fixture
def api_client():
    """FastAPI test client that keeps one event loop across requests.

    The context manager matters: ADPKD starts its job as a background task on
    upload, which must survive until the following interaction request.
    """
    session_manager.sessions.clear()
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
    session_manager.sessions.clear()
