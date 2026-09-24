"""Tests for the FastAPI server: helpers and the HTTP API with the ADPKD model."""

import threading

import numpy as np
import pytest
import SimpleITK as sitk

from itksnap_dls.server import call_model, read_sitk_image
from itksnap_dls.session import session_manager

from .test_toolbox import (
    IMAGE_SHAPE_ZYX,
    decode_mask,
    encode_upload,
    make_label_array,
)

EXPECTED_MASK = (make_label_array() > 0).astype(np.int8)


class SyncFakeWrapper:
    """Synchronous wrapper standing in for nnInteractive/SAM2 in API tests."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.threads: set[int] = set()
        self.image = None

    def set_image(self, sitk_image):
        self.image = sitk_image

    def _record(self, name, *args):
        self.calls.append((name, *args))
        self.threads.add(threading.get_ident())

    def add_point_interaction(self, index_itk, include_interaction):
        self._record("point", list(index_itk), include_interaction)

    def add_scribble_interaction(self, sitk_image, include_interaction):
        self._record("scribble", include_interaction)

    def add_lasso_interaction(self, sitk_image, include_interaction):
        self._record("lasso", include_interaction)

    def reset_interactions(self):
        self._record("reset")

    def get_result(self):
        self._record("result")
        labels = np.zeros(IMAGE_SHAPE_ZYX, dtype=np.uint8)
        labels[0, 0, 0] = 5
        return sitk.GetImageFromArray(labels)


def start_adpkd_session(api_client) -> str:
    """Start an ADPKD session and upload the sample image; return the session id."""
    session_id = api_client.get("/v2/start_session/ADPKD").json()["session_id"]
    payload, metadata = encode_upload(np.ones(IMAGE_SHAPE_ZYX))
    response = api_client.post(
        f"/v2/upload_raw/{session_id}",
        files={"file": ("image", payload)},
        data={"metadata": metadata},
    )
    assert response.status_code == 200, response.text
    return session_id


def post_image_interaction(api_client, kind, session_id, foreground=True):
    """POST a scribble or lasso interaction with an empty mask image."""
    payload, metadata = encode_upload(np.zeros(IMAGE_SHAPE_ZYX))
    return api_client.post(
        f"/process_{kind}_interaction/{session_id}",
        params={"foreground": foreground},
        files={"file": ("mask", payload)},
        data={"metadata": metadata},
    )


@pytest.mark.unit
class TestReadSitkImage:
    """Tests for decoding uploaded images."""

    def test_scalar_image(self):
        """A scalar upload becomes an image with ITK (x, y, z) size and same values."""
        array = np.arange(np.prod(IMAGE_SHAPE_ZYX), dtype=np.float32).reshape(
            IMAGE_SHAPE_ZYX
        )
        payload, metadata = encode_upload(array)

        image = read_sitk_image(payload, metadata)

        assert image.GetSize() == (6, 5, 4)
        assert image.GetNumberOfComponentsPerPixel() == 1
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), array)

    def test_vector_image(self):
        """A 3-component upload becomes a vector image."""
        array = np.random.default_rng(0).random((5, 6, 3)).astype(np.float32)
        payload, metadata = encode_upload(array, components_per_pixel=3)

        image = read_sitk_image(payload, metadata)

        assert image.GetSize() == (6, 5)
        assert image.GetNumberOfComponentsPerPixel() == 3
        np.testing.assert_array_equal(sitk.GetArrayFromImage(image), array)


@pytest.mark.unit
@pytest.mark.anyio
class TestCallModel:
    """Tests for dispatching to sync and async wrapper methods."""

    async def test_async_method_is_awaited(self):
        """An async method is awaited and its result returned."""

        async def method(value, *, scale):
            return value * scale

        assert await call_model(method, 2, scale=3) == 6

    async def test_sync_method_runs_in_threadpool(self):
        """A sync method runs off the event loop thread and its result is returned."""
        loop_thread = threading.get_ident()

        def method(value, *, scale):
            return value * scale, threading.get_ident()

        result, method_thread = await call_model(method, 2, scale=3)

        assert result == 6
        assert method_thread != loop_thread


@pytest.mark.unit
class TestSessionManager:
    """Tests for session bookkeeping."""

    def test_create_get_delete(self):
        """A created session can be fetched and then deleted exactly once."""
        session_id = session_manager.create_session("data", "sid-1")
        try:
            assert session_manager.get_session(session_id) == "data"
            assert session_manager.delete_session(session_id) is True
            assert session_manager.delete_session(session_id) is False
        finally:
            session_manager.sessions.pop("sid-1", None)

    @pytest.mark.xfail(
        strict=True,
        reason="create_session's default id is evaluated once at import time",
    )
    def test_sessions_get_unique_ids(self):
        """Two sessions without an explicit id get different ids."""
        first = session_manager.create_session("a")
        second = session_manager.create_session("b")
        try:
            assert first != second
        finally:
            session_manager.sessions.pop(first, None)
            session_manager.sessions.pop(second, None)


@pytest.mark.integration
class TestAdpkdApi:
    """End-to-end ADPKD flow through the HTTP API with a fake container."""

    def test_models_endpoint_lists_adpkd(self, api_client):
        """/v2/models includes the ADPKD model."""
        models = api_client.get("/v2/models").json()["models"]
        assert "ADPKD" in [model["id"] for model in models]

    def test_upload_submits_job_before_interaction(self, api_client, fake_client):
        """Uploading an image starts the container job without waiting for a click."""
        session_id = start_adpkd_session(api_client)
        # Any follow-up request lets the background task run on the shared loop
        api_client.get(f"/v2/reset_interactions/{session_id}")

        assert len(fake_client.submitted) == 1

    @pytest.mark.parametrize("foreground", [True, False])
    def test_point_interaction_returns_mask(self, api_client, fake_client, foreground):
        """A point interaction returns the binarised ADPKD segmentation."""
        session_id = start_adpkd_session(api_client)

        response = api_client.get(
            f"/v2/process_point_interaction/{session_id}",
            params={"point": [1, 2, 3], "foreground": foreground},
        )

        assert response.status_code == 200, response.text
        assert response.json()["status"] == "success"
        np.testing.assert_array_equal(
            decode_mask(response.json()["result"]), EXPECTED_MASK
        )

    def test_legacy_point_interaction_returns_mask(self, api_client, fake_client):
        """The legacy x/y/z point endpoint returns the same mask."""
        session_id = start_adpkd_session(api_client)

        response = api_client.get(
            f"/process_point_interaction/{session_id}",
            params={"x": 1, "y": 2, "z": 3, "foreground": True},
        )

        assert response.status_code == 200, response.text
        np.testing.assert_array_equal(
            decode_mask(response.json()["result"]), EXPECTED_MASK
        )

    @pytest.mark.parametrize("kind", ["scribble", "lasso"])
    def test_image_interaction_returns_mask(self, api_client, fake_client, kind):
        """Scribble and lasso interactions return the ADPKD segmentation."""
        session_id = start_adpkd_session(api_client)

        response = post_image_interaction(api_client, kind, session_id)

        assert response.status_code == 200, response.text
        np.testing.assert_array_equal(
            decode_mask(response.json()["result"]), EXPECTED_MASK
        )

    def test_repeated_interactions_and_reset_submit_one_job(
        self, api_client, fake_client
    ):
        """Clicks, resets and scribbles on one image reuse a single job."""
        session_id = start_adpkd_session(api_client)
        point_url = f"/v2/process_point_interaction/{session_id}"

        api_client.get(point_url, params={"point": [0, 0, 0], "foreground": True})
        api_client.get(f"/v2/reset_interactions/{session_id}")
        api_client.get(point_url, params={"point": [1, 1, 1], "foreground": False})
        post_image_interaction(api_client, "scribble", session_id)

        assert len(fake_client.submitted) == 1

    def test_new_upload_submits_new_job(self, api_client, fake_client):
        """Uploading another image to the same session runs a new job."""
        session_id = start_adpkd_session(api_client)
        api_client.get(
            f"/v2/process_point_interaction/{session_id}",
            params={"point": [0, 0, 0], "foreground": True},
        )
        payload, metadata = encode_upload(np.ones(IMAGE_SHAPE_ZYX))
        api_client.post(
            f"/v2/upload_raw/{session_id}",
            files={"file": ("image", payload)},
            data={"metadata": metadata},
        )
        api_client.get(
            f"/v2/process_point_interaction/{session_id}",
            params={"point": [0, 0, 0], "foreground": True},
        )

        assert len(fake_client.submitted) == 2

    def test_failed_job_returns_server_error(self, api_client, fake_client):
        """A failed container job results in an HTTP 500, not an empty mask."""
        fake_client.statuses = ["failed"]
        session_id = start_adpkd_session(api_client)

        response = api_client.get(
            f"/v2/process_point_interaction/{session_id}",
            params={"point": [0, 0, 0], "foreground": True},
        )

        assert response.status_code == 500

    def test_interaction_before_upload_returns_server_error(self, api_client):
        """An interaction on a session without an image fails cleanly."""
        session_id = api_client.get("/v2/start_session/ADPKD").json()["session_id"]

        response = api_client.get(
            f"/v2/process_point_interaction/{session_id}",
            params={"point": [0, 0, 0], "foreground": True},
        )

        assert response.status_code == 500

    def test_end_session(self, api_client, fake_client):
        """Ending a session removes it; ending it again reports it as invalid."""
        session_id = start_adpkd_session(api_client)

        first = api_client.get(f"/v2/end_session/{session_id}").json()
        second = api_client.get(f"/v2/end_session/{session_id}").json()

        assert first == {"message": "Session ended"}
        assert second == {"message": "Invalid session"}


@pytest.mark.integration
class TestSyncModelApi:
    """The API still drives synchronous wrappers (nnInteractive, SAM2)."""

    @pytest.fixture
    def sync_session(self, api_client):
        """Register a synchronous fake wrapper as a session."""
        wrapper = SyncFakeWrapper()
        session_manager.create_session(wrapper, "sync-session")
        return wrapper

    def test_point_interaction(self, api_client, sync_session):
        """Point interactions call the sync wrapper and return its mask."""
        response = api_client.get(
            "/v2/process_point_interaction/sync-session",
            params={"point": [1, 2, 3], "foreground": True},
        )

        assert response.status_code == 200, response.text
        assert sync_session.calls == [("point", [1, 2, 3], True), ("result",)]
        assert decode_mask(response.json()["result"]).sum() == 1

    @pytest.mark.parametrize("kind", ["scribble", "lasso"])
    def test_image_interaction(self, api_client, sync_session, kind):
        """Scribble and lasso interactions call the sync wrapper."""
        response = post_image_interaction(api_client, kind, "sync-session", False)

        assert response.status_code == 200, response.text
        assert sync_session.calls == [(kind, False), ("result",)]

    def test_reset(self, api_client, sync_session):
        """Reset calls the wrapper's reset_interactions."""
        response = api_client.get("/v2/reset_interactions/sync-session")

        assert response.json() == {"status": "success"}
        assert sync_session.calls == [("reset",)]


@pytest.mark.integration
class TestInvalidSession:
    """Every session endpoint reports an unknown session id."""

    @pytest.mark.parametrize(
        ("method", "url", "kwargs"),
        [
            ("get", "/v2/process_point_interaction/nope", {"params": {"point": [0]}}),
            ("get", "/process_point_interaction/nope", {"params": {"x": 0, "y": 0, "z": 0}}),
            ("get", "/v2/reset_interactions/nope", {}),
        ],
        ids=["point", "legacy-point", "reset"],
    )
    def test_get_endpoints(self, api_client, method, url, kwargs):
        """GET endpoints return an invalid-session error."""
        response = getattr(api_client, method)(url, **kwargs)
        assert response.json() == {"error": "Invalid session"}

    @pytest.mark.parametrize("kind", ["scribble", "lasso"])
    def test_image_interaction_endpoints(self, api_client, kind):
        """Scribble and lasso endpoints return an invalid-session error."""
        response = post_image_interaction(api_client, kind, "nope")
        assert response.json() == {"error": "Invalid session"}
