"""Tests for the ADPKD segmentation pipeline (submit, poll, read result)."""

import httpx
import numpy as np
import pytest
import SimpleITK as sitk

from itksnap_dls.adpkd_seg import adpkd_segmentation
from itksnap_dls.settings import get_plugin_settings

from .test_toolbox import IMAGE_ORIGIN, IMAGE_SPACING, make_label_array

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class TestAdpkdSegmentationResult:
    """Tests for the returned segmentation image."""

    async def test_returns_labels_from_container(self, fake_client, sample_image):
        """The label values written by the container are returned unchanged."""
        seg = await adpkd_segmentation(sample_image)
        np.testing.assert_array_equal(sitk.GetArrayFromImage(seg), make_label_array())

    async def test_result_is_uint8(self, fake_client, sample_image):
        """The result is cast to UInt8 regardless of the container's pixel type."""
        seg = await adpkd_segmentation(sample_image)
        assert seg.GetPixelID() == sitk.sitkUInt8

    async def test_result_has_input_geometry(self, fake_client, sample_image):
        """Spacing and origin come from the input, not from the container output."""
        seg = await adpkd_segmentation(sample_image)
        np.testing.assert_allclose(seg.GetSpacing(), IMAGE_SPACING)
        np.testing.assert_allclose(seg.GetOrigin(), IMAGE_ORIGIN)


class TestAdpkdSegmentationJob:
    """Tests for how the job is submitted and polled."""

    async def test_client_uses_configured_url(self, fake_client, sample_image, mocker):
        """The HTTP client is created with the URL from the settings."""
        client_cls = mocker.patch(
            "itksnap_dls.adpkd_seg.AdpkdClient", return_value=fake_client
        )
        await adpkd_segmentation(sample_image)
        client_cls.assert_called_once_with("http://adpkd-test:9000")

    async def test_input_written_to_shared_dir(
        self, fake_client, sample_image, adpkd_settings
    ):
        """The input exists in the shared input dir when the job is submitted."""
        await adpkd_segmentation(sample_image)

        submitted = fake_client.submitted[0]
        assert submitted["input_path"].parent == adpkd_settings.adpkd_shared_input_dir
        assert submitted["input_path"].name.endswith(".nii.gz")
        assert submitted["input_exists"], "input was not written before submitting"

    async def test_missing_input_dir_is_created(
        self, fake_client, sample_image, tmp_path, monkeypatch
    ):
        """A nested shared input dir that does not exist yet is created."""
        monkeypatch.setenv("ADPKD_SHARED_INPUT_DIR", str(tmp_path / "a" / "b"))
        get_plugin_settings.cache_clear()

        await adpkd_segmentation(sample_image)

        assert fake_client.submitted[0]["input_exists"]

    async def test_each_run_uses_unique_input(self, fake_client, sample_image):
        """Concurrent-safe: two runs never share an input file name."""
        await adpkd_segmentation(sample_image)
        await adpkd_segmentation(sample_image)

        paths = [job["input_path"] for job in fake_client.submitted]
        assert paths[0] != paths[1]

    @pytest.mark.parametrize(
        ("small", "cpu"), [(False, False), (True, False), (False, True), (True, True)]
    )
    async def test_run_flags_from_settings(
        self, fake_client, sample_image, monkeypatch, small, cpu
    ):
        """The small and cpu flags are taken from the settings."""
        monkeypatch.setenv("ADPKD_RUN_SMALL", str(small))
        monkeypatch.setenv("ADPKD_RUN_CPU", str(cpu))
        get_plugin_settings.cache_clear()

        await adpkd_segmentation(sample_image)

        assert fake_client.submitted[0]["small"] is small
        assert fake_client.submitted[0]["cpu"] is cpu

    async def test_polls_until_succeeded(self, fake_client, sample_image):
        """Queued/running statuses are polled until the job succeeds."""
        fake_client.statuses = ["queued", "running", "running", "succeeded"]

        await adpkd_segmentation(sample_image)

        assert fake_client.polls == 4

    async def test_sleeps_poll_interval_between_polls(
        self, fake_client, sample_image, monkeypatch, mocker
    ):
        """The configured poll interval is awaited between status checks."""
        monkeypatch.setenv("ADPKD_POLL_INTERVAL", "1.5")
        get_plugin_settings.cache_clear()
        sleep = mocker.patch("itksnap_dls.adpkd_seg.asyncio.sleep")
        fake_client.statuses = ["running", "running", "succeeded"]

        await adpkd_segmentation(sample_image)

        assert sleep.await_count == 2
        sleep.assert_awaited_with(1.5)


class TestAdpkdSegmentationErrors:
    """Tests for failure handling."""

    async def test_failed_job_raises(self, fake_client, sample_image):
        """A job reported as failed raises instead of returning a result."""
        fake_client.statuses = ["running", "failed"]
        with pytest.raises(RuntimeError, match="failed"):
            await adpkd_segmentation(sample_image)

    async def test_timeout_raises(self, fake_client, sample_image, monkeypatch):
        """A job that never finishes raises once the timeout is exceeded."""
        monkeypatch.setenv("ADPKD_TIMEOUT", "0")
        get_plugin_settings.cache_clear()
        fake_client.statuses = ["running"]

        with pytest.raises(RuntimeError, match="timed out"):
            await adpkd_segmentation(sample_image)

    async def test_missing_result_raises(self, fake_client, sample_image):
        """A succeeded job without seg.nii.gz raises."""
        fake_client.labels = None
        with pytest.raises(RuntimeError, match="not found"):
            await adpkd_segmentation(sample_image)

    async def test_size_mismatch_raises(self, fake_client, sample_image):
        """A result whose size differs from the input raises."""
        fake_client.labels = np.zeros((2, 2, 2), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="does not match"):
            await adpkd_segmentation(sample_image)

    async def test_http_error_propagates(self, fake_client, sample_image, mocker):
        """Connection errors from the service are not swallowed."""
        mocker.patch.object(
            fake_client, "submit_job", side_effect=httpx.ConnectError("refused")
        )
        with pytest.raises(httpx.ConnectError):
            await adpkd_segmentation(sample_image)


class TestAdpkdSegmentationCleanup:
    """Tests that the shared input dir is cleaned up."""

    async def test_input_removed_after_success(
        self, fake_client, sample_image, adpkd_settings
    ):
        """The input file is deleted after a successful run."""
        await adpkd_segmentation(sample_image)
        assert list(adpkd_settings.adpkd_shared_input_dir.iterdir()) == []

    @pytest.mark.parametrize(
        ("statuses", "labels"),
        [
            (["failed"], make_label_array()),
            (["succeeded"], None),
            (["succeeded"], np.zeros((2, 2, 2), dtype=np.uint8)),
        ],
        ids=["job-failed", "missing-result", "size-mismatch"],
    )
    async def test_input_removed_after_failure(
        self, fake_client, sample_image, adpkd_settings, statuses, labels
    ):
        """The input file is deleted even when the run raises."""
        fake_client.statuses = statuses
        fake_client.labels = labels

        with pytest.raises(RuntimeError):
            await adpkd_segmentation(sample_image)

        assert list(adpkd_settings.adpkd_shared_input_dir.iterdir()) == []
