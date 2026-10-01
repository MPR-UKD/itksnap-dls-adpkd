"""Tests for ADPKDWrapper and its registration in the model registry."""

import asyncio

import numpy as np
import pytest
import SimpleITK as sitk

from itksnap_dls.segment import (
    ADPKDWrapper,
    get_model_listing,
    global_config,
    instantiate_model_wrapper,
)

from .test_toolbox import (
    IMAGE_SHAPE_ZYX,
    KIDNEY_INDEX,
    LIVER_INDEX,
    make_image,
    make_label_array,
)

INTERACTIONS = [
    ("add_point_interaction", ([1, 2, 3],)),
    ("add_scribble_interaction", (make_image(),)),
    ("add_lasso_interaction", (make_image(),)),
]
INTERACTION_IDS = ["point", "scribble", "lasso"]

# ITK (x, y, z) index outside both organs
BACKGROUND_INDEX = [5, 4, 3]


def drawing(zyx_slices) -> sitk.Image:
    """Return a scribble/lasso mask image that is 1 inside ``zyx_slices``."""
    array = np.zeros(IMAGE_SHAPE_ZYX, dtype=np.uint8)
    array[zyx_slices] = 1
    return sitk.GetImageFromArray(array)


def mask_array(image: sitk.Image) -> np.ndarray:
    """Return the (z, y, x) array of a result mask."""
    return sitk.GetArrayFromImage(image)


@pytest.fixture
def segmentation_result():
    """Image returned by the mocked ADPKD pipeline, with input-like geometry."""
    seg = sitk.GetImageFromArray(make_label_array())
    seg.CopyInformation(make_image())
    return seg


@pytest.fixture
def mock_segmentation(mocker, segmentation_result):
    """Replace the ADPKD pipeline in the wrapper with an AsyncMock."""
    return mocker.patch(
        "itksnap_dls.segment.adpkd_segmentation",
        new_callable=mocker.AsyncMock,
        return_value=segmentation_result,
    )


@pytest.mark.unit
class TestADPKDRegistration:
    """Tests for listing and instantiating the ADPKD model."""

    def test_listed_with_capabilities(self):
        """The model listing advertises ADPKD as a 3D single-channel model."""
        listing = {model["id"]: model for model in get_model_listing()}
        assert listing["ADPKD"] == {
            "id": "ADPKD",
            "channels": [1],
            "dimensions": 3,
            "interactions": ["point", "scribble", "lasso"],
        }

    def test_instantiate_by_id(self):
        """instantiate_model_wrapper builds an ADPKDWrapper with the given config."""
        wrapper = instantiate_model_wrapper("ADPKD")
        assert isinstance(wrapper, ADPKDWrapper)
        assert wrapper.config is global_config

    def test_instantiate_unknown_id_raises(self):
        """An unknown model ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            instantiate_model_wrapper("does-not-exist")


@pytest.mark.unit
class TestADPKDWrapperSetImage:
    """Tests for starting the segmentation on upload."""

    def test_set_image_requires_running_loop(self, mock_segmentation):
        """set_image outside an event loop raises instead of silently not running."""
        with pytest.raises(RuntimeError):
            ADPKDWrapper().set_image(make_image())
        mock_segmentation.assert_not_called()

    @pytest.mark.anyio
    async def test_set_image_starts_job_immediately(self, mock_segmentation):
        """The job starts on upload, before any interaction arrives."""
        image = make_image()
        ADPKDWrapper().set_image(image)
        await asyncio.sleep(0)

        mock_segmentation.assert_awaited_once_with(image)

    @pytest.mark.anyio
    async def test_new_image_cancels_previous_job(self, mocker):
        """Uploading a new image cancels the running job and uses the new one."""
        first_started = asyncio.Event()
        first_cancelled = asyncio.Event()
        second_result = sitk.Image(2, 2, 2, sitk.sitkUInt8)

        async def fake_segmentation(image):
            if image.GetSize() == (6, 5, 4):
                first_started.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    first_cancelled.set()
                    raise
            return second_result

        mocker.patch("itksnap_dls.segment.adpkd_segmentation", fake_segmentation)
        wrapper = ADPKDWrapper()

        wrapper.set_image(make_image())
        await first_started.wait()
        wrapper.set_image(make_image(shape_zyx=(2, 2, 2)))

        assert (await wrapper.get_result()).GetSize() == second_result.GetSize()
        assert first_cancelled.is_set()


@pytest.mark.unit
class TestADPKDWrapperClose:
    """Tests for stopping the background job when a session ends."""

    def test_close_without_image_is_noop(self):
        """close() on a wrapper that never received an image does nothing."""
        ADPKDWrapper().close()

    @pytest.mark.anyio
    async def test_close_cancels_running_job(self, mocker):
        """close() cancels a job that is still running."""
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def fake_segmentation(image):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        mocker.patch("itksnap_dls.segment.adpkd_segmentation", fake_segmentation)
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())
        await started.wait()

        wrapper.close()
        await asyncio.sleep(0)

        assert cancelled.is_set()

    @pytest.mark.anyio
    async def test_get_result_after_close_raises(self, mock_segmentation):
        """After close(), the wrapper behaves as if no image was uploaded."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())
        wrapper.close()

        with pytest.raises(RuntimeError, match="No image"):
            await wrapper.get_result()


@pytest.mark.unit
@pytest.mark.anyio
class TestADPKDWrapperResult:
    """Tests for interactions and result retrieval."""

    async def test_get_result_without_image_raises(self):
        """get_result before an upload raises a clear error."""
        with pytest.raises(RuntimeError, match="No image"):
            await ADPKDWrapper().get_result()

    @pytest.mark.parametrize(("method", "args"), INTERACTIONS, ids=INTERACTION_IDS)
    async def test_interaction_without_image_raises(self, method, args):
        """Interactions before an upload raise a clear error."""
        with pytest.raises(RuntimeError, match="No image"):
            await getattr(ADPKDWrapper(), method)(*args, include_interaction=True)

    async def test_get_result_without_selection_is_empty(
        self, mock_segmentation, segmentation_result
    ):
        """Before any interaction, the mask is empty but has the segmentation's geometry."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())

        result = await wrapper.get_result()

        assert not mask_array(result).any()
        assert result.GetSize() == segmentation_result.GetSize()
        assert result.GetSpacing() == segmentation_result.GetSpacing()
        assert result.GetOrigin() == segmentation_result.GetOrigin()

    @pytest.mark.parametrize(("method", "args"), INTERACTIONS, ids=INTERACTION_IDS)
    @pytest.mark.parametrize("include_interaction", [True, False])
    async def test_interactions_reuse_single_job(
        self, mock_segmentation, method, args, include_interaction
    ):
        """Repeated interactions of any kind wait for the one job started on upload."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())

        for _ in range(3):
            await getattr(wrapper, method)(*args, include_interaction=include_interaction)
            await wrapper.get_result()

        mock_segmentation.assert_awaited_once()

    async def test_reset_clears_selection_but_keeps_job(self, mock_segmentation):
        """reset_interactions starts a new selection without rerunning the job."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())
        await wrapper.add_point_interaction(KIDNEY_INDEX, include_interaction=True)

        wrapper.reset_interactions()

        assert not mask_array(await wrapper.get_result()).any()
        mock_segmentation.assert_awaited_once()

    async def test_job_failure_propagates(self, mocker):
        """A failed job surfaces as an error on the next interaction."""
        mocker.patch(
            "itksnap_dls.segment.adpkd_segmentation",
            new_callable=mocker.AsyncMock,
            side_effect=RuntimeError("ADPKD job job-1 failed"),
        )
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())

        with pytest.raises(RuntimeError, match="failed"):
            await wrapper.add_point_interaction([0, 0, 0], include_interaction=True)


@pytest.mark.unit
@pytest.mark.anyio
class TestADPKDWrapperSelection:
    """Tests for picking kidneys or liver into the active ITK-SNAP label."""

    @pytest.mark.parametrize(
        ("index", "label"),
        [(KIDNEY_INDEX, 1), (LIVER_INDEX, 2)],
        ids=["kidneys", "liver"],
    )
    async def test_point_selects_organ_under_click(
        self, mock_segmentation, index, label
    ):
        """A click on an organ returns exactly that organ's label as the mask."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())

        await wrapper.add_point_interaction(index, include_interaction=True)

        np.testing.assert_array_equal(
            mask_array(await wrapper.get_result()), make_label_array() == label
        )

    @pytest.mark.parametrize(
        "index",
        [BACKGROUND_INDEX, [6, 0, 0], [-1, 0, 0]],
        ids=["background", "beyond", "negative"],
    )
    async def test_point_off_organ_leaves_selection(self, mock_segmentation, index):
        """Clicks on background or outside the image select nothing."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())

        await wrapper.add_point_interaction(index, include_interaction=True)

        assert not mask_array(await wrapper.get_result()).any()

    async def test_points_combine_and_right_click_removes(self, mock_segmentation):
        """Left clicks add organs to the mask; a right click removes one again."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())

        await wrapper.add_point_interaction(KIDNEY_INDEX, include_interaction=True)
        await wrapper.add_point_interaction(LIVER_INDEX, include_interaction=True)
        np.testing.assert_array_equal(
            mask_array(await wrapper.get_result()), make_label_array() > 0
        )

        await wrapper.add_point_interaction(KIDNEY_INDEX, include_interaction=False)
        np.testing.assert_array_equal(
            mask_array(await wrapper.get_result()), make_label_array() == 2
        )

    @pytest.mark.parametrize(
        "method", ["add_scribble_interaction", "add_lasso_interaction"]
    )
    async def test_drawing_selects_majority_organ(self, mock_segmentation, method):
        """A drawing selects the organ it covers most, ignoring background voxels."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())
        # One kidney voxel, two liver voxels and some background
        drawn = drawing((1, 1, slice(1, 6)))

        await getattr(wrapper, method)(drawn, include_interaction=True)

        np.testing.assert_array_equal(
            mask_array(await wrapper.get_result()), make_label_array() == 2
        )

    async def test_drawing_off_organs_leaves_selection(self, mock_segmentation):
        """A drawing that only covers background selects nothing."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())

        await wrapper.add_scribble_interaction(
            drawing((3, 4, slice(None))), include_interaction=True
        )

        assert not mask_array(await wrapper.get_result()).any()

    async def test_new_image_clears_selection(self, mock_segmentation):
        """Uploading a new image starts with an empty selection."""
        wrapper = ADPKDWrapper()
        wrapper.set_image(make_image())
        await wrapper.add_point_interaction(KIDNEY_INDEX, include_interaction=True)

        wrapper.set_image(make_image())

        assert not mask_array(await wrapper.get_result()).any()
