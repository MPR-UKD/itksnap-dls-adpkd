"""Shared helpers for the test suite (not a test module)."""

import base64
import gzip
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

# Array shape in numpy (z, y, x) order; the ITK size is the reverse
IMAGE_SHAPE_ZYX = (4, 5, 6)
IMAGE_SPACING = (0.8, 0.8, 3.0)
IMAGE_ORIGIN = (-10.0, 20.0, 5.0)

# Row-major LPS direction of a coronal acquisition: the third voxel axis (slices)
# runs along anterior-posterior and the second along inferior-superior
CORONAL_DIRECTION = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0)


def make_image(
    shape_zyx=IMAGE_SHAPE_ZYX, spacing=IMAGE_SPACING, origin=IMAGE_ORIGIN
) -> sitk.Image:
    """Return a float32 image with deterministic content and non-default geometry."""
    array = np.arange(np.prod(shape_zyx), dtype=np.float32).reshape(shape_zyx)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing(spacing)
    image.SetOrigin(origin)
    return image


def make_label_array(shape_zyx=IMAGE_SHAPE_ZYX) -> np.ndarray:
    """Return a label array with two kidneys (labels 1 and 2), 8 voxels each."""
    labels = np.zeros(shape_zyx, dtype=np.uint8)
    labels[1:3, 1:3, 0:2] = 1
    labels[1:3, 1:3, 3:5] = 2
    return labels


def write_segmentation(path: Path, labels: np.ndarray) -> None:
    """Write a label array as NIfTI with geometry that differs from the input image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seg = sitk.GetImageFromArray(labels.astype(np.int16))
    seg.SetSpacing((9.0, 9.0, 9.0))
    sitk.WriteImage(seg, str(path))


def encode_upload(
    array: np.ndarray,
    components_per_pixel: int = 1,
    dtype=np.float32,
    spacing=None,
    origin=None,
    direction=None,
    component_type=None,
):
    """Encode an array like the ITK-SNAP client: gzipped pixels plus JSON metadata.

    Geometry keys and ``component_type`` are only added when given, so the default
    call produces the minimal metadata of older clients.
    """
    payload = gzip.compress(array.astype(dtype).tobytes())
    spatial_shape = array.shape if components_per_pixel == 1 else array.shape[:-1]
    metadata = {
        "dimensions": list(spatial_shape[::-1]),
        "components_per_pixel": components_per_pixel,
    }
    optional = {
        "spacing": spacing,
        "origin": origin,
        "direction": direction,
        "component_type": component_type,
    }
    metadata.update(
        {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in optional.items()
            if value is not None
        }
    )
    return payload, json.dumps(metadata)


def decode_mask(result_b64: str, shape_zyx=IMAGE_SHAPE_ZYX) -> np.ndarray:
    """Decode a server interaction result into a (z, y, x) int8 mask."""
    raw = gzip.decompress(base64.b64decode(result_b64))
    return np.frombuffer(raw, dtype=np.int8).reshape(shape_zyx)


class FakeAdpkdClient:
    """In-memory stand-in for :class:`AdpkdClient` that emulates the container.

    Each ``get_job`` call returns the next entry of ``statuses`` (the last one
    repeats). When it reports ``succeeded``, ``labels`` is written to
    ``<output_dir>/<job_id>/seg.nii.gz`` like the real container.
    """

    def __init__(self, output_dir: Path, labels=None, statuses=("succeeded",)):
        self.healthy = True
        self.output_dir = output_dir
        self.labels = make_label_array() if labels is None else labels
        self.statuses = list(statuses)
        self.submitted: list[dict] = []
        self.polls = 0

    @property
    def job_id(self) -> str:
        return f"job-{len(self.submitted)}"

    async def health(self):
        return self.healthy

    async def submit_job(self, input_path, small=False, cpu=False):
        # Record the geometry now: the input file is deleted after the run
        input_exists = Path(input_path).is_file()
        input_image = sitk.ReadImage(str(input_path)) if input_exists else None
        self.submitted.append(
            {
                "input_path": Path(input_path),
                "input_exists": input_exists,
                "input_spacing": input_image.GetSpacing() if input_image else None,
                "input_origin": input_image.GetOrigin() if input_image else None,
                "input_direction": input_image.GetDirection() if input_image else None,
                "small": small,
                "cpu": cpu,
            }
        )
        return {"job_id": self.job_id, "status": "queued"}

    async def get_job(self, job_id):
        status = self.statuses[min(self.polls, len(self.statuses) - 1)]
        self.polls += 1
        if status == "succeeded" and self.labels is not None:
            write_segmentation(self.output_dir / job_id / "seg.nii.gz", self.labels)
        return {"job_id": job_id, "status": status}
