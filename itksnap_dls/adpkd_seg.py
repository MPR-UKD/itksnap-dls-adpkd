import asyncio
import time
import uuid

import SimpleITK as sitk

from .adpkd_client import AdpkdClient
from .settings import get_plugin_settings


async def adpkd_segmentation(input_image: sitk.Image) -> sitk.Image:
    """Run the adpkd-net container on an image and return the segmentation.

    The input is written to the shared input directory (mounted in the container),
    a job is submitted and polled until it finishes, and the resulting
    ``seg.nii.gz`` is read from the shared output directory.

    Args:
        input_image: 3D image to segment.

    Returns:
        UInt8 label image with the same geometry as ``input_image``.

    Raises:
        RuntimeError: If the service is unreachable, or the job fails, times out
            or returns a mismatched result.
        httpx.HTTPError: If the service cannot be reached.
    """
    settings = get_plugin_settings()
    client = AdpkdClient(settings.adpkd_url)
    if not await client.health():
        raise RuntimeError(f"adpkd-net service not reachable at {settings.adpkd_url}")

    # 1. Write the image into the directory shared with the container
    run_id = str(uuid.uuid4())[:8]
    settings.adpkd_shared_input_dir.mkdir(parents=True, exist_ok=True)
    input_path = settings.adpkd_shared_input_dir / f"{run_id}.nii.gz"
    sitk.WriteImage(input_image, str(input_path))
    print(f"[ADPKD] Starting run {run_id}, input: {input_path}")

    try:
        # 2. Submit the job
        job = await client.submit_job(
            str(input_path),
            small=settings.adpkd_run_small,
            cpu=settings.adpkd_run_cpu,
        )
        job_id = job["job_id"]
        print(f"[ADPKD] Job submitted: {job_id}")

        # 3. Poll until the job is done
        start_time = time.time()
        while True:
            status = (await client.get_job(job_id))["status"]
            if status == "succeeded":
                break
            if status == "failed":
                raise RuntimeError(f"ADPKD job {job_id} failed")
            if time.time() - start_time > settings.adpkd_timeout:
                raise RuntimeError(
                    f"ADPKD job {job_id} timed out after {settings.adpkd_timeout} s"
                )
            await asyncio.sleep(settings.adpkd_poll_interval)
        print(f"[ADPKD] Job succeeded in {time.time() - start_time:.1f} seconds")

        # 4. Read the result and map it onto the input geometry
        seg_path = settings.adpkd_shared_output_dir / job_id / "seg.nii.gz"
        if not seg_path.is_file():
            raise RuntimeError(f"ADPKD result not found: {seg_path}")
        seg = sitk.ReadImage(str(seg_path))
        if seg.GetSize() != input_image.GetSize():
            raise RuntimeError(
                f"ADPKD result size {seg.GetSize()} does not match "
                f"input size {input_image.GetSize()}"
            )
        seg = sitk.Cast(seg, sitk.sitkUInt8)
        seg.CopyInformation(input_image)
        return seg

    finally:
        input_path.unlink(missing_ok=True)
