"""Async HTTP client for the adpkd-net segmentation service."""

from __future__ import annotations

from typing import Any

import httpx


class AdpkdClient:
    """Thin async client for the adpkd-net job API."""

    def __init__(self, base_url: str, timeout: float = 10.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def health(self) -> bool:
        """Return whether the adpkd-net service reports healthy."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.get(f"{self._base_url}/health")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def submit_job(
        self, input_path: str, small: bool = False, cpu: bool = False
    ) -> dict[str, Any]:
        """Submit a segmentation job and return the created job."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.post(
                f"{self._base_url}/jobs",
                json={"input_path": input_path, "small": small, "cpu": cpu},
            )
            response.raise_for_status()
            return response.json()

    async def get_job(self, job_id: str) -> dict[str, Any]:
        """Return the current status of a job."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self._base_url}/jobs/{job_id}")
            response.raise_for_status()
            return response.json()

    async def get_result(self, job_id: str) -> dict[str, Any]:
        """Return the parsed results for a succeeded job."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self._base_url}/jobs/{job_id}/result")
            response.raise_for_status()
            return response.json()

    async def get_status(self) -> dict[str, Any]:
        """Return GPU availability and per-task trained-model readiness."""
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(f"{self._base_url}/status")
            response.raise_for_status()
            return response.json()
