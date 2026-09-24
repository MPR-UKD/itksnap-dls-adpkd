"""Tests for the adpkd-net HTTP client."""

import json

import httpx
import pytest

from itksnap_dls.adpkd_client import AdpkdClient

pytestmark = [pytest.mark.unit, pytest.mark.anyio]


class ServiceStub:
    """Records requests and replies with canned responses keyed by (method, path)."""

    def __init__(self):
        self.requests: list[httpx.Request] = []
        self.responses: dict[tuple[str, str], httpx.Response] = {}
        self.client_kwargs: list[dict] = []
        self.error: Exception | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return self.responses.get(
            (request.method, request.url.path), httpx.Response(404)
        )


@pytest.fixture
def service(mocker):
    """Route every ``httpx.AsyncClient`` created by the client to a stub service."""
    stub = ServiceStub()
    real_async_client = httpx.AsyncClient

    def client_factory(**kwargs):
        stub.client_kwargs.append(kwargs)
        return real_async_client(transport=httpx.MockTransport(stub.handler), **kwargs)

    mocker.patch(
        "itksnap_dls.adpkd_client.httpx.AsyncClient", side_effect=client_factory
    )
    return stub


class TestAdpkdClientHealth:
    """Tests for the health check."""

    async def test_health_returns_true_on_200(self, service):
        """A 200 response from /health means the service is healthy."""
        service.responses[("GET", "/health")] = httpx.Response(200)
        assert await AdpkdClient("http://adpkd:9000").health() is True

    async def test_health_returns_false_on_error_status(self, service):
        """A non-200 response from /health means the service is unhealthy."""
        service.responses[("GET", "/health")] = httpx.Response(503)
        assert await AdpkdClient("http://adpkd:9000").health() is False

    async def test_health_returns_false_when_unreachable(self, service):
        """A connection error is reported as unhealthy instead of raising."""
        service.error = httpx.ConnectError("connection refused")
        assert await AdpkdClient("http://adpkd:9000").health() is False


class TestAdpkdClientJobs:
    """Tests for the job endpoints."""

    async def test_submit_job_posts_payload(self, service):
        """submit_job POSTs the input path and flags as JSON and returns the job."""
        service.responses[("POST", "/jobs")] = httpx.Response(
            201, json={"job_id": "abc", "status": "queued"}
        )

        job = await AdpkdClient("http://adpkd:9000").submit_job(
            "/data/in.nii.gz", small=True, cpu=True
        )

        assert job == {"job_id": "abc", "status": "queued"}
        request = service.requests[0]
        assert str(request.url) == "http://adpkd:9000/jobs"
        assert json.loads(request.content) == {
            "input_path": "/data/in.nii.gz",
            "small": True,
            "cpu": True,
        }

    async def test_submit_job_defaults_flags_to_false(self, service):
        """submit_job sends small=False and cpu=False unless requested."""
        service.responses[("POST", "/jobs")] = httpx.Response(201, json={"job_id": "a"})

        await AdpkdClient("http://adpkd:9000").submit_job("/data/in.nii.gz")

        payload = json.loads(service.requests[0].content)
        assert payload["small"] is False and payload["cpu"] is False

    @pytest.mark.parametrize(
        ("method_name", "path"),
        [
            ("get_job", "/jobs/abc"),
            ("get_result", "/jobs/abc/result"),
        ],
    )
    async def test_job_getters_call_expected_path(self, service, method_name, path):
        """Job getters GET the job-specific path and return the parsed JSON."""
        service.responses[("GET", path)] = httpx.Response(200, json={"ok": 1})

        result = await getattr(AdpkdClient("http://adpkd:9000"), method_name)("abc")

        assert result == {"ok": 1}
        assert service.requests[0].url.path == path

    async def test_get_status_calls_status_endpoint(self, service):
        """get_status GETs /status and returns the parsed JSON."""
        service.responses[("GET", "/status")] = httpx.Response(200, json={"gpu": True})
        assert await AdpkdClient("http://adpkd:9000").get_status() == {"gpu": True}

    @pytest.mark.parametrize(
        ("method_name", "args"),
        [
            ("submit_job", ("/data/in.nii.gz",)),
            ("get_job", ("abc",)),
            ("get_result", ("abc",)),
            ("get_status", ()),
        ],
    )
    async def test_error_status_raises(self, service, method_name, args):
        """Every job call raises HTTPStatusError on an error response."""
        with pytest.raises(httpx.HTTPStatusError):
            await getattr(AdpkdClient("http://adpkd:9000"), method_name)(*args)


class TestAdpkdClientConfiguration:
    """Tests for base URL and timeout handling."""

    async def test_trailing_slash_is_stripped(self, service):
        """A trailing slash in the base URL does not produce a double slash."""
        service.responses[("GET", "/jobs/abc")] = httpx.Response(200, json={})

        await AdpkdClient("http://adpkd:9000/").get_job("abc")

        assert str(service.requests[0].url) == "http://adpkd:9000/jobs/abc"

    async def test_timeout_is_passed_to_http_client(self, service):
        """The configured timeout is used for the underlying HTTP client."""
        service.responses[("GET", "/status")] = httpx.Response(200, json={})

        await AdpkdClient("http://adpkd:9000", timeout=3.5).get_status()

        assert service.client_kwargs[0]["timeout"] == 3.5
