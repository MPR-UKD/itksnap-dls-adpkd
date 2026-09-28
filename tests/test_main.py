"""Tests for the server command-line options."""

import sys

import pytest

from itksnap_dls.__main__ import get_args

pytestmark = pytest.mark.unit


@pytest.fixture
def run_cli(monkeypatch):
    """Parse the given command-line arguments with a clean thread env var."""
    monkeypatch.delenv("ITKSNAP_DLS_CPU_THREADS", raising=False)

    def parse(*argv):
        monkeypatch.setattr(sys, "argv", ["itksnap_dls", *argv])
        return get_args()

    return parse


class TestCpuThreads:
    """Tests for the --cpu-threads option."""

    def test_default_is_two(self, run_cli):
        """Without flag or env var, nnInteractive uses 2 CPU threads."""
        assert run_cli().cpu_threads == 2

    def test_env_var_sets_default(self, run_cli, monkeypatch):
        """ITKSNAP_DLS_CPU_THREADS sets the default (used by the Docker image)."""
        monkeypatch.setenv("ITKSNAP_DLS_CPU_THREADS", "8")
        assert run_cli().cpu_threads == 8

    def test_flag_overrides_env_var(self, run_cli, monkeypatch):
        """--cpu-threads takes precedence over the env var."""
        monkeypatch.setenv("ITKSNAP_DLS_CPU_THREADS", "8")
        assert run_cli("--cpu-threads", "4").cpu_threads == 4

    @pytest.mark.parametrize("value", ["0", "-1", "abc"])
    def test_invalid_value_exits(self, run_cli, value):
        """Values below 1 or non-integers are rejected."""
        with pytest.raises(SystemExit):
            run_cli("--cpu-threads", value)
