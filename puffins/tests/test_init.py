"""Tests for package-level attributes of puffins."""

import importlib
import importlib.metadata

import pytest

import puffins


class TestVersion:
    """Tests for puffins.__version__."""

    def test_matches_installed_metadata(self) -> None:
        """__version__ is the version recorded in the installed metadata."""
        assert puffins.__version__ == importlib.metadata.version("puffins")

    def test_fallback_when_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without installed metadata, __version__ is the PEP 440 placeholder."""

        def not_installed(name: str) -> str:
            raise importlib.metadata.PackageNotFoundError(name)

        monkeypatch.setattr(importlib.metadata, "version", not_installed)
        try:
            importlib.reload(puffins)
            assert puffins.__version__ == "0+unknown"
        finally:
            monkeypatch.undo()
            importlib.reload(puffins)
