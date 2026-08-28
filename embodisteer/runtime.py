"""Runtime metadata and safe path helpers for public launchers."""

from __future__ import annotations

from pathlib import Path


def repository_root() -> Path:
    """Return the checkout root without relying on a developer's home path."""

    # ``embodisteer/runtime.py`` is two levels below the checkout root.
    return Path(__file__).resolve().parents[1]


def third_party_manifest_path() -> Path:
    """Return the version manifest used by bootstrap and diagnostics."""

    return repository_root() / "third_party" / "manifest.yaml"


__all__ = ["repository_root", "third_party_manifest_path"]
