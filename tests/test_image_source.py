import os
import subprocess

import pytest

from darktable_vlm_tagger import image_source
from darktable_vlm_tagger.image_source import (
    DarktableNotFoundError,
    _require_darktable_bin,
    _resolve_darktable_bin,
    load_bytes_from_file,
)


def _magick_available():
    return subprocess.run(["magick", "-version"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _magick_available(), reason="ImageMagick not installed")
def test_load_bytes_from_file_downscales_to_long_edge(tmp_path):
    # Simulates darktable's own Lua-side export handed straight to the tool,
    # bypassing the mipmap cache/render fallback entirely - see the
    # docstring on load_bytes_from_file for why the live-integration path
    # needs this instead of image_source.load_image_bytes.
    src = tmp_path / "exported.jpg"
    subprocess.run(["magick", "-size", "2000x1000", "xc:red", str(src)],
                    check=True, capture_output=True)

    result = load_bytes_from_file(src, long_edge=500)

    assert isinstance(result, bytes)
    assert result.startswith(b"\xff\xd8")  # JPEG magic bytes

    dims = subprocess.run(["identify", "-format", "%wx%h", "-"], input=result,
                           check=True, capture_output=True).stdout.decode()
    assert dims == "500x250"


@pytest.fixture(autouse=True)
def _clear_bin_cache():
    _resolve_darktable_bin.cache_clear()
    yield
    _resolve_darktable_bin.cache_clear()


def test_resolve_prefers_path(monkeypatch, tmp_path):
    fake = tmp_path / "darktable-cli"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setattr(image_source.shutil, "which",
                        lambda name: str(fake) if name == "darktable-cli" else None)
    monkeypatch.delenv("DARKTABLE_BIN_DIR", raising=False)
    assert _resolve_darktable_bin("darktable-cli") == str(fake)


def test_resolve_falls_back_to_bundle_dir(monkeypatch, tmp_path):
    bundle = tmp_path / "darktable.app/Contents/MacOS"
    bundle.mkdir(parents=True)
    binary = bundle / "darktable-generate-cache"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(image_source.shutil, "which", lambda name: None)
    monkeypatch.setattr(image_source, "_BUNDLE_BIN_DIRS", [bundle])
    monkeypatch.delenv("DARKTABLE_BIN_DIR", raising=False)
    assert _resolve_darktable_bin("darktable-generate-cache") == str(binary)


def test_resolve_honours_env_override(monkeypatch, tmp_path):
    binary = tmp_path / "darktable-cli"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    monkeypatch.setattr(image_source.shutil, "which", lambda name: "/usr/bin/darktable-cli")
    monkeypatch.setenv("DARKTABLE_BIN_DIR", str(tmp_path))
    assert _resolve_darktable_bin("darktable-cli") == str(binary)


def test_require_raises_clear_error_when_missing(monkeypatch):
    monkeypatch.setattr(image_source.shutil, "which", lambda name: None)
    monkeypatch.setattr(image_source, "_BUNDLE_BIN_DIRS", [])
    monkeypatch.delenv("DARKTABLE_BIN_DIR", raising=False)
    with pytest.raises(DarktableNotFoundError, match="darktable-cli"):
        _require_darktable_bin("darktable-cli")
