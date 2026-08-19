import subprocess

import pytest

from darktable_vlm_tagger.image_source import load_bytes_from_file


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
