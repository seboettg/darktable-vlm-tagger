"""Turn a darktable image record into JPEG bytes ready to send to the model.

Priority, cheapest first:

1. darktable's own mipmap cache (~/.cache/darktable/mipmaps-<hash>.d/<level>/
   <image_id>.jpg). Verified against a live library: level 4 alone covers
   ~99% of a 16k-image library, and the file is keyed by darktable's own
   image id, which is exactly `images.id` in library.db. Its presence alone
   is trusted - no staleness check against the image's last edit.
   `images.write_timestamp` looked like an obvious staleness signal but
   turned out not to be one: checked against a live library, it trails the
   mipmap file's mtime by hours to weeks on effectively every image (0 fresh
   out of 1996 sampled), so it clearly advances for reasons unrelated to
   actual edits (e.g. darktable's own housekeeping) and would make this
   check reject almost every cached thumbnail, defeating the entire point of
   reusing them. darktable invalidates its own mipmap cache when history
   actually changes; this tool relies on that rather than re-deriving it
   from an unreliable column.
2. If missing entirely: ask darktable itself to generate it
   (`darktable-generate-cache`, the official tool for exactly this), then
   look again.
3. If the file was never imported into the library, or generation still
   fails: render directly (darktable-cli for RAW/edited files, ImageMagick
   for a plain JPEG/PNG/TIFF with no sidecar - same sidecar rule as the
   original proxy-rendering script).

Every path ends the same way: downscale to a configurable long edge
(default 1024, matching the resolution the model/prompt were evaluated at)
with ImageMagick before returning the bytes. darktable keeps the long edge
fixed per mipmap level regardless of aspect ratio (verified: level 4 is
800x1200 for 3:2 and 900x1200 for 4:3 sources), so this step also
normalises 4:3 (e.g. Micro Four Thirds) sources down to the same budget.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from functools import lru_cache
from pathlib import Path

from .darktable_db import ImageRecord

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "darktable"
RENDERABLE_WITHOUT_DARKTABLE = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# darktable's CLI helpers are not on $PATH when it is installed as a
# self-contained app bundle (macOS) or Windows install - only the GUI is
# launched via Finder/Start menu. Look inside the usual bundle locations
# before giving up. Override with DARKTABLE_BIN_DIR for anything unusual.
_BUNDLE_BIN_DIRS = [
    Path("/Applications/darktable.app/Contents/MacOS"),
    Path.home() / "Applications/darktable.app/Contents/MacOS",
    Path("C:/Program Files/darktable/bin"),
]


class DarktableRunningError(RuntimeError):
    """Raised when a darktable-cli render is attempted while darktable itself
    is open - it must be closed, or gmic-compressed LUTs silently fail to
    apply and the render looks plausible but wrong."""


class DarktableNotFoundError(RuntimeError):
    """Raised when a darktable CLI helper (darktable-cli /
    darktable-generate-cache) cannot be located on $PATH or in a known app
    bundle. Install darktable, or point DARKTABLE_BIN_DIR at the directory
    holding its command-line tools."""


@lru_cache(maxsize=None)
def _resolve_darktable_bin(name: str) -> str | None:
    """Absolute path to a darktable CLI helper, or None if not found.

    Order: $DARKTABLE_BIN_DIR, then $PATH, then the known app-bundle dirs.
    """
    override = os.environ.get("DARKTABLE_BIN_DIR")
    search_dirs = [Path(override)] if override else []
    on_path = shutil.which(name)
    if on_path and not override:
        return on_path
    for d in search_dirs + _BUNDLE_BIN_DIRS:
        for candidate in (d / name, d / f"{name}.exe"):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return on_path  # honours $PATH even when an override dir was set but empty


def _require_darktable_bin(name: str) -> str:
    resolved = _resolve_darktable_bin(name)
    if resolved is None:
        hint = ""
        if sys.platform == "darwin":
            hint = (" On macOS the tools ship inside darktable.app "
                    "(Contents/MacOS) and are not on $PATH by default.")
        raise DarktableNotFoundError(
            f"{name!r} not found on $PATH or in a known darktable install."
            f"{hint} Install darktable or set DARKTABLE_BIN_DIR to the "
            "directory containing its command-line tools."
        )
    return resolved


def darktable_is_running() -> bool:
    return subprocess.run(["pgrep", "-x", "darktable"], capture_output=True).returncode == 0


def _find_mip(cache_dir: Path, image_id: int, level: int) -> Path | None:
    matches = list(cache_dir.glob(f"mipmaps-*.d/{level}/{image_id}.jpg"))
    return matches[0] if matches else None


def _generate_cache(library_dir: Path, cache_dir: Path, image_id: int,
                     level: int, timeout: int = 120) -> None:
    dt_bin = _resolve_darktable_bin("darktable-generate-cache")
    if dt_bin is None:
        # Best-effort step: if darktable isn't installed here, skip straight
        # to the render fallback, which raises a clear error itself when it
        # actually needs darktable (a RAW/edited file).
        return
    subprocess.run(
        [
            dt_bin,
            "--min-mip", str(level), "--max-mip", str(level),
            "--min-imgid", str(image_id), "--max-imgid", str(image_id),
            "--core",
            "--configdir", str(library_dir),
            "--cachedir", str(cache_dir),
        ],
        capture_output=True, timeout=timeout, check=False,
    )


def _resize_with_imagemagick(src: Path, long_edge: int) -> bytes:
    result = subprocess.run(
        ["magick", str(src), "-auto-orient",
         "-resize", f"{long_edge}x{long_edge}>",
         "-quality", "88", "-strip", "jpg:-"],
        capture_output=True, check=True,
    )
    return result.stdout


def load_bytes_from_file(source_file: Path, *, long_edge: int) -> bytes:
    """Bytes for an already-rendered image file, downscaled the same way as
    every other source path.

    Used by the Lua/UI integration: while darktable is running, no external
    process can safely re-render an image itself (see
    `_render_with_darktable_cli`'s guard), but darktable's own live process
    can - it exports via its own Lua API and passes the result here,
    bypassing the mipmap-cache/render dance entirely.
    """
    return _resize_with_imagemagick(source_file, long_edge)


def _render_with_darktable_cli(path: Path, sidecar: Path, long_edge: int,
                                library_dir: Path, cache_dir: Path,
                                timeout: int = 120) -> bytes:
    if darktable_is_running():
        raise DarktableRunningError(
            "darktable is running - close it before rendering, or film "
            "simulation LUTs silently fail to apply and the render looks "
            "plausible but is wrong."
        )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "proxy.jpg"
        cmd = [_require_darktable_bin("darktable-cli"), str(path)]
        if sidecar.exists():
            cmd.append(str(sidecar))
        cmd += [
            str(out),
            "--width", str(long_edge),
            "--height", str(long_edge),
            "--hq", "true",
            "--apply-custom-presets", "false",
            "--core",
            "--configdir", str(library_dir),
            "--cachedir", str(cache_dir),
            "--library", ":memory:",
            "--conf", "plugins/imageio/format/jpeg/quality=88",
        ]
        subprocess.run(cmd, capture_output=True, timeout=timeout, check=True)
        return out.read_bytes()


def _render_fallback(path: Path, sidecar: Path, long_edge: int,
                      library_dir: Path, cache_dir: Path) -> bytes:
    if path.suffix.lower() in RENDERABLE_WITHOUT_DARKTABLE and not sidecar.exists():
        return _resize_with_imagemagick(path, long_edge)
    return _render_with_darktable_cli(path, sidecar, long_edge, library_dir, cache_dir)


def load_image_bytes(record: ImageRecord, *, library_dir: Path, long_edge: int,
                      mip_level: int, cache_dir: Path = DEFAULT_CACHE_DIR) -> bytes:
    """Return JPEG bytes for `record`, downscaled to `long_edge`.

    For a duplicate/version (record.version > 0), the mipmap cache is keyed
    by image id and already reflects that specific version's development, so
    the mip path needs no extra handling; the fallback render path does,
    since darktable-cli must be pointed at that version's own sidecar
    (record.sidecar_path) rather than the physical file's plain `.xmp`.
    """
    if record.id is not None:
        mip = _find_mip(cache_dir, record.id, mip_level)
        if mip is None:
            _generate_cache(library_dir, cache_dir, record.id, mip_level)
            mip = _find_mip(cache_dir, record.id, mip_level)
        if mip is not None:
            return _resize_with_imagemagick(mip, long_edge)

    return _render_fallback(record.path, record.sidecar_path, long_edge,
                             library_dir, cache_dir)
