"""Read-only access to darktable's library.db.

This module never writes to the database - resolving a folder or file to a
darktable image id is the only thing it is used for. All actual tagging
output goes through sidecar.py instead.
"""

import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

DEFAULT_LIBRARY_DIR = Path.home() / ".config" / "darktable"


@dataclass(frozen=True)
class ImageRecord:
    id: int | None  # darktable image id; None if the file was never imported
    path: Path  # absolute path to the RAW/JPEG file
    version: int  # darktable duplicate/version number, 0 for the base image
    write_timestamp: int | None  # unix epoch seconds, informational only

    @property
    def sidecar_path(self) -> Path:
        """darktable's on-disk sidecar naming: plain `<file>.xmp` for version
        0, but `<stem>_<NN><suffix>.xmp` for duplicates - e.g. version 1 of
        `photo.RAF` is `photo_01.RAF.xmp`, sharing the same physical RAW.
        Getting this wrong either overwrites the wrong duplicate's tags or
        renders with the wrong duplicate's development history."""
        if self.version == 0:
            return Path(f"{self.path}.xmp")
        return self.path.with_name(
            f"{self.path.stem}_{self.version:02d}{self.path.suffix}.xmp")


def _connect(library_dir: Path) -> sqlite3.Connection:
    db_path = library_dir / "library.db"
    if not db_path.exists():
        raise FileNotFoundError(f"no library.db found under {library_dir}")
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)


def images_in_folder(library_dir: Path, folder: Path) -> list[ImageRecord]:
    """All images darktable has imported from this exact folder (film roll).

    Files physically present in `folder` but never imported into this
    library are not returned - the caller falls back to a direct render for
    those, since there is no image id and thus no mipmap cache entry.
    """
    folder = folder.resolve()
    with closing(_connect(library_dir)) as conn:
        rows = conn.execute(
            """
            SELECT images.id, images.filename, images.version, images.write_timestamp
            FROM images
            JOIN film_rolls ON film_rolls.id = images.film_id
            WHERE film_rolls.folder = ?
            ORDER BY images.filename, images.version
            """,
            (str(folder),),
        ).fetchall()
    return [
        ImageRecord(id=row[0], path=folder / row[1], version=row[2], write_timestamp=row[3])
        for row in rows
    ]


def image_for_id(library_dir: Path, image_id: int) -> ImageRecord | None:
    """Look up a single image by darktable's own image id - the only way to
    address a specific duplicate/version unambiguously (image_for_file can
    only ever resolve version 0). Used by the Lua/UI integration, which
    always has the exact id from the live selection. Returns None if this
    id doesn't exist in the library.
    """
    with closing(_connect(library_dir)) as conn:
        row = conn.execute(
            """
            SELECT images.filename, film_rolls.folder, images.version, images.write_timestamp
            FROM images
            JOIN film_rolls ON film_rolls.id = images.film_id
            WHERE images.id = ?
            """,
            (image_id,),
        ).fetchone()
    if row is None:
        return None
    filename, folder, version, write_timestamp = row
    return ImageRecord(id=image_id, path=Path(folder) / filename,
                        version=version, write_timestamp=write_timestamp)


def image_for_file(library_dir: Path, file_path: Path) -> ImageRecord:
    """Look up a single file; falls back to an id-less record if darktable
    has never imported it.

    If the file has duplicates/versions in darktable, this returns the base
    version (0) - a specific duplicate can only be addressed via --folder,
    since a bare file path can't disambiguate between them.
    """
    file_path = file_path.resolve()
    with closing(_connect(library_dir)) as conn:
        row = conn.execute(
            """
            SELECT images.id, images.write_timestamp
            FROM images
            JOIN film_rolls ON film_rolls.id = images.film_id
            WHERE film_rolls.folder = ? AND images.filename = ? AND images.version = 0
            """,
            (str(file_path.parent), file_path.name),
        ).fetchone()
    if row is None:
        return ImageRecord(id=None, path=file_path, version=0, write_timestamp=None)
    return ImageRecord(id=row[0], path=file_path, version=0, write_timestamp=row[1])
