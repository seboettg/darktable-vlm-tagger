import sqlite3
from pathlib import Path

from darktable_vlm_tagger.darktable_db import (
    ImageRecord,
    image_for_id,
    images_in_folder,
)


def _make_library(tmp_path):
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT)")
    conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, "
                  "filename TEXT, version INTEGER, write_timestamp INTEGER, "
                  "group_id INTEGER)")
    conn.execute("INSERT INTO film_rolls VALUES (1, '/roll')")
    conn.execute("INSERT INTO images VALUES (10, 1, 'photo.RAF', 0, 1000, 10)")
    conn.execute("INSERT INTO images VALUES (11, 1, 'photo.RAF', 1, 1001, 10)")
    conn.commit()
    conn.close()
    return tmp_path


def test_sidecar_path_for_base_version():
    record = ImageRecord(id=1, path=Path("/roll/photo.RAF"), version=0, write_timestamp=None)
    assert record.sidecar_path == Path("/roll/photo.RAF.xmp")


def test_sidecar_path_for_duplicate_version():
    # darktable duplicates share the physical RAW but get their own sidecar,
    # named with a zero-padded version suffix before the original extension.
    record = ImageRecord(id=2, path=Path("/roll/photo.RAF"), version=1, write_timestamp=None)
    assert record.sidecar_path == Path("/roll/photo_01.RAF.xmp")


def test_sidecar_path_for_higher_duplicate_version():
    record = ImageRecord(id=3, path=Path("/roll/photo.RAF"), version=12, write_timestamp=None)
    assert record.sidecar_path == Path("/roll/photo_12.RAF.xmp")


def test_image_for_id_finds_base_version(tmp_path):
    library_dir = _make_library(tmp_path)
    record = image_for_id(library_dir, 10)
    assert record == ImageRecord(id=10, path=Path("/roll/photo.RAF"),
                                  version=0, write_timestamp=1000, group_id=10)


def test_image_for_id_finds_specific_duplicate(tmp_path):
    # this is the whole point of image_for_id over image_for_file: it can
    # address a duplicate/version > 0 unambiguously by darktable's own id.
    library_dir = _make_library(tmp_path)
    record = image_for_id(library_dir, 11)
    assert record == ImageRecord(id=11, path=Path("/roll/photo.RAF"),
                                  version=1, write_timestamp=1001, group_id=10)
    assert record.sidecar_path == Path("/roll/photo_01.RAF.xmp")


def test_image_for_id_returns_none_for_unknown_id(tmp_path):
    library_dir = _make_library(tmp_path)
    assert image_for_id(library_dir, 999) is None


def test_images_in_folder_populates_shared_group_id(tmp_path):
    # darktable auto-groups a RAW+JPEG frame on import: both rows carry the
    # same group_id, which pairing.py keys off.
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT)")
    conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, "
                  "filename TEXT, version INTEGER, write_timestamp INTEGER, "
                  "group_id INTEGER)")
    folder = tmp_path.resolve()
    conn.execute("INSERT INTO film_rolls VALUES (1, ?)", (str(folder),))
    conn.execute("INSERT INTO images VALUES (10, 1, 'shot.RAF', 0, 1000, 10)")
    conn.execute("INSERT INTO images VALUES (11, 1, 'shot.JPG', 0, 1000, 10)")
    conn.commit()
    conn.close()

    records = images_in_folder(tmp_path, folder)
    assert {r.path.name: r.group_id for r in records} == {
        "shot.RAF": 10, "shot.JPG": 10,
    }
