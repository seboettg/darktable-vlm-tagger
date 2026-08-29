import json
import sqlite3

import pytest

from darktable_vlm_tagger import batch
from darktable_vlm_tagger.cli import _load_existing_results, build_arg_parser, main

_IMAGES_SCHEMA = ("CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, "
                  "filename TEXT, version INTEGER, write_timestamp INTEGER, "
                  "group_id INTEGER)")


def _make_empty_library(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT)")
    conn.execute(_IMAGES_SCHEMA)
    conn.commit()
    conn.close()
    return tmp_path


def test_source_group_is_mutually_exclusive():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--file", "a.jpg", "--image-id", "1"])


def test_source_group_requires_one_option():
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_image_id_accepts_an_integer():
    parser = build_arg_parser()
    args = parser.parse_args(["--image-id", "42"])
    assert args.image_id == 42
    assert args.folder is None
    assert args.file is None


def test_source_file_accepts_a_path_alongside_image_id():
    parser = build_arg_parser()
    args = parser.parse_args(["--image-id", "42", "--source-file", "/tmp/exported.jpg"])
    assert args.image_id == 42
    assert str(args.source_file) == "/tmp/exported.jpg"


def test_load_existing_results_treats_missing_file_as_empty(tmp_path):
    assert _load_existing_results(tmp_path / "nonexistent.json") == {}


def test_load_existing_results_treats_pre_created_empty_file_as_empty(tmp_path):
    # This is exactly what df.create_tmp_file() does in the Lua integration:
    # opens the path for writing (truncating it to 0 bytes) before dt-vlm-tag
    # ever runs, purely to check writability. json.loads("") used to crash
    # this with JSONDecodeError - see the Lua-triggered bug this regresses.
    out_path = tmp_path / "pre-touched.json"
    out_path.touch()
    assert out_path.stat().st_size == 0
    assert _load_existing_results(out_path) == {}


def test_load_existing_results_parses_real_content(tmp_path):
    out_path = tmp_path / "results.json"
    out_path.write_text('{"a.xmp": {"title": "x"}}', encoding="utf-8")
    assert _load_existing_results(out_path) == {"a.xmp": {"title": "x"}}


def test_nonexistent_image_id_exits_1(tmp_path, capsys):
    library_dir = _make_empty_library(tmp_path / "darktable")
    config_dir = tmp_path / "tool-config"
    exit_code = main(["--image-id", "999", "--library", str(library_dir),
                       "--config-dir", str(config_dir)])
    assert exit_code == 1
    assert "999" in capsys.readouterr().err


def test_pair_raw_jpeg_flag_defaults_off_and_parses():
    parser = build_arg_parser()
    assert parser.parse_args(["--folder", "x"]).pair_raw_jpeg is False
    assert parser.parse_args(["--folder", "x", "--pair-raw-jpeg"]).pair_raw_jpeg is True


def _make_folder_library(tmp_path, folder, rows):
    """rows: list of (id, filename, version, group_id)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT)")
    conn.execute(_IMAGES_SCHEMA)
    conn.execute("INSERT INTO film_rolls VALUES (1, ?)", (str(folder),))
    for img_id, filename, version, group_id in rows:
        conn.execute("INSERT INTO images VALUES (?, 1, ?, ?, 1000, ?)",
                      (img_id, filename, version, group_id))
    conn.commit()
    conn.close()
    return tmp_path


def test_pair_raw_jpeg_fans_result_out_to_both_siblings(tmp_path, monkeypatch, capsys):
    folder = (tmp_path / "roll").resolve()
    folder.mkdir()
    library_dir = _make_folder_library(tmp_path / "darktable", folder,
                                        [(10, "shot.RAF", 0, 4), (11, "shot.JPG", 0, 4)])
    config_dir = tmp_path / "tool-config"
    out = tmp_path / "results.json"
    log_file = tmp_path / "run.log"

    calls = []

    def fake_process_image(record, *a, **kw):
        calls.append(record)
        return {"outcome": "tagged",
                "data": {"subject": ["cat"], "title": "T", "description": "D"},
                "seconds": 1.0}

    monkeypatch.setattr(batch, "process_image", fake_process_image)

    exit_code = main(["--folder", str(folder), "--mode", "json", "--pair-raw-jpeg",
                       "--library", str(library_dir), "--config-dir", str(config_dir),
                       "--out", str(out), "--log-file", str(log_file)])

    assert exit_code == 0
    # one inference, on the RAW (default render_source)
    assert [r.path.name for r in calls] == ["shot.RAF"]
    # identical result under both sidecar keys
    results = json.loads(out.read_text())
    assert set(results) == {str(folder / "shot.RAF.xmp"), str(folder / "shot.JPG.xmp")}
    assert results[str(folder / "shot.RAF.xmp")] == results[str(folder / "shot.JPG.xmp")]
    # both logged, the sibling tagged as paired
    log_lines = [json.loads(x) for x in log_file.read_text().splitlines()]
    assert len(log_lines) == 2
    assert log_lines[1]["paired_with"] == str(folder / "shot.RAF.xmp")
    assert "+1 paired" in capsys.readouterr().out


def test_without_flag_raw_and_jpeg_are_processed_separately(tmp_path, monkeypatch):
    folder = (tmp_path / "roll").resolve()
    folder.mkdir()
    library_dir = _make_folder_library(tmp_path / "darktable", folder,
                                        [(10, "shot.RAF", 0, 4), (11, "shot.JPG", 0, 4)])
    config_dir = tmp_path / "tool-config"

    calls = []
    monkeypatch.setattr(batch, "process_image",
                         lambda record, *a, **kw: calls.append(record) or {
                             "outcome": "tagged",
                             "data": {"subject": ["x"], "title": "", "description": ""},
                             "seconds": 1.0})

    main(["--folder", str(folder), "--mode", "print",
          "--library", str(library_dir), "--config-dir", str(config_dir)])

    assert sorted(r.path.name for r in calls) == ["shot.JPG", "shot.RAF"]
