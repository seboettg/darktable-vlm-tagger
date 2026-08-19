import sqlite3

import pytest

from darktable_vlm_tagger.cli import _load_existing_results, build_arg_parser, main


def _make_empty_library(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "library.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE film_rolls (id INTEGER PRIMARY KEY, folder TEXT)")
    conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, film_id INTEGER, "
                  "filename TEXT, version INTEGER, write_timestamp INTEGER)")
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
