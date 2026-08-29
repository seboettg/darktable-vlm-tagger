from pathlib import Path

from darktable_vlm_tagger.darktable_db import ImageRecord
from darktable_vlm_tagger.pairing import build_work_units


def _rec(img_id, name, *, version=0, group_id=None):
    return ImageRecord(id=img_id, path=Path("/roll") / name, version=version,
                       write_timestamp=None,
                       group_id=img_id if group_id is None else group_id)


def _units(records, **kw):
    kw.setdefault("pair_raw_jpeg", True)
    return build_work_units(records, **kw)


def test_flag_off_is_one_unit_per_record_in_order():
    recs = [_rec(1, "a.RAF"), _rec(2, "a.JPG", group_id=1), _rec(3, "b.RAF")]
    units = build_work_units(recs, pair_raw_jpeg=False)
    assert [u.primary for u in units] == recs
    assert all(u.secondaries == () for u in units)


def test_raw_jpeg_group_is_paired_with_raw_primary():
    raw, jpg = _rec(10, "shot.RAF", group_id=10), _rec(11, "shot.JPG", group_id=10)
    units = _units([raw, jpg])
    assert len(units) == 1
    assert units[0].primary is raw
    assert units[0].secondaries == (jpg,)


def test_render_source_jpeg_flips_primary_and_secondary():
    raw, jpg = _rec(10, "shot.RAF", group_id=10), _rec(11, "shot.JPG", group_id=10)
    units = _units([raw, jpg], render_source="jpeg")
    assert units[0].primary is jpg
    assert units[0].secondaries == (raw,)


def test_secondary_dropped_from_top_level_order_preserved():
    a = _rec(1, "a.RAF")
    raw, jpg = _rec(10, "shot.RAF", group_id=10), _rec(11, "shot.JPG", group_id=10)
    z = _rec(20, "z.RAF")
    units = _units([a, raw, jpg, z])
    assert [u.primary for u in units] == [a, raw, z]


def test_not_paired_when_group_ids_differ():
    raw, jpg = _rec(10, "shot.RAF", group_id=10), _rec(11, "shot.JPG", group_id=11)
    units = _units([raw, jpg])
    assert [u.secondaries for u in units] == [(), ()]


def test_not_paired_when_stems_differ():
    raw, jpg = _rec(10, "aaa.RAF", group_id=10), _rec(11, "bbb.JPG", group_id=10)
    units = _units([raw, jpg])
    assert [u.secondaries for u in units] == [(), ()]


def test_not_paired_when_a_duplicate_version_is_in_the_group():
    raw = _rec(10, "shot.RAF", group_id=10)
    jpg = _rec(11, "shot.JPG", group_id=10)
    dup = _rec(12, "shot.RAF", version=1, group_id=10)
    units = _units([raw, jpg, dup])
    # version-0 members are RAW + JPEG with matching stem -> still a valid pair;
    # the duplicate is its own standalone unit.
    assert len(units) == 2
    paired = next(u for u in units if u.secondaries)
    assert paired.primary is raw and paired.secondaries == (jpg,)
    assert any(u.primary is dup and not u.secondaries for u in units)


def test_not_paired_when_group_has_three_version0_members():
    a = _rec(10, "burst.RAF", group_id=10)
    b = _rec(11, "burst.JPG", group_id=10)
    c = _rec(12, "burst.jpeg", group_id=10)
    units = _units([a, b, c])
    assert [u.secondaries for u in units] == [(), (), ()]


def test_not_paired_when_group_is_two_raws():
    a = _rec(10, "shot.RAF", group_id=10)
    b = _rec(11, "shot.DNG", group_id=10)
    units = _units([a, b])
    assert [u.secondaries for u in units] == [(), ()]


def test_lone_raw_and_jpeg_only_group_are_not_paired():
    lone = _rec(10, "only.RAF", group_id=10)
    j1 = _rec(11, "pair.JPG", group_id=11)
    j2 = _rec(12, "pair.jpeg", group_id=11)
    units = _units([lone, j1, j2])
    assert [u.secondaries for u in units] == [(), (), ()]


def test_suffix_and_stem_matching_is_case_insensitive():
    raw = _rec(10, "IMG_1.rw2", group_id=10)
    jpg = _rec(11, "img_1.JPG", group_id=10)
    units = _units([raw, jpg])
    assert units[0].primary is raw and units[0].secondaries == (jpg,)
