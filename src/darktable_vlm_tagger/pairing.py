"""Group a RAW+JPEG shot of the same frame so it is tagged once, not twice.

Only active when the CLI is run with ``--pair-raw-jpeg`` and only for
``--folder`` mode. The rule is deliberately fixed and narrow - no inspection
of develop history, no guessing. Two records are a pair *iff all* of:

1. both have ``version == 0`` (virtual copies / duplicates are never paired);
2. they share the same ``images.group_id`` - darktable's own "same shot"
   grouping, assigned automatically on import for RAW+JPEG;
3. the group has exactly two members: one RAW and one JPEG;
4. their filename stems match case-insensitively.

Anything else - lone files, a group of one RAW, JPEG-only bursts, manually
grouped bracket/panorama sets (distinct stems), groups of three or more, any
``version > 0`` record - passes through untouched and is tagged individually.

The config setting ``[pairing] render_source`` decides which member of a pair
is the *primary* (the one actually rendered and sent to the model); the other
is a *secondary* that receives a verbatim copy of the primary's result.
"""

from dataclasses import dataclass

from .darktable_db import ImageRecord

# Fixed lowercase list, leading dot. Add an unusual camera's extension here.
RAW_SUFFIXES = frozenset({
    ".3fr", ".ari", ".arw", ".bay", ".cap", ".cr2", ".cr3", ".crw", ".dcr",
    ".dcs", ".dng", ".drf", ".eip", ".erf", ".fff", ".gpr", ".iiq", ".k25",
    ".kdc", ".mdc", ".mef", ".mos", ".mrw", ".nef", ".nrw", ".orf", ".ori",
    ".pef", ".ptx", ".pxn", ".raf", ".raw", ".rw2", ".rwl", ".rwz", ".sr2",
    ".srf", ".srw", ".x3f",
})
JPEG_SUFFIXES = frozenset({".jpg", ".jpeg"})


@dataclass(frozen=True)
class WorkUnit:
    """One primary image to process, plus any secondaries that receive an
    identical copy of its result. ``secondaries`` is empty for every image
    when pairing is off, and for every image that does not match the pairing
    rule when it is on."""

    primary: ImageRecord
    secondaries: tuple[ImageRecord, ...] = ()

    @property
    def all_records(self) -> tuple[ImageRecord, ...]:
        return (self.primary, *self.secondaries)


def _is_raw(record: ImageRecord) -> bool:
    return record.path.suffix.lower() in RAW_SUFFIXES


def _is_jpeg(record: ImageRecord) -> bool:
    return record.path.suffix.lower() in JPEG_SUFFIXES


def _pair_primary_secondary(
    members: list[ImageRecord], render_source: str
) -> tuple[ImageRecord, ImageRecord] | None:
    """(primary, secondary) if ``members`` is a valid RAW+JPEG pair, else None."""
    if len(members) != 2:
        return None
    raws = [m for m in members if _is_raw(m)]
    jpegs = [m for m in members if _is_jpeg(m)]
    if len(raws) != 1 or len(jpegs) != 1:
        return None
    raw, jpeg = raws[0], jpegs[0]
    if raw.path.stem.lower() != jpeg.path.stem.lower():
        return None
    return (jpeg, raw) if render_source == "jpeg" else (raw, jpeg)


def build_work_units(
    records: list[ImageRecord], *, pair_raw_jpeg: bool, render_source: str = "raw"
) -> list[WorkUnit]:
    """Turn a flat record list into the units the CLI loop processes.

    Output order matches input order: a pair's ``WorkUnit`` is emitted at the
    position of its primary, and the secondary is dropped from the top level
    (it lives on ``WorkUnit.secondaries`` instead).
    """
    if not pair_raw_jpeg:
        return [WorkUnit(record) for record in records]

    by_group: dict[int, list[ImageRecord]] = {}
    for record in records:
        if record.version == 0 and record.group_id is not None:
            by_group.setdefault(record.group_id, []).append(record)

    primary_to_secondaries: dict[int, tuple[ImageRecord, ...]] = {}
    secondary_ids: set[int] = set()
    for members in by_group.values():
        pair = _pair_primary_secondary(members, render_source)
        if pair is None:
            continue
        primary, secondary = pair
        primary_to_secondaries[primary.id] = (secondary,)
        secondary_ids.add(secondary.id)

    units: list[WorkUnit] = []
    for record in records:
        if record.id in secondary_ids:
            continue
        secondaries = primary_to_secondaries.get(record.id, ())
        units.append(WorkUnit(record, secondaries))
    return units
