"""dt-vlm-tag: tag, title and describe darktable photographs with a local VLM."""

import argparse
import json
import sys
from pathlib import Path

from . import batch, config, darktable_db, image_source, vocab


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="dt-vlm-tag",
        description="Tag, title and describe darktable photographs with a local "
                     "vision-language model.",
    )
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--folder", type=Path,
                         help="a darktable film roll (folder) to process")
    source.add_argument("--file", type=Path,
                         help="a single image file to process")
    source.add_argument("--image-id", type=int,
                         help="darktable image id to process (used by the Lua/UI "
                              "integration to address a specific duplicate/version)")

    ap.add_argument("--source-file", type=Path,
                     help="use this already-rendered image file directly instead of "
                          "sourcing one from the mipmap cache/render fallback - for use "
                          "with --image-id when darktable itself has already exported "
                          "the current develop state (the Lua/UI integration does this, "
                          "since no external process can safely re-render while "
                          "darktable is open)")
    ap.add_argument("--mode", choices=["print", "json", "sidecar"],
                     help="output mode (default: from config.toml, initially 'print')")
    ap.add_argument("--out", type=Path, default=Path("results.json"),
                     help="output file for --mode json (default: ./results.json)")
    ap.add_argument("--library", type=Path,
                     help="darktable config dir (default: ~/.config/darktable)")
    ap.add_argument("--config-dir", type=Path,
                     help="tool config dir (default: ~/.darktable-vlm-tagger)")
    ap.add_argument("--model", help="Ollama model tag (default: from config.toml)")
    ap.add_argument("--target-long-edge", type=int,
                     help="long edge in px sent to the model (default: from config.toml)")
    ap.add_argument("--mip-level", type=int,
                     help="darktable mipmap cache level to source from (default: from config.toml)")
    ap.add_argument("--force", action="store_true",
                     help="reprocess images that already carry the tagger's marker tag")
    ap.add_argument("--log-file", type=Path,
                     help="JSON-lines log file (default: <config-dir>/run.log)")
    return ap


def _load_existing_results(out_path: Path) -> dict:
    """Existing --mode json results to merge new ones into, if any.

    A pre-existing but empty file is treated as "nothing to merge yet", not
    an error - callers (e.g. the Lua integration's tmp-file helper) may
    pre-create the path to check writability before dt-vlm-tag ever runs.
    """
    if out_path.exists() and out_path.stat().st_size > 0:
        return json.loads(out_path.read_text(encoding="utf-8"))
    return {}


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    cfg = config.load_config(
        config_dir=args.config_dir or config.DEFAULT_CONFIG_DIR,
        model=args.model,
        target_long_edge=args.target_long_edge,
        mip_level=args.mip_level,
        library=args.library,
        mode=args.mode,
    )
    library_dir = cfg.library or darktable_db.DEFAULT_LIBRARY_DIR

    if cfg.mode == "sidecar" and image_source.darktable_is_running():
        print("darktable is running - close it before writing sidecars, or "
              "the changes race with darktable's own database.", file=sys.stderr)
        return 1

    vocab_data = vocab.load_vocab(cfg.vocab_path)
    schema = vocab.build_schema(vocab_data)
    closed_values = vocab.closed_values(vocab_data)
    prompt = cfg.prompt_path.read_text(encoding="utf-8").replace(
        "{VOCAB}", vocab.vocab_block(vocab_data))

    if args.folder:
        records = darktable_db.images_in_folder(library_dir, args.folder)
        if not records:
            print(f"No images found for {args.folder} in {library_dir} - "
                  f"has this folder been imported into this darktable library?",
                  file=sys.stderr)
            return 1
    elif args.file:
        records = [darktable_db.image_for_file(library_dir, args.file)]
    else:
        record = darktable_db.image_for_id(library_dir, args.image_id)
        if record is None:
            print(f"No image with id {args.image_id} found in {library_dir}.",
                  file=sys.stderr)
            return 1
        records = [record]

    log_path = args.log_file or (cfg.config_dir / "run.log")

    results = {}
    if cfg.mode == "json":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        results = _load_existing_results(args.out)

    total = len(records)
    exit_code = 0
    for i, record in enumerate(records, 1):
        label = record.path.name if record.version == 0 else \
            f"{record.path.name} (v{record.version})"
        status = batch.process_image(record, cfg, library_dir, schema, prompt,
                                      closed_values=closed_values,
                                      source_file=args.source_file, force=args.force)
        batch.append_log(log_path, record, status)

        outcome = status["outcome"]
        if outcome == "error":
            exit_code = 1
            print(f"[{i}/{total}] {label}: ERROR - {status['reason']}", file=sys.stderr)
            continue
        if outcome == "skipped":
            print(f"[{i}/{total}] {label}: skipped ({status['reason']})")
            continue

        data = status["data"]
        print(f"[{i}/{total}] {label}: {status['seconds']}s, "
              f"{len(data.get('subject') or [])} subject tags")

        if cfg.mode == "print":
            print(json.dumps(data, indent=2, ensure_ascii=False))
        elif cfg.mode == "json":
            # sidecar_path (not path) is the key: two duplicates/versions of
            # the same RAW share a path but have distinct edits and tags.
            results[str(record.sidecar_path)] = data
            args.out.write_text(json.dumps(results, indent=2, ensure_ascii=False),
                                 encoding="utf-8")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
