"""Per-image processing: source the image, ask the model, write the result.

Resume works by checking the sidecar itself for the marker tag before doing
any work, so a re-run over a whole folder skips already-tagged images
without needing a separate state file.
"""

import base64
import json
import time
from pathlib import Path

from . import image_source, ollama_client, parsing, sidecar, vocab
from .config import Config
from .darktable_db import ImageRecord


def tag_paths_from_result(data: dict) -> list[str]:
    return [
        f"{field}|{value}"
        for field in vocab.FIELDS
        for value in (data.get(field) or [])
    ]


def process_image(record: ImageRecord, cfg: Config, library_dir: Path,
                   schema: dict, prompt: str, closed_values: frozenset[str] = frozenset(),
                   source_file: Path | None = None, force: bool = False) -> dict:
    """Returns a status dict: {"outcome": "tagged" | "skipped" | "error", ...}.

    `source_file`, if given, is an already-rendered image file to use
    directly instead of sourcing one from the mipmap cache/render fallback -
    see image_source.load_bytes_from_file for why the Lua/UI integration
    needs this.
    """
    sidecar_path = record.sidecar_path
    if cfg.mode == "sidecar" and not force and sidecar.is_already_tagged(sidecar_path):
        return {"outcome": "skipped", "reason": "already tagged"}

    started = time.monotonic()
    try:
        if source_file is not None:
            image_bytes = image_source.load_bytes_from_file(
                source_file, long_edge=cfg.target_long_edge)
        else:
            image_bytes = image_source.load_image_bytes(
                record, library_dir=library_dir, long_edge=cfg.target_long_edge,
                mip_level=cfg.mip_level,
            )
    except Exception as exc:
        return {"outcome": "error", "reason": f"could not source image: {exc}"}

    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    data, note = None, None
    for _attempt in range(cfg.retries + 1):
        try:
            raw, _elapsed, _diag = ollama_client.ask(
                cfg.host, cfg.model, prompt, image_b64, schema,
                cfg.timeout, cfg.num_ctx, cfg.num_predict,
            )
        except Exception as exc:
            # Transient (Ollama/llama-server hiccups, timeouts) and permanent
            # failures look the same from here - both get the same retry
            # budget as a parse failure, and only surface as an error once
            # that budget is exhausted.
            note = f"ollama request failed: {exc}"
            continue

        data, note = parsing.parse_json_ish(raw) if raw.strip() else (None, "empty response")
        if data is not None and closed_values and isinstance(data.get("subject"), list):
            data["subject"] = [s for s in data["subject"] if s.lower() not in closed_values]
        if data is not None and not vocab.tag_list(data):
            data, note = None, "valid JSON but no subject tags"
        if data is not None:
            break

    seconds = round(time.monotonic() - started, 2)
    if data is None:
        return {"outcome": "error", "reason": note or "unparseable response", "seconds": seconds}

    if cfg.mode == "sidecar":
        try:
            sidecar.write_tags(
                sidecar_path,
                tag_paths=tag_paths_from_result(data),
                title=data.get("title", ""),
                description=data.get("description", ""),
            )
        except Exception as exc:
            return {"outcome": "error",
                    "reason": f"could not write sidecar: {exc}", "seconds": seconds}

    return {"outcome": "tagged", "data": data, "seconds": seconds}


def append_log(log_path: Path, record: ImageRecord, status: dict) -> None:
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "path": str(record.path),
        "version": record.version,
        "sidecar": str(record.sidecar_path),
        **status,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
