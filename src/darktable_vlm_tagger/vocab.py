"""Load the closed-vocabulary spec and turn it into a JSON schema / prompt block."""

import json
from pathlib import Path

FIELDS = ["category", "color", "tone", "light", "composition",
          "technique", "mood", "subject"]


def load_vocab(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def build_schema(vocab: dict) -> dict:
    """JSON schema for Ollama's `format` field. The enum constraint makes the
    model structurally unable to produce a value outside the vocabulary."""
    props, required = {}, []
    for field in FIELDS:
        spec = vocab.get(field)
        if not spec:
            continue
        item = {"type": "string"}
        if spec.get("values"):
            item["enum"] = spec["values"]
        props[field] = {
            "type": "array",
            "items": item,
            "minItems": spec.get("min", 0),
            "maxItems": spec.get("max", 12),
            "uniqueItems": True,
        }
        required.append(field)
    props["title"] = {"type": "string"}
    required.append("title")
    props["description"] = {"type": "string"}
    required.append("description")
    return {"type": "object", "properties": props, "required": required}


def vocab_block(vocab: dict) -> str:
    """The vocabulary section of the prompt. The schema enforces the values
    regardless, but a model that has seen the list picks better from it."""
    lines = []
    for field in FIELDS:
        spec = vocab.get(field)
        if not spec:
            continue
        lo, hi = spec.get("min", 0), spec.get("max", 12)
        count = f"exactly {hi}" if lo == hi else f"{lo}-{hi}"
        lines.append(f'"{field}" ({count})' +
                     (f" — {spec['hint']}" if spec.get("hint") else ""))
        if spec.get("values"):
            lines.append("    choose only from: " + ", ".join(spec["values"]))
        else:
            lines.append("    free vocabulary")
    return "\n".join(lines)


def closed_values(vocab: dict) -> frozenset[str]:
    """Every value from every closed-vocabulary field (lowercased), i.e.
    everything except "subject" itself. Used to strip words out of "subject"
    that belong to another field - the model doesn't reliably follow the
    prompt rule against this on its own (e.g. "people", "reflection" leak
    through because they read as ordinary descriptive words)."""
    return frozenset(
        v.lower()
        for field in FIELDS
        if field != "subject"
        for v in (vocab.get(field, {}).get("values") or [])
    )


def tag_list(data: dict) -> list:
    if not isinstance(data, dict):
        return []
    return data.get("subject") or []
