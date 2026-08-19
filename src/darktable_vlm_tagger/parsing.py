"""Robust-ish extraction of a JSON object from a VLM's text response.

VLMs like to wrap JSON in code fences, add prose around it, or get cut off
mid-object when num_predict is too small. This tries the clean path first,
then progressively more aggressive repairs.
"""

import json
import re

FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

# Matches '"mood": "a", "b", "c"' - the model wanted a list and forgot the
# brackets. Only matches lines that consist purely of a key and several
# strings; a nested key-value pair would need a colon and won't match.
LIST_LEAK = re.compile(
    r'^(\s*"[A-Za-z_]+"\s*:\s*)'
    r'("(?:[^"\\]|\\.)*"(?:\s*,\s*"(?:[^"\\]|\\.)*")+)'
    r'(\s*,?)\s*$',
    re.MULTILINE)


def close_brackets(text: str) -> str:
    """Best-effort close of a truncated response."""
    stack, in_str, esc = [], False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in "[{":
            stack.append(ch)
        elif ch in "]}" and stack:
            stack.pop()
    out = text
    if in_str:
        out += '"'
    out = out.rstrip().rstrip(",")
    for ch in reversed(stack):
        out += "]" if ch == "[" else "}"
    return out


def repair(text: str) -> str:
    return LIST_LEAK.sub(lambda m: f"{m.group(1)}[{m.group(2)}]{m.group(3)}", text)


def parse_json_ish(raw: str):
    """VLMs love to append fences or prose. Try clean, then loose, then repaired.

    Returns (data, note) where note is None on a clean parse, or a short
    description of which repair path succeeded / why it failed.
    """
    cleaned = FENCE.sub("", raw).strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1]), "extracted from prose"
        except json.JSONDecodeError:
            pass

    if start != -1:
        body = cleaned[start:]
        for label, candidate in (("list repaired", repair(body)),
                                 ("truncated, closed", close_brackets(body)),
                                 ("repaired and closed",
                                  close_brackets(repair(body)))):
            try:
                return json.loads(candidate), label
            except json.JSONDecodeError as exc:
                last = exc
        return None, f"unparseable: {last}"
    return None, "no JSON object in the response"
