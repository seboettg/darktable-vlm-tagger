from darktable_vlm_tagger.parsing import close_brackets, parse_json_ish, repair


def test_clean_json_parses_directly():
    data, note = parse_json_ish('{"subject": ["a", "b"]}')
    assert data == {"subject": ["a", "b"]}
    assert note is None


def test_json_wrapped_in_code_fence():
    raw = '```json\n{"subject": ["a"]}\n```'
    data, note = parse_json_ish(raw)
    assert data == {"subject": ["a"]}


def test_json_surrounded_by_prose():
    raw = 'Sure, here is the JSON:\n{"subject": ["a"]}\nHope that helps!'
    data, note = parse_json_ish(raw)
    assert data == {"subject": ["a"]}
    assert note == "extracted from prose"


def test_list_leak_repaired():
    # The regex is line-based (real model output puts each field on its own
    # line); a bare fragment is enough to exercise it directly.
    broken = '"mood": "calm", "tense"'
    fixed = repair(broken)
    assert fixed == '"mood": ["calm", "tense"]'


def test_truncated_response_is_closed():
    truncated = '{"subject": ["a", "b"'
    closed = close_brackets(truncated)
    assert closed == '{"subject": ["a", "b"]}'


def test_truncated_mid_string_is_closed():
    truncated = '{"description": "a tram in the fo'
    closed = close_brackets(truncated)
    assert closed == '{"description": "a tram in the fo"}'


def test_unparseable_garbage_returns_none_with_note():
    data, note = parse_json_ish("not json at all")
    assert data is None
    assert "no JSON object" in note


def test_list_leak_and_truncation_combined():
    # Multi-line, like real model output - and missing the closing brace.
    raw = '{\n  "mood": "calm", "tense"'
    data, note = parse_json_ish(raw)
    assert data == {"mood": ["calm", "tense"]}
    assert note == "repaired and closed"
