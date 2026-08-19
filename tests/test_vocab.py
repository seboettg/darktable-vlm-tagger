import json

from darktable_vlm_tagger import vocab

SAMPLE = {
    "_comment": ["ignored"],
    "category": {"min": 1, "max": 1, "hint": "pick one",
                 "values": ["landscape", "portrait"]},
    "color": {"min": 0, "max": 2, "hint": "stands out",
              "values": ["blue", "red"]},
    "subject": {"min": 2, "max": 4, "hint": "free vocabulary", "values": None},
}


def test_load_vocab_drops_underscore_keys(tmp_path):
    path = tmp_path / "vocab.json"
    path.write_text(json.dumps(SAMPLE), encoding="utf-8")
    data = vocab.load_vocab(path)
    assert "_comment" not in data
    assert "category" in data


def test_build_schema_enforces_enum_and_min_max():
    schema = vocab.build_schema(SAMPLE)
    category = schema["properties"]["category"]
    assert category["items"]["enum"] == ["landscape", "portrait"]
    assert category["minItems"] == 1
    assert category["maxItems"] == 1

    subject = schema["properties"]["subject"]
    assert "enum" not in subject["items"]
    assert subject["minItems"] == 2
    assert subject["maxItems"] == 4

    assert schema["properties"]["title"] == {"type": "string"}
    assert schema["properties"]["description"] == {"type": "string"}
    assert "title" in schema["required"]
    assert "description" in schema["required"]


def test_vocab_block_lists_closed_values_and_marks_free_field():
    block = vocab.vocab_block(SAMPLE)
    assert "choose only from: landscape, portrait" in block
    assert "free vocabulary" in block


def test_closed_values_collects_all_fields_except_subject():
    values = vocab.closed_values(SAMPLE)
    assert values == {"landscape", "portrait", "blue", "red"}


def test_tag_list_reads_subject_field():
    assert vocab.tag_list({"subject": ["a", "b"]}) == ["a", "b"]
    assert vocab.tag_list({"subject": []}) == []
    assert vocab.tag_list({}) == []
    assert vocab.tag_list(None) == []
