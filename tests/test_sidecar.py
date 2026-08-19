import xml.etree.ElementTree as ET

from darktable_vlm_tagger import sidecar

EXISTING_SIDECAR = """<?xml version="1.0" encoding="UTF-8"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="XMP Core 4.4.0-Exiv2">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:darktable="http://darktable.sf.net/"
    xmlns:dc="http://purl.org/dc/elements/1.1/"
    xmlns:lr="http://ns.adobe.com/lightroom/1.0/"
   darktable:history_end="3">
   <dc:subject>
    <rdf:Bag>
     <rdf:li>changed</rdf:li>
     <rdf:li>darktable</rdf:li>
     <rdf:li>format</rdf:li>
     <rdf:li>raf</rdf:li>
    </rdf:Bag>
   </dc:subject>
   <lr:hierarchicalSubject>
    <rdf:Bag>
     <rdf:li>darktable|changed</rdf:li>
     <rdf:li>darktable|format|raf</rdf:li>
    </rdf:Bag>
   </lr:hierarchicalSubject>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
"""


def _find_bag_texts(root, tag_qname):
    ns = {"rdf": sidecar.RDF}
    desc = root.find(f".//{{{sidecar.RDF}}}Description")
    el = desc.find(tag_qname)
    bag = el.find(f"{{{sidecar.RDF}}}Bag")
    return [li.text for li in bag.findall(f"{{{sidecar.RDF}}}li")]


def test_is_already_tagged_false_when_no_sidecar(tmp_path):
    assert sidecar.is_already_tagged(tmp_path / "missing.xmp") is False


def test_write_tags_creates_new_sidecar_when_missing(tmp_path):
    path = tmp_path / "fresh.RAF.xmp"
    sidecar.write_tags(path, tag_paths=["category|landscape", "subject|cyclist"],
                        title="Cyclist at dusk", description="A cyclist crosses an intersection.")

    tree = ET.parse(path)
    root = tree.getroot()
    hier = _find_bag_texts(root, f"{{{sidecar.LR}}}hierarchicalSubject")
    assert "category|landscape" in hier
    assert "subject|cyclist" in hier
    assert sidecar.MARKER_TAG in hier
    assert sidecar.is_already_tagged(path) is True


def test_write_tags_merges_with_existing_tags(tmp_path):
    path = tmp_path / "existing.RAF.xmp"
    path.write_text(EXISTING_SIDECAR, encoding="utf-8")

    sidecar.write_tags(path, tag_paths=["category|landscape", "color|blue"],
                        title="Title", description="Description.")

    root = ET.parse(path).getroot()
    hier = _find_bag_texts(root, f"{{{sidecar.LR}}}hierarchicalSubject")
    # existing darktable-internal tags survive
    assert "darktable|changed" in hier
    assert "darktable|format|raf" in hier
    # new tags were added
    assert "category|landscape" in hier
    assert "color|blue" in hier


def test_category_segment_excluded_from_flat_subject(tmp_path):
    path = tmp_path / "category.RAF.xmp"
    sidecar.write_tags(path, tag_paths=["category|landscape", "color|blue"],
                        title="Title", description="Description.")

    root = ET.parse(path).getroot()
    flat = _find_bag_texts(root, f"{{{sidecar.DC}}}subject")
    assert "category" not in flat
    assert "landscape" in flat
    assert "color" in flat
    assert "blue" in flat


def test_write_tags_sets_title_and_description(tmp_path):
    path = tmp_path / "meta.RAF.xmp"
    sidecar.write_tags(path, tag_paths=[], title="A short title",
                        description="One factual sentence.")

    root = ET.parse(path).getroot()
    desc = root.find(f".//{{{sidecar.RDF}}}Description")
    title_li = desc.find(f"{{{sidecar.DC}}}title/{{{sidecar.RDF}}}Alt/{{{sidecar.RDF}}}li")
    description_li = desc.find(
        f"{{{sidecar.DC}}}description/{{{sidecar.RDF}}}Alt/{{{sidecar.RDF}}}li")
    assert title_li.text == "A short title"
    assert description_li.text == "One factual sentence."
    assert title_li.get("{http://www.w3.org/XML/1998/namespace}lang") == "x-default"


def test_write_tags_is_idempotent_on_marker(tmp_path):
    path = tmp_path / "idempotent.RAF.xmp"
    sidecar.write_tags(path, tag_paths=["subject|cyclist"], title="A", description="B.")
    sidecar.write_tags(path, tag_paths=["subject|cyclist"], title="A", description="B.")

    root = ET.parse(path).getroot()
    hier = _find_bag_texts(root, f"{{{sidecar.LR}}}hierarchicalSubject")
    assert hier.count(sidecar.MARKER_TAG) == 1
    assert hier.count("subject|cyclist") == 1
