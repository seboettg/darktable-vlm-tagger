"""Read, merge and write darktable XMP sidecars.

Only touches dc:subject, lr:hierarchicalSubject, dc:title and dc:description.
Everything else in an existing sidecar (history stack, masks, timestamps) is
preserved untouched: this parses the whole file and edits only the elements
it owns.

XMP layout verified against `libdarktable.so` strings and a real sidecar from
the archive: darktable decomposes every hierarchical tag path into its
individual segments and writes each segment into the flat dc:subject list,
except for segments flagged as a "category" node in darktable's own tag
database (data.db - not the sidecar). This tool never touches that database,
so it reproduces the same effect purely at the sidecar layer for the one
namespace segment the vocabulary intends as a category ("category" itself):
that literal segment is excluded when building the flat list, everything
else (including the actual category value, e.g. "landscape") is included
normally. Flagging the `category` tag as an actual darktable category node
is a one-off manual step (documented in the README) that only affects
cosmetics - without it "category" additionally shows up as a flat keyword.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

NS = {
    "x": "adobe:ns:meta/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "lr": "http://ns.adobe.com/lightroom/1.0/",
    "exif": "http://ns.adobe.com/exif/1.0/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
    "xmpMM": "http://ns.adobe.com/xap/1.0/mm/",
    "darktable": "http://darktable.sf.net/",
}
for _prefix, _uri in NS.items():
    ET.register_namespace(_prefix, _uri)

RDF = NS["rdf"]
DC = NS["dc"]
LR = NS["lr"]
XML_LANG = "{http://www.w3.org/XML/1998/namespace}lang"

MARKER_TAG = "darktable-vlm-tagger|tagged"
CATEGORY_SEGMENT = "category"  # excluded from the flat dc:subject list


def _q(ns: str, tag: str) -> str:
    return f"{{{ns}}}{tag}"


def _new_tree() -> ET.ElementTree:
    root = ET.Element(_q(NS["x"], "xmpmeta"), {_q(NS["x"], "xmptk"): "darktable-vlm-tagger"})
    rdf = ET.SubElement(root, _q(RDF, "RDF"))
    ET.SubElement(rdf, _q(RDF, "Description"), {_q(RDF, "about"): ""})
    return ET.ElementTree(root)


def _description(tree: ET.ElementTree) -> ET.Element:
    root = tree.getroot()
    rdf = root.find(_q(RDF, "RDF"))
    desc = rdf.find(_q(RDF, "Description"))
    if desc is None:
        desc = ET.SubElement(rdf, _q(RDF, "Description"), {_q(RDF, "about"): ""})
    return desc


def _bag_items(desc: ET.Element, tag_qname: str) -> list[str]:
    el = desc.find(tag_qname)
    if el is None:
        return []
    bag = el.find(_q(RDF, "Bag"))
    if bag is None:
        return []
    return [li.text or "" for li in bag.findall(_q(RDF, "li"))]


def _set_bag(desc: ET.Element, tag_qname: str, items: list[str]) -> None:
    el = desc.find(tag_qname)
    if el is None:
        el = ET.SubElement(desc, tag_qname)
    else:
        el.clear()
    bag = ET.SubElement(el, _q(RDF, "Bag"))
    for item in items:
        li = ET.SubElement(bag, _q(RDF, "li"))
        li.text = item


def _set_alt(desc: ET.Element, tag_qname: str, value: str) -> None:
    el = desc.find(tag_qname)
    if el is None:
        el = ET.SubElement(desc, tag_qname)
    else:
        el.clear()
    alt = ET.SubElement(el, _q(RDF, "Alt"))
    li = ET.SubElement(alt, _q(RDF, "li"))
    li.set(XML_LANG, "x-default")
    li.text = value


def is_already_tagged(sidecar_path: Path) -> bool:
    if not sidecar_path.exists():
        return False
    tree = ET.parse(sidecar_path)
    desc = _description(tree)
    return MARKER_TAG in _bag_items(desc, _q(LR, "hierarchicalSubject"))


def write_tags(sidecar_path: Path, *, tag_paths: list[str], title: str,
               description: str) -> None:
    """Merge `tag_paths` (full hierarchical paths, e.g. "category|landscape")
    plus the resume marker into the sidecar, and set title/description.

    Existing tags (e.g. darktable's own "kept" functional `darktable|...`
    tags) are preserved: hierarchicalSubject is a union, never a replacement.
    """
    tree = ET.parse(sidecar_path) if sidecar_path.exists() else _new_tree()
    desc = _description(tree)

    hier_qname = _q(LR, "hierarchicalSubject")
    existing_paths = _bag_items(desc, hier_qname)
    new_paths = list(dict.fromkeys(existing_paths + tag_paths + [MARKER_TAG]))
    _set_bag(desc, hier_qname, new_paths)

    existing_flat = _bag_items(desc, _q(DC, "subject"))
    segments = {
        segment
        for path in new_paths
        for segment in path.split("|")
        if segment != CATEGORY_SEGMENT
    }
    flat = list(dict.fromkeys(existing_flat + sorted(segments)))
    _set_bag(desc, _q(DC, "subject"), flat)

    _set_alt(desc, _q(DC, "title"), title)
    _set_alt(desc, _q(DC, "description"), description)

    ET.indent(tree, space="  ")
    tree.write(sidecar_path, encoding="UTF-8", xml_declaration=True)
