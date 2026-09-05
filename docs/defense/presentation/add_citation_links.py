"""Attach external source hyperlinks to named citation text boxes in a draft.

Artifact Tool authors the complete deck. This package step supplies click
targets before the standard finalizer validates and delivers the PowerPoint.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
import zipfile

NS = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def attach_links(pptx: Path, manifest: Path) -> None:
    links = json.loads(manifest.read_text())
    with zipfile.ZipFile(pptx) as archive:
        entries = {entry.filename: (entry, archive.read(entry.filename))
                   for entry in archive.infolist()}
    for link in links:
        assert link["url"].startswith("https://arxiv.org/")
        slide_path = f"ppt/slides/slide{link['slide']}.xml"
        rels_path = f"ppt/slides/_rels/slide{link['slide']}.xml.rels"
        slide = ET.fromstring(entries[slide_path][1])
        rels = ET.fromstring(entries[rels_path][1])
        matches = [item for item in slide.findall(".//p:sp/p:nvSpPr/p:cNvPr", NS)
                   if item.get("name") == link["name"]]
        assert len(matches) == 1, link["name"]
        assert matches[0].find("a:hlinkClick", NS) is None
        used = {item.get("Id") for item in rels}
        number = 1
        while f"rIdCitation{number}" in used:
            number += 1
        identifier = f"rIdCitation{number}"
        ET.SubElement(rels, f"{{{PKG}}}Relationship", {
            "Id": identifier,
            "Type": NS["r"] + "/hyperlink",
            "Target": link["url"],
            "TargetMode": "External",
        })
        ET.SubElement(matches[0], f"{{{NS['a']}}}hlinkClick", {
            f"{{{NS['r']}}}id": identifier,
            "tooltip": "Open the cited paper",
        })
        for name, root in [(slide_path, slide), (rels_path, rels)]:
            entries[name] = (entries[name][0], ET.tostring(
                root, encoding="utf-8", xml_declaration=True))
    temporary = pptx.with_suffix(".linked.pptx")
    with zipfile.ZipFile(temporary, "w") as archive:
        for info, content in entries.values():
            archive.writestr(info, content)
    temporary.replace(pptx)
    print(f"Attached {len(links)} source hyperlinks")


if __name__ == "__main__":
    attach_links(Path(sys.argv[1]), Path(sys.argv[2]))
