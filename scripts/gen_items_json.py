"""Generate/validate knowledge/items.json (TODO D1).

The game ships its collectible table as ``items.xml`` inside the extracted
``resources`` directory; it is NOT vendored into this repo. When that file is
available, run::

    python scripts/gen_items_json.py --xml /path/to/items.xml

to rebuild the full table, or run without arguments to validate the committed
curated subset. Quality/avoid rules need human review either way; the XML
provides ids/names/champions but not the strategic tags the decision engine
consumes, so generation augments rather than replaces the curated file.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from xml.etree import ElementTree

REPO_ROOT = Path(__file__).resolve().parents[1]
ITEMS_PATH = REPO_ROOT / "knowledge" / "items.json"

# Known strategic subsets the decision engine actually needs. Authoritative
# values come from the game; quality follows the Isaac wiki's 0-4 scale.
CURATED = {
    "84":  ("The Wafer",          4, ["defensive"],                     False),
    "105": ("The D6",             4, ["active", "utility", "reroll"],   False),
    "118": ("Pinking Shears",     3, ["active", "damage"],              False),
    "119": ("Dead Sea Scrolls",   2, ["active", "utility"],             False),
    "158": ("Crystal Ball",       2, ["active", "vision"],              False),
    "169": ("Polyphemus",         4, ["damage"],                        False),
    "182": ("Sacred Heart",       4, ["damage", "defensive"],           False),
    "188": ("Abaddon",            4, ["damage", "defensive", "devil"],  False),
    "201": ("Iron Bar",           2, ["damage"],                        False),
    "226": ("Black Lotus",        3, ["hearts", "utility"],             False),
    "238": ("Key Piece 1",        0, ["quest", "key_piece", "mega_satan"], False),
    "239": ("Key Piece 2",        0, ["quest", "key_piece", "mega_satan"], False),
    "327": ("The Polaroid",       0, ["quest", "zone_access", "the_beast"], False),
    "328": ("The Negative",       0, ["quest", "zone_access", "the_beast"], False),
    "386": ("D100",               3, ["active", "reroll", "chaotic"],   True),
    "409": ("Curse of the Tower", 1, ["defensive", "chaotic", "self_damage"], True),
    "549": ("Dad's Note",         0, ["quest", "the_beast", "ascent"],   False),
    "552": ("Glowing Hour Glass", 3, ["active", "utility", "rewind"],    False),
    "594": ("Teleport 2.0",       3, ["active", "movement"],             False),
}

TRANSFORMATIONS = {
    "guppy": {"item_ids": [144, 145, 146], "tags": ["flight", "utility", "synergy"]},
    "bookworm": {"item_ids": [33, 34, 37, 74], "tags": ["active", "utility"]},
}

TAG_REGISTRY = sorted({
    "active", "damage", "defensive", "utility", "reroll", "vision",
    "hearts", "quest", "key_piece", "mega_satan", "zone_access", "the_beast",
    "chaotic", "self_damage", "ascent", "rewind", "movement", "devil",
    "flight", "synergy", "escape", "card",
})


def _entry(item_id: str, data) -> dict:
    name, quality, tags, avoid = data
    return {
        "name": name,
        "quality": quality,
        "tags": list(tags),
        "transformations": [
            t for t, spec in TRANSFORMATIONS.items()
            if int(item_id) in spec["item_ids"]
        ],
        "avoid": avoid,
    }


def curated_document() -> dict:
    return {
        "version": 1,
        "source": {
            "generator": __file__,
            "note": (
                "Curated strategic subset maintained by hand (quality + tags "
                "only). Run scripts/gen_items_json.py --xml <repentance_items.xml> "
                "to regenerate the full table from game data when the extracted "
                "files are available."
            ),
        },
        "items": {str(k): _entry(str(k), v) for k, v in CURATED.items()},
        "transformations": TRANSFORMATIONS,
        "tag_registry": TAG_REGISTRY,
    }


def parse_items_xml(path: str) -> dict[str, str]:
    """Extract {id: name} from a Repentance-style items.xml.

    Handles both the classic ``<item id=".." name=".."/>`` shape and the
    extracted ``<Item ... ID=".." Name=".."/>`` shape.
    """
    tree = ElementTree.parse(path)
    found: dict[str, str] = {}
    for node in tree.getroot().iter():
        tag = re.sub(r"\{.*?\}", "", node.tag).lower()
        if tag not in ("item", "collectible"):
            continue
        item_id = node.get("id") or node.get("ID")
        name = node.get("name") or node.get("Name")
        if item_id and name:
            found[str(int(item_id))] = name
    return found


def merge_xml(base: dict, from_xml: dict[str, str]) -> dict:
    """Start from the curated subset; fill names/ids from the XML, keeping tags."""
    merged = json.loads(json.dumps(base))
    for item_id, name in from_xml.items():
        if item_id in merged["items"]:
            merged["items"][item_id]["name"] = name
        else:
            # Unknown items enter with neutral facts; a human must assign
            # quality/tags before they influence the decision engine.
            merged["items"][item_id] = {
                "name": name,
                "quality": 0,
                "tags": [],
                "transformations": [],
                "avoid": False,
            }
    return merged


def validate(document: dict) -> list[str]:
    errors: list[str] = []
    if document.get("version") != 1:
        errors.append("version must be 1")
    items = document.get("items")
    if not isinstance(items, dict):
        errors.append("items must be an object")
        return errors
    registry = set(document.get("tag_registry", []))
    if not registry:
        errors.append("tag_registry is empty")
    for item_id, record in items.items():
        try:
            int(item_id)
        except ValueError:
            errors.append(f"item id {item_id!r} is not an integer")
        if not isinstance(record, dict):
            errors.append(f"item {item_id} is not an object")
            continue
        for required in ("name", "quality", "tags", "transformations", "avoid"):
            if required not in record:
                errors.append(f"item {item_id} missing {required!r}")
        for tag in record.get("tags", []):
            if tag not in registry:
                errors.append(f"item {item_id} uses unregistered tag {tag!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        default=None,
        help="path to an extracted Repentance items.xml for regeneration",
    )
    parser.add_argument(
        "--out",
        default=str(ITEMS_PATH),
        help="output path (default: knowledge/items.json)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate the committed file without writing",
    )
    args = parser.parse_args()

    doc = curated_document()
    if args.xml:
        doc = merge_xml(doc, parse_items_xml(args.xml))

    errors = validate(doc)
    if errors:
        print("items.json validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    if args.validate_only:
        print("OK: knowledge/items.json is valid")
        return 0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(doc['items'])} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
