from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "repertoires" / "llm_knowledge+search" / "species"

# Only records whose own descriptions identify them as biosonar or mechanically
# generated are removed. Social click-like vocalizations remain in the database.
REMOVE_CALLS = {
    "corvus-brachyrhynchos.yaml": {"click"},
    "corvus-corax.yaml": {"knock call", "bill snap"},
    "delphinapterus-leucas.yaml": {"echolocation click train"},
    "desmodus-rotundus.yaml": {"echolocation pulse"},
    "heterocephalus-glaber.yaml": {"tooth chatter"},
    "leptonychotes-weddellii.yaml": {"jaw clap", "click"},
    "macaca-sylvanus.yaml": {"teeth chatter"},
    "mandrillus-sphinx.yaml": {"tooth grind"},
    "marmota-flaviventris.yaml": {"tooth chatter"},
    "megaptera-novaeangliae.yaml": {
        "pectoral slap sound",
        "lobtail sound",
        "breach sound",
    },
    "mimus-polyglottos.yaml": {"wing-flash display"},
    "octodon-degus.yaml": {"teeth snap"},
    "orcinus-orca.yaml": {
        "echolocation click train",
        "echolocation buzz",
        "prey-handling sound",
        "jaw clap",
        "tail slap",
    },
    "phyllostomus-discolor.yaml": {"echolocation call"},
    "phyllostomus-hastatus.yaml": {"echolocation pulse"},
    "physeter-macrocephalus.yaml": {
        "usual click train",
        "creak",
        "rapid click train",
    },
    "pongo-pygmaeus.yaml": {"nest smacks"},
    "rousettus-aegyptiacus.yaml": {"click sequence"},
    "tursiops-truncatus.yaml": {"echolocation click train", "terminal buzz"},
}


def _call_blocks(lines: list[str]) -> list[tuple[int, int, str]]:
    starts = []
    for index, line in enumerate(lines):
        match = re.match(r"^(\s*)- name:\s*(.+?)\s*$", line)
        if match:
            starts.append((index, len(match.group(1)), str(yaml.safe_load(match.group(2)))))
    blocks = []
    for offset, (start, indent, name) in enumerate(starts):
        end = len(lines)
        for later_start, later_indent, _ in starts[offset + 1 :]:
            if later_indent == indent:
                end = later_start
                break
        for index in range(start + 1, end):
            if re.match(r"^(references|provenance):\s*$", lines[index]):
                end = index
                break
        blocks.append((start, end, name))
    return blocks


def _remove_calls(lines: list[str], names: set[str]) -> tuple[list[str], list[str]]:
    removed = []
    for start, end, name in reversed(_call_blocks(lines)):
        if name in names:
            del lines[start:end]
            removed.append(name)
    return lines, sorted(removed)


def _append_provenance(lines: list[str], note: str) -> list[str]:
    if any(note in line for line in lines):
        return lines
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    provenance_index = next(
        index for index, line in enumerate(lines) if re.match(r"^provenance:\s*$", line)
    )
    section_end = next(
        (
            index
            for index in range(provenance_index + 1, len(lines))
            if re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*:\s*", lines[index])
        ),
        len(lines),
    )
    first_item = next(
        line
        for line in lines[provenance_index + 1 : section_end]
        if re.match(r"^\s*-\s", line)
    )
    item_indent = re.match(r"^(\s*)", first_item).group(1)
    field_indent = item_indent + "  "
    entry = [
        f"{item_indent}- timestamp: {json.dumps(timestamp)}\n",
        f'{field_indent}model: "GPT-5 Codex"\n',
        f"{field_indent}action: updated\n",
        f"{field_indent}notes: {json.dumps(note)}\n",
    ]
    lines[section_end:section_end] = entry
    return lines


def process(path: Path, *, write: bool) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    lines, removed = _remove_calls(lines, REMOVE_CALLS.get(path.name, set()))
    if removed and write:
        lines = _append_provenance(
            lines, "Removed explicit echolocation/non-vocal records and a duplicate call label."
        )
        path.write_text("".join(lines), encoding="utf-8")
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    removed = []
    for path in sorted(args.source.glob("*.yaml")):
        removed.extend(f"{path.stem}: {name}" for name in process(path, write=args.write))
    print(f"Calls removed: {len(removed)}")
    for item in removed:
        print(f"  - {item}")


if __name__ == "__main__":
    main()
