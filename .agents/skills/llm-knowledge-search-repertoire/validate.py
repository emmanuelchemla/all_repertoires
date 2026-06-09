#!/usr/bin/env python3
"""Validate a search repertoire YAML file against schema.json.

Usage: python .agents/skills/llm-knowledge-search-repertoire/validate.py repertoires/llm_knowledge+search/species/<file>.yaml [more files...]
Exits 0 on success, 1 on any failure.
"""
import json
import sys
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).parent
SCHEMA = json.loads((ROOT / "schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def check(path: Path) -> list[str]:
    errors: list[str] = []
    data = yaml.safe_load(path.read_text())

    for err in VALIDATOR.iter_errors(data):
        loc = "/".join(str(p) for p in err.absolute_path) or "<root>"
        errors.append(f"{loc}: {err.message}")

    taxonomy = data.get("taxonomy") or {}
    if taxonomy.get("genus") and taxonomy.get("species") and data.get("scientific_name"):
        expected = f"{taxonomy['genus']} {taxonomy['species']}"
        if expected != data["scientific_name"]:
            errors.append(
                f"taxonomy: genus + species is '{expected}', expected scientific_name '{data['scientific_name']}'"
            )

    ref_ids = set((data.get("references") or {}).keys())
    for i, call in enumerate(data.get("calls") or []):
        scope = call.get("scope") or {}
        note = (scope.get("note") or "").strip()
        if scope.get("population_specific"):
            if not note:
                errors.append(f"calls[{i}].scope.note: required when population_specific is true")
        elif note:
            errors.append(f"calls[{i}].scope.note: must be empty when population_specific is false")

        for rid in call.get("references") or []:
            if rid not in ref_ids:
                errors.append(f"calls[{i}].references: unknown reference id '{rid}'")

    return errors


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    failed = False
    for arg in sys.argv[1:]:
        path = Path(arg)
        errs = check(path)
        if errs:
            failed = True
            print(f"FAIL {path}")
            for err in errs:
                print(f"  - {err}")
        else:
            print(f"OK   {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
