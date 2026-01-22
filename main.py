"""
Build a CSV database of vocal call repertoires for a list of species using an LLM.

Requirements:
- OpenAI: set OPENAI_API_KEY and pass --provider openai --model gpt-4o (default).
- Gemini: install google-generativeai, set GEMINI_API_KEY, and pass --provider gemini.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class CallRecord:
    species: str
    usual_name: str
    acoustic_description: str
    context_description: str
    references: str
    demographic_notes: str
    reliability: str
    provider: str
    model: str


def build_prompt(species: str) -> str:
    """Prompt that coerces the LLM to return a structured repertoire for one species."""
    return f"""You are a bioacoustics expert who writes concise factual summaries.
Provide the vocal call repertoire for the species "{species}".
Return ONLY a single JSON object with this exact shape:
{{
  "species": "{species}",
  "calls": [
    {{
      "usual_name": "...",
      "acoustic_description": "...",
      "context_description": "...",
      "references": ["...", "..."],
      "demographic_notes": "...",
      "reliability": {{
        "level": "high|medium|low",
        "explanation": "..."
      }}
    }}
  ]
}}
Rules:
- Include as many distinct calls as are well-documented.
- If a field is unknown, use "unknown".
- references should be short citations or URLs when possible.
"""


def _require_env(var_name: str) -> str:
    value = os.environ.get(var_name)
    if not value:
        raise RuntimeError(
            f"Environment variable {var_name} is required for this provider."
        )
    return value


def call_openai(prompt: str, model: str, temperature: float) -> str:
    from openai import OpenAI

    api_key = _require_env("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You return compact JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    return completion.choices[0].message.content


def call_gemini(prompt: str, model: str, temperature: float) -> str:
    try:
        import google.generativeai as genai
    except ImportError as exc:
        raise RuntimeError(
            "google-generativeai is required for Gemini. Install with `pip install google-generativeai`."
        ) from exc

    api_key = _require_env("GEMINI_API_KEY")
    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model)
    response = gemini_model.generate_content(
        prompt,
        generation_config={
            "temperature": temperature,
            "response_mime_type": "application/json",
        },
    )
    return response.text


def extract_json_blob(text: str) -> str:
    """Extract the first JSON object or array from arbitrary text."""
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    if not match:
        raise ValueError("No JSON object or array found in the model response.")
    return match.group(1)


def parse_calls(
    species: str, content: str, provider: str, model: str
) -> List[CallRecord]:
    json_payload = extract_json_blob(content)
    data = json.loads(json_payload)
    calls: Sequence[dict] = data.get("calls", [])
    records: List[CallRecord] = []

    for call in calls:
        references = call.get("references") or []
        if isinstance(references, list):
            references_str = "; ".join(references)
        else:
            references_str = str(references)

        reliability = call.get("reliability") or {}
        if isinstance(reliability, dict):
            rel_level = reliability.get("level", "unknown")
            rel_note = reliability.get("explanation", "")
            reliability_str = f"{rel_level}: {rel_note}".strip(": ")
        else:
            reliability_str = str(reliability)

        records.append(
            CallRecord(
                species=data.get("species", species),
                usual_name=call.get("usual_name", "unknown"),
                acoustic_description=call.get("acoustic_description", "unknown"),
                context_description=call.get("context_description", "unknown"),
                references=references_str or "unknown",
                demographic_notes=call.get("demographic_notes", "unknown"),
                reliability=reliability_str or "unknown",
                provider=provider,
                model=model,
            )
        )
    return records


def write_csv(records: Iterable[CallRecord], out_path: str) -> None:
    fieldnames = [
        "species",
        "usual_name",
        "acoustic_description",
        "context_description",
        "references",
        "demographic_notes",
        "reliability",
        "provider",
        "model",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.__dict__)


def read_species(args: argparse.Namespace) -> List[str]:
    species: List[str] = []
    if args.species:
        species.extend(args.species)
    if args.species_file:
        with open(args.species_file, "r", encoding="utf-8") as f:
            species.extend(line.strip() for line in f if line.strip())
    deduped = sorted(set(species))
    if not deduped:
        raise ValueError(
            "Provide at least one species via --species or --species-file."
        )
    return deduped


def query_provider(provider: str, prompt: str, model: str, temperature: float) -> str:
    if provider == "openai":
        return call_openai(prompt, model, temperature)
    if provider == "gemini":
        return call_gemini(prompt, model, temperature)
    raise ValueError(f"Unsupported provider: {provider}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build vocal call repertoire CSV via LLM."
    )
    parser.add_argument(
        "--species", nargs="*", help="Species names (e.g., 'Pan troglodytes')."
    )
    parser.add_argument(
        "--species-file", help="Path to a text file with one species per line."
    )
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument(
        "--model", default="gpt-4o", help="Model name for the chosen provider."
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--out", default="call_repertoire.csv", help="Output CSV path.")
    args = parser.parse_args()

    species_list = read_species(args)
    all_records: List[CallRecord] = []

    for species in species_list:
        prompt = build_prompt(species)
        content = query_provider(args.provider, prompt, args.model, args.temperature)
        records = parse_calls(species, content, args.provider, args.model)
        all_records.extend(records)

    write_csv(all_records, args.out)
    print(f"Wrote {len(all_records)} rows to {args.out}")


if __name__ == "__main__":
    main()
