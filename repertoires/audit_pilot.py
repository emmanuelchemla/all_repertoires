#!/usr/bin/env python3
"""Run the 10-species repertoire verification pilot.

The script verifies cited reference metadata, tries to download legal open-access
PDFs, creates a manifest, and writes concise per-species audit reports. It does
not edit species YAML files.
"""

from __future__ import annotations

import datetime as dt
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
SPECIES_DIR = ROOT / "species"
AUDIT_DIR = ROOT / "audit"
SPECIES_AUDIT_DIR = AUDIT_DIR / "species"
PAPERS_OA_DIR = ROOT / "papers" / "oa"
PAPERS_MANUAL_DIR = ROOT / "papers" / "manual"

PILOT_SPECIES = [
    "pongo-pygmaeus",
    "cercopithecus-neglectus",
    "globicephala-melas",
    "saccopteryx-bilineata",
    "corvus-brachyrhynchos",
    "leptonychotes-weddellii",
    "forpus-passerinus",
    "hyla-arborea",
    "phyllostomus-discolor",
    "rousettus-aegyptiacus",
]

KNOWN_OA_PDF_URLS = {
    # OpenAlex/Crossref do not always expose author-hosted or society archive PDFs.
    "berg_etal_2011": "https://www.utrgv.edu/avianecology/_files/documents/publications/2012-berg-etal.pdf",
    "chamberlain_auger_1990": "https://digitalcommons.usf.edu/cgi/viewcontent.cgi?article=9512&context=wilson_bulletin",
    "knornschild_2014": "https://mirjam-knoernschild.org/wp-content/uploads/2012/06/Knoernschild-2014.pdf",
    "yorzinski_etal_2006": "https://yorzinskilab.org/wp-content/uploads/2021/02/yorzinski_etal_2006.pdf",
}

USER_AGENT = "repertoire-audit-pilot/0.1 (mailto:erossi@example.invalid)"
REQUEST_TIMEOUT = 25
SLEEP_SECONDS = 0.15


def normalize_doi(value: str | None) -> str | None:
    if not value:
        return None
    doi = value.strip()
    doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", doi, flags=re.I)
    doi = re.sub(r"^doi:\s*", "", doi, flags=re.I)
    return doi.strip().rstrip(".") or None


def request(url: str, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": accept},
    )
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
        return response.read()


def fetch_json(url: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        time.sleep(SLEEP_SECONDS)
        return json.loads(request(url).decode("utf-8")), None
    except Exception as exc:  # noqa: BLE001 - report exact retrieval failure.
        return None, f"{type(exc).__name__}: {exc}"


def crossref_by_doi(doi: str) -> tuple[dict[str, Any] | None, str | None]:
    encoded = urllib.parse.quote(doi, safe="")
    payload, error = fetch_json(f"https://api.crossref.org/works/{encoded}")
    if error or not payload:
        return None, error
    return payload.get("message"), None


def crossref_search(citation: str) -> tuple[dict[str, Any] | None, str | None]:
    query = urllib.parse.urlencode({"query.bibliographic": citation, "rows": "1"})
    payload, error = fetch_json(f"https://api.crossref.org/works?{query}")
    if error or not payload:
        return None, error
    items = payload.get("message", {}).get("items", [])
    return (items[0] if items else None), None


def openalex_by_doi(doi: str) -> tuple[dict[str, Any] | None, str | None]:
    encoded_doi_url = urllib.parse.quote(f"https://doi.org/{doi}", safe="")
    payload, error = fetch_json(f"https://api.openalex.org/works/{encoded_doi_url}")
    return payload, error


def openalex_search(query: str, limit: int = 5) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {"search": query, "per-page": str(limit), "sort": "cited_by_count:desc"}
    )
    payload, error = fetch_json(f"https://api.openalex.org/works?{params}")
    if error or not payload:
        return []
    return payload.get("results", [])[:limit]


def first_year(item: dict[str, Any] | None) -> int | None:
    if not item:
        return None
    year = item.get("published-print", {}).get("date-parts", [[None]])[0][0]
    year = year or item.get("published-online", {}).get("date-parts", [[None]])[0][0]
    year = year or item.get("issued", {}).get("date-parts", [[None]])[0][0]
    year = year or item.get("publication_year")
    return year


def title_from_crossref(item: dict[str, Any] | None) -> str | None:
    if not item:
        return None
    title = item.get("title") or []
    return title[0] if title else None


def container_from_crossref(item: dict[str, Any] | None) -> str | None:
    if not item:
        return None
    container = item.get("container-title") or []
    return container[0] if container else None


def authors_from_crossref(item: dict[str, Any] | None) -> list[str]:
    authors = []
    for author in (item or {}).get("author", [])[:8]:
        family = author.get("family")
        given = author.get("given")
        if family and given:
            authors.append(f"{family}, {given}")
        elif family:
            authors.append(family)
    return authors


def title_similarity(title: str | None, citation: str) -> float | None:
    if not title:
        return None
    return round(difflib.SequenceMatcher(None, title.lower(), citation.lower()).ratio(), 3)


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", (value or "").lower())).strip()


def title_is_in_citation(title: str | None, citation: str) -> bool:
    title_norm = normalize_text(title)
    citation_norm = normalize_text(citation)
    if not title_norm:
        return False
    return title_norm in citation_norm


def metadata_status(ref: dict[str, Any], item: dict[str, Any] | None) -> tuple[str, list[str]]:
    issues: list[str] = []
    if not item:
        return "not_verified", ["No metadata record found."]
    citation = ref.get("citation", "")
    title = title_from_crossref(item) or item.get("title")
    year = first_year(item)
    if not title_is_in_citation(title, citation):
        similarity = title_similarity(title, citation)
        if similarity is not None and similarity < 0.25:
            issues.append(f"Metadata title weakly matches citation text (similarity={similarity}).")
    if year and str(year) not in citation:
        citation_years = [int(match) for match in re.findall(r"\b(19\d{2}|20\d{2})\b", citation)]
        if not any(abs(year - citation_year) <= 1 for citation_year in citation_years):
            issues.append(f"Metadata year {year} is not present in citation.")
    if ref.get("doi"):
        returned_doi = normalize_doi(item.get("DOI") or item.get("doi"))
        expected_doi = normalize_doi(ref.get("doi"))
        if returned_doi and expected_doi and returned_doi.lower() != expected_doi.lower():
            issues.append(f"Returned DOI {returned_doi} differs from YAML DOI {expected_doi}.")
    return ("verified" if not issues else "needs_review"), issues


def pdf_candidates(ref_id: str, ref: dict[str, Any], openalex: dict[str, Any] | None) -> list[str]:
    candidates: list[str] = []
    if ref_id in KNOWN_OA_PDF_URLS:
        candidates.append(KNOWN_OA_PDF_URLS[ref_id])
    url = ref.get("url")
    if url and ".pdf" in url.lower():
        candidates.append(url)
    if url and "arxiv.org/abs/" in url.lower():
        candidates.append(url.replace("/abs/", "/pdf/") + ".pdf")
    if openalex:
        for location in [openalex.get("best_oa_location")] + (openalex.get("locations") or []):
            if not location:
                continue
            pdf_url = location.get("pdf_url")
            if pdf_url:
                candidates.append(pdf_url)
    seen = set()
    unique = []
    for candidate in candidates:
        if candidate not in seen:
            unique.append(candidate)
            seen.add(candidate)
    return unique


def download_pdf(urls: list[str], output: Path) -> tuple[str, str | None]:
    if output.exists() and output.stat().st_size > 5 and output.read_bytes()[:5] == b"%PDF-":
        return "downloaded", None
    errors = []
    for url in urls:
        try:
            data = request(url, accept="application/pdf,*/*")
            if not data.startswith(b"%PDF-"):
                errors.append(f"{url}: response was not a PDF")
                continue
            output.write_bytes(data)
            return "downloaded", None
        except Exception as exc:  # noqa: BLE001 - keep per-URL failure details.
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    if urls:
        return "download_failed", "; ".join(errors)
    return "no_oa_pdf_found", "No legal OA PDF URL found from DOI/OpenAlex/direct URL checks."


def pdf_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = path.read_bytes()
    info = {
        "path": str(path.relative_to(ROOT)),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "valid_pdf_header": data.startswith(b"%PDF-"),
    }
    pdftotext = shutil.which("pdftotext")
    if pdftotext:
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "out.txt"
            proc = subprocess.run(
                [pdftotext, "-enc", "UTF-8", str(path), str(text_path)],
                text=True,
                capture_output=True,
                check=False,
            )
            info["text_extractable"] = proc.returncode == 0 and text_path.exists()
            info["text_characters"] = len(text_path.read_text(errors="ignore")) if text_path.exists() else 0
            if proc.returncode != 0:
                info["text_extraction_error"] = (proc.stderr or proc.stdout).strip()[:300]
    return info


def load_species(slug: str) -> dict[str, Any]:
    return yaml.safe_load((SPECIES_DIR / f"{slug}.yaml").read_text())


def call_terms(call: dict[str, Any]) -> list[str]:
    terms = [call.get("name", "")]
    terms.extend(call.get("alternative_names") or [])
    return [term for term in terms if term]


def primary_call_presence(species: dict[str, Any], manifest_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    primary_id = (species.get("primary_inventory") or {}).get("id")
    if not primary_id:
        return []
    entry = manifest_by_id.get(primary_id, {})
    pdf_path = entry.get("pdf", {}).get("path")
    if not pdf_path:
        return []
    absolute_pdf = ROOT / pdf_path
    pdftotext = shutil.which("pdftotext")
    if not pdftotext or not absolute_pdf.exists():
        return []
    with tempfile.TemporaryDirectory() as tmp:
        text_path = Path(tmp) / "primary.txt"
        subprocess.run([pdftotext, "-enc", "UTF-8", str(absolute_pdf), str(text_path)], check=False)
        text = text_path.read_text(errors="ignore").lower() if text_path.exists() else ""
    presence = []
    for call in species.get("calls") or []:
        terms = call_terms(call)
        matched = [term for term in terms if term.lower() in text]
        presence.append(
            {
                "name": call.get("name"),
                "in_primary_inventory": call.get("in_primary_inventory"),
                "matched_terms_in_primary_pdf": matched,
                "status": "term_found" if matched else "term_not_found",
            }
        )
    return presence


def candidate_spine_papers(species: dict[str, Any]) -> list[dict[str, Any]]:
    scientific = species["scientific_name"]
    common = species["common_name"]
    queries = [
        f"{scientific} vocal repertoire",
        f"{scientific} acoustic repertoire",
        f"{scientific} vocalization calls",
        f"{common} vocal repertoire",
    ]
    candidates: dict[str, dict[str, Any]] = {}
    species_tokens = {
        token
        for token in normalize_text(f"{scientific} {common}").split()
        if len(token) > 3 and token not in {"monkey", "frog", "crow", "parrotlet", "spear", "nosed"}
    }
    for query in queries:
        for item in openalex_search(query, limit=3):
            key = item.get("doi") or item.get("id") or item.get("title")
            if not key or key in candidates:
                continue
            title = item.get("title")
            if not title:
                continue
            title_tokens = set(normalize_text(title).split())
            abstract_index = item.get("abstract_inverted_index") or {}
            abstract_tokens = {normalize_text(token) for token in abstract_index.keys()}
            if not (species_tokens & title_tokens or len(species_tokens & abstract_tokens) >= 2):
                continue
            candidates[key] = {
                "title": title,
                "year": item.get("publication_year"),
                "doi": item.get("doi"),
                "cited_by_count": item.get("cited_by_count"),
                "openalex_id": item.get("id"),
                "source": ((item.get("primary_location") or {}).get("source") or {}).get("display_name"),
            }
    return list(candidates.values())[:8]


def write_markdown_report(
    slug: str,
    species: dict[str, Any],
    ref_entries: list[dict[str, Any]],
    manual_entries: list[dict[str, Any]],
    primary_presence: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> None:
    primary_id = (species.get("primary_inventory") or {}).get("id")
    lines = [
        f"# {species['common_name']} ({species['scientific_name']})",
        "",
        f"- File: `species/{slug}.yaml`",
        f"- Primary inventory: `{primary_id or 'NONE'}`",
        f"- Calls in YAML: {len(species.get('calls') or [])}",
        f"- References in YAML: {len(species.get('references') or {})}",
        "",
        "## Reference Verification",
        "",
    ]
    for entry in ref_entries:
        issues = entry.get("metadata_issues") or []
        lines.append(
            f"- `{entry['reference_id']}`: metadata `{entry['metadata_status']}`, "
            f"access `{entry['access_status']}`"
        )
        title = entry.get("metadata", {}).get("title")
        if title:
            lines.append(f"  - Metadata title: {title}")
        if issues:
            lines.append(f"  - Issues: {'; '.join(issues)}")
        if entry.get("access_error"):
            lines.append(f"  - Access note: {entry['access_error']}")
    lines += ["", "## Primary Inventory Coverage", ""]
    if primary_id:
        if primary_presence:
            for item in primary_presence:
                matched = ", ".join(item["matched_terms_in_primary_pdf"]) or "none"
                lines.append(
                    f"- `{item['name']}` (`in_primary_inventory={item['in_primary_inventory']}`): "
                    f"{item['status']} (matched: {matched})"
                )
            lines.append(
                "\nManual follow-up: inspect the primary paper tables/typology for named call categories "
                "that are not already represented by the YAML call names or alternatives."
            )
        else:
            lines.append(
                "- Primary inventory call coverage could not be text-checked because the primary PDF was not downloaded or was not extractable."
            )
    else:
        lines.append("- No `primary_inventory.id` is present in the YAML.")
        lines.append("- Candidate spine papers from OpenAlex search:")
        if candidates:
            for candidate in candidates:
                doi = candidate.get("doi") or "no DOI"
                source = candidate.get("source") or "unknown source"
                lines.append(
                    f"  - {candidate.get('title')} ({candidate.get('year')}, {source}, {doi})"
                )
        else:
            lines.append("  - No strong candidate found by the automated search.")
    lines += ["", "## Access Gaps", ""]
    relevant_manual = [entry for entry in manual_entries if entry["species"] == slug]
    if relevant_manual:
        for entry in relevant_manual:
            lines.append(f"- `{entry['reference_id']}`: {entry['reason']}")
    else:
        lines.append("- No manual PDFs needed from this automated pass.")
    lines.append("")
    (SPECIES_AUDIT_DIR / f"{slug}.md").write_text("\n".join(lines))


def main() -> int:
    AUDIT_DIR.mkdir(exist_ok=True)
    SPECIES_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_OA_DIR.mkdir(parents=True, exist_ok=True)
    PAPERS_MANUAL_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pilot_species": PILOT_SPECIES,
        "references": [],
    }
    manual_needed: list[dict[str, Any]] = []
    species_entries: dict[str, list[dict[str, Any]]] = {slug: [] for slug in PILOT_SPECIES}

    seen_reference_species: dict[str, set[str]] = {}
    for slug in PILOT_SPECIES:
        species = load_species(slug)
        for ref_id, ref in (species.get("references") or {}).items():
            seen_reference_species.setdefault(ref_id, set()).add(slug)

            doi = normalize_doi(ref.get("doi") or ref.get("url"))
            metadata = None
            metadata_error = None
            openalex = None
            openalex_error = None
            if doi:
                metadata, metadata_error = crossref_by_doi(doi)
                openalex, openalex_error = openalex_by_doi(doi)
            if not metadata:
                metadata, metadata_error = crossref_search(ref.get("citation", ""))

            status, metadata_issues = metadata_status(ref, metadata)
            pdf_urls = pdf_candidates(ref_id, ref, openalex)
            pdf_path = PAPERS_OA_DIR / f"{ref_id}.pdf"
            access_status, access_error = download_pdf(pdf_urls, pdf_path)
            pdf = pdf_info(pdf_path) if access_status == "downloaded" else {}

            entry = {
                "species": slug,
                "reference_id": ref_id,
                "citation": ref.get("citation"),
                "doi": doi,
                "url": ref.get("url"),
                "metadata_status": status,
                "metadata_issues": metadata_issues,
                "metadata_error": metadata_error,
                "openalex_error": openalex_error,
                "metadata": {
                    "title": title_from_crossref(metadata) or (metadata or {}).get("title"),
                    "year": first_year(metadata),
                    "container": container_from_crossref(metadata),
                    "authors": authors_from_crossref(metadata),
                    "doi": normalize_doi((metadata or {}).get("DOI") or (metadata or {}).get("doi")),
                },
                "oa_pdf_candidates": pdf_urls,
                "access_status": access_status,
                "access_error": access_error,
                "pdf": pdf,
            }
            if access_status != "downloaded":
                manual_needed.append(
                    {
                        "species": slug,
                        "reference_id": ref_id,
                        "citation": ref.get("citation"),
                        "doi": doi,
                        "url": ref.get("url"),
                        "reason": access_error or access_status,
                    }
                )
            manifest["references"].append(entry)
            species_entries[slug].append(entry)

    manifest_by_ref = {entry["reference_id"]: entry for entry in manifest["references"]}
    all_candidate_spines: dict[str, list[dict[str, Any]]] = {}
    for slug in PILOT_SPECIES:
        species = load_species(slug)
        candidates = [] if (species.get("primary_inventory") or {}).get("id") else candidate_spine_papers(species)
        all_candidate_spines[slug] = candidates
        write_markdown_report(
            slug=slug,
            species=species,
            ref_entries=species_entries[slug],
            manual_entries=manual_needed,
            primary_presence=primary_call_presence(species, manifest_by_ref),
            candidates=candidates,
        )

    (AUDIT_DIR / "reference_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))
    (AUDIT_DIR / "manual_pdf_needed.yaml").write_text(
        yaml.safe_dump(
            {
                "generated_at": manifest["generated_at"],
                "instructions": "Place manually supplied PDFs in papers/manual/<reference_id>.pdf.",
                "references": manual_needed,
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )
    (AUDIT_DIR / "candidate_spine_papers.yaml").write_text(
        yaml.safe_dump(
            {"generated_at": manifest["generated_at"], "species": all_candidate_spines},
            sort_keys=False,
            allow_unicode=True,
        )
    )
    (PAPERS_MANUAL_DIR / "README.md").write_text(
        "# Manual PDFs\n\n"
        "Place paywalled or otherwise inaccessible full-text PDFs here using the reference ID as the filename, "
        "for example `hardus_etal_2009.pdf`.\n\n"
        "The audit script expects manually supplied files at `papers/manual/<reference_id>.pdf`.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
