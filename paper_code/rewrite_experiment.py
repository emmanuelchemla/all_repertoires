"""
Experiment: rewrite new-database descriptions to old-database style,
then test whether Mantel r improves.

Steps:
1. Load 33-species subset (>=6 calls in new DB)
2. Rewrite descriptions using Claude Haiku via Anthropic SDK
3. Save rewritten descriptions to paper_code/rewritten_descriptions.json
4. Run Mantel test on original vs rewritten descriptions
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sentence_transformers import SentenceTransformer

from paper_code.data_sources import load_calls
from paper_code.mantel import (
    EMBEDDING_MODEL,
    bootstrap_ci,
    embed_texts,
    run_mantel_subset,
    similarity_matrix,
)

CACHE_PATH = ROOT / ".embedding_cache.json"
REWRITTEN_PATH = ROOT / "paper_code" / "rewritten_descriptions.json"

SYSTEM_PROMPT = """\
You are a comparative biologist writing concise, function-focused descriptions of animal vocalizations.
Your descriptions should be SHORT (under 100 characters each) and focus on:
- Acoustic: key physical features (frequency, duration, tonality, structure)
- Semantic: function and behavioral context

Style examples:
Acoustic: "High-frequency, narrowband tonal call that is difficult for predators to localize."
Acoustic: "Short, tonal call exchanged between nearby flock members."
Semantic: "Aerial predator alarm eliciting freezing or cover-seeking behavior."
Semantic: "Short-range contact and group cohesion during foraging."

Return ONLY valid JSON, no extra text."""


def build_prompt(species: str, calls: list[dict]) -> str:
    calls_block = []
    for i, c in enumerate(calls):
        calls_block.append(
            f"Call {i+1}: \"{c['call_name']}\"\n"
            f"  Current acoustic: {c['acoustic_description'][:300]}\n"
            f"  Current semantic: {c['semantic_description'][:300]}\n"
            f"  Semantic keywords: {', '.join(c['semantic_keywords'][:5])}"
        )
    calls_text = "\n\n".join(calls_block)

    return (
        f"Rewrite the following {len(calls)} call descriptions for {species}.\n"
        f"For each call, produce a concise, function-focused acoustic and semantic description "
        f"(under 100 characters each).\n\n"
        f"{calls_text}\n\n"
        f"Return a JSON array with one object per call, in order:\n"
        f'[{{"call_name": "...", "acoustic": "...", "semantic": "..."}}, ...]\n\n'
        f"Return ONLY the JSON array, no extra text."
    )


def rewrite_species_via_cli(species: str, calls: list[dict]) -> list[dict]:
    """Rewrite all calls for one species via the claude CLI subprocess."""
    import subprocess

    prompt = (
        SYSTEM_PROMPT + "\n\n" + build_prompt(species, calls)
    )

    for attempt in range(3):
        try:
            result = subprocess.run(
                ["claude", "-p", prompt],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode != 0:
                raise RuntimeError(f"claude CLI error: {result.stderr[:200]}")
            text = result.stdout.strip()
            # Strip markdown code blocks if present
            if "```" in text:
                # find content between first ``` and last ```
                parts = text.split("```")
                for part in parts[1::2]:  # odd-indexed parts are code blocks
                    if part.startswith("json"):
                        part = part[4:]
                    text = part.strip()
                    break
            result_json = json.loads(text)
            if isinstance(result_json, list) and len(result_json) == len(calls):
                return result_json
            print(f"  Warning: got {len(result_json)} results for {len(calls)} calls, retrying...")
        except (json.JSONDecodeError, IndexError, KeyError, RuntimeError) as e:
            print(f"  Error on attempt {attempt+1}: {e}")
            if attempt == 2:
                raise
        time.sleep(2)
    raise RuntimeError(f"Failed to rewrite calls for {species}")


def load_or_build_rewritten(calls: list[dict]) -> dict[str, dict]:
    """Load cached rewrites or call via claude CLI to generate them."""
    if REWRITTEN_PATH.exists():
        with open(REWRITTEN_PATH) as f:
            data = json.load(f)
        # Check if rewrites are different from originals (not just fallbacks)
        # by checking description lengths
        sample_key = list(data.keys())[0] if data else None
        if sample_key:
            rew_len = len(data[sample_key].get("acoustic", ""))
            orig = next((c for c in calls if f"{c['species']}||{c['call_name']}" == sample_key), None)
            if orig and abs(rew_len - len(str(orig["acoustic_description"]))) < 5:
                print(f"Existing rewrites at {REWRITTEN_PATH} appear to be fallbacks (same length as originals). Regenerating...")
            else:
                print(f"Loading existing rewrites from {REWRITTEN_PATH}")
                return data

    print("Rewriting descriptions via claude CLI...")

    # Group by species
    by_species: dict[str, list[dict]] = {}
    for c in calls:
        by_species.setdefault(c["species"], []).append(c)

    rewrites: dict[str, dict] = {}

    for i, (species, sp_calls) in enumerate(sorted(by_species.items())):
        print(f"  [{i+1}/{len(by_species)}] {species} ({len(sp_calls)} calls)...", end="", flush=True)
        try:
            results = rewrite_species_via_cli(species, sp_calls)
            for call, rewritten in zip(sp_calls, results):
                key = f"{species}||{call['call_name']}"
                rewrites[key] = {
                    "acoustic": rewritten.get("acoustic", ""),
                    "semantic": rewritten.get("semantic", ""),
                }
            print(f" ok (rewritten)")
        except Exception as e:
            print(f" ERROR: {e}")
            # Fallback: use originals
            for call in sp_calls:
                key = f"{species}||{call['call_name']}"
                rewrites[key] = {
                    "acoustic": str(call["acoustic_description"]),
                    "semantic": str(call["semantic_description"]),
                }
        time.sleep(0.5)

    with open(REWRITTEN_PATH, "w") as f:
        json.dump(rewrites, f, indent=2)
    print(f"Saved rewrites to {REWRITTEN_PATH}")

    return rewrites


def run_mantel_analysis(calls: list[dict], acoustic_texts: list[str], semantic_texts: list[str],
                        label: str, encoder: SentenceTransformer) -> dict:
    """Embed texts and run Mantel test suite."""
    print(f"\n--- {label} ---")

    ac_emb, _ = embed_texts(acoustic_texts, EMBEDDING_MODEL, CACHE_PATH, encoder)
    se_emb, _ = embed_texts(semantic_texts, EMBEDDING_MODEL, CACHE_PATH, encoder)

    Sa = similarity_matrix(ac_emb)
    Ss = similarity_matrix(se_emb)
    n = len(calls)

    sp_arr = np.array([c["species"] for c in calls])
    fa_arr = np.array([c["family"] for c in calls])

    # All pairs
    mask_all = np.ones((n, n), dtype=bool)
    r_all, p_all, k_all, av_all, sv_all = run_mantel_subset(Sa, Ss, mask_all)
    ci_all = bootstrap_ci(av_all, sv_all)

    # Within-species
    mask_within = sp_arr[:, None] == sp_arr[None, :]
    r_within, p_within, k_within, av_within, sv_within = run_mantel_subset(Sa, Ss, mask_within)
    ci_within = bootstrap_ci(av_within, sv_within)

    # Same family cross-species
    mask_fam = (fa_arr[:, None] == fa_arr[None, :]) & (sp_arr[:, None] != sp_arr[None, :])
    r_fam, p_fam, k_fam, av_fam, sv_fam = run_mantel_subset(Sa, Ss, mask_fam)
    ci_fam = bootstrap_ci(av_fam, sv_fam)

    results = {
        "all_pairs": {"r": r_all, "p": p_all, "n_pairs": k_all, "ci_lo": ci_all[0], "ci_hi": ci_all[1]},
        "within_species": {"r": r_within, "p": p_within, "n_pairs": k_within, "ci_lo": ci_within[0], "ci_hi": ci_within[1]},
        "same_family_cross_species": {"r": r_fam, "p": p_fam, "n_pairs": k_fam, "ci_lo": ci_fam[0], "ci_hi": ci_fam[1]},
    }

    print(f"  {'Subset':<35s}  {'r':>7s}  {'95% CI':>16s}  {'p':>8s}  {'n_pairs':>8s}")
    for subset, v in results.items():
        r, p, k = v["r"], v["p"], v["n_pairs"]
        ci = f"[{v['ci_lo']:+.3f}, {v['ci_hi']:+.3f}]"
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        print(f"  {subset:<35s}  {r:+.3f}  {ci:>16s}  {p:.4f} {sig:4s}  {k:>8d}")

    return results


def main():
    # 1. Load 33-species subset
    print("Loading new database calls...")
    calls = load_calls("new")
    sp_counts = Counter(c["species"] for c in calls)
    eligible = {s for s, n in sp_counts.items() if n >= 6}
    subset = [c for c in calls if c["species"] in eligible]

    print(f"Eligible species: {len(eligible)}")
    print(f"Subset calls: {len(subset)}")

    # 2. Load or generate rewrites
    rewrites = load_or_build_rewritten(subset)

    # 3. Check coverage
    original_ac = [str(c["acoustic_description"]) for c in subset]
    original_se = [str(c["semantic_description"]) for c in subset]

    rewritten_ac = []
    rewritten_se = []
    missing = 0
    for c in subset:
        key = f"{c['species']}||{c['call_name']}"
        if key in rewrites:
            rewritten_ac.append(rewrites[key]["acoustic"])
            rewritten_se.append(rewrites[key]["semantic"])
        else:
            missing += 1
            rewritten_ac.append(c["acoustic_description"])
            rewritten_se.append(c["semantic_description"])

    if missing:
        print(f"Warning: {missing} calls not found in rewrites, using originals")

    # Show description length comparison
    orig_ac_lens = [len(t) for t in original_ac]
    orig_se_lens = [len(t) for t in original_se]
    rew_ac_lens = [len(t) for t in rewritten_ac]
    rew_se_lens = [len(t) for t in rewritten_se]

    print(f"\nDescription length comparison:")
    print(f"  Original acoustic:  mean={sum(orig_ac_lens)/len(orig_ac_lens):.0f}, max={max(orig_ac_lens)}")
    print(f"  Rewritten acoustic: mean={sum(rew_ac_lens)/len(rew_ac_lens):.0f}, max={max(rew_ac_lens)}")
    print(f"  Original semantic:  mean={sum(orig_se_lens)/len(orig_se_lens):.0f}, max={max(orig_se_lens)}")
    print(f"  Rewritten semantic: mean={sum(rew_se_lens)/len(rew_se_lens):.0f}, max={max(rew_se_lens)}")

    # Show a sample rewrite
    print("\nSample rewrites (first 3 calls):")
    for i, c in enumerate(subset[:3]):
        key = f"{c['species']}||{c['call_name']}"
        print(f"\n  [{c['species']} / {c['call_name']}]")
        print(f"  ORIG acoustic: {c['acoustic_description'][:120]}...")
        print(f"  NEW  acoustic: {rewrites.get(key, {}).get('acoustic', 'N/A')}")
        print(f"  ORIG semantic: {c['semantic_description'][:120]}...")
        print(f"  NEW  semantic: {rewrites.get(key, {}).get('semantic', 'N/A')}")

    # 4. Load encoder and run Mantel tests
    print(f"\nLoading sentence encoder: {EMBEDDING_MODEL}")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    results_original = run_mantel_analysis(
        subset, original_ac, original_se,
        "ORIGINAL descriptions (new DB style)", encoder
    )

    results_rewritten = run_mantel_analysis(
        subset, rewritten_ac, rewritten_se,
        "REWRITTEN descriptions (old DB style)", encoder
    )

    # 5. Summary comparison
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (33-species subset, >=6 calls)")
    print("=" * 80)
    print(f"{'Subset':<35s}  {'Original r':>10s}  {'Rewritten r':>11s}  {'Delta':>8s}")
    print("-" * 68)
    for subset_name in ["all_pairs", "within_species", "same_family_cross_species"]:
        r_orig = results_original[subset_name]["r"]
        r_rew = results_rewritten[subset_name]["r"]
        delta = r_rew - r_orig
        direction = "↑" if delta > 0 else "↓"
        print(f"  {subset_name:<33s}  {r_orig:+.4f}      {r_rew:+.4f}      {delta:+.4f} {direction}")

    # Also compare to full new DB results
    try:
        with open(ROOT / "paper_code" / "mantel_results_new.json") as f:
            full_new = json.load(f)
        print("\nFor reference — full new DB (all 633 calls, all species):")
        print(f"  all_pairs:          r = {full_new['all_pairs']['r']:+.4f}")
        print(f"  within_species:     r = {full_new['within_species (pooled)']['r']:+.4f}")
        print(f"  same_family:        r = {full_new['same_family_cross_species']['r']:+.4f}")
    except FileNotFoundError:
        pass

    # Save comparison
    comparison = {
        "n_calls": len(subset),
        "n_species": len(eligible),
        "original": results_original,
        "rewritten": results_rewritten,
    }
    out_path = ROOT / "paper_code" / "rewrite_experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
