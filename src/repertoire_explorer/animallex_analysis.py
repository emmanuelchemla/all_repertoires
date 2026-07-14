from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import fisher_exact, linregress

from .datasets import CanonicalDataset


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RELATIONSHIP_GROUPS = ("within species", "same family", "different families")
CLASS_ORDER = {"Amphibia": 0, "Aves": 1, "Mammalia": 2}

SEMANTIC_GROUPS = {
    "social cohesion": {"contact", "group_coordination", "affiliation"},
    "agonistic": {"threat", "aggression", "submission"},
    "danger": {"alarm", "predator"},
    "distress and care": {"distress", "begging", "caregiving"},
    "reproduction": {"courtship", "mating"},
    "resources": {"food", "recruitment"},
    "territorial spacing": {"territorial", "spacing"},
    "identity and attention": {"identity", "attention"},
    "metacommunicative": {"play", "display"},
    "combinatorial": {"combinatorial"},
}
ACOUSTIC_GROUPS = {
    "frequency": {"high_frequency", "low_frequency", "frequency_modulated"},
    "spectral": {"tonal", "broadband", "noisy", "harmonic"},
    "temporal": {"short", "long", "abrupt", "repetitive", "pulsed", "multi_component"},
    "amplitude": {"loud", "quiet"},
    "variation": {"graded"},
}


@dataclass(frozen=True)
class AnalysisConfig:
    analysis_version: str = "3"
    embedding_model: str = EMBEDDING_MODEL
    random_seed: int = 42
    n_permutations: int = 9999
    scatter_sample_per_group: int = 1000
    species_pair_minimum: int = 6
    pmi_minimum: int = 4
    pmi_alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _keyword_group(keyword: str, groups: dict[str, set[str]]) -> str:
    return next((name for name, values in groups.items() if keyword in values), "other")


def _taxonomic_group(class_name: str, order: str, family: str) -> str:
    if family in {"Hominidae", "Hylobatidae"}:
        return "Apes and gibbons"
    if family == "Cercopithecidae":
        return "Old World monkeys"
    if order == "Primates":
        return "Other primates"
    if order == "Carnivora":
        return "Carnivorans"
    if order == "Rodentia":
        return "Rodents"
    if order == "Chiroptera":
        return "Bats"
    if order in {"Artiodactyla", "Perissodactyla"}:
        return "Ungulates"
    if class_name == "Mammalia":
        return "Other mammals"
    if order == "Passeriformes":
        return "Songbirds"
    if order == "Psittaciformes":
        return "Parrots"
    if class_name == "Aves":
        return "Other birds"
    if class_name == "Amphibia":
        return "Frogs"
    return "Other animals"


def compute_overview(dataset: CanonicalDataset) -> dict[str, Any]:
    calls = dataset.calls
    semantic = Counter(k for values in calls["semantic_keywords"] for k in values)
    acoustic = Counter(k for values in calls["acoustic_keywords"] for k in values)
    species_rows = []
    for species, frame in calls.groupby("species", sort=True):
        first = frame.iloc[0]
        common_name = dataset.species_metadata.get(species, {}).get("common_name", species)
        species_rows.append(
            {
                "species": species,
                "common_name": common_name,
                "n_calls": int(len(frame)),
                "class": str(first["class"]),
                "order": str(first["order"]),
                "family": str(first["family"]),
                "taxonomic_group": _taxonomic_group(
                    str(first["class"]), str(first["order"]), str(first["family"])
                ),
            }
        )
    group_order = {
        group: index
        for index, group in enumerate(
            [
                "Apes and gibbons",
                "Old World monkeys",
                "Other primates",
                "Carnivorans",
                "Rodents",
                "Bats",
                "Ungulates",
                "Other mammals",
                "Songbirds",
                "Parrots",
                "Other birds",
                "Frogs",
                "Other animals",
            ]
        )
    }
    call_count_distribution = Counter(row["n_calls"] for row in species_rows)
    return {
        "n_calls": int(len(calls)),
        "n_species": int(calls["species"].nunique()),
        "semantic_keywords": [
            {
                "keyword": key,
                "count": count,
                "percent_calls": 100 * count / len(calls),
                "group": _keyword_group(key, SEMANTIC_GROUPS),
            }
            for key, count in semantic.most_common()
        ],
        "acoustic_keywords": [
            {
                "keyword": key,
                "count": count,
                "percent_calls": 100 * count / len(calls),
                "group": _keyword_group(key, ACOUSTIC_GROUPS),
            }
            for key, count in acoustic.most_common()
        ],
        "call_count_distribution": [
            {"n_calls": n_calls, "n_species": n_species}
            for n_calls, n_species in sorted(call_count_distribution.items())
        ],
        "species_counts": sorted(
            species_rows,
            key=lambda row: (
                group_order[row["taxonomic_group"]],
                -row["n_calls"],
                row["species"],
            ),
        ),
    }


def _load_embedding_cache(path: Path, model: str) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("embeddings", {}) if payload.get("model") == model else {}


def compute_description_embeddings(
    texts: list[str],
    *,
    model_name: str = EMBEDDING_MODEL,
    cache_path: Path | str,
    encoder: Any | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Embed descriptions with an exact text cache shared by all consumers."""
    cache_path = Path(cache_path)
    cache = _load_embedding_cache(cache_path, model_name)
    missing = list(dict.fromkeys(text for text in texts if text not in cache))
    if missing:
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_name)
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            vectors = encoder.encode(
                chunk,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            for text, vector in zip(chunk, vectors):
                cache[text] = np.asarray(vector, dtype=float).tolist()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps({"model": model_name, "embeddings": cache}),
            encoding="utf-8",
        )
    return np.asarray([cache[text] for text in texts], dtype=np.float32)


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    return normalized @ normalized.T


def mantel(
    acoustic_values: np.ndarray,
    semantic_values: np.ndarray,
    *,
    n_perm: int = 9999,
    seed: int = 42,
) -> tuple[float, float]:
    """Preserve the permutation behavior used by the current paper code."""
    rng = np.random.default_rng(seed)
    observed = float(np.corrcoef(acoustic_values, semantic_values)[0, 1])
    count = 0
    for _ in range(n_perm):
        permutation = rng.permutation(len(semantic_values))
        permuted = float(np.corrcoef(acoustic_values, semantic_values[permutation])[0, 1])
        if permuted >= observed:
            count += 1
    return observed, (count + 1) / (n_perm + 1)


def compute_form_meaning_alignment(
    dataset: CanonicalDataset,
    acoustic_embeddings: np.ndarray,
    semantic_embeddings: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, Any]:
    ac_similarity = similarity_matrix(acoustic_embeddings)
    sem_similarity = similarity_matrix(semantic_embeddings)
    n = len(dataset.calls)
    left, right = np.triu_indices(n, k=1)
    species = dataset.calls["species"].astype(str).to_numpy()
    families = dataset.calls["family"].astype(str).to_numpy()
    labels = np.where(
        species[left] == species[right],
        "within species",
        np.where(families[left] == families[right], "same family", "different families"),
    )
    ac_distance = 1 - ac_similarity[left, right]
    sem_distance = 1 - sem_similarity[left, right]
    rng = np.random.default_rng(config.random_seed)
    groups: dict[str, Any] = {}
    for offset, group in enumerate(RELATIONSHIP_GROUPS):
        indices = np.flatnonzero(labels == group)
        if len(indices) < 2:
            groups[group] = {
                "r": None,
                "p": None,
                "n_pairs": int(len(indices)),
                "slope": None,
                "intercept": None,
                "sample": [],
            }
            continue
        sample = rng.choice(
            indices,
            size=min(config.scatter_sample_per_group, len(indices)),
            replace=False,
        )
        r_value, p_value = mantel(
            ac_distance[indices],
            sem_distance[indices],
            n_perm=config.n_permutations,
            seed=config.random_seed + offset,
        )
        regression = linregress(ac_distance[indices], sem_distance[indices])
        slope, intercept = regression.slope, regression.intercept
        groups[group] = {
            "r": r_value,
            "p": p_value,
            "n_pairs": int(len(indices)),
            "slope": float(slope),
            "intercept": float(intercept),
            "sample": [
                {
                    "acoustic_distance": float(ac_distance[index]),
                    "semantic_distance": float(sem_distance[index]),
                    "call_id_1": str(dataset.calls.iloc[left[index]]["call_id"]),
                    "call_id_2": str(dataset.calls.iloc[right[index]]["call_id"]),
                }
                for index in sample
            ],
        }
    return {"groups": groups}


def _reorder_within_classes(
    species: list[str], matrix: np.ndarray, classes: list[str]
) -> list[int]:
    order: list[int] = []
    start = 0
    while start < len(classes):
        end = start + 1
        while end < len(classes) and classes[end] == classes[start]:
            end += 1
        if end - start <= 2:
            order.extend(range(start, end))
        else:
            block = np.nan_to_num(matrix[start:end, :], nan=0.0)
            leaves = leaves_list(linkage(block, method="average", metric="euclidean"))
            order.extend(start + int(index) for index in leaves)
        start = end
    return order


def compute_species_pair_correlations(
    dataset: CanonicalDataset,
    acoustic_similarity: np.ndarray,
    semantic_similarity: np.ndarray,
    *,
    minimum_pairs: int = 6,
) -> dict[str, Any]:
    calls = dataset.calls
    classes_by_species = calls.groupby("species")["class"].first().to_dict()
    species = sorted(
        dataset.species,
        key=lambda value: (CLASS_ORDER.get(str(classes_by_species.get(value, "")), 9), value),
    )
    indices = {
        value: np.flatnonzero(calls["species"].astype(str).to_numpy() == value)
        for value in species
    }
    matrix = np.full((len(species), len(species)), np.nan)
    counts = np.zeros_like(matrix, dtype=int)
    for i, source in enumerate(species):
        for j in range(i + 1, len(species)):
            target = species[j]
            ac_values = acoustic_similarity[np.ix_(indices[source], indices[target])].ravel()
            sem_values = semantic_similarity[np.ix_(indices[source], indices[target])].ravel()
            counts[i, j] = counts[j, i] = len(ac_values)
            if len(ac_values) >= minimum_pairs and np.std(ac_values) and np.std(sem_values):
                matrix[i, j] = matrix[j, i] = np.corrcoef(ac_values, sem_values)[0, 1]
    classes = [str(classes_by_species.get(value, "")) for value in species]
    order = _reorder_within_classes(species, matrix, classes)
    return {
        "species": [species[index] for index in order],
        "classes": [classes[index] for index in order],
        "matrix": matrix[np.ix_(order, order)].tolist(),
        "pair_counts": counts[np.ix_(order, order)].tolist(),
    }


def compute_pairwise_r(
    calls: list[dict[str, Any]],
    acoustic_similarity: np.ndarray,
    semantic_similarity: np.ndarray,
    *,
    minimum_pairs: int = 6,
) -> tuple[list[str], np.ndarray, list[str]]:
    """Compatibility interface for the paper's species pair calculation."""
    species_values = np.array([str(call["species"]) for call in calls])
    classes_by_species = {str(call["species"]): str(call.get("class", "")) for call in calls}
    species = sorted(
        set(species_values),
        key=lambda value: (CLASS_ORDER.get(classes_by_species.get(value, ""), 9), value),
    )
    indices = {value: np.flatnonzero(species_values == value) for value in species}
    matrix = np.full((len(species), len(species)), np.nan)
    for i, source in enumerate(species):
        for j in range(i + 1, len(species)):
            target = species[j]
            ac_values = acoustic_similarity[np.ix_(indices[source], indices[target])].ravel()
            sem_values = semantic_similarity[np.ix_(indices[source], indices[target])].ravel()
            if len(ac_values) >= minimum_pairs:
                matrix[i, j] = matrix[j, i] = np.corrcoef(ac_values, sem_values)[0, 1]
    return species, matrix, [classes_by_species.get(value, "") for value in species]


def compute_keyword_pmi(
    dataset: CanonicalDataset,
    *,
    minimum_calls: int = 4,
    alpha: float = 0.05,
) -> dict[str, Any]:
    acoustic_counts = Counter(k for values in dataset.calls["acoustic_keywords"] for k in values)
    semantic_counts = Counter(k for values in dataset.calls["semantic_keywords"] for k in values)
    acoustic = sorted(key for key, count in acoustic_counts.items() if count >= minimum_calls)
    semantic = sorted(key for key, count in semantic_counts.items() if count >= minimum_calls)
    n = len(dataset.calls)
    matrix = np.zeros((len(acoustic), len(semantic)), dtype=float)
    joint = np.zeros_like(matrix, dtype=int)
    p_values = np.ones_like(matrix, dtype=float)
    matches: dict[str, list[str]] = {}
    for i, ac_keyword in enumerate(acoustic):
        for j, sem_keyword in enumerate(semantic):
            mask = dataset.calls.apply(
                lambda row: ac_keyword in row["acoustic_keywords"]
                and sem_keyword in row["semantic_keywords"],
                axis=1,
            )
            joint[i, j] = int(mask.sum())
            acoustic_only = acoustic_counts[ac_keyword] - joint[i, j]
            semantic_only = semantic_counts[sem_keyword] - joint[i, j]
            neither = n - joint[i, j] - acoustic_only - semantic_only
            p_values[i, j] = float(
                fisher_exact(
                    [[joint[i, j], acoustic_only], [semantic_only, neither]],
                    alternative="two-sided",
                ).pvalue
            )
            if joint[i, j]:
                matrix[i, j] = np.log2(
                    (joint[i, j] / n)
                    / ((acoustic_counts[ac_keyword] / n) * (semantic_counts[sem_keyword] / n))
                )
                matches[f"{i}:{j}"] = dataset.calls.loc[mask, "call_id"].astype(str).tolist()
    flat_p = p_values.ravel()
    order = np.argsort(flat_p)
    ranked = flat_p[order]
    adjusted_ranked = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    flat_q = np.empty_like(flat_p)
    flat_q[order] = np.clip(adjusted_ranked, 0, 1)
    q_values = flat_q.reshape(p_values.shape)
    significant = q_values < alpha
    return {
        "acoustic_keywords": acoustic,
        "semantic_keywords": semantic,
        "matrix": matrix.tolist(),
        "joint_counts": joint.tolist(),
        "p_values": p_values.tolist(),
        "q_values": q_values.tolist(),
        "significant": significant.tolist(),
        "alpha": alpha,
        "significance_test": "two-sided Fisher exact test",
        "multiple_testing_correction": "Benjamini-Hochberg FDR",
        "acoustic_counts": [acoustic_counts[key] for key in acoustic],
        "semantic_counts": [semantic_counts[key] for key in semantic],
        "matches": matches,
    }
