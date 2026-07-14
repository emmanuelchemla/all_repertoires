from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import linregress

from .datasets import CanonicalDataset


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RELATIONSHIP_GROUPS = ("within species", "same family", "different families")
CLASS_ORDER = {"Amphibia": 0, "Aves": 1, "Mammalia": 2}
COVERAGE_SPECIES_PERCENTAGES = (25, 50, 75, 100)
TAXONOMIC_GROUP_ORDER = (
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
)

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
    analysis_version: str = "5"
    embedding_model: str = EMBEDDING_MODEL
    random_seed: int = 42
    n_permutations: int = 9999
    scatter_sample_per_group: int = 1000
    species_pair_minimum: int = 6
    pmi_minimum: int = 4
    pmi_alpha: float = 0.05
    coverage_threshold_step: float = 0.01
    coverage_default_threshold: float = 0.6

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
    group_order = {group: index for index, group in enumerate(TAXONOMIC_GROUP_ORDER)}
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


def _coverage_for_similarity(
    similarity: np.ndarray,
    source_indices: np.ndarray,
    target_species_indices: list[np.ndarray],
    thresholds: list[float],
) -> dict[str, list[list[float]] | list[list[int]]]:
    best_by_species = np.column_stack(
        [
            similarity[np.ix_(source_indices, indices)].max(axis=1)
            for indices in target_species_indices
        ]
    )
    for column, indices in enumerate(target_species_indices):
        best_by_species[np.isin(source_indices, indices), column] = 1.0
    n_calls = len(source_indices)
    n_species = len(target_species_indices)
    count_rows: list[list[int]] = []
    percent_rows: list[list[float]] = []
    for threshold in thresholds:
        represented_species = np.count_nonzero(best_by_species >= threshold, axis=1)
        counts = []
        percentages = []
        for species_percent in COVERAGE_SPECIES_PERCENTAGES:
            minimum_species = int(np.ceil(species_percent * n_species / 100))
            matching_calls = int(np.count_nonzero(represented_species >= minimum_species))
            counts.append(matching_calls)
            percentages.append(100 * matching_calls / n_calls)
        count_rows.append(counts)
        percent_rows.append(percentages)
    return {"n_calls": count_rows, "percent_calls": percent_rows}


def compute_cross_species_coverage(
    dataset: CanonicalDataset,
    acoustic_similarity: np.ndarray,
    semantic_similarity: np.ndarray,
    *,
    thresholds: list[float] | np.ndarray | None = None,
    default_threshold: float = 0.6,
) -> dict[str, Any]:
    """Compute stored coverage curves for all calls and broad taxonomic groups."""
    if thresholds is None:
        thresholds = np.round(np.arange(0, 1.001, 0.01), 2)
    threshold_values = sorted({float(value) for value in thresholds})
    if not threshold_values or threshold_values[0] < 0 or threshold_values[-1] > 1:
        raise ValueError("coverage thresholds must be between 0 and 1")

    calls = dataset.calls
    species_values = calls["species"].astype(str).to_numpy()
    species = sorted(set(species_values))
    call_indices_by_species = {
        value: np.flatnonzero(species_values == value) for value in species
    }
    group_by_species: dict[str, str] = {}
    for value in species:
        row = calls.iloc[call_indices_by_species[value][0]]
        group_by_species[value] = _taxonomic_group(
            str(row.get("class", "")),
            str(row.get("order", "")),
            str(row.get("family", "")),
        )

    group_members: list[tuple[str, str, list[str]]] = [
        ("all", "All AnimalLex", species)
    ]
    for group in TAXONOMIC_GROUP_ORDER:
        members = [value for value in species if group_by_species[value] == group]
        if len(members) >= 2:
            group_members.append((group.lower().replace(" ", "_"), group, members))

    groups: dict[str, Any] = {}
    for key, label, members in group_members:
        source_indices = np.concatenate([call_indices_by_species[value] for value in members])
        target_indices = [call_indices_by_species[value] for value in members]
        groups[key] = {
            "label": label,
            "n_species": len(members),
            "n_calls": len(source_indices),
            "minimum_species": [
                int(np.ceil(percent * len(members) / 100))
                for percent in COVERAGE_SPECIES_PERCENTAGES
            ],
            "semantic": _coverage_for_similarity(
                semantic_similarity, source_indices, target_indices, threshold_values
            ),
            "acoustic": _coverage_for_similarity(
                acoustic_similarity, source_indices, target_indices, threshold_values
            ),
        }
    if float(default_threshold) not in threshold_values:
        raise ValueError("default_threshold must be included in thresholds")
    return {
        "default_group": "all",
        "default_threshold": float(default_threshold),
        "thresholds": threshold_values,
        "species_percentages": list(COVERAGE_SPECIES_PERCENTAGES),
        "source_species_included": True,
        "groups": groups,
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
    n_permutations: int = 9999,
    random_seed: int = 42,
    permutation_batch_size: int = 128,
) -> dict[str, Any]:
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1")
    acoustic_counts = Counter(k for values in dataset.calls["acoustic_keywords"] for k in values)
    semantic_counts = Counter(k for values in dataset.calls["semantic_keywords"] for k in values)
    acoustic = sorted(key for key, count in acoustic_counts.items() if count >= minimum_calls)
    semantic = sorted(key for key, count in semantic_counts.items() if count >= minimum_calls)
    n = len(dataset.calls)
    acoustic_presence = np.asarray(
        [
            [keyword in values for keyword in acoustic]
            for values in dataset.calls["acoustic_keywords"]
        ],
        dtype=np.int16,
    )
    semantic_presence = np.asarray(
        [
            [keyword in values for keyword in semantic]
            for values in dataset.calls["semantic_keywords"]
        ],
        dtype=np.int16,
    )
    joint = acoustic_presence.T @ semantic_presence
    matrix = np.zeros(joint.shape, dtype=float)

    species_values = dataset.calls["species"].astype(str).to_numpy()
    species_groups = [
        np.flatnonzero(species_values == species)
        for species in sorted(set(species_values))
    ]
    expected = np.zeros(joint.shape, dtype=float)
    for indices in species_groups:
        expected += np.outer(
            acoustic_presence[indices].sum(axis=0),
            semantic_presence[indices].sum(axis=0),
        ) / len(indices)

    observed_deviation = np.abs(joint - expected)
    extreme_counts = np.zeros(joint.shape, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    completed = 0
    while completed < n_permutations:
        batch_size = min(permutation_batch_size, n_permutations - completed)
        permuted_joint = np.zeros((batch_size, *joint.shape), dtype=np.int32)
        for indices in species_groups:
            group_acoustic = acoustic_presence[indices]
            group_semantic = semantic_presence[indices]
            if len(indices) == 1:
                contribution = group_acoustic.T @ group_semantic
                permuted_joint += contribution
                continue
            permutations = np.argsort(
                rng.random((batch_size, len(indices))), axis=1
            )
            shuffled_acoustic = group_acoustic[permutations]
            permuted_joint += (
                shuffled_acoustic.transpose(0, 2, 1) @ group_semantic
            )
        extreme_counts += np.count_nonzero(
            np.abs(permuted_joint - expected) >= observed_deviation - 1e-12,
            axis=0,
        )
        completed += batch_size
    p_values = (extreme_counts + 1) / (n_permutations + 1)

    matches: dict[str, list[str]] = {}
    for i, ac_keyword in enumerate(acoustic):
        for j, sem_keyword in enumerate(semantic):
            mask = (acoustic_presence[:, i] & semantic_presence[:, j]).astype(bool)
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
        "expected_counts": expected.tolist(),
        "p_values": p_values.tolist(),
        "q_values": q_values.tolist(),
        "significant": significant.tolist(),
        "alpha": alpha,
        "significance_test": "two-sided within-species permutation test",
        "permutation_unit": "complete acoustic keyword sets",
        "n_permutations": n_permutations,
        "random_seed": random_seed,
        "null_model": (
            "Acoustic keyword sets are shuffled among calls within each species; "
            "semantic keyword sets remain fixed."
        ),
        "multiple_testing_correction": "Benjamini-Hochberg FDR",
        "acoustic_counts": [acoustic_counts[key] for key in acoustic],
        "semantic_counts": [semantic_counts[key] for key in semantic],
        "matches": matches,
    }
