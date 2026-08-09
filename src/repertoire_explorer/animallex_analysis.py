from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import json
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.stats import linregress
from sklearn.linear_model import Ridge

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
    analysis_version: str = "12"
    embedding_model: str = EMBEDDING_MODEL
    random_seed: int = 42
    n_permutations: int = 999
    scatter_sample_per_group: int = 1000
    species_pair_minimum: int = 6
    pmi_minimum: int = 4
    pmi_alpha: float = 0.05
    coverage_threshold_step: float = 0.01
    coverage_default_threshold: float = 0.6
    prediction_folds: int = 10
    prediction_ridge_alpha: float = 10.0
    prediction_bootstrap_samples: int = 2000
    motif_minimum_species: int = 4
    motif_minimum_families: int = 1
    motif_minimum_orders: int = 1
    motif_acoustic_similarity: float = 0.65
    motif_semantic_similarity: float = 0.65
    motif_overlap_jaccard: float = 1.1
    motif_max_per_signature: int = 10_000
    motif_max_results: int = 10_000
    motif_exclude_low_confidence: bool = False

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


def set_similarity_matrix(keyword_sets: list[list[str]]) -> np.ndarray:
    """Return pairwise Jaccard similarity for a sequence of keyword sets."""
    normalized = [set(values) for values in keyword_sets]
    vocabulary = sorted(set().union(*normalized)) if normalized else []
    if not vocabulary:
        return np.ones((len(normalized), len(normalized)), dtype=np.float32)
    positions = {keyword: index for index, keyword in enumerate(vocabulary)}
    incidence = np.zeros((len(normalized), len(vocabulary)), dtype=np.float32)
    for row, values in enumerate(normalized):
        incidence[row, [positions[value] for value in values]] = 1
    intersections = incidence @ incidence.T
    sizes = incidence.sum(axis=1)
    unions = sizes[:, None] + sizes[None, :] - intersections
    return np.divide(
        intersections,
        unions,
        out=np.ones_like(intersections),
        where=unions != 0,
    )


MOTIF_UNINFORMATIVE_SEMANTIC_KEYWORDS = {"attention"}


def _maximal_cliques(adjacency: list[set[int]]) -> list[list[int]]:
    """Enumerate maximal cliques with a deterministic Bron--Kerbosch search."""
    cliques: list[list[int]] = []

    def visit(current: set[int], candidates: set[int], excluded: set[int]) -> None:
        if not candidates and not excluded:
            cliques.append(sorted(current))
            return
        pivot_pool = candidates | excluded
        pivot = (
            max(pivot_pool, key=lambda value: (len(candidates & adjacency[value]), -value))
            if pivot_pool
            else None
        )
        remaining = candidates - (adjacency[pivot] if pivot is not None else set())
        for vertex in sorted(remaining):
            visit(
                current | {vertex},
                candidates & adjacency[vertex],
                excluded & adjacency[vertex],
            )
            candidates.remove(vertex)
            excluded.add(vertex)

    visit(set(), set(range(len(adjacency))), set())
    return cliques


def compute_cross_species_motifs(
    dataset: CanonicalDataset,
    acoustic_similarity: np.ndarray,
    semantic_similarity: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Find curated reciprocal cross-species matches in both embedding spaces.

    An edge joins calls from two species only when each call is the other's
    within-species top-1 acoustic and semantic match. Similarity floors are
    applied before maximal cliques are enumerated, so every pair in a reported
    motif clears both thresholds.
    """
    calls = dataset.calls.reset_index(drop=True)
    n_calls = len(calls)
    if acoustic_similarity.shape != (n_calls, n_calls):
        raise ValueError("acoustic_similarity must match the number of calls")
    if semantic_similarity.shape != (n_calls, n_calls):
        raise ValueError("semantic_similarity must match the number of calls")

    species_values = calls["species"].astype(str).to_numpy()
    species = sorted(set(species_values))
    species_indices = {
        value: np.flatnonzero(species_values == value) for value in species
    }
    adjacency = [set() for _ in range(n_calls)]
    edge_values: dict[tuple[int, int], tuple[float, float]] = {}
    for source_offset, source in enumerate(species):
        source_indices = species_indices[source]
        for target in species[source_offset + 1 :]:
            target_indices = species_indices[target]
            for target_index in target_indices:
                acoustic_source = int(
                    source_indices[
                        np.argmax(acoustic_similarity[target_index, source_indices])
                    ]
                )
                semantic_source = int(
                    source_indices[
                        np.argmax(semantic_similarity[target_index, source_indices])
                    ]
                )
                if acoustic_source != semantic_source:
                    continue
                acoustic_target = int(
                    target_indices[
                        np.argmax(acoustic_similarity[acoustic_source, target_indices])
                    ]
                )
                semantic_target = int(
                    target_indices[
                        np.argmax(semantic_similarity[acoustic_source, target_indices])
                    ]
                )
                if acoustic_target != target_index or semantic_target != target_index:
                    continue
                acoustic_value = float(
                    acoustic_similarity[acoustic_source, target_index]
                )
                semantic_value = float(
                    semantic_similarity[acoustic_source, target_index]
                )
                if (
                    acoustic_value < config.motif_acoustic_similarity
                    or semantic_value < config.motif_semantic_similarity
                ):
                    continue
                left, right = sorted((acoustic_source, int(target_index)))
                adjacency[left].add(right)
                adjacency[right].add(left)
                edge_values[(left, right)] = (acoustic_value, semantic_value)

    eligible: list[dict[str, Any]] = []
    for clique in _maximal_cliques(adjacency):
        if len(clique) < config.motif_minimum_species:
            continue
        rows = calls.iloc[clique]
        if rows["family"].astype(str).nunique() < config.motif_minimum_families:
            continue
        if rows["order"].astype(str).nunique() < config.motif_minimum_orders:
            continue
        if config.motif_exclude_low_confidence and "low" in set(rows["confidence"]):
            continue
        shared_semantic = set.intersection(
            *(set(values) for values in rows["semantic_keywords"])
        ) - MOTIF_UNINFORMATIVE_SEMANTIC_KEYWORDS
        shared_acoustic = set.intersection(
            *(set(values) for values in rows["acoustic_keywords"])
        )
        pairs = [tuple(sorted(pair)) for pair in combinations(clique, 2)]
        acoustic_values = [edge_values[pair][0] for pair in pairs]
        semantic_values = [edge_values[pair][1] for pair in pairs]
        joint_values = [
            (acoustic_value + semantic_value) / 2
            for acoustic_value, semantic_value in zip(
                acoustic_values, semantic_values
            )
        ]
        signature = tuple(sorted(shared_semantic))
        eligible.append(
            {
                "indices": set(clique),
                "signature": signature,
                "n_species": len(clique),
                "n_classes": int(rows["class"].astype(str).nunique()),
                "n_orders": int(rows["order"].astype(str).nunique()),
                "n_families": int(rows["family"].astype(str).nunique()),
                "minimum_acoustic_similarity": min(acoustic_values),
                "minimum_semantic_similarity": min(semantic_values),
                "mean_joint_similarity": float(np.mean(joint_values)),
                "shared_semantic_keywords": list(signature),
                "shared_acoustic_keywords": sorted(shared_acoustic),
                "members": [
                    {
                        "call_id": str(row["call_id"]),
                        "species": str(row["species"]),
                        "common_name": str(
                            dataset.species_metadata.get(str(row["species"]), {}).get(
                                "common_name", row["species"]
                            )
                        ),
                        "call_name": str(row["call_name"]),
                        "class": str(row["class"]),
                        "order": str(row["order"]),
                        "family": str(row["family"]),
                        "confidence": str(row["confidence"]),
                        "acoustic_keywords": list(row["acoustic_keywords"]),
                        "semantic_keywords": list(row["semantic_keywords"]),
                        "acoustic_description": str(row["acoustic_description"]),
                        "semantic_description": str(row["semantic_description"]),
                    }
                    for _, row in rows.sort_values(
                        ["class", "order", "family", "species", "call_name"]
                    ).iterrows()
                ],
            }
        )

    eligible.sort(
        key=lambda motif: (
            -min(
                motif["minimum_acoustic_similarity"],
                motif["minimum_semantic_similarity"],
            ),
            -motif["mean_joint_similarity"],
            -motif["n_classes"],
            -motif["n_orders"],
            -motif["n_families"],
            -motif["n_species"],
            motif["signature"],
            tuple(member["call_id"] for member in motif["members"]),
        )
    )
    selected: list[dict[str, Any]] = []
    signature_counts: Counter[tuple[str, ...]] = Counter()
    for motif in eligible:
        if signature_counts[motif["signature"]] >= config.motif_max_per_signature:
            continue
        if any(
            len(motif["indices"] & other["indices"])
            / len(motif["indices"] | other["indices"])
            >= config.motif_overlap_jaccard
            for other in selected
        ):
            continue
        signature_counts[motif["signature"]] += 1
        selected.append(motif)
        if len(selected) >= config.motif_max_results:
            break

    motifs = []
    for index, motif in enumerate(selected, start=1):
        motif = dict(motif)
        motif.pop("indices")
        motif.pop("signature")
        motif["motif_id"] = f"motif-{index}"
        motifs.append(motif)
    return {
        "criteria": {
            "minimum_species": config.motif_minimum_species,
            "minimum_families": config.motif_minimum_families,
            "minimum_orders": config.motif_minimum_orders,
            "minimum_acoustic_similarity": config.motif_acoustic_similarity,
            "minimum_semantic_similarity": config.motif_semantic_similarity,
            "excluded_semantic_keywords": sorted(
                MOTIF_UNINFORMATIVE_SEMANTIC_KEYWORDS
            ),
            "exclude_low_confidence": config.motif_exclude_low_confidence,
            "overlap_jaccard": config.motif_overlap_jaccard,
            "maximum_per_semantic_signature": config.motif_max_per_signature,
            "maximum_results": config.motif_max_results,
        },
        "n_thresholded_edges": len(edge_values),
        "n_eligible_before_deduplication": len(eligible),
        "n_motifs": len(motifs),
        "ranking_metric": "minimum of the acoustic and semantic pairwise minima",
        "motifs": motifs,
    }


def mantel(
    acoustic_distance: np.ndarray,
    semantic_distance: np.ndarray,
    *,
    pair_indices: tuple[np.ndarray, np.ndarray] | None = None,
    permutation_blocks: list[np.ndarray] | None = None,
    n_perm: int = 9999,
    seed: int = 42,
) -> tuple[float, float]:
    """Mantel test using simultaneous row/column label permutations.

    ``permutation_blocks`` constrains label shuffling to exchangeable groups. For
    AnimalLex those groups are species, which preserves species composition and
    repertoire sizes while breaking call-level acoustic/semantic correspondence.
    """
    acoustic_distance = np.asarray(acoustic_distance, dtype=float)
    semantic_distance = np.asarray(semantic_distance, dtype=float)
    if (
        acoustic_distance.ndim != 2
        or acoustic_distance.shape[0] != acoustic_distance.shape[1]
    ):
        raise ValueError("acoustic_distance must be a square matrix")
    if semantic_distance.shape != acoustic_distance.shape:
        raise ValueError(
            "semantic_distance must have the same shape as acoustic_distance"
        )
    n = acoustic_distance.shape[0]
    if pair_indices is None:
        left, right = np.triu_indices(n, k=1)
    else:
        left, right = pair_indices
        left = np.asarray(left, dtype=int)
        right = np.asarray(right, dtype=int)
    if len(left) < 2:
        return float("nan"), float("nan")
    blocks = permutation_blocks or [np.arange(n)]
    covered = np.concatenate(blocks) if blocks else np.array([], dtype=int)
    if len(covered) != n or not np.array_equal(np.sort(covered), np.arange(n)):
        raise ValueError("permutation_blocks must partition all matrix rows")

    acoustic_values = acoustic_distance[left, right]
    semantic_values = semantic_distance[left, right]
    rng = np.random.default_rng(seed)
    observed = float(np.corrcoef(acoustic_values, semantic_values)[0, 1])
    count = 0
    permutation = np.arange(n)
    for _ in range(n_perm):
        for block in blocks:
            permutation[block] = rng.permutation(block)
        permuted_values = semantic_distance[permutation[left], permutation[right]]
        permuted = float(np.corrcoef(acoustic_values, permuted_values)[0, 1])
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
    result = _compute_form_meaning_from_similarity(
        dataset, ac_similarity, sem_similarity, config
    )
    result["similarity_method"] = "cosine similarity of description embeddings"
    return result


def compute_keyword_form_meaning_alignment(
    dataset: CanonicalDataset,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Compute form-to-meaning alignment from acoustic and semantic keyword sets."""
    ac_similarity = set_similarity_matrix(
        dataset.calls["acoustic_keywords"].tolist()
    )
    sem_similarity = set_similarity_matrix(
        dataset.calls["semantic_keywords"].tolist()
    )
    result = _compute_form_meaning_from_similarity(
        dataset, ac_similarity, sem_similarity, config
    )
    result["similarity_method"] = "Jaccard similarity of keyword sets"
    return result


def _compute_form_meaning_from_similarity(
    dataset: CanonicalDataset,
    ac_similarity: np.ndarray,
    sem_similarity: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, Any]:
    ac_distance_matrix = 1 - ac_similarity
    sem_distance_matrix = 1 - sem_similarity
    n = len(dataset.calls)
    left, right = np.triu_indices(n, k=1)
    species = dataset.calls["species"].astype(str).to_numpy()
    families = dataset.calls["family"].astype(str).to_numpy()
    labels = np.where(
        species[left] == species[right],
        "within species",
        np.where(families[left] == families[right], "same family", "different families"),
    )
    ac_distance = ac_distance_matrix[left, right]
    sem_distance = sem_distance_matrix[left, right]
    permutation_blocks = [
        np.flatnonzero(species == value) for value in sorted(set(species))
    ]
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
            ac_distance_matrix,
            sem_distance_matrix,
            pair_indices=(left[indices], right[indices]),
            permutation_blocks=permutation_blocks,
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
    return {
        "groups": groups,
        "significance_test": "one-sided Mantel test with blocked label permutations",
        "permutation_unit": "semantic call identities shuffled within species",
        "n_permutations": config.n_permutations,
    }


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return values / norms


def _random_call_folds(n_calls: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [fold for fold in np.array_split(rng.permutation(n_calls), n_folds) if len(fold)]


def _balanced_group_folds(
    groups: np.ndarray, n_folds: int, seed: int
) -> list[np.ndarray]:
    """Assign complete groups to folds while approximately balancing call counts."""
    group_values, counts = np.unique(groups.astype(str), return_counts=True)
    rng = np.random.default_rng(seed)
    tie_breakers = rng.random(len(group_values))
    order = np.lexsort((tie_breakers, -counts))
    fold_groups: list[list[str]] = [[] for _ in range(n_folds)]
    fold_sizes = np.zeros(n_folds, dtype=int)
    for index in order:
        target = int(np.argmin(fold_sizes))
        fold_groups[target].append(str(group_values[index]))
        fold_sizes[target] += int(counts[index])
    return [
        np.flatnonzero(np.isin(groups.astype(str), values))
        for values in fold_groups
        if values
    ]


def _prediction_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    predicted = _normalize_rows(predicted)
    truth = _normalize_rows(truth)
    cosine = float(np.mean(np.sum(predicted * truth, axis=1)))
    scores = predicted @ truth.T
    true_scores = np.diag(scores)
    ranks = 1 + np.count_nonzero(scores > true_scores[:, None] + 1e-12, axis=1)
    return {"cosine": cosine, "mrr": float(np.mean(1 / ranks))}


def _bootstrap_interval(
    values: list[float], *, n_samples: int, seed: int
) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_samples, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def compute_acoustic_semantic_prediction(
    dataset: CanonicalDataset,
    acoustic_embeddings: np.ndarray,
    semantic_embeddings: np.ndarray,
    config: AnalysisConfig,
) -> dict[str, Any]:
    """Evaluate acoustic-text to semantic-text prediction under three holdouts."""
    acoustic = _normalize_rows(acoustic_embeddings)
    semantic = _normalize_rows(semantic_embeddings)
    n_calls = len(dataset.calls)
    n_folds = min(config.prediction_folds, n_calls)
    species = dataset.calls["species"].astype(str).to_numpy()
    families = dataset.calls["family"].astype(str).to_numpy()
    conditions = [
        (
            "held_out_calls",
            "Held-out calls",
            _random_call_folds(n_calls, n_folds, config.random_seed),
            "Calls are randomly assigned to folds; species may occur in train and test.",
        ),
        (
            "held_out_species",
            "Held-out species",
            _balanced_group_folds(
                species, min(config.prediction_folds, len(set(species))), config.random_seed
            ),
            "Every test species is absent from its training fold.",
        ),
        (
            "held_out_families",
            "Held-out families",
            _balanced_group_folds(
                families,
                min(config.prediction_folds, len(set(families))),
                config.random_seed,
            ),
            "Every test family is absent from its training fold.",
        ),
    ]
    model_labels = {
        "random": "Random pairing",
        "retrieval": "Acoustic nearest neighbor",
        "ridge": "Linear ridge",
    }
    metric_labels = {"cosine": "Cosine similarity", "mrr": "Mean reciprocal rank"}
    results: dict[str, dict[str, dict[str, Any]]] = {}
    condition_metadata = []
    all_indices = np.arange(n_calls)
    for condition_offset, (key, label, test_folds, description) in enumerate(conditions):
        condition_metadata.append(
            {
                "key": key,
                "label": label,
                "description": description,
                "n_folds": len(test_folds),
            }
        )
        fold_results = {
            model: {metric: [] for metric in metric_labels} for model in model_labels
        }
        for fold_offset, test_indices in enumerate(test_folds):
            train_indices = np.setdiff1d(all_indices, test_indices, assume_unique=True)
            if len(test_indices) < 2 or len(train_indices) < 2:
                continue
            acoustic_train = acoustic[train_indices]
            semantic_train = semantic[train_indices]
            acoustic_test = acoustic[test_indices]
            semantic_test = semantic[test_indices]
            fold_rng = np.random.default_rng(
                config.random_seed + 1000 * condition_offset + fold_offset
            )
            predictions = {
                "random": semantic_test[fold_rng.permutation(len(test_indices))],
                "retrieval": semantic_train[
                    np.argmax(acoustic_test @ acoustic_train.T, axis=1)
                ],
            }
            ridge = Ridge(alpha=config.prediction_ridge_alpha, solver="cholesky")
            ridge.fit(acoustic_train, semantic_train)
            predictions["ridge"] = ridge.predict(acoustic_test)
            for model, predicted in predictions.items():
                metrics = _prediction_metrics(predicted, semantic_test)
                for metric, value in metrics.items():
                    fold_results[model][metric].append(value)

        results[key] = {}
        for model_offset, model in enumerate(model_labels):
            results[key][model] = {}
            for metric_offset, metric in enumerate(metric_labels):
                values = fold_results[model][metric]
                low, high = _bootstrap_interval(
                    values,
                    n_samples=config.prediction_bootstrap_samples,
                    seed=(
                        config.random_seed
                        + 10_000 * condition_offset
                        + 100 * model_offset
                        + metric_offset
                    ),
                )
                results[key][model][metric] = {
                    "mean": float(np.mean(values)),
                    "ci_low": low,
                    "ci_high": high,
                    "fold_values": values,
                }
    return {
        "conditions": condition_metadata,
        "models": model_labels,
        "metrics": metric_labels,
        "results": results,
        "n_calls": n_calls,
        "n_species": int(len(set(species))),
        "n_families": int(len(set(families))),
        "n_folds": config.prediction_folds,
        "ridge_alpha": config.prediction_ridge_alpha,
        "uncertainty": "95% percentile bootstrap confidence interval across outer folds",
        "bootstrap_samples": config.prediction_bootstrap_samples,
        "candidate_pool": "Calls in the test fold for each split",
        "input": "sentence embedding of the acoustic text description",
        "target": "sentence embedding of the semantic text description",
    }


def _reorder_within_classes(matrix: np.ndarray, classes: list[str]) -> list[int]:
    """Keep class blocks contiguous while clustering species inside each block."""
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
        key=lambda value: (
            CLASS_ORDER.get(str(classes_by_species.get(value, "")), 9),
            value,
        ),
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
    order = _reorder_within_classes(matrix, classes)
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
    # A call must be represented by another species. Exclude every call's own
    # species rather than counting the diagonal/self-repertoire match as 1.0.
    for column, indices in enumerate(target_species_indices):
        best_by_species[np.isin(source_indices, indices), column] = -np.inf
    n_calls = len(source_indices)
    n_species = len(target_species_indices) - 1
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
            "n_target_species": len(members) - 1,
            "n_calls": len(source_indices),
            "minimum_species": [
                int(np.ceil(percent * (len(members) - 1) / 100))
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
        "source_species_included": False,
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
    expected = np.outer(
        acoustic_presence.sum(axis=0), semantic_presence.sum(axis=0)
    ) / n

    observed_deviation = np.abs(joint - expected)
    extreme_counts = np.zeros(joint.shape, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    completed = 0
    while completed < n_permutations:
        batch_size = min(permutation_batch_size, n_permutations - completed)
        permutations = np.argsort(rng.random((batch_size, n)), axis=1)
        shuffled_acoustic = acoustic_presence[permutations]
        permuted_joint = shuffled_acoustic.transpose(0, 2, 1) @ semantic_presence
        extreme_counts += np.count_nonzero(
            np.abs(permuted_joint - expected) >= observed_deviation - 1e-12,
            axis=0,
        )
        completed += batch_size
    p_values = (extreme_counts + 1) / (n_permutations + 1)

    # Standard PMI in bits, written as log2(observed / globally expected count).
    # This expectation matches the global permutation null used above.
    matrix = np.full(joint.shape, np.nan, dtype=float)
    positive = (joint > 0) & (expected > 0)
    matrix[positive] = np.log2(joint[positive] / expected[positive])

    matches: dict[str, list[str]] = {}
    for i in range(len(acoustic)):
        for j in range(len(semantic)):
            mask = (acoustic_presence[:, i] & semantic_presence[:, j]).astype(bool)
            if joint[i, j]:
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
        "global_pmi_matrix": matrix.tolist(),
        "joint_counts": joint.tolist(),
        "expected_counts": expected.tolist(),
        "p_values": p_values.tolist(),
        "q_values": q_values.tolist(),
        "significant": significant.tolist(),
        "alpha": alpha,
        "significance_test": "two-sided global permutation test",
        "effect_size": "pointwise mutual information (bits)",
        "permutation_unit": "complete acoustic keyword sets shuffled across all calls",
        "n_permutations": n_permutations,
        "random_seed": random_seed,
        "null_model": (
            "Acoustic keyword sets are shuffled among all calls; "
            "semantic keyword sets remain fixed."
        ),
        "multiple_testing_correction": "Benjamini-Hochberg FDR",
        "acoustic_counts": [acoustic_counts[key] for key in acoustic],
        "semantic_counts": [semantic_counts[key] for key in semantic],
        "matches": matches,
    }
