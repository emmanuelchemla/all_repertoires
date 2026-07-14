from __future__ import annotations

import numpy as np

from paper_code.fig_supp_mantel_matrix import compute_pairwise_r
from paper_code.mantel import mantel, similarity_matrix


def test_similarity_matrix_regression() -> None:
    embeddings = np.array([[3.0, 0.0], [0.0, 4.0], [1.0, 1.0]])

    result = similarity_matrix(embeddings)

    np.testing.assert_allclose(
        result,
        np.array(
            [
                [1.0, 0.0, 1 / np.sqrt(2)],
                [0.0, 1.0, 1 / np.sqrt(2)],
                [1 / np.sqrt(2), 1 / np.sqrt(2), 1.0],
            ]
        ),
        atol=1e-9,
    )


def test_mantel_regression() -> None:
    acoustic = np.array([0.1, 0.4, 0.7, 0.8, 0.3, 0.9])
    semantic = np.array([0.2, 0.5, 0.65, 0.7, 0.4, 0.95])

    result = mantel(acoustic, semantic, n_perm=25, seed=7)

    np.testing.assert_allclose(result, (0.9709191848, 0.0384615385), atol=1e-9)


def test_species_pair_correlation_regression() -> None:
    calls = [
        {"species": "A", "class": "Mammalia"},
        {"species": "A", "class": "Mammalia"},
        {"species": "B", "class": "Aves"},
        {"species": "B", "class": "Aves"},
        {"species": "B", "class": "Aves"},
    ]
    acoustic = np.eye(5)
    semantic = np.eye(5)
    acoustic_cross = np.array([[0.1, 0.2, 0.4], [0.3, 0.5, 0.9]])
    semantic_cross = np.array([[0.2, 0.3, 0.5], [0.4, 0.6, 1.0]])
    acoustic[:2, 2:] = acoustic_cross
    acoustic[2:, :2] = acoustic_cross.T
    semantic[:2, 2:] = semantic_cross
    semantic[2:, :2] = semantic_cross.T

    species, matrix, classes = compute_pairwise_r(calls, acoustic, semantic)

    assert species == ["B", "A"]
    assert classes == ["Aves", "Mammalia"]
    np.testing.assert_allclose(matrix[0, 1], 1.0, atol=1e-9)
