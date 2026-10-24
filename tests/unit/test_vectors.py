import numpy as np
import pytest

from omniseek_multimodal.vectors import normalize, search


@pytest.mark.parametrize("dimension", [768, 512, 256, 128])
def test_truncation_is_normalized_without_mutating_original(dimension):
    original = np.random.default_rng(21).normal(size=(4, 768)).astype(np.float32)
    before = original.copy()
    output = normalize(original, dimension)
    assert output.shape == (4, dimension)
    assert output.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(output, axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(original, before)


@pytest.mark.parametrize("invalid", [np.zeros(768), np.full(768, np.nan), np.full(768, np.inf), np.ones(127)])
def test_invalid_embeddings_fail(invalid):
    with pytest.raises(ValueError):
        normalize(invalid)


def test_incompatible_dimension_fails():
    with pytest.raises(ValueError):
        normalize(np.ones(768), 64)


def test_renormalization_changes_wrong_dot_product_order():
    # Both full vectors have unit norm; truncation changes their norms differently.
    corpus = np.zeros((2, 768), dtype=np.float32)
    corpus[0, :2] = [0.1, 0]
    corpus[0, 700] = np.sqrt(0.99)
    corpus[1, :2] = [0.5, 0.5]
    corpus[1, 700] = np.sqrt(0.5)
    query = np.zeros(768, dtype=np.float32)
    query[0] = 1
    assert np.argmax(corpus[:, :128] @ query[:128]) == 1
    scores, ids = search(corpus, query, 128, 2)
    assert ids[0] == 0
    assert scores[0] == pytest.approx(1.0)


@pytest.mark.parametrize("approximate", [False, True])
def test_cosine_ranking(approximate):
    corpus = np.eye(768, dtype=np.float32)[:4]
    scores, ids = search(corpus, corpus[2], 768, 3, approximate)
    assert ids[0] == 2
    assert scores[0] == pytest.approx(1)
