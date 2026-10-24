import numpy as np
import pytest

from omniseek_multimodal.storage import Store


def seeded(path):
    store = Store(path)
    store.execute("INSERT INTO collections (id,name,description) VALUES ('c','test','')")
    store.execute(
        "INSERT INTO files (id,collection_id,name,modality,sha256,size,metadata) "
        "VALUES ('f','c','a.txt','text','hash',10,'{}')"
    )
    return store


def test_vectors_and_metadata_survive_reload(tmp_path):
    path = tmp_path / "db.sqlite3"
    store = seeded(path)
    vector = np.random.default_rng(1).normal(size=(1, 768)).astype(np.float32)
    store.replace_segments("f", [{"start": 10.0, "end": 20.0, "text": "passage"}], vector)
    loaded = Store(path)
    rows, vectors = loaded.corpus("c")
    assert rows[0]["metadata"]["start"] == 10
    assert vectors.shape == (1, 768)
    assert np.linalg.norm(vectors[0]) == pytest.approx(1)
    assert loaded.files()[0]["status"] == "indexed"


def test_revision_and_index_version_mismatch_are_rejected(tmp_path):
    store = seeded(tmp_path / "db")
    store.replace_segments("f", [{}], np.ones((1, 768), dtype=np.float32))
    store.execute("UPDATE segments SET revision='different'")
    with pytest.raises(ValueError, match="Incompatible"):
        store.corpus()


def test_rebuild_replaces_instead_of_accumulating_and_delete_cascades(tmp_path):
    store = seeded(tmp_path / "db")
    for _ in range(2):
        store.replace_segments("f", [{}], np.ones((1, 768), dtype=np.float32))
    assert len(store.corpus()[0]) == 1
    store.execute("DELETE FROM collections WHERE id='c'")
    assert store.files() == []
    assert store.corpus()[1].shape == (0, 768)


def test_bad_vector_batch_does_not_replace_good_index(tmp_path):
    store = seeded(tmp_path / "db")
    store.replace_segments("f", [{}], np.ones((1, 768), dtype=np.float32))
    with pytest.raises(ValueError):
        store.replace_segments("f", [{}], np.zeros((1, 768), dtype=np.float32))
    assert len(store.corpus()[0]) == 1
