"""
The FAISS index is a flat inner-product index over unit vectors, which is what
makes its three operations — add, search, remove — hold no surprises. These
tests pin that down: the index type and metric, that ids survive a round trip,
that removal is exact and permanent, and that all of it survives being written
to disk and read back.

Everything here runs against a real FAISS index in a temporary directory, so
nothing touches the application's data/ folder.
"""

from __future__ import annotations

import faiss
import numpy as np
import pytest

from services import faiss_store
from services.embedder import EMBEDDING_DIM


def unit(*rows: list[float]) -> np.ndarray:
    """Rows as L2-normalised float32 vectors, the way embedder.embed() returns them."""
    arr = np.array(rows, dtype=np.float32)
    return arr / np.linalg.norm(arr, axis=1, keepdims=True)


def basis(dim_index: int) -> list[float]:
    """A one-hot vector: two different ones are orthogonal, so cosine 0."""
    v = [0.0] * EMBEDDING_DIM
    v[dim_index] = 1.0
    return v


@pytest.fixture
def index(tmp_path, monkeypatch):
    """A fresh empty index that persists into tmp_path instead of data/."""
    monkeypatch.setattr(faiss_store, "_DATA_DIR", tmp_path)
    monkeypatch.setattr(faiss_store, "_INDEX_PATH", tmp_path / "faiss.index")
    monkeypatch.setattr(faiss_store, "_META_PATH", tmp_path / "faiss_meta.json")
    faiss_store.reset()
    yield faiss_store
    faiss_store.reset()


class TestIndexType:
    def test_it_is_a_flat_inner_product_index_of_the_embedding_dimension(self, index) -> None:
        idx = index._get_index()
        assert idx.metric_type == faiss.METRIC_INNER_PRODUCT
        assert idx.d == EMBEDDING_DIM
        # IndexIDMap so each vector carries its chunk's faiss_index_id, over a
        # flat (exhaustive) index so remove_ids() is supported.
        assert isinstance(idx, faiss.IndexIDMap)
        assert isinstance(faiss.downcast_index(idx.index), faiss.IndexFlat)

    def test_ids_are_handed_out_in_sequence_and_never_reused(self, index) -> None:
        first = [index.next_id() for _ in range(3)]
        assert first == [0, 1, 2]

        index.add(first, unit(basis(0), basis(1), basis(2)), persist=False)
        index.remove(first, persist=False)
        assert index.next_id() == 3   # not rewound by the removal


class TestAddAndSearch:
    def test_vectors_come_back_under_the_ids_they_were_added_with(self, index) -> None:
        index.add([101, 102, 103], unit(basis(0), basis(1), basis(2)), persist=False)
        assert index.index_size() == 3

        ids, sims = index.search(unit(basis(1))[0], k=3)
        assert ids[0] == 102
        assert sims[0] == pytest.approx(1.0, abs=1e-5)

    def test_the_score_is_the_cosine_similarity(self, index) -> None:
        # Over unit vectors the inner product IS the cosine, so an identical
        # vector scores 1 and an orthogonal one scores 0 — no conversion.
        index.add([1, 2], unit(basis(0), basis(1)), persist=False)
        ids, sims = index.search(unit(basis(0))[0], k=2)

        assert dict(zip(ids, sims))[1] == pytest.approx(1.0, abs=1e-5)
        assert dict(zip(ids, sims))[2] == pytest.approx(0.0, abs=1e-5)

    def test_results_are_ordered_best_first(self, index) -> None:
        near, far = basis(0), basis(1)
        blend = [a + 0.9 * b for a, b in zip(near, far)]
        index.add([1, 2, 3], unit(near, blend, far), persist=False)

        ids, sims = index.search(unit(near)[0], k=3)
        assert ids == [1, 2, 3]
        assert sims == sorted(sims, reverse=True)

    def test_an_empty_index_returns_nothing(self, index) -> None:
        assert index.search(unit(basis(0))[0], k=5) == ([], [])


class TestRemove:
    def test_removed_vectors_are_gone_from_the_index(self, index) -> None:
        index.add([101, 102, 103], unit(basis(0), basis(1), basis(2)), persist=False)

        assert index.remove([101, 102], persist=False) == 2
        assert index.index_size() == 1

        ids, _ = index.search(unit(basis(0))[0], k=10)
        assert ids == [103]   # not merely ranked last — absent

    def test_unrelated_vectors_stay_searchable(self, index) -> None:
        index.add([101, 102], unit(basis(0), basis(1)), persist=False)
        index.remove([101], persist=False)

        ids, sims = index.search(unit(basis(1))[0], k=10)
        assert ids == [102]
        assert sims[0] == pytest.approx(1.0, abs=1e-5)

    def test_removing_nothing_is_a_no_op(self, index) -> None:
        index.add([101], unit(basis(0)), persist=False)
        assert index.remove([], persist=False) == 0
        assert index.index_size() == 1

    def test_ids_can_be_added_again_after_a_removal(self, index) -> None:
        # Deleting a document must not corrupt the index for later ingests.
        index.add([101, 102], unit(basis(0), basis(1)), persist=False)
        index.remove([101], persist=False)
        index.add([103], unit(basis(2)), persist=False)

        ids, _ = index.search(unit(basis(2))[0], k=10)
        assert ids[0] == 103
        assert index.index_size() == 2


class TestPersistence:
    def test_the_index_survives_a_save_and_reload(self, index) -> None:
        index.add([101, 102], unit(basis(0), basis(1)))   # persists
        index.reset()
        index.load()

        assert index.index_size() == 2
        ids, _ = index.search(unit(basis(1))[0], k=1)
        assert ids == [102]

    def test_a_deletion_survives_a_save_and_reload(self, index) -> None:
        index.add([101, 102], unit(basis(0), basis(1)))
        index.remove([101])      # persists the smaller index
        index.reset()
        index.load()

        assert index.index_size() == 1
        ids, _ = index.search(unit(basis(0))[0], k=10)
        assert ids == [102]      # 101 did not come back

    def test_the_id_counter_survives_a_save_and_reload(self, index) -> None:
        ids = [index.next_id() for _ in range(2)]
        index.add(ids, unit(basis(0), basis(1)))
        index.reset()
        index.load()

        assert index.next_id() == 2   # no id is ever handed out twice

    def test_an_index_from_an_older_build_is_refused(self, index, tmp_path, caplog) -> None:
        # The index used to be an L2 HNSW graph. Its scores mean the opposite
        # of what search() now returns, so loading one would silently invert
        # every dense ranking — it must be ignored, not used.
        legacy = faiss.IndexIDMap(faiss.IndexHNSWFlat(EMBEDDING_DIM, 32))
        legacy.add_with_ids(unit(basis(0)), np.array([1], dtype=np.int64))
        faiss.write_index(legacy, str(tmp_path / "faiss.index"))

        index.load()

        assert index.index_size() == 0
        assert "rebuild_indexes" in caplog.text
