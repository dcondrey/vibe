"""VIBE wrapper for HMS (holographic-memory), the quantized graph index.

The index is a proximity graph whose traversal reads 4-bit or 8-bit codes (`code_bits`) of the
mean-centered rows; the `rerank` best candidates are re-scored with an 8-bit residual code.
Cosine and normalized data are searched by angle, euclidean data by Euclidean distance, and ip
data by inner product: the graph is built on rows lifted by one coordinate (MIPS reduced to
Euclidean search) and served directly on the original coordinates by negative inner product.

On out-of-distribution datasets the build adds query-aware edges computed from a seeded sample
of `learn_queries` rows of the learn split; the learn neighbors, test queries and ground truth
are never used. `reservoir` keeps the best scored candidates independently of the traversal
beam (0 uses the final beam); `patience` stops after that many expansions without improving the
estimated top k (0 disables the early stop).
"""
import numpy as np
from holographic_memory import VectorIndex

from ..base.module import BaseANN

METRICS = {"cosine": "angular", "normalized": "angular", "euclidean": "euclidean", "ip": "ip"}
LEARN_SEED = 24301


class HMS(BaseANN):
    def __init__(self, metric, degree, build_ef, code_bits=4, learn_queries=0):
        if metric not in METRICS:
            raise NotImplementedError(f"hms does not support the {metric} distance")
        self.metric = metric
        self.degree = degree
        self.build_ef = build_ef
        self.code_bits = code_bits
        self.learn_queries = learn_queries
        self.ef = None
        self.rerank = None
        self.reservoir = 0
        self.patience = 0
        self.index = None

    def _new_index(self):
        return VectorIndex(
            METRICS[self.metric],
            degree=self.degree,
            build_ef=self.build_ef,
            threads=1,
            code_bits=self.code_bits,
            direct_ip=self.metric == "ip",
        )

    def fit(self, X):
        self.learn_queries = 0
        self.index = self._new_index()
        self.index.fit(np.ascontiguousarray(X, dtype=np.float32))

    def fit_ood(self, X, X_learn, X_learn_neighbors):
        if self.learn_queries <= 0:
            self.fit(X)
            return
        order = np.random.default_rng(LEARN_SEED).permutation(len(X_learn))[: self.learn_queries]
        training = np.ascontiguousarray(X_learn[order], dtype=np.float32)
        self.index = self._new_index()
        self.index.fit(np.ascontiguousarray(X, dtype=np.float32), training_queries=training)

    def set_query_arguments(self, ef, rerank, reservoir=0, patience=0):
        self.ef = ef
        self.rerank = rerank
        self.reservoir = reservoir
        self.patience = patience
        self.index.set_query_arguments(ef, rerank, patience=patience, reservoir=reservoir)

    def query(self, v, n):
        return self.index.query(v, n)

    def __str__(self):
        return (
            f"HMS(metric={self.metric}, degree={self.degree}, build_ef={self.build_ef}, "
            f"code_bits={self.code_bits}, learn_queries={self.learn_queries}, ef={self.ef}, "
            f"rerank={self.rerank}, reservoir={self.reservoir}, patience={self.patience})"
        )
