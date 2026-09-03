"""CuPy primitives for the source-free CBIB-1 parity fixture.

There is deliberately no file or payload handling in this module.
"""

from __future__ import annotations

import cupy as cp
import numpy as np

from clustered_ib_core import ALPHABET, KT_ALPHA


def model_counts_gpu(group_labels: cp.ndarray,
                     assignments: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    q = cp.asarray(group_labels, dtype=cp.uint8)
    u = cp.asarray(assignments, dtype=cp.uint8)
    if q.ndim != 2 or u.shape != (q.shape[1],):
        raise ValueError("model count shapes")
    if bool(cp.any(q >= ALPHABET).item()) or bool(cp.any(u >= 2).item()):
        raise ValueError("model count values")
    latent = cp.bincount(u, minlength=2).astype(cp.int64)
    conditional = cp.zeros((q.shape[0], 2, ALPHABET), dtype=cp.int64)
    for expert in range(q.shape[0]):
        for state in range(2):
            conditional[expert, state] = cp.bincount(
                q[expert, u == state], minlength=ALPHABET
            )
    return latent, conditional


def assignment_costs_gpu(group_labels: cp.ndarray, latent: cp.ndarray,
                         conditional: cp.ndarray) -> cp.ndarray:
    q = cp.asarray(group_labels, dtype=cp.uint8)
    latent = cp.asarray(latent, dtype=cp.int64)
    conditional = cp.asarray(conditional, dtype=cp.int64)
    k, n = q.shape
    if latent.shape != (2,) or conditional.shape != (k, 2, ALPHABET):
        raise ValueError("assignment model shapes")
    latent_logp = cp.log2((latent.astype(cp.float64) + KT_ALPHA) /
                          (latent.sum(dtype=cp.int64) + 2 * KT_ALPHA))
    costs = cp.empty((2, n), dtype=cp.float64)
    for state in range(2):
        cost = cp.full(n, -latent_logp[state], dtype=cp.float64)
        for expert in range(k):
            logp = cp.log2(
                (conditional[expert, state].astype(cp.float64) + KT_ALPHA)
                / (latent[state] + ALPHABET * KT_ALPHA)
            )
            cost -= logp[q[expert]]
        costs[state] = cost
    return costs


def evaluate_binary_model_gpu(group_labels: cp.ndarray, latent: cp.ndarray,
                              conditional: cp.ndarray) -> dict:
    q = cp.asarray(group_labels, dtype=cp.uint8)
    latent = cp.asarray(latent, dtype=cp.int64)
    conditional = cp.asarray(conditional, dtype=cp.int64)
    costs = assignment_costs_gpu(q, latent, conditional)
    u = cp.argmin(costs, axis=0).astype(cp.uint8)
    latent_logp = cp.log2((latent.astype(cp.float64) + KT_ALPHA) /
                          (latent.sum(dtype=cp.int64) + 2 * KT_ALPHA))
    latent_bits = float((-cp.sum(latent_logp[u], dtype=cp.float64)).item())
    private = []
    for expert in range(q.shape[0]):
        bits = cp.asarray(0.0, dtype=cp.float64)
        for state in range(2):
            mask = u == state
            logp = cp.log2(
                (conditional[expert, state].astype(cp.float64) + KT_ALPHA)
                / (latent[state] + ALPHABET * KT_ALPHA)
            )
            bits -= cp.sum(logp[q[expert, mask]], dtype=cp.float64)
        private.append(float(bits.item()))
    return {
        "assignments": cp.asnumpy(u),
        "latent_bits": latent_bits,
        "private_bits": private,
        "total_bits": latent_bits + sum(private),
    }


def cpu_bytes(array: cp.ndarray) -> bytes:
    return np.ascontiguousarray(cp.asnumpy(array)).tobytes(order="C")

