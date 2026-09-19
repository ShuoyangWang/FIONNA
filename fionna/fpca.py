from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if len(grid) == 1:
        return np.ones(1)
    step = np.diff(grid)
    weights = np.zeros(len(grid))
    weights[0], weights[-1] = step[0] / 2, step[-1] / 2
    weights[1:-1] = (step[:-1] + step[1:]) / 2
    return weights


@dataclass
class FPCAResult:
    grid: np.ndarray
    eigvals: np.ndarray
    eigfuns: np.ndarray
    scores_std: np.ndarray
    mean_function: np.ndarray
    weights: np.ndarray

    def transform_new(self, trajectories: np.ndarray) -> np.ndarray:
        centered = np.asarray(trajectories, dtype=float) - self.mean_function
        raw = (centered * self.weights) @ self.eigfuns
        return raw / np.sqrt(self.eigvals)


def fpca_dense(trajectories: np.ndarray, grid: np.ndarray,
               n_components: int) -> FPCAResult:
    trajectories = np.asarray(trajectories, dtype=float)
    grid = np.asarray(grid, dtype=float)
    n, points = trajectories.shape
    if len(grid) != points:
        raise ValueError("grid length must match trajectory width")

    weights = trapezoid_weights(grid)
    mean = trajectories.mean(axis=0)
    centered = trajectories - mean
    covariance = centered.T @ centered / n
    root_weights = np.sqrt(weights)
    symmetric = (root_weights[:, None] * covariance) * root_weights[None, :]
    symmetric = (symmetric + symmetric.T) / 2

    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    keep = values > 1e-12 * max(values[0], 1e-12)
    values, vectors = values[keep], vectors[:, keep]
    k = min(int(n_components), len(values))
    values, vectors = values[:k], vectors[:, :k]
    functions = vectors / root_weights[:, None]

    for column in range(k):
        anchor = int(np.argmax(np.abs(functions[:, column])))
        if functions[anchor, column] < 0:
            functions[:, column] *= -1

    raw = (centered * weights) @ functions
    scores = raw / np.sqrt(values)
    return FPCAResult(grid, values, functions, scores, mean, weights)
