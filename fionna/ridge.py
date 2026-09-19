from __future__ import annotations

import numpy as np
from sklearn.model_selection import KFold


def ridge_fit(design: np.ndarray, outcome: np.ndarray, penalty: float) -> np.ndarray:
    p = design.shape[1]
    diagonal = np.zeros(p)
    diagonal[1:] = penalty
    return np.linalg.solve(
        design.T @ design / len(design) + np.diag(diagonal),
        design.T @ outcome / len(design),
    )


def select_gamma_lambda(design: np.ndarray, outcome: np.ndarray,
                        seed: int) -> tuple[float, list[dict[str, float]]]:
    grid = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    folds = KFold(3, shuffle=True, random_state=seed)
    scores = []
    for penalty in grid:
        losses = []
        for train, test in folds.split(design):
            try:
                coefficient = ridge_fit(design[train], outcome[train], penalty)
                losses.append(float(np.mean((outcome[test] - design[test] @ coefficient) ** 2)))
            except np.linalg.LinAlgError:
                losses.append(float("inf"))
        scores.append({"lambda": penalty, "loss": float(np.mean(losses))})
    best = min(scores, key=lambda item: (item["loss"], -item["lambda"]))
    return float(best["lambda"]), scores
