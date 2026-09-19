from __future__ import annotations

import json
import math
import time
from typing import Any

import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .fpca import fpca_dense
from .ridge import ridge_fit, select_gamma_lambda


ARCHITECTURES = [(2, 32), (2, 64), (2, 128), (3, 32), (3, 64), (3, 128)]
E3 = {
    "solver": "adam",
    "learning_rate_init": 5e-4,
    "batch_size": 64,
    "max_iter": 900,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 30,
    "alpha": 1e-4,
    "tol": 1e-4,
}


class TunedNuisance:
    def __init__(self, seed: int):
        self.seed = int(seed)

    def _model(self, depth: int, width: int, seed: int):
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(width,) * depth,
                activation="relu",
                random_state=int(seed),
                **E3,
            ),
        )

    def fit(self, features: np.ndarray, outcome: np.ndarray) -> "TunedNuisance":
        folds = list(KFold(3, shuffle=True, random_state=self.seed + 701).split(features))
        candidates = []
        for depth, width in ARCHITECTURES:
            losses = []
            for inner, (train, test) in enumerate(folds):
                model = self._model(depth, width,
                                    self.seed + inner * 1009 + depth * 37 + width)
                model.fit(features[train], outcome[train])
                error = np.asarray(outcome[test]) - np.asarray(model.predict(features[test]))
                if error.ndim == 1:
                    losses.append(float(np.mean(error**2)))
                else:
                    scale = np.std(outcome[train], axis=0, ddof=1)
                    scale[scale < 1e-8] = 1
                    losses.append(float(np.mean((error / scale) ** 2)))
            candidates.append((float(np.mean(losses)), depth, width))

        self.loss_, self.depth_, self.width_ = min(
            candidates, key=lambda item: (item[0], item[1], item[2])
        )
        self.model_ = self._model(self.depth_, self.width_, self.seed + 900001)
        self.model_.fit(features, outcome)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.asarray(self.model_.predict(features))


def _model_log(model: TunedNuisance) -> dict[str, Any]:
    return {"depth": model.depth_, "width": model.width_, "inner_loss": model.loss_}


def _mediator_blocks(data: Any, K: int | list[int] | tuple[int, ...]):
    if getattr(data, "M_list", None) is not None:
        raw = data.M_list
        blocks = ([np.asarray(raw[j], dtype=float) for j in range(raw.shape[0])]
                  if isinstance(raw, np.ndarray)
                  else [np.asarray(block, dtype=float) for block in raw])
    else:
        blocks = [np.asarray(data.M, dtype=float)]
    levels = [int(K)] * len(blocks) if np.isscalar(K) else [int(value) for value in K]
    if len(levels) != len(blocks) or any(value < 1 for value in levels):
        raise ValueError("one positive K is required per mediator")
    return blocks, levels


def _train_test_scores(mediator: np.ndarray, train: np.ndarray, test: np.ndarray,
                       grid: np.ndarray, K: int):
    if mediator.shape[1] <= 300:
        fitted = fpca_dense(mediator[train], grid, K)
        return fitted.scores_std, fitted.transform_new(mediator[test])
    location = mediator[train].mean(axis=0)
    pca = PCA(n_components=K, svd_solver="randomized", random_state=1941)
    train_scores = pca.fit_transform(mediator[train] - location)
    test_scores = pca.transform(mediator[test] - location)
    center = train_scores.mean(axis=0)
    scale = train_scores.std(axis=0, ddof=1)
    scale[scale < 1e-10] = 1
    return (train_scores - center) / scale, (test_scores - center) / scale


def _fit_fold(uy: np.ndarray, ux: np.ndarray, scores: np.ndarray,
              uy_train: np.ndarray, ux_train: np.ndarray,
              scores_train: np.ndarray, seed: int):
    count, K = scores.shape
    projection, _, rank, _ = np.linalg.lstsq(scores, ux, rcond=None)
    if rank < K:
        raise np.linalg.LinAlgError(f"score design rank {rank} < K {K}")
    residual_x = ux - scores @ projection
    joint = np.linalg.lstsq(np.column_stack([ux, scores]), uy, rcond=None)[0]
    alpha = float(residual_x @ uy / (residual_x @ ux))
    theta = float(ux @ uy / (ux @ ux))
    delta = theta - alpha

    penalty, penalty_grid = select_gamma_lambda(
        np.column_stack([ux_train, scores_train]), uy_train, seed
    )
    gamma = ridge_fit(np.column_stack([ux, scores]), uy, penalty)[1:]
    h_alpha = float(np.mean(residual_x * ux))
    h_theta = float(np.mean(ux**2))
    error_alpha = uy - alpha * ux - scores @ gamma
    error_theta = uy - theta * ux
    psi_alpha = residual_x * error_alpha / h_alpha
    psi_theta = ux * error_theta / h_theta
    return {
        "theta": theta,
        "alpha": alpha,
        "delta": delta,
        "psi": np.column_stack([psi_theta, psi_alpha]),
        "fwl_gap": abs(alpha - float(joint[0])),
        "identity_gap": abs(theta - alpha - delta),
        "normal_gap": float(np.max(np.abs(scores.T @ residual_x / count))),
        "score_min_eig": float(np.linalg.eigvalsh(scores.T @ scores / count).min()),
        "h_alpha": h_alpha,
        "h_theta": h_theta,
        "gamma_lambda": penalty,
        "gamma_grid": penalty_grid,
    }


def fit_fionna(data: Any, K: int | list[int] | tuple[int, ...], seed: int = 1,
               return_observations: bool = False) -> dict[str, Any]:
    """Fit the revised foldwise-FPCA FIONNA estimator."""
    started = time.time()
    y, x, z = map(np.asarray, (data.Y, data.X, data.Z))
    blocks, levels = _mediator_blocks(data, K)
    n = len(y)
    estimates, logs = [], []
    psi = np.empty((n, 2))
    residual_x = np.empty(n)
    fold_ids = np.full(n, -1)
    max_fwl = max_identity = max_normal = 0.0
    min_score_eig = math.inf

    folds = KFold(5, shuffle=True, random_state=seed)
    for fold, (train, test) in enumerate(folds.split(y)):
        pairs = [_train_test_scores(mediator, train, test, data.grid, level)
                 for mediator, level in zip(blocks, levels)]
        scores_train = np.column_stack([pair[0] for pair in pairs])
        scores_test = np.column_stack([pair[1] for pair in pairs])

        models = []
        for offset, outcome in ((1, y[train]), (2, x[train]), (3, scores_train)):
            model = TunedNuisance(seed + fold * 10000 + offset)
            model.fit(z[train], outcome)
            models.append(model)
        model_y, model_x, model_s = models

        py_train, px_train = model_y.predict(z[train]), model_x.predict(z[train])
        ps_train = model_s.predict(z[train])
        py_test, px_test = model_y.predict(z[test]), model_x.predict(z[test])
        ps_test = model_s.predict(z[test])
        if ps_train.ndim == 1:
            ps_train = ps_train[:, None]
        if ps_test.ndim == 1:
            ps_test = ps_test[:, None]

        answer = _fit_fold(
            y[test] - py_test,
            x[test] - px_test,
            scores_test - ps_test,
            y[train] - py_train,
            x[train] - px_train,
            scores_train - ps_train,
            seed + 7000 + fold,
        )
        estimates.append((answer["theta"], answer["alpha"], len(test)))
        psi[test] = answer["psi"]
        residual_x[test] = x[test] - px_test
        fold_ids[test] = fold
        max_fwl = max(max_fwl, answer["fwl_gap"])
        max_identity = max(max_identity, answer["identity_gap"])
        max_normal = max(max_normal, answer["normal_gap"])
        min_score_eig = min(min_score_eig, answer["score_min_eig"])
        logs.append({
            "fold": fold,
            "K_by_block": levels,
            "Y": _model_log(model_y),
            "X": _model_log(model_x),
            "S": _model_log(model_s),
            "gamma_lambda": answer["gamma_lambda"],
            "gamma_grid": answer["gamma_grid"],
        })

    theta = sum(value * size for value, _, size in estimates) / n
    alpha = sum(value * size for _, value, size in estimates) / n
    delta = theta - alpha
    covariance = psi.T @ psi / n**2
    covariance = (covariance + covariance.T) / 2
    contrast = np.array([1.0, -1.0])
    se_theta, se_alpha = np.sqrt(np.maximum(np.diag(covariance), 0))
    se_delta = float(np.sqrt(max(float(contrast @ covariance @ contrast), 0)))
    p_delta = float(2 * norm.sf(abs(delta / se_delta))) if se_delta > 0 else float(delta != 0)

    result = {
        "theta_hat": theta,
        "alpha_hat": alpha,
        "delta_hat": delta,
        "se_theta": float(se_theta),
        "se_alpha": float(se_alpha),
        "se_delta": se_delta,
        "cov_ta": float(covariance[0, 1]),
        "p_delta": p_delta,
        "ci_delta": (delta - norm.ppf(0.975) * se_delta,
                     delta + norm.ppf(0.975) * se_delta),
        "fwl_gap": max_fwl,
        "identity_gap": max_identity,
        "normal_gap": max_normal,
        "min_score_eig": min_score_eig,
        "min_cov_eig": float(np.linalg.eigvalsh(covariance).min()),
        "selected_details": json.dumps(logs, separators=(",", ":")),
        "elapsed_sec": time.time() - started,
    }
    if return_observations:
        result["observations"] = {
            "fold": fold_ids.tolist(),
            "residual_x": residual_x.tolist(),
            "psi_theta": psi[:, 0].tolist(),
            "psi_alpha": psi[:, 1].tolist(),
        }
    return result
