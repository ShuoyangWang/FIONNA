from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SimData:
    Y: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    grid: np.ndarray
    alpha0: float
    delta0: float
    theta0: float
    M: np.ndarray | None = None
    M_list: list[np.ndarray] | None = None


def _basis(grid: np.ndarray, count: int) -> np.ndarray:
    return np.column_stack([
        np.sqrt(2.0) * np.sin(k * np.pi * grid) for k in range(1, count + 1)
    ])


def _vector(length: int, scale: float, phase: float = 0.0) -> np.ndarray:
    index = np.arange(1, length + 1, dtype=float)
    return scale * np.sin(index + phase) / np.sqrt(index)


def _matrix(rows: int, columns: int, scale: float,
            phase: float = 0.0) -> np.ndarray:
    row = np.arange(1, rows + 1, dtype=float)[:, None]
    column = np.arange(1, columns + 1, dtype=float)[None, :]
    return scale * np.sin(0.7 * row * column + phase) / np.sqrt(row)


def _z(confounders: np.ndarray, column: int) -> np.ndarray:
    return confounders[:, column] if column < confounders.shape[1] else np.zeros(len(confounders))


def generate_linear_null(n: int = 400, p: int = 8, T: int = 40,
                         seed: int = 20260710) -> SimData:
    """Linear-confounding null used in the R=200 size experiment."""
    rng = np.random.default_rng(seed)
    basis_count = 20
    grid = np.linspace(0, 1, T)
    basis = _basis(grid, basis_count)
    eigenvalues = np.array([(k + 1.0) ** -2 for k in range(basis_count)])
    Z = rng.standard_normal((n, p))
    mean_x = Z @ _vector(p, 0.35, 0.1)
    X = mean_x + rng.standard_normal(n)

    beta = np.array([np.exp(-0.5 * k) if k < 6 else 0.0
                     for k in range(basis_count)])
    a_path = np.zeros(basis_count)
    mediator_mean = Z @ _matrix(p, basis_count, 0.18, 0.2)
    score_noise = rng.standard_normal((n, basis_count)) * np.sqrt(eigenvalues)
    mediator_coefficients = np.outer(X, a_path) + mediator_mean + score_noise
    M = mediator_coefficients @ basis.T + 0.3 * rng.standard_normal((n, T))
    outcome_confounding = Z @ _vector(p, 0.35, 1.1)
    Y = X + mediator_coefficients @ beta + outcome_confounding + 0.5 * rng.standard_normal(n)
    return SimData(Y=Y, X=X, Z=Z, M=M, M_list=None, grid=grid,
                   alpha0=1.0, delta0=0.0, theta0=1.0)


def generate_shared_profile_null(n: int = 400, p: int = 8, T: int = 60,
                                 seed: int = 20267010) -> SimData:
    """Four-mediator shared-profile null used in the R=200 size experiment."""
    rng = np.random.default_rng(seed)
    mediator_count, basis_count = 4, 28
    grid = np.linspace(0, 1, max(T, 60))
    basis = _basis(grid, basis_count)
    eigenvalues = np.array([(k + 1.0) ** -1.15 for k in range(basis_count)])
    Z = rng.standard_normal((n, p))
    shared1 = np.sin(_z(Z, 0) * _z(Z, 1))
    shared2 = _z(Z, 2) ** 2 - 1
    shared3 = np.tanh(_z(Z, 3))
    mean_x = 0.55 * shared1 + 0.35 * shared2 + 0.45 * shared3
    X = mean_x + rng.standard_normal(n)

    index = np.arange(basis_count, dtype=float)
    weights_a = [0.50, 0.42, 0.36, 0.30]
    weights_b = [0.48, 0.38, 0.34, 0.28]
    b_paths = []
    for mediator, weight in enumerate(weights_b):
        phase = 0.35 * mediator
        path = weight * np.exp(-0.18 * index) * (1 + 0.10 * np.cos(0.7 * index + phase))
        path[index >= 14] = 0
        b_paths.append(path)

    shared_noise = rng.standard_normal((n, basis_count)) * np.sqrt(eigenvalues)
    mediator_matrices, mediator_coefficients = [], []
    active = min(p, 12)
    for mediator in range(mediator_count):
        phase = 0.6 * mediator
        nonlinear = (
            shared1[:, None] * _vector(basis_count, 0.12 + 0.02 * mediator, 0.4 + phase)
            + shared2[:, None] * _vector(basis_count, 0.08, 1.3 + phase)
            + shared3[:, None] * _vector(basis_count, 0.07, 2.1 + phase)
            + np.tanh(_z(Z, 2) * _z(Z, 3))[:, None]
            * _vector(basis_count, 0.04, 2.8 + phase)
        )
        linear = Z[:, :active] @ _matrix(
            active, basis_count, 0.035 + 0.005 * mediator, 2.7 + phase
        )
        specific = rng.standard_normal((n, basis_count)) * np.sqrt(eigenvalues)
        noise = 0.82 * shared_noise + np.sqrt(1 - 0.82**2) * specific
        coefficients = nonlinear + linear + noise
        observed = coefficients @ basis.T
        observed += (0.38 + 0.04 * mediator) * rng.standard_normal(observed.shape)
        mediator_coefficients.append(coefficients)
        mediator_matrices.append(observed)

    signal = sum(coefficients @ path
                 for coefficients, path in zip(mediator_coefficients, b_paths))
    outcome_confounding = (
        0.55 * np.sin(_z(Z, 0) * _z(Z, 2))
        + 0.35 * np.log1p(np.abs(_z(Z, 3) * _z(Z, 4)))
        + 0.35 * (_z(Z, 1) ** 2 - 1)
        + 0.20 * np.tanh(_z(Z, 5) * _z(Z, 6))
    )
    Y = X + signal + outcome_confounding + 0.55 * rng.standard_normal(n)
    return SimData(Y=Y, X=X, Z=Z, M=mediator_matrices[0],
                   M_list=mediator_matrices, grid=grid,
                   alpha0=1.0, delta0=0.0, theta0=1.0)
