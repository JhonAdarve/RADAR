"""
Ponderacion de criterios para RADAR Cibest.

Implementa BWM (Best-Worst Method, Rezaei 2015) como tecnica principal y
AHP (Saaty 1980) como alternativa, con patron Strategy intercambiable.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from loguru import logger
from scipy.optimize import minimize

from src.utils import ConsistencyError, ScoringError, save_yaml


# Tabla CI segun Rezaei (2015) por valor de a_BW
BWM_CONSISTENCY_INDEX: Dict[int, float] = {
    1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63,
    5: 2.30, 6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23,
}

# Random Index de Saaty
AHP_RANDOM_INDEX: Dict[int, float] = {
    1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12,
    6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49,
    11: 1.51, 12: 1.48, 13: 1.56, 14: 1.57, 15: 1.59,
}


class WeightingMethod(ABC):
    """Clase base abstracta para tecnicas de ponderacion multicriteria."""

    @abstractmethod
    def compute_weights(self, judgments: Dict[str, Any]) -> Dict[str, float]:
        """Calcula pesos a partir de juicios de expertos."""

    @abstractmethod
    def validate_consistency(self, judgments: Dict[str, Any]) -> Dict[str, float]:
        """Valida la consistencia de los juicios."""


class BWMWeighting(WeightingMethod):
    """BWM: Best-Worst Method (Rezaei, 2015)."""

    def __init__(self, max_consistency_ratio: float = 0.10) -> None:
        self.max_cr = max_consistency_ratio

    def compute_weights(self, judgments: Dict[str, Any]) -> Dict[str, float]:
        """Resuelve el modelo de optimizacion BWM via SLSQP."""
        criteria = judgments["criteria"]
        best = judgments["best"]
        worst = judgments["worst"]
        bto = judgments["best_to_others"]
        otw = judgments["others_to_worst"]

        if best not in criteria or worst not in criteria:
            raise ScoringError(f"best/worst no estan en criteria: {best}, {worst}")
        if best == worst:
            raise ScoringError("El criterio best y worst no pueden ser iguales")

        n = len(criteria)
        idx = {c: i for i, c in enumerate(criteria)}

        def objective(x: np.ndarray) -> float:
            return x[-1]

        def constraints_list() -> List[Dict[str, Any]]:
            cons: List[Dict[str, Any]] = []
            cons.append({"type": "eq", "fun": lambda x: np.sum(x[:n]) - 1.0})
            for c in criteria:
                if c == best:
                    continue
                a_Bj = bto[c]
                j, b = idx[c], idx[best]
                cons.append({
                    "type": "ineq",
                    "fun": lambda x, b=b, j=j, a=a_Bj: x[-1] - abs(x[b] - a * x[j]),
                })
            for c in criteria:
                if c == worst:
                    continue
                a_jW = otw[c]
                j, w = idx[c], idx[worst]
                cons.append({
                    "type": "ineq",
                    "fun": lambda x, j=j, w=w, a=a_jW: x[-1] - abs(x[j] - a * x[w]),
                })
            return cons

        x0 = np.concatenate([np.full(n, 1.0 / n), [0.1]])
        bounds = [(0.0, 1.0)] * n + [(0.0, None)]

        result = minimize(
            objective, x0=x0, method="SLSQP",
            bounds=bounds, constraints=constraints_list(),
            options={"maxiter": 500, "ftol": 1e-9},
        )

        if not result.success:
            logger.warning("Optimizacion BWM no convergio: {m}", m=result.message)

        weights_arr = result.x[:n]
        xi_star = result.x[-1]
        weights_arr = weights_arr / weights_arr.sum()

        self._last_xi = float(xi_star)
        weights = {c: float(weights_arr[idx[c]]) for c in criteria}

        logger.info(
            "BWM pesos: {w} | xi* = {x:.4f}",
            w={k: round(v, 3) for k, v in weights.items()},
            x=xi_star,
        )
        return weights

    def validate_consistency(self, judgments: Dict[str, Any]) -> Dict[str, float]:
        if not hasattr(self, "_last_xi"):
            self.compute_weights(judgments)

        worst = judgments["worst"]
        a_BW = judgments["best_to_others"][worst]
        ci = BWM_CONSISTENCY_INDEX.get(int(round(a_BW)), BWM_CONSISTENCY_INDEX[9])
        cr = self._last_xi / ci if ci > 0 else 0.0

        logger.info("BWM consistencia: xi*={x:.4f} | CI={c:.2f} | CR={r:.4f}",
                    x=self._last_xi, c=ci, r=cr)

        if cr > self.max_cr:
            raise ConsistencyError(
                f"BWM inconsistente: CR={cr:.4f} > umbral {self.max_cr}. "
                f"Solicite al ejecutivo revisar sus juicios."
            )

        return {"xi": self._last_xi, "ci": ci, "ratio": cr}


class AHPWeighting(WeightingMethod):
    """AHP: comparaciones pareadas con matriz reciproca completa (Saaty, 1980)."""

    def __init__(self, max_consistency_ratio: float = 0.10) -> None:
        self.max_cr = max_consistency_ratio

    def compute_weights(self, judgments: Dict[str, Any]) -> Dict[str, float]:
        criteria = judgments["criteria"]
        matrix = np.asarray(judgments["pairwise"], dtype=float)

        if matrix.shape != (len(criteria), len(criteria)):
            raise ScoringError(f"Matriz pairwise no cuadrada: {matrix.shape}")

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        idx_max = np.argmax(eigenvalues.real)
        principal = eigenvectors[:, idx_max].real
        weights_arr = principal / principal.sum()

        self._lambda_max = float(eigenvalues[idx_max].real)
        self._matrix = matrix

        return {c: float(weights_arr[i]) for i, c in enumerate(criteria)}

    def validate_consistency(self, judgments: Dict[str, Any]) -> Dict[str, float]:
        if not hasattr(self, "_lambda_max"):
            self.compute_weights(judgments)

        n = self._matrix.shape[0]
        ci = (self._lambda_max - n) / (n - 1) if n > 1 else 0.0
        ri = AHP_RANDOM_INDEX.get(n, 1.59)
        cr = ci / ri if ri > 0 else 0.0

        logger.info("AHP consistencia: lambda_max={l:.3f} | CI={c:.4f} | CR={r:.4f}",
                    l=self._lambda_max, c=ci, r=cr)

        if cr > self.max_cr:
            raise ConsistencyError(f"AHP inconsistente: CR={cr:.4f} > umbral {self.max_cr}")

        return {"lambda_max": self._lambda_max, "ci": ci, "ratio": cr}


def aggregate_expert_weights(
    expert_weights: List[Dict[str, float]],
    method: str = "geometric_mean",
) -> Dict[str, float]:
    """Agrega pesos de varios ejecutivos del panel BWM."""
    if not expert_weights:
        raise ScoringError("Lista de pesos expertos vacia")

    criteria = list(expert_weights[0].keys())
    matrix = np.array([[e[c] for c in criteria] for e in expert_weights])

    if method == "geometric_mean":
        aggregated = np.exp(np.mean(np.log(matrix + 1e-12), axis=0))
    elif method == "arithmetic_mean":
        aggregated = matrix.mean(axis=0)
    else:
        raise ValueError(f"Metodo de agregacion invalido: {method}")

    aggregated = aggregated / aggregated.sum()
    return {c: float(aggregated[i]) for i, c in enumerate(criteria)}


def compute_hierarchical_weights(
    dimension_weights: Dict[str, float],
    variable_weights_by_dim: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Calcula los pesos finales de cada variable multiplicando dim x variable."""
    total: Dict[str, float] = {}
    for dim, dim_w in dimension_weights.items():
        vars_in_dim = variable_weights_by_dim.get(dim, {})
        for var, vw in vars_in_dim.items():
            total[var] = dim_w * vw

    s = sum(total.values())
    if s > 0:
        total = {k: v / s for k, v in total.items()}
    return total


def export_weights(weights: Dict[str, Any], path: Path | str) -> Path:
    """Exporta la estructura de pesos a YAML."""
    return save_yaml(weights, path)


def build_weighting_method(
    name: str,
    max_consistency_ratio: float = 0.10,
) -> WeightingMethod:
    """Factoria de tecnicas de ponderacion."""
    name_lower = name.lower()
    if name_lower == "bwm":
        return BWMWeighting(max_consistency_ratio=max_consistency_ratio)
    if name_lower == "ahp":
        return AHPWeighting(max_consistency_ratio=max_consistency_ratio)
    raise ValueError(f"Tecnica de ponderacion no soportada: {name}")
