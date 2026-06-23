"""
Ranking de paises para RADAR Cibest.

Implementa TOPSIS (Hwang & Yoon, 1981) como tecnica principal de ranking,
con scores parciales por dimension que alimentan el motor de senales,
y VIKOR (Opricovic & Tzeng, 2004) como validacion cruzada.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from src.utils import ScoringError


class RankingMethod(ABC):
    """Clase base abstracta para tecnicas de ranking multicriteria."""

    @abstractmethod
    def rank(
        self,
        decision_matrix: pd.DataFrame,
        weights: Dict[str, float],
        variable_catalog: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Ejecuta el ranking de alternativas."""


class TOPSISRanking(RankingMethod):
    """TOPSIS: Technique for Order of Preference by Similarity to Ideal Solution."""

    def __init__(self, distance_metric: str = "euclidean") -> None:
        self.distance_metric = distance_metric

    @staticmethod
    def apply_weights(
        normalized_matrix: pd.DataFrame,
        weights: Dict[str, float],
    ) -> pd.DataFrame:
        """Aplica V_ij = w_j * R_ij."""
        missing = [v for v in normalized_matrix.columns if v not in weights]
        if missing:
            raise ScoringError(f"Faltan pesos para variables: {missing}")
        weight_series = pd.Series(weights).reindex(normalized_matrix.columns)
        return normalized_matrix.mul(weight_series, axis=1)

    @staticmethod
    def compute_ideal_solutions(weighted_matrix: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """Calcula A+ (max) y A- (min) por columna."""
        return weighted_matrix.max(), weighted_matrix.min()

    def _distance(self, row: np.ndarray, reference: np.ndarray) -> float:
        diff = row - reference
        if self.distance_metric == "euclidean":
            return float(np.sqrt(np.sum(diff ** 2)))
        if self.distance_metric == "manhattan":
            return float(np.sum(np.abs(diff)))
        if self.distance_metric == "chebyshev":
            return float(np.max(np.abs(diff)))
        raise ScoringError(f"Metrica de distancia invalida: {self.distance_metric}")

    def compute_distances(
        self,
        weighted_matrix: pd.DataFrame,
        ideal_pos: pd.Series,
        ideal_neg: pd.Series,
    ) -> pd.DataFrame:
        """Distancias d+ y d- de cada pais."""
        ip = ideal_pos.values
        ine = ideal_neg.values
        d_pos = weighted_matrix.apply(lambda r: self._distance(r.values, ip), axis=1)
        d_neg = weighted_matrix.apply(lambda r: self._distance(r.values, ine), axis=1)
        return pd.DataFrame({"d_pos": d_pos, "d_neg": d_neg})

    @staticmethod
    def compute_closeness_coefficient(d_pos: pd.Series, d_neg: pd.Series) -> pd.Series:
        """C* = d- / (d+ + d-)."""
        denom = d_pos + d_neg
        cc = d_neg / denom.replace(0, np.nan)
        return cc.fillna(0.0)

    def _compute_dimension_scores(
        self,
        weighted_matrix: pd.DataFrame,
        variable_catalog: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Score parcial por dimension: TOPSIS sobre cada subconjunto."""
        by_dim: Dict[str, List[str]] = {}
        for var in weighted_matrix.columns:
            meta = variable_catalog.get(var)
            if meta is None:
                continue
            by_dim.setdefault(meta["dimension"], []).append(var)

        dim_scores: Dict[str, pd.Series] = {}
        for dim, vars_list in by_dim.items():
            sub_matrix = weighted_matrix[vars_list]
            if sub_matrix.sum().sum() == 0:
                dim_scores[dim] = pd.Series(0.0, index=weighted_matrix.index)
                continue
            ip, ine = self.compute_ideal_solutions(sub_matrix)
            distances = self.compute_distances(sub_matrix, ip, ine)
            cc = self.compute_closeness_coefficient(distances["d_pos"], distances["d_neg"])
            dim_scores[dim] = cc

        return pd.DataFrame(dim_scores)

    def rank(
        self,
        decision_matrix: pd.DataFrame,
        weights: Dict[str, float],
        variable_catalog: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Ejecuta el flujo TOPSIS completo."""
        weighted = self.apply_weights(decision_matrix, weights)
        ideal_pos, ideal_neg = self.compute_ideal_solutions(weighted)
        distances = self.compute_distances(weighted, ideal_pos, ideal_neg)
        cc = self.compute_closeness_coefficient(distances["d_pos"], distances["d_neg"])
        dim_scores = self._compute_dimension_scores(weighted, variable_catalog)

        out = pd.DataFrame({"score": cc}).join(distances)
        out = out.join(dim_scores.add_prefix("score_"))
        out["rank"] = out["score"].rank(ascending=False, method="min").astype(int)
        out = out.sort_values("score", ascending=False)

        logger.info(
            "TOPSIS completado: {n} paises | score max={s:.3f} ({c})",
            n=len(out), s=out["score"].iloc[0], c=out.index[0],
        )
        return out


class VIKORRanking(RankingMethod):
    """VIKOR: Compromise Ranking Method (Opricovic & Tzeng, 2004)."""

    def __init__(self, v: float = 0.5) -> None:
        if not 0 <= v <= 1:
            raise ScoringError("El parametro v de VIKOR debe estar en [0, 1]")
        self.v = v

    def rank(
        self,
        decision_matrix: pd.DataFrame,
        weights: Dict[str, float],
        variable_catalog: Dict[str, Dict[str, Any]],
    ) -> pd.DataFrame:
        """Ejecuta VIKOR sobre la matriz de decision."""
        f_best = decision_matrix.max()
        f_worst = decision_matrix.min()
        rng = (f_best - f_worst).replace(0, np.nan)

        weight_series = pd.Series(weights).reindex(decision_matrix.columns)
        normalized_gap = (f_best - decision_matrix) / rng
        weighted_gap = normalized_gap.mul(weight_series, axis=1).fillna(0)

        s = weighted_gap.sum(axis=1)
        r = weighted_gap.max(axis=1)

        s_star, s_minus = s.min(), s.max()
        r_star, r_minus = r.min(), r.max()

        q = self.v * (s - s_star) / max(s_minus - s_star, 1e-12) + (
            (1 - self.v) * (r - r_star) / max(r_minus - r_star, 1e-12)
        )

        q_norm = (q - q.min()) / max(q.max() - q.min(), 1e-12)
        score = 1 - q_norm

        out = pd.DataFrame({"score": score, "S": s, "R": r, "Q": q})
        out["rank"] = out["score"].rank(ascending=False, method="min").astype(int)
        out = out.sort_values("score", ascending=False)

        logger.info("VIKOR completado: {n} paises | v={v}", n=len(out), v=self.v)
        return out


def build_ranking_method(name: str, **kwargs: Any) -> RankingMethod:
    """Factoria de tecnicas de ranking."""
    name_lower = name.lower()
    if name_lower == "topsis":
        return TOPSISRanking(**kwargs)
    if name_lower == "vikor":
        return VIKORRanking(**kwargs)
    raise ValueError(f"Tecnica de ranking no soportada: {name}")
