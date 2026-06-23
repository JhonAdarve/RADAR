from typing import Dict, Any, Optional, List
import pandas as pd
from src.scoring.ranking import TOPSISRanking


def compute_score_contributions(
    decision_matrix: pd.DataFrame,
    weights_audit: pd.DataFrame,
    business_line: str,
) -> pd.DataFrame:
    """Calcula contribuciones y brechas por variable para una línea de negocio.

    La contribución se calcula como:

        contribution = normalized_value * final_topsis_weight

    La brecha se calcula como:

        shortfall = (1 - normalized_value) * final_topsis_weight

    Args:
        decision_matrix: Matriz normalizada país x variable, con dirección aplicada.
        weights_audit: Auditoría de pesos efectivos por línea de negocio.
        business_line: Código de línea de negocio, por ejemplo 'PF', 'IB', 'AD'.

    Returns:
        DataFrame largo con contribución y brecha por país-variable.
    """

    weight_slice = weights_audit[
        (weights_audit["business_line"] == business_line)
        & (weights_audit["in_decision_matrix"])
    ].copy()

    if weight_slice.empty:
        raise ValueError(
            f"No hay pesos efectivos para la línea de negocio: {business_line}"
        )

    weights = (
        weight_slice
        .set_index("variable")["final_topsis_weight"]
        .reindex(decision_matrix.columns)
        .dropna()
    )

    missing_weights = [
        col for col in decision_matrix.columns
        if col not in weights.index
    ]

    if missing_weights:
        raise ValueError(
            f"Faltan pesos para variables en decision_matrix: {missing_weights}"
        )

    weighted_values = decision_matrix[weights.index].mul(weights, axis=1)
    weighted_shortfalls = (1.0 - decision_matrix[weights.index]).mul(weights, axis=1)

    contributions_long = (
        weighted_values
        .reset_index()
        .melt(
            id_vars=decision_matrix.index.name or "country_iso3",
            var_name="variable",
            value_name="contribution",
        )
    )

    shortfalls_long = (
        weighted_shortfalls
        .reset_index()
        .melt(
            id_vars=decision_matrix.index.name or "country_iso3",
            var_name="variable",
            value_name="shortfall",
        )
    )

    country_col = decision_matrix.index.name or "country_iso3"

    out = contributions_long.merge(
        shortfalls_long,
        on=[country_col, "variable"],
        how="left",
    )

    out = out.rename(columns={country_col: "country_iso3"})

    metadata_cols = [
        "business_line",
        "dimension",
        "variable",
        "final_topsis_weight",
        "has_override",
        "override_variable_weight_in_dim",
    ]

    metadata = weight_slice[
        [c for c in metadata_cols if c in weight_slice.columns]
    ].drop_duplicates(subset=["variable"])

    out = out.merge(
        metadata,
        on="variable",
        how="left",
    )

    out["normalized_value"] = out["contribution"] / out["final_topsis_weight"]
    out["business_line"] = business_line

    return out[
        [
            "business_line",
            "country_iso3",
            "dimension",
            "variable",
            "normalized_value",
            "final_topsis_weight",
            "contribution",
            "shortfall",
            "has_override",
            "override_variable_weight_in_dim",
        ]
    ].sort_values(
        ["country_iso3", "contribution"],
        ascending=[True, False],
    )


def compute_all_business_line_contributions(
    decision_matrix: pd.DataFrame,
    weights_audit: pd.DataFrame,
    business_lines: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Calcula contribuciones por variable para todas las líneas de negocio.

    Args:
        decision_matrix: Matriz normalizada país x variable.
        weights_audit: Auditoría de pesos efectivos.
        business_lines: Lista opcional de líneas. Si None, usa todas.

    Returns:
        Diccionario {business_line: DataFrame de contribuciones}.
    """

    if business_lines is None:
        business_lines = sorted(weights_audit["business_line"].unique())

    return {
        business_line: compute_score_contributions(
            decision_matrix=decision_matrix,
            weights_audit=weights_audit,
            business_line=business_line,
        )
        for business_line in business_lines
    }


def get_top_contributors(
    contributions: pd.DataFrame,
    country_iso3: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Retorna las variables que más contribuyen positivamente para un país."""

    return (
        contributions[
            contributions["country_iso3"] == country_iso3
        ]
        .sort_values("contribution", ascending=False)
        .head(top_n)
        [
            [
                "business_line",
                "country_iso3",
                "dimension",
                "variable",
                "normalized_value",
                "final_topsis_weight",
                "contribution",
            ]
        ]
    )


def get_top_shortfalls(
    contributions: pd.DataFrame,
    country_iso3: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Retorna las variables que más limitan el atractivo relativo de un país."""

    return (
        contributions[
            contributions["country_iso3"] == country_iso3
        ]
        .sort_values("shortfall", ascending=False)
        .head(top_n)
        [
            [
                "business_line",
                "country_iso3",
                "dimension",
                "variable",
                "normalized_value",
                "final_topsis_weight",
                "shortfall",
            ]
        ]
    )


def summarize_contributions_by_dimension(
    contributions: pd.DataFrame,
) -> pd.DataFrame:
    """Resume contribuciones y brechas por país, línea y dimensión."""

    return (
        contributions
        .groupby(
            ["business_line", "country_iso3", "dimension"],
            as_index=False,
        )
        .agg(
            contribution=("contribution", "sum"),
            shortfall=("shortfall", "sum"),
        )
        .sort_values(
            ["business_line", "country_iso3", "contribution"],
            ascending=[True, True, False],
        )
    )


def generate_country_line_explanation(
    contributions: pd.DataFrame,
    country_iso3: str,
    top_n: int = 3,
) -> str:
    """Genera una explicación ejecutiva automática para país-línea."""

    country_df = contributions[
        contributions["country_iso3"] == country_iso3
    ].copy()

    if country_df.empty:
        return f"No hay contribuciones disponibles para {country_iso3}."

    business_line = country_df["business_line"].iloc[0]

    top_drivers = (
        country_df
        .sort_values("contribution", ascending=False)
        .head(top_n)
    )

    top_constraints = (
        country_df
        .sort_values("shortfall", ascending=False)
        .head(top_n)
    )

    driver_text = ", ".join(
        [
            f"{row.variable} ({row.contribution:.3f})"
            for row in top_drivers.itertuples()
        ]
    )

    constraint_text = ", ".join(
        [
            f"{row.variable} ({row.shortfall:.3f})"
            for row in top_constraints.itertuples()
        ]
    )

    return (
        f"{country_iso3} en {business_line} está impulsado principalmente por "
        f"{driver_text}. Sus principales restricciones relativas son "
        f"{constraint_text}."
    )


def compare_country_across_lines(
    contrib_by_line: Dict[str, pd.DataFrame],
    country_iso3: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Compara los principales drivers de un país entre líneas de negocio."""

    frames = []

    for business_line, contributions in contrib_by_line.items():
        top = (
            contributions[
                contributions["country_iso3"] == country_iso3
            ]
            .sort_values("contribution", ascending=False)
            .head(top_n)
            .copy()
        )

        frames.append(top)

    return pd.concat(frames, ignore_index=True)[
        [
            "business_line",
            "country_iso3",
            "dimension",
            "variable",
            "normalized_value",
            "final_topsis_weight",
            "contribution",
            "shortfall",
        ]
    ]


def compare_countries_in_line(
    contributions: pd.DataFrame,
    countries: List[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Compara contribuciones de varios países en una misma línea."""

    subset = contributions[
        contributions["country_iso3"].isin(countries)
    ].copy()

    top_variables = (
        subset
        .groupby("variable")["contribution"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )

    return (
        subset[
            subset["variable"].isin(top_variables)
        ]
        .pivot_table(
            index=["dimension", "variable"],
            columns="country_iso3",
            values="contribution",
            aggfunc="sum",
            fill_value=0.0,
        )
        .reset_index()
    )


def build_explainability_table_for_line(
    ranking_df: pd.DataFrame,
    contributions: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Construye tabla ejecutiva de ranking con drivers y restricciones."""

    ranking = ranking_df.copy()

    if "country_iso3" in ranking.columns:
        ranking = ranking.set_index("country_iso3")

    rows = []

    for country in ranking.index:
        country_contrib = contributions[
            contributions["country_iso3"] == country
        ].copy()

        top_drivers = (
            country_contrib
            .sort_values("contribution", ascending=False)
            .head(top_n)
        )

        top_constraints = (
            country_contrib
            .sort_values("shortfall", ascending=False)
            .head(top_n)
        )

        rows.append(
            {
                "country_iso3": country,
                "score": ranking.loc[country, "score"],
                "rank": ranking.loc[country, "rank"],
                "top_drivers": "; ".join(top_drivers["variable"].tolist()),
                "top_constraints": "; ".join(top_constraints["variable"].tolist()),
            }
        )

    return pd.DataFrame(rows).sort_values("rank")


def _score_series_from_ranking(ranking_df: pd.DataFrame) -> pd.Series:
    """Extrae una serie de score indexada por country_iso3.

    Args:
        ranking_df: DataFrame de ranking generado por TOPSISRanking.rank().

    Returns:
        Serie de scores TOPSIS indexada por país.
    """

    ranking = ranking_df.copy()

    if "country_iso3" in ranking.columns:
        ranking = ranking.set_index("country_iso3")

    return ranking["score"]


def _rank_series_from_ranking(ranking_df: pd.DataFrame) -> pd.Series:
    """Extrae una serie de rank indexada por country_iso3.

    Args:
        ranking_df: DataFrame de ranking generado por TOPSISRanking.rank().

    Returns:
        Serie de rankings indexada por país.
    """

    ranking = ranking_df.copy()

    if "country_iso3" in ranking.columns:
        ranking = ranking.set_index("country_iso3")

    return ranking["rank"]


def _get_effective_weights_for_line(
    weights_audit: pd.DataFrame,
    business_line: str,
    decision_matrix: pd.DataFrame,
) -> pd.Series:
    """Obtiene pesos finales efectivos para una línea de negocio.

    Usa la auditoría de pesos ya filtrada por línea y conserva únicamente
    variables presentes en decision_matrix. Luego renormaliza para asegurar
    que la suma sea 1.

    Args:
        weights_audit: DataFrame de auditoría de pesos efectivos.
        business_line: Código de línea de negocio.
        decision_matrix: Matriz normalizada país x variable.

    Returns:
        Serie de pesos finales indexada por variable.
    """

    weight_slice = weights_audit[
        (weights_audit["business_line"] == business_line)
        & (weights_audit["in_decision_matrix"])
    ].copy()

    if weight_slice.empty:
        raise ValueError(
            f"No hay pesos efectivos para la línea de negocio '{business_line}'."
        )

    weights = (
        weight_slice
        .set_index("variable")["final_topsis_weight"]
        .reindex(decision_matrix.columns)
        .dropna()
    )

    if weights.empty:
        raise ValueError(
            f"No hay pesos compatibles con decision_matrix para '{business_line}'."
        )

    total_weight = weights.sum()

    if total_weight <= 0:
        raise ValueError(
            f"La suma de pesos efectivos para '{business_line}' es cero."
        )

    return weights / total_weight


def compute_marginal_variable_effects(
    decision_matrix: pd.DataFrame,
    weights_audit: pd.DataFrame,
    business_line: str,
    variable_catalog: Dict[str, Dict[str, Any]],
    distance_metric: str = "euclidean",
    variables: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Calcula efectos marginales leave-one-variable-out por país y variable.

    Para cada variable se recalcula TOPSIS sin esa variable y con los pesos
    restantes renormalizados. El efecto marginal se define como:

        score_effect = score_full - score_without_variable

    Interpretación:
        score_effect > 0:
            la variable ayuda al país; al removerla, su score baja.

        score_effect < 0:
            la variable penaliza al país; al removerla, su score sube.

        abs(score_effect) alto:
            la variable es materialmente relevante para ese país/línea.

    También calcula:

        rank_effect = rank_without_variable - rank_full

    Interpretación:
        rank_effect > 0:
            al remover la variable el país empeora en ranking; la variable ayuda.

        rank_effect < 0:
            al remover la variable el país mejora; la variable penaliza.

    Args:
        decision_matrix: Matriz normalizada país x variable con dirección aplicada.
        weights_audit: Auditoría de pesos efectivos por línea.
        business_line: Código de línea de negocio.
        variable_catalog: Catálogo de variables.
        distance_metric: Métrica de distancia TOPSIS.
        variables: Lista opcional de variables a evaluar. Si None, evalúa todas.

    Returns:
        DataFrame largo con efectos marginales por país-variable.
    """

    weights = _get_effective_weights_for_line(
        weights_audit=weights_audit,
        business_line=business_line,
        decision_matrix=decision_matrix,
    )

    available_vars = list(weights.index)

    if variables is not None:
        variables_to_test = [
            var for var in variables
            if var in available_vars
        ]
    else:
        variables_to_test = available_vars

    if len(available_vars) < 2:
        raise ValueError(
            "Se requieren al menos dos variables para calcular efectos marginales."
        )

    ranker = TOPSISRanking(distance_metric=distance_metric)

    full_ranking = ranker.rank(
        decision_matrix[available_vars],
        weights.to_dict(),
        variable_catalog,
    )

    full_scores = _score_series_from_ranking(full_ranking)
    full_ranks = _rank_series_from_ranking(full_ranking)

    metadata_cols = [
        "business_line",
        "dimension",
        "variable",
        "final_topsis_weight",
        "has_override",
        "override_variable_weight_in_dim",
    ]

    metadata = (
        weights_audit[
            (weights_audit["business_line"] == business_line)
            & (weights_audit["variable"].isin(available_vars))
        ][[col for col in metadata_cols if col in weights_audit.columns]]
        .drop_duplicates(subset=["variable"])
    )

    rows: List[pd.DataFrame] = []

    for removed_variable in variables_to_test:
        remaining_vars = [
            var for var in available_vars
            if var != removed_variable
        ]

        if not remaining_vars:
            continue

        weights_without = weights.drop(index=removed_variable)
        weights_without = weights_without / weights_without.sum()

        ranking_without = ranker.rank(
            decision_matrix[remaining_vars],
            weights_without.to_dict(),
            variable_catalog,
        )

        scores_without = _score_series_from_ranking(ranking_without)
        ranks_without = _rank_series_from_ranking(ranking_without)

        effect = pd.DataFrame(
            {
                "country_iso3": full_scores.index,
                "business_line": business_line,
                "removed_variable": removed_variable,
                "score_full": full_scores,
                "score_without_variable": scores_without.reindex(full_scores.index),
                "rank_full": full_ranks,
                "rank_without_variable": ranks_without.reindex(full_ranks.index),
            }
        ).reset_index(drop=True)

        effect["score_effect"] = (
            effect["score_full"] - effect["score_without_variable"]
        )

        effect["rank_effect"] = (
            effect["rank_without_variable"] - effect["rank_full"]
        )

        rows.append(effect)

    if not rows:
        return pd.DataFrame(
            columns=[
                "business_line",
                "country_iso3",
                "removed_variable",
                "dimension",
                "final_topsis_weight",
                "score_full",
                "score_without_variable",
                "score_effect",
                "rank_full",
                "rank_without_variable",
                "rank_effect",
                "effect_type",
            ]
        )

    out = pd.concat(rows, ignore_index=True)

    out = out.merge(
        metadata.rename(columns={"variable": "removed_variable"}),
        on=["business_line", "removed_variable"],
        how="left",
    )

    out["abs_score_effect"] = out["score_effect"].abs()

    out["effect_type"] = out["score_effect"].apply(
        lambda x: "Driver" if x > 0 else ("Restriccion" if x < 0 else "Neutro")
    )

    return out[
        [
            "business_line",
            "country_iso3",
            "removed_variable",
            "dimension",
            "final_topsis_weight",
            "score_full",
            "score_without_variable",
            "score_effect",
            "abs_score_effect",
            "rank_full",
            "rank_without_variable",
            "rank_effect",
            "effect_type",
            "has_override",
            "override_variable_weight_in_dim",
        ]
    ].sort_values(
        ["country_iso3", "abs_score_effect"],
        ascending=[True, False],
    )


def compute_all_marginal_effects(
    decision_matrix: pd.DataFrame,
    weights_audit: pd.DataFrame,
    variable_catalog: Dict[str, Dict[str, Any]],
    distance_metric: str = "euclidean",
    business_lines: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Calcula efectos marginales para todas las líneas de negocio.

    Args:
        decision_matrix: Matriz normalizada país x variable.
        weights_audit: Auditoría de pesos efectivos.
        variable_catalog: Catálogo de variables.
        distance_metric: Métrica TOPSIS.
        business_lines: Lista opcional de líneas. Si None, usa todas.

    Returns:
        Diccionario {business_line: DataFrame de efectos marginales}.
    """

    if business_lines is None:
        business_lines = sorted(weights_audit["business_line"].unique())

    return {
        business_line: compute_marginal_variable_effects(
            decision_matrix=decision_matrix,
            weights_audit=weights_audit,
            business_line=business_line,
            variable_catalog=variable_catalog,
            distance_metric=distance_metric,
        )
        for business_line in business_lines
    }

def combine_contribution_and_marginal(
    contributions: pd.DataFrame,
    marginal_effects: pd.DataFrame,
) -> pd.DataFrame:
    """Combina contribución aditiva y efecto marginal por país-variable.

    Args:
        contributions: DataFrame de contribuciones aditivas.
        marginal_effects: DataFrame de efectos marginales.

    Returns:
        DataFrame integrado de explicabilidad.
    """

    contrib = contributions.rename(
        columns={"variable": "removed_variable"}
    ).copy()

    merged = contrib.merge(
        marginal_effects,
        on=["business_line", "country_iso3", "removed_variable"],
        how="left",
        suffixes=("_contribution", "_marginal"),
    )

    return merged


def classify_driver_robustness(row: pd.Series) -> str:
    """Clasifica la robustez de un driver combinando contribución y efecto marginal."""

    contribution = row.get("contribution", 0.0)
    score_effect = row.get("score_effect", 0.0)

    if contribution > 0.03 and score_effect > 0.01:
        return "Driver robusto"
    if contribution > 0.03 and abs(score_effect) <= 0.01:
        return "Driver descriptivo"
    if score_effect < -0.01:
        return "Restriccion critica"
    if abs(score_effect) <= 0.005:
        return "Efecto marginal bajo"
    return "Driver moderado"


def build_country_driver_table(
    explainability_df: pd.DataFrame,
    country_iso3: str,
    top_n: int = 5,
) -> pd.DataFrame:
    """Construye tabla ejecutiva de drivers y restricciones para un país."""

    country_df = explainability_df[
        explainability_df["country_iso3"] == country_iso3
    ].copy()

    top_positive = (
        country_df
        .sort_values("score_effect", ascending=False)
        .head(top_n)
        .assign(driver_side="Driver")
    )

    top_negative = (
        country_df
        .sort_values("score_effect", ascending=True)
        .head(top_n)
        .assign(driver_side="Restriccion")
    )

    return pd.concat(
        [top_positive, top_negative],
        ignore_index=True,
    )[
        [
            "business_line",
            "country_iso3",
            "removed_variable",
            "dimension_contribution",
            "normalized_value",
            "final_topsis_weight_contribution",
            "contribution",
            "shortfall",
            "score_effect",
            "rank_effect",
            "effect_type",
            "driver_side",
            "driver_class",
        ]
    ]