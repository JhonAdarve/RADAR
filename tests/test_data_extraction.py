"""
Tests unitarios para src/data_extraction/world_bank.py.

Usa mocks para evitar llamadas reales a la API.

Autor: Jhon Adarve - Direccion de Estrategia, Grupo Cibest
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data_extraction.world_bank import fetch_indicator, validate_response
from src.utils import DataExtractionError


# ---------------------------------------------------------------------------
# Validacion de respuesta
# ---------------------------------------------------------------------------
def test_validate_response_ok() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = [
        {"page": 1, "pages": 1, "per_page": 1000, "total": 1},
        [{"countryiso3code": "COL", "country": {"value": "Colombia"},
          "date": "2023", "value": 16500.5}],
    ]
    records = validate_response(mock_resp)
    assert isinstance(records, list)
    assert len(records) == 1


def test_validate_response_http_error() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error"
    mock_resp.url = "http://x"
    with pytest.raises(DataExtractionError):
        validate_response(mock_resp)


def test_validate_response_invalid_json() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.side_effect = json.JSONDecodeError("err", "", 0)
    with pytest.raises(DataExtractionError):
        validate_response(mock_resp)


def test_validate_response_unexpected_structure() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"not_a_list": True}
    with pytest.raises(DataExtractionError):
        validate_response(mock_resp)


# ---------------------------------------------------------------------------
# Extraccion con mock de session
# ---------------------------------------------------------------------------
@patch("src.data_extraction.world_bank.requests.Session")
def test_fetch_indicator_returns_dataframe(mock_session_cls: MagicMock) -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"x"
    mock_resp.json.return_value = [
        {"page": 1, "pages": 1, "per_page": 1000, "total": 2},
        [
            {"countryiso3code": "COL", "country": {"value": "Colombia"},
             "date": "2023", "value": 16500},
            {"countryiso3code": "MEX", "country": {"value": "Mexico"},
             "date": "2023", "value": 21000},
        ],
    ]
    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    mock_session_cls.return_value = mock_session

    df = fetch_indicator(
        indicator_code="NY.GDP.PCAP.CD",
        countries=["COL", "MEX"],
        start_year=2020, end_year=2024,
        cache_dir=None,
    )
    assert isinstance(df, pd.DataFrame)
    assert "country_iso3" in df.columns
    assert "value" in df.columns
    assert len(df) == 2
