"""Dados de mentira só para os testes do site (nunca em data/ nem no site). Contrato §1.4: nada disto sai de tests/.

Janela real do params.yaml: 1981-01..2026-09 (549 meses). Três municípios de mentira (A, B, C).
"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import chirps_raw
from src.data.config import load_config

RAIZ = Path(__file__).resolve().parents[1]
PARAMS_REAL = RAIZ / "params.yaml"
MESES = pd.period_range("1981-01", "2026-09", freq="M")
NT = len(MESES)
A, B, C = "2200001", "2200002", "2200003"
AGORA = datetime(2026, 11, 19, 12, 30, 45, tzinfo=timezone.utc)
VERIFICADO = datetime(2026, 11, 19, 15, 0, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ dados sintéticos
def quadrado(lon, lat, lado=0.5):
    return [[lon, lat], [lon + lado, lat], [lon + lado, lat - lado], [lon, lat - lado], [lon, lat]]


def feature(cod, nome, geom):
    return {"type": "Feature", "properties": {"cod_ibge": cod, "nome": nome, "situacao": "res150"}, "geometry": geom}


def malha_tres():
    """A no norte (lat maior), C no meio, B no sul; C é um MultiPolygon com um furo no primeiro polígono."""
    furo = quadrado(-44.9, -6.6, 0.1)[::-1]
    return {"type": "FeatureCollection", "features": [
        feature(B, "Beta do Sul", {"type": "Polygon", "coordinates": [quadrado(-44.0, -8.0)]}),
        feature(A, "Alfa do Norte", {"type": "Polygon", "coordinates": [quadrado(-44.0, -4.0)]}),
        feature(C, "Tanque do Piauí", {"type": "MultiPolygon", "coordinates": [
            [quadrado(-45.0, -6.0, 0.8), furo], [quadrado(-43.0, -6.0)]]}),
    ]}


def raw_sintetico(cods=(A, B, C), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quadros = []
    for cod in cods:
        p = rng.gamma(2.0, 60.0, NT).astype("float32")
        p[rng.random(NT) < 0.1] = 0.0
        quadros.append(pd.DataFrame({
            "cod_ibge": cod, "mes": MESES.to_timestamp(), "precip_mm": p,
            "precip_max_dia_mm": (p * 0.4).astype("float32"),
            "dias_chuva": np.minimum(MESES.days_in_month, np.floor(p / 8)).astype("float32"),
            "dias_validos": MESES.days_in_month.astype("int16"),
            "dias_no_mes": MESES.days_in_month.astype("int16"),
            "pixels_validos_frac": np.float32(1.0)}))
    return pd.concat(quadros, ignore_index=True)


def escrever_raw(raiz: Path, raw: pd.DataFrame, cods) -> None:
    cfg = load_config(raiz / "params.yaml")
    d = chirps_raw.diretorio_raw(cfg)
    lotes = (("1981-01", "2000-12", "2026-11-19T00:40:00+00:00"), ("2001-01", "2026-09", "2026-11-18T23:10:00+00:00"))
    for ini, fim, quando in lotes:
        i, f = pd.Period(ini, "M"), pd.Period(fim, "M")
        m = (raw["mes"] >= i.to_timestamp()) & (raw["mes"] <= f.to_timestamp())
        chirps_raw.gravar_lote(raw[m].reset_index(drop=True), d, cods, i, f, meta={"extraido_em_utc": quando})
