"""Qualidade da série mensal de chuva (Item 3). Sem I/O e sem rede.

Entrada: o RAW do CHIRPS (`chirps_raw.ler_raw`), um DataFrame município x mês. Cada etapa é uma
função pura:

  1. `cobertura_mensal`: relatório município x mês, gerado ANTES de qualquer imputação (contrato 6.3).
     Um mês é VÁLIDO quando `precip_mm` existe, `dias_validos / dias_no_mes` >= `qualidade.min_cobertura_dias_mes`
     e `pixels_validos_frac` >= `qualidade.min_cobertura_pixels` (limiares inclusivos).
  2. `serie_valida`: as 3 colunas de chuva viram NaN em mês inválido (nunca 0). É a entrada do `spi.py`
     (D9: o SPI nunca usa valor imputado; mês sem dado fica "sem dado").
  3. `climatologia`: mediana por (cod_ibge, mês-calendário) só dos meses VÁLIDOS com mes <= `split.treino_fim`.
  4. `imputar`: preenche com essa mediana, só para alimentar features. Regra causal (CA-06): o k-ésimo mês
     SEGUIDO sem dado só é preenchido se k <= `qualidade.max_meses_imputados_consecutivos`; decidir um mês
     usa apenas o passado, nunca "quanto falta" da lacuna. `precip_mm_missing = True` em todo mês que não
     era válido (preenchido ou não).
  5. `qualidade_municipio`: fração de meses válidos no TREINO (D10). Abaixo de
     `qualidade.min_fracao_meses_validos_municipio` => `excluded` com motivo (o município continua nos
     216 e aparece como "sem dado"; nunca é removido em silêncio).

Limitação registrada: a mediana do treino usa todos os anos do treino, inclusive os posteriores ao mês
imputado (estatística de treino, como um scaler; não cruza o corte para validação ou teste).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import splits
from src.data.config import Config, exigir_resolvido

COLUNAS_CHUVA = ["precip_mm", "precip_max_dia_mm", "dias_chuva"]
COLUNAS_ENTRADA = ["cod_ibge", "mes", *COLUNAS_CHUVA, "dias_validos", "dias_no_mes", "pixels_validos_frac"]
COLUNAS_COBERTURA = ["cod_ibge", "mes", "cobertura_dias", "cobertura_pixels", "valido", "motivo"]
COLUNAS_SERIE = ["cod_ibge", "mes", *COLUNAS_CHUVA]
COLUNAS_IMPUTADA = [*COLUNAS_SERIE, "precip_mm_missing"]
COLUNAS_CLIMATOLOGIA = ["cod_ibge", "mes_calendario", "mediana_mm", "mediana_max_dia_mm",
                        "mediana_dias_chuva", "n_amostras"]
COLUNAS_MUNICIPIO = ["cod_ibge", "frac_meses_validos", "excluded", "motivo"]

_COD = re.compile(r"\d{7}")


class QualidadeError(ValueError):
    """Entrada ou configuração incompatível com a etapa de qualidade."""


@dataclass(frozen=True)
class ResultadoQualidade:
    cobertura: pd.DataFrame         # COLUNAS_COBERTURA: relatório antes de imputar
    serie_valida: pd.DataFrame      # COLUNAS_SERIE: NaN em mês inválido (entrada do spi.py)
    serie_imputada: pd.DataFrame    # COLUNAS_IMPUTADA: entrada das features
    climatologia: pd.DataFrame      # COLUNAS_CLIMATOLOGIA (artefato `climatologia_precip`)
    municipios: pd.DataFrame        # COLUNAS_MUNICIPIO (tabela `qualidade_municipio`)


# ---------------------------------------------------------------- entrada
def _meses_janela(cfg: Config) -> pd.PeriodIndex:
    return pd.period_range(cfg.janela.inicio, cfg.janela.fim, freq="M")


def _grade(df: pd.DataFrame, cfg: Config) -> None:
    """Exige: mes = dia 1 sem hora, dentro da janela; sem duplicata; mesma grade completa em todo município."""
    if df["mes"].isna().any() or df["cod_ibge"].isna().any():
        raise QualidadeError("cod_ibge ou mes ausente")
    mes = df["mes"]
    if (mes != mes.dt.to_period("M").dt.to_timestamp()).any():
        raise QualidadeError("`mes` deve ser o dia 1 às 00:00")
    maus = ~df["cod_ibge"].astype(str).map(lambda c: bool(_COD.fullmatch(c)))
    if maus.any():
        raise QualidadeError(f"cod_ibge fora do formato de 7 dígitos, ex.: {df.loc[maus, 'cod_ibge'].head(3).tolist()}")
    janela = _meses_janela(cfg)
    esperado = set((janela.year * 12 + janela.month).tolist())
    obtido = (mes.dt.year * 12 + mes.dt.month).astype("int64")
    fora = ~obtido.isin(esperado)
    if fora.any():
        raise QualidadeError(f"{int(fora.sum())} linha(s) fora da janela [{janela[0]}, {janela[-1]}]")
    if df.duplicated(["cod_ibge", "mes"]).any():
        raise QualidadeError("duplicatas (cod_ibge, mes)")
    n = df.groupby("cod_ibge").size()
    if (n != len(janela)).any():
        ruins = n[n != len(janela)].index.tolist()[:3]
        raise QualidadeError(f"grade incompleta: cada município deve ter {len(janela)} meses; ex.: {ruins}")


def _preparar(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raise QualidadeError("esperado um DataFrame")
    faltam = set(COLUNAS_ENTRADA) - set(raw.columns)
    if faltam:
        raise QualidadeError(f"colunas ausentes: {sorted(faltam)}")
    if raw.empty:
        raise QualidadeError("DataFrame vazio")
    df = raw[COLUNAS_ENTRADA].copy().reset_index(drop=True)
    try:
        df["cod_ibge"] = df["cod_ibge"].astype(str)
        df["mes"] = pd.to_datetime(df["mes"]).astype("datetime64[ns]")
        for c in (*COLUNAS_CHUVA, "pixels_validos_frac"):
            df[c] = pd.to_numeric(df[c], errors="raise").astype("float32")
        for c in ("dias_validos", "dias_no_mes"):
            df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")
    except (ValueError, TypeError) as e:
        raise QualidadeError(f"conversão de tipos falhou: {e}") from e
    _grade(df, cfg)
    if (df["dias_no_mes"] <= 0).any():
        raise QualidadeError("dias_no_mes deve ser > 0")
    if ((df["dias_validos"] < 0) | (df["dias_validos"] > df["dias_no_mes"])).any():
        raise QualidadeError("dias_validos fora de [0, dias_no_mes]")
    frac = df["pixels_validos_frac"]
    if frac.isna().any() or (frac < 0).any() or (frac > 1).any():
        raise QualidadeError("pixels_validos_frac deve estar em [0, 1] e sem NaN")
    for c in COLUNAS_CHUVA:
        v = df[c].to_numpy()
        if np.isinf(v).any() or (v[~np.isnan(v)] < 0).any():
            raise QualidadeError(f"{c} negativo ou infinito")
    return df.sort_values(["cod_ibge", "mes"], kind="stable").reset_index(drop=True)


def _checar_serie(serie: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if not isinstance(serie, pd.DataFrame):
        raise QualidadeError("esperado um DataFrame")
    faltam = set(COLUNAS_SERIE) - set(serie.columns)
    if faltam:
        raise QualidadeError(f"colunas ausentes: {sorted(faltam)}")
    if serie.empty:
        raise QualidadeError("DataFrame vazio")
    df = serie[COLUNAS_SERIE].copy().reset_index(drop=True)
    try:
        df["cod_ibge"] = df["cod_ibge"].astype(str)
        df["mes"] = pd.to_datetime(df["mes"]).astype("datetime64[ns]")
        for c in COLUNAS_CHUVA:
            df[c] = pd.to_numeric(df[c], errors="raise").astype("float32")
    except (ValueError, TypeError) as e:
        raise QualidadeError(f"conversão de tipos falhou: {e}") from e
    _grade(df, cfg)
    return df.sort_values(["cod_ibge", "mes"], kind="stable").reset_index(drop=True)


def _treino_fim(cfg: Config) -> pd.Timestamp:
    """Primeiro dia do mês `split.treino_fim`. Também valida o embargo para todas as escalas do SPI."""
    return splits.cortes(cfg).treino_fim.to_timestamp()


# ---------------------------------------------------------------- 1. cobertura
def _cobertura(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    q = cfg.qualidade
    cob_dias = df["dias_validos"].to_numpy("float64") / df["dias_no_mes"].to_numpy("float64")
    cob_pix = df["pixels_validos_frac"].to_numpy("float32")
    ok_dado = df["precip_mm"].notna().to_numpy()
    ok_dias = cob_dias >= q.min_cobertura_dias_mes
    # float32 dos dois lados: converter o pixel para float64 reprovaria 7/10 no limiar 0.7 (arredondamento)
    ok_pix = cob_pix >= np.float32(q.min_cobertura_pixels)
    nomes = np.array(["sem_dado", "cobertura_dias", "cobertura_pixels"])
    falhas = np.column_stack([~ok_dado, ~ok_dias, ~ok_pix])
    return pd.DataFrame({
        "cod_ibge": df["cod_ibge"].to_numpy(),
        "mes": df["mes"].to_numpy(),
        "cobertura_dias": cob_dias.astype("float32"),
        "cobertura_pixels": cob_pix,
        "valido": ok_dado & ok_dias & ok_pix,
        "motivo": ["+".join(nomes[f]) for f in falhas],
    })[COLUNAS_COBERTURA]


def cobertura_mensal(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Relatório município x mês (COLUNAS_COBERTURA). `motivo` = critérios reprovados, separados por '+'."""
    exigir_resolvido(cfg, "qualidade")
    return _cobertura(_preparar(raw, cfg), cfg)


# ---------------------------------------------------------------- 2. série válida
def _serie_valida(df: pd.DataFrame, cob: pd.DataFrame) -> pd.DataFrame:
    valido = cob["valido"].to_numpy()
    out = df[COLUNAS_SERIE].copy()
    for c in COLUNAS_CHUVA:
        out[c] = out[c].where(valido).astype("float32")
    return out.reset_index(drop=True)


def serie_valida(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """COLUNAS_SERIE com NaN (nunca 0) em mês inválido. Sem imputação: é a entrada do SPI (D9)."""
    exigir_resolvido(cfg, "qualidade")
    df = _preparar(raw, cfg)
    return _serie_valida(df, _cobertura(df, cfg))


# ---------------------------------------------------------------- 3. climatologia
def climatologia(serie: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Mediana por (cod_ibge, mês-calendário) dos meses válidos <= `split.treino_fim` (COLUNAS_CLIMATOLOGIA).

    Inclui zeros. Célula sem mês válido no treino fica NaN com `n_amostras = 0`. Todas as 12 células de
    cada município aparecem."""
    exigir_resolvido(cfg, "qualidade")
    df = _checar_serie(serie, cfg)
    corte = _treino_fim(cfg)
    tr = df[df["mes"] <= corte].copy()
    tr["mes_calendario"] = tr["mes"].dt.month.astype("int8")
    g = tr.groupby(["cod_ibge", "mes_calendario"], sort=True)
    med = g[COLUNAS_CHUVA].median()
    n = g["precip_mm"].count()
    cods = sorted(df["cod_ibge"].unique())
    grade = pd.MultiIndex.from_product([cods, np.arange(1, 13, dtype="int8")], names=["cod_ibge", "mes_calendario"])
    med = med.reindex(grade)
    n = n.reindex(grade).fillna(0)
    out = pd.DataFrame({
        "cod_ibge": grade.get_level_values(0).astype(str),
        "mes_calendario": np.asarray(grade.get_level_values(1), dtype="int8"),
        "mediana_mm": med["precip_mm"].to_numpy("float64").astype("float32"),
        "mediana_max_dia_mm": med["precip_max_dia_mm"].to_numpy("float64").astype("float32"),
        "mediana_dias_chuva": med["dias_chuva"].to_numpy("float64").astype("float32"),
        "n_amostras": n.to_numpy().astype("int16"),
    })
    return out[COLUNAS_CLIMATOLOGIA].reset_index(drop=True)


# ---------------------------------------------------------------- 4. imputação causal
def _posicao_na_lacuna(falta: np.ndarray) -> np.ndarray:
    """Para cada mês sem dado, quantos meses SEGUIDOS sem dado já se passaram até ele, inclusive (1, 2, 3...).
    0 onde há dado. Usa só o passado."""
    c = np.cumsum(falta, axis=0)
    zera = np.maximum.accumulate(np.where(~falta, c, 0), axis=0)
    return (c - zera).astype("int64")


def _matriz(df: pd.DataFrame, col: str) -> np.ndarray:
    return df.pivot(index="mes", columns="cod_ibge", values=col).to_numpy("float64")


def imputar(serie: pd.DataFrame, clim: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Imputação causal por mediana do treino (COLUNAS_IMPUTADA).

    `serie` deve ser a série válida (NaN onde o mês não vale). O k-ésimo mês seguido sem `precip_mm` é
    preenchido (nas 3 colunas de chuva) se k <= `qualidade.max_meses_imputados_consecutivos` e a mediana
    existir; senão fica NaN. `precip_mm_missing` marca todo mês sem `precip_mm` na entrada."""
    exigir_resolvido(cfg, "qualidade")
    df = _checar_serie(serie, cfg)
    faltam = set(COLUNAS_CLIMATOLOGIA) - set(clim.columns)
    if faltam:
        raise QualidadeError(f"climatologia sem colunas: {sorted(faltam)}")
    cods = sorted(df["cod_ibge"].unique())
    meses = np.sort(pd.DatetimeIndex(df["mes"].unique()).to_numpy())
    mes_cal = pd.DatetimeIndex(meses).month.to_numpy() - 1
    P = {c: _matriz(df, c) for c in COLUNAS_CHUVA}
    falta = np.isnan(P["precip_mm"])
    ok_pos = _posicao_na_lacuna(falta) <= cfg.qualidade.max_meses_imputados_consecutivos
    mapa = {"precip_mm": "mediana_mm", "precip_max_dia_mm": "mediana_max_dia_mm",
            "dias_chuva": "mediana_dias_chuva"}
    cl = clim.assign(cod_ibge=clim["cod_ibge"].astype(str))
    saida = {}
    for c in COLUNAS_CHUVA:
        m = cl.pivot(index="mes_calendario", columns="cod_ibge", values=mapa[c])
        m = m.reindex(index=range(1, 13), columns=cods).to_numpy("float64")
        preenche = m[mes_cal, :]                       # [T, M]: mediana do mês-calendário de cada linha
        pode = falta & ok_pos & np.isfinite(preenche)
        saida[c] = np.where(pode, preenche, P[c])
    T, M = falta.shape
    out = pd.DataFrame({"cod_ibge": np.repeat(np.array(cods, dtype=object), T).astype(str),
                        "mes": np.tile(meses, M)})
    for c in COLUNAS_CHUVA:
        out[c] = saida[c].T.ravel().astype("float32")
    out["precip_mm_missing"] = falta.T.ravel()
    return out[COLUNAS_IMPUTADA].reset_index(drop=True)


# ---------------------------------------------------------------- 5. município
def qualidade_municipio(cobertura: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Fração de meses válidos no TREINO por município e exclusão (COLUNAS_MUNICIPIO).

    Denominador = meses de `janela.inicio` a `split.treino_fim` (linha ausente conta como inválida).
    `excluded` quando fração < `min_fracao_meses_validos_municipio` (estritamente abaixo)."""
    exigir_resolvido(cfg, "qualidade")
    faltam = {"cod_ibge", "mes", "valido"} - set(cobertura.columns)
    if faltam:
        raise QualidadeError(f"cobertura sem colunas: {sorted(faltam)}")
    if cobertura.empty:
        raise QualidadeError("DataFrame vazio")
    corte = _treino_fim(cfg)
    n_treino = len(pd.period_range(cfg.janela.inicio, corte.to_period("M"), freq="M"))
    cod = cobertura["cod_ibge"].astype(str)
    mes = pd.to_datetime(cobertura["mes"])
    em_treino = (mes <= corte).to_numpy()
    validos = pd.Series(cobertura["valido"].to_numpy(dtype=bool) & em_treino).groupby(cod.to_numpy()).sum()
    frac = validos.to_numpy("float64") / n_treino
    limiar = cfg.qualidade.min_fracao_meses_validos_municipio
    excluded = frac < limiar
    motivo = [f"cobertura no treino {f:.3f} < {limiar:g}" if e else "" for f, e in zip(frac, excluded)]
    return pd.DataFrame({"cod_ibge": validos.index.astype(str), "frac_meses_validos": frac.astype("float32"),
                         "excluded": excluded, "motivo": motivo})[COLUNAS_MUNICIPIO].reset_index(drop=True)


# ---------------------------------------------------------------- tudo
def avaliar_qualidade(raw: pd.DataFrame, cfg: Config) -> ResultadoQualidade:
    exigir_resolvido(cfg, "qualidade")
    df = _preparar(raw, cfg)
    cob = _cobertura(df, cfg)
    valida = _serie_valida(df, cob)
    clim = climatologia(valida, cfg)
    return ResultadoQualidade(cobertura=cob, serie_valida=valida, serie_imputada=imputar(valida, clim, cfg),
                              climatologia=clim, municipios=qualidade_municipio(cob, cfg))
