"""SPI-3 e SPI-6 por município (Item 5). Sem I/O e sem rede.

Método (contrato, seção 7), por escala k em `spi.janelas_meses`:
  1. S_k(t) = soma de P(t-k+1) ... P(t). Qualquer mês ausente na janela torna S_k(t) NaN.
  2. Para cada (cod_ibge, mês-calendário do mês final) ajusta-se uma Gama (MLE, loc = 0) aos S_k
     POSITIVOS com mês final <= `split.treino_fim`; q = fração de zeros desse mesmo conjunto.
  3. H(x) = q + (1 - q) G(x) e SPI = Phi^-1(H(x)). Para x = 0, G(0) = 0 e portanto H(0) = q
     (convenção D3 = "q", padrão OMM). Os parâmetros do treino valem para todo o período.
  4. Células sem condição de ajuste (poucas amostras, poucos positivos, série constante, MLE que
     não converge) ficam `valido = False`: o SPI é NaN e `spiK_invalid = True` (nunca ±inf).
     H é limitado a [EPS_H, 1 - EPS_H], então |SPI| <= ~4,75.
  5. `spiK_baixa_conf = True` (com SPI válido) quando há pouca chuva esperada na janela, por um destes motivos:
       a) q > `spi.q_baixa_confianca` (muitos zeros no mês-calendário); ou
       b) média do treino da janela por mês da janela < `spi.media_min_mm_por_mes`, com
          média = (1 - q) * shape * scale / k (no MLE da Gama com loc = 0, shape * scale é a média dos positivos).
     Nesses casos poucos milímetros viram SPI alto ou baixo (limitação conhecida do índice em estação seca);
     o SPI não muda, só a marcação. Depende só dos parâmetros do treino, então não vaza.

O SPI das linhas de treino é in-sample (limitação registrada no contrato). A função não lê nem
grava arquivos: persistir `parametros` em `artifacts/spi_gamma_params` é papel do build.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from src.data import splits
from src.data.config import Config, exigir_resolvido

CONVENCOES_ZERO = ("q",)      # vocabulário de `spi.convencao_zero`; "q": H(0) = q
EPS_H = 1e-6                  # H fica em [EPS_H, 1 - EPS_H]: evita ±inf
COLUNAS_PARAMETROS = ["cod_ibge", "mes_calendario", "escala_meses", "q", "shape", "scale",
                      "n_amostras", "n_positivos", "valido"]


class SpiError(ValueError):
    """Entrada ou configuração incompatível com o cálculo do SPI."""


@dataclass(frozen=True)
class ResultadoSpi:
    spi: pd.DataFrame          # longo: cod_ibge, mes, spi{k}, spi{k}_invalid, spi{k}_baixa_conf
    parametros: pd.DataFrame   # COLUNAS_PARAMETROS (artefato `spi_gamma_params`)


# ---------------------------------------------------------------- entrada
def _validar_cfg(cfg: Config) -> None:
    exigir_resolvido(cfg, "spi")
    s = cfg.spi
    if s.convencao_zero not in CONVENCOES_ZERO:
        raise SpiError(f"spi.convencao_zero={s.convencao_zero!r} não suportada; use um de {CONVENCOES_ZERO}")
    if not s.zero_inflado:
        raise SpiError("spi.zero_inflado=false não é suportado (o semiárido tem meses com chuva zero)")
    if tuple(s.agrupar_por) != ("cod_ibge", "mes_calendario"):
        raise SpiError(f"spi.agrupar_por={tuple(s.agrupar_por)} não suportado; esperado (cod_ibge, mes_calendario)")


def para_matriz(precip: pd.DataFrame) -> pd.DataFrame:
    """Longo (cod_ibge, mes, precip_mm) -> matriz: índice PeriodIndex mensal contíguo, colunas = cod_ibge.

    Mês que falta no intervalo vira NaN (nunca 0). Duplicata, valor negativo ou infinito são erro."""
    if not isinstance(precip, pd.DataFrame):
        raise SpiError("esperado um DataFrame")
    faltam = {"cod_ibge", "mes", "precip_mm"} - set(precip.columns)
    if faltam:
        raise SpiError(f"colunas ausentes: {sorted(faltam)}")
    if precip.empty:
        raise SpiError("DataFrame vazio")
    try:
        v = pd.to_numeric(precip["precip_mm"], errors="raise").astype("float64").to_numpy()
        mes = pd.to_datetime(precip["mes"]).dt.to_period("M")
    except (ValueError, TypeError) as e:
        raise SpiError(f"conversão de tipos falhou: {e}") from e
    if mes.isna().any() or precip["cod_ibge"].isna().any():
        raise SpiError("cod_ibge ou mes ausente")
    if np.isinf(v).any() or (v < 0).any():           # NaN passa (mês sem dado); negativo e inf não
        raise SpiError("precip_mm negativo ou infinito")
    df = pd.DataFrame({"cod_ibge": precip["cod_ibge"].astype(str).to_numpy(),
                       "mes": mes.reset_index(drop=True), "v": v})
    if df.duplicated(["cod_ibge", "mes"]).any():
        raise SpiError("duplicatas (cod_ibge, mes)")
    w = df.pivot(index="mes", columns="cod_ibge", values="v")
    w = w.reindex(pd.period_range(w.index.min(), w.index.max(), freq="M"))
    return w.reindex(sorted(w.columns), axis=1).astype("float64")


# ---------------------------------------------------------------- acumulado e ajuste
def acumular(matriz: pd.DataFrame, k: int) -> pd.DataFrame:
    """S_k(t): soma móvel de k meses terminando em t. Primeiros k-1 meses e janelas com NaN viram NaN."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise SpiError(f"escala inválida: {k!r}")
    v = matriz.to_numpy(dtype="float64")
    out = np.full_like(v, np.nan)
    if v.shape[0] >= k:
        out[k - 1:] = np.lib.stride_tricks.sliding_window_view(v, k, axis=0).sum(axis=-1)
    return pd.DataFrame(out, index=matriz.index, columns=matriz.columns)


def _ajustar_celula(x: np.ndarray, min_positivos: int, min_amostras: int):
    """x: S_k sem NaN de uma célula. Devolve (q, shape, scale, n, n_positivos, valido)."""
    n = int(x.size)
    pos = x[x > 0]
    npos = int(pos.size)
    q = (n - npos) / n if n else float("nan")
    nan = float("nan")
    if n < min_amostras or npos < min_positivos or float(np.ptp(pos)) == 0.0:
        return q, nan, nan, n, npos, False
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            a, _, esc = stats.gamma.fit(pos, floc=0)
    except (ValueError, RuntimeError, FloatingPointError, stats.FitError):
        return q, nan, nan, n, npos, False
    if not (np.isfinite(a) and np.isfinite(esc) and a > 0 and esc > 0):
        return q, nan, nan, n, npos, False
    return q, float(a), float(esc), n, npos, True


def ajustar_gama(S: pd.DataFrame, k: int, cfg: Config, treino_fim: pd.Period) -> pd.DataFrame:
    """Parâmetros por (cod_ibge, mês-calendário), usando SÓ meses finais <= `treino_fim`."""
    treino = np.asarray(S.index <= treino_fim)
    mes_cal = np.asarray(S.index.month)
    celulas = [np.flatnonzero(treino & (mes_cal == m)) for m in range(1, 13)]
    vals = S.to_numpy(dtype="float64")
    min_pos, min_n = cfg.spi.min_positivos_ajuste, cfg.spi.min_anos_treino
    linhas = []
    for j, cod in enumerate(S.columns):
        for m, ix in enumerate(celulas, start=1):
            x = vals[ix, j]
            linhas.append((str(cod), m, k, *_ajustar_celula(x[~np.isnan(x)], min_pos, min_n)))
    df = pd.DataFrame(linhas, columns=COLUNAS_PARAMETROS)
    return df.astype({"mes_calendario": "int8", "escala_meses": "int8", "q": "float64", "shape": "float64",
                      "scale": "float64", "n_amostras": "int32", "n_positivos": "int32", "valido": "bool"})


def _escala_unica(params: pd.DataFrame) -> int:
    ks = pd.unique(params["escala_meses"])
    if len(ks) != 1:
        raise SpiError(f"parametros de uma única escala esperados, recebi {sorted(int(k) for k in ks)}")
    return int(ks[0])


def _aplicar(S: pd.DataFrame, params: pd.DataFrame, cfg: Config):
    """SPI, invalid e baixa_conf (matrizes T x J) a partir de S_k e dos parâmetros da escala."""
    k = _escala_unica(params)
    def por_mes(col):
        p = params.pivot(index="mes_calendario", columns="cod_ibge", values=col)
        return p.reindex(index=range(1, 13), columns=S.columns).to_numpy()
    cal = np.asarray(S.index.month) - 1
    q, a, esc = (por_mes(c).astype("float64")[cal] for c in ("q", "shape", "scale"))
    valido = pd.DataFrame(por_mes("valido")).fillna(False).to_numpy(dtype=bool)[cal]
    x = S.to_numpy(dtype="float64")
    ok = valido & np.isfinite(x) & np.isfinite(q) & np.isfinite(a) & np.isfinite(esc)
    xs, a, esc, q = (np.where(ok, arr, fill) for arr, fill in ((x, 1.0), (a, 1.0), (esc, 1.0), (q, 0.0)))
    h = np.clip(q + (1.0 - q) * stats.gamma.cdf(xs, a, scale=esc), EPS_H, 1.0 - EPS_H)
    spi = np.where(ok, stats.norm.ppf(h), np.nan)
    media_por_mes = (1.0 - q) * a * esc / k              # mm por mês da janela; valores de preenchimento só onde ~ok
    pouca_chuva = media_por_mes < cfg.spi.media_min_mm_por_mes
    return spi, ~ok, ok & ((q > cfg.spi.q_baixa_confianca) | pouca_chuva)


# ---------------------------------------------------------------- orquestração
def calcular_spi(precip: pd.DataFrame, cfg: Config, *, parametros: pd.DataFrame | None = None) -> ResultadoSpi:
    """SPI de todas as escalas de `spi.janelas_meses`.

    Sem `parametros`, a Gama é ajustada só no treino. Com `parametros` (ex.: carregados de
    `artifacts/spi_gamma_params`), eles são apenas aplicados, sem nenhum ajuste."""
    _validar_cfg(cfg)
    corte = splits.cortes(cfg).treino_fim                  # também valida o embargo para o SPI-6
    M = para_matriz(precip)
    meses = M.index.to_timestamp().astype("datetime64[ns]")
    n_mes, cods = len(M.index), np.asarray(M.columns, dtype=object)
    saida = {"cod_ibge": np.repeat(cods, n_mes), "mes": np.tile(meses.to_numpy(), len(cods))}
    todos = []
    for k in cfg.spi.janelas_meses:
        S = acumular(M, k)
        if parametros is None:
            pk = ajustar_gama(S, k, cfg, corte)
        else:
            if not set(COLUNAS_PARAMETROS) <= set(parametros.columns):
                raise SpiError(f"parametros sem as colunas {COLUNAS_PARAMETROS}")
            pk = parametros[parametros["escala_meses"] == k]
            if pk.empty:
                raise SpiError(f"parametros sem a escala SPI-{k}")
        spi, inval, baixa = _aplicar(S, pk, cfg)
        saida[f"spi{k}"] = spi.T.ravel().astype("float32")
        saida[f"spi{k}_invalid"] = inval.T.ravel()
        saida[f"spi{k}_baixa_conf"] = baixa.T.ravel()
        todos.append(pk.reset_index(drop=True))
    return ResultadoSpi(spi=pd.DataFrame(saida), parametros=pd.concat(todos, ignore_index=True))
