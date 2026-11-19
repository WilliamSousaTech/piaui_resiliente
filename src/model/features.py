"""Features, alvos e padronização do modelo (Item 9). Sem I/O, sem rede e sem torch.

Unidade de amostra (contrato, seção 3): a linha é (`cod_ibge`, `mes_origem` = t). Toda feature usa só
dados <= t. O alvo de cada cabeça é o SPI da escala K (cada K em `spi.janelas_meses`) em t + h, com
h = `alvo.horizonte_h`, e o split é definido por `mes_alvo` (nunca por `mes_origem`).

Colunas (L = `features.max_lag_meses`; "lag k" = mês t - k):
  chuva_lag{0..L-1}           log1p(precip_mm) da série imputada de `quality.py` (imputação causal)
  chuva_falta_lag{0..L-1}     1 quando aquele mês não tinha dado válido (preenchido ou não)
  spi{K}_lag{0..2}            SPI-K de t, t-1 e t-2 (NaN quando inválido)
  spi{K}_ausente_lag{0..2}    1 quando o SPI-K daquele mês é inválido
  spi{K}_baixa_lag0           1 quando o SPI-K de t tem baixa confiança
  mes_alvo_sin, mes_alvo_cos  calendário do mês-alvo (conhecido de antemão)
  clim_mm_alvo                log1p da mediana de chuva do mês-alvo: estatística só do treino (`quality.climatologia`)

Regras anti-vazamento deste módulo:
  1. Só se olha para trás: a coluna de lag k lê o mês t-k; o alvo é o único dado de t+h.
  2. O alvo ausente (SPI inválido) vira NaN com máscara falsa. Nunca é preenchido com 0.
  3. Município `excluded` sai das linhas (ele continua "sem dado" no site).
  4. Origens com mes_alvo depois de `split.teste_fim` ficam no bloco `pos_teste` (só observado). Entre elas
     está a origem mais recente, cujo alvo ainda não existe: serve à inferência e tem alvo NaN.
  5. Com `split.teste_liberado: false` os alvos do bloco `test` são NaN já na construção, e `selecionar`
     recusa o bloco. Não há como treinar, escolher hiperparâmetro nem pontuar no teste por engano.
  6. O `Padronizador` só se ajusta em linhas do bloco `train` e recusa qualquer outra coisa.

Limitação registrada: o SPI das linhas de treino é in-sample (Gama ajustada no próprio treino).
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import splits
from src.data.config import Config

_NOME_ALVO = re.compile(r"alvo_spi(\d+)_h(\d+)")
LAGS_SPI = 3                       # SPI de t, t-1 e t-2
POS_TESTE = "pos_teste"
BLOCOS = (*splits.BLOCOS, POS_TESTE)
DESVIO_MIN = 1e-6                  # desvio abaixo disso vira 1 (coluna constante no treino)


class FeaturesError(ValueError):
    """Entrada ou configuração incompatível com a construção das features."""


@dataclass(frozen=True, eq=False)
class Conjunto:
    X: np.ndarray                  # (n, d) float32; NaN onde não há dado (padronizar antes de treinar)
    Y: np.ndarray                  # (n, c) float32; NaN = sem rótulo (inválido, congelado ou ainda inexistente)
    M: np.ndarray                  # (n, c) bool; True = rótulo presente (entra na perda e nas métricas)
    colunas: tuple[str, ...]
    continuas: np.ndarray          # (d,) bool; só estas colunas são padronizadas
    alvos: tuple[str, ...]         # nome de cada cabeça, ex.: alvo_spi3_h1
    meta: pd.DataFrame             # cod_ibge, mes_origem, mes_alvo, split, alvo_spiK_baixa_conf

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def _alvo(self, i: int) -> tuple[int, int]:
        m = _NOME_ALVO.fullmatch(self.alvos[i])
        if m is None:
            raise FeaturesError(f"nome de alvo fora do padrão alvo_spiK_hH: {self.alvos[i]!r}")
        return int(m.group(1)), int(m.group(2))

    @property
    def escalas(self) -> tuple[int, ...]:
        """Escala K do SPI de cada cabeça, na ordem de `alvos`."""
        return tuple(self._alvo(i)[0] for i in range(len(self.alvos)))

    @property
    def horizonte(self) -> int:
        """h (meses entre a origem e o alvo), igual para todas as cabeças."""
        return self._alvo(0)[1]

    def fatiar(self, linhas: np.ndarray) -> "Conjunto":
        """Subconjunto por máscara booleana (n,). Mantém a ordem."""
        linhas = np.asarray(linhas)
        if linhas.dtype != bool or linhas.shape != (len(self),):
            raise FeaturesError(f"máscara de linhas deve ser bool com shape ({len(self)},)")
        return Conjunto(self.X[linhas], self.Y[linhas], self.M[linhas], self.colunas, self.continuas,
                        self.alvos, self.meta.loc[linhas].reset_index(drop=True))


# ---------------------------------------------------------------- entrada
def _matriz(df: pd.DataFrame, col: str, meses: pd.DatetimeIndex, cods: Sequence[str], onde: str) -> pd.DataFrame:
    """Matriz (meses x municípios) de `col`. Exige a grade completa e sem repetição."""
    faltam = [c for c in ("cod_ibge", "mes", col) if c not in df.columns]
    if faltam:
        raise FeaturesError(f"{onde}: coluna(s) ausente(s): {faltam}")
    sub = df.loc[df["cod_ibge"].isin(cods), ["cod_ibge", "mes", col]].copy()
    sub["mes"] = pd.to_datetime(sub["mes"])
    if sub.duplicated(["cod_ibge", "mes"]).any():
        raise FeaturesError(f"{onde}: pares (cod_ibge, mes) repetidos")
    if len(sub) != len(meses) * len(cods) or not sub["mes"].isin(meses).all():
        raise FeaturesError(f"{onde}: a grade município x mês não cobre {len(cods)} município(s) x "
                            f"{len(meses)} mês(es) da janela")
    return sub.pivot(index="mes", columns="cod_ibge", values=col).reindex(index=meses, columns=list(cods))


def _municipios(municipios: pd.DataFrame) -> list[str]:
    """Municípios que entram no modelo: todos os não `excluded`, ordenados por cod_ibge."""
    if not {"cod_ibge", "excluded"} <= set(municipios.columns):
        raise FeaturesError("qualidade_municipio sem as colunas cod_ibge e excluded")
    if municipios["cod_ibge"].duplicated().any():
        raise FeaturesError("qualidade_municipio com cod_ibge repetido")
    cods = sorted(municipios.loc[~municipios["excluded"].astype(bool), "cod_ibge"].astype(str))
    if not cods:
        raise FeaturesError("nenhum município utilizável: todos estão excluded")
    return cods


def _clim_log(clim: pd.DataFrame, cods: Sequence[str]) -> np.ndarray:
    """(12, J): log1p da mediana de chuva por mês-calendário (linha 0 = janeiro)."""
    faltam = [c for c in ("cod_ibge", "mes_calendario", "mediana_mm") if c not in clim.columns]
    if faltam:
        raise FeaturesError(f"climatologia: coluna(s) ausente(s): {faltam}")
    c = clim.loc[clim["cod_ibge"].isin(cods), ["cod_ibge", "mes_calendario", "mediana_mm"]]
    if c.duplicated(["cod_ibge", "mes_calendario"]).any() or len(c) != 12 * len(cods):
        raise FeaturesError(f"climatologia deve ter 12 meses por município ({len(cods)} município(s))")
    p = c.pivot(index="mes_calendario", columns="cod_ibge", values="mediana_mm")
    p = p.reindex(index=range(1, 13), columns=list(cods)).to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        return np.log1p(p)


# ---------------------------------------------------------------- construção
def construir(demo: pd.DataFrame, serie_imputada: pd.DataFrame, climatologia: pd.DataFrame,
              municipios: pd.DataFrame, cfg: Config) -> Conjunto:
    """Monta features e alvos de todas as origens com histórico completo.

    `demo` = `demo_observado` (SPI e flags), `serie_imputada` = `quality.avaliar_qualidade(...).serie_imputada`,
    `climatologia` = artefato `climatologia_precip`, `municipios` = `qualidade_municipio`."""
    escalas = tuple(cfg.spi.janelas_meses)
    L, h = cfg.features.max_lag_meses, cfg.alvo.horizonte_h
    if L < LAGS_SPI:
        raise FeaturesError(f"features.max_lag_meses={L}: use pelo menos {LAGS_SPI} (lags do SPI)")
    periodos = pd.period_range(cfg.janela.inicio, cfg.janela.fim, freq="M")
    T = len(periodos)
    if T < L + 1:
        raise FeaturesError(f"janela de {T} mês(es) curta demais para {L} meses de histórico")
    meses = periodos.to_timestamp()
    cods = _municipios(municipios)
    J = len(cods)
    n = T - L + 1                                                # origens: t = janela.inicio + L - 1 .. janela.fim

    chuva = _matriz(serie_imputada, "precip_mm", meses, cods, "serie_imputada").to_numpy(dtype="float64")
    falta = _matriz(serie_imputada, "precip_mm_missing", meses, cods, "serie_imputada").to_numpy(dtype="float64") > 0.5
    if np.any(chuva < 0):                                        # NaN compara falso: só chuva negativa real cai aqui
        raise FeaturesError("serie_imputada com precip_mm negativa")
    falta = falta | np.isnan(chuva)
    with np.errstate(invalid="ignore"):
        chuva = np.log1p(chuva)

    spi, ausente, baixa = {}, {}, {}
    for k in escalas:
        v = _matriz(demo, f"spi{k}", meses, cods, "demo_observado").to_numpy(dtype="float64")
        inv = _matriz(demo, f"spi{k}_invalid", meses, cods, "demo_observado").to_numpy(dtype="float64") > 0.5
        bx = _matriz(demo, f"spi{k}_baixa_conf", meses, cods, "demo_observado").to_numpy(dtype="float64") > 0.5
        ausente[k] = inv | np.isnan(v)
        spi[k] = np.where(ausente[k], np.nan, v)
        baixa[k] = bx & ~ausente[k]
    clim = _clim_log(climatologia, cods)

    origem, alvo_p = periodos[L - 1:], periodos[L - 1:] + h
    colunas: list[np.ndarray] = []
    nomes: list[str] = []
    continuas: list[bool] = []

    def add(nome: str, arr: np.ndarray, continua: bool) -> None:     # arr (n, J) -> cod-major (J * n,)
        colunas.append(np.asarray(arr, dtype="float64").T.ravel())
        nomes.append(nome)
        continuas.append(continua)

    def lag(M: np.ndarray, k: int) -> np.ndarray:                     # valor do mês t-k para cada origem t
        return M[L - 1 - k: T - k]

    for k in range(L):
        add(f"chuva_lag{k}", lag(chuva, k), True)
    for k in range(L):
        add(f"chuva_falta_lag{k}", lag(falta, k), False)
    for K in escalas:
        for j in range(LAGS_SPI):
            add(f"spi{K}_lag{j}", lag(spi[K], j), True)
        for j in range(LAGS_SPI):
            add(f"spi{K}_ausente_lag{j}", lag(ausente[K], j), False)
        add(f"spi{K}_baixa_lag0", lag(baixa[K], 0), False)
    ang = 2.0 * np.pi * np.asarray(alvo_p.month, dtype="float64") / 12.0
    add("mes_alvo_sin", np.repeat(np.sin(ang)[:, None], J, axis=1), False)
    add("mes_alvo_cos", np.repeat(np.cos(ang)[:, None], J, axis=1), False)
    add("clim_mm_alvo", clim[np.asarray(alvo_p.month) - 1], True)     # (n, J)

    def alvo(M: np.ndarray, vazio) -> np.ndarray:                    # valor do mês t+h; vazio depois do fim da janela
        pad = np.concatenate([M, np.full((h, J), vazio, dtype=M.dtype)], axis=0)
        return pad[L - 1 + h: T + h]

    Y = np.column_stack([alvo(spi[k], np.nan).T.ravel() for k in escalas]).astype("float32")
    meta = pd.DataFrame({
        "cod_ibge": np.repeat(np.asarray(cods, dtype=object), n),
        "mes_origem": np.tile(origem.to_timestamp().to_numpy(), J),
        "mes_alvo": np.tile(alvo_p.to_timestamp().to_numpy(), J)})
    for k in escalas:
        meta[f"alvo_spi{k}_baixa_conf"] = alvo(baixa[k], False).T.ravel()

    rot = np.full(n, POS_TESTE, dtype=object)                         # splits.rotular recusa mes_alvo > teste_fim
    na_janela = alvo_p <= cfg.split.teste_fim
    if na_janela.any():
        rot[na_janela] = splits.rotular(alvo_p[na_janela].to_timestamp(), cfg, escalas).astype(str).to_numpy()
    meta["split"] = pd.Categorical(np.tile(rot, J), categories=list(BLOCOS))
    if not cfg.split.teste_liberado:
        Y[(meta["split"] == "test").to_numpy()] = np.nan             # teste congelado: o alvo nem entra na memória
    M = np.isfinite(Y)

    meta = meta[["cod_ibge", "mes_origem", "mes_alvo", "split", *[f"alvo_spi{k}_baixa_conf" for k in escalas]]]
    return Conjunto(X=np.column_stack(colunas).astype("float32"), Y=Y, M=M, colunas=tuple(nomes),
                    continuas=np.asarray(continuas, dtype=bool),
                    alvos=tuple(f"alvo_spi{k}_h{h}" for k in escalas), meta=meta)


def selecionar(c: Conjunto, nome: str, cfg: Config, *, so_com_rotulo: bool = False) -> Conjunto:
    """Linhas do bloco `nome`. O bloco `test` recusa antes de qualquer cálculo se não liberado."""
    if nome not in BLOCOS:
        raise FeaturesError(f"bloco {nome!r} desconhecido; use um de {BLOCOS}")
    if nome == "test":
        splits.exigir_teste_liberado(cfg)
    linhas = (c.meta["split"] == nome).to_numpy()
    if so_com_rotulo:
        linhas = linhas & c.M.any(axis=1)
    return c.fatiar(linhas)


# ---------------------------------------------------------------- padronização (só treino)
@dataclass(frozen=True, eq=False)
class Padronizador:
    colunas: tuple[str, ...]
    continuas: np.ndarray          # (d,) bool
    media: np.ndarray              # (d,) float64; 0 nas colunas que não são contínuas
    desvio: np.ndarray             # (d,) float64; 1 nas colunas que não são contínuas

    @classmethod
    def ajustar(cls, treino: Conjunto) -> "Padronizador":
        """Média e desvio das colunas contínuas, calculados só com linhas do bloco `train`."""
        if len(treino) == 0:
            raise FeaturesError("conjunto de treino vazio")
        if not (treino.meta["split"].astype(str) == "train").all():
            raise FeaturesError("o padronizador só pode ser ajustado com linhas do bloco 'train'")
        X = treino.X.astype("float64")
        ok = np.isfinite(X)
        cont = np.maximum(ok.sum(axis=0), 1)
        media = np.where(ok, X, 0.0).sum(axis=0) / cont
        var = np.where(ok, (X - media) ** 2, 0.0).sum(axis=0) / cont
        desvio = np.sqrt(var)
        desvio = np.where(desvio < DESVIO_MIN, 1.0, desvio)
        media = np.where(treino.continuas, media, 0.0)
        desvio = np.where(treino.continuas, desvio, 1.0)
        return cls(colunas=treino.colunas, continuas=treino.continuas.copy(), media=media, desvio=desvio)

    def transformar(self, c: Conjunto) -> np.ndarray:
        """X padronizado (float32, sem NaN): NaN vira 0, a média do treino. As flags de ausência já estão em X."""
        if c.colunas != self.colunas:
            raise FeaturesError("as colunas do conjunto não batem com as do padronizador")
        Z = (c.X.astype("float64") - self.media) / self.desvio
        Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)
        return Z.astype("float32")

    def para_dict(self) -> dict:
        """JSON puro, para guardar junto dos pesos."""
        return {"colunas": list(self.colunas), "continuas": [bool(x) for x in self.continuas],
                "media": [float(x) for x in self.media], "desvio": [float(x) for x in self.desvio]}

    @classmethod
    def de_dict(cls, d: dict) -> "Padronizador":
        try:
            cols, cont = tuple(d["colunas"]), np.asarray(d["continuas"], dtype=bool)
            media, desvio = np.asarray(d["media"], dtype="float64"), np.asarray(d["desvio"], dtype="float64")
        except (KeyError, TypeError, ValueError) as e:
            raise FeaturesError(f"padronizador inválido: {e}") from e
        if not (len(cols) == len(cont) == len(media) == len(desvio)):
            raise FeaturesError("padronizador com tamanhos inconsistentes")
        if not np.isfinite(media).all() or not (np.isfinite(desvio).all() and (desvio > 0).all()):
            raise FeaturesError("padronizador com média ou desvio inválido")
        return cls(colunas=cols, continuas=cont, media=media, desvio=desvio)
