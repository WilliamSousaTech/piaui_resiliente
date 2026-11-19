"""Rede de correção sobre o baseline parcial, contexto regional e conjunto de redes (Item 9). Sem I/O.

Ideia: o SPI-K de t+1 já contém K-1 meses de chuva conhecidos. O baseline `ParcialClimatologica` faz essa conta
(com a chuva mediana do mês que falta) e erra menos que uma rede que precisa reaprender a conta. Aqui a rede
recebe o baseline como entrada e aprende só o que falta: `Y' = Y - baseline`; a previsão final é
`baseline + rede(X, baseline)`. A rede continua treinada do zero em PyTorch; o baseline não aprende nada com o
rótulo (usa os parâmetros Gama e a climatologia do treino) e não é Ridge, floresta nem XGBoost.

Anti-vazamento:
- o contexto regional de uma origem t usa só linhas da MESMA origem (todas com dados até t); como o split é
  pelo mês-alvo, todas essas linhas caem no mesmo bloco;
- a padronização das colunas novas é ajustada só no treino (`Padronizador.ajustar` sobre o conjunto aumentado);
- o rótulo ausente continua NaN (a máscara `M` não muda).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.model import features as fe
from src.model.metricas import LIMIAR_SECA

PREFIXO_REGIONAL = "reg_"
PREFIXO_BASE = "base_"
LAGS_CHUVA_REGIONAL = 3


class ResiduoError(ValueError):
    """Conjunto ou previsão incompatível com a correção sobre o baseline."""


def _indices(c: fe.Conjunto, nomes: Sequence[str]) -> list[int]:
    faltam = [n for n in nomes if n not in c.colunas]
    if faltam:
        raise ResiduoError(f"colunas ausentes do conjunto: {faltam}")
    return [c.colunas.index(n) for n in nomes]


def adicionar_regional(c: fe.Conjunto) -> fe.Conjunto:
    """Acrescenta ao conjunto a média entre municípios, no mesmo mês de origem, de `chuva_lag0..2` e do SPI de
    cada escala no lag 0, mais a fração de municípios em seca (SPI <= -1,0) por escala.

    Chamar no conjunto COMPLETO, antes de `selecionar`: assim todos os municípios da origem entram na média.
    Valores ausentes ficam de fora da média; origem sem nenhum valor vira NaN (a padronização troca por 0)."""
    if any(n.startswith(PREFIXO_REGIONAL) for n in c.colunas):
        raise ResiduoError("o contexto regional já foi acrescentado a este conjunto")
    chuva = [f"chuva_lag{j}" for j in range(LAGS_CHUVA_REGIONAL)]
    spi = [f"spi{K}_lag0" for K in c.escalas]
    X = c.X.astype("float64")
    grupos = c.meta["mes_origem"].to_numpy()
    cont = pd.DataFrame(X[:, _indices(c, chuva + spi)], columns=chuva + spi)
    medias = cont.groupby(grupos).transform("mean")                              # média ignora NaN
    seca = pd.DataFrame({f"{n}_seca": np.where(np.isfinite(cont[n]), (cont[n] <= LIMIAR_SECA).astype("float64"), np.nan)
                         for n in spi})
    fracao = seca.groupby(grupos).transform("mean")
    novas = pd.concat([medias, fracao], axis=1)
    novas.columns = [PREFIXO_REGIONAL + n for n in novas.columns]
    return dataclasses.replace(
        c, X=np.column_stack([c.X, novas.to_numpy(dtype="float32")]), colunas=(*c.colunas, *novas.columns),
        continuas=np.concatenate([c.continuas, np.ones(novas.shape[1], dtype=bool)]))


def com_base(c: fe.Conjunto, base: np.ndarray) -> fe.Conjunto:
    """Conjunto de correção: `X' = [X, base]`, `Y' = Y - base` onde há rótulo (NaN onde não há) e `M` igual.

    Só serve para treinar e prever a correção; para medir erro use o conjunto original e a previsão completa."""
    base = np.asarray(base)
    if base.shape != (len(c), len(c.alvos)) or not np.isfinite(base).all():
        raise ResiduoError(f"base deve ser finita com shape {(len(c), len(c.alvos))} (recebi {base.shape})")
    if any(n.startswith(PREFIXO_BASE) for n in c.colunas):
        raise ResiduoError("o baseline já foi acrescentado a este conjunto")
    Y = np.where(c.M, c.Y.astype("float64") - base.astype("float64"), np.nan).astype("float32")
    nomes = tuple(PREFIXO_BASE + a for a in c.alvos)
    return dataclasses.replace(
        c, X=np.column_stack([c.X, base.astype("float32")]), Y=Y, colunas=(*c.colunas, *nomes),
        continuas=np.concatenate([c.continuas, np.ones(len(nomes), dtype=bool)]))


class ResiduoPreditor:
    """`baseline + rede(X, baseline)`, com a mesma interface dos demais: `prever(Conjunto) -> (n, cabeças)`."""
    nome = "rede_residuo"

    def __init__(self, base, preditor):
        self.base, self.preditor = base, preditor

    def prever(self, c: fe.Conjunto) -> np.ndarray:
        b = np.asarray(self.base.prever(c))
        return (b.astype("float64") + self.preditor.prever(com_base(c, b)).astype("float64")).astype("float32")


class MediaDePreditores:
    """Média de redes treinadas com sementes diferentes (conjunto). `desvio` é a dispersão entre elas."""
    nome = "conjunto"

    def __init__(self, preditores: Sequence):
        if len(preditores) < 1:
            raise ResiduoError("o conjunto precisa de pelo menos um preditor")
        self.preditores = tuple(preditores)

    def _todas(self, c: fe.Conjunto) -> np.ndarray:
        return np.stack([p.prever(c).astype("float64") for p in self.preditores], axis=0)

    def prever(self, c: fe.Conjunto) -> np.ndarray:
        return self._todas(c).mean(axis=0).astype("float32")

    def desvio(self, c: fe.Conjunto) -> np.ndarray:
        """Desvio-padrão entre as redes por linha e cabeça (0 com um só membro)."""
        return self._todas(c).std(axis=0).astype("float32")
