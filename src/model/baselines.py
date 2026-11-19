"""Baselines comparativos (Item 9). SÓ para comparar: o núcleo preditivo é a rede PyTorch treinada do zero.

Todo baseline devolve, para cada linha de um `Conjunto`, uma previsão por cabeça: array (n, c) float32,
sem NaN. Onde o baseline não consegue prever (entrada ausente, parâmetro inválido), a previsão é 0, o SPI
esperado pela construção do índice. Assim todos são pontuados nas mesmas linhas (`metricas.py`).

  Persistencia            SPI-K de t como previsão do SPI-K de t+h
  Climatologia            média do alvo no treino por (município, mês-calendário do mês-alvo)
  ParcialClimatologica    soma de chuva dos K-1 meses já observados mais a chuva mediana do mês-alvo,
                          convertida em SPI pela Gama do treino (só h = 1). É o baseline "esperto" a bater:
                          a rede só ganha se prever algo além da chuva mediana do mês que falta
  RidgeBaseline           regressão linear regularizada; alfa escolhido na validação
  Floresta                Random Forest (scikit-learn)

Anti-vazamento: `ajustar` só aceita linhas do bloco `train` (a validação serve apenas para escolher o
alfa do Ridge); padronização vem do `Padronizador` ajustado no treino; o teste nunca passa por aqui.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from src.data.spi import COLUNAS_PARAMETROS, EPS_H
from src.model import features as fe
from src.model import metricas as mt

ALFAS = (0.1, 1.0, 10.0, 100.0, 1000.0)


class BaselineError(ValueError):
    """Ajuste ou previsão incompatível com o baseline."""


def _exigir_bloco(c: fe.Conjunto, nome: str, quem: str) -> None:
    if len(c) == 0 or not (c.meta["split"].astype(str) == nome).all():
        raise BaselineError(f"{quem}: use só linhas do bloco {nome!r} (e pelo menos uma)")


def _saida(P: np.ndarray) -> np.ndarray:
    return np.nan_to_num(np.asarray(P, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def _coluna(c: fe.Conjunto, nome: str) -> np.ndarray:
    if nome not in c.colunas:
        raise BaselineError(f"coluna {nome!r} ausente do conjunto")
    return c.X[:, c.colunas.index(nome)].astype("float64")


# ---------------------------------------------------------------- sem parâmetros aprendidos
class Persistencia:
    nome = "persistencia"

    def prever(self, c: fe.Conjunto) -> np.ndarray:
        return _saida(np.column_stack([_coluna(c, f"spi{K}_lag0") for K in c.escalas]))


class Climatologia:
    nome = "climatologia"

    def __init__(self, tabelas: Sequence[pd.Series]):
        self.tabelas = tuple(tabelas)

    @classmethod
    def ajustar(cls, treino: fe.Conjunto) -> "Climatologia":
        _exigir_bloco(treino, "train", "climatologia")
        cod = treino.meta["cod_ibge"].to_numpy()
        mes = treino.meta["mes_alvo"].dt.month.to_numpy()
        tabelas = []
        for i in range(treino.Y.shape[1]):
            m = treino.M[:, i]
            df = pd.DataFrame({"cod": cod[m], "mes": mes[m], "y": treino.Y[m, i].astype("float64")})
            tabelas.append(df.groupby(["cod", "mes"])["y"].mean())
        return cls(tabelas)

    def prever(self, c: fe.Conjunto) -> np.ndarray:
        if len(c.alvos) != len(self.tabelas):
            raise BaselineError("número de cabeças diferente do usado no ajuste")
        idx = pd.MultiIndex.from_arrays([c.meta["cod_ibge"].to_numpy(), c.meta["mes_alvo"].dt.month.to_numpy()])
        return _saida(np.column_stack([t.reindex(idx).to_numpy(dtype="float64") for t in self.tabelas]))


class ParcialClimatologica:
    nome = "parcial_climatologica"

    def __init__(self, params_gama: pd.DataFrame):
        if not set(COLUNAS_PARAMETROS) <= set(params_gama.columns):
            raise BaselineError(f"params_gama sem as colunas {COLUNAS_PARAMETROS}")
        self._p = params_gama.set_index(["escala_meses", "cod_ibge", "mes_calendario"])[["q", "shape", "scale", "valido"]]
        if self._p.index.duplicated().any():
            raise BaselineError("params_gama com chave (escala, cod_ibge, mês) repetida")

    def prever(self, c: fe.Conjunto) -> np.ndarray:
        if c.horizonte != 1:
            raise BaselineError("o baseline parcial só está definido para h = 1")
        cod = c.meta["cod_ibge"].to_numpy()
        mes = c.meta["mes_alvo"].dt.month.to_numpy()
        mediana = np.expm1(_coluna(c, "clim_mm_alvo"))
        saidas = []
        for K in c.escalas:
            S = mediana.copy()                                          # o mês que falta: chuva mediana do treino
            for j in range(K - 1):                                      # os K-1 meses já observados: t, t-1, ...
                S += np.expm1(_coluna(c, f"chuva_lag{j}"))
            p = self._p.reindex(pd.MultiIndex.from_arrays([np.full(len(c), K), cod, mes]))
            q, a, esc = (p[k].to_numpy(dtype="float64") for k in ("q", "shape", "scale"))
            ok = p["valido"].fillna(False).to_numpy(dtype=bool) & np.isfinite(S) & np.isfinite(q) \
                & np.isfinite(a) & np.isfinite(esc)
            S, a, esc, q = (np.where(ok, v, f) for v, f in ((S, 1.0), (a, 1.0), (esc, 1.0), (q, 0.0)))
            h = np.clip(q + (1.0 - q) * stats.gamma.cdf(S, a, scale=esc), EPS_H, 1.0 - EPS_H)   # mesma conta do spi.py
            saidas.append(np.where(ok, stats.norm.ppf(h), 0.0))
        return _saida(np.column_stack(saidas))


# ---------------------------------------------------------------- aprendidos (só no treino)
class RidgeBaseline:
    nome = "ridge"

    def __init__(self, padronizador: fe.Padronizador, modelos: Sequence[Ridge], alfas: Sequence[float]):
        self.padronizador, self.modelos, self.alfas = padronizador, tuple(modelos), tuple(alfas)

    @classmethod
    def ajustar(cls, treino: fe.Conjunto, val: fe.Conjunto, padronizador: fe.Padronizador,
                alfas: Sequence[float] = ALFAS) -> "RidgeBaseline":
        """Ajusta no treino; a validação só escolhe o alfa de cada cabeça (menor MAE)."""
        _exigir_bloco(treino, "train", "ridge")
        _exigir_bloco(val, "val", "ridge")
        Ztr, Zva = padronizador.transformar(treino), padronizador.transformar(val)
        modelos, escolhidos = [], []
        for i in range(treino.Y.shape[1]):
            mtr, mva = treino.M[:, i], val.M[:, i]
            if not mtr.any() or not mva.any():
                raise BaselineError(f"ridge: cabeça {treino.alvos[i]} sem rótulo no treino ou na validação")
            melhor = None
            for alfa in alfas:
                r = Ridge(alpha=float(alfa)).fit(Ztr[mtr], treino.Y[mtr, i])
                mae = mt.erros(val.Y[:, i], r.predict(Zva).astype("float32"), mva)["mae"]
                if melhor is None or mae < melhor[0]:
                    melhor = (mae, float(alfa), r)
            escolhidos.append(melhor[1])
            modelos.append(melhor[2])
        return cls(padronizador, modelos, escolhidos)

    def prever(self, c: fe.Conjunto) -> np.ndarray:
        Z = self.padronizador.transformar(c)
        return _saida(np.column_stack([m.predict(Z) for m in self.modelos]))


class Floresta:
    nome = "floresta_aleatoria"

    def __init__(self, padronizador: fe.Padronizador, modelos: Sequence[RandomForestRegressor]):
        self.padronizador, self.modelos = padronizador, tuple(modelos)

    @classmethod
    def ajustar(cls, treino: fe.Conjunto, padronizador: fe.Padronizador, *, seed: int, n_arvores: int = 100,
                folha_min: int = 20, max_features: float = 0.5) -> "Floresta":
        _exigir_bloco(treino, "train", "floresta")
        Z = padronizador.transformar(treino)
        modelos = []
        for i in range(treino.Y.shape[1]):
            m = treino.M[:, i]
            if not m.any():
                raise BaselineError(f"floresta: cabeça {treino.alvos[i]} sem rótulo no treino")
            rf = RandomForestRegressor(n_estimators=n_arvores, min_samples_leaf=folha_min, max_features=max_features,
                                       n_jobs=-1, random_state=seed)
            modelos.append(rf.fit(Z[m], treino.Y[m, i]))
        return cls(padronizador, modelos)

    def prever(self, c: fe.Conjunto) -> np.ndarray:
        Z = self.padronizador.transformar(c)
        return _saida(np.column_stack([m.predict(Z) for m in self.modelos]))


def ajustar_todos(treino: fe.Conjunto, val: fe.Conjunto, padronizador: fe.Padronizador,
                  params_gama: pd.DataFrame, *, seed: int, n_arvores: int = 100) -> dict[str, object]:
    """Os cinco baselines, na ordem em que aparecem no relatório."""
    b = [Persistencia(), Climatologia.ajustar(treino), ParcialClimatologica(params_gama),
         RidgeBaseline.ajustar(treino, val, padronizador),
         Floresta.ajustar(treino, padronizador, seed=seed, n_arvores=n_arvores)]
    return {x.nome: x for x in b}
