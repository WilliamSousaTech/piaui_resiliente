"""Destaques do site calculados só com o que já foi observado: mês de abertura, maiores secas e séries do clima.

Entrada: a matriz de classes do SPI (município x mês; 255 = sem dado) e a de baixa confiança (0/1).
Nada aqui treina, prevê ou lê arquivo. "Leitura confiável" = há SPI e o mês não está marcado como baixa
confiança (regra D13). As duas funções só olham meses em que a maioria das leituras é confiável, para
que um mês de estiagem, em que quase tudo é baixa confiança, não vire manchete.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEM_DADO = 255
ULTIMA_CLASSE_SECA = 2          # 0 seca extrema, 1 severa, 2 moderada (SPI <= -1,0)
MINIMO_CONFIAVEL = 0.5          # fatia mínima de leituras confiáveis para o mês poder ser destaque
SEPARACAO_MESES = 12            # o SPI-3 de meses vizinhos descreve quase a mesma chuva: um destaque por episódio


def _matrizes(cls, baixa) -> tuple[np.ndarray, np.ndarray]:
    c, b = np.asarray(cls), np.asarray(baixa)
    if c.ndim != 2 or c.shape != b.shape:
        raise ValueError(f"classes e baixa confiança precisam ter a mesma forma (município x mês): {c.shape} e {b.shape}")
    valido = c != SEM_DADO
    return valido, valido & (b == 0)


def mes_abertura(cls, baixa, minimo_confiavel: float = MINIMO_CONFIAVEL) -> int:
    """Índice do último mês em que a maioria dos municípios com SPI tem leitura confiável.

    Sem nenhum mês assim, devolve o último mês com algum SPI. Sem SPI em lugar nenhum, é erro."""
    valido, confiavel = _matrizes(cls, baixa)
    n_valido, n_conf = valido.sum(axis=0), confiavel.sum(axis=0)
    if not n_valido.any():
        raise ValueError("nenhum mês tem SPI")
    boa = (n_valido > 0) & (n_conf >= minimo_confiavel * n_valido)
    candidatos = np.flatnonzero(boa if boa.any() else n_valido > 0)
    return int(candidatos.max())


def maiores_secas(cls, baixa, *, n: int = 8, separacao: int = SEPARACAO_MESES,
                  minimo_confiavel: float = MINIMO_CONFIAVEL) -> list[dict]:
    """Os `n` meses com maior fatia de municípios em seca entre as leituras confiáveis, um por episódio.

    Escolha gulosa: pega o pior mês, descarta os que estão a menos de `separacao` meses dele e repete.
    Empate na fatia: mais municípios em seca primeiro; depois, o mês mais recente."""
    if n < 1 or separacao < 1:
        raise ValueError("n e separacao precisam ser >= 1")
    c = np.asarray(cls)
    valido, confiavel = _matrizes(cls, baixa)
    n_valido, n_conf = valido.sum(axis=0), confiavel.sum(axis=0)
    n_seca = ((c <= ULTIMA_CLASSE_SECA) & confiavel).sum(axis=0)
    elegivel = (n_valido > 0) & (n_conf >= minimo_confiavel * n_valido) & (n_conf > 0)
    fatia = np.where(elegivel, n_seca / np.maximum(n_conf, 1), -1.0)
    ordem = sorted(np.flatnonzero(elegivel).tolist(), key=lambda t: (-fatia[t], -n_seca[t], -t))
    escolhidos: list[int] = []
    for t in ordem:
        if all(abs(t - s) >= separacao for s in escolhidos):
            escolhidos.append(t)
        if len(escolhidos) == n:
            break
    return [{"t": int(t), "n_seca": int(n_seca[t]), "n_confiaveis": int(n_conf[t]), "n_validos": int(n_valido[t])}
            for t in escolhidos]


# ------------------------------------------------------------------ clima: quando chove e como foram as estações
def _como_periodo(mes) -> "pd.Period":
    return mes if isinstance(mes, pd.Period) else pd.Period(str(mes)[:7], freq="M")


def _preparar_chuva(chuva, meses, incluir):
    a = np.asarray(chuva, dtype="float64")
    idx = pd.PeriodIndex(meses, freq="M")
    if a.ndim != 2 or a.shape[1] != len(idx):
        raise ValueError(f"chuva precisa ser município x mês com {len(idx)} meses; veio {a.shape}")
    mascara = np.ones(a.shape[0], dtype=bool) if incluir is None else np.asarray(incluir, dtype=bool)
    if mascara.shape != (a.shape[0],) or not mascara.any():
        raise ValueError("`incluir` precisa marcar ao menos um município e ter um valor por município")
    return a[mascara], idx


def climatologia_mensal(chuva, meses, treino_fim, incluir=None) -> list[float]:
    """Chuva média (mm) de cada mês do calendário: média entre municípios incluídos e entre os anos até `treino_fim`."""
    a, idx = _preparar_chuva(chuva, meses, incluir)
    fim = _como_periodo(treino_fim)
    saida = []
    for m in range(1, 13):
        cols = np.flatnonzero((idx.month == m) & (idx <= fim))
        vals = a[:, cols]
        if cols.size == 0 or np.isnan(vals).all():
            raise ValueError(f"sem chuva observada para o mês {m} até {fim}")
        saida.append(round(float(np.nanmean(vals)), 1))
    return saida


def estacao_chuvosa(chuva, meses, treino_fim, incluir=None, largura: int = 4) -> dict:
    """Os `largura` meses seguidos mais chuvosos (pela climatologia) e o total de cada estação, ano a ano.

    A estação de um ano começa no mês `inicio_mes` desse ano e pode passar para o ano seguinte. Só entram estações
    completas, sem lacuna. `desvio_pct` compara o total com a média das estações que terminam até `treino_fim`."""
    if not 1 <= largura <= 11:
        raise ValueError("largura precisa estar entre 1 e 11")
    a, idx = _preparar_chuva(chuva, meses, incluir)
    clim = climatologia_mensal(chuva, meses, treino_fim, incluir)
    somas = [sum(clim[(s + j) % 12] for j in range(largura)) for s in range(12)]
    s0 = int(np.argmax(somas))                                   # primeiro mês (0 = janeiro) da estação
    cal = [(s0 + j) % 12 + 1 for j in range(largura)]            # meses do calendário da estação
    posicao = {(p.year, p.month): i for i, p in enumerate(idx)}
    fim = _como_periodo(treino_fim)
    anos, totais_treino = [], []
    for ano in range(int(idx.year.min()), int(idx.year.max()) + 1):
        cols = [posicao.get((ano + (1 if (s0 + j) >= 12 else 0), m)) for j, m in enumerate(cal)]
        if any(c is None for c in cols):
            continue
        blocos = a[:, cols]
        if np.isnan(blocos).any():
            continue
        total = float(blocos.sum(axis=1).mean())
        ultimo = idx[cols[-1]]
        anos.append({"ano": ano, "total_mm": round(total, 1), "_treino": bool(ultimo <= fim)})
        if ultimo <= fim:
            totais_treino.append(total)
    if not totais_treino:
        raise ValueError("nenhuma estação completa dentro do período de treino")
    media = float(np.mean(totais_treino))
    return {"meses": cal, "largura": largura, "media_mm": round(media, 1),
            "anos": [{"ano": x["ano"], "total_mm": x["total_mm"], "desvio_pct": round((x["total_mm"] / media - 1) * 100, 1)}
                     for x in anos]}
