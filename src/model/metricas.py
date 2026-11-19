"""Métricas de erro do modelo e dos baselines (Item 9). Puro numpy: sem I/O, sem rede, sem torch.

Regra de justiça: todo modelo é pontuado nas MESMAS linhas, as que têm rótulo (`M` verdadeiro). Previsão
NaN numa linha com rótulo é erro do modelo, não motivo para descartar a linha: a função recusa em vez de
calcular a métrica só sobre as linhas "fáceis".

"Em seca" segue o contrato: SPI <= -1,0 (`LIMIAR_SECA`).
"""
from __future__ import annotations

import numpy as np

LIMIAR_SECA = -1.0


class MetricasError(ValueError):
    """Entrada incoerente para o cálculo das métricas."""


def _preparar(y, p, m) -> tuple[np.ndarray, np.ndarray]:
    y, p, m = np.asarray(y), np.asarray(p), np.asarray(m)
    if y.ndim != 1 or y.shape != p.shape or y.shape != m.shape:
        raise MetricasError(f"y, p e m devem ser vetores do mesmo tamanho (recebi {y.shape}, {p.shape}, {m.shape})")
    if m.dtype != bool:
        raise MetricasError("a máscara m deve ser bool")
    ys, ps = y[m].astype("float64"), p[m].astype("float64")
    if not np.isfinite(ys).all():
        raise MetricasError("rótulo não finito em linha com máscara verdadeira")
    if not np.isfinite(ps).all():
        raise MetricasError("previsão NaN ou infinita em linha com rótulo: o modelo precisa prever todas as linhas")
    return ys, ps


def erros(y, p, m) -> dict:
    """MAE, RMSE, viés (previsão - alvo) e n, sobre as linhas com rótulo."""
    ys, ps = _preparar(y, p, m)
    if len(ys) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "vies": float("nan")}
    e = ps - ys
    return {"n": int(len(ys)), "mae": float(np.abs(e).mean()), "rmse": float(np.sqrt((e ** 2).mean())),
            "vies": float(e.mean())}


def seca(y, p, m, limiar: float = LIMIAR_SECA) -> dict:
    """Acerto de "em seca" (SPI <= limiar): contagens, precisão, revocação e F1. NaN quando o denominador é 0."""
    ys, ps = _preparar(y, p, m)
    real, prev = ys <= limiar, ps <= limiar
    tp, fp = int((real & prev).sum()), int((~real & prev).sum())
    fn, tn = int((real & ~prev).sum()), int((~real & ~prev).sum())
    div = lambda a, b: float(a / b) if b > 0 else float("nan")        # noqa: E731
    prec, rev = div(tp, tp + fp), div(tp, tp + fn)
    f1 = div(2 * prec * rev, prec + rev) if np.isfinite(prec) and np.isfinite(rev) else float("nan")
    return {"n": int(len(ys)), "secas_reais": tp + fn, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precisao": prec, "revocacao": rev, "f1": f1}


def quantil_erro_abs(y, p, m, q: float) -> float:
    """Quantil `q` do |erro| nas linhas com rótulo: a margem de erro que o site mostra ao lado da previsão.

    Calcular na validação e aplicar a previsões novas (nunca no próprio teste)."""
    if not (0.0 < q < 1.0):
        raise MetricasError(f"q deve estar entre 0 e 1 (recebi {q!r})")
    ys, ps = _preparar(y, p, m)
    if len(ys) == 0:
        return float("nan")
    return float(np.quantile(np.abs(ps - ys), q))


def cobertura_da_margem(y, p, m, margem: float) -> float:
    """Fração das linhas com rótulo em que |erro| <= margem (conferência da margem em outro bloco)."""
    ys, ps = _preparar(y, p, m)
    if len(ys) == 0:
        return float("nan")
    return float((np.abs(ps - ys) <= margem).mean())


def ic_bootstrap_dif_mae(y, p_a, p_b, m, grupo, *, n_boot: int = 2000, seed: int, nivel: float = 0.95) -> dict:
    """Diferença de MAE (A - B) com intervalo por bootstrap de GRUPOS (meses-alvo inteiros).

    Os erros dos municípios no mesmo mês são correlacionados (a seca é regional), então reamostrar linhas
    soltas encolheria o intervalo. Aqui reamostram-se meses. `dif` negativa = A erra menos que B; o intervalo
    que não contém 0 indica diferença distinguível de ruído (nunca prova de ganho fora da amostra)."""
    if not (0.0 < nivel < 1.0) or n_boot < 100:
        raise MetricasError("nivel deve estar entre 0 e 1 e n_boot >= 100")
    ys, pa = _preparar(y, p_a, m)
    _, pb = _preparar(y, p_b, m)
    g = np.asarray(grupo)
    if g.shape != np.asarray(m).shape:
        raise MetricasError("grupo deve ter o mesmo tamanho de m")
    d = np.abs(pa - ys) - np.abs(pb - ys)
    _, inv = np.unique(g[np.asarray(m)], return_inverse=True)
    n_grupos = int(inv.max()) + 1 if len(inv) else 0
    if n_grupos < 2:
        return {"dif": float(d.mean()) if len(d) else float("nan"), "ic_inf": float("nan"), "ic_sup": float("nan"),
                "n_grupos": n_grupos, "n": int(len(d))}
    soma, cont = np.bincount(inv, weights=d), np.bincount(inv)
    sorteio = np.random.default_rng(seed).integers(0, n_grupos, size=(n_boot, n_grupos))
    est = soma[sorteio].sum(axis=1) / cont[sorteio].sum(axis=1)
    lo, hi = np.quantile(est, [(1 - nivel) / 2, 1 - (1 - nivel) / 2])
    return {"dif": float(d.mean()), "ic_inf": float(lo), "ic_sup": float(hi), "n_grupos": n_grupos, "n": int(len(d))}
