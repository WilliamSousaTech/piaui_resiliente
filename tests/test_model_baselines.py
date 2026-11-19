"""Testes de src/model/baselines.py (Item 9). Dados sintéticos só aqui; números sem sentido climático.

O que se prova: mecânica (formatos, ausências), anti-vazamento (só treino ajusta; validação só escolhe o
alfa) e a conta do baseline parcial contra o SPI do próprio projeto (oráculo com a chuva real do mês-alvo)."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from dados_sinteticos import A, B, C, PARAMS_REAL, raw_sintetico
from src.data import build_demo as bd
from src.data import quality
from src.data.config import load_config
from src.model import baselines as bl
from src.model import features as fe
from src.model import metricas as mt


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def raw():
    return raw_sintetico()


@pytest.fixture(scope="module")
def dados(cfg, raw):
    r = bd.montar_demo(raw, cfg)
    q = quality.avaliar_qualidade(raw, cfg)
    c = fe.construir(r.demo_observado, q.serie_imputada, r.climatologia_precip, r.qualidade_municipio, cfg)
    tr, va = fe.selecionar(c, "train", cfg), fe.selecionar(c, "val", cfg)
    return {"c": c, "r": r, "tr": tr, "va": va, "p": fe.Padronizador.ajustar(tr)}


def com_y(c, Y):
    return dataclasses.replace(c, Y=np.where(c.M, Y, np.nan).astype("float32"))


def tudo(c):
    return np.ones(len(c), dtype=bool)


# ------------------------------------------------------------------ persistência
def test_persistencia_e_o_spi_de_t_com_zero_onde_falta(dados):
    va = dados["va"]
    X = va.X.copy()
    k6 = va.colunas.index("spi6_lag0")
    X[:5, k6] = np.nan
    P = bl.Persistencia().prever(dataclasses.replace(va, X=X))
    assert P.shape == (len(va), 2) and P.dtype == np.float32 and np.isfinite(P).all()
    np.testing.assert_array_equal(P[5:, 0], np.nan_to_num(va.X[5:, va.colunas.index("spi3_lag0")]))
    np.testing.assert_array_equal(P[5:, 1], np.nan_to_num(va.X[5:, k6]))
    assert (P[:5, 1] == 0).all()


# ------------------------------------------------------------------ climatologia
def test_climatologia_media_do_treino_por_municipio_e_mes(dados):
    tr, va = dados["tr"], dados["va"]
    cl = bl.Climatologia.ajustar(tr)
    P = cl.prever(va)
    assert P.shape == (len(va), 2) and P.dtype == np.float32 and np.isfinite(P).all()
    g = tr.meta[(tr.meta["cod_ibge"] == B) & (tr.meta["mes_alvo"].dt.month == 7)].index
    esperado = tr.Y[g, 0][tr.M[g, 0]].mean()
    i = va.meta.index[(va.meta["cod_ibge"] == B) & (va.meta["mes_alvo"].dt.month == 7)][0]
    assert P[i, 0] == pytest.approx(float(esperado), abs=1e-6)


def test_climatologia_desconhecida_vira_zero_e_nao_ve_validacao(dados):
    tr, va = dados["tr"], dados["va"]
    estranho = dataclasses.replace(va, meta=va.meta.assign(cod_ibge="9999999"))
    assert (bl.Climatologia.ajustar(tr).prever(estranho) == 0).all()
    with pytest.raises(bl.BaselineError, match="train"):
        bl.Climatologia.ajustar(va)
    with pytest.raises(bl.BaselineError, match="cabeças"):
        bl.Climatologia.ajustar(tr).prever(dataclasses.replace(va, alvos=("alvo_spi3_h1",)))


# ------------------------------------------------------------------ parcial: a conta tem que ser a do spi.py
def test_parcial_com_a_chuva_real_do_mes_alvo_reproduz_o_spi(dados, raw):
    c, r = dados["c"], dados["r"]
    P = raw.set_index(["cod_ibge", "mes"])["precip_mm"]
    real = P.reindex(pd.MultiIndex.from_arrays([c.meta["cod_ibge"], c.meta["mes_alvo"]])).to_numpy(dtype="float64")
    tem = np.isfinite(real)
    X = c.X.copy()
    X[tem, c.colunas.index("clim_mm_alvo")] = np.log1p(real[tem]).astype("float32")     # oráculo: o mês que falta é o real
    pred = bl.ParcialClimatologica(r.spi_gamma_params).prever(dataclasses.replace(c, X=X))
    for i in (0, 1):                                                                    # SPI-3 e SPI-6
        sel = c.M[:, i] & tem
        assert sel.sum() > 1000
        assert np.abs(pred[sel, i] - c.Y[sel, i]).max() < 5e-3, c.alvos[i]


def test_parcial_com_a_mediana_e_monotono_na_chuva_observada(dados):
    c, r = dados["va"], dados["r"]
    b = bl.ParcialClimatologica(r.spi_gamma_params)
    base_ = b.prever(c)
    X = c.X.copy()
    X[:, c.colunas.index("chuva_lag0")] += 0.5                                          # mais chuva no último mês observado
    mais = b.prever(dataclasses.replace(c, X=X))
    assert (mais >= base_ - 1e-6).all() and (mais > base_).mean() > 0.9


def test_parcial_parametro_invalido_ou_ausente_vira_zero(dados):
    va, r = dados["va"], dados["r"]
    p = r.spi_gamma_params.copy()
    p.loc[p["cod_ibge"] == A, "valido"] = False
    pred = bl.ParcialClimatologica(p).prever(va)
    do_a = (va.meta["cod_ibge"] == A).to_numpy()
    assert (pred[do_a] == 0).all() and (pred[~do_a] != 0).any()
    sem_b = bl.ParcialClimatologica(p[p["cod_ibge"] != B]).prever(va)
    assert (sem_b[(va.meta["cod_ibge"] == B).to_numpy()] == 0).all()


def test_parcial_recusa_entradas_incoerentes(dados):
    va, r = dados["va"], dados["r"]
    with pytest.raises(bl.BaselineError, match="colunas"):
        bl.ParcialClimatologica(r.spi_gamma_params.drop(columns=["q"]))
    with pytest.raises(bl.BaselineError, match="repetida"):
        bl.ParcialClimatologica(pd.concat([r.spi_gamma_params, r.spi_gamma_params.iloc[:1]]))
    h2 = dataclasses.replace(va, alvos=("alvo_spi3_h2", "alvo_spi6_h2"))
    with pytest.raises(bl.BaselineError, match="h = 1"):
        bl.ParcialClimatologica(r.spi_gamma_params).prever(h2)
    sem_col = dataclasses.replace(va, colunas=tuple("x" + n if n == "clim_mm_alvo" else n for n in va.colunas))
    with pytest.raises(bl.BaselineError, match="clim_mm_alvo"):
        bl.ParcialClimatologica(r.spi_gamma_params).prever(sem_col)


# ------------------------------------------------------------------ ridge
def test_ridge_recupera_relacao_linear_e_escolhe_o_menor_alfa_sem_ruido(dados):
    c, p = dados["c"], dados["p"]
    w = np.linspace(-1.0, 1.0, len(c.colunas))
    y = p.transformar(c) @ w
    plantado = com_y(c, np.column_stack([y, 0.5 * y]))
    tr2 = plantado.fatiar((plantado.meta["split"] == "train").to_numpy())
    va2 = plantado.fatiar((plantado.meta["split"] == "val").to_numpy())
    rd = bl.RidgeBaseline.ajustar(tr2, va2, p)
    assert rd.alfas == (0.1, 0.1)
    err = mt.erros(va2.Y[:, 0], rd.prever(va2)[:, 0], va2.M[:, 0])
    assert err["mae"] < 0.02 and rd.prever(va2).dtype == np.float32


def test_ridge_so_o_treino_ajusta_e_a_validacao_so_escolhe_o_alfa(dados):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    rd = bl.RidgeBaseline.ajustar(tr, va, p, alfas=(10.0,))
    Ztr = p.transformar(tr)
    manual = Ridge(alpha=10.0).fit(Ztr[tr.M[:, 0]], tr.Y[tr.M[:, 0], 0])
    np.testing.assert_allclose(rd.modelos[0].coef_, manual.coef_)
    va_lixo = dataclasses.replace(va, Y=np.where(va.M, 99.0, np.nan).astype("float32"))
    rd2 = bl.RidgeBaseline.ajustar(tr, va_lixo, p, alfas=(10.0,))                      # validação diferente, mesmo alfa
    np.testing.assert_array_equal(rd.modelos[0].coef_, rd2.modelos[0].coef_)
    with pytest.raises(bl.BaselineError, match="train"):
        bl.RidgeBaseline.ajustar(va, va, p)
    with pytest.raises(bl.BaselineError, match="val"):
        bl.RidgeBaseline.ajustar(tr, tr, p)
    sem_rotulo = dataclasses.replace(va, M=np.zeros_like(va.M))
    with pytest.raises(bl.BaselineError, match="sem rótulo"):
        bl.RidgeBaseline.ajustar(tr, sem_rotulo, p)


# ------------------------------------------------------------------ floresta
def test_floresta_e_deterministica_com_a_mesma_semente_e_aprende_nao_linear(dados):
    c, p = dados["c"], dados["p"]
    y = np.sign(p.transformar(c)[:, 0])                                                 # degrau: o Ridge não pega
    plantado = com_y(c, np.column_stack([y, y]))
    tr2 = plantado.fatiar((plantado.meta["split"] == "train").to_numpy())
    va2 = plantado.fatiar((plantado.meta["split"] == "val").to_numpy())
    f1 = bl.Floresta.ajustar(tr2, p, seed=7, n_arvores=30)
    f2 = bl.Floresta.ajustar(tr2, p, seed=7, n_arvores=30)
    f3 = bl.Floresta.ajustar(tr2, p, seed=8, n_arvores=30)
    P1 = f1.prever(va2)
    np.testing.assert_array_equal(P1, f2.prever(va2))
    assert not np.array_equal(P1, f3.prever(va2))
    assert P1.shape == (len(va2), 2) and P1.dtype == np.float32 and np.isfinite(P1).all()
    assert mt.erros(va2.Y[:, 0], P1[:, 0], va2.M[:, 0])["mae"] < 0.2
    with pytest.raises(bl.BaselineError, match="train"):
        bl.Floresta.ajustar(va2, p, seed=7, n_arvores=5)
    sem_rotulo = dataclasses.replace(tr2, M=np.zeros_like(tr2.M))
    with pytest.raises(bl.BaselineError, match="sem rótulo"):
        bl.Floresta.ajustar(sem_rotulo, p, seed=7, n_arvores=5)


# ------------------------------------------------------------------ conjunto completo
def test_ajustar_todos_devolve_os_cinco_e_todos_se_pontuam_nas_mesmas_linhas(dados):
    tr, va, p, r = dados["tr"], dados["va"], dados["p"], dados["r"]
    todos = bl.ajustar_todos(tr, va, p, r.spi_gamma_params, seed=42, n_arvores=10)
    assert list(todos) == ["persistencia", "climatologia", "parcial_climatologica", "ridge", "floresta_aleatoria"]
    for nome, b in todos.items():
        P = b.prever(va)
        assert P.shape == (len(va), 2) and P.dtype == np.float32 and np.isfinite(P).all(), nome
        e = mt.erros(va.Y[:, 0], P[:, 0], va.M[:, 0])
        assert e["n"] == int(va.M[:, 0].sum()) and np.isfinite(e["mae"]), nome
