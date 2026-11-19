"""Correção sobre o baseline parcial, contexto regional e conjunto de redes (Item 9). Só dados sintéticos."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from dados_sinteticos import PARAMS_REAL, raw_sintetico
from src.data import build_demo as bd
from src.data import quality
from src.data.config import load_config
from src.model import baselines as bl
from src.model import features as fe
from src.model import metricas as mt
from src.model import residuo as rs
from src.model import treino as tn


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def mundo(cfg):
    raw = raw_sintetico()
    r = bd.montar_demo(raw, cfg)
    q = quality.avaliar_qualidade(raw, cfg)
    c = fe.construir(r.demo_observado, q.serie_imputada, r.climatologia_precip, r.qualidade_municipio, cfg)
    return {"c": c, "base": bl.ParcialClimatologica(r.spi_gamma_params), "r": r}


# ------------------------------------------------------------------ contexto regional
def test_regional_bate_com_conta_independente_e_nao_mexe_no_original(mundo):
    c = mundo["c"]
    X0 = c.X.copy()
    g = rs.adicionar_regional(c)
    assert np.array_equal(c.X, X0) and len(c.colunas) == 41                              # original intacto
    novas = [n for n in g.colunas if n.startswith("reg_")]
    assert novas == ["reg_chuva_lag0", "reg_chuva_lag1", "reg_chuva_lag2", "reg_spi3_lag0", "reg_spi6_lag0",
                     "reg_spi3_lag0_seca", "reg_spi6_lag0_seca"]
    assert g.colunas[:41] == c.colunas and g.X.shape == (len(c), 48) and g.continuas[41:].all()
    assert np.array_equal(g.Y, c.Y, equal_nan=True) and np.array_equal(g.M, c.M)         # rótulos e máscara intocados
    # conta independente, mês de origem por mês de origem
    for mes in pd.to_datetime(c.meta["mes_origem"]).drop_duplicates().iloc[[0, 50, 300]]:
        linhas = (c.meta["mes_origem"] == mes).to_numpy()
        esperado = np.nanmean(c.X[linhas][:, c.colunas.index("chuva_lag1")].astype("float64"))
        assert g.X[linhas][:, g.colunas.index("reg_chuva_lag1")] == pytest.approx(esperado, rel=1e-5)
        spi = c.X[linhas][:, c.colunas.index("spi3_lag0")].astype("float64")
        ok = np.isfinite(spi)
        if ok.any():
            assert g.X[linhas][:, g.colunas.index("reg_spi3_lag0_seca")] == pytest.approx((spi[ok] <= -1.0).mean(), abs=1e-6)


def test_regional_so_olha_a_mesma_origem_mudar_o_futuro_nao_muda_o_passado(mundo):
    c = mundo["c"]
    corte = pd.Timestamp("2005-06-01")
    base = rs.adicionar_regional(c)
    X = c.X.copy()
    futuro = (c.meta["mes_origem"] > corte).to_numpy()
    X[futuro] = np.random.default_rng(0).normal(size=X[futuro].shape).astype("float32")      # destrói tudo depois do corte
    mudado = rs.adicionar_regional(dataclasses.replace(c, X=X))
    passado = ~futuro
    assert np.array_equal(base.X[passado], mudado.X[passado], equal_nan=True)
    assert not np.array_equal(base.X[futuro], mudado.X[futuro], equal_nan=True)


def test_regional_ausente_fica_de_fora_da_media_e_origem_sem_valor_vira_nan(mundo):
    c = mundo["c"]
    X = c.X.copy()
    j = c.colunas.index("spi3_lag0")
    mes = pd.to_datetime(c.meta["mes_origem"]).drop_duplicates().iloc[100]
    linhas = np.flatnonzero((c.meta["mes_origem"] == mes).to_numpy())
    X[linhas[0], j] = np.nan                                                                # um município sem SPI
    X[linhas[1], j] = -1.0                                                                  # exatamente no limiar: conta como seca
    X[linhas[2], j] = -0.99
    g = rs.adicionar_regional(dataclasses.replace(c, X=X))
    resto = X[linhas, j].astype("float64")[1:]
    assert g.X[linhas[2], g.colunas.index("reg_spi3_lag0_seca")] == pytest.approx((resto <= -1.0).mean(), abs=1e-6)
    valor = g.X[linhas[2], g.colunas.index("reg_spi3_lag0")]
    assert valor == pytest.approx(np.nanmean(X[linhas, j].astype("float64")), rel=1e-5) and np.isfinite(valor)
    X[linhas, j] = np.nan                                                                   # todos sem SPI naquela origem
    g2 = rs.adicionar_regional(dataclasses.replace(c, X=X))
    assert np.isnan(g2.X[linhas, g2.colunas.index("reg_spi3_lag0")]).all()
    assert np.isnan(g2.X[linhas, g2.colunas.index("reg_spi3_lag0_seca")]).all()


def test_regional_recusa_duas_vezes_e_coluna_ausente(mundo):
    g = rs.adicionar_regional(mundo["c"])
    with pytest.raises(rs.ResiduoError, match="já foi acrescentado"):
        rs.adicionar_regional(g)
    sem = dataclasses.replace(mundo["c"], colunas=tuple("x" + n for n in mundo["c"].colunas))
    with pytest.raises(rs.ResiduoError, match="ausentes"):
        rs.adicionar_regional(sem)


# ------------------------------------------------------------------ correção sobre o baseline
def test_com_base_subtrai_so_onde_ha_rotulo_e_acrescenta_a_entrada(mundo, cfg):
    va = fe.selecionar(mundo["c"], "val", cfg, so_com_rotulo=True)
    b = mundo["base"].prever(va)
    r = rs.com_base(va, b)
    assert r.colunas[-2:] == ("base_alvo_spi3_h1", "base_alvo_spi6_h1") and r.X.shape == (len(va), 43)
    assert np.array_equal(r.X[:, :41], va.X, equal_nan=True) and np.array_equal(r.X[:, 41:], b)
    assert np.allclose(r.Y[va.M], va.Y[va.M] - b[va.M], atol=1e-6) and np.array_equal(r.M, va.M)
    M = va.M.copy()
    M[::3, 0] = False
    Y = np.where(M, va.Y, np.nan).astype("float32")
    r2 = rs.com_base(dataclasses.replace(va, Y=Y, M=M), b)
    assert np.isnan(r2.Y[~M]).all() and np.isfinite(r2.Y[M]).all()                        # sem rótulo continua NaN, nunca 0
    assert r.continuas[-2:].all() and len(r.continuas) == 43
    assert len(va.colunas) == 41 and va.X.shape[1] == 41                                   # o original não mudou


def test_com_base_recusa_formato_errado_nan_e_repeticao(mundo, cfg):
    va = fe.selecionar(mundo["c"], "val", cfg, so_com_rotulo=True)
    b = mundo["base"].prever(va)
    with pytest.raises(rs.ResiduoError, match="shape"):
        rs.com_base(va, b[:, :1])
    ruim = b.copy()
    ruim[0, 0] = np.nan
    with pytest.raises(rs.ResiduoError, match="finita"):
        rs.com_base(va, ruim)
    with pytest.raises(rs.ResiduoError, match="já foi acrescentado"):
        rs.com_base(rs.com_base(va, b), b)


class Fixo:
    """Preditor falso: devolve uma constante e guarda o conjunto que recebeu."""
    def __init__(self, v):
        self.v, self.visto = v, None

    def prever(self, c):
        self.visto = c
        return np.full((len(c), len(c.alvos)), self.v, dtype="float32")


def test_previsao_e_baseline_mais_correcao(mundo, cfg):
    va = fe.selecionar(mundo["c"], "val", cfg, so_com_rotulo=True)
    fixo = Fixo(0.25)
    p = rs.ResiduoPreditor(mundo["base"], fixo).prever(va)
    b = mundo["base"].prever(va)
    assert p.dtype == np.float32 and np.allclose(p, b + 0.25, atol=1e-6)
    assert fixo.visto.colunas[-2:] == ("base_alvo_spi3_h1", "base_alvo_spi6_h1")              # a rede viu o baseline como entrada


def test_rede_de_correcao_aprende_o_que_falta_e_com_correcao_nula_iguala_o_baseline(mundo, cfg):
    c = mundo["c"]
    tr, va = fe.selecionar(c, "train", cfg, so_com_rotulo=True), fe.selecionar(c, "val", cfg, so_com_rotulo=True)
    # rótulo plantado = baseline + sinal que depende de uma feature (a rede só precisa aprender o sinal)
    def plantar(d):
        b = mundo["base"].prever(d)
        sinal = 0.8 * np.tanh(d.X[:, d.colunas.index("chuva_lag0")].astype("float64") - 3.2)[:, None]
        Y = np.where(d.M, b + sinal, np.nan).astype("float32")
        return dataclasses.replace(d, Y=Y)
    tr, va = plantar(tr), plantar(va)
    btr, bva = mundo["base"].prever(tr), mundo["base"].prever(va)
    tr_r, va_r = rs.com_base(tr, btr), rs.com_base(va, bva)
    pad = fe.Padronizador.ajustar(tr_r)
    hp = tn.Hiperparametros(ocultas=(16,), dropout=0.0, lr=1e-2, lote=128, epocas_max=60, paciencia=60)
    res = tn.treinar(tr_r, va_r, pad, seed=1, hp=hp)
    pred = rs.ResiduoPreditor(mundo["base"], tn.RedePreditor(res.rede, pad)).prever(va)
    com, sem = mt.erros(va.Y[:, 0], pred[:, 0], va.M[:, 0]), mt.erros(va.Y[:, 0], bva[:, 0], va.M[:, 0])
    assert com["mae"] < 0.5 * sem["mae"]                                                      # aprendeu o sinal plantado
    nula = rs.ResiduoPreditor(mundo["base"], Fixo(0.0)).prever(va)
    assert np.array_equal(nula, bva)


# ------------------------------------------------------------------ conjunto de redes
def test_media_de_preditores_e_desvio(mundo, cfg):
    va = fe.selecionar(mundo["c"], "val", cfg, so_com_rotulo=True)
    m = rs.MediaDePreditores([Fixo(1.0), Fixo(3.0), Fixo(8.0)])
    assert np.allclose(m.prever(va), 4.0) and np.allclose(m.desvio(va), np.std([1.0, 3.0, 8.0]))   # média, não mediana
    assert m.prever(va).dtype == np.float32 and m.desvio(va).shape == (len(va), 2)
    um = rs.MediaDePreditores([Fixo(2.0)])
    assert np.allclose(um.prever(va), 2.0) and np.allclose(um.desvio(va), 0.0)
    with pytest.raises(rs.ResiduoError, match="pelo menos um"):
        rs.MediaDePreditores([])
