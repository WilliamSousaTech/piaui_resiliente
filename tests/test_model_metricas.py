"""Testes de src/model/metricas.py. Números feitos à mão; nada de arquivo nem rede."""
import math

import numpy as np
import pytest

from src.model import metricas as mt


def test_erros_batem_com_a_conta_a_mao():
    y = np.array([0.0, 1.0, -2.0, 5.0])
    p = np.array([1.0, 1.0, -1.0, 9.0])          # erros: +1, 0, +1, +4 (a 4a linha fica fora pela máscara)
    m = np.array([True, True, True, False])
    r = mt.erros(y, p, m)
    assert r["n"] == 3 and r["mae"] == pytest.approx(2 / 3) and r["rmse"] == pytest.approx(math.sqrt(2 / 3))
    assert r["vies"] == pytest.approx(2 / 3)


def test_vies_tem_sinal_previsao_menos_alvo():
    m = np.array([True, True])
    assert mt.erros(np.array([1.0, 1.0]), np.array([0.0, 0.0]), m)["vies"] == -1.0
    assert mt.erros(np.array([0.0, 0.0]), np.array([1.0, 1.0]), m)["vies"] == 1.0


def test_so_linhas_com_rotulo_entram_mesmo_com_lixo_fora_da_mascara():
    y = np.array([0.5, np.nan, 2.0])
    p = np.array([0.5, np.nan, 1.0])
    r = mt.erros(y, p, np.array([True, False, True]))
    assert r["n"] == 2 and r["mae"] == pytest.approx(0.5)


def test_previsao_nan_em_linha_com_rotulo_e_recusada_nao_descartada():
    y, m = np.array([0.0, 1.0]), np.array([True, True])
    for ruim in (np.array([0.0, np.nan]), np.array([0.0, np.inf])):
        with pytest.raises(mt.MetricasError, match="previsão"):
            mt.erros(y, ruim, m)
        with pytest.raises(mt.MetricasError, match="previsão"):
            mt.seca(y, ruim, m)
    with pytest.raises(mt.MetricasError, match="rótulo"):
        mt.erros(np.array([0.0, np.nan]), np.array([0.0, 0.0]), m)


def test_formato_e_mascara_invalidos():
    with pytest.raises(mt.MetricasError, match="mesmo tamanho"):
        mt.erros(np.zeros(3), np.zeros(2), np.ones(3, dtype=bool))
    with pytest.raises(mt.MetricasError, match="mesmo tamanho"):
        mt.erros(np.zeros((3, 1)), np.zeros((3, 1)), np.ones((3, 1), dtype=bool))
    with pytest.raises(mt.MetricasError, match="bool"):
        mt.erros(np.zeros(3), np.zeros(3), np.ones(3, dtype=int))


def test_sem_linhas_com_rotulo_devolve_nan_e_n_zero():
    z = np.zeros(2)
    r = mt.erros(z, z, np.zeros(2, dtype=bool))
    assert r["n"] == 0 and all(math.isnan(r[k]) for k in ("mae", "rmse", "vies"))
    s = mt.seca(z, z, np.zeros(2, dtype=bool))
    assert s["n"] == 0 and math.isnan(s["f1"])
    assert math.isnan(mt.quantil_erro_abs(z, z, np.zeros(2, dtype=bool), 0.8))
    assert math.isnan(mt.cobertura_da_margem(z, z, np.zeros(2, dtype=bool), 1.0))


def test_seca_contagens_precisao_revocacao_f1():
    #            real em seca? S  S  S  n  n  n  n  n
    y = np.array([-1.0, -1.5, -2.0, 0.0, 0.5, -0.9, 1.0, -1.01])
    p = np.array([-1.0, -0.5, -1.2, -1.1, 0.0, -0.9, -1.0, 0.3])    # prevê seca: 0, 2, 3, 6
    r = mt.seca(y, p, np.ones(8, dtype=bool))
    # reais (<= -1): índices 0,1,2,7. previstos (<= -1): índices 0,2,3,6.
    assert (r["tp"], r["fp"], r["fn"], r["tn"]) == (2, 2, 2, 2) and r["secas_reais"] == 4 and r["n"] == 8
    assert r["precisao"] == 0.5 and r["revocacao"] == 0.5 and r["f1"] == pytest.approx(0.5)


def test_limiar_de_seca_e_inclusivo():
    y, m = np.array([-1.0, -1.0]), np.array([True, True])
    r = mt.seca(y, np.array([-1.0, -0.99]), m)
    assert r["tp"] == 1 and r["fn"] == 1                                  # -1,00 já é seca, -0,99 não
    assert mt.LIMIAR_SECA == -1.0


def test_seca_sem_positivos_ou_sem_previsoes_nao_divide_por_zero():
    m = np.ones(3, dtype=bool)
    nada = mt.seca(np.array([0.0, 0.5, 1.0]), np.array([0.0, 0.5, 1.0]), m)       # nunca há seca
    assert math.isnan(nada["precisao"]) and math.isnan(nada["revocacao"]) and math.isnan(nada["f1"])
    nunca_preve = mt.seca(np.array([-2.0, -1.5, 0.0]), np.zeros(3), m)
    assert math.isnan(nunca_preve["precisao"]) and nunca_preve["revocacao"] == 0.0 and math.isnan(nunca_preve["f1"])
    cega = mt.seca(np.array([-2.0, 0.0]), np.array([0.0, -2.0]), np.ones(2, dtype=bool))
    assert cega["precisao"] == 0.0 and cega["revocacao"] == 0.0
    assert math.isnan(cega["f1"])                                                  # f1 = 0/0: NaN, nunca exceção


def test_quantil_do_erro_absoluto_e_cobertura_da_margem():
    y = np.zeros(5)
    p = np.array([0.1, -0.2, 0.3, -0.4, 5.0])
    m = np.array([True, True, True, True, False])                          # o 5,0 fica fora da máscara
    q = mt.quantil_erro_abs(y, p, m, 0.5)
    assert q == pytest.approx(0.25)                                         # |erros| = .1 .2 .3 .4 -> mediana .25
    assert mt.cobertura_da_margem(y, p, m, q) == 0.5
    assert mt.cobertura_da_margem(y, p, m, 0.4) == 1.0
    for ruim in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(mt.MetricasError, match="entre 0 e 1"):
            mt.quantil_erro_abs(y, p, m, ruim)


def test_tipos_float32_dao_o_mesmo_resultado_que_float64():
    rng = np.random.default_rng(1)
    y = rng.normal(size=200).astype("float32")
    p = (y + rng.normal(scale=0.3, size=200)).astype("float32")
    m = rng.random(200) < 0.8
    a = mt.erros(y, p, m)
    b = mt.erros(y.astype("float64"), p.astype("float64"), m)
    assert a["mae"] == pytest.approx(b["mae"]) and a["n"] == b["n"] == int(m.sum())


# ------------------------------------------------------------------ bootstrap por grupos (meses)
def _cenario(n_meses=40, n_mun=30, seed=0, ganho=0.0):
    rng = np.random.default_rng(seed)
    mes = np.repeat(np.arange(n_meses), n_mun)
    comum = np.repeat(rng.normal(size=n_meses), n_mun)                  # erro regional, igual para todos no mês
    y = rng.normal(size=n_meses * n_mun)
    pb = y + comum + rng.normal(scale=0.3, size=y.size)
    pa = y + comum * (1 - ganho) + rng.normal(scale=0.3, size=y.size)
    return y, pa, pb, np.ones(y.size, dtype=bool), mes


def test_bootstrap_detecta_ganho_real_e_nao_inventa_ganho_onde_nao_ha():
    y, pa, pb, m, g = _cenario(ganho=0.8)
    r = mt.ic_bootstrap_dif_mae(y, pa, pb, m, g, seed=1)
    assert r["dif"] < 0 and r["ic_sup"] < 0 and r["n_grupos"] == 40 and r["n"] == 1200      # A claramente melhor
    y, pa, pb, m, g = _cenario(ganho=0.0, seed=3)
    r0 = mt.ic_bootstrap_dif_mae(y, pa, pb, m, g, seed=1)
    assert r0["ic_inf"] < 0 < r0["ic_sup"]                                                   # empate: o intervalo contém 0
    inv = mt.ic_bootstrap_dif_mae(y, pb, pa, m, g, seed=1)
    assert inv["dif"] == pytest.approx(-r0["dif"])                                           # trocar A e B inverte o sinal


def test_bootstrap_por_mes_e_mais_largo_que_por_linha_quando_o_erro_e_regional():
    y, pa, pb, m, g = _cenario(ganho=0.3, seed=5)
    por_mes = mt.ic_bootstrap_dif_mae(y, pa, pb, m, g, seed=2)
    por_linha = mt.ic_bootstrap_dif_mae(y, pa, pb, m, np.arange(y.size), seed=2)             # cada linha seu próprio grupo
    assert (por_mes["ic_sup"] - por_mes["ic_inf"]) > 2 * (por_linha["ic_sup"] - por_linha["ic_inf"])


def test_bootstrap_e_reprodutivel_e_respeita_a_mascara():
    y, pa, pb, m, g = _cenario(ganho=0.5)
    a, b = mt.ic_bootstrap_dif_mae(y, pa, pb, m, g, seed=9), mt.ic_bootstrap_dif_mae(y, pa, pb, m, g, seed=9)
    assert a == b and a != mt.ic_bootstrap_dif_mae(y, pa, pb, m, g, seed=10)
    m2 = m.copy()
    m2[::2] = False
    pa2, pb2 = pa.copy(), pb.copy()
    pa2[::2] = np.nan                                                                        # lixo fora da máscara
    pb2[::2] = np.nan
    r = mt.ic_bootstrap_dif_mae(y, pa2, pb2, m2, g, seed=9)
    assert r["n"] == int(m2.sum()) and np.isfinite(r["dif"])


def test_bootstrap_com_um_grupo_ou_sem_linhas_nao_calcula_intervalo_e_entradas_ruins_sao_recusadas():
    y = np.array([0.0, 1.0, 2.0])
    p = np.array([0.5, 1.5, 2.5])
    um = mt.ic_bootstrap_dif_mae(y, p, p, np.ones(3, dtype=bool), np.zeros(3), seed=1)
    assert um["n_grupos"] == 1 and math.isnan(um["ic_inf"]) and um["dif"] == 0.0
    vazio = mt.ic_bootstrap_dif_mae(y, p, p, np.zeros(3, dtype=bool), np.zeros(3), seed=1)
    assert vazio["n"] == 0 and math.isnan(vazio["dif"])
    ok = np.ones(3, dtype=bool)
    with pytest.raises(mt.MetricasError, match="nivel"):
        mt.ic_bootstrap_dif_mae(y, p, p, ok, np.arange(3), seed=1, nivel=1.0)
    with pytest.raises(mt.MetricasError, match="n_boot"):
        mt.ic_bootstrap_dif_mae(y, p, p, ok, np.arange(3), seed=1, n_boot=10)
    with pytest.raises(mt.MetricasError, match="grupo"):
        mt.ic_bootstrap_dif_mae(y, p, p, ok, np.arange(2), seed=1)
    with pytest.raises(mt.MetricasError, match="previsão"):
        mt.ic_bootstrap_dif_mae(y, np.array([0.0, np.nan, 1.0]), p, ok, np.arange(3), seed=1)
