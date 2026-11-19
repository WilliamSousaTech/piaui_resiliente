"""Previsão do próximo mês (src/model/prever.py). Só dados sintéticos e um preditor de mentira; nada de rede."""
import copy
import dataclasses
import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from dados_sinteticos import A, B, C, PARAMS_REAL, raw_sintetico
from src.data import build_demo as bd
from src.data import manifest as mf
from src.data import quality
from src.data import spi as spimod
from src.data.config import load_config
from src.model import busca as bu
from src.model import entrada as en
from src.model import features as fe
from src.model import prever as pv
from src.model import residuo as rs
from test_build_site import montar_proj


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def mundo(cfg):
    raw = raw_sintetico()
    r = bd.montar_demo(raw, cfg)
    q = quality.avaliar_qualidade(raw, cfg)
    c = fe.construir(r.demo_observado, q.serie_imputada, r.climatologia_precip, r.qualidade_municipio, cfg)
    return {"c": c, "params": r.spi_gamma_params}


class Falso:
    """Preditor de mentira: devolve valores fixos por linha e anota o que viu."""

    def __init__(self, P, D=0.05):
        self.P, self.D, self.vistos = np.asarray(P, dtype="float32"), D, []

    def prever(self, c):
        self.vistos.append(c)
        assert len(c) == len(self.P), (len(c), len(self.P))
        return self.P

    def desvio(self, c):
        return np.full(self.P.shape, self.D, dtype="float32")


def _cabeca_val(mae, n=6048):
    return {"todas": {"n": n, "mae": mae, "classes": {"acerto": 0.81, "acerto_sempre_normal": 0.70}},
            "por_mes_calendario": {str(m): 0.1 * m for m in range(1, 13)}}


def receita_e_validacao(sha="ab" * 32):
    alvos = ["alvo_spi3_h1", "alvo_spi6_h1"]
    receita = {"dataset_sha256": sha, "alvos": alvos, "limiares_alerta": {alvos[0]: -1.0, alvos[1]: -0.95},
               "n_membros": 5, "config": {"residuo": True, "regional": False}, "colunas": []}
    val = {"dataset_sha256": sha, "principal": "rede_final", "corte_b": "2018-09",
           "modelos": {"rede_final": {a: _cabeca_val(0.40 + 0.1 * i) for i, a in enumerate(alvos)},
                       "persistencia": {a: {"todas": {"mae": 0.64 + 0.1 * i}, "por_mes_calendario": {str(m): 0.2 * m for m in range(1, 13)}}
                                        for i, a in enumerate(alvos)},
                       "parcial_climatologica": {a: {"todas": {"mae": 0.41 + 0.1 * i}} for i, a in enumerate(alvos)}},
           "alerta_seca": {a: {"rede_final": {"limiar_previsao": lim, "secas_reais": 200, "tp": 50, "fp": 40,
                                              "revocacao": 0.254980079, "precisao": 0.55}}
                           for a, lim in zip(alvos, (-1.0, -0.95))},
           "comparacao": {a: {"persistencia": {"veredito": vp}, "melhor_baseline": {"contra": b, "veredito": v}}
                          for a, vp, b, v in zip(alvos, ("superou", "indistinguivel"), ("parcial_climatologica", "ridge"),
                                                 ("indistinguivel", "superou"))},
           "margem": {a: {"q": 0.8, "valor": v, "cobertura_validacao": 0.80, "cobertura_2a_metade": 0.76, "estavel": est}
                      for a, v, est in zip(alvos, (0.6624, 0.5486), (True, False))}}
    return receita, val


def chamar(mundo, cfg, P, *, receita=None, val=None, hoje=date(2026, 11, 19), c=None):
    r0, v0 = receita_e_validacao()
    falso = Falso(P)
    doc = pv.prever_proximo_mes(c if c is not None else mundo["c"], falso, params_gama=mundo["params"], cfg=cfg,
                                receita=receita or r0, validacao=val or v0, gerado_em=hoje)
    return doc, falso


def ultima(c):
    return (c.meta["mes_origem"] == c.meta["mes_origem"].max()).to_numpy()


def ordem_dos_cods(c):
    return c.fatiar(ultima(c)).meta["cod_ibge"].astype(str).tolist()


# ------------------------------------------------------------------ documento
def test_documento_traz_origem_alvo_e_so_a_ultima_origem(mundo, cfg):
    c = mundo["c"]
    n = int(ultima(c).sum())
    assert n == 3
    doc, falso = chamar(mundo, cfg, np.zeros((n, 2)))
    assert (doc["origem"], doc["alvo"], doc["horizonte_meses"]) == ("2026-09", "2026-10", 1)
    assert len(falso.vistos) == 1 and set(falso.vistos[0].meta["mes_origem"]) == {pd.Timestamp("2026-09-01")}
    assert [m["cod_ibge"] for m in doc["municipios"]] == [A, B, C]            # ordenado por código, não pela ordem de entrada
    assert doc["versao"] == pv.VERSAO and doc["aviso"] == pv.AVISO and "MVP" in doc["aviso"]
    assert doc["gerado_em"] == "2026-11-19" and doc["alvo_ja_passou"] is True and doc["corte_b"] == "2018-09"
    assert doc["teste_congelado_lido"] is False


def test_alvo_ja_passou_depende_da_data(mundo, cfg):
    n = int(ultima(mundo["c"]).sum())
    antes, _ = chamar(mundo, cfg, np.zeros((n, 2)), hoje=date(2026, 10, 31))
    depois, _ = chamar(mundo, cfg, np.zeros((n, 2)), hoje=date(2026, 11, 1))
    assert antes["alvo_ja_passou"] is False and depois["alvo_ja_passou"] is True


def test_alerta_vale_para_o_numero_arredondado_e_inclui_o_limiar(mundo, cfg):
    cods = ordem_dos_cods(mundo["c"])
    P = np.zeros((3, 2))
    P[cods.index(A)] = (-0.996, -0.9449)      # -1,00 (alerta, no limiar) e -0,94 (não alerta: limiar -0,95)
    P[cods.index(B)] = (-1.004, -0.95)        # -1,00 e -0,95: os dois exatamente no limiar
    P[cods.index(C)] = (-0.99, -0.001)        # -0,99 não alerta; -0,001 vira 0,0 (nunca -0,0)
    doc, _ = chamar(mundo, cfg, P)
    m = {x["cod_ibge"]: x for x in doc["municipios"]}
    assert (m[A]["spi3"]["previsto"], m[A]["spi3"]["alerta"], m[A]["spi6"]["alerta"]) == (-1.0, True, False)
    assert (m[B]["spi3"]["alerta"], m[B]["spi6"]["alerta"]) == (True, True)
    assert (m[C]["spi3"]["previsto"], m[C]["spi3"]["alerta"]) == (-0.99, False)
    assert m[C]["spi6"]["previsto"] == 0.0 and str(m[C]["spi6"]["previsto"]) == "0.0"
    assert doc["resumo"]["n_alerta_spi3"] == 2 and doc["resumo"]["n_alerta_spi6"] == 1 and doc["resumo"]["n_alerta_algum"] == 2
    assert doc["resumo"]["n_municipios"] == 3


def test_cabecas_trazem_a_validacao_do_mesmo_limiar_e_a_margem(mundo, cfg):
    doc, _ = chamar(mundo, cfg, np.zeros((3, 2)))
    h3, h6 = doc["cabecas"]
    assert (h3["id"], h3["escala"], h3["limiar_alerta"], h6["id"], h6["escala"], h6["limiar_alerta"]) == ("spi3", 3, -1.0, "spi6", 6, -0.95)
    assert (h3["margem"], h6["margem"]) == (0.6624, 0.5486)
    assert (h3["margem_estavel"], h6["margem_estavel"]) == (True, False)
    assert (h3["margem_q"], h3["margem_cobertura_validacao"], h3["margem_cobertura_2a_metade"]) == (0.8, 0.8, 0.76)
    assert (h3["mae"], h3["mae_persistencia"], h3["mae_parcial_climatologica"]) == (0.4, 0.64, 0.41)
    assert (h3["mae_mes_alvo"], h3["mae_mes_alvo_persistencia"]) == (1.0, 2.0)  # mês-calendário 10 do alvo
    assert (h3["secas_reais"], h3["secas_pegas"], h3["alarmes_falsos"], h3["revocacao"], h3["precisao"]) == (200, 50, 40, 0.25498, 0.55)   # sem arredondar cedo
    assert (h3["acerto_classe"], h3["acerto_sempre_normal"]) == (0.81, 0.7)
    assert (h3["veredito_persistencia"], h3["melhor_baseline"], h3["veredito_melhor_baseline"]) == ("superou", "parcial_climatologica", "indistinguivel")
    assert (h6["veredito_persistencia"], h6["melhor_baseline"], h6["veredito_melhor_baseline"]) == ("indistinguivel", "ridge", "superou")


def test_baixa_confianca_do_alvo_vem_dos_parametros_e_nao_da_coluna_do_conjunto(mundo, cfg):
    c = mundo["c"]
    assert not c.meta.loc[ultima(c), ["alvo_spi3_baixa_conf", "alvo_spi6_baixa_conf"]].to_numpy().any()   # a coluna vem toda falsa
    p = mundo["params"].copy()
    alvo_cel = (p["cod_ibge"] == B) & (p["mes_calendario"] == 10) & (p["escala_meses"] == 3)
    p.loc[alvo_cel, "q"] = 0.9                                                                          # muitos zeros em outubro
    doc = pv.prever_proximo_mes(c, Falso(np.zeros((3, 2))), params_gama=p, cfg=cfg, gerado_em=date(2026, 11, 19),
                                **dict(zip(("receita", "validacao"), receita_e_validacao())))
    m = {x["cod_ibge"]: x for x in doc["municipios"]}
    assert m[B]["spi3"]["baixa_conf"] is True and m[A]["spi3"]["baixa_conf"] is False
    assert doc["resumo"]["n_baixa_conf_spi3"] == sum(x["spi3"]["baixa_conf"] for x in doc["municipios"]) == 1
    assert doc["resumo"]["n_baixa_conf_spi6"] == sum(x["spi6"]["baixa_conf"] for x in doc["municipios"])


def test_referencia_do_mes_bate_com_a_regra_do_spi(mundo, cfg):
    p = mundo["params"].copy()
    k = 3
    p.loc[(p["cod_ibge"] == A) & (p["mes_calendario"] == 10) & (p["escala_meses"] == k), "q"] = 0.9           # q alto
    p.loc[(p["cod_ibge"] == B) & (p["mes_calendario"] == 10) & (p["escala_meses"] == k), "scale"] = 0.01      # média baixa
    p.loc[(p["cod_ibge"] == C) & (p["mes_calendario"] == 10) & (p["escala_meses"] == k), "valido"] = False   # sem parâmetro
    meses = pd.date_range("2025-10-01", periods=1, freq="MS")
    S = pd.DataFrame(50.0, index=meses, columns=[A, B, C])
    _, inval_ref, baixa_ref = spimod._aplicar(S, p[p["escala_meses"] == k], cfg)
    inval, baixa = pv.referencia_do_mes(p, cfg, [A, B, C], (k,), 10)
    assert inval[:, 0].tolist() == inval_ref[0].tolist() == [False, False, True]
    assert baixa[:, 0].tolist() == baixa_ref[0].tolist() == [True, True, False]
    # e também no conjunto inteiro de parâmetros intactos, em todos os meses
    q = mundo["params"]
    for mes in range(1, 13):
        S = pd.DataFrame(50.0, index=pd.date_range(f"2025-{mes:02d}-01", periods=1, freq="MS"), columns=[A, B, C])
        for kk in (3, 6):
            _, i_ref, b_ref = spimod._aplicar(S, q[q["escala_meses"] == kk], cfg)
            i, b = pv.referencia_do_mes(q, cfg, [A, B, C], (kk,), mes)
            assert i[:, 0].tolist() == i_ref[0].tolist() and b[:, 0].tolist() == b_ref[0].tolist(), (mes, kk)


def test_municipio_sem_parametro_do_mes_derruba_a_previsao(mundo, cfg):
    p = mundo["params"].copy()
    p.loc[(p["cod_ibge"] == C) & (p["mes_calendario"] == 10) & (p["escala_meses"] == 6), "valido"] = False
    r, v = receita_e_validacao()
    with pytest.raises(pv.PreverError, match="sem parâmetro Gama válido"):
        pv.prever_proximo_mes(mundo["c"], Falso(np.zeros((3, 2))), params_gama=p, cfg=cfg, receita=r, validacao=v,
                              gerado_em=date(2026, 11, 19))


def test_params_repetidos_ou_sem_coluna_sao_recusados(mundo, cfg):
    p = mundo["params"]
    with pytest.raises(pv.PreverError, match="repetida"):
        pv.referencia_do_mes(pd.concat([p, p.iloc[:1]]), cfg, [A], (3,), 10)
    with pytest.raises(pv.PreverError, match="sem as colunas"):
        pv.referencia_do_mes(p.drop(columns=["scale"]), cfg, [A], (3,), 10)


# ------------------------------------------------------------------ recusas (anti-vazamento e honestidade)

def test_ordem_do_documento_nao_depende_da_ordem_das_linhas(mundo, cfg):
    c = mundo["c"]
    linhas = np.where(ultima(c))[0]
    perm = np.concatenate([np.setdiff1d(np.arange(len(c)), linhas), linhas[::-1]])           # última origem em ordem invertida
    c2 = dataclasses.replace(c, X=c.X[perm], Y=c.Y[perm], M=c.M[perm], meta=c.meta.iloc[perm].reset_index(drop=True))
    cods = ordem_dos_cods(c2)
    assert cods == [C, B, A]
    P = np.zeros((3, 2))
    P[cods.index(A)] = (-1.5, 0.0)
    doc, _ = chamar(mundo, cfg, P, c=c2)
    assert [m["cod_ibge"] for m in doc["municipios"]] == [A, B, C]
    assert [m["spi3"]["previsto"] for m in doc["municipios"]] == [-1.5, 0.0, 0.0]            # cada valor ficou com o seu município


def test_so_aceita_horizonte_um(mundo, cfg):
    c = mundo["c"]
    meta = c.meta.copy()
    meta["mes_alvo"] = meta["mes_origem"] + pd.offsets.MonthBegin(2)
    c2 = dataclasses.replace(c, meta=meta, alvos=("alvo_spi3_h2", "alvo_spi6_h2"))
    r, v = receita_e_validacao()
    r["alvos"] = list(c2.alvos)
    with pytest.raises(pv.PreverError, match="h = 1"):
        chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v, c=c2)


def test_recusa_mes_alvo_que_nao_e_origem_mais_horizonte(mundo, cfg):
    c = mundo["c"]
    meta = c.meta.copy()
    meta.loc[ultima(c), "mes_alvo"] = pd.Timestamp("2027-03-01")
    with pytest.raises(pv.PreverError, match="mês-alvo"):
        chamar(mundo, cfg, np.zeros((3, 2)), c=dataclasses.replace(c, meta=meta))


def test_recusa_municipio_repetido_na_ultima_origem(mundo, cfg):
    c = mundo["c"]
    meta = c.meta.copy()
    idx = np.where(ultima(c))[0]
    meta.loc[idx[1], "cod_ibge"] = meta.loc[idx[0], "cod_ibge"]
    with pytest.raises(pv.PreverError, match="repetido"):
        chamar(mundo, cfg, np.zeros((3, 2)), c=dataclasses.replace(c, meta=meta))


def test_recusa_desvio_com_nan_ou_shape_errado(mundo, cfg):
    r, v = receita_e_validacao()

    class DesvioRuim(Falso):
        def __init__(self, D):
            super().__init__(np.zeros((3, 2)))
            self.D_ruim = D

        def desvio(self, c):
            return self.D_ruim
    for D, msg in ((np.full((3, 2), np.nan), "NaN ou infinito"), (np.zeros((3, 3)), "shapes")):
        with pytest.raises(pv.PreverError, match=msg):
            pv.prever_proximo_mes(mundo["c"], DesvioRuim(D), params_gama=mundo["params"], cfg=cfg, receita=r, validacao=v,
                                  gerado_em=date(2026, 11, 19))


@pytest.mark.parametrize("coluna", ["q", "shape", "scale"])
def test_parametro_nao_finito_vira_invalido(mundo, cfg, coluna):
    p = mundo["params"].copy()
    cel = (p["cod_ibge"] == A) & (p["mes_calendario"] == 10) & (p["escala_meses"] == 3)
    p.loc[cel, coluna] = np.nan
    inval, baixa = pv.referencia_do_mes(p, cfg, [A, B], (3,), 10)
    assert inval[:, 0].tolist() == [True, False]
    assert not baixa[0, 0]                                                       # sem parâmetro não há SPI, logo não há "baixa confiança"


def test_limite_da_media_por_mes_usa_a_media_dividida_por_k(mundo, cfg):
    p = mundo["params"].copy()
    # (q, média por mês da janela depois de (1 - q) e de dividir por k) -> esperado "baixa confiança"
    casos = {A: (0.0, 9.9, True), B: (0.0, 10.1, False), C: (0.4, 7.2, True)}       # C: 7,2 mm com o fator (1 - q); sem ele seriam 12 mm
    for k in (3, 6):
        for cod, (q, media, _) in casos.items():
            cel = (p["cod_ibge"] == cod) & (p["mes_calendario"] == 10) & (p["escala_meses"] == k)
            p.loc[cel, ["q", "shape", "scale", "valido"]] = (q, 2.0, media * k / (1.0 - q) / 2.0, True)
    for k in (3, 6):
        _, baixa = pv.referencia_do_mes(p, cfg, [A, B, C], (k,), 10)
        S = pd.DataFrame(50.0, index=pd.date_range("2025-10-01", periods=1, freq="MS"), columns=[A, B, C])
        _, _, ref = spimod._aplicar(S, p[p["escala_meses"] == k], cfg)
        assert baixa[:, 0].tolist() == ref[0].tolist() == [True, False, True]


def test_media_exatamente_no_minimo_nao_e_baixa_confianca(mundo, cfg):
    p = mundo["params"].copy()
    cel = (p["cod_ibge"] == A) & (p["mes_calendario"] == 10) & (p["escala_meses"] == 3)
    p.loc[cel, ["q", "shape", "scale", "valido"]] = (0.0, 2.0, 15.0, True)           # 2 * 15 / 3 = 10,0 mm por mês
    _, baixa = pv.referencia_do_mes(p, cfg, [A], (3,), 10)
    S = pd.DataFrame(50.0, index=pd.date_range("2025-10-01", periods=1, freq="MS"), columns=[A, B, C])
    _, _, ref = spimod._aplicar(S, p[p["escala_meses"] == 3], cfg)
    assert cfg.spi.media_min_mm_por_mes == 10.0 and baixa[0, 0] == ref[0][0] == False   # noqa: E712


def test_parametro_invalido_nunca_e_baixa_confianca_mesmo_com_poucos_mm(mundo, cfg):
    p = mundo["params"].copy()
    cel = (p["cod_ibge"] == A) & (p["mes_calendario"] == 10) & (p["escala_meses"] == 3)
    p.loc[cel, ["q", "valido"]] = (0.9, False)                                       # q alto, mas sem parâmetro válido
    inval, baixa = pv.referencia_do_mes(p, cfg, [A], (3,), 10)
    assert inval[0, 0] and not baixa[0, 0]


def test_recusa_se_a_ultima_origem_nao_e_pos_teste(mundo, cfg):
    c = mundo["c"]
    meta = c.meta.copy()
    meta["split"] = meta["split"].astype(str)
    meta.loc[ultima(c), "split"] = "test"                               # como se o teste congelado fosse a última origem
    with pytest.raises(pv.PreverError, match="teste congelado"):
        chamar(mundo, cfg, np.zeros((3, 2)), c=dataclasses.replace(c, meta=meta))


def test_recusa_conjunto_vazio(mundo, cfg):
    c = mundo["c"]
    with pytest.raises(pv.PreverError, match="vazio"):
        chamar(mundo, cfg, np.zeros((0, 2)), c=c.fatiar(np.zeros(len(c), dtype=bool)))


def test_recusa_se_o_mes_alvo_ja_tem_rotulo(mundo, cfg):
    c = mundo["c"]
    M = c.M.copy()
    M[ultima(c)] = True
    with pytest.raises(pv.PreverError, match="já tem rótulo"):
        chamar(mundo, cfg, np.zeros((3, 2)), c=dataclasses.replace(c, M=M))


@pytest.mark.parametrize("ruim", [np.nan, np.inf, -np.inf])
def test_nao_publica_previsao_com_nan_ou_infinito(mundo, cfg, ruim):
    P = np.zeros((3, 2))
    P[1, 0] = ruim
    with pytest.raises(pv.PreverError, match="NaN ou infinito"):
        chamar(mundo, cfg, P)


def test_recusa_shape_errado_do_preditor(mundo, cfg):
    class Torto(Falso):
        def prever(self, c):
            return np.zeros((len(c), 3), dtype="float32")
    r, v = receita_e_validacao()
    with pytest.raises(pv.PreverError, match="shapes"):
        pv.prever_proximo_mes(mundo["c"], Torto(np.zeros((3, 2))), params_gama=mundo["params"], cfg=cfg, receita=r, validacao=v,
                              gerado_em=date(2026, 11, 19))


def test_recusa_cabecas_diferentes_das_da_receita(mundo, cfg):
    r, v = receita_e_validacao()
    r["alvos"] = ["alvo_spi6_h1", "alvo_spi3_h1"]
    with pytest.raises(pv.PreverError, match="cabeças"):
        chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v)


def test_recusa_limiar_diferente_entre_receita_e_validacao(mundo, cfg):
    r, v = receita_e_validacao()
    v["alerta_seca"]["alvo_spi6_h1"]["rede_final"]["limiar_previsao"] = -0.8
    with pytest.raises(pv.PreverError, match="limiar de alerta"):
        chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v)


@pytest.mark.parametrize("caminho", [("margem", "alvo_spi3_h1"), ("modelos", "persistencia", "alvo_spi6_h1", "por_mes_calendario"),
                                     ("alerta_seca", "alvo_spi3_h1", "rede_final"), ("modelos", "rede_final", "alvo_spi3_h1", "todas", "classes")])
def test_validacao_incompleta_da_erro_claro(mundo, cfg, caminho):
    r, v = receita_e_validacao()
    d = v
    for k in caminho[:-1]:
        d = d[k]
    del d[caminho[-1]]
    with pytest.raises(pv.PreverError, match="não traz"):
        chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v)


def test_validacao_com_bloco_que_nao_e_dicionario(mundo, cfg):
    r, v = receita_e_validacao()
    v["margem"] = 0.5
    with pytest.raises(pv.PreverError, match="não traz"):
        chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v)


def test_validacao_sem_o_mes_calendario_do_alvo(mundo, cfg):
    r, v = receita_e_validacao()
    del v["modelos"]["rede_final"]["alvo_spi3_h1"]["por_mes_calendario"]["10"]
    with pytest.raises(pv.PreverError, match="mês-calendário 10"):
        chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v)


@pytest.mark.parametrize("ruim", [float("nan"), float("inf"), True, "0.6"])
def test_margem_que_nao_e_numero_finito_e_recusada(mundo, cfg, ruim):
    r, v = receita_e_validacao()
    v["margem"]["alvo_spi3_h1"]["valor"] = ruim
    with pytest.raises(pv.PreverError, match="número finito"):
        chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v)


def test_entrada_nao_e_alterada(mundo, cfg):
    c = mundo["c"]
    X0, Y0, meta0 = c.X.copy(), c.Y.copy(), c.meta.copy()
    r, v = receita_e_validacao()
    r0, v0 = copy.deepcopy(r), copy.deepcopy(v)
    chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v)
    assert np.array_equal(c.X, X0, equal_nan=True) and np.array_equal(c.Y, Y0, equal_nan=True) and c.meta.equals(meta0)
    assert r == r0 and v == v0


def test_mesmo_resultado_duas_vezes(mundo, cfg):
    P = np.array([[-1.2, -0.5], [0.1, 0.2], [0.8, -0.96]])
    a, _ = chamar(mundo, cfg, P)
    b, _ = chamar(mundo, cfg, P)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ------------------------------------------------------------------ gravação
def test_gravar_escreve_json_inteiro_sem_sobras(mundo, cfg, tmp_path):
    doc, _ = chamar(mundo, cfg, np.zeros((3, 2)))
    destino = tmp_path / "novo" / "dir" / "previsao.json"
    assert pv.gravar(doc, destino) == destino
    assert json.loads(destino.read_text(encoding="utf-8")) == doc
    assert destino.read_text(encoding="utf-8").endswith("}\n") and "é" in destino.read_text(encoding="utf-8")
    assert [p.name for p in destino.parent.iterdir()] == ["previsao.json"]       # sem .tmp


def test_gravar_recusa_nan_e_nao_estraga_o_arquivo_anterior(tmp_path):
    destino = tmp_path / "p.json"
    destino.write_text('{"ok": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        pv.gravar({"x": float("nan")}, destino)
    assert destino.read_text(encoding="utf-8") == '{"ok": 1}\n'
    assert [p.name for p in tmp_path.iterdir()] == ["p.json"]


# ------------------------------------------------------------------ ponta a ponta com um projeto de mentira
@pytest.fixture
def proj(tmp_path):
    return montar_proj(tmp_path, raw_sintetico())


def _montar_modelo(cfg, raiz, sha, *, regional=False):
    c, _ = en.montar_conjunto(cfg)
    if regional:
        c = rs.adicionar_regional(c)
    receita, val = receita_e_validacao(sha)
    receita["config"]["regional"] = regional
    receita["colunas"] = [*c.colunas, "base_alvo_spi3_h1", "base_alvo_spi6_h1"]
    pasta = raiz / "artifacts" / "modelo_x"
    pasta.mkdir(parents=True)
    (pasta / "receita.json").write_text(json.dumps(receita), encoding="utf-8")
    arq_val = raiz / "artifacts" / "busca" / "confirmacao_B.json"
    arq_val.parent.mkdir(parents=True)
    arq_val.write_text(json.dumps(val), encoding="utf-8")
    return pasta, arq_val, receita, val


def test_gerar_ponta_a_ponta(proj, monkeypatch):
    cfg, raiz = proj
    sha = mf.verificar_manifest(cfg)["dataset_sha256"]
    pasta, arq_val, receita, _ = _montar_modelo(cfg, raiz, sha)
    falso = Falso(np.array([[-1.5, -1.0], [0.0, 0.0], [0.5, 0.5]]))
    monkeypatch.setattr(bu, "carregar_final", lambda p, base: falso)
    destino = raiz / "data" / "reference" / "previsao_proximo_mes.json"
    doc = pv.gerar(cfg, pasta_modelo=pasta, arquivo_validacao=arq_val, destino=destino, gerado_em=date(2026, 11, 19))
    assert json.loads(destino.read_text(encoding="utf-8")) == doc
    assert doc["dataset_sha256"] == sha and doc["modelo"]["n_membros"] == 5
    assert doc["modelo"]["receita_sha256"] == hashlib.sha256((pasta / "receita.json").read_bytes()).hexdigest()
    assert doc["alvo"] == "2026-10" and doc["resumo"]["n_municipios"] == 3
    assert doc["resumo"]["n_alerta_spi3"] == 1 and doc["resumo"]["n_alerta_spi6"] == 1          # [-1,5; -1,0] vs limiares -1,0 e -0,95
    assert len(falso.vistos) == 1 and "reg_chuva_lag0" not in falso.vistos[0].colunas           # sem contexto regional nesta receita


def test_gerar_com_contexto_regional_na_receita(proj, monkeypatch):
    cfg, raiz = proj
    sha = mf.verificar_manifest(cfg)["dataset_sha256"]
    pasta, arq_val, _, _ = _montar_modelo(cfg, raiz, sha, regional=True)
    falso = Falso(np.zeros((3, 2)))
    monkeypatch.setattr(bu, "carregar_final", lambda p, base: falso)
    pv.gerar(cfg, pasta_modelo=pasta, arquivo_validacao=arq_val, destino=raiz / "p.json", gerado_em=date(2026, 11, 19))
    assert "reg_chuva_lag0" in falso.vistos[0].colunas                                           # o preditor recebeu as colunas regionais


def test_gerar_nao_grava_o_que_o_site_recusaria(proj, monkeypatch):
    cfg, raiz = proj
    sha = mf.verificar_manifest(cfg)["dataset_sha256"]
    pasta, arq_val, receita, _ = _montar_modelo(cfg, raiz, sha)
    receita["n_membros"] = 0                                             # o leitor do site exige pelo menos 1 rede
    (pasta / "receita.json").write_text(json.dumps(receita), encoding="utf-8")
    monkeypatch.setattr(bu, "carregar_final", lambda p, base: Falso(np.zeros((3, 2))))
    destino = raiz / "p.json"
    with pytest.raises(pv.PreverError, match="conferência do site"):
        pv.gerar(cfg, pasta_modelo=pasta, arquivo_validacao=arq_val, destino=destino)
    assert not destino.exists()


def test_validacao_sem_corte_b(mundo, cfg):
    r, v = receita_e_validacao()
    del v["corte_b"]
    with pytest.raises(pv.PreverError, match="não traz corte_b"):
        chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v)


def test_gerar_recusa_rodar_com_o_teste_liberado(proj, monkeypatch):
    cfg, raiz = proj
    sha = mf.verificar_manifest(cfg)["dataset_sha256"]
    pasta, arq_val, _, _ = _montar_modelo(cfg, raiz, sha)
    monkeypatch.setattr(bu, "carregar_final", lambda p, base: pytest.fail("não podia nem carregar a rede"))
    liberado = cfg.model_copy(update={"split": cfg.split.model_copy(update={"teste_liberado": True})})
    with pytest.raises(pv.PreverError, match="teste_liberado"):
        pv.gerar(liberado, pasta_modelo=pasta, arquivo_validacao=arq_val, destino=raiz / "p.json")


@pytest.mark.parametrize("caminho", [("comparacao", "alvo_spi3_h1", "persistencia"), ("comparacao", "alvo_spi6_h1", "melhor_baseline")])
def test_validacao_sem_comparacao_da_erro_claro(mundo, cfg, caminho):
    r, v = receita_e_validacao()
    d = v
    for k in caminho[:-1]:
        d = d[k]
    del d[caminho[-1]]
    with pytest.raises(pv.PreverError, match="não traz"):
        chamar(mundo, cfg, np.zeros((3, 2)), receita=r, val=v)


def test_gerar_recusa_modelo_de_outro_dataset(proj, monkeypatch):
    cfg, raiz = proj
    sha = mf.verificar_manifest(cfg)["dataset_sha256"]
    pasta, arq_val, receita, _ = _montar_modelo(cfg, raiz, sha)
    receita["dataset_sha256"] = "cd" * 32
    (pasta / "receita.json").write_text(json.dumps(receita), encoding="utf-8")
    monkeypatch.setattr(bu, "carregar_final", lambda p, base: pytest.fail("não podia nem carregar a rede"))
    destino = raiz / "previsao.json"
    with pytest.raises(pv.PreverError, match="outro dataset_sha256"):
        pv.gerar(cfg, pasta_modelo=pasta, arquivo_validacao=arq_val, destino=destino)
    assert not destino.exists()


def test_gerar_recusa_validacao_de_outro_dataset(proj, monkeypatch):
    cfg, raiz = proj
    sha = mf.verificar_manifest(cfg)["dataset_sha256"]
    pasta, arq_val, _, val = _montar_modelo(cfg, raiz, sha)
    val["dataset_sha256"] = "cd" * 32
    arq_val.write_text(json.dumps(val), encoding="utf-8")
    monkeypatch.setattr(bu, "carregar_final", lambda p, base: pytest.fail("não podia nem carregar a rede"))
    with pytest.raises(pv.PreverError, match="a validação é de outro dataset_sha256"):
        pv.gerar(cfg, pasta_modelo=pasta, arquivo_validacao=arq_val, destino=raiz / "p.json")


def test_gerar_recusa_colunas_diferentes_das_do_treino(proj, monkeypatch):
    cfg, raiz = proj
    sha = mf.verificar_manifest(cfg)["dataset_sha256"]
    pasta, arq_val, receita, _ = _montar_modelo(cfg, raiz, sha)
    receita["colunas"] = receita["colunas"][::-1]
    (pasta / "receita.json").write_text(json.dumps(receita), encoding="utf-8")
    monkeypatch.setattr(bu, "carregar_final", lambda p, base: pytest.fail("não podia nem carregar a rede"))
    with pytest.raises(pv.PreverError, match="colunas"):
        pv.gerar(cfg, pasta_modelo=pasta, arquivo_validacao=arq_val, destino=raiz / "p.json")


def test_gerar_recusa_json_que_nao_e_objeto(proj):
    cfg, raiz = proj
    sha = mf.verificar_manifest(cfg)["dataset_sha256"]
    pasta, arq_val, _, _ = _montar_modelo(cfg, raiz, sha)
    arq_val.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(pv.PreverError, match="não é um objeto"):
        pv.gerar(cfg, pasta_modelo=pasta, arquivo_validacao=arq_val, destino=raiz / "p.json")


@pytest.mark.parametrize("quebrar", ["receita", "validacao"])
def test_gerar_com_arquivo_ausente_ou_estragado(proj, quebrar):
    cfg, raiz = proj
    sha = mf.verificar_manifest(cfg)["dataset_sha256"]
    pasta, arq_val, _, _ = _montar_modelo(cfg, raiz, sha)
    (pasta / "receita.json" if quebrar == "receita" else arq_val).write_text("{não é json", encoding="utf-8")
    with pytest.raises(pv.PreverError, match="ausente ou inválido"):
        pv.gerar(cfg, pasta_modelo=pasta, arquivo_validacao=arq_val, destino=raiz / "p.json")
    with pytest.raises(pv.PreverError, match="ausente ou inválido"):
        pv.gerar(cfg, pasta_modelo=pasta / "nao_existe", arquivo_validacao=arq_val, destino=raiz / "p.json")


# ------------------------------------------------------------------ linha de comando
def test_main_imprime_resumo_e_aviso(monkeypatch, mundo, cfg):
    doc, _ = chamar(mundo, cfg, np.array([[-1.5, -1.0], [0.0, 0.0], [0.5, 0.5]]))
    monkeypatch.setattr(pv, "gerar", lambda *a, **k: doc)
    monkeypatch.setattr(pv, "load_config", lambda p: cfg)
    linhas = []
    assert pv.main(["--saida", "x.json"], saida=linhas.append) == 0
    texto = "\n".join(linhas)
    assert "2026-10" in texto and "3 municípios" in texto and "SPI-3 1, SPI-6 1" in texto
    assert "já passou" in texto and pv.AVISO in texto


def test_main_nao_diz_que_o_mes_passou_quando_nao_passou(monkeypatch, mundo, cfg):
    doc, _ = chamar(mundo, cfg, np.zeros((3, 2)), hoje=date(2026, 10, 15))
    monkeypatch.setattr(pv, "gerar", lambda *a, **k: doc)
    monkeypatch.setattr(pv, "load_config", lambda p: cfg)
    linhas = []
    assert pv.main([], saida=linhas.append) == 0
    assert "já passou" not in "\n".join(linhas)


def test_main_devolve_1_com_mensagem_quando_falha(monkeypatch, cfg):
    def falha(*a, **k):
        raise pv.PreverError("a receita é de outro dataset")
    monkeypatch.setattr(pv, "gerar", falha)
    monkeypatch.setattr(pv, "load_config", lambda p: cfg)
    linhas = []
    assert pv.main([], saida=linhas.append) == 1
    assert linhas == ["ERRO: a receita é de outro dataset"]
