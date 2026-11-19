"""Testes de src/model/features.py (Item 9). Sem internet; dados sintéticos só aqui (contrato §1.4)."""
import json

import numpy as np
import pandas as pd
import pytest

from dados_sinteticos import A, B, C, PARAMS_REAL, raw_sintetico
from src.data import build_demo as bd
from src.data import quality, splits
from src.data.config import load_config
from src.model import features as fe

D = pd.Timestamp


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def raw():
    return raw_sintetico()


def montar(raw_, cfg_, municipios=None):
    r = bd.montar_demo(raw_, cfg_)
    q = quality.avaliar_qualidade(raw_, cfg_)
    mun = r.qualidade_municipio if municipios is None else municipios
    c = fe.construir(r.demo_observado, q.serie_imputada, r.climatologia_precip, mun, cfg_)
    return c, r, q


@pytest.fixture(scope="module")
def base(raw, cfg):
    return montar(raw, cfg)


def liberado(cfg_):
    return cfg_.model_copy(update={"split": cfg_.split.model_copy(update={"teste_liberado": True})})


def linha(c, cod, mes):
    m = c.meta
    return m.index[(m["cod_ibge"] == cod) & (m["mes_origem"] == D(mes))][0]


def col(c, nome):
    return c.X[:, c.colunas.index(nome)]


def meses_entre(m_ini, m_fim):
    return (m_fim.dt.year * 12 + m_fim.dt.month) - (m_ini.dt.year * 12 + m_ini.dt.month)


# ------------------------------------------------------------------ formato
def test_formato_e_tipos(base):
    c = base[0]
    n = 3 * (549 - 12 + 1)
    assert c.X.shape == (n, 41) and c.X.dtype == np.float32
    assert c.Y.shape == (n, 2) and c.Y.dtype == np.float32
    assert c.M.dtype == bool and np.array_equal(c.M, np.isfinite(c.Y))
    assert len(c.colunas) == len(set(c.colunas)) == 41 and c.continuas.dtype == bool and len(c.continuas) == 41
    assert c.alvos == ("alvo_spi3_h1", "alvo_spi6_h1") and len(c) == n
    assert len(c.meta) == n and list(c.meta.columns[:4]) == ["cod_ibge", "mes_origem", "mes_alvo", "split"]
    assert {"chuva_lag0", "chuva_lag11", "chuva_falta_lag11", "spi3_lag2", "spi6_ausente_lag2", "spi3_baixa_lag0",
            "mes_alvo_sin", "mes_alvo_cos", "clim_mm_alvo"} <= set(c.colunas)
    assert "chuva_lag12" not in c.colunas                                  # 12 meses de histórico: t-11..t


def test_ordem_cod_major_e_origens(base):
    m = base[0].meta
    assert list(m["cod_ibge"].unique()) == [A, B, C]
    for _, g in m.groupby("cod_ibge"):
        assert g["mes_origem"].is_monotonic_increasing and g["mes_origem"].min() == D("1981-12-01")
        assert g["mes_origem"].max() == D("2026-09-01") and g["mes_origem"].is_unique
    assert (meses_entre(m["mes_origem"], m["mes_alvo"]) == 1).all()


def test_escalas_e_horizonte_vem_do_nome_dos_alvos(base, raw, cfg):
    c = base[0]
    assert c.escalas == (3, 6) and c.horizonte == 1
    cfg2 = cfg.model_copy(update={"alvo": cfg.alvo.model_copy(update={"horizonte_h": 2})})
    assert montar(raw, cfg2)[0].horizonte == 2
    import dataclasses
    ruim = dataclasses.replace(c, alvos=("alvo_spi3", "alvo_spi6_h1"))
    with pytest.raises(fe.FeaturesError, match="padrão"):
        ruim.escalas
    with pytest.raises(fe.FeaturesError, match="padrão"):
        ruim.horizonte


def test_determinismo(raw, cfg, base):
    c2 = montar(raw, cfg)[0]
    np.testing.assert_array_equal(base[0].X, c2.X)
    np.testing.assert_array_equal(base[0].Y, c2.Y)


# ------------------------------------------------------------------ valores
def test_alvo_e_o_spi_de_t_mais_h(base):
    c, r, _ = base
    j = c.meta.copy()
    j["y3"], j["y6"] = c.Y[:, 0], c.Y[:, 1]
    spi = r.demo_observado[["cod_ibge", "mes", "spi3", "spi6"]].rename(columns={"mes": "mes_alvo"})
    j = j.merge(spi, on=["cod_ibge", "mes_alvo"], how="left")
    visivel = ((j["split"] != "test") & (j["mes_alvo"] <= D("2026-09-01"))).to_numpy()
    np.testing.assert_array_equal(j.loc[visivel, "y3"].to_numpy(), j.loc[visivel, "spi3"].to_numpy())
    np.testing.assert_array_equal(j.loc[visivel, "y6"].to_numpy(), j.loc[visivel, "spi6"].to_numpy())
    assert np.isfinite(j.loc[visivel, "y3"]).sum() > 1000                  # o teste não é vazio por acaso


def test_lags_de_chuva_leem_o_passado(base):
    c, _, q = base
    s = q.serie_imputada.set_index(["cod_ibge", "mes"])["precip_mm"]
    for cod, mes in ((A, "2003-07-01"), (B, "1999-02-01"), (C, "2026-09-01")):
        i = linha(c, cod, mes)
        for k in (0, 1, 5, 11):
            mes_k = (D(mes).to_period("M") - k).to_timestamp()
            esperado = np.float32(np.log1p(float(s.loc[(cod, mes_k)])))
            assert col(c, f"chuva_lag{k}")[i] == pytest.approx(float(esperado), rel=1e-6)


def test_lags_do_spi_e_calendario_do_alvo(base):
    c, r, _ = base
    spi = r.demo_observado.set_index(["cod_ibge", "mes"])
    i = linha(c, B, "2010-04-01")
    for j, mes in enumerate(("2010-04-01", "2010-03-01", "2010-02-01")):
        assert col(c, f"spi3_lag{j}")[i] == pytest.approx(float(spi.loc[(B, D(mes)), "spi3"]), abs=1e-6)
        assert col(c, f"spi6_lag{j}")[i] == pytest.approx(float(spi.loc[(B, D(mes)), "spi6"]), abs=1e-6)
    assert col(c, "mes_alvo_sin")[i] == pytest.approx(np.sin(2 * np.pi * 5 / 12), abs=1e-6)     # alvo = maio
    assert col(c, "mes_alvo_cos")[i] == pytest.approx(np.cos(2 * np.pi * 5 / 12), abs=1e-6)


def test_climatologia_do_mes_alvo(base):
    c, r, _ = base
    cl = r.climatologia_precip.set_index(["cod_ibge", "mes_calendario"])["mediana_mm"]
    i = linha(c, C, "2012-08-01")                                          # alvo = setembro
    assert col(c, "clim_mm_alvo")[i] == pytest.approx(float(np.log1p(cl.loc[(C, 9)])), rel=1e-6)


def test_mes_sem_dado_vira_flag_e_nan_e_alvo_sem_rotulo(raw, cfg):
    r3 = raw.copy()
    r3.loc[(r3["cod_ibge"] == A) & (r3["mes"] == D("2000-05-01")), "precip_mm"] = np.nan
    c, _, q = montar(r3, cfg)
    i_mai, i_abr, i_jun = (linha(c, A, m) for m in ("2000-05-01", "2000-04-01", "2000-06-01"))
    assert col(c, "chuva_falta_lag0")[i_mai] == 1 and col(c, "chuva_falta_lag1")[i_jun] == 1
    assert col(c, "chuva_falta_lag0")[i_abr] == 0
    for m in ("2000-05-01", "2000-06-01", "2000-07-01"):                  # janela do SPI-3 contém o mês faltante
        i = linha(c, A, m)
        assert col(c, "spi3_ausente_lag0")[i] == 1 and np.isnan(col(c, "spi3_lag0")[i])
    assert np.isnan(c.Y[i_abr]).all() and not c.M[i_abr].any()             # alvo de maio inválido: NaN, nunca 0
    imp = q.serie_imputada.set_index(["cod_ibge", "mes"]).loc[(A, D("2000-05-01")), "precip_mm"]
    assert col(c, "chuva_lag0")[i_mai] == pytest.approx(float(np.log1p(imp)), rel=1e-6)
    assert col(c, "chuva_lag0")[i_mai] == pytest.approx(col(c, "clim_mm_alvo")[i_abr], rel=1e-6)   # imputado pela mediana


# ------------------------------------------------------------------ anti-vazamento
def test_o_futuro_nao_altera_features_nem_alvos_passados(raw, cfg, base):
    r2 = raw.copy()
    fut = r2["mes"] >= D("2018-01-01")
    r2.loc[fut, "precip_mm"] = r2.loc[fut, "precip_mm"] * 3 + 5
    c1, (c2, _, _) = base[0], montar(r2, cfg)
    assert c1.colunas == c2.colunas and c1.meta.equals(c2.meta)
    passado = (c1.meta["mes_origem"] <= D("2017-12-01")).to_numpy()
    np.testing.assert_array_equal(c1.X[passado], c2.X[passado])
    ate_2017 = (c1.meta["mes_alvo"] <= D("2017-12-01")).to_numpy() & (c1.meta["split"] != "test").to_numpy()
    np.testing.assert_array_equal(c1.Y[ate_2017], c2.Y[ate_2017])
    # controle positivo: o teste só vale se o futuro alterado realmente muda o que vem depois
    futuro = (c1.meta["mes_origem"] >= D("2018-01-01")).to_numpy()
    assert not np.array_equal(c1.X[futuro], c2.X[futuro])
    assert not np.array_equal(c1.Y[~ate_2017 & c1.M.any(axis=1)], c2.Y[~ate_2017 & c1.M.any(axis=1)])


def test_blocos_seguem_o_mes_alvo(base, cfg):
    m = base[0].meta
    lim = {"train": ("1982-01-01", "2015-12-01"), "val": ("2016-06-01", "2020-12-01"),
           "test": ("2021-06-01", "2025-12-01"), "pos_teste": ("2026-01-01", "2026-10-01")}
    for nome, (ini, fim) in lim.items():
        g = m.loc[m["split"] == nome, "mes_alvo"]
        assert len(g) > 0 and g.min() >= D(ini) and g.max() <= D(fim), nome
    e = m.loc[m["split"] == "embargo", "mes_alvo"]
    assert set(e.dt.strftime("%Y-%m")) == {"2016-01", "2016-02", "2016-03", "2016-04", "2016-05",
                                           "2021-01", "2021-02", "2021-03", "2021-04", "2021-05"}
    # origem em 2015-12 tem alvo 2016-01: é embargo, não treino (o split é por mes_alvo, não por mes_origem)
    assert m.loc[linha(base[0], A, "2015-12-01"), "split"] == "embargo"
    assert m.loc[linha(base[0], A, "2015-11-01"), "split"] == "train"
    assert (m["split"].astype(str) == splits.rotular(m["mes_alvo"].clip(upper=D("2025-12-01")), cfg).astype(str)
            ).where(m["split"] != "pos_teste", True).all()


def test_teste_congelado_por_padrao(base, cfg):
    c = base[0]
    teste = (c.meta["split"] == "test").to_numpy()
    assert teste.sum() > 0 and np.isnan(c.Y[teste]).all() and not c.M[teste].any()
    assert c.M[(c.meta["split"] == "val").to_numpy()].any() and c.M[(c.meta["split"] == "train").to_numpy()].any()
    with pytest.raises(splits.TesteBloqueadoError):
        fe.selecionar(c, "test", cfg)


def test_teste_liberado_traz_os_rotulos_sem_mexer_nos_outros_blocos(raw, cfg, base):
    cfg2 = liberado(cfg)
    c2 = montar(raw, cfg2)[0]
    c1 = base[0]
    teste = (c2.meta["split"] == "test").to_numpy()
    assert c2.M[teste].any() and len(fe.selecionar(c2, "test", cfg2, so_com_rotulo=True)) > 0
    np.testing.assert_array_equal(c1.X, c2.X)
    np.testing.assert_array_equal(c1.Y[~teste], c2.Y[~teste])


def test_pos_teste_e_a_origem_mais_recente(base):
    c = base[0]
    pos = c.meta["split"] == "pos_teste"
    assert pos.sum() == 3 * 10                                             # origens 2025-12 .. 2026-09 por município
    ultima = linha(c, A, "2026-09-01")
    assert c.meta.loc[ultima, "split"] == "pos_teste" and c.meta.loc[ultima, "mes_alvo"] == D("2026-10-01")
    assert np.isnan(c.Y[ultima]).all() and not c.M[ultima].any()
    anterior = linha(c, A, "2026-08-01")                                   # alvo 2026-09 existe e é só observado
    assert c.M[anterior].any() and c.meta.loc[anterior, "split"] == "pos_teste"


def test_horizonte_dois(raw, cfg):
    cfg2 = cfg.model_copy(update={"alvo": cfg.alvo.model_copy(update={"horizonte_h": 2})})
    c, r, _ = montar(raw, cfg2)
    assert c.alvos == ("alvo_spi3_h2", "alvo_spi6_h2")
    assert (meses_entre(c.meta["mes_origem"], c.meta["mes_alvo"]) == 2).all()
    i = linha(c, B, "2010-04-01")
    spi = r.demo_observado.set_index(["cod_ibge", "mes"])
    assert c.Y[i, 0] == pytest.approx(float(spi.loc[(B, D("2010-06-01")), "spi3"]), abs=1e-6)
    for m in ("2026-08-01", "2026-09-01"):
        assert np.isnan(c.Y[linha(c, A, m)]).all()


# ------------------------------------------------------------------ entradas e erros
def test_municipio_excluido_sai_das_linhas(raw, cfg):
    r = bd.montar_demo(raw, cfg)
    mun = r.qualidade_municipio.copy()
    mun.loc[mun["cod_ibge"] == B, "excluded"] = True
    c, _, _ = montar(raw, cfg, mun)
    assert set(c.meta["cod_ibge"]) == {A, C} and len(c) == 2 * (549 - 12 + 1)
    mun["excluded"] = True
    with pytest.raises(fe.FeaturesError, match="nenhum município"):
        montar(raw, cfg, mun)


def test_grade_incompleta_repetida_ou_sem_coluna(raw, cfg):
    r = bd.montar_demo(raw, cfg)
    q = quality.avaliar_qualidade(raw, cfg)
    ok = (r.demo_observado, q.serie_imputada, r.climatologia_precip, r.qualidade_municipio)

    def construir(demo=ok[0], serie=ok[1], clim=ok[2], mun=ok[3]):
        return fe.construir(demo, serie, clim, mun, cfg)

    with pytest.raises(fe.FeaturesError, match="grade"):
        construir(demo=ok[0].iloc[1:])
    with pytest.raises(fe.FeaturesError, match="repetidos"):
        construir(serie=pd.concat([ok[1], ok[1].iloc[:1]], ignore_index=True))
    with pytest.raises(fe.FeaturesError, match="ausente"):
        construir(demo=ok[0].drop(columns=["spi6_baixa_conf"]))
    with pytest.raises(fe.FeaturesError, match="12 meses"):
        construir(clim=ok[2].iloc[1:])
    with pytest.raises(fe.FeaturesError, match="repetido"):
        construir(mun=pd.concat([ok[3], ok[3].iloc[:1]], ignore_index=True))
    neg = ok[1].copy()
    neg.loc[neg.index[100], "precip_mm"] = -1.0
    with pytest.raises(fe.FeaturesError, match="negativa"):
        construir(serie=neg)


def test_baixa_confianca_nunca_acompanha_spi_ausente(raw, cfg):
    r = bd.montar_demo(raw, cfg)
    q = quality.avaliar_qualidade(raw, cfg)
    demo = r.demo_observado.copy()
    ruim = (demo["cod_ibge"] == A) & demo["mes"].between(D("2005-03-01"), D("2005-05-01"))
    demo.loc[ruim, ["spi3_invalid", "spi3"]] = [True, np.nan]               # 3 meses sem SPI-3 em um município
    demo["spi3_baixa_conf"] = True                                          # entrada incoerente: baixa confiança até no inválido
    c = fe.construir(demo, q.serie_imputada, r.climatologia_precip, r.qualidade_municipio, cfg)
    aus, baixa = col(c, "spi3_ausente_lag0") == 1, col(c, "spi3_baixa_lag0") == 1
    assert aus.any() and not (aus & baixa).any() and baixa[~aus].all()
    fora_do_teste = (c.meta["split"] != "test").to_numpy()                  # no teste o alvo está congelado de propósito
    assert not (c.meta["alvo_spi3_baixa_conf"].to_numpy() & ~c.M[:, 0] & fora_do_teste).any()   # idem para o alvo


def test_lag_menor_que_o_spi_e_recusado(raw, cfg):
    cfg2 = cfg.model_copy(update={"features": cfg.features.model_copy(update={"max_lag_meses": 2})})
    with pytest.raises(fe.FeaturesError, match="max_lag_meses"):
        montar(raw, cfg2)


# ------------------------------------------------------------------ selecionar e fatiar
def test_selecionar_e_fatiar(base, cfg):
    c = base[0]
    tr = fe.selecionar(c, "train", cfg)
    assert (tr.meta["split"].astype(str) == "train").all() and len(tr) < len(c)
    rot = fe.selecionar(c, "val", cfg, so_com_rotulo=True)
    assert rot.M.any(axis=1).all()
    assert fe.selecionar(c, "embargo", cfg).meta["split"].astype(str).eq("embargo").all()
    with pytest.raises(fe.FeaturesError, match="desconhecido"):
        fe.selecionar(c, "treino", cfg)
    with pytest.raises(fe.FeaturesError, match="máscara"):
        c.fatiar(np.ones(len(c) - 1, dtype=bool))
    with pytest.raises(fe.FeaturesError, match="máscara"):
        c.fatiar(np.ones(len(c), dtype=int))


# ------------------------------------------------------------------ padronizador
def test_padronizador_so_ajusta_no_treino(base, cfg):
    c = base[0]
    with pytest.raises(fe.FeaturesError, match="train"):
        fe.Padronizador.ajustar(fe.selecionar(c, "val", cfg))
    with pytest.raises(fe.FeaturesError, match="train"):
        fe.Padronizador.ajustar(c)                                         # mistura de blocos
    with pytest.raises(fe.FeaturesError, match="vazio"):
        fe.Padronizador.ajustar(c.fatiar(np.zeros(len(c), dtype=bool)))


def test_padronizador_zera_media_e_unifica_desvio_so_nas_continuas(base, cfg):
    c = base[0]
    tr = fe.selecionar(c, "train", cfg)
    p = fe.Padronizador.ajustar(tr)
    Z = p.transformar(tr)
    assert Z.dtype == np.float32 and Z.shape == tr.X.shape and np.isfinite(Z).all()
    cont = tr.continuas
    ok = np.isfinite(tr.X)
    for k in np.flatnonzero(cont):
        v = Z[ok[:, k], k]
        assert abs(v.mean()) < 1e-3 and v.std() == pytest.approx(1.0, abs=1e-3), tr.colunas[k]
    np.testing.assert_array_equal(Z[:, ~cont], tr.X[:, ~cont])             # flags e calendário ficam como estão
    assert (p.media[~cont] == 0).all() and (p.desvio[~cont] == 1).all()


def test_padronizador_ignora_validacao_e_teste(raw, cfg, base):
    r2 = raw.copy()
    fut = r2["mes"] >= D("2017-01-01")                                     # depois do último mês-alvo de treino
    r2.loc[fut, "precip_mm"] = r2.loc[fut, "precip_mm"] * 9 + 40
    c2 = montar(r2, cfg)[0]
    p1 = fe.Padronizador.ajustar(fe.selecionar(base[0], "train", cfg))
    p2 = fe.Padronizador.ajustar(fe.selecionar(c2, "train", cfg))
    np.testing.assert_array_equal(p1.media, p2.media)
    np.testing.assert_array_equal(p1.desvio, p2.desvio)


def test_padronizador_nan_constante_e_colunas(base, cfg):
    import dataclasses
    tr = fe.selecionar(base[0], "train", cfg)
    k_const, k_nan = tr.colunas.index("chuva_lag0"), tr.colunas.index("chuva_lag1")
    X = tr.X.copy()
    X[:, k_const] = 3.0
    X[:, k_nan] = np.nan
    tr2 = dataclasses.replace(tr, X=X)
    p = fe.Padronizador.ajustar(tr2)
    assert p.desvio[k_const] == 1.0 and p.media[k_const] == pytest.approx(3.0)
    assert p.desvio[k_nan] == 1.0 and p.media[k_nan] == 0.0               # coluna toda NaN: sem estatística
    Z = p.transformar(tr2)
    assert (Z[:, k_const] == 0).all() and (Z[:, k_nan] == 0).all() and np.isfinite(Z).all()
    outro = dataclasses.replace(tr2, colunas=tuple(reversed(tr2.colunas)))
    with pytest.raises(fe.FeaturesError, match="colunas"):
        p.transformar(outro)


def test_padronizador_ida_e_volta_em_json(base, cfg):
    p = fe.Padronizador.ajustar(fe.selecionar(base[0], "train", cfg))
    q = fe.Padronizador.de_dict(json.loads(json.dumps(p.para_dict())))
    assert q.colunas == p.colunas
    np.testing.assert_array_equal(q.media, p.media)
    np.testing.assert_array_equal(q.desvio, p.desvio)
    np.testing.assert_array_equal(q.continuas, p.continuas)
    d = p.para_dict()
    for ruim in ({**d, "desvio": d["desvio"][:-1]}, {**d, "desvio": [0.0] * len(d["desvio"])},
                 {**d, "media": [float("nan")] * len(d["media"])}, {"colunas": d["colunas"]}):
        with pytest.raises(fe.FeaturesError):
            fe.Padronizador.de_dict(ruim)
