"""Testes de src/data/spi.py (Item 5). Sem internet; séries sintéticas só aqui (nunca em data/).

A fórmula do SPI é conferida contra uma conta independente (scipy) feita no próprio teste.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from src.data import spi as sp
from src.data.config import PendenciaError, load_config

PARAMS_REAL = Path(__file__).resolve().parents[1] / "params.yaml"
COD = "2200001"

ESCALA_MM = np.array([110, 130, 150, 120, 55, 14, 8, 6, 8, 20, 45, 80], float)   # escala gama por mês-calendário
PZERO = np.array([0, 0, 0, 0, .03, .15, .30, .40, .35, .15, .05, .01])            # chance de mês sem chuva


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


def com_media_min(cfg, mm):
    """Cópia da config com outro limite de chuva esperada (mm por mês da janela); 0 desliga a regra."""
    return cfg.model_copy(update={"spi": cfg.spi.model_copy(update={"media_min_mm_por_mes": mm})})


def meses(inicio="1981-01", fim="2025-12"):
    return pd.period_range(inicio, fim, freq="M")


def precip_sintetica(n_cod=3, seed=0, inicio="1981-01", fim="2025-12") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    mm = meses(inicio, fim)
    cal = mm.month.to_numpy() - 1
    quadros = []
    for i in range(n_cod):
        p = rng.gamma(2.0, ESCALA_MM[cal])
        p = np.where(rng.random(len(mm)) < PZERO[cal], 0.0, p)
        quadros.append(pd.DataFrame({"cod_ibge": f"22{i:05d}", "mes": mm.to_timestamp(), "precip_mm": p}))
    return pd.concat(quadros, ignore_index=True)


def uma_serie(valores, cod=COD, inicio="1981-01") -> pd.DataFrame:
    mm = pd.period_range(inicio, periods=len(valores), freq="M")
    return pd.DataFrame({"cod_ibge": cod, "mes": mm.to_timestamp(), "precip_mm": np.asarray(valores, float)})


def serie_com_zeros_no_trimestre_jun_ago(anos_zero, seed=5):
    """Série única onde Jun-Jul-Ago têm chuva 0 nos `anos_zero` (então S3 final em ago = 0)."""
    mm = meses()
    p = np.random.default_rng(seed).gamma(2.0, 60.0, len(mm))
    for ano in anos_zero:
        p[(mm.year == ano) & np.isin(mm.month, (6, 7, 8))] = 0.0
    return uma_serie(p)


def indexado(res):
    return res.spi.set_index(["cod_ibge", "mes"])


def linha_param(res, cod, k, mes_cal):
    p = res.parametros
    return p[(p.cod_ibge == cod) & (p.escala_meses == k) & (p.mes_calendario == mes_cal)].iloc[0]


def cfg_variante(tmp_path, **ajustes_spi):
    d = yaml.safe_load(PARAMS_REAL.read_text(encoding="utf-8"))
    d["spi"].update(ajustes_spi)
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_config(tmp_path / "params.yaml")


def params_manual(cod, q, shape, scale, valido=True):
    return pd.DataFrame({"cod_ibge": cod, "mes_calendario": np.arange(1, 13, dtype="int8"), "escala_meses": np.int8(3),
                         "q": q, "shape": shape, "scale": scale, "n_amostras": 35, "n_positivos": 30,
                         "valido": valido})


# ================================================================== acumular
def test_acumular_soma_exata_e_primeiros_meses_nan():
    m = pd.DataFrame({"A": [1.0, 2, 3, 4, 5], "B": [10.0, 0, 0, 5, 1]}, index=meses("2000-01", "2000-05"))
    s = sp.acumular(m, 3)
    assert s["A"].isna().tolist() == [True, True, False, False, False]
    assert s["A"].dropna().tolist() == [6.0, 9.0, 12.0]
    assert s["B"].dropna().tolist() == [10.0, 5.0, 6.0]
    assert s.index.equals(m.index) and list(s.columns) == ["A", "B"]


def test_acumular_nan_na_janela_contamina_so_as_janelas_que_o_contem():
    m = pd.DataFrame({"A": [1.0, 2, np.nan, 4, 5, 6, 7]}, index=meses("2000-01", "2000-07"))
    s = sp.acumular(m, 3)["A"]
    assert s.isna().tolist() == [True, True, True, True, True, False, False]
    assert s.dropna().tolist() == [15.0, 18.0]


def test_acumular_k1_e_identidade_e_k_maior_que_a_serie_e_tudo_nan():
    m = pd.DataFrame({"A": [1.0, 0.0, 3.0]}, index=meses("2000-01", "2000-03"))
    assert sp.acumular(m, 1)["A"].tolist() == [1.0, 0.0, 3.0]
    assert sp.acumular(m, 4)["A"].isna().all()
    assert sp.acumular(m, 3)["A"].isna().tolist() == [True, True, False]
    assert sp.acumular(m, 3)["A"].iloc[-1] == 4.0                       # série com exatamente k meses: 1 soma


def test_acumular_nao_acumula_erro_de_ponto_flutuante():
    rng = np.random.default_rng(0)
    v = rng.random(540) * 1e6 + 1e-3
    s = sp.acumular(pd.DataFrame({"A": v}, index=meses()), 6)["A"].to_numpy()
    esperado = np.array([v[i - 5:i + 1].sum() for i in range(5, 540)])
    assert np.allclose(s[5:], esperado, rtol=1e-13, atol=0)


@pytest.mark.parametrize("k", [0, -3, True, "3", 2.0])
def test_acumular_escala_invalida(k):
    with pytest.raises(sp.SpiError, match="escala"):
        sp.acumular(pd.DataFrame({"A": [1.0]}, index=meses("2000-01", "2000-01")), k)


# ================================================================== para_matriz
def test_para_matriz_ordena_colunas_preenche_lacuna_com_nan_e_indice_mensal():
    df = pd.concat([uma_serie([1, 2, 3, 4], cod="B"), uma_serie([5, 6, 7, 8], cod="A")])
    df = df[~((df.cod_ibge == "A") & (df.mes == pd.Timestamp("2001-02-01")))]          # sem linha = NaN
    df = df.assign(mes=df.mes + pd.Timedelta(days=9))                                  # qualquer dia do mês
    m = sp.para_matriz(df)
    assert list(m.columns) == ["A", "B"] and isinstance(m.index, pd.PeriodIndex)
    assert m.index[0] == pd.Period("1981-01", "M") and len(m) == 4
    assert m["B"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert m["A"].isna().sum() == 0


def test_para_matriz_mes_ausente_no_meio_vira_nan_nunca_zero():
    df = uma_serie([1, 2, 3, 4, 5])
    m = sp.para_matriz(df.drop(index=2))
    assert len(m) == 5 and np.isnan(m[COD].iloc[2]) and m[COD].iloc[[0, 1, 3, 4]].tolist() == [1, 2, 4, 5]


def test_para_matriz_aceita_float32_e_nan():
    df = uma_serie([1, np.nan, 3])
    df["precip_mm"] = df["precip_mm"].astype("float32")
    m = sp.para_matriz(df)
    assert m.dtypes.iloc[0] == "float64" and np.isnan(m[COD].iloc[1])


@pytest.mark.parametrize("quebra, trecho", [
    (lambda d: pd.concat([d, d.iloc[[0]]]), "duplicatas"),
    (lambda d: d.assign(precip_mm=[1.0, -0.1, 3.0]), "negativo"),
    (lambda d: d.assign(precip_mm=[1.0, np.inf, 3.0]), "infinito"),
    (lambda d: d.drop(columns="precip_mm"), "colunas ausentes"),
    (lambda d: d.iloc[0:0], "vazio"),
    (lambda d: d.assign(mes=[pd.NaT, pd.NaT, pd.NaT]), "ausente"),
    (lambda d: d.assign(cod_ibge=[COD, None, COD]), "ausente"),
    (lambda d: d.assign(precip_mm=["a", "b", "c"]), "conversão"),
])
def test_para_matriz_entradas_invalidas(quebra, trecho):
    with pytest.raises(sp.SpiError, match=trecho):
        sp.para_matriz(quebra(uma_serie([1.0, 2.0, 3.0])))


def test_para_matriz_exige_dataframe():
    with pytest.raises(sp.SpiError, match="DataFrame"):
        sp.para_matriz([1, 2, 3])


# ================================================================== ajuste da Gama
def test_ajuste_recupera_parametros_conhecidos():
    x = np.random.default_rng(1).gamma(2.5, 40.0, 20000)
    q, a, esc, n, npos, ok = sp._ajustar_celula(x, 10, 30)
    assert ok and q == 0.0 and n == npos == 20000
    assert a == pytest.approx(2.5, rel=0.03) and esc == pytest.approx(40.0, rel=0.03)


def test_ajuste_q_conta_zeros_e_ignora_zeros_no_ajuste():
    pos = np.random.default_rng(2).gamma(2.0, 30.0, 30)
    x = np.concatenate([np.zeros(10), pos])
    q, a, esc, n, npos, ok = sp._ajustar_celula(x, 10, 30)
    assert ok and (n, npos) == (40, 30) and q == pytest.approx(10 / 40)
    _, a2, esc2, *_ = sp._ajustar_celula(pos, 10, 30)
    assert (a, esc) == (a2, esc2)                                    # zeros não entram no MLE


@pytest.mark.parametrize("x, rotulo", [
    (np.zeros(35), "toda zero"),
    (np.concatenate([np.zeros(30), np.arange(1.0, 6.0)]), "poucos positivos"),
    (np.full(35, 7.0), "constante"),
    (7.0 + np.random.default_rng(3).normal(0, 1e-9, 35), "quase constante"),
    (np.random.default_rng(4).gamma(2.0, 30.0, 29), "poucas amostras"),
    (np.array([]), "vazia"),
])
def test_celula_sem_condicao_de_ajuste_fica_invalida_sem_levantar(x, rotulo):
    q, a, esc, n, npos, ok = sp._ajustar_celula(x, 10, 30)
    assert not ok and np.isnan(a) and np.isnan(esc), rotulo
    assert n == x.size


def test_limites_exatos_de_amostras_e_positivos():
    x = np.random.default_rng(6).gamma(2.0, 30.0, 30)
    assert sp._ajustar_celula(x, 10, 30)[-1] is True                 # n == mínimo
    assert sp._ajustar_celula(x[:29], 10, 30)[-1] is False           # n == mínimo - 1
    assert sp._ajustar_celula(np.concatenate([np.zeros(20), x[:10]]), 10, 30)[-1] is True   # npos == mínimo
    assert sp._ajustar_celula(np.concatenate([np.zeros(20), x[:9]]), 10, 30)[-1] is False


# ================================================================== SPI: fórmula e convenção D3
def test_spi_positivo_bate_com_conta_independente(cfg):
    df = precip_sintetica(n_cod=2, seed=7)
    res = sp.calcular_spi(df, cfg)
    S = sp.acumular(sp.para_matriz(df), 3)
    idx = indexado(res)
    for cod, mes in [("2200000", "2010-02"), ("2200001", "2019-11"), ("2200000", "2023-04")]:
        x = S.loc[pd.Period(mes, "M"), cod]
        p = linha_param(res, cod, 3, pd.Period(mes, "M").month)
        h = p.q + (1 - p.q) * stats.gamma.cdf(x, p["shape"], scale=p["scale"])
        assert idx.loc[(cod, pd.Timestamp(mes + "-01")), "spi3"] == pytest.approx(float(stats.norm.ppf(h)), abs=2e-6)


def test_zero_usa_h0_igual_a_q_na_validacao(cfg):
    anos = [1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2013, 2018]      # 10 no treino + 2018 (val)
    res = sp.calcular_spi(serie_com_zeros_no_trimestre_jun_ago(anos), cfg)
    p = linha_param(res, COD, 3, 8)
    assert p.n_amostras == 35 and p.q == pytest.approx(10 / 35) and p.valido
    idx = indexado(res)
    esperado = float(stats.norm.ppf(10 / 35))
    for ano in (1983, 2013, 2018):                                                   # treino e validação
        assert idx.loc[(COD, pd.Timestamp(f"{ano}-08-01")), "spi3"] == pytest.approx(esperado, abs=2e-6)
    assert not idx.loc[(COD, pd.Timestamp("2018-08-01")), "spi3_invalid"]


def test_baixa_confianca_quando_q_acima_do_limite(cfg):
    anos = list(range(1981, 2001))                                                   # 20 de 35 -> q = 0,571
    res = sp.calcular_spi(serie_com_zeros_no_trimestre_jun_ago(anos), cfg)
    d = res.spi
    ago = d["mes"].dt.month == 8
    assert linha_param(res, COD, 3, 8).q == pytest.approx(20 / 35)
    assert d.loc[ago & ~d.spi3_invalid, "spi3_baixa_conf"].all()
    assert not d.loc[~ago, "spi3_baixa_conf"].any()


def test_baixa_confianca_e_estritamente_maior_que_o_limite(cfg):
    S = pd.DataFrame({"A": np.arange(1.0, 25.0)}, index=meses("2001-01", "2002-12"))
    for q, esperado in [(0.5, False), (0.5001, True)]:
        _, _, baixa = sp._aplicar(S, params_manual("A", q, 2.0, 10.0), com_media_min(cfg, 0.0))   # isola o critério de zeros
        assert bool(baixa.any()) is esperado


def test_h_limitado_nos_extremos_sem_infinito(cfg):
    S = pd.DataFrame({"A": [0.0, 1e9, np.nan, 20.0]}, index=meses("2001-01", "2001-04"))
    spi, inval, _ = sp._aplicar(S, params_manual("A", 0.0, 2.0, 10.0), cfg)          # q = 0 e x = 0 -> H = 0
    lim = float(stats.norm.ppf(sp.EPS_H))
    assert spi[0, 0] == pytest.approx(lim) and spi[1, 0] == pytest.approx(-lim)
    assert np.isnan(spi[2, 0]) and inval[:, 0].tolist() == [False, False, True, False]
    assert np.isfinite(spi[[0, 1, 3], 0]).all() and sp.EPS_H == 1e-6 and 4.7 < abs(lim) < 4.8


def test_parametro_invalido_ou_ausente_gera_nan_e_invalid(cfg):
    S = pd.DataFrame({"A": [10.0, 20.0], "B": [10.0, 20.0]}, index=meses("2001-01", "2001-02"))
    params = params_manual("A", 0.1, 2.0, 10.0, valido=False)                        # B nem consta na tabela
    spi, inval, baixa = sp._aplicar(S, params, cfg)
    assert np.isnan(spi).all() and inval.all() and not baixa.any()


# ================================================================== anti-vazamento
def test_alterar_o_futuro_nao_muda_parametros_nem_spi_passado(cfg):
    df = precip_sintetica(n_cod=3, seed=11)
    a = sp.calcular_spi(df, cfg)
    futuro = df["mes"] > pd.Timestamp("2015-12-31")
    df2 = df.copy()
    df2.loc[futuro, "precip_mm"] = df2.loc[futuro, "precip_mm"] * 7 + 500.0
    b = sp.calcular_spi(df2, cfg)
    pd.testing.assert_frame_equal(a.parametros, b.parametros, check_exact=True)
    passado = (a.spi["mes"] <= pd.Timestamp("2015-12-01")).to_numpy()
    pd.testing.assert_frame_equal(a.spi[passado].reset_index(drop=True), b.spi[passado].reset_index(drop=True),
                                  check_exact=True)
    assert not a.spi.loc[~passado, "spi3"].equals(b.spi.loc[~passado, "spi3"])        # o futuro mudou de fato


def test_alterar_o_treino_muda_os_parametros(cfg):
    df = precip_sintetica(n_cod=1, seed=12)
    a = sp.calcular_spi(df, cfg)
    df2 = df.copy()
    df2.loc[df2["mes"] == pd.Timestamp("2000-03-01"), "precip_mm"] += 400.0
    b = sp.calcular_spi(df2, cfg)
    assert not a.parametros.equals(b.parametros)


def test_n_amostras_so_conta_meses_finais_ate_o_treino(cfg):
    res = sp.calcular_spi(precip_sintetica(n_cod=1, seed=13), cfg)
    n = lambda k, m: int(linha_param(res, "2200000", k, m).n_amostras)            # noqa: E731
    assert n(3, 1) == 34 and n(3, 2) == 34 and n(3, 3) == 35 and n(3, 12) == 35   # 1ª soma de 3 meses: 1981-03
    assert n(6, 5) == 34 and n(6, 6) == 35                                        # 1ª soma de 6 meses: 1981-06


def test_parametros_dados_sao_apenas_aplicados_sem_reajuste(cfg):
    base = precip_sintetica(n_cod=2, seed=14)
    ref = sp.calcular_spi(base, cfg)
    outro = precip_sintetica(n_cod=2, seed=15)
    reaj = sp.calcular_spi(outro, cfg)
    aplic = sp.calcular_spi(outro, cfg, parametros=ref.parametros)
    pd.testing.assert_frame_equal(aplic.parametros, ref.parametros.reset_index(drop=True))
    assert not aplic.spi["spi3"].equals(reaj.spi["spi3"])
    # mesmos dados + parâmetros do próprio treino = mesmo resultado do ajuste direto
    assert sp.calcular_spi(base, cfg, parametros=ref.parametros).spi.equals(ref.spi)


def test_parametros_incompletos_sao_recusados(cfg):
    ref = sp.calcular_spi(precip_sintetica(n_cod=1, seed=16), cfg)
    with pytest.raises(sp.SpiError, match="colunas"):
        sp.calcular_spi(precip_sintetica(n_cod=1, seed=16), cfg, parametros=ref.parametros.drop(columns="q"))
    with pytest.raises(sp.SpiError, match="SPI-6"):
        sp.calcular_spi(precip_sintetica(n_cod=1, seed=16), cfg,
                        parametros=ref.parametros[ref.parametros.escala_meses == 3])


# ================================================================== qualidade: células inválidas
def test_serie_toda_zero_nao_gera_infinito(cfg):
    res = sp.calcular_spi(uma_serie(np.zeros(540)), cfg)
    assert res.spi["spi3_invalid"].all() and res.spi["spi6_invalid"].all()
    assert res.spi[["spi3", "spi6"]].isna().all().all() and not res.parametros["valido"].any()
    assert (res.parametros["q"] == 1.0).all()


def test_serie_constante_fica_invalida(cfg):
    res = sp.calcular_spi(uma_serie(np.full(540, 10.0)), cfg)
    assert res.spi["spi3_invalid"].all() and not np.isinf(res.spi[["spi3", "spi6"]].to_numpy()).any()


def test_poucos_positivos_invalida_so_a_celula_afetada(cfg):
    anos = list(range(1981, 2011))                                                    # 30 de 35 anos zerados em ago
    res = sp.calcular_spi(serie_com_zeros_no_trimestre_jun_ago(anos), cfg)
    assert not linha_param(res, COD, 3, 8).valido and linha_param(res, COD, 3, 9).valido
    d = res.spi
    ago = d["mes"].dt.month == 8
    assert d.loc[ago, "spi3_invalid"].all() and d.loc[ago, "spi3"].isna().all()
    assert not d.loc[~ago & (d["mes"] >= pd.Timestamp("1981-03-01")), "spi3_invalid"].any()


def test_mes_sem_dado_zera_as_janelas_que_o_contem(cfg):
    df = precip_sintetica(n_cod=1, seed=17)
    df.loc[df["mes"] == pd.Timestamp("2000-05-01"), "precip_mm"] = np.nan
    res = sp.calcular_spi(df, cfg)
    d = res.spi.set_index("mes")
    for mes in ("2000-05-01", "2000-06-01", "2000-07-01"):                            # S3 que contém mai/2000
        assert d.loc[mes, "spi3_invalid"] and np.isnan(d.loc[mes, "spi3"])
    assert not d.loc["2000-08-01", "spi3_invalid"] and not d.loc["2000-04-01", "spi3_invalid"]
    assert d.loc["2000-10-01", "spi6_invalid"] and not d.loc["2000-11-01", "spi6_invalid"]    # S6: mai..out/2000
    assert linha_param(res, "2200000", 3, 5).n_amostras == 34                         # a amostra perdeu 2000


def test_menos_anos_que_o_minimo_invalida_tudo(cfg):
    res = sp.calcular_spi(precip_sintetica(n_cod=1, seed=18, inicio="1990-01"), cfg)  # 26 anos de treino < 30
    assert not res.parametros["valido"].any() and res.spi["spi3_invalid"].all()


# ================================================================== distribuição e schema
def test_spi_de_treino_tem_media_zero_e_desvio_um(cfg):
    res = sp.calcular_spi(precip_sintetica(n_cod=60, seed=21), cfg)
    d = res.spi
    for k in (3, 6):
        z = d.loc[(d["mes"] <= pd.Timestamp("2015-12-01")) & ~d[f"spi{k}_invalid"] & ~d[f"spi{k}_baixa_conf"], f"spi{k}"]
        assert len(z) > 10000
        assert abs(z.mean()) < 0.05 and abs(z.std() - 1.0) < 0.07, (k, z.mean(), z.std())
        assert abs((z <= -1).mean() - 0.1587) < 0.03


def test_schema_ordem_e_tamanho_da_saida(cfg):
    df = precip_sintetica(n_cod=4, seed=22)
    res = sp.calcular_spi(df, cfg)
    d = res.spi
    assert list(d.columns) == ["cod_ibge", "mes", "spi3", "spi3_invalid", "spi3_baixa_conf",
                               "spi6", "spi6_invalid", "spi6_baixa_conf"]
    assert str(d["mes"].dtype) == "datetime64[ns]" and (d["mes"].dt.day == 1).all()
    assert d["spi3"].dtype == "float32" and d["spi6"].dtype == "float32"
    assert d["spi3_invalid"].dtype == bool and d["spi6_baixa_conf"].dtype == bool
    assert len(d) == 4 * 540 and not d.duplicated(["cod_ibge", "mes"]).any()
    assert d.equals(d.sort_values(["cod_ibge", "mes"]).reset_index(drop=True))
    primeiros = d[d["mes"] < pd.Timestamp("1981-03-01")]
    assert primeiros["spi3_invalid"].all() and primeiros["spi3"].isna().all()         # janela incompleta
    seis = d[d["mes"] < pd.Timestamp("1981-06-01")]
    assert seis["spi6_invalid"].all() and not d.loc[d["mes"] == pd.Timestamp("1981-06-01"), "spi6_invalid"].all()
    assert not np.isinf(d[["spi3", "spi6"]].to_numpy()).any()


def test_schema_dos_parametros_segue_o_contrato(cfg):
    p = sp.calcular_spi(precip_sintetica(n_cod=4, seed=23), cfg).parametros
    assert list(p.columns) == sp.COLUNAS_PARAMETROS
    assert len(p) == 4 * 12 * 2 and set(p["escala_meses"]) == {3, 6}
    assert sorted(p["mes_calendario"].unique()) == list(range(1, 13))
    assert p["valido"].dtype == bool and p["n_amostras"].dtype == "int32"
    ok = p[p["valido"]]
    assert ((ok["shape"] > 0) & (ok["scale"] > 0) & ok["q"].between(0, 1)).all()
    assert p.loc[~p["valido"], ["shape", "scale"]].isna().all().all()
    assert not p.duplicated(["cod_ibge", "mes_calendario", "escala_meses"]).any()


def test_resultado_e_deterministico(cfg):
    df = precip_sintetica(n_cod=3, seed=24)
    a, b = sp.calcular_spi(df, cfg), sp.calcular_spi(df, cfg)
    assert a.spi.equals(b.spi) and a.parametros.equals(b.parametros)


def test_ordem_das_linhas_de_entrada_nao_importa(cfg):
    df = precip_sintetica(n_cod=3, seed=25)
    a = sp.calcular_spi(df, cfg)
    b = sp.calcular_spi(df.sample(frac=1.0, random_state=0), cfg)
    assert a.spi.equals(b.spi) and a.parametros.equals(b.parametros)


# ================================================================== configuração
def test_convencao_zero_pendente_bloqueia(tmp_path):
    with pytest.raises(PendenciaError):
        sp.calcular_spi(uma_serie([1.0] * 12), cfg_variante(tmp_path, convencao_zero="VERIFICAR"))


@pytest.mark.parametrize("ajuste, trecho", [({"convencao_zero": "centro"}, "convencao_zero"),
                                            ({"zero_inflado": False}, "zero_inflado"),
                                            ({"agrupar_por": ["cod_ibge"]}, "agrupar_por")])
def test_configuracao_nao_suportada_e_recusada(tmp_path, ajuste, trecho):
    with pytest.raises(sp.SpiError, match=trecho):
        sp.calcular_spi(uma_serie([1.0] * 12), cfg_variante(tmp_path, **ajuste))


def test_janelas_da_config_definem_as_colunas_de_saida(tmp_path):
    cfg3 = cfg_variante(tmp_path, janelas_meses=[3])
    d = sp.calcular_spi(precip_sintetica(n_cod=1, seed=26), cfg3).spi
    assert "spi3" in d.columns and "spi6" not in d.columns


# ================================================================== baixa confiança por pouca chuva esperada
def _baixa(cfg, mm, q, shape, scale, k=3, valido=True, S=None):
    S = S if S is not None else pd.DataFrame({"A": np.arange(1.0, 25.0)}, index=meses("2001-01", "2002-12"))
    p = params_manual("A", q, shape, scale, valido).assign(escala_meses=np.int8(k))
    spi, inval, baixa = sp._aplicar(S, p, com_media_min(cfg, mm))
    return spi, inval, baixa


def test_media_por_mes_e_um_menos_q_vezes_forma_vezes_escala_sobre_k(cfg):
    # q = 0,25; forma 2; escala 8; k = 3 -> (0,75 * 16) / 3 = 4,0 mm por mês, conta exata em ponto flutuante
    assert not _baixa(cfg, 4.0, 0.25, 2.0, 8.0)[2].any()            # estritamente menor: 4,0 < 4,0 é falso
    assert _baixa(cfg, 4.0000001, 0.25, 2.0, 8.0)[2].all()
    assert not _baixa(cfg, 3.9999999, 0.25, 2.0, 8.0)[2].any()


def test_media_por_mes_divide_pela_escala_do_spi(cfg):
    # mesmos parâmetros: k = 3 dá 4,0 mm/mês (acima de 3 -> sem marca); k = 6 dá 2,0 mm/mês (abaixo de 3 -> marca)
    assert not _baixa(cfg, 3.0, 0.25, 2.0, 8.0, k=3)[2].any()
    assert _baixa(cfg, 3.0, 0.25, 2.0, 8.0, k=6)[2].all()


def test_zeros_do_treino_reduzem_a_media_esperada(cfg):
    # mesma Gama (forma 2, escala 10 -> 20 mm por mês de janela em k=1... aqui k=3): q = 0 dá 6,67; q = 0,6 dá 2,67
    assert not _baixa(cfg, 5.0, 0.0, 2.0, 10.0)[2].any()
    assert _baixa(cfg, 5.0, 0.6, 2.0, 10.0)[2].all()


def test_limite_zero_desliga_a_regra(cfg):
    assert not _baixa(cfg, 0.0, 0.0, 2.0, 0.001)[2].any()            # chuva esperada ínfima, mas a regra está desligada


def test_celula_invalida_nunca_e_baixa_confianca(cfg):
    spi, inval, baixa = _baixa(cfg, 50.0, 0.0, 2.0, 0.5, valido=False)
    assert inval.all() and np.isnan(spi).all() and not baixa.any()


def test_aplicar_recusa_parametros_de_mais_de_uma_escala(cfg):
    S = pd.DataFrame({"A": np.arange(1.0, 25.0)}, index=meses("2001-01", "2002-12"))
    p = pd.concat([params_manual("A", 0.0, 2.0, 10.0), params_manual("A", 0.0, 2.0, 10.0).assign(escala_meses=np.int8(6))])
    with pytest.raises(sp.SpiError, match="única escala"):
        sp._aplicar(S, p, cfg)


def test_regra_so_acrescenta_marcas_e_nao_muda_o_spi(cfg):
    base = precip_sintetica(n_cod=3, seed=11)
    com = sp.calcular_spi(base, cfg)                                  # limite do params.yaml (10 mm por mês)
    sem = sp.calcular_spi(base, com_media_min(cfg, 0.0))
    for k in (3, 6):
        np.testing.assert_array_equal(com.spi[f"spi{k}"].to_numpy(), sem.spi[f"spi{k}"].to_numpy())
        assert (com.spi[f"spi{k}_invalid"] == sem.spi[f"spi{k}_invalid"]).all()
        assert (com.spi[f"spi{k}_baixa_conf"] | ~sem.spi[f"spi{k}_baixa_conf"]).all()          # sem => com
    assert com.spi["spi3_baixa_conf"].sum() > sem.spi["spi3_baixa_conf"].sum()
    marcado = com.spi.loc[com.spi["spi3_baixa_conf"] & ~sem.spi["spi3_baixa_conf"], "mes"].dt.month
    assert set(marcado) <= {5, 6, 7, 8, 9, 10, 11}                  # só meses secos; a estação chuvosa não é marcada


def test_marca_por_pouca_chuva_depende_so_do_treino(cfg):
    """Anti-vazamento: inflar a chuva depois do treino não pode mudar nenhuma marca."""
    base = precip_sintetica(n_cod=2, seed=3)
    alterado = base.copy()
    alterado.loc[alterado["mes"] > pd.Timestamp("2015-12-01"), "precip_mm"] *= 100.0
    a, b = sp.calcular_spi(base, cfg), sp.calcular_spi(alterado, cfg)
    for k in (3, 6):
        assert (a.spi[f"spi{k}_baixa_conf"] == b.spi[f"spi{k}_baixa_conf"]).all()


def test_media_derivada_dos_parametros_bate_com_a_media_amostral_do_treino(cfg):
    """A regra usa (1 - q) * forma * escala no lugar da média do treino: no MLE com loc = 0 elas coincidem."""
    base = precip_sintetica(n_cod=2, seed=21)
    res = sp.calcular_spi(base, cfg)
    S = sp.acumular(sp.para_matriz(base), 3)
    treino = S[S.index <= cfg.split.treino_fim]
    p = res.parametros
    p = p[(p["escala_meses"] == 3) & p["valido"]]
    assert len(p) > 0
    for _, r in p.iterrows():
        x = treino.loc[treino.index.month == r["mes_calendario"], r["cod_ibge"]].dropna()
        assert (1 - r["q"]) * r["shape"] * r["scale"] == pytest.approx(x.mean(), rel=1e-4)
