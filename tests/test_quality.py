"""Testes de src/data/quality.py (Item 3). Sem internet; séries sintéticas só aqui (nunca em data/).

Janela real do params.yaml: 1981-01..2025-12 (540 meses). Treino até 2015-12 (420 meses).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pandas.testing import assert_frame_equal

from src.data import quality as qd
from src.data import spi as sp
from src.data.config import PendenciaError, load_config

PARAMS_REAL = Path(__file__).resolve().parents[1] / "params.yaml"
MESES = pd.period_range("1981-01", "2025-12", freq="M")
N_TREINO = 420          # 1981-01..2015-12
A, B = "2200001", "2200002"


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


def cfg_variante(tmp_path, **ajustes_qualidade):
    d = yaml.safe_load(PARAMS_REAL.read_text(encoding="utf-8"))
    d["qualidade"].update(ajustes_qualidade)
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return load_config(tmp_path / "params.yaml")


def raw_sintetico(cods=(A, B), seed=0) -> pd.DataFrame:
    """RAW completo e perfeito (cobertura 100%): grade município x mês de toda a janela."""
    rng = np.random.default_rng(seed)
    quadros = []
    for cod in cods:
        p = rng.gamma(2.0, 60.0, len(MESES)).astype("float32")
        p[rng.random(len(MESES)) < 0.1] = 0.0
        quadros.append(pd.DataFrame({
            "cod_ibge": cod, "mes": MESES.to_timestamp(), "precip_mm": p,
            "precip_max_dia_mm": (p * 0.4).astype("float32"),
            "dias_chuva": np.minimum(MESES.days_in_month, np.floor(p / 8)).astype("float32"),
            "dias_validos": MESES.days_in_month.astype("int16"),
            "dias_no_mes": MESES.days_in_month.astype("int16"),
            "pixels_validos_frac": np.float32(1.0)}))
    return pd.concat(quadros, ignore_index=True)


def sel(df, cod, ini, fim=None) -> np.ndarray:
    ini = pd.Period(ini, "M").to_timestamp()
    fim = pd.Period(fim or str(pd.Period(ini, "M")), "M").to_timestamp()
    return ((df["cod_ibge"] == cod) & (df["mes"] >= ini) & (df["mes"] <= fim)).to_numpy()


def sem_dado(df, cod, ini, fim=None):
    """Mês sem nenhum dado do satélite (como o extrator grava: trio NaN, 0 dias, 0 pixels)."""
    m = sel(df, cod, ini, fim)
    df.loc[m, ["precip_mm", "precip_max_dia_mm", "dias_chuva"]] = np.nan
    df.loc[m, "dias_validos"] = 0
    df.loc[m, "pixels_validos_frac"] = np.float32(0.0)
    return df


def linha(df, cod, mes):
    return df[sel(df, cod, mes)].iloc[0]


# ================================================================== cobertura
def test_raw_perfeito_todos_validos(cfg):
    cob = qd.cobertura_mensal(raw_sintetico(), cfg)
    assert cob["valido"].all()
    assert (cob["motivo"] == "").all()
    assert (cob["cobertura_dias"] == 1.0).all() and (cob["cobertura_pixels"] == 1.0).all()
    assert len(cob) == 2 * len(MESES)


def test_schema_e_dtypes_da_cobertura(cfg):
    cob = qd.cobertura_mensal(raw_sintetico(), cfg)
    assert list(cob.columns) == qd.COLUNAS_COBERTURA
    assert cob["mes"].dtype == "datetime64[ns]"
    assert cob["cobertura_dias"].dtype == "float32" and cob["cobertura_pixels"].dtype == "float32"
    assert cob["valido"].dtype == bool
    assert cob["cod_ibge"].tolist()[:1] == [A]


def test_cobertura_ordenada_e_independente_da_ordem_de_entrada(cfg):
    raw = sem_dado(raw_sintetico(), A, "2000-06")
    embaralhado = raw.sample(frac=1.0, random_state=3)
    assert_frame_equal(qd.cobertura_mensal(raw, cfg), qd.cobertura_mensal(embaralhado, cfg))
    cob = qd.cobertura_mensal(embaralhado, cfg)
    assert cob.sort_values(["cod_ibge", "mes"]).index.equals(cob.index)


def test_dias_limiar_inclusivo_e_um_dia_a_menos_invalida(cfg, tmp_path):
    raw = raw_sintetico()
    # abril tem 30 dias: 29/30 reprova o limiar 1.0 padrão
    raw.loc[sel(raw, A, "2000-04"), "dias_validos"] = 29
    c = linha(qd.cobertura_mensal(raw, cfg), A, "2000-04")
    assert not c["valido"] and c["motivo"] == "cobertura_dias"
    # com limiar 0.9: 27/30 = 0.9 exato é válido (inclusivo) e 26/30 não
    c9 = cfg_variante(tmp_path, min_cobertura_dias_mes=0.9)
    raw.loc[sel(raw, A, "2000-04"), "dias_validos"] = 27
    raw.loc[sel(raw, A, "2001-04"), "dias_validos"] = 26
    cob = qd.cobertura_mensal(raw, c9)
    assert linha(cob, A, "2000-04")["valido"]
    assert not linha(cob, A, "2001-04")["valido"]


def test_pixels_limiar_inclusivo(cfg):
    raw = raw_sintetico()
    raw.loc[sel(raw, A, "2000-01"), "pixels_validos_frac"] = np.float32(0.8)
    raw.loc[sel(raw, A, "2000-02"), "pixels_validos_frac"] = np.float32(0.79)
    cob = qd.cobertura_mensal(raw, cfg)
    assert linha(cob, A, "2000-01")["valido"]
    c = linha(cob, A, "2000-02")
    assert not c["valido"] and c["motivo"] == "cobertura_pixels"


def test_pixels_float32_nao_reprova_por_arredondamento(tmp_path):
    """float32(0.7) convertido para float64 é < 0.7; comparar em float64 reprovaria 7/10 pixels no limiar 0.7."""
    assert float(np.float32(0.7)) < 0.7
    c7 = cfg_variante(tmp_path, min_cobertura_pixels=0.7)
    raw = raw_sintetico()
    raw.loc[sel(raw, A, "2000-01"), "pixels_validos_frac"] = np.float32(7 / 10)
    raw.loc[sel(raw, A, "2000-02"), "pixels_validos_frac"] = np.float32(6 / 10)
    cob = qd.cobertura_mensal(raw, c7)
    assert linha(cob, A, "2000-01")["valido"]
    assert not linha(cob, A, "2000-02")["valido"]


def test_precip_nan_com_cobertura_perfeita_so_sem_dado(cfg):
    raw = raw_sintetico()
    raw.loc[sel(raw, A, "2000-05"), "precip_mm"] = np.nan
    c = linha(qd.cobertura_mensal(raw, cfg), A, "2000-05")
    assert not c["valido"] and c["motivo"] == "sem_dado"


def test_mes_sem_nenhum_dado_lista_os_tres_motivos(cfg):
    raw = sem_dado(raw_sintetico(), A, "2000-05")
    c = linha(qd.cobertura_mensal(raw, cfg), A, "2000-05")
    assert not c["valido"]
    assert c["motivo"] == "sem_dado+cobertura_dias+cobertura_pixels"
    assert c["cobertura_dias"] == 0.0 and c["cobertura_pixels"] == 0.0


def test_zero_mm_real_e_valido(cfg):
    raw = raw_sintetico()
    for col in ("precip_mm", "precip_max_dia_mm", "dias_chuva"):
        raw.loc[sel(raw, A, "2000-07"), col] = np.float32(0.0)
    cob = qd.cobertura_mensal(raw, cfg)
    assert linha(cob, A, "2000-07")["valido"]          # 0 mm real != NaN
    assert linha(qd.serie_valida(raw, cfg), A, "2000-07")["precip_mm"] == 0.0


def test_cobertura_nao_altera_a_entrada(cfg):
    raw = sem_dado(raw_sintetico(), A, "2000-06")
    antes = raw.copy(deep=True)
    qd.avaliar_qualidade(raw, cfg)
    assert_frame_equal(raw, antes)


# ================================================================== entrada inválida
def _com(raw, **mud):
    raw = raw.copy()
    for k, v in mud.items():
        raw[k] = v
    return raw


def test_entrada_nao_dataframe(cfg):
    with pytest.raises(qd.QualidadeError, match="DataFrame"):
        qd.cobertura_mensal([1, 2, 3], cfg)


def test_coluna_ausente(cfg):
    with pytest.raises(qd.QualidadeError, match="colunas ausentes"):
        qd.cobertura_mensal(raw_sintetico().drop(columns=["dias_chuva"]), cfg)


def test_dataframe_vazio(cfg):
    with pytest.raises(qd.QualidadeError, match="vazio"):
        qd.cobertura_mensal(raw_sintetico().iloc[0:0], cfg)


def test_mes_que_nao_e_dia_1(cfg):
    raw = raw_sintetico()
    raw.loc[0, "mes"] = raw.loc[0, "mes"] + pd.Timedelta(days=3)
    with pytest.raises(qd.QualidadeError, match="dia 1"):
        qd.cobertura_mensal(raw, cfg)


def test_mes_fora_da_janela(cfg):
    raw = raw_sintetico()
    extra = raw.iloc[[0]].copy()
    extra["mes"] = pd.Timestamp("2026-01-01")
    with pytest.raises(qd.QualidadeError, match="fora da janela"):
        qd.cobertura_mensal(pd.concat([raw, extra], ignore_index=True), cfg)


def test_duplicata(cfg):
    raw = raw_sintetico()
    with pytest.raises(qd.QualidadeError, match="duplicatas"):
        qd.cobertura_mensal(pd.concat([raw, raw.iloc[[5]]], ignore_index=True), cfg)


def test_grade_incompleta(cfg):
    raw = raw_sintetico()
    with pytest.raises(qd.QualidadeError, match="grade incompleta"):
        qd.cobertura_mensal(raw.drop(index=10), cfg)


@pytest.mark.parametrize("cod", ["220000", "22000001", "abc1234", "nan"])
def test_codigo_fora_do_formato(cfg, cod):
    raw = raw_sintetico()
    raw["cod_ibge"] = raw["cod_ibge"].where(raw["cod_ibge"] != B, cod)
    with pytest.raises(qd.QualidadeError, match="7 dígitos"):
        qd.cobertura_mensal(raw, cfg)


def test_dias_validos_maior_que_dias_no_mes(cfg):
    raw = raw_sintetico()
    raw.loc[3, "dias_validos"] = raw.loc[3, "dias_no_mes"] + 1
    with pytest.raises(qd.QualidadeError, match="dias_validos"):
        qd.cobertura_mensal(raw, cfg)


def test_dias_no_mes_zero(cfg):
    raw = raw_sintetico()
    raw.loc[3, "dias_no_mes"] = 0
    raw.loc[3, "dias_validos"] = 0
    with pytest.raises(qd.QualidadeError, match="dias_no_mes"):
        qd.cobertura_mensal(raw, cfg)


@pytest.mark.parametrize("valor", [np.nan, -0.1, 1.1])
def test_pixels_frac_invalida(cfg, valor):
    raw = raw_sintetico()
    raw.loc[3, "pixels_validos_frac"] = np.float32(valor)
    with pytest.raises(qd.QualidadeError, match="pixels_validos_frac"):
        qd.cobertura_mensal(raw, cfg)


@pytest.mark.parametrize("valor", [-1.0, np.inf])
def test_precip_negativa_ou_infinita(cfg, valor):
    raw = raw_sintetico()
    raw.loc[3, "precip_mm"] = np.float32(valor)
    with pytest.raises(qd.QualidadeError, match="negativo ou infinito"):
        qd.cobertura_mensal(raw, cfg)


def test_gate_de_pendencia_na_qualidade(cfg):
    pendente = cfg.model_copy(update={"qualidade": cfg.qualidade.model_copy(
        update={"min_cobertura_pixels": "VERIFICAR"})})
    with pytest.raises(PendenciaError, match="qualidade"):
        qd.cobertura_mensal(raw_sintetico(), pendente)


# ================================================================== serie_valida
def test_serie_valida_nan_em_mes_invalido_e_preserva_o_resto(cfg):
    raw = raw_sintetico()
    raw.loc[sel(raw, A, "2000-03"), "dias_validos"] = 20      # incompleto: vale NaN nas 3 colunas
    raw = sem_dado(raw, B, "1999-12")
    v = qd.serie_valida(raw, cfg)
    for cod, mes in ((A, "2000-03"), (B, "1999-12")):
        r = linha(v, cod, mes)
        assert r[["precip_mm", "precip_max_dia_mm", "dias_chuva"]].isna().all()
    ok = linha(v, A, "2000-04")
    orig = linha(raw, A, "2000-04")
    assert ok["precip_mm"] == orig["precip_mm"] and ok["precip_max_dia_mm"] == orig["precip_max_dia_mm"]
    assert v["precip_mm"].isna().sum() == 2
    assert list(v.columns) == qd.COLUNAS_SERIE
    assert all(v[c].dtype == "float32" for c in qd.COLUNAS_CHUVA)


def test_serie_valida_nunca_vira_zero(cfg):
    raw = sem_dado(raw_sintetico(), A, "2000-01", "2000-12")
    v = qd.serie_valida(raw, cfg)
    assert v.loc[sel(v, A, "2000-01", "2000-12"), "precip_mm"].isna().all()


# ================================================================== climatologia
def test_climatologia_bate_com_conta_manual(cfg):
    raw = raw_sintetico()
    raw = sem_dado(raw, A, "1990-03")                    # mês inválido não entra na mediana de março
    v = qd.serie_valida(raw, cfg)
    clim = qd.climatologia(v, cfg).set_index(["cod_ibge", "mes_calendario"])
    tr = v[(v["cod_ibge"] == A) & (v["mes"] <= "2015-12-01") & (v["mes"].dt.month == 3)]
    assert len(tr.dropna(subset=["precip_mm"])) == 34
    esperado = np.float32(np.median(tr["precip_mm"].dropna().astype("float64")))
    assert clim.loc[(A, 3), "mediana_mm"] == esperado
    assert clim.loc[(A, 3), "n_amostras"] == 34
    assert clim.loc[(A, 4), "n_amostras"] == 35


def test_climatologia_inclui_zeros_na_mediana(cfg):
    raw = raw_sintetico()
    ano = raw["mes"].dt.year
    for col in ("precip_mm", "precip_max_dia_mm", "dias_chuva"):
        raw.loc[(raw["cod_ibge"] == A) & (raw["mes"].dt.month == 8) & (ano <= 2015), col] = np.float32(0.0)
    clim = qd.climatologia(qd.serie_valida(raw, cfg), cfg).set_index(["cod_ibge", "mes_calendario"])
    assert clim.loc[(A, 8), "mediana_mm"] == 0.0
    assert clim.loc[(A, 8), "n_amostras"] == 35


def test_climatologia_so_usa_o_treino_corte_exato(cfg):
    raw = raw_sintetico()
    base = qd.climatologia(qd.serie_valida(raw, cfg), cfg)
    # 2016-01 é o primeiro mês depois do corte: nada que aconteça de 2016-01 em diante muda a climatologia
    fora = raw.copy()
    fora.loc[sel(fora, A, "2016-01", "2025-12"), "precip_mm"] = np.float32(1999.0)
    assert_frame_equal(qd.climatologia(qd.serie_valida(fora, cfg), cfg), base)
    assert_frame_equal(qd.climatologia(qd.serie_valida(sem_dado(raw.copy(), A, "2016-01", "2025-12"), cfg), cfg), base)
    # 2015-12 é o último mês do treino: invalidá-lo tira uma amostra de dezembro (e só de dezembro) do município A
    dentro = qd.climatologia(qd.serie_valida(sem_dado(raw.copy(), A, "2015-12"), cfg), cfg)
    n0 = base.set_index(["cod_ibge", "mes_calendario"])["n_amostras"]
    n1 = dentro.set_index(["cod_ibge", "mes_calendario"])["n_amostras"]
    assert (n0 - n1)[(n0 - n1) != 0].to_dict() == {(A, 12): 1}


def test_climatologia_celula_sem_mes_valido_no_treino(cfg):
    raw = raw_sintetico()
    ano = raw["mes"].dt.year
    m = (raw["cod_ibge"] == A) & (raw["mes"].dt.month == 1) & (ano <= 2015)
    raw.loc[m, ["precip_mm", "precip_max_dia_mm", "dias_chuva"]] = np.nan
    raw.loc[m, "dias_validos"] = 0
    raw.loc[m, "pixels_validos_frac"] = np.float32(0.0)
    clim = qd.climatologia(qd.serie_valida(raw, cfg), cfg).set_index(["cod_ibge", "mes_calendario"])
    assert np.isnan(clim.loc[(A, 1), "mediana_mm"]) and clim.loc[(A, 1), "n_amostras"] == 0
    assert clim.loc[(B, 1), "n_amostras"] == 35


def test_climatologia_schema_12_linhas_por_municipio(cfg):
    clim = qd.climatologia(qd.serie_valida(raw_sintetico(), cfg), cfg)
    assert list(clim.columns) == qd.COLUNAS_CLIMATOLOGIA
    assert len(clim) == 24 and clim.groupby("cod_ibge").size().eq(12).all()
    assert clim["mes_calendario"].dtype == "int8" and clim["n_amostras"].dtype == "int16"
    assert all(clim[c].dtype == "float32" for c in ("mediana_mm", "mediana_max_dia_mm", "mediana_dias_chuva"))
    assert sorted(clim["mes_calendario"].unique()) == list(range(1, 13))


# ================================================================== imputar
def _clim_valida(raw, cfg):
    v = qd.serie_valida(raw, cfg)
    return v, qd.climatologia(v, cfg)


def _mediana_mm(clim, cod, mes_cal):
    return clim.set_index(["cod_ibge", "mes_calendario"]).loc[(cod, mes_cal), "mediana_mm"]


def test_sem_lacuna_a_serie_fica_identica_e_sem_flag(cfg):
    v, clim = _clim_valida(raw_sintetico(), cfg)
    imp = qd.imputar(v, clim, cfg)
    assert_frame_equal(imp[qd.COLUNAS_SERIE], v)
    assert not imp["precip_mm_missing"].any()
    assert list(imp.columns) == qd.COLUNAS_IMPUTADA and imp["precip_mm_missing"].dtype == bool


def test_um_mes_faltando_recebe_a_mediana_do_mes_calendario(cfg):
    v, clim = _clim_valida(sem_dado(raw_sintetico(), A, "2000-06"), cfg)
    imp = qd.imputar(v, clim, cfg)
    r = linha(imp, A, "2000-06")
    assert r["precip_mm"] == _mediana_mm(clim, A, 6)
    assert r["precip_mm_missing"]
    assert imp["precip_mm_missing"].sum() == 1
    # os três campos recebem a PRÓPRIA mediana e o máximo diário continua <= total do mês
    c = clim.set_index(["cod_ibge", "mes_calendario"]).loc[(A, 6)]
    assert r["precip_max_dia_mm"] == c["mediana_max_dia_mm"] and r["dias_chuva"] == c["mediana_dias_chuva"]
    assert r["precip_max_dia_mm"] <= r["precip_mm"]


def test_regra_dos_meses_consecutivos_para_no_terceiro(cfg):
    raw = sem_dado(raw_sintetico(), A, "2000-01", "2000-05")      # 5 meses seguidos sem dado
    v, clim = _clim_valida(raw, cfg)
    imp = qd.imputar(v, clim, cfg)
    r = imp[sel(imp, A, "2000-01", "2000-06")]
    assert r["precip_mm"].notna().tolist() == [True, True, False, False, False, True]   # 1º e 2º preenchidos
    assert r["precip_mm_missing"].tolist() == [True, True, True, True, True, False]
    assert r["precip_mm"].iloc[5] == linha(v, A, "2000-06")["precip_mm"]


def test_contador_reinicia_depois_de_um_mes_valido(cfg):
    raw = sem_dado(raw_sintetico(), A, "2000-01", "2000-03")      # 3 seguidos: o 3º fica NaN
    raw = sem_dado(raw, A, "2000-05", "2000-06")                  # 2 seguidos depois de um mês válido: os dois preenchidos
    v, clim = _clim_valida(raw, cfg)
    imp = qd.imputar(v, clim, cfg)
    r = imp[sel(imp, A, "2000-01", "2000-06")]
    assert r["precip_mm"].notna().tolist() == [True, True, False, True, True, True]


def test_limite_vem_da_configuracao(cfg, tmp_path):
    raw = sem_dado(raw_sintetico(), A, "2000-01", "2000-05")
    v, clim = _clim_valida(raw, cfg)
    zero = qd.imputar(v, clim, cfg_variante(tmp_path, max_meses_imputados_consecutivos=0))
    assert zero["precip_mm"].isna().sum() == 5
    quatro = qd.imputar(v, clim, cfg_variante(tmp_path, max_meses_imputados_consecutivos=4))
    assert quatro["precip_mm"].isna().sum() == 1


def test_lacuna_no_inicio_da_serie_usa_a_mediana_do_treino(cfg):
    v, clim = _clim_valida(sem_dado(raw_sintetico(), A, "1981-01", "1981-02"), cfg)
    imp = qd.imputar(v, clim, cfg)
    assert linha(imp, A, "1981-01")["precip_mm"] == _mediana_mm(clim, A, 1)
    assert linha(imp, A, "1981-02")["precip_mm"] == _mediana_mm(clim, A, 2)


def test_lacuna_na_validacao_e_no_teste_usa_a_climatologia_do_treino(cfg):
    raw = sem_dado(sem_dado(raw_sintetico(), A, "2018-05"), A, "2024-05")
    v, clim = _clim_valida(raw, cfg)
    imp = qd.imputar(v, clim, cfg)
    assert linha(imp, A, "2018-05")["precip_mm"] == _mediana_mm(clim, A, 5)
    assert linha(imp, A, "2024-05")["precip_mm"] == _mediana_mm(clim, A, 5)


def test_climatologia_sem_valor_deixa_nan(cfg):
    raw = raw_sintetico()
    ano = raw["mes"].dt.year
    m = (raw["cod_ibge"] == A) & (raw["mes"].dt.month == 1) & (ano <= 2015)
    raw.loc[m, ["precip_mm", "precip_max_dia_mm", "dias_chuva"]] = np.nan
    raw.loc[m, "dias_validos"] = 0
    raw.loc[m, "pixels_validos_frac"] = np.float32(0.0)
    v, clim = _clim_valida(raw, cfg)
    imp = qd.imputar(v, clim, cfg)
    assert np.isnan(linha(imp, A, "1990-01")["precip_mm"]) and linha(imp, A, "1990-01")["precip_mm_missing"]


def test_municipios_sao_independentes(cfg):
    v, clim = _clim_valida(sem_dado(raw_sintetico(), A, "2000-01", "2000-03"), cfg)
    imp = qd.imputar(v, clim, cfg)
    assert not imp.loc[imp["cod_ibge"] == B, "precip_mm_missing"].any()
    assert_frame_equal(imp[imp["cod_ibge"] == B][qd.COLUNAS_SERIE].reset_index(drop=True),
                       v[v["cod_ibge"] == B].reset_index(drop=True))


def test_imputacao_e_causal_futuro_nao_muda_o_passado(cfg):
    """CA-06: alterar o que vem depois do mês t não pode mudar nenhuma linha em <= t."""
    rng = np.random.default_rng(11)
    raw = raw_sintetico()
    for cod in (A, B):                                       # lacunas aleatórias de 1 a 4 meses
        for ini in rng.choice(len(MESES) - 6, 25, replace=False):
            sem_dado(raw, cod, str(MESES[ini]), str(MESES[ini + int(rng.integers(0, 4))]))
    v, clim = _clim_valida(raw, cfg)
    base = qd.imputar(v, clim, cfg)
    for corte in ("1990-06", "2010-01", "2016-03", "2023-09"):
        mod = v.copy()
        fut = (mod["mes"] > pd.Timestamp(corte)).to_numpy()
        sorteio = rng.random(len(mod)) < 0.5
        for c in qd.COLUNAS_CHUVA:
            mod.loc[fut, c] = np.where(sorteio[fut], np.nan, np.float32(rng.uniform(1, 300)))
        novo = qd.imputar(mod, clim, cfg)
        passado = (base["mes"] <= pd.Timestamp(corte)).to_numpy()
        assert_frame_equal(base[passado].reset_index(drop=True), novo[passado].reset_index(drop=True))


def test_imputar_nao_altera_a_entrada(cfg):
    v, clim = _clim_valida(sem_dado(raw_sintetico(), A, "2000-06"), cfg)
    antes, clim_antes = v.copy(deep=True), clim.copy(deep=True)
    qd.imputar(v, clim, cfg)
    assert_frame_equal(v, antes)
    assert_frame_equal(clim, clim_antes)


def test_imputar_exige_colunas_da_climatologia(cfg):
    v, clim = _clim_valida(raw_sintetico(), cfg)
    with pytest.raises(qd.QualidadeError, match="climatologia sem colunas"):
        qd.imputar(v, clim.drop(columns=["mediana_mm"]), cfg)


# ================================================================== qualidade_municipio
def _municipios(raw, cfg):
    return qd.qualidade_municipio(qd.cobertura_mensal(raw, cfg), cfg).set_index("cod_ibge")


def _invalidar_n_meses_de_treino(raw, cod, n):
    """Torna inválidos os primeiros `n` meses (1981-01 em diante), todos dentro do treino."""
    return sem_dado(raw, cod, "1981-01", str(MESES[n - 1]))


def test_municipio_perfeito(cfg):
    m = _municipios(raw_sintetico(), cfg)
    assert (m["frac_meses_validos"] == 1.0).all() and not m["excluded"].any() and (m["motivo"] == "").all()
    assert list(qd.qualidade_municipio(qd.cobertura_mensal(raw_sintetico(), cfg), cfg).columns) == qd.COLUNAS_MUNICIPIO


def test_exclusao_no_limite_exato_de_95_por_cento(cfg):
    # 420 meses de treino: 399 válidos = 0.95 exato (NÃO é "abaixo"); 398 válidos exclui
    m = _municipios(_invalidar_n_meses_de_treino(raw_sintetico(), A, 21), cfg)
    assert m.loc[A, "frac_meses_validos"] == np.float32(0.95) and not m.loc[A, "excluded"]
    m = _municipios(_invalidar_n_meses_de_treino(raw_sintetico(), A, 22), cfg)
    assert m.loc[A, "excluded"] and not m.loc[B, "excluded"]
    assert "0.948" in m.loc[A, "motivo"] and "0.95" in m.loc[A, "motivo"]


def test_exclusao_so_olha_o_treino(cfg):
    raw = sem_dado(raw_sintetico(), A, "2016-01", "2025-12")        # 120 meses ruins, todos depois do treino
    m = _municipios(raw, cfg)
    assert m.loc[A, "frac_meses_validos"] == 1.0 and not m.loc[A, "excluded"]


def test_corte_do_treino_inclui_2015_12_e_exclui_2016_01(cfg):
    ruim_no_corte = _municipios(sem_dado(raw_sintetico(), A, "2015-12"), cfg)
    assert ruim_no_corte.loc[A, "frac_meses_validos"] == np.float32(419 / 420)
    ruim_depois = _municipios(sem_dado(raw_sintetico(), A, "2016-01"), cfg)
    assert ruim_depois.loc[A, "frac_meses_validos"] == 1.0


def test_linha_ausente_conta_como_invalida(cfg):
    cob = qd.cobertura_mensal(raw_sintetico(), cfg)
    cob = cob[~sel(cob, A, "1981-01", "2015-12")]                     # município A sem nenhuma linha de treino
    m = qd.qualidade_municipio(cob, cfg).set_index("cod_ibge")
    assert m.loc[A, "frac_meses_validos"] == 0.0 and m.loc[A, "excluded"]


def test_municipio_ordenado_e_um_por_codigo(cfg):
    m = qd.qualidade_municipio(qd.cobertura_mensal(raw_sintetico(cods=(B, A)), cfg), cfg)
    assert m["cod_ibge"].tolist() == [A, B]
    assert m["frac_meses_validos"].dtype == "float32" and m["excluded"].dtype == bool


def test_qualidade_municipio_exige_colunas(cfg):
    with pytest.raises(qd.QualidadeError, match="cobertura sem colunas"):
        qd.qualidade_municipio(pd.DataFrame({"cod_ibge": [A]}), cfg)


# ================================================================== integração
def test_avaliar_qualidade_e_deterministico_e_consistente(cfg):
    raw = sem_dado(raw_sintetico(), A, "2000-06", "2000-08")
    r1, r2 = qd.avaliar_qualidade(raw, cfg), qd.avaliar_qualidade(raw, cfg)
    for nome in ("cobertura", "serie_valida", "serie_imputada", "climatologia", "municipios"):
        assert_frame_equal(getattr(r1, nome), getattr(r2, nome))
    assert_frame_equal(r1.serie_valida, qd.serie_valida(raw, cfg))
    assert_frame_equal(r1.cobertura, qd.cobertura_mensal(raw, cfg))
    assert r1.cobertura["valido"].sum() == len(raw) - 3
    assert r1.serie_imputada["precip_mm_missing"].sum() == 3
    assert r1.serie_imputada["precip_mm"].isna().sum() == 1          # 3 meses seguidos: o 3º não é preenchido


def test_spi_usa_a_serie_valida_e_nao_a_imputada(cfg):
    """D9: mês sem dado vira SPI NaN ('sem dado'); a imputação não entra no cálculo do SPI."""
    raw = sem_dado(raw_sintetico(), A, "2000-06")
    r = qd.avaliar_qualidade(raw, cfg)
    spi_ok = sp.calcular_spi(r.serie_valida[["cod_ibge", "mes", "precip_mm"]], cfg).spi
    spi_ok = spi_ok[spi_ok["cod_ibge"] == A].set_index("mes")
    for mes in ("2000-06-01", "2000-07-01", "2000-08-01"):           # janelas de 3 meses que contêm junho
        assert np.isnan(spi_ok.loc[pd.Timestamp(mes), "spi3"])
    assert not np.isnan(spi_ok.loc[pd.Timestamp("2000-09-01"), "spi3"])
    # a série imputada teria 'preenchido' o buraco: o contraste prova que não é ela que alimenta o SPI
    spi_imp = sp.calcular_spi(r.serie_imputada[["cod_ibge", "mes", "precip_mm"]], cfg).spi
    spi_imp = spi_imp[spi_imp["cod_ibge"] == A].set_index("mes")
    assert not np.isnan(spi_imp.loc[pd.Timestamp("2000-06-01"), "spi3"])


def test_escala_real_216_municipios(cfg):
    cods = tuple(f"22{i:05d}" for i in range(216))
    r = qd.avaliar_qualidade(raw_sintetico(cods=cods, seed=2), cfg)
    assert len(r.cobertura) == 216 * 540 and len(r.municipios) == 216 and len(r.climatologia) == 216 * 12
    assert r.cobertura["valido"].all() and not r.municipios["excluded"].any()
