"""Testes de src/data/build_demo.py (Item 10). Sem internet; séries sintéticas só aqui (nunca em data/).

Janela real do params.yaml: 1981-01..2026-09 (549 meses). Treino até 2015-12 (420 meses).
"""
import ast
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data import build_demo as bd
from src.data import chirps_raw, manifest as mf, quality, reference, spi, splits
from src.data.config import Caminhos, load_config

RAIZ = Path(__file__).resolve().parents[1]
PARAMS_REAL = RAIZ / "params.yaml"
MESES = pd.period_range("1981-01", "2026-09", freq="M")
N_MESES = len(MESES)
A, B, C = "2200001", "2200002", "2200003"
AGORA = datetime(2026, 11, 17, 12, 30, 45, tzinfo=timezone.utc)


# ------------------------------------------------------------------ dados sintéticos
def raw_sintetico(cods=(A, B, C), seed=0) -> pd.DataFrame:
    """RAW completo e perfeito (cobertura 100%): grade município x mês de toda a janela."""
    rng = np.random.default_rng(seed)
    quadros = []
    for cod in cods:
        p = rng.gamma(2.0, 60.0, N_MESES).astype("float32")
        p[rng.random(N_MESES) < 0.1] = 0.0
        quadros.append(pd.DataFrame({
            "cod_ibge": cod, "mes": MESES.to_timestamp(), "precip_mm": p,
            "precip_max_dia_mm": (p * 0.4).astype("float32"),
            "dias_chuva": np.minimum(MESES.days_in_month, np.floor(p / 8)).astype("float32"),
            "dias_validos": MESES.days_in_month.astype("int16"),
            "dias_no_mes": MESES.days_in_month.astype("int16"),
            "pixels_validos_frac": np.float32(1.0)}))
    return pd.concat(quadros, ignore_index=True)


def sem_dado(df, cod, ini, fim=None):
    """Mês sem nenhum dado do satélite (como o extrator grava: trio NaN, 0 dias, 0 pixels)."""
    ini = pd.Period(ini, "M").to_timestamp()
    fim = pd.Period(fim or str(pd.Period(ini, "M")), "M").to_timestamp()
    m = ((df["cod_ibge"] == cod) & (df["mes"] >= ini) & (df["mes"] <= fim)).to_numpy()
    df.loc[m, ["precip_mm", "precip_max_dia_mm", "dias_chuva"]] = np.nan
    df.loc[m, "dias_validos"] = 0
    df.loc[m, "pixels_validos_frac"] = np.float32(0.0)
    return df


def multiplicar_depois(df, a_partir_de, fator):
    """Muda a chuva dos meses >= `a_partir_de` (para provar que o futuro não vaza para o treino)."""
    m = (df["mes"] >= pd.Period(a_partir_de, "M").to_timestamp()).to_numpy()
    df.loc[m, "precip_mm"] = (df.loc[m, "precip_mm"] * fator).astype("float32")
    df.loc[m, "precip_max_dia_mm"] = (df.loc[m, "precip_max_dia_mm"] * fator).astype("float32")
    return df


def linha(df, cod, mes):
    return df[(df["cod_ibge"] == cod) & (df["mes"] == pd.Period(mes, "M").to_timestamp())].iloc[0]


def escrever_raw(raiz: Path, raw: pd.DataFrame, cods, cortes=("1981-01", "2000-12", "2001-01", "2026-09")) -> None:
    """Grava o RAW em dois lotes imutáveis, como o extrator faria."""
    cfg = load_config(raiz / "params.yaml")
    d = chirps_raw.diretorio_raw(cfg)
    for ini, fim in ((cortes[0], cortes[1]), (cortes[2], cortes[3])):
        i, f = pd.Period(ini, "M"), pd.Period(fim, "M")
        m = (raw["mes"] >= i.to_timestamp()) & (raw["mes"] <= f.to_timestamp())
        chirps_raw.gravar_lote(raw[m].reset_index(drop=True), d, cods, i, f)


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def raw_base():
    return raw_sintetico()


@pytest.fixture(scope="module")
def res_base(cfg, raw_base):
    return bd.montar_demo(raw_base, cfg)


@pytest.fixture
def proj(tmp_path):
    """Projeto de mentira: params.yaml real, selo da referência e RAW de 3 municípios em 2 lotes."""
    shutil.copy(PARAMS_REAL, tmp_path / "params.yaml")
    ref = tmp_path / "data" / "reference"
    ref.mkdir(parents=True)
    (ref / "referencia.sha256.json").write_text(json.dumps(
        {"conteudo_sha256": "ab" * 32, "contagens": {"malha": 224, "escopo": 3},
         "entradas": {"PI_Municipios_2022.zip": "cd" * 32}, "parametros": {"incluir_transicao": True}}),
        encoding="utf-8")
    (ref / "municipios_semiarido.csv").write_bytes(
        "cod_ibge,nome,uf,situacao,area_km2\n2200001,Aaa,PI,res150,10.0\n".encode("utf-8"))
    pd.DataFrame({"CD_MUN": [A], "geometry": [b"wkb"]}).to_parquet(ref / "malha_pi_2022.parquet", index=False)
    (ref / "malha_pi_2022_gee.geojson").write_bytes(b'{"type": "FeatureCollection", "features": []}\n')
    cfg = load_config(tmp_path / "params.yaml")
    escrever_raw(tmp_path, raw_sintetico(), (A, B, C))
    return cfg, tmp_path


def rodar(proj, **kw):
    cfg, _ = proj
    return bd.construir_demo(cfg, municipios=(A, B, C), agora=AGORA, **kw)


# ================================================================== montar_demo: forma
def test_cinco_tabelas_com_os_nomes_do_contrato(res_base):
    assert tuple(bd.ARTEFATOS) == tuple(bd.ResultadoDemo.__dataclass_fields__)
    for nome in bd.ARTEFATOS:
        assert isinstance(getattr(res_base, nome), pd.DataFrame)


def test_colunas_do_demo_observado(res_base, cfg):
    esperado = ["cod_ibge", "mes"]
    for k in cfg.spi.janelas_meses:
        esperado += [f"spi{k}", f"spi{k}_invalid", f"spi{k}_baixa_conf"]
    assert list(res_base.demo_observado.columns) == esperado


def test_dtypes_do_demo_observado(res_base, cfg):
    d = res_base.demo_observado
    assert pd.api.types.is_string_dtype(d["cod_ibge"])
    assert d["mes"].dtype == np.dtype("datetime64[ns]")
    for k in cfg.spi.janelas_meses:
        assert d[f"spi{k}"].dtype == np.float32
        assert d[f"spi{k}_invalid"].dtype == bool
        assert d[f"spi{k}_baixa_conf"].dtype == bool


def test_colunas_e_dtypes_das_demais_tabelas(res_base):
    assert list(res_base.qualidade_municipio.columns) == quality.COLUNAS_MUNICIPIO
    assert list(res_base.climatologia_precip.columns) == quality.COLUNAS_CLIMATOLOGIA
    assert list(res_base.spi_gamma_params.columns) == spi.COLUNAS_PARAMETROS
    assert res_base.cobertura_mensal["valido"].dtype == bool
    assert res_base.spi_gamma_params["valido"].dtype == bool


def test_grade_completa_ordenada_e_sem_duplicata(res_base):
    d = res_base.demo_observado
    assert len(d) == 3 * N_MESES
    assert not d.duplicated(["cod_ibge", "mes"]).any()
    assert d.equals(d.sort_values(["cod_ibge", "mes"]).reset_index(drop=True))
    assert d["mes"].min() == pd.Timestamp("1981-01-01") and d["mes"].max() == pd.Timestamp("2026-09-01")
    assert sorted(d["cod_ibge"].unique()) == [A, B, C]


def test_grade_do_spi_igual_a_da_cobertura(res_base):
    chaves = ["cod_ibge", "mes"]
    assert res_base.demo_observado[chaves].equals(res_base.cobertura_mensal[chaves])


def test_indice_do_demo_e_simples(res_base):
    assert isinstance(res_base.demo_observado.index, pd.RangeIndex)
    assert isinstance(res_base.spi_gamma_params.index, pd.RangeIndex)


def test_sem_alvo_split_ou_metrica_nas_tabelas(res_base):
    """CA-12: a demonstração é observacional; alvo, bloco e métrica pertencem ao treino."""
    proibidos = ("alvo", "target", "split", "bloco", "train", "val", "test", "mae", "rmse", "mes_alvo")
    for nome in bd.ARTEFATOS:
        for col in getattr(res_base, nome).columns:
            assert not any(p == col or col.startswith(p + "_") for p in proibidos), (nome, col)


# ================================================================== montar_demo: conteúdo
def test_spi_igual_ao_calculo_direto_sem_exclusao(res_base, raw_base, cfg):
    q = quality.avaliar_qualidade(raw_base, cfg)
    assert not q.municipios["excluded"].any()
    direto = spi.calcular_spi(q.serie_valida[["cod_ibge", "mes", "precip_mm"]], cfg)
    assert_frame_equal(res_base.demo_observado, direto.spi, check_exact=True)
    assert_frame_equal(res_base.spi_gamma_params, direto.parametros.reset_index(drop=True), check_exact=True)


def test_qualidade_e_climatologia_iguais_as_do_modulo(res_base, raw_base, cfg):
    q = quality.avaliar_qualidade(raw_base, cfg)
    assert_frame_equal(res_base.cobertura_mensal, q.cobertura, check_exact=True)
    assert_frame_equal(res_base.qualidade_municipio, q.municipios, check_exact=True)
    assert_frame_equal(res_base.climatologia_precip, q.climatologia, check_exact=True)


def test_parametros_reproduzem_o_demo(res_base, raw_base, cfg):
    """Os artefatos são coerentes entre si: aplicar `spi_gamma_params` à série válida devolve o demo."""
    serie = quality.avaliar_qualidade(raw_base, cfg).serie_valida[["cod_ibge", "mes", "precip_mm"]]
    de_novo = spi.calcular_spi(serie, cfg, parametros=res_base.spi_gamma_params)
    assert_frame_equal(de_novo.spi, res_base.demo_observado, check_exact=True)


def test_primeiros_meses_da_janela_sao_invalidos(res_base, cfg):
    d = res_base.demo_observado
    for k in cfg.spi.janelas_meses:
        for cod in (A, B, C):
            sub = d[d["cod_ibge"] == cod].reset_index(drop=True)
            assert sub.loc[: k - 2, f"spi{k}_invalid"].all()
            assert sub.loc[: k - 2, f"spi{k}"].isna().all()
            assert not sub.loc[k - 1, f"spi{k}_invalid"]
            assert np.isfinite(sub.loc[k - 1, f"spi{k}"])


def test_spi_invalido_se_e_somente_se_nan(res_base, cfg):
    d = res_base.demo_observado
    for k in cfg.spi.janelas_meses:
        assert (d[f"spi{k}_invalid"] == d[f"spi{k}"].isna()).all()
        assert np.isfinite(d[f"spi{k}"].dropna()).all()


def test_spi_com_distribuicao_plausivel(res_base):
    d = res_base.demo_observado
    t = d[d["mes"] <= pd.Timestamp("2015-12-01")]["spi3"].dropna()
    assert abs(float(t.mean())) < 0.15 and 0.8 < float(t.std()) < 1.2     # in-sample: ~N(0,1)


def test_determinismo(cfg, raw_base, res_base):
    de_novo = bd.montar_demo(raw_base.copy(), cfg)
    for nome in bd.ARTEFATOS:
        assert_frame_equal(getattr(de_novo, nome), getattr(res_base, nome), check_exact=True)


def test_entrada_nao_e_mutada(cfg, raw_base):
    copia = raw_base.copy(deep=True)
    bd.montar_demo(raw_base, cfg)
    assert_frame_equal(raw_base, copia, check_exact=True)


def test_ordem_das_linhas_do_raw_nao_importa(cfg, raw_base, res_base):
    embaralhado = raw_base.sample(frac=1.0, random_state=1).reset_index(drop=True)
    de_novo = bd.montar_demo(embaralhado, cfg)
    for nome in bd.ARTEFATOS:
        assert_frame_equal(getattr(de_novo, nome), getattr(res_base, nome), check_exact=True)


def test_raw_invalido_propaga_erro_da_qualidade(cfg, raw_base):
    with pytest.raises(ValueError):
        bd.montar_demo(raw_base.drop(columns=["pixels_validos_frac"]), cfg)


def test_grade_inconsistente_levanta_demoerror(cfg, raw_base, monkeypatch):
    original = spi.calcular_spi

    def sem_uma_linha(*a, **k):
        r = original(*a, **k)
        return spi.ResultadoSpi(spi=r.spi.iloc[1:].reset_index(drop=True), parametros=r.parametros)
    monkeypatch.setattr(spi, "calcular_spi", sem_uma_linha)
    with pytest.raises(bd.DemoError, match="grade município x mês"):
        bd.montar_demo(raw_base, cfg)


# ================================================================== D9: lacuna não é imputada no SPI
def test_d9_lacuna_no_treino_fica_sem_spi_e_nao_e_imputada(cfg, raw_base):
    raw = sem_dado(raw_base.copy(), B, "2000-06")
    r = bd.montar_demo(raw, cfg)
    d = r.demo_observado
    assert not r.qualidade_municipio.set_index("cod_ibge").loc[B, "excluded"]
    for mes in ("2000-06", "2000-07", "2000-08"):                 # as 3 janelas SPI-3 que contêm a lacuna
        x = linha(d, B, mes)
        assert x["spi3_invalid"] and np.isnan(x["spi3"])
    assert not linha(d, B, "2000-09")["spi3_invalid"]
    assert not linha(d, B, "2000-05")["spi3_invalid"]
    for mes in ("2000-06", "2000-11"):                            # SPI-6: de 06/2000 a 11/2000
        assert linha(d, B, mes)["spi6_invalid"]
    assert not linha(d, B, "2000-12")["spi6_invalid"]
    # contraste: se a imputação (feature) entrasse no SPI, haveria valor em 2000-06
    q = quality.avaliar_qualidade(raw, cfg)
    imp = q.serie_imputada
    assert linha(imp, B, "2000-06")["precip_mm"] >= 0
    errado = spi.calcular_spi(imp[["cod_ibge", "mes", "precip_mm"]], cfg).spi
    assert np.isfinite(linha(errado, B, "2000-06")["spi3"])


def test_d9_lacuna_de_um_municipio_nao_altera_outro(cfg, raw_base, res_base):
    r = bd.montar_demo(sem_dado(raw_base.copy(), B, "2000-06"), cfg)
    for cod in (A, C):
        a = r.demo_observado[r.demo_observado["cod_ibge"] == cod].reset_index(drop=True)
        b = res_base.demo_observado[res_base.demo_observado["cod_ibge"] == cod].reset_index(drop=True)
        assert_frame_equal(a, b, check_exact=True)


# ================================================================== D11: município excluído
@pytest.fixture(scope="module")
def raw_com_exclusao(raw_base):
    """22 meses seguidos sem dado em 420 de treino => 94,76% < 95% => excluído.

    Uma corrida única (e não meses espalhados) mantém válidas as células da Gama de A: cada mês-calendário
    perde só 2 amostras (33 >= 30). Assim a regra D11 é a ÚNICA coisa que apaga o SPI de A."""
    return sem_dado(raw_base.copy(), A, "1985-01", "1986-10")


@pytest.fixture(scope="module")
def res_excl(cfg, raw_com_exclusao):
    return bd.montar_demo(raw_com_exclusao, cfg)


def test_d11_contraste_sem_a_regra_o_municipio_teria_spi(raw_com_exclusao, cfg):
    """Garante que o teste exercita a regra: sem D11 as células de A seriam válidas e o SPI existiria."""
    serie = quality.avaliar_qualidade(raw_com_exclusao, cfg).serie_valida[["cod_ibge", "mes", "precip_mm"]]
    cru = spi.calcular_spi(serie, cfg)
    a = cru.spi[cru.spi["cod_ibge"] == A]
    for k in cfg.spi.janelas_meses:
        assert a[f"spi{k}"].notna().sum() > 400
    assert cru.parametros.loc[cru.parametros["cod_ibge"] == A, "valido"].all()


def test_d11_excluido_fica_na_grade_e_marcado(res_excl):
    qm = res_excl.qualidade_municipio.set_index("cod_ibge")
    assert qm.loc[A, "excluded"] and not qm.loc[B, "excluded"] and not qm.loc[C, "excluded"]
    d = res_excl.demo_observado
    assert len(d) == 3 * N_MESES
    assert (d["cod_ibge"] == A).sum() == N_MESES


def test_d11_excluido_nao_tem_spi_nem_baixa_confianca(res_excl, cfg):
    a = res_excl.demo_observado[res_excl.demo_observado["cod_ibge"] == A]
    for k in cfg.spi.janelas_meses:
        assert a[f"spi{k}"].isna().all()
        assert a[f"spi{k}_invalid"].all()
        assert not a[f"spi{k}_baixa_conf"].any()


def test_d11_outros_municipios_iguais_a_execucao_sem_o_excluido(res_excl, raw_com_exclusao, cfg):
    sozinhos = bd.montar_demo(raw_com_exclusao[raw_com_exclusao["cod_ibge"] != A].reset_index(drop=True), cfg)
    for cod in (B, C):
        a = res_excl.demo_observado[res_excl.demo_observado["cod_ibge"] == cod].reset_index(drop=True)
        b = sozinhos.demo_observado[sozinhos.demo_observado["cod_ibge"] == cod].reset_index(drop=True)
        assert_frame_equal(a, b, check_exact=True)


def test_d11_parametros_do_excluido_ficam_invalidos(res_excl, cfg):
    p = res_excl.spi_gamma_params
    assert not p.loc[p["cod_ibge"] == A, "valido"].any()
    assert p.loc[p["cod_ibge"] != A, "valido"].all()
    assert len(p.loc[p["cod_ibge"] == A]) == 12 * len(cfg.spi.janelas_meses)


def test_d11_parametros_reproduzem_o_demo_com_excluido(res_excl, raw_com_exclusao, cfg):
    serie = quality.avaliar_qualidade(raw_com_exclusao, cfg).serie_valida[["cod_ibge", "mes", "precip_mm"]]
    de_novo = spi.calcular_spi(serie, cfg, parametros=res_excl.spi_gamma_params)
    assert_frame_equal(de_novo.spi, res_excl.demo_observado, check_exact=True)


def test_d11_cobertura_e_climatologia_do_excluido_continuam_publicadas(res_excl):
    assert (res_excl.cobertura_mensal["cod_ibge"] == A).sum() == N_MESES
    assert (res_excl.climatologia_precip["cod_ibge"] == A).sum() == 12
    c = res_excl.cobertura_mensal
    assert not c.loc[(c["cod_ibge"] == A) & (c["mes"] == pd.Timestamp("1986-03-01")), "valido"].iloc[0]


def test_exclusao_esta_no_limite_estrito_de_95_por_cento(cfg, raw_base):
    """21 meses sem dado em 420 = 95,0% exatos => NÃO exclui; 22 => exclui."""
    r21 = bd.montar_demo(sem_dado(raw_base.copy(), A, "1985-01", "1986-09"), cfg)
    r22 = bd.montar_demo(sem_dado(raw_base.copy(), A, "1985-01", "1986-10"), cfg)
    assert not r21.qualidade_municipio.set_index("cod_ibge").loc[A, "excluded"]
    assert r22.qualidade_municipio.set_index("cod_ibge").loc[A, "excluded"]
    assert r21.demo_observado.query("cod_ibge == @A")["spi3"].notna().any()


# ================================================================== anti-vazamento
def test_futuro_nao_muda_parametros_nem_climatologia_nem_spi_do_treino(cfg, raw_base, res_base):
    r = bd.montar_demo(multiplicar_depois(raw_base.copy(), "2016-01", 3.0), cfg)
    assert_frame_equal(r.spi_gamma_params, res_base.spi_gamma_params, check_exact=True)
    assert_frame_equal(r.climatologia_precip, res_base.climatologia_precip, check_exact=True)
    assert_frame_equal(r.qualidade_municipio, res_base.qualidade_municipio, check_exact=True)
    ate = pd.Timestamp("2015-12-01")
    a, b = r.demo_observado, res_base.demo_observado
    assert_frame_equal(a[a["mes"] <= ate].reset_index(drop=True),
                       b[b["mes"] <= ate].reset_index(drop=True), check_exact=True)
    depois = r.demo_observado["mes"] > pd.Timestamp("2016-06-01")
    assert not r.demo_observado.loc[depois, "spi3"].equals(res_base.demo_observado.loc[depois, "spi3"])


def test_nao_usa_nenhuma_funcao_do_teste_congelado(cfg, raw_base, monkeypatch):
    """CA-12: montar o demo não rotula, recorta nem libera o teste."""
    def proibido(*a, **k):
        raise AssertionError("o demo não pode tocar rotular/bloco/exigir_teste_liberado")
    for nome in ("rotular", "bloco", "exigir_teste_liberado"):
        monkeypatch.setattr(splits, nome, proibido)
    bd.montar_demo(raw_base, cfg)


# ================================================================== gravar_demo
def test_artefatos_apontam_para_pastas_do_params(proj):
    cfg, raiz = proj
    for nome, (pasta, arquivo) in bd.ARTEFATOS.items():
        assert pasta in Caminhos.model_fields
        assert bd.destino(cfg, nome) == cfg.caminho(pasta) / arquivo
    assert bd.destino(cfg, "demo_observado") == raiz / "data" / "curated" / "demo_observado.parquet"
    assert bd.destino(cfg, "spi_gamma_params") == raiz / "artifacts" / "spi_gamma_params.parquet"
    assert bd.destino(cfg, "climatologia_precip") == raiz / "artifacts" / "climatologia_precip.parquet"
    assert bd.destino(cfg, "cobertura_mensal") == raiz / "data" / "processed" / "cobertura_mensal.parquet"
    assert bd.destino(cfg, "qualidade_municipio") == raiz / "data" / "processed" / "qualidade_municipio.parquet"


def test_grava_cinco_arquivos_e_devolve_gerado(proj, res_base):
    cfg, _ = proj
    status = bd.gravar_demo(cfg, res_base)
    assert status == dict.fromkeys(bd.ARTEFATOS, "gerado")
    assert list(status) == list(bd.ARTEFATOS)
    assert all(bd.destino(cfg, n).is_file() for n in bd.ARTEFATOS)


def test_ida_e_volta_pelo_parquet_sem_perda(proj, res_base):
    cfg, _ = proj
    bd.gravar_demo(cfg, res_base)
    for nome in bd.ARTEFATOS:
        lido = pd.read_parquet(bd.destino(cfg, nome), engine="pyarrow")
        assert_frame_equal(lido, getattr(res_base, nome), check_exact=True)


def test_gravacao_e_idempotente(proj, res_base):
    cfg, _ = proj
    bd.gravar_demo(cfg, res_base)
    antes = {n: bd.destino(cfg, n).read_bytes() for n in bd.ARTEFATOS}
    assert bd.gravar_demo(cfg, res_base) == dict.fromkeys(bd.ARTEFATOS, "ja_existente")
    assert {n: bd.destino(cfg, n).read_bytes() for n in bd.ARTEFATOS} == antes


def test_regravacao_parcial_mantem_a_ordem_do_contrato(proj, res_base):
    cfg, _ = proj
    bd.gravar_demo(cfg, res_base)
    bd.destino(cfg, "spi_gamma_params").unlink()
    status = bd.gravar_demo(cfg, res_base)
    assert list(status) == list(bd.ARTEFATOS)
    assert status["spi_gamma_params"] == "gerado"
    assert [v for n, v in status.items() if n != "spi_gamma_params"] == ["ja_existente"] * 4


def test_conflito_levanta_e_nao_grava_nenhum_faltante(proj, res_base):
    """Todo-ou-nada: um artefato diferente barra a gravação dos que ainda faltam."""
    cfg, _ = proj
    destino = bd.destino(cfg, "cobertura_mensal")
    destino.parent.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(destino, index=False)
    antes = destino.read_bytes()
    with pytest.raises(bd.DemoImutavelError, match="cobertura_mensal"):
        bd.gravar_demo(cfg, res_base)
    for nome in bd.ARTEFATOS:
        if nome != "cobertura_mensal":
            assert not bd.destino(cfg, nome).exists(), nome
    assert destino.read_bytes() == antes


def test_conflito_com_um_valor_diferente_e_detectado(proj, res_base):
    cfg, _ = proj
    bd.gravar_demo(cfg, res_base)
    alterado = res_base.demo_observado.copy()
    v = alterado.loc[100, "spi3"]
    assert np.isfinite(v)
    alterado.loc[100, "spi3"] = np.nextafter(v, np.float32(np.inf))      # 1 ULP: só a comparação exata pega
    outro = bd.ResultadoDemo(**{**{n: getattr(res_base, n) for n in bd.ARTEFATOS}, "demo_observado": alterado})
    with pytest.raises(bd.DemoImutavelError, match="demo_observado"):
        bd.gravar_demo(cfg, outro)


def test_artefato_ilegivel_conta_como_conflito(proj, res_base):
    cfg, _ = proj
    destino = bd.destino(cfg, "spi_gamma_params")
    destino.parent.mkdir(parents=True)
    destino.write_bytes(b"isto nao e parquet")
    with pytest.raises(bd.DemoImutavelError, match="spi_gamma_params"):
        bd.gravar_demo(cfg, res_base)
    assert not bd.destino(cfg, "demo_observado").exists()


def test_nao_deixa_arquivo_temporario(proj, res_base):
    cfg, raiz = proj
    bd.gravar_demo(cfg, res_base)
    assert list(raiz.rglob("*.tmp")) == []


def test_falha_na_escrita_nao_deixa_parcial_nem_temporario(proj, res_base, monkeypatch):
    cfg, raiz = proj

    def falha(self, caminho, *a, **k):
        Path(caminho).write_bytes(b"meio arquivo")
        raise OSError("disco cheio")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", falha)
    with pytest.raises(OSError, match="disco cheio"):
        bd.gravar_demo(cfg, res_base)
    assert list(raiz.rglob("*.tmp")) == []
    assert not bd.destino(cfg, "demo_observado").exists()


# ================================================================== construir_demo (fim a fim)
def test_fim_a_fim_gera_artefatos_e_manifest(proj):
    cfg, _ = proj
    r = rodar(proj)
    assert r["artefatos"] == dict.fromkeys(bd.ARTEFATOS, "gerado")
    assert r["manifest"] == "gerado"
    assert all(bd.destino(cfg, n).is_file() for n in bd.ARTEFATOS)
    m = mf.verificar_manifest(cfg)
    assert m["dataset_sha256"] == r["dataset_sha256"]


def test_manifest_lista_artefatos_lotes_e_referencia(proj):
    cfg, _ = proj
    rodar(proj)
    nomes = set(mf.verificar_manifest(cfg)["arquivos"])
    assert set(bd.ARTEFATOS) <= nomes
    assert {"raw_chirps/chirps_mensal_198101_200012.parquet",
            "raw_chirps/chirps_mensal_200101_202609.parquet"} <= nomes
    assert {"referencia/municipios_semiarido.csv", "referencia/malha_pi_2022.parquet",
            "referencia/malha_pi_2022_gee.geojson", "referencia/referencia.sha256.json"} <= nomes
    assert not any(n.endswith(".sha256.json") and n.startswith("raw_chirps") for n in nomes)


def test_manifest_traz_aviso_de_maturidade_e_resumo_do_demo(proj):
    cfg, _ = proj
    rodar(proj)
    m = mf.verificar_manifest(cfg)
    assert m["maturity"] == cfg.projeto.maturidade and "MVP" in m["maturity"]
    d = m["extras"]["demo"]
    assert d == {"n_municipios": 3, "n_meses": N_MESES, "n_excluidos": 0, "escalas_spi": [3, 6],
                 "spi_calibrado_ate": "2015-12", "spi_treino_in_sample": True}
    assert m["split"]["teste_liberado"] is False


def test_resumo_devolvido(proj):
    r = rodar(proj)
    assert r["n_municipios"] == 3 and r["n_meses"] == N_MESES and r["n_excluidos"] == 0
    assert r["maturity"].startswith("MVP experimental")
    assert 0.0 < r["frac_alvo_invalido"] < 0.05          # só os 2 primeiros meses (+ células sem ajuste)
    assert len(r["dataset_sha256"]) == 64
    proj_cfg, _ = proj
    d = pd.read_parquet(bd.destino(proj_cfg, "demo_observado"))
    assert r["frac_alvo_invalido"] == float(d["spi3_invalid"].mean()) != float(d["spi6_invalid"].mean())


def test_segunda_execucao_e_idempotente(proj):
    cfg, _ = proj
    r1 = rodar(proj)
    manifest_antes = (cfg.raiz / cfg.manifest.arquivo).read_bytes()
    r2 = bd.construir_demo(cfg, municipios=(A, B, C), agora=datetime(2030, 1, 1, tzinfo=timezone.utc))
    assert r2["artefatos"] == dict.fromkeys(bd.ARTEFATOS, "ja_existente")
    assert r2["manifest"] == "ja_existente"
    assert r2["dataset_sha256"] == r1["dataset_sha256"]
    assert (cfg.raiz / cfg.manifest.arquivo).read_bytes() == manifest_antes


def test_caminho_verificado_e_a_porta_da_interface(proj):
    cfg, _ = proj
    rodar(proj)
    p = mf.caminho_verificado(cfg, "demo_observado")
    assert p == bd.destino(cfg, "demo_observado")
    d = pd.read_parquet(p)
    assert {"spi3", "spi3_invalid", "spi3_baixa_conf"} <= set(d.columns)


def test_adulteracao_de_artefato_e_detectada(proj):
    cfg, _ = proj
    rodar(proj)
    destino = bd.destino(cfg, "demo_observado")
    dados = bytearray(destino.read_bytes())
    dados[len(dados) // 2] ^= 0xFF
    destino.write_bytes(bytes(dados))
    with pytest.raises(mf.ManifestIntegridadeError, match="demo_observado"):
        mf.verificar_manifest(cfg)
    with pytest.raises(mf.ManifestIntegridadeError):
        mf.caminho_verificado(cfg, "demo_observado")


def test_adulteracao_do_raw_e_detectada_pelo_manifest_e_pela_leitura(proj):
    cfg, _ = proj
    rodar(proj)
    lote = chirps_raw.diretorio_raw(cfg) / "chirps_mensal_200101_202609.parquet"
    dados = bytearray(lote.read_bytes())
    dados[len(dados) // 2] ^= 0xFF
    lote.write_bytes(bytes(dados))
    with pytest.raises(mf.ManifestIntegridadeError, match="chirps_mensal_200101_202609"):
        mf.verificar_manifest(cfg)
    with pytest.raises(chirps_raw.RawIntegridadeError):
        rodar(proj)


def test_params_alterado_depois_do_manifest_e_detectado(proj):
    cfg, raiz = proj
    rodar(proj)
    p = raiz / "params.yaml"
    p.write_text(p.read_text(encoding="utf-8") + "\n# alterado\n", encoding="utf-8")
    with pytest.raises(mf.ManifestIntegridadeError, match="params.yaml"):
        mf.verificar_manifest(cfg)


def test_raw_incompleto_nao_grava_nada(proj):
    cfg, _ = proj
    (chirps_raw.diretorio_raw(cfg) / "chirps_mensal_200101_202609.parquet").unlink()
    with pytest.raises(chirps_raw.RawSchemaError, match="faltam 927 linha"):
        rodar(proj)
    assert not any(bd.destino(cfg, n).exists() for n in bd.ARTEFATOS)
    assert not (cfg.raiz / cfg.manifest.arquivo).exists()


def test_raw_com_municipio_a_menos_nao_grava_nada(proj):
    cfg, _ = proj
    with pytest.raises(chirps_raw.RawSchemaError):
        bd.construir_demo(cfg, municipios=(A, B, C, "2200004"), agora=AGORA)
    assert not any(bd.destino(cfg, n).exists() for n in bd.ARTEFATOS)


def test_sem_raw_levanta_erro_claro(tmp_path):
    shutil.copy(PARAMS_REAL, tmp_path / "params.yaml")
    cfg = load_config(tmp_path / "params.yaml")
    with pytest.raises(chirps_raw.RawIntegridadeError, match="nenhum lote RAW"):
        bd.construir_demo(cfg, municipios=(A,), agora=AGORA)


def test_manifest_de_outro_conjunto_exige_substituir(proj):
    cfg, _ = proj
    destino = cfg.raiz / cfg.manifest.arquivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"dataset_sha256": "00" * 32}), encoding="utf-8")
    with pytest.raises(mf.ManifestImutavelError):
        rodar(proj)
    assert json.loads(destino.read_text(encoding="utf-8")) == {"dataset_sha256": "00" * 32}
    r = rodar(proj, substituir_manifest=True)
    assert r["manifest"] == "substituido"
    mf.verificar_manifest(cfg)


def test_sem_selo_da_referencia_manifest_falha_e_nova_execucao_e_segura(proj):
    cfg, raiz = proj
    selo = raiz / "data" / "reference" / "referencia.sha256.json"
    conteudo = selo.read_bytes()
    selo.unlink()
    with pytest.raises(mf.ManifestError, match="selo da referência ausente"):
        rodar(proj)
    assert all(bd.destino(cfg, n).is_file() for n in bd.ARTEFATOS)      # artefatos já gravados e íntegros
    selo.write_bytes(conteudo)
    r = rodar(proj)
    assert r["artefatos"] == dict.fromkeys(bd.ARTEFATOS, "ja_existente")
    assert r["manifest"] == "gerado"


def test_fim_a_fim_com_municipio_excluido(proj, tmp_path):
    cfg, raiz = proj
    # reescreve o RAW de um projeto novo com o município A sem dados em 22 meses seguidos do treino
    raiz2 = tmp_path / "outro"
    raiz2.mkdir()
    shutil.copytree(raiz / "data" / "reference", raiz2 / "data" / "reference")
    shutil.copy(PARAMS_REAL, raiz2 / "params.yaml")
    escrever_raw(raiz2, sem_dado(raw_sintetico(), A, "1985-01", "1986-10"), (A, B, C))
    cfg2 = load_config(raiz2 / "params.yaml")
    r = bd.construir_demo(cfg2, municipios=(A, B, C), agora=AGORA)
    assert r["n_excluidos"] == 1 and r["n_municipios"] == 3
    m = mf.verificar_manifest(cfg2)
    assert m["extras"]["demo"]["n_excluidos"] == 1
    d = pd.read_parquet(mf.caminho_verificado(cfg2, "demo_observado"))
    assert d.loc[d["cod_ibge"] == A, "spi3"].isna().all()
    assert d.loc[d["cod_ibge"] == B, "spi3"].notna().any()


def test_teste_congelado_continua_bloqueado_depois_do_demo(proj, monkeypatch):
    cfg, _ = proj
    def proibido(*a, **k):
        raise AssertionError("o demo não pode tocar rotular/bloco")
    monkeypatch.setattr(splits, "rotular", proibido)
    monkeypatch.setattr(splits, "bloco", proibido)
    rodar(proj)
    assert cfg.split.teste_liberado is False
    with pytest.raises(splits.TesteBloqueadoError):
        splits.exigir_teste_liberado(cfg)


# ================================================================== CLI
@pytest.fixture
def cli_proj(proj, monkeypatch):
    monkeypatch.setattr(reference, "ler_municipios_escopo",
                        lambda cfg: pd.DataFrame({"cod_ibge": [A, B, C]}))
    return proj


def test_cli_gera_e_mostra_aviso_e_status(cli_proj, capsys):
    cfg, raiz = cli_proj
    assert bd.main(["--params", str(raiz / "params.yaml")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("AVISO: MVP experimental")
    for nome in bd.ARTEFATOS:
        assert f"[gerado] {nome}" in out
    assert "[gerado] manifest" in out and "municípios=3" in out and "in-sample" in out
    assert all(bd.destino(cfg, n).is_file() for n in bd.ARTEFATOS)


def test_cli_segunda_vez_so_confere(cli_proj, capsys):
    _, raiz = cli_proj
    args = ["--params", str(raiz / "params.yaml")]
    assert bd.main(args) == 0
    capsys.readouterr()
    assert bd.main(args) == 0
    out = capsys.readouterr().out
    assert "[ja_existente] demo_observado" in out and "[ja_existente] manifest" in out


def test_cli_erro_vira_codigo_1_e_mensagem_no_stderr(tmp_path, capsys, monkeypatch):
    shutil.copy(PARAMS_REAL, tmp_path / "params.yaml")
    monkeypatch.setattr(reference, "ler_municipios_escopo", lambda cfg: pd.DataFrame({"cod_ibge": [A]}))
    assert bd.main(["--params", str(tmp_path / "params.yaml")]) == 1
    cap = capsys.readouterr()
    assert cap.err.startswith("ERRO:") and "nenhum lote RAW" in cap.err
    assert cap.out == ""


def test_cli_params_inexistente_vira_erro(tmp_path, capsys):
    assert bd.main(["--params", str(tmp_path / "nao_existe.yaml")]) == 1
    assert capsys.readouterr().err.startswith("ERRO:")


def test_cli_manifest_de_outro_conjunto_so_com_flag(cli_proj, capsys):
    cfg, raiz = cli_proj
    destino = cfg.raiz / cfg.manifest.arquivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps({"dataset_sha256": "00" * 32}), encoding="utf-8")
    args = ["--params", str(raiz / "params.yaml")]
    assert bd.main(args) == 1
    assert "outro conjunto de dados" in capsys.readouterr().err
    assert bd.main(args + ["--substituir-manifest"]) == 0
    assert "[substituido] manifest" in capsys.readouterr().out


# ================================================================== coerência do módulo
def test_nomes_da_referencia_iguais_aos_do_modulo_reference():
    assert bd.PARQUET_REFERENCIA == reference.PARQUET
    assert bd.GEOJSON_REFERENCIA == reference.GEOJSON
    assert mf.SELO_REFERENCIA == reference.SELO


def test_dominio_puro_sem_framework_web_sql_ou_geopandas_na_importacao():
    fonte = (RAIZ / "src" / "data" / "build_demo.py").read_text(encoding="utf-8")
    raiz_ast = ast.parse(fonte)
    topo = set()
    for no in raiz_ast.body:                                   # só importações de nível de módulo
        if isinstance(no, ast.Import):
            topo |= {a.name.split(".")[0] for a in no.names}
        elif isinstance(no, ast.ImportFrom):
            topo.add((no.module or "").split(".")[0])
    proibidos = {"fastapi", "flask", "starlette", "uvicorn", "sqlite3", "sqlalchemy", "geopandas", "ee",
                 "geemap", "torch", "requests"}
    assert not (topo & proibidos), topo & proibidos
    assert "reference" not in {a.name for no in raiz_ast.body if isinstance(no, ast.ImportFrom)
                               for a in no.names}


def test_nenhum_dado_de_exemplo_em_src():
    fonte = (RAIZ / "src" / "data" / "build_demo.py").read_text(encoding="utf-8")
    assert "np.random" not in fonte and "default_rng" not in fonte and "import random" not in fonte
