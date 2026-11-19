"""Testes de src/data/chirps_raw.py (Item 2a). Sem internet, sem GEE."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import chirps_raw as r
from src.data.config import load_config

PARAMS_REAL = Path(__file__).resolve().parents[1] / "params.yaml"
CODS = ["2207801", "2200000", "2201234"]
MESES = list(pd.period_range("2020-01", "2020-03", freq="M"))


def fabrica(cods=CODS, meses=MESES) -> pd.DataFrame:
    linhas = [(c, m.to_timestamp(), 10.0, 5.0, 3.0, m.days_in_month, m.days_in_month, 1.0)
              for c in cods for m in meses]
    return pd.DataFrame(linhas, columns=r.COLUNAS)


TRES = ["precip_mm", "precip_max_dia_mm", "dias_chuva"]


def sem_dado(df, linha):
    """Município sem pixel válido: as 3 colunas de chuva NaN e cobertura zero."""
    df.loc[linha, TRES] = np.nan
    df.loc[linha, ["pixels_validos_frac", "dias_validos"]] = [0.0, 0]


def valida(df, cods=CODS, meses=MESES):
    r.validar(r.padronizar(df), cods, meses)


# ------------------------------------------------------------------ meses e lotes
def test_meses_janela_e_diretorio_do_params_real():
    cfg = load_config(PARAMS_REAL)
    ms = r.meses_janela(cfg)
    assert (len(ms), str(ms[0]), str(ms[-1])) == (549, "1981-01", "2026-09")
    assert r.diretorio_raw(cfg) == cfg.raiz / "data" / "raw" / "chirps"


@pytest.mark.parametrize("n", [1, 7, 185, 224, 4000])
def test_lotes_cobrem_tudo_sem_sobreposicao_e_respeitam_limite(n):
    ms = list(pd.period_range("1981-01", "2025-12", freq="M"))
    lotes = r.dividir_em_lotes(ms, n, 4000)
    cobertos = [m for ini, fim in lotes for m in pd.period_range(ini, fim, freq="M")]
    assert cobertos == ms
    assert all(n * len(pd.period_range(i, f, freq="M")) <= 4000 for i, f in lotes)


def test_lotes_com_185_municipios_sao_26():
    ms = list(pd.period_range("1981-01", "2025-12", freq="M"))
    lotes = r.dividir_em_lotes(ms, 185, 4000)
    assert r.meses_por_lote(185, 4000) == 21 and len(lotes) == 26
    assert str(lotes[-1][0]) == "2024-10" and str(lotes[-1][1]) == "2025-12"


@pytest.mark.parametrize("n,maximo", [(0, 4000), (-1, 4000), (10, 0), (5000, 4000)])
def test_meses_por_lote_rejeita_entradas_invalidas(n, maximo):
    with pytest.raises(ValueError):
        r.meses_por_lote(n, maximo)


@pytest.mark.parametrize("meses", [
    [],
    [pd.Period("2020-01", "M"), pd.Period("2020-03", "M")],
    [pd.Period("2020-02", "M"), pd.Period("2020-01", "M")],
    [pd.Period("2020-01", "M"), pd.Period("2020-01", "M")],
    [pd.Period("2020-01-01", "D"), pd.Period("2020-01-02", "D")],
])
def test_dividir_em_lotes_rejeita_meses_invalidos(meses):
    with pytest.raises(ValueError):
        r.dividir_em_lotes(meses, 10, 4000)


# ------------------------------------------------------------------ padronizar
def test_padronizar_aplica_dtypes_do_contrato_e_ordena():
    df = fabrica().sample(frac=1, random_state=0)
    df["mes"] = df["mes"].astype("datetime64[us]")
    df["cod_ibge"] = df["cod_ibge"].astype(int)          # int de 7 dígitos vira str
    out = r.padronizar(df)
    assert list(out.columns) == r.COLUNAS
    assert {c: str(out[c].dtype) for c in r.DTYPES} == r.DTYPES
    assert pd.api.types.is_string_dtype(out["cod_ibge"])
    assert out.equals(out.sort_values(["cod_ibge", "mes"]).reset_index(drop=True))


def test_padronizar_nao_altera_o_dataframe_de_entrada():
    df = fabrica()
    antes = df.copy()
    r.padronizar(df)
    pd.testing.assert_frame_equal(df, antes)


def _sem_coluna(d):
    return d.drop(columns=["precip_mm"])


def _coluna_extra(d):
    return d.assign(extra=1)


def _dias_nan(d):
    d = d.astype({"dias_validos": "float64"})
    d.loc[0, "dias_validos"] = np.nan
    return d


def _dias_fracionario(d):
    d = d.astype({"dias_validos": "float64"})
    d.loc[0, "dias_validos"] = 3.5
    return d


def _dias_gigante(d):
    return d.assign(dias_validos=40000)


def _max_dia_texto(d):
    d = d.astype({"precip_max_dia_mm": "object"})
    d.loc[0, "precip_max_dia_mm"] = "abc"
    return d


def _precip_texto(d):
    d = d.astype({"precip_mm": "object"})
    d.loc[0, "precip_mm"] = "abc"
    return d


@pytest.mark.parametrize("muta", [_sem_coluna, _coluna_extra, _dias_nan, _dias_fracionario,
                                  _dias_gigante, _precip_texto, _max_dia_texto])
def test_padronizar_rejeita_entradas_fora_do_contrato(muta):
    with pytest.raises(r.RawSchemaError):
        r.padronizar(muta(fabrica()))


def test_padronizar_rejeita_o_que_nao_e_dataframe():
    with pytest.raises(r.RawSchemaError):
        r.padronizar([1, 2, 3])


# ------------------------------------------------------------------ validar
def test_validar_aceita_baseline_e_precip_nan_sem_virar_zero():
    valida(fabrica())
    df = fabrica()
    sem_dado(df, 0)
    df.loc[1, TRES] = np.nan                              # NaN com cobertura cheia também é aceito
    padrao = r.padronizar(df)
    r.validar(padrao, CODS, MESES)
    assert padrao["precip_mm"].isna().sum() == 2
    assert padrao["precip_max_dia_mm"].isna().sum() == padrao["dias_chuva"].isna().sum() == 2


def _bordas(d):
    d = d.copy()
    d.loc[0, ["precip_mm", "precip_max_dia_mm"]] = [10.0, 10.0]          # um único dia concentra o mês
    d.loc[1, ["precip_mm", "precip_max_dia_mm", "dias_chuva"]] = [0.0, 0.0, 0.0]   # mês seco
    d.loc[2, "dias_chuva"] = float(d.loc[2, "dias_no_mes"])               # chuva todos os dias
    d.loc[3, "precip_max_dia_mm"] = 10.0004                               # dentro da folga de float32
    return d


def test_validar_aceita_valores_de_borda_das_colunas_novas():
    valida(_bordas(fabrica()))


def test_validar_rejeita_max_dia_acima_da_folga():
    d = fabrica()
    d.loc[0, "precip_max_dia_mm"] = 10.002
    with pytest.raises(r.RawSchemaError, match="precip_max_dia_mm maior que precip_mm"):
        valida(d)


def test_validar_aceita_fevereiro_bissexto():
    ms = [pd.Period("2020-02", "M")]
    valida(fabrica(CODS, ms), CODS, ms)                   # 29 dias
    ms2 = [pd.Period("2021-02", "M")]
    valida(fabrica(CODS, ms2), CODS, ms2)                 # 28 dias


def _set(coluna, valor, linha=0):
    def f(d):
        d = d.copy()
        d[coluna] = d[coluna].astype("object") if isinstance(valor, str) else d[coluna]
        d.loc[linha, coluna] = valor
        return d
    return f


def _precip_com_cobertura_zero(d):
    d = d.copy()
    d.loc[0, ["precip_mm", "pixels_validos_frac"]] = [5.0, 0.0]
    return d


def _precip_sem_dia_valido(d):
    d = d.copy()
    d.loc[0, ["precip_mm", "dias_validos"]] = [5.0, 0]
    return d


VIOLACOES = [
    ("precip negativa", _set("precip_mm", -1.0), "precip_mm fora de"),
    ("precip infinita", _set("precip_mm", np.inf), "precip_mm fora de"),
    ("precip acima do teto", _set("precip_mm", 2500.0), "precip_mm fora de"),
    ("precip sem pixel", _precip_com_cobertura_zero, "sem dia/pixel válido"),
    ("precip sem dia", _precip_sem_dia_valido, "sem dia/pixel válido"),
    ("fracao > 1", _set("pixels_validos_frac", 1.5), "pixels_validos_frac fora"),
    ("fracao < 0", _set("pixels_validos_frac", -0.1), "pixels_validos_frac fora"),
    ("fracao NaN", _set("pixels_validos_frac", np.nan), "pixels_validos_frac fora"),
    ("dias_validos > dias_no_mes", _set("dias_validos", 32), "dias_validos fora"),
    ("dias_validos < 0", _set("dias_validos", -1), "dias_validos fora"),
    ("dias_no_mes errado", _set("dias_no_mes", 30), "dias_no_mes diferente"),
    ("mes dia 15", _set("mes", pd.Timestamp("2020-01-15")), "mes deve ser"),
    ("mes NaT", _set("mes", pd.NaT), "mes deve ser"),
    ("max_dia negativo", _set("precip_max_dia_mm", -1.0), "precip_max_dia_mm negativo"),
    ("max_dia infinito", _set("precip_max_dia_mm", np.inf), "precip_max_dia_mm negativo ou infinito"),
    ("max_dia maior que a soma", _set("precip_max_dia_mm", 11.0), "precip_max_dia_mm maior que precip_mm"),
    ("dias_chuva negativo", _set("dias_chuva", -0.5), "dias_chuva fora de"),
    ("dias_chuva acima do mes", _set("dias_chuva", 31.5), "dias_chuva fora de"),
    ("dias_chuva infinito", _set("dias_chuva", np.inf), "dias_chuva fora de"),
    ("max_dia NaN sozinho", _set("precip_max_dia_mm", np.nan), "devem ser NaN juntos"),
    ("dias_chuva NaN sozinho", _set("dias_chuva", np.nan), "devem ser NaN juntos"),
    ("precip NaN sozinha", _set("precip_mm", np.nan), "devem ser NaN juntos"),
    ("soma zero com max_dia", _set("precip_mm", 0.0), "precip_mm = 0 com chuva"),
    ("cod curto", _set("cod_ibge", "22078"), "cod_ibge inválido"),
    ("cod com letra", _set("cod_ibge", "22078AB"), "cod_ibge inválido"),
]


@pytest.mark.parametrize("nome,muta,msg", VIOLACOES, ids=[v[0] for v in VIOLACOES])
def test_validar_rejeita_violacoes_de_valor(nome, muta, msg):
    with pytest.raises(r.RawSchemaError, match=msg):
        valida(muta(fabrica()))


def test_validar_aponta_duplicata_linha_faltando_e_linha_extra():
    base = fabrica()
    with pytest.raises(r.RawSchemaError, match="duplicatas"):
        valida(pd.concat([base, base.iloc[:1]], ignore_index=True))
    with pytest.raises(r.RawSchemaError, match="faltam 1"):
        valida(base.drop(index=2))
    extra = pd.concat([base, fabrica(["2299999"], MESES[:1])], ignore_index=True)
    with pytest.raises(r.RawSchemaError, match="fora do esperado"):
        valida(extra)
    fora = pd.concat([base, fabrica(CODS[:1], [pd.Period("2020-04", "M")])], ignore_index=True)
    with pytest.raises(r.RawSchemaError, match="fora do esperado"):
        valida(fora)


def test_validar_junta_varias_violacoes_numa_mensagem():
    df = fabrica()
    df.loc[0, "precip_mm"] = -1.0
    df.loc[1, "dias_no_mes"] = 30
    with pytest.raises(r.RawSchemaError) as e:
        valida(df)
    assert "precip_mm fora de" in str(e.value) and "dias_no_mes diferente" in str(e.value)


def test_validar_rejeita_dtype_errado_sem_padronizar():
    df = r.padronizar(fabrica()).astype({"precip_mm": "float64"})
    with pytest.raises(r.RawSchemaError, match="dtypes fora do contrato"):
        r.validar(df, CODS, MESES)


@pytest.mark.parametrize("municipios", [[], ["2207801", "2207801"], ["22078"], [2207801]])
def test_lista_de_municipios_invalida_e_rejeitada(municipios):
    with pytest.raises(r.RawSchemaError):
        r.validar(r.padronizar(fabrica()), municipios, MESES)


# ------------------------------------------------------------------ gravar / ler
def _grava(tmp_path, df=None):
    return r.gravar_lote(fabrica() if df is None else df, tmp_path, CODS, MESES[0], MESES[-1])


def test_gravar_cria_parquet_selo_e_hash_confere(tmp_path):
    res = _grava(tmp_path)
    assert res.status == "gravado" and res.n_linhas == 9 and res.n_precip_nan == 0
    assert res.caminho.name == "chirps_mensal_202001_202003.parquet"
    assert r.sha256_arquivo(res.caminho) == res.sha256 == r.verificar_integridade(res.caminho)
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "chirps_mensal_202001_202003.parquet", "chirps_mensal_202001_202003.parquet.sha256.json"]


def test_roundtrip_preserva_dtypes_e_nan(tmp_path):
    df = fabrica()
    sem_dado(df, 0)
    res = _grava(tmp_path, df)
    lido = r.ler_lote(res.caminho)
    assert {c: str(lido[c].dtype) for c in r.DTYPES} == r.DTYPES
    assert res.n_precip_nan == 1 and lido["precip_mm"].isna().sum() == 1
    assert lido["precip_max_dia_mm"].isna().sum() == lido["dias_chuva"].isna().sum() == 1
    assert (lido["precip_mm"].dropna() > 0).all()          # NaN nunca virou 0


def test_gravar_de_novo_com_mesmo_conteudo_pula_sem_alterar_o_arquivo(tmp_path):
    primeiro = _grava(tmp_path)
    segundo = _grava(tmp_path, fabrica().sample(frac=1, random_state=1))
    assert segundo.status == "pulado" and segundo.sha256 == primeiro.sha256
    assert r.sha256_arquivo(primeiro.caminho) == primeiro.sha256


def test_gravar_com_conteudo_diferente_levanta_imutavel_e_nao_altera(tmp_path):
    primeiro = _grava(tmp_path)
    outro = fabrica()
    outro.loc[0, "precip_mm"] = 11.0
    with pytest.raises(r.RawImutavelError):
        _grava(tmp_path, outro)
    assert r.sha256_arquivo(primeiro.caminho) == primeiro.sha256


def test_imutavel_detecta_alteracao_de_1_bit_em_float32(tmp_path):
    primeiro = _grava(tmp_path)
    vizinho = float(np.nextafter(np.float32(10.0), np.float32(11.0)))   # menor passo acima de 10.0
    outro = fabrica()
    outro.loc[0, "precip_mm"] = vizinho
    with pytest.raises(r.RawImutavelError):
        _grava(tmp_path, outro)
    assert r.sha256_arquivo(primeiro.caminho) == primeiro.sha256


def test_gravar_dados_invalidos_nao_deixa_nenhum_arquivo(tmp_path):
    ruim = fabrica()
    ruim.loc[0, "precip_mm"] = -5.0
    with pytest.raises(r.RawSchemaError):
        _grava(tmp_path, ruim)
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_arquivo_adulterado_e_detectado(tmp_path):
    res = _grava(tmp_path)
    res.caminho.write_bytes(res.caminho.read_bytes() + b"x")
    with pytest.raises(r.RawIntegridadeError, match="divergente"):
        r.verificar_integridade(res.caminho)
    with pytest.raises(r.RawIntegridadeError):
        _grava(tmp_path)


def test_meio_estado_e_erro_nunca_falso(tmp_path):
    res = _grava(tmp_path)
    selo = res.caminho.with_name(res.caminho.name + ".sha256.json")
    selo.unlink()
    with pytest.raises(r.RawIntegridadeError, match="meio-estado"):
        r.lote_gravado(tmp_path, MESES[0], MESES[-1])
    with pytest.raises(r.RawIntegridadeError):
        _grava(tmp_path)


def test_lote_gravado_false_quando_nao_existe_e_true_quando_existe(tmp_path):
    assert r.lote_gravado(tmp_path, MESES[0], MESES[-1]) is False
    _grava(tmp_path)
    assert r.lote_gravado(tmp_path, MESES[0], MESES[-1]) is True


def test_ler_raw_junta_lotes_em_ordem_e_exige_completude(tmp_path):
    for ini, fim in [(MESES[0], MESES[1]), (MESES[2], MESES[2])]:
        ms = list(pd.period_range(ini, fim, freq="M"))
        r.gravar_lote(fabrica(CODS, ms), tmp_path, CODS, ini, fim)
    todo = r.ler_raw(tmp_path, CODS, MESES)
    assert len(todo) == 9 and todo.equals(r.padronizar(fabrica()))
    with pytest.raises(r.RawSchemaError, match="faltam"):
        r.ler_raw(tmp_path, CODS, MESES + [pd.Period("2020-04", "M")])


def test_ler_raw_sem_lotes_ou_com_lote_adulterado_falha(tmp_path):
    with pytest.raises(r.RawIntegridadeError, match="nenhum lote"):
        r.ler_raw(tmp_path / "nao_existe", CODS, MESES)
    res = _grava(tmp_path)
    res.caminho.write_bytes(res.caminho.read_bytes() + b"x")
    with pytest.raises(r.RawIntegridadeError):
        r.ler_raw(tmp_path, CODS, MESES)


def test_ler_raw_detecta_lotes_sobrepostos(tmp_path):
    ms2 = MESES[:2]
    r.gravar_lote(fabrica(), tmp_path, CODS, MESES[0], MESES[-1])
    r.gravar_lote(fabrica(CODS, ms2), tmp_path, CODS, ms2[0], ms2[-1])
    with pytest.raises(r.RawSchemaError, match="duplicatas"):
        r.ler_raw(tmp_path, CODS, MESES)


def test_meta_vai_para_o_selo_e_nao_muda_hash_nem_idempotencia(tmp_path):
    import json
    meta = {"n_imagens_por_mes": {"2020-01": 31, "2020-02": 29, "2020-03": 31}, "gee_id": "X"}
    a = r.gravar_lote(fabrica(), tmp_path, CODS, MESES[0], MESES[-1], meta=meta)
    selo = json.loads(a.caminho.with_name(a.caminho.name + ".sha256.json").read_text("utf-8"))
    assert selo["meta"] == meta and selo["sha256"] == a.sha256
    b = r.gravar_lote(fabrica(), tmp_path, CODS, MESES[0], MESES[-1], meta={"outro": 1})
    assert b.status == "pulado" and b.sha256 == a.sha256       # meta não entra na comparação
    sem = r.gravar_lote(fabrica(), tmp_path / "sem", CODS, MESES[0], MESES[-1])
    assert "meta" not in json.loads(
        sem.caminho.with_name(sem.caminho.name + ".sha256.json").read_text("utf-8"))


@pytest.mark.parametrize("meta", [{"x": float("nan")}, {"x": np.int64(3)}, {"x": {1, 2}}])
def test_meta_invalida_e_rejeitada_sem_deixar_arquivo(tmp_path, meta):
    with pytest.raises(r.RawSchemaError, match="meta"):
        r.gravar_lote(fabrica(), tmp_path, CODS, MESES[0], MESES[-1], meta=meta)
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_modulo_e_puro_nao_importa_ee():
    assert not hasattr(r, "ee")
