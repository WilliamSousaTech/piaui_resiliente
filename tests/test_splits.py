"""Testes de src/data/splits.py (Item 4). Sem internet; dados sintéticos só aqui.

Os cortes são fixados em cada teste (não dependem dos valores do params.yaml real, exceto nos
testes marcados como "params real"), para que mudar o embargo no params não quebre o contrato do módulo.
"""
import itertools
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data import splits as sp
from src.data.config import load_config

PARAMS_REAL = Path(__file__).resolve().parents[1] / "params.yaml"
TREINO_FIM, VAL_FIM, TESTE_FIM = "2015-12", "2020-12", "2025-12"


def cfg_com(tmp_path: Path, *, embargo=3, liberado=False, escala=3, janelas=None):
    d = yaml.safe_load(PARAMS_REAL.read_text(encoding="utf-8"))
    d["alvo"]["escala_spi_meses"] = escala
    d["spi"]["janelas_meses"] = list(janelas) if janelas is not None else [escala]
    d["split"].update({"treino_fim": TREINO_FIM, "val_fim": VAL_FIM, "teste_fim": TESTE_FIM,
                       "embargo_meses": embargo, "teste_liberado": liberado})
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False),
                                          encoding="utf-8")
    return load_config(tmp_path / "params.yaml")


def janela_inteira() -> pd.Series:
    return pd.Series(pd.period_range("1981-01", "2025-12", freq="M").to_timestamp())


def blocos_por_mes(cfg, escalas=None) -> dict[str, str]:
    meses = janela_inteira()
    rot = sp.rotular(meses, cfg, escalas)
    return {str(p): r for p, r in zip(meses.dt.to_period("M"), rot)}


def P(s: str) -> pd.Period:
    return pd.Period(s, freq="M")


# ================================================================== cortes
@pytest.mark.parametrize("e, val_ini, teste_ini", [(2, "2016-03", "2021-03"), (3, "2016-04", "2021-04"),
                                                    (5, "2016-06", "2021-06")])
def test_cortes_somam_o_embargo(tmp_path, e, val_ini, teste_ini):
    c = sp.cortes(cfg_com(tmp_path, embargo=e))
    assert (c.treino_fim, c.val_fim, c.teste_fim) == (P(TREINO_FIM), P(VAL_FIM), P(TESTE_FIM))
    assert c.val_ini == P(val_ini) and c.teste_ini == P(teste_ini) and c.embargo_meses == e
    assert c.inicio == P("1981-01")


def test_cortes_para_manifest_e_json_puro(tmp_path):
    d = sp.cortes(cfg_com(tmp_path, embargo=3)).para_manifest()
    assert json.loads(json.dumps(d)) == d
    assert d == {"inicio": "1981-01", "treino_fim": "2015-12", "val_ini": "2016-04", "val_fim": "2020-12",
                 "teste_ini": "2021-04", "teste_fim": "2025-12", "embargo_meses": 3}


def test_embargo_insuficiente_para_spi6_e_recusado(tmp_path):
    cfg = cfg_com(tmp_path, embargo=3)
    with pytest.raises(sp.SplitError, match="SPI-6"):
        sp.cortes(cfg, escalas=(3, 6))
    with pytest.raises(sp.SplitError):
        sp.rotular(janela_inteira(), cfg, escalas=(3, 6))
    with pytest.raises(sp.SplitError):
        sp.bloco(pd.DataFrame({"mes_alvo": janela_inteira()}), "train", cfg, escalas=(3, 6))


def test_embargo_5_atende_spi3_e_spi6(tmp_path):
    c = sp.cortes(cfg_com(tmp_path, embargo=5), escalas=(3, 6))
    assert c.val_ini == P("2016-06")


def test_embargo_no_minimo_exato_passa(tmp_path):
    sp.cortes(cfg_com(tmp_path, embargo=5), escalas=(6,))            # 6 - 1 = 5
    with pytest.raises(sp.SplitError):
        sp.cortes(cfg_com(tmp_path, embargo=4), escalas=(6,))        # 4 < 5


def test_escala_padrao_vem_de_spi_janelas_meses(tmp_path):
    assert sp.cortes(cfg_com(tmp_path, embargo=2, escala=3)).val_ini == P("2016-03")
    assert sp.cortes(cfg_com(tmp_path, embargo=6, escala=7)).val_ini == P("2016-07")
    cfg = cfg_com(tmp_path, embargo=5, janelas=[3, 6])
    assert sp.cortes(cfg).val_ini == P("2016-06")
    # a própria config impede embargo < maior janela - 1; aqui ela é contornada (model_copy não valida)
    # para provar que o padrão do módulo é a lista de janelas (SPI-6 incluído), não só o alvo
    curto = cfg.model_copy(update={"split": cfg.split.model_copy(update={"embargo_meses": 3})})
    with pytest.raises(sp.SplitError, match="SPI-6"):
        sp.cortes(curto)


@pytest.mark.parametrize("escalas", [(), (0,), (-1,), (True,), ("3",), (3.0,)])
def test_escalas_invalidas(tmp_path, escalas):
    with pytest.raises(sp.SplitError, match="escalas"):
        sp.cortes(cfg_com(tmp_path), escalas=escalas)


# ================================================================== rotular: limites exatos
@pytest.mark.parametrize("e", [0, 2, 3, 5])
def test_limites_exatos_de_cada_bloco(tmp_path, e):
    cfg = cfg_com(tmp_path, embargo=e, escala=max(1, e + 1) if e < 2 else 3)
    b = blocos_por_mes(cfg)
    assert b["1981-01"] == "train" and b[TREINO_FIM] == "train"
    for k in range(1, e + 1):
        assert b[str(P(TREINO_FIM) + k)] == "embargo", k
        assert b[str(P(VAL_FIM) + k)] == "embargo", k
    assert b[str(P(TREINO_FIM) + e + 1)] == "val" and b[VAL_FIM] == "val"
    assert b[str(P(VAL_FIM) + e + 1)] == "test" and b[TESTE_FIM] == "test"


@pytest.mark.parametrize("e, n_emb, n_val, n_test", [(0, 0, 60, 60), (3, 6, 57, 57), (5, 10, 55, 55)])
def test_particao_completa_da_janela_e_contagens(tmp_path, e, n_emb, n_val, n_test):
    cfg = cfg_com(tmp_path, embargo=e, escala=3 if e >= 2 else 1)
    rot = sp.rotular(janela_inteira(), cfg)
    cont = rot.value_counts().to_dict()
    assert cont["train"] == 420 and cont["embargo"] == n_emb
    assert cont["val"] == n_val and cont["test"] == n_test
    assert sum(cont.values()) == 540 == len(rot)


@pytest.mark.parametrize("e, esperado", [
    (0, ["train", "val", "test"]),
    (3, ["train", "embargo", "val", "embargo", "test"])])
def test_ordem_cronologica_dos_blocos_sem_embaralhar(tmp_path, e, esperado):
    cfg = cfg_com(tmp_path, embargo=e, escala=3 if e >= 2 else 1)
    rot = sp.rotular(janela_inteira(), cfg)
    assert [k for k, _ in itertools.groupby(rot)] == esperado


def test_dia_do_mes_nao_muda_o_bloco(tmp_path):
    cfg = cfg_com(tmp_path, embargo=3)
    rot = sp.rotular(pd.Series(pd.to_datetime(["2015-12-01", "2015-12-31", "2016-01-01", "2016-01-31",
                                               "2016-04-15", "2021-04-30"])), cfg)
    assert list(rot) == ["train", "train", "embargo", "embargo", "val", "test"]


# ================================================================== rotular: entradas e erros
def test_resultado_e_categorico_com_categorias_do_contrato(tmp_path):
    rot = sp.rotular(janela_inteira(), cfg_com(tmp_path))
    assert isinstance(rot.dtype, pd.CategoricalDtype) and not rot.dtype.ordered
    assert list(rot.dtype.categories) == ["train", "val", "test", "embargo"] == list(sp.BLOCOS)
    assert rot.name == "split"


def test_aceita_lista_de_textos_period_e_preserva_o_indice(tmp_path):
    cfg = cfg_com(tmp_path)
    assert list(sp.rotular(["2015-12", "2016-01", "2016-04"], cfg)) == ["train", "embargo", "val"]
    per = pd.Series(pd.PeriodIndex(["2015-12", "2016-04"], freq="M"))
    assert list(sp.rotular(per, cfg)) == ["train", "val"]
    s = pd.Series(pd.to_datetime(["2020-12-01", "2015-12-01"]), index=[10, 20])
    r = sp.rotular(s, cfg)
    assert list(r.index) == [10, 20] and list(r) == ["val", "train"]


def test_entrada_vazia_devolve_vazio(tmp_path):
    r = sp.rotular(pd.Series([], dtype="datetime64[ns]"), cfg_com(tmp_path))
    assert len(r) == 0 and list(r.dtype.categories) == list(sp.BLOCOS)


@pytest.mark.parametrize("mes", ["1980-12-01", "2026-01-01", "2030-06-01"])
def test_mes_fora_da_janela_e_erro(tmp_path, mes):
    with pytest.raises(sp.SplitError, match="fora da janela"):
        sp.rotular(pd.Series(pd.to_datetime(["2020-01-01", mes])), cfg_com(tmp_path))


def test_mes_ausente_e_erro(tmp_path):
    with pytest.raises(sp.SplitError, match="NaT"):
        sp.rotular(pd.Series(pd.to_datetime(["2020-01-01", None])), cfg_com(tmp_path))


def test_rotular_e_deterministico(tmp_path):
    cfg = cfg_com(tmp_path)
    assert sp.rotular(janela_inteira(), cfg).equals(sp.rotular(janela_inteira(), cfg))


# ================================================================== gate do teste e filtro por bloco
def df_exemplo() -> pd.DataFrame:
    meses = janela_inteira()
    return pd.DataFrame({"mes_alvo": meses.to_numpy(), "x": range(len(meses))},
                        index=range(1000, 1000 + len(meses)))   # to_numpy: evita alinhar por índice


def test_teste_bloqueado_recusa_mesmo_sem_linhas_de_teste(tmp_path):
    cfg = cfg_com(tmp_path, liberado=False)
    so_treino = df_exemplo().iloc[:10]
    with pytest.raises(sp.TesteBloqueadoError):
        sp.bloco(so_treino, "test", cfg)
    with pytest.raises(sp.TesteBloqueadoError):
        sp.exigir_teste_liberado(cfg)


def test_gate_recusa_antes_de_validar_a_entrada(tmp_path):
    cfg = cfg_com(tmp_path, liberado=False)
    lixo = pd.DataFrame({"mes_alvo": pd.to_datetime(["1900-01-01"])})        # fora da janela
    with pytest.raises(sp.TesteBloqueadoError):
        sp.bloco(lixo, "test", cfg)


def test_teste_liberado_devolve_so_o_teste(tmp_path):
    cfg = cfg_com(tmp_path, embargo=3, liberado=True)
    t = sp.bloco(df_exemplo(), "test", cfg)
    assert len(t) == 57 and t["mes_alvo"].min() == pd.Timestamp("2021-04-01")
    assert t["mes_alvo"].max() == pd.Timestamp("2025-12-01")
    sp.exigir_teste_liberado(cfg)


def test_treino_e_validacao_nao_dependem_do_gate(tmp_path):
    cfg = cfg_com(tmp_path, embargo=3, liberado=False)
    tr, va = sp.bloco(df_exemplo(), "train", cfg), sp.bloco(df_exemplo(), "val", cfg)
    assert len(tr) == 420 and tr["mes_alvo"].max() == pd.Timestamp("2015-12-01")
    assert len(va) == 57 and va["mes_alvo"].min() == pd.Timestamp("2016-04-01")
    assert va["mes_alvo"].max() == pd.Timestamp("2020-12-01")


def test_blocos_particionam_o_dataframe_sem_perder_indice(tmp_path):
    cfg = cfg_com(tmp_path, embargo=3, liberado=True)
    df = df_exemplo()
    partes = [sp.bloco(df, b, cfg) for b in sp.BLOCOS]
    idx = [i for p in partes for i in p.index]
    assert sorted(idx) == list(df.index) and len(set(idx)) == len(idx)
    assert len(partes[sp.BLOCOS.index("embargo")]) == 6
    assert partes[0].loc[1000, "x"] == 0                                   # valores e índice preservados


def test_bloco_nao_altera_o_dataframe_original(tmp_path):
    cfg = cfg_com(tmp_path, liberado=True)
    df = df_exemplo()
    antes = df.copy()
    treino = sp.bloco(df, "train", cfg)
    treino["x"] = -1
    pd.testing.assert_frame_equal(df, antes)


def test_bloco_desconhecido_e_coluna_ausente(tmp_path):
    cfg = cfg_com(tmp_path)
    with pytest.raises(sp.SplitError, match="desconhecido"):
        sp.bloco(df_exemplo(), "validacao", cfg)
    with pytest.raises(sp.SplitError, match="ausente"):
        sp.bloco(df_exemplo().rename(columns={"mes_alvo": "mes"}), "train", cfg)
    out = sp.bloco(df_exemplo().rename(columns={"mes_alvo": "mes"}), "train", cfg, col="mes")
    assert len(out) == 420


# ================================================================== params real
def test_params_real_rotula_a_janela_inteira_sem_erro():
    cfg = load_config(PARAMS_REAL)
    rot = sp.rotular(janela_inteira(), cfg)
    assert rot.notna().all() and set(rot.unique()) == set(sp.BLOCOS)
    assert (rot == "train").sum() == 420
    assert not cfg.split.teste_liberado
    with pytest.raises(sp.TesteBloqueadoError):
        sp.bloco(df_exemplo(), "test", cfg)
