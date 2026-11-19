"""Contexto do El Niño: o arquivo versionado é válido e o validador recusa o que sai da forma."""
import copy
import json
from datetime import date
from pathlib import Path

import pytest

from src.data import contexto_enso as ce

REAL = Path(__file__).resolve().parents[1] / "data" / "reference" / "contexto_enso.json"


def bruto():
    return json.loads(REAL.read_text(encoding="utf-8"))


def com(tmp_path, d) -> Path:
    p = tmp_path / "c.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_arquivo_versionado_e_valido_e_tem_fonte_e_data_em_cada_numero():
    c = ce.carregar(REAL)
    assert c.situacao.fonte == "cpc"
    f = c.fonte("cpc")
    assert f.emitido_em == date(2026, 11, 12) and f.url.startswith("https://www.cpc.ncep.noaa.gov/")
    assert c.situacao.nino34_c == 2.5 and c.situacao.prob_forte_ou_muito_forte_min_pct == 79
    assert c.situacao.prob_evento_historico_pct == 60
    assert all(x.fonte in {g.id for g in c.fontes} for x in c.leituras)


def test_nao_ha_numero_nem_probabilidade_inventada_fora_da_situacao():
    """Só a `situacao` carrega números do CPC; as leituras do INMET são texto, sem percentuais."""
    c = ce.carregar(REAL)
    assert not any("%" in x.texto for x in c.leituras)


def test_desatualizado_depois_da_proxima_discussao():
    c = ce.carregar(REAL)
    assert not c.desatualizado(date(2026, 11, 21))
    assert not c.desatualizado(date(2026, 12, 10))
    assert c.desatualizado(date(2026, 12, 11))


def test_para_pacote_e_json_puro_com_datas_iso():
    p = ce.para_pacote(ce.carregar(REAL))
    json.dumps(p, allow_nan=False)
    assert p["fontes"][0]["emitido_em"] == "2026-11-12" and p["obtido_em"] == "2026-11-18"


def test_arquivo_ausente_e_json_quebrado(tmp_path):
    with pytest.raises(ce.ContextoEnsoError, match="não encontrado"):
        ce.carregar(tmp_path / "nao_existe.json")
    q = tmp_path / "x.json"
    q.write_text("{", encoding="utf-8")
    with pytest.raises(ce.ContextoEnsoError, match="não consegui ler"):
        ce.carregar(q)


def mutar(tmp_path, f):
    d = copy.deepcopy(bruto())
    f(d)
    return com(tmp_path, d)


@pytest.mark.parametrize("nome,f,trecho", [
    ("fonte citada ausente", lambda d: d["situacao"].update(fonte="noaa"), "não está em `fontes`"),
    ("id repetido", lambda d: d["fontes"][1].update(id="cpc"), "repetidos"),
    ("http simples", lambda d: d["fontes"][0].update(url="http://x.gov/a"), "https"),
    ("link com espaço", lambda d: d["fontes"][0].update(url="https://x.gov/a b"), "https"),
    ("pct acima de 100", lambda d: d["situacao"].update(prob_evento_historico_pct=101), "less than or equal to 100"),
    ("pct negativo", lambda d: d["situacao"].update(prob_forte_ou_muito_forte_min_pct=-1), "greater than or equal to 0"),
    ("campo extra", lambda d: d.update(inventado=1), "Extra inputs"),
    ("sem leituras", lambda d: d.update(leituras=[]), "at least 1"),
    ("fonte do futuro", lambda d: d["fontes"][0].update(emitido_em="2026-12-01"), "depois de `obtido_em`"),
    ("data torta", lambda d: d["fontes"][0].update(emitido_em="12/11/2026"), "date"),
    ("texto curto demais", lambda d: d["leituras"][0].update(texto="chove"), "at least 10"),
])
def test_forma_invalida_e_recusada(tmp_path, nome, f, trecho):
    with pytest.raises(ce.ContextoEnsoError, match=trecho):
        ce.carregar(mutar(tmp_path, f))


def test_modulo_nao_importa_rede_web_nem_ml():
    src = (Path(ce.__file__)).read_text(encoding="utf-8")
    for proibido in ("requests", "urllib", "torch", "fastapi", "flask", "sqlalchemy", "sqlite3"):
        assert f"import {proibido}" not in src and f"from {proibido}" not in src
