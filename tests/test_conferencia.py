"""Leitor da conferência: aceita o que o script grava, recusa arquivo adulterado ou de outro conjunto de dados."""
import copy
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import conferencia as cf
from src.data import conferencia_outubro as co

RAIZ = Path(__file__).resolve().parents[1]
ALVO = pd.Period("2026-10", freq="M")
SHA = "a" * 64
CODS = [f"22000{i:02d}" for i in range(1, 9)]
SINAL = {"2200001", "2200002"}


def _om(out):
    return {"2026-08": {"mm": 0.0, "dias_validos": 31, "dias_no_mes": 31},
            "2026-09": {"mm": 1.0, "dias_validos": 30, "dias_no_mes": 30},
            "2026-10": {"mm": out, "dias_validos": 31 if out is not None else 30, "dias_no_mes": 31}}


def _doc(com_buraco=False):
    """Documento no formato do script (o contrato escritor→leitor é o que este teste protege)."""
    chuva = {c: _om(5.0 if c in SINAL else 25.0 + i) for i, c in enumerate(CODS)}
    if com_buraco:
        chuva[CODS[-1]] = _om(None)
    chirps = pd.DataFrame({c: [0.5, 2.0] for c in CODS}, index=pd.period_range("2026-08", "2026-09", freq="M"))
    normal = pd.Series({c: 20.0 for c in CODS})
    hist = {c: np.arange(35, dtype=float) for c in CODS}
    regs = co.montar_registros(CODS, chuva, chirps, ALVO, normal, hist, SINAL)
    doc = co.montar_documento(regs, alvo=ALVO, ini=date(2026, 8, 1), fim=date(2026, 10, 31), hoje=date(2026, 11, 20),
                              dataset_sha256=SHA, janela_historica="1981-01 a 2015-12", n_invalidos=0)
    return json.loads(json.dumps(doc))


def _validar(doc):
    return cf.Conferencia.model_validate(doc)


def test_aceita_o_que_o_script_grava():
    c = _validar(_doc())
    assert c.mes_alvo == "2026-10" and c.n_municipios == 8 and {r.cod_ibge for r in c.com_sinal} == SINAL
    assert c.grupos.com_sinal.n == 2 and c.grupos.com_sinal.n_secos == 2 and c.grupos.sem_sinal.n_secos == 0


def test_aceita_municipio_incompleto():
    c = _validar(_doc(com_buraco=True))
    assert c.n_incompletos == 1 and not c.municipios[-1].completo


def _mut(f):
    d = copy.deepcopy(_doc())
    f(d)
    return d


MUTACOES = {
    "versao": (lambda d: d.update(versao=2), "versao"),
    "campo extra": (lambda d: d.update(extra=1), "extra"),
    "sha curto": (lambda d: d.update(previsao_dataset_sha256="abc"), "previsao_dataset_sha256"),
    "n_municipios": (lambda d: d.update(n_municipios=9), "n_municipios"),
    "municipio repetido": (lambda d: d["municipios"].__setitem__(1, d["municipios"][0]), "repetido"),
    "sem atribuição": (lambda d: d["fonte_conferencia"].update(atribuicao="Dados de chuva do clima aberto"), "Open-Meteo"),
    "licença trocada": (lambda d: d["fonte_conferencia"].update(licenca="CC0"), "licenca"),
    "url sem https": (lambda d: d["fonte_conferencia"].update(url="http://open-meteo.com/"), "https"),
    "período não termina no alvo": (lambda d: d["fonte_conferencia"].update(periodo=["2026-08-01", "2026-09-30"]), "período"),
    "gerado antes do fim": (lambda d: d.update(gerado_em="2026-10-15"), "período"),
    "alvo na comparação": (lambda d: d["comparacao_fontes"].__setitem__("2026-10", d["comparacao_fontes"]["2026-09"]), "mês-alvo"),
    "chave de mês ruim": (lambda d: d["comparacao_fontes"].__setitem__("set", d["comparacao_fontes"]["2026-09"]), "AAAA-MM"),
    "completo mentindo": (lambda d: d["municipios"][0].update(completo=False), "completo"),
    "incompleto com percentil": (lambda d: (d["municipios"][0]["chuva_mm"].update({"2026-10": None}),
                                            d["municipios"][0].update(completo=False)), "incompleto"),
    "n_incompletos": (lambda d: d.update(n_incompletos=3), "n_incompletos"),
    "desvio errado": (lambda d: d["municipios"][0].update(anomalia_mm=d["municipios"][0]["anomalia_mm"] + 5), "desvio"),
    "% da normal errada": (lambda d: d["municipios"][0].update(pct_normal=300.0), "% da normal"),
    "contagem de secos": (lambda d: d["grupos"]["com_sinal"].update(n_secos=1), "contagens"),
    "grupo trocado": (lambda d: d["municipios"][0].update(sinal_modelo=False), "contagens"),
    "fração": (lambda d: d["grupos"]["com_sinal"].update(fracao_secos=0.5), "fração"),
    "mediana do percentil": (lambda d: d["grupos"]["sem_sinal"].update(mediana_percentil=99.0), "mediana"),
    "mediana do desvio": (lambda d: d["grupos"]["com_sinal"].update(mediana_anomalia_mm=-99.0), "mediana"),
    "percentil fora de 0-100": (lambda d: d["municipios"][0].update(percentil=120.0), "percentil"),
    "chuva negativa": (lambda d: d["municipios"][0]["chuva_mm"].update({"2026-09": -1.0}), "chuva_mm"),
    "cod curto": (lambda d: d["municipios"][0].update(cod_ibge="123"), "cod_ibge"),
    "comparação n=0 com números": (lambda d: d["comparacao_fontes"]["2026-08"].update(n=0), "n = 0"),
}


@pytest.mark.parametrize("nome", list(MUTACOES))
def test_recusa_arquivo_adulterado(nome):
    f, trecho = MUTACOES[nome]
    with pytest.raises(ValueError, match=trecho):
        _validar(_mut(f))


def test_recusa_nan_e_infinito():
    for ruim in (float("nan"), float("inf")):
        d = _doc()
        d["municipios"][0]["normal_mm"] = ruim
        with pytest.raises(ValueError):
            _validar(d)


def test_pct_normal_some_quando_normal_baixa():
    d = _doc()
    m = d["municipios"][0]
    m["normal_mm"] = 2.0                                # abaixo do mínimo: % da normal não pode existir
    m["anomalia_mm"] = round(m["chuva_mm"]["2026-10"] - 2.0, 1)       # mantém o desvio coerente para isolar a regra
    with pytest.raises(ValueError, match="% da normal"):
        _validar(d)


# ------------------------------------------------------------------ carregar e conferir contra a previsão
def test_carregar_ausente_ilegivel_e_invalido(tmp_path):
    with pytest.raises(cf.ConferenciaError, match="não encontrado"):
        cf.carregar(tmp_path / "nao_existe.json")
    (tmp_path / "ruim.json").write_text("{não é json", encoding="utf-8")
    with pytest.raises(cf.ConferenciaError, match="não consegui ler"):
        cf.carregar(tmp_path / "ruim.json")
    (tmp_path / "vazio.json").write_text("{}", encoding="utf-8")
    with pytest.raises(cf.ConferenciaError, match="inválido"):
        cf.carregar(tmp_path / "vazio.json")


def test_carregar_roundtrip_e_pacote(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(_doc()), encoding="utf-8")
    c = cf.carregar(p)
    pac = cf.para_pacote(c)
    json.dumps(pac, allow_nan=False)
    assert pac["fonte_conferencia"]["periodo"] == ["2026-08-01", "2026-10-31"] and pac["gerado_em"] == "2026-11-20"


def test_conferir_previsao_aceita_e_recusa():
    c = _validar(_doc())
    cf.conferir_previsao(c, SHA, "2026-10", SINAL, CODS)
    with pytest.raises(cf.ConferenciaError, match="dataset"):
        cf.conferir_previsao(c, "b" * 64, "2026-10", SINAL, CODS)
    with pytest.raises(cf.ConferenciaError, match="2026-11"):
        cf.conferir_previsao(c, SHA, "2026-11", SINAL, CODS)
    with pytest.raises(cf.ConferenciaError, match="mesmos municípios"):
        cf.conferir_previsao(c, SHA, "2026-10", SINAL, CODS[:-1])
    with pytest.raises(cf.ConferenciaError, match="com sinal"):
        cf.conferir_previsao(c, SHA, "2026-10", {"2200001", "2200003"}, CODS)


def test_arquivo_real_do_repositorio_e_da_previsao_real():
    arq = RAIZ / cf.ARQUIVO_PADRAO
    prev = RAIZ / "data" / "reference" / "previsao_proximo_mes.json"
    if not (arq.is_file() and prev.is_file()):
        pytest.skip("arquivos de referência ausentes")
    c = cf.carregar(arq)
    p = json.loads(prev.read_text(encoding="utf-8"))
    sinal = {m["cod_ibge"] for m in p["municipios"] if m["spi3"]["alerta"] or m["spi6"]["alerta"]}
    cf.conferir_previsao(c, p["dataset_sha256"], p["alvo"], sinal, [m["cod_ibge"] for m in p["municipios"]])
    assert c.n_municipios == p["resumo"]["n_municipios"] and len(c.com_sinal) == p["resumo"]["n_alerta_algum"]


def test_modulo_sem_rede_ml_ou_web():
    fonte = (RAIZ / "src" / "data" / "conferencia.py").read_text(encoding="utf-8")
    for proibido in ("requests", "urllib", "torch", "fastapi", "flask", "sqlalchemy", "sqlite3", "socket"):
        assert f"import {proibido}" not in fonte and f"from {proibido}" not in fonte
