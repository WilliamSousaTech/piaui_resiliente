"""Leitura e conferência de data/reference/previsao_proximo_mes.json (src/data/previsao.py). Documentos escritos à mão."""
import copy
import json
from pathlib import Path

import pytest

from src.data import previsao as pr

SHA = "ab" * 32
RAIZ = Path(__file__).resolve().parents[1]


def cabeca(escala, limiar, margem=0.66, estavel=True):
    return {"id": f"spi{escala}", "escala": escala, "nome_alvo": f"alvo_spi{escala}_h1", "limiar_alerta": limiar, "margem": margem,
            "margem_q": 0.8, "margem_cobertura_validacao": 0.8, "margem_cobertura_2a_metade": 0.76, "margem_estavel": estavel,
            "n_linhas_validacao": 6048, "mae": 0.395, "mae_persistencia": 0.638, "mae_parcial_climatologica": 0.414,
            "mae_mes_alvo": 0.562, "mae_mes_alvo_persistencia": 0.579, "veredito_persistencia": "superou",
            "melhor_baseline": "parcial_climatologica", "veredito_melhor_baseline": "indistinguivel", "acerto_classe": 0.809, "acerto_sempre_normal": 0.701,
            "secas_reais": 251, "secas_pegas": 64, "alarmes_falsos": 50, "revocacao": 0.255, "precisao": 0.561}


def cel(previsto, limiar, baixa=False):
    return {"previsto": previsto, "dispersao": 0.04, "baixa_conf": baixa, "alerta": previsto <= limiar}


def doc():
    mun = [{"cod_ibge": "2200053", "spi3": cel(-1.2, -1.0, True), "spi6": cel(-0.5, -0.95)},
           {"cod_ibge": "2200103", "spi3": cel(0.1, -1.0), "spi6": cel(-0.96, -0.95)},
           {"cod_ibge": "2200202", "spi3": cel(0.4, -1.0), "spi6": cel(0.2, -0.95, True)}]
    return {"versao": 1, "aviso": "MVP experimental, em calibração com S2iD/TCE-PI. Previsão com margem de erro.",
            "gerado_em": "2026-11-19", "origem": "2026-09", "alvo": "2026-10", "horizonte_meses": 1, "corte_b": "2018-09",
            "alvo_ja_passou": True, "teste_congelado_lido": False, "dataset_sha256": SHA,
            "modelo": {"pasta": "artifacts/modelo_v2", "n_membros": 5, "receita_sha256": "cd" * 32},
            "cabecas": [cabeca(3, -1.0), cabeca(6, -0.95, 0.549, False)],
            "resumo": {"n_municipios": 3, "n_alerta_algum": 2, "n_alerta_spi3": 1, "n_alerta_spi6": 1,
                       "n_baixa_conf_spi3": 1, "n_baixa_conf_spi6": 1},
            "municipios": mun}


def gravar(tmp_path, d):
    p = tmp_path / "previsao.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def test_documento_valido_le_e_vira_pacote(tmp_path):
    d = doc()
    prev = pr.carregar(gravar(tmp_path, d))
    assert prev.alvo == "2026-10" and prev.cabeca("spi6").margem == 0.549 and len(prev.municipios) == 3
    assert pr.para_pacote(prev) == d                                  # ida e volta sem perder nada


@pytest.mark.parametrize("mexer,trecho", [
    (lambda d: d.update(versao=2), "versão 2"),
    (lambda d: d.update(aviso="Previsão com margem de erro, ainda sem rótulo de maturidade."), "aviso de maturidade"),
    (lambda d: d.update(alvo="2026-11"), "origem"),
    (lambda d: d.update(alvo="2026-09"), "origem"),
    (lambda d: d.update(horizonte_meses=2), "origem"),
    (lambda d: d.update(origem="2026-13"), "origem"),
    (lambda d: d.update(corte_b="x"), "corte_b"),
    (lambda d: d.update(dataset_sha256="zz"), "dataset_sha256"),
    (lambda d: d.update(extra=1), "extra"),
    (lambda d: d["cabecas"].reverse(), "cabeças esperadas"),
    (lambda d: d["cabecas"].pop(), "cabeças esperadas"),
    (lambda d: d["cabecas"][0].update(escala=4), "incoerente"),
    (lambda d: d["cabecas"][0].update(nome_alvo="alvo_spi3_h2"), "incoerente"),
    (lambda d: d["cabecas"][0].update(secas_pegas=300), "maior que as reais"),
    (lambda d: d["cabecas"][0].update(margem=0), "margem"),
    (lambda d: d["cabecas"][0].update(limiar_alerta=0.5), "limiar_alerta"),
    (lambda d: d["cabecas"][0].update(revocacao=1.5), "revocacao"),
    (lambda d: d["cabecas"][0].update(veredito_persistencia="ganhou"), "veredito_persistencia"),
    (lambda d: d["cabecas"][1].update(melhor_baseline="xgboost"), "melhor_baseline"),
    (lambda d: d.update(teste_congelado_lido=True), "teste_congelado_lido"),
    (lambda d: d.pop("teste_congelado_lido"), "teste_congelado_lido"),
    (lambda d: d["municipios"].reverse(), "fora da ordem"),
    (lambda d: d["municipios"].append(copy.deepcopy(d["municipios"][-1])), "repetidos"),
    (lambda d: d["municipios"][0].update(cod_ibge="220005"), "cod_ibge"),
    (lambda d: d["municipios"][0]["spi3"].update(alerta=False), "não bate"),
    (lambda d: d["municipios"][2]["spi6"].update(alerta=True), "não bate"),
    (lambda d: d["municipios"][1]["spi6"].update(previsto=-0.94), "não bate"),
    (lambda d: d["municipios"][0]["spi3"].update(previsto=-9.0), "previsto"),
    (lambda d: d["municipios"][0]["spi3"].update(dispersao=-0.1), "dispersao"),
    (lambda d: d["municipios"][0].pop("spi6"), "spi6"),
    (lambda d: d["resumo"].update(n_alerta_algum=3), "resumo"),
    (lambda d: d["resumo"].update(n_baixa_conf_spi6=0), "resumo"),
    (lambda d: d["resumo"].update(n_municipios=216), "resumo"),
    (lambda d: d["modelo"].update(n_membros=0), "n_membros"),
    (lambda d: d["municipios"].clear(), "municipios"),
])
def test_documento_incoerente_e_recusado(tmp_path, mexer, trecho):
    d = doc()
    mexer(d)
    with pytest.raises(pr.PrevisaoError, match=trecho):
        pr.carregar(gravar(tmp_path, d))


def test_alerta_no_limiar_exato_vale(tmp_path):
    d = doc()
    d["municipios"][2]["spi6"] = cel(-0.95, -0.95)                    # igual ao limiar: alerta
    d["resumo"].update(n_alerta_spi6=2, n_alerta_algum=3, n_baixa_conf_spi6=0)
    assert pr.carregar(gravar(tmp_path, d)).municipios[2].spi6.alerta is True


def test_arquivo_ausente_e_json_estragado(tmp_path):
    with pytest.raises(pr.PrevisaoError, match="não encontrado"):
        pr.carregar(tmp_path / "nao_existe.json")
    p = tmp_path / "ruim.json"
    p.write_text("{não é json", encoding="utf-8")
    with pytest.raises(pr.PrevisaoError, match="não consegui ler"):
        pr.carregar(p)
    p.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(pr.PrevisaoError, match="inválido"):
        pr.carregar(p)


def test_nan_no_arquivo_e_recusado(tmp_path):
    d = doc()
    texto = json.dumps(d).replace('"previsto": -1.2', '"previsto": NaN', 1)
    p = tmp_path / "nan.json"
    p.write_text(texto, encoding="utf-8")
    with pytest.raises(pr.PrevisaoError):
        pr.carregar(p)


def test_conferir_dataset(tmp_path):
    prev = pr.carregar(gravar(tmp_path, doc()))
    pr.conferir_dataset(prev, SHA, ["2200053", "2200103", "2200202"])               # não levanta
    with pytest.raises(pr.PrevisaoError, match="python -m src.model.prever"):
        pr.conferir_dataset(prev, "ef" * 32, ["2200053", "2200103", "2200202"])
    with pytest.raises(pr.PrevisaoError, match="mesmos municípios"):
        pr.conferir_dataset(prev, SHA, ["2200053", "2200103"])
    with pytest.raises(pr.PrevisaoError, match="mesmos municípios"):
        pr.conferir_dataset(prev, SHA, ["2200053", "2200103", "2200202", "2200301"])


def test_arquivo_real_do_repositorio_passa_na_conferencia():
    caminho = RAIZ / pr.ARQUIVO_PADRAO
    if not caminho.is_file():
        pytest.skip("previsão ainda não gerada")
    prev = pr.carregar(caminho)
    assert prev.resumo.n_municipios == len(prev.municipios) == 216
    assert prev.horizonte_meses == 1 and "MVP" in prev.aviso
