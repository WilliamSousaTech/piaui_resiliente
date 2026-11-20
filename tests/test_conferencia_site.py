"""Seção "O que choveu em <mês>" na página Previsão, texto do Método e ligação com o gerador do site.

Dados sintéticos só em tests/. Os municípios A e B têm sinal de seca na previsão sintética; C não tem."""
import copy
import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from dados_sinteticos import A, B, C
from src.data import conferencia as cf
from src.data import conferencia_outubro as co
from src.data import manifest as mf
from src.data import previsao as pr
from src.web import build_site as bs
from src.web import paginas
from test_previsao_site import com_prev, doc_do_site, pac, pagina, projeto  # noqa: F401  (fixtures)

ALVO = pd.Period("2026-10", freq="M")


def _om(out):
    return {"2026-08": {"mm": 0.0, "dias_validos": 31, "dias_no_mes": 31},
            "2026-09": {"mm": 1.0, "dias_validos": 30, "dias_no_mes": 30},
            "2026-10": {"mm": out, "dias_validos": 31 if out is not None else 30, "dias_no_mes": 31}}


def conf_doc(sha, chuva=None, normal=20.0):
    """A: 5 mm (seco, percentil ~16); B: 9 mm (fora da faixa seca, ~27); C: 40 mm (sem sinal)."""
    prev = pr.Previsao.model_validate(doc_do_site(sha))
    sinal = {m.cod_ibge for m in prev.municipios if m.spi3.alerta or m.spi6.alerta}
    chuva = chuva or {A: _om(5.0), B: _om(9.0), C: _om(40.0)}
    cods = [A, B, C]
    chirps = pd.DataFrame({c: [0.5, 2.0] for c in cods}, index=pd.period_range("2026-08", "2026-09", freq="M"))
    regs = co.montar_registros(cods, chuva, chirps, ALVO, pd.Series({c: normal for c in cods}),
                               {c: np.arange(35, dtype=float) for c in cods}, sinal)
    doc = co.montar_documento(regs, alvo=ALVO, ini=date(2026, 8, 1), fim=date(2026, 10, 31), hoje=date(2026, 11, 20),
                              dataset_sha256=sha, janela_historica="1981-01 a 2015-12", n_invalidos=0)
    return json.loads(json.dumps(doc))


def conf_pac(sha="a" * 64, **kw):
    return cf.para_pacote(cf.Conferencia.model_validate(conf_doc(sha, **kw)))


@pytest.fixture
def com_conf(com_prev):
    return {**com_prev, "conferencia": conf_pac(com_prev["previsao"]["dataset_sha256"])}


def secao(h):
    i = h.index("<h2>O que choveu")
    return h[h.rfind("<section", 0, i):h.index("</section>", i) + 10]


# ------------------------------------------------------------------ a seção da página Previsão
def test_secao_traz_fatos_tabela_fonte_e_aviso(com_conf):
    s = secao(pagina(com_conf))
    assert "Conferência indicativa, não o acerto oficial." in s
    assert "<strong>1 de 2</strong>" in s and "entre os 25% mais secos de 1981 a 2015" in s     # A seco; B não
    assert "Nos outros 1, foram 0 (0%)." in s
    assert "Weather data by Open-Meteo.com (CC BY 4.0)" in s and 'href="https://open-meteo.com/"' in s
    assert "Chuva diária de 01/08/2026 a 31/10/2026, buscada em 20/11/2026" in s
    assert "Conferência indicativa: a chuva vem de outra fonte" in s                            # os avisos do arquivo aparecem
    assert "agosto e setembro" in s and "{{" not in s


def test_tabela_so_tem_municipios_com_sinal_do_mais_seco_ao_menos_seco(com_conf):
    nome = {m["cod"]: m["nome"] for m in com_conf["municipios"]}
    s = secao(pagina(com_conf))
    corpo = s.split("<tbody>")[1].split("</tbody>")[0]
    linhas = corpo.split("<tr>")[1:]
    assert len(linhas) == 2 and nome[A] in linhas[0] and nome[B] in linhas[1]
    assert "Outubro seco" in linhas[0] and "Fora da faixa seca" in linhas[1] and "Outubro seco" not in linhas[1]
    assert nome[C] not in corpo                                # C não tem sinal: fica fora da tabela
    assert "5,0" in linhas[0] and "20,0" in linhas[0] and "−15,0" in linhas[0] and "25%" in linhas[0]
    assert "84%" in linhas[0]                                  # percentil 15,7 => mais seco que 84,3% => 84%
    assert f'href="monitor.html#m={A}"' in linhas[0]


def test_percentual_da_normal_some_quando_a_normal_e_baixa(payload, com_prev):
    pc = conf_pac(com_prev["previsao"]["dataset_sha256"], normal=3.0)
    s = secao(pagina({**com_prev, "conferencia": pc}))
    assert "% do normal" in s and "normal baixa demais para uma porcentagem" in s
    assert "167%" not in s                                    # 5 mm ÷ 3 mm enganaria


def test_municipio_com_dia_faltante_aparece_sem_numero(com_prev):
    pc = conf_pac(com_prev["previsao"]["dataset_sha256"], chuva={A: _om(None), B: _om(9.0), C: _om(40.0)})
    s = secao(pagina({**com_prev, "conferencia": pc}))
    assert "dias sem dado na outra fonte: sem conferência" in s and "Outubro seco" not in s


def test_nome_de_municipio_e_escapado_na_conferencia(com_conf):
    p = copy.deepcopy(com_conf)
    p["municipios"][0]["nome"] = '<script>alert(1)</script>'
    s = secao(pagina(p))
    assert "<script>alert(1)" not in s and "&lt;script&gt;" in s


def test_sem_conferencia_nada_aparece(com_prev):
    h = pagina(com_prev)
    assert "O que choveu" not in h and "Open-Meteo" not in h and "{{" not in h
    assert "conferência indicativa" not in h


def test_conferencia_de_municipio_que_nao_esta_no_site_e_erro(com_conf):
    p = copy.deepcopy(com_conf)
    p["municipios"] = [m for m in p["municipios"] if m["cod"] != A]
    with pytest.raises(paginas.PaginaError, match="não está no site"):
        pagina(p)


def test_aviso_da_previsao_aponta_para_a_secao(com_conf):
    assert "a seção “O que choveu em outubro” traz uma conferência indicativa" in pagina(com_conf)


def test_secao_vem_antes_da_tabela_de_municipios(com_conf):
    h = pagina(com_conf)
    assert h.index("O que choveu em outubro nos municípios") < h.index("do mais seco ao mais úmido")


def test_sem_nenhum_municipio_com_dado_a_secao_some(com_prev):
    d = conf_doc(com_prev["previsao"]["dataset_sha256"], chuva={A: _om(None), B: _om(None), C: _om(40.0)})
    pc = cf.para_pacote(cf.Conferencia.model_validate(d))
    assert "O que choveu" not in pagina({**com_prev, "conferencia": pc})


# ------------------------------------------------------------------ Método
def test_metodo_explica_a_segunda_fonte_so_quando_ha_conferencia(com_conf, com_prev):
    h = pagina(com_conf, "metodo")
    assert "Segunda fonte de chuva, só para conferir" in h and "Núcleo auditável." in h and "Conferência com outra fonte." in h
    assert "não altera os dados nem o SHA-256" in h and "teste final continua fechado" in h
    assert "A página Previsão traz também uma conferência indicativa de outubro de 2026" in h
    assert "{{" not in h
    sem = pagina(com_prev, "metodo")
    assert "Segunda fonte de chuva" not in sem and "Open-Meteo" not in sem


def test_nenhuma_promessa_de_tempo_real(com_conf):
    for id_ in ("previsao", "metodo"):
        h = pagina(com_conf, id_).lower()
        assert "tempo real" not in h and "ao vivo" not in h


# ------------------------------------------------------------------ gerador do site
def escrever_conf(raiz, d):
    destino = raiz / cf.ARQUIVO_PADRAO
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(d), encoding="utf-8")


def escrever_prev(raiz, d):
    destino = raiz / pr.ARQUIVO_PADRAO
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(d), encoding="utf-8")


def sha_do(cfg):
    return mf.verificar_manifest(cfg)["dataset_sha256"]


def test_site_com_previsao_e_conferencia_ganha_a_secao(projeto, tmp_path):
    cfg, raiz = projeto
    sha = sha_do(cfg)
    escrever_prev(raiz, doc_do_site(sha))
    escrever_conf(raiz, conf_doc(sha))
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["conferencia"] == "2026-10" and r["avisos"] == []
    h = (tmp_path / "out" / "previsao.html").read_text(encoding="utf-8")
    assert "O que choveu em outubro nos municípios com sinal" in h
    assert "Segunda fonte de chuva" in (tmp_path / "out" / "metodo.html").read_text(encoding="utf-8")
    dados = (tmp_path / "out" / "assets" / "dados.js").read_text(encoding="utf-8")
    assert "Open-Meteo" not in dados and "conferencia" not in dados and "percentil" not in dados      # só vira HTML


def test_site_sem_arquivo_de_conferencia_continua_igual(projeto, tmp_path):
    cfg, raiz = projeto
    escrever_prev(raiz, doc_do_site(sha_do(cfg)))
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["conferencia"] is None and r["avisos"] == []
    assert "O que choveu" not in (tmp_path / "out" / "previsao.html").read_text(encoding="utf-8")


def test_conferencia_sem_previsao_e_ignorada_sem_aviso(projeto, tmp_path):
    cfg, raiz = projeto
    escrever_conf(raiz, conf_doc(sha_do(cfg)))
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["conferencia"] is None and r["previsao"] is None and r["avisos"] == []
    assert not (tmp_path / "out" / "previsao.html").exists()


@pytest.mark.parametrize("mexer, trecho", [
    (lambda d: d.update(previsao_dataset_sha256="ef" * 32), "dataset"),
    (lambda d: d.update(mes_alvo="2026-11"), "período"),
    (lambda d: d["grupos"]["com_sinal"].update(n_secos=0), "contagens"),
    (lambda d: d["fonte_conferencia"].update(atribuicao="sem a fonte"), "atribuição"),
])
def test_conferencia_incoerente_ou_de_outra_previsao_e_ignorada_com_aviso(projeto, tmp_path, mexer, trecho):
    cfg, raiz = projeto
    sha = sha_do(cfg)
    escrever_prev(raiz, doc_do_site(sha))
    d = conf_doc(sha)
    mexer(d)
    escrever_conf(raiz, d)
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["conferencia"] is None and any("conferência de 2026-10 ignorada" in a and trecho in a for a in r["avisos"]), r["avisos"]
    assert r["previsao"] == "2026-10" and (tmp_path / "out" / "previsao.html").is_file()           # a página segue de pé
    assert "O que choveu" not in (tmp_path / "out" / "previsao.html").read_text(encoding="utf-8")


def test_conferencia_com_municipios_diferentes_da_previsao_e_ignorada(projeto, tmp_path):
    cfg, raiz = projeto
    sha = sha_do(cfg)
    escrever_prev(raiz, doc_do_site(sha))
    d = conf_doc(sha, chuva={A: _om(5.0), B: _om(9.0), C: _om(40.0)})
    d["municipios"][2]["cod_ibge"] = "2299999"
    escrever_conf(raiz, d)
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["conferencia"] is None and any("mesmos municípios" in a for a in r["avisos"])


def test_conferencia_que_marca_outros_municipios_com_sinal_e_ignorada(projeto, tmp_path):
    cfg, raiz = projeto
    sha = sha_do(cfg)
    escrever_prev(raiz, doc_do_site(sha))
    d = conf_doc(sha)
    for m in d["municipios"]:                                  # troca quem tem sinal e refaz os grupos para ficarem coerentes
        m["sinal_modelo"] = m["cod_ibge"] == C
    comsinal = [m for m in d["municipios"] if m["sinal_modelo"]]
    d["grupos"] = {"com_sinal": cf.Grupo(n=1, n_com_dado=1, n_secos=0, fracao_secos=0.0,
                                         mediana_percentil=comsinal[0]["percentil"], mediana_anomalia_mm=comsinal[0]["anomalia_mm"],
                                         n_abaixo_da_normal=0).model_dump(),
                   "sem_sinal": cf.Grupo(n=2, n_com_dado=2, n_secos=1, fracao_secos=0.5, mediana_percentil=None,
                                         mediana_anomalia_mm=None, n_abaixo_da_normal=2).model_dump()}
    outros = [m for m in d["municipios"] if not m["sinal_modelo"]]
    d["grupos"]["sem_sinal"]["mediana_percentil"] = float(np.median([m["percentil"] for m in outros]))
    d["grupos"]["sem_sinal"]["mediana_anomalia_mm"] = float(np.median([m["anomalia_mm"] for m in outros]))
    escrever_conf(raiz, d)
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["conferencia"] is None and any("não são os que a previsão marcou" in a for a in r["avisos"]), r["avisos"]


def test_cli_do_site_imprime_a_conferencia(projeto, tmp_path, capsys):
    cfg, raiz = projeto
    sha = sha_do(cfg)
    escrever_prev(raiz, doc_do_site(sha))
    escrever_conf(raiz, conf_doc(sha))
    assert bs.main(["--params", str(raiz / "params.yaml"), "--saida", str(tmp_path / "o")]) == 0
    assert "conferência indicativa de 2026-10 com outra fonte de chuva (Open-Meteo)" in capsys.readouterr().out
