"""Página Previsão e a ligação com o gerador do site. Dados sintéticos só em tests/."""
import copy
import json

import pytest

from dados_sinteticos import A, B, C, raw_sintetico
from src.data import manifest as mf
from src.data import previsao as pr
from src.web import build_site as bs
from src.web import paginas
from test_build_site import montar_proj
from test_previsao import SHA, doc as doc_base


def doc_do_site(sha=SHA):
    d = doc_base()
    for m, cod in zip(d["municipios"], (A, B, C)):
        m["cod_ibge"] = cod
    d["dataset_sha256"] = sha
    d["cabecas"][1].update(mae=0.309, mae_persistencia=0.473, mae_parcial_climatologica=0.326, mae_mes_alvo=0.323,
                           margem_cobertura_2a_metade=0.713, secas_reais=126, secas_pegas=37, alarmes_falsos=93,
                           revocacao=0.29365, precisao=0.28462, veredito_melhor_baseline="superou")
    d["cabecas"][0].update(revocacao=0.25498, precisao=0.5614)
    d["municipios"][1]["spi6"] = {"previsto": -1.5, "dispersao": 0.04, "baixa_conf": False, "alerta": True}     # B: o menor dos dois é o SPI-6
    d["municipios"][2]["spi6"]["baixa_conf"] = False
    d["resumo"].update(n_baixa_conf_spi6=0)
    return d


def pacote(d=None):
    return pr.para_pacote(pr.Previsao.model_validate(d or doc_do_site()))


@pytest.fixture
def pac():
    return pacote()


@pytest.fixture
def com_prev(payload, pac):
    return {**payload, "previsao": pac, "tem_previsao": True}


def pagina(payload, id_="previsao"):
    lista = paginas.paginas_do_site(payload)
    p = next(x for x in lista if x.id == id_)
    return paginas.renderizar_pagina(p, lista, payload)


def linhas(h):
    corpo = h.split('<tbody id="corpoPrev">')[1].split("</tbody>")[0]
    return [x for x in corpo.split("<tr ")[1:]]


# ------------------------------------------------------------------ lista de páginas e textos do site
def test_previsao_entra_entre_municipios_e_el_nino_e_planejamento_continua_antes_do_metodo(payload, pac):
    assert paginas.paginas_do_site({**payload, "previsao": None}) == paginas.paginas_do_site(payload)
    assert [p.id for p in paginas.paginas_do_site({**payload, "previsao": pac})] == \
        ["inicio", "monitor", "municipios", "previsao", "el-nino", "metodo", "sobre"]
    com_os_dois = {**payload, "previsao": pac, "planejamento": {"rascunhos": [1]}}
    assert [p.id for p in paginas.paginas_do_site(com_os_dois)] == \
        ["inicio", "monitor", "municipios", "previsao", "el-nino", "planejamento", "metodo", "sobre"]


def test_sem_previsao_o_site_continua_dizendo_que_ela_esta_indisponivel(payload):
    for nome in ("metodo", "sobre"):
        h = pagina(payload, nome)
        assert "Indisponível" in h and "Previsão" not in paginas.html_nav(paginas.paginas_do_site(payload), nome)
    assert "não uma previsão" in pagina(payload, "metodo")                                   # aviso do topo


def test_com_previsao_o_aviso_do_topo_o_metodo_e_o_sobre_apontam_para_a_pagina(com_prev):
    for nome in ("metodo", "sobre", "inicio"):
        h = pagina(com_prev, nome)
        assert "a previsão do modelo está na página Previsão, com margem de erro" in h and "previsão de chuva da Open-Meteo, outra fonte" in h
        assert '<a href="previsao.html">Previsão</a>' in h
    sobre, metodo = pagina(com_prev, "sobre"), pagina(com_prev, "metodo")
    assert "Disponível só como experimento, na página Previsão" in sobre and "o teste final ainda não foi aberto" in sobre
    assert "Só como experimento: a página Previsão mostra o SPI de outubro de 2026" in metodo
    assert "Previsão com margem de erro</b><span>Indisponível" not in sobre


def test_aviso_do_topo_nao_promete_previsao_quando_nao_ha():
    assert "não uma previsão" in paginas.html_aviso("MVP experimental")
    assert "não uma previsão" not in paginas.html_aviso("MVP experimental", True)
    assert "página Previsão" in paginas.html_aviso("MVP experimental", True)


def test_dados_js_leva_so_o_sinal_de_que_ha_previsao(com_prev, payload):
    com = bs._dados_js(com_prev)
    sem = bs._dados_js(payload)
    assert '"tem_previsao":true' in com and '"previsao"' not in com and "n_alerta" not in com
    assert '"tem_previsao":false' in sem or '"tem_previsao"' not in sem            # o payload de teste vem sem a chave
    assert "D.tem_previsao" in (bs.ASSETS / "monitor.js").read_text(encoding="utf-8")


# ------------------------------------------------------------------ a página
def test_pagina_traz_titulo_aviso_fatos_e_nenhum_marcador_solto(com_prev):
    h = pagina(com_prev)
    assert "<h1 class=\"titulo-pagina\">Previsão do próximo mês</h1>" in h
    assert "Experimento, não alerta." in h and "MVP experimental" in h
    assert "O mês previsto, outubro de 2026, já passou: a chuva medida que o site tem termina em setembro de 2026" in h
    assert "<strong>2 de 3</strong>" in h and "1 pelo SPI-3 e 1 pelo SPI-6" in h
    assert "<strong>±0,66 e ±0,55</strong>" in h and "80% dos erros ficaram dentro dela" in h
    assert "<strong>25% e 29%</strong>" in h and "acertou 56% e 28%" in h                           # 25,498% e 29,365%: sem arredondar duas vezes
    assert "{{" not in h and "}}" not in h
    assert 'src="assets/previsao.js"' in h


def test_alvo_que_ainda_nao_passou_muda_o_texto(payload):
    d = doc_do_site()
    d["alvo_ja_passou"] = False
    h = pagina({**payload, "previsao": pacote(d)})
    assert "ainda não tem chuva medida no site" in h and "já passou" not in h


def test_linhas_do_mais_seco_ao_mais_umido_com_sinal_e_baixa_confianca(com_prev):
    ls = linhas(pagina(com_prev))
    assert len(ls) == 3
    nomes = [x.split("</a>")[0].rsplit(">", 1)[1] for x in ls]
    assert nomes == ["Beta do Sul", "Alfa do Norte", "Tanque do Piauí"]                        # min(spi3, spi6): −1,5; −1,2; 0,2
    assert 'data-sinal="1"' in ls[0] and 'data-sinal="1"' in ls[1] and 'data-sinal="0"' in ls[2]
    assert "Sinal de seca: SPI-6<" in ls[0] and "Sinal de seca: SPI-3<" in ls[1] and "Sem sinal" in ls[2]
    assert ls[0].count("baixa confiança") == 0 and ls[1].count("baixa confiança") == 1 and ls[2].count("baixa confiança") == 0
    assert f'href="monitor.html#m={B}"' in ls[0] and f'href="monitor.html#m={A}"' in ls[1]
    assert "−1,20" in ls[1] and "−0,50" in ls[1] and "0,40" in ls[2] and "−1,50" in ls[0]          # sinal de menos de verdade, duas casas
    assert "faixa provável de −1,86 a −0,54" in ls[1]                                            # previsto ± margem (0,66)
    mista = paginas.html_previsao_tabela({**com_prev["previsao"], "municipios": [
        {**com_prev["previsao"]["municipios"][0], "spi6": {**com_prev["previsao"]["municipios"][0]["spi6"], "previsto": -1.0, "alerta": True}}]}, com_prev["municipios"])
    assert "Sinal de seca: SPI-3 e SPI-6<" in mista


def test_faixa_desenha_ponto_barra_e_zona_nas_posicoes_certas(com_prev):
    cab = com_prev["previsao"]["cabecas"][0]
    cel = {"previsto": 0.0, "dispersao": 0, "baixa_conf": False, "alerta": False}
    f = paginas._faixa(cel, cab)
    assert 'class="pt" style="left:50.0%"' in f                                                 # SPI 0 no meio
    assert 'class="zona" style="width:30.0%"' in f                                              # limiar −1,0 na escala −2,5..2,5
    assert 'class="mg" style="left:36.8%;width:26.4%"' in f                                     # ±0,66
    f2 = paginas._faixa({**cel, "previsto": -4.0}, cab)
    assert 'class="pt" style="left:0.0%"' in f2 and 'class="mg" style="left:0.0%;width:' in f2   # corta no fim da escala
    assert paginas._pos_faixa(9) == 100.0 and paginas._pos_faixa(-9) == 0.0 and paginas._pos_faixa(0) == 50.0


def test_nome_de_municipio_e_escapado(payload, pac):
    municipios = copy.deepcopy(payload["municipios"])
    municipios[0]["nome"] = "<script>x</script> & Cia"
    h = paginas.html_previsao_tabela(pac, municipios)
    assert "<script>" not in h and "&lt;script&gt;x&lt;/script&gt; &amp; Cia" in h


def test_municipio_da_previsao_que_nao_esta_no_site_e_erro(payload, pac):
    with pytest.raises(paginas.PaginaError, match="não está no site"):
        paginas.html_previsao_tabela(pac, payload["municipios"][:2])


def test_confianca_traz_os_numeros_do_arquivo(com_prev):
    h = pagina(com_prev)
    for trecho in ("Erro médio no SPI-3: 0,40", "Erro médio no SPI-6: 0,31", "repetir o mês anterior (0,64)", "repetir o mês anterior (0,47)",
                   "(0,41)", "(0,33)", "Só nos meses de outubro da validação, o erro médio foi 0,56 no SPI-3 e 0,32 no SPI-6, contra 0,40 e 0,31",
                   "Pegou 64 de 251 secas reais no SPI-3 e 37 de 126 no SPI-6, com 50 e 93 alarmes falsos",
                   "Sem sinal de seca não quer dizer sem seca", "cobriu 76% dos erros da segunda (SPI-3) e 71% (SPI-6)",
                   "A segunda metade começa em setembro de 2018", "Em outubro se espera pouca chuva em 1 município no SPI-3 e 0 no SPI-6",
                   "O teste final ainda não foi aberto", "Em 19/11/2026", "até dezembro de 2025"):
        assert trecho in h, trecho


def test_margem_instavel_so_e_dita_quando_e(payload):
    d = doc_do_site()
    assert [c["margem_estavel"] for c in d["cabecas"]] == [True, False]
    assert "No SPI-6 ela não se manteve estável entre as duas metades" in pagina({**payload, "previsao": pacote(d)})
    d["cabecas"][1]["margem_estavel"] = True
    assert "não se manteve estável" not in pagina({**payload, "previsao": pacote(d)})
    d["cabecas"][0]["margem_estavel"] = False
    assert "No SPI-3 ela não se manteve estável entre as duas metades" in pagina({**payload, "previsao": pacote(d)})
    d["cabecas"][1]["margem_estavel"] = False
    assert "No SPI-3 e SPI-6 ela não se manteve estável" in pagina({**payload, "previsao": pacote(d)})


@pytest.mark.parametrize("ver,trecho", [("superou", "errou um pouco mais que a previsão"), ("pior", "errou menos que a previsão"),
                                        ("indistinguivel", "não há diferença clara"), ("indefinido", "não é conclusiva")])
def test_veredito_contra_o_baseline_muda_a_frase(payload, ver, trecho):
    d = doc_do_site()
    d["cabecas"][0]["veredito_melhor_baseline"] = ver
    h = pagina({**payload, "previsao": pacote(d)})
    erro_spi3 = h.split("Erro médio no SPI-3")[1].split("</li>")[0]
    assert trecho in erro_spi3


@pytest.mark.parametrize("ver,trecho", [("superou", "errou menos que repetir"), ("pior", "errou mais que repetir"),
                                        ("indistinguivel", "sem diferença clara de repetir"), ("indefinido", "não deu para comparar com repetir")])
def test_veredito_contra_a_persistencia_muda_a_frase(payload, ver, trecho):
    d = doc_do_site()
    d["cabecas"][1]["veredito_persistencia"] = ver
    assert trecho in pagina({**payload, "previsao": pacote(d)}).split("Erro médio no SPI-6")[1].split("</li>")[0]


def test_baseline_que_nao_e_o_parcial_nao_inventa_numero(payload):
    d = doc_do_site()
    d["cabecas"][0]["melhor_baseline"] = "ridge"
    erro = pagina({**payload, "previsao": pacote(d)}).split("Erro médio no SPI-3")[1].split("</li>")[0]
    assert "a regressão linear (Ridge)" in erro and "(0,41)" not in erro


def test_pct_e_municipios_em_texto():
    assert paginas._pct_br(0.25498) == "25" and paginas._pct_br(0.255) == "26" and paginas._pct_br(1.0) == "100"
    assert paginas._n_municipios(1) == "1 município" and paginas._n_municipios(0) == "0 municípios" and paginas._n_municipios(91) == "91 municípios"


def test_legenda_explica_ponto_barra_zona_e_zero(com_prev):
    h = pagina(com_prev)
    for t in ("<b>Ponto</b>", "<b>Barra</b>", "<b>Fundo terracota</b>", "<b>Linha fina</b>", "−1,00 no SPI-3"):
        assert t in h


# ------------------------------------------------------------------ gerador do site
def escrever(raiz, d):
    destino = raiz / pr.ARQUIVO_PADRAO
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(d), encoding="utf-8")


@pytest.fixture
def projeto(tmp_path):
    return montar_proj(tmp_path, raw_sintetico())


def test_site_com_previsao_conferida_ganha_a_pagina(projeto, tmp_path):
    cfg, raiz = projeto
    escrever(raiz, doc_do_site(mf.verificar_manifest(cfg)["dataset_sha256"]))
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["previsao"] == "2026-10" and r["avisos"] == []
    h = (tmp_path / "out" / "previsao.html").read_text(encoding="utf-8")
    assert "Alfa do Norte" in h and "Experimento, não alerta." in h
    assert 'href="previsao.html"' in (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert (tmp_path / "out" / "assets" / "previsao.js").is_file()
    dados = (tmp_path / "out" / "assets" / "dados.js").read_text(encoding="utf-8")
    assert '"tem_previsao":true' in dados and "n_alerta" not in dados


def test_site_sem_arquivo_de_previsao_nao_tem_a_pagina_nem_aviso(projeto, tmp_path):
    cfg, _ = projeto
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["previsao"] is None and not any("previsão" in a for a in r["avisos"])
    assert not (tmp_path / "out" / "previsao.html").exists() and not (tmp_path / "out" / "assets" / "previsao.js").exists()
    assert '"tem_previsao":false' in (tmp_path / "out" / "assets" / "dados.js").read_text(encoding="utf-8")


def test_previsao_de_outro_dataset_e_ignorada_com_aviso(projeto, tmp_path):
    cfg, raiz = projeto
    escrever(raiz, doc_do_site("ef" * 32))
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["previsao"] is None and any("previsão do próximo mês ignorada" in a and "outro" not in a or "dataset" in a for a in r["avisos"])
    assert not (tmp_path / "out" / "previsao.html").exists()


def test_previsao_de_outros_municipios_e_ignorada_com_aviso(projeto, tmp_path):
    cfg, raiz = projeto
    d = doc_do_site(mf.verificar_manifest(cfg)["dataset_sha256"])
    d["municipios"] = d["municipios"][:2]
    d["resumo"].update(n_municipios=2, n_alerta_algum=2, n_alerta_spi3=1, n_alerta_spi6=1, n_baixa_conf_spi3=1, n_baixa_conf_spi6=0)
    escrever(raiz, d)
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["previsao"] is None and any("mesmos municípios" in a for a in r["avisos"])


def test_previsao_invalida_e_ignorada_sem_derrubar_o_site(projeto, tmp_path):
    cfg, raiz = projeto
    d = doc_do_site(mf.verificar_manifest(cfg)["dataset_sha256"])
    d["municipios"][0]["spi3"]["alerta"] = False                                  # alerta que não bate com o número
    escrever(raiz, d)
    r = bs.gerar_site(cfg, tmp_path / "out")
    assert r["previsao"] is None and any("não bate" in a for a in r["avisos"])
    assert (tmp_path / "out" / "index.html").is_file()


def test_cli_do_site_imprime_a_previsao(projeto, tmp_path, capsys):
    cfg, raiz = projeto
    escrever(raiz, doc_do_site(mf.verificar_manifest(cfg)["dataset_sha256"]))
    assert bs.main(["--params", str(raiz / "params.yaml"), "--saida", str(tmp_path / "o")]) == 0
    assert "página Previsão: previsão de 2026-10" in capsys.readouterr().out
