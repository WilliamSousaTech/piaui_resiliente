"""Testes de src/web/paginas.py e da pasta do site montada (offline, links, ids do JS, cores e contraste).

Os dados vêm de tests/dados_sinteticos.py (3 municípios de mentira). O contexto do El Niño é o arquivo real de
data/reference/contexto_enso.json, que é transcrição de fonte oficial e já vai no repositório.
"""
import copy
import dataclasses
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from dados_sinteticos import RAIZ, VERIFICADO
from src.data import contexto_enso as ce
from src.web import build_site as bs
from src.web import paginas
from src.web.paginas import Html, PaginaError
from util_cor import contraste

ARQUIVOS = [p.arquivo for p in paginas.PAGINAS]
ASSETS = RAIZ / "src" / "web" / "assets"


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def enso():
    return ce.carregar(RAIZ / ce.ARQUIVO_PADRAO)


@pytest.fixture(scope="module")
def payload_enso(insumos, cfg, enso):
    return bs.montar_payload(dataclasses.replace(insumos, enso=enso), cfg, verificado_em=VERIFICADO)


@pytest.fixture(scope="module")
def site(payload_enso):
    return bs.montar_arquivos(payload_enso)


@pytest.fixture(scope="module")
def site_sem_enso(payload):
    return bs.montar_arquivos(payload)


def tags(texto: str, nome: str) -> list[str]:
    return re.findall(rf"<{nome}\b[^>]*>", texto)


# ================================================================== formatadores
@pytest.mark.parametrize("x,casas,esperado", [(1234.5, 1, "1.234,5"), (0.0, 0, "0"), (-1.25, 2, "−1,25"),
                                              (1234567, 0, "1.234.567"), (2.5, 1, "2,5"), (-40.0, 0, "−40"),
                                              # meio exato arredonda para longe do zero, igual ao JS (core.js: arred)
                                              (-37.5, 0, "−38"), (-36.5, 0, "−37"), (12.5, 0, "13"), (0.5, 0, "1"),
                                              (2.675, 2, "2,68"), (-0.4, 0, "0"), (-0.0, 1, "0,0"), (99.95, 1, "100,0")])
def test_num_br(x, casas, esperado):
    assert paginas.num_br(x, casas) == esperado


@pytest.mark.parametrize("ruim", [float("nan"), float("inf"), float("-inf")])
def test_num_br_recusa_numero_nao_finito(ruim):
    with pytest.raises(PaginaError, match="não finito"):
        paginas.num_br(ruim)


@pytest.mark.parametrize("entrada", ["2026-07", "2026-07-01", "2026-07-01T00:00:00"])
def test_mes_extenso(entrada):
    assert paginas.mes_extenso(entrada) == "julho de 2026"


def test_dia_br():
    assert paginas.dia_br("2026-11-12") == "12/11/2026" and paginas.dia_br("2026-11-12T10:00:00Z") == "12/11/2026"


@pytest.mark.parametrize("iso", ["2026-11-19T15:00:00Z", "2026-11-19T15:00:00+00:00", "2026-11-19T12:00:00-03:00"])
def test_data_hora_br_converte_para_o_horario_de_fortaleza(iso):
    assert paginas.data_hora_br(iso) == "19/11/2026 12:00"


def test_data_hora_br_sem_fuso_e_erro_para_nao_depender_da_maquina():
    with pytest.raises(PaginaError, match="sem fuso"):
        paginas.data_hora_br("2026-11-19T15:00:00")


@pytest.mark.parametrize("itens,conj,esperado", [([], "ou", ""), (["a"], "ou", "a"), (["a", "b"], "ou", "a ou b"),
                                                 (["a", "b", "c"], "e", "a, b e c"), ([1, 2], "e", "1 e 2")])
def test_lista_pt(itens, conj, esperado):
    assert paginas.lista_pt(itens, conj) == esperado


def test_intervalo_meses_atravessa_o_ano():
    assert paginas.intervalo_meses([12, 1, 2, 3]) == "dezembro a março"
    assert paginas.intervalo_meses([1, 2, 3, 4]) == "janeiro a abril"


def test_esc_escapa_aspas_e_tags():
    assert paginas.esc("""<a href="x">'&'</a>""") == "&lt;a href=&quot;x&quot;&gt;&#x27;&amp;&#x27;&lt;/a&gt;"


# ================================================================== preenchimento
def test_preencher_escapa_texto_e_deixa_html_pronto():
    r = paginas.preencher("<p>{{a}}</p><i>{{b}}</i>", {"a": "<b>x</b>", "b": Html("<b>y</b>")}, "teste")
    assert r == "<p>&lt;b&gt;x&lt;/b&gt;</p><i><b>y</b></i>"


def test_preencher_marcador_sem_valor_e_erro_com_o_nome_e_o_lugar():
    with pytest.raises(PaginaError, match=r"monitor\.html.*\{\{falta_isto\}\}"):
        paginas.preencher("oi {{falta_isto}}", {}, "monitor.html")


def test_preencher_aceita_numero_e_nao_mexe_em_chaves_soltas():
    assert paginas.preencher("{{n}} e {x} e {{ }}", {"n": 3}, "t") == "3 e {x} e {{ }}"


def test_nav_marca_so_a_pagina_atual():
    nav = paginas.html_nav(paginas.PAGINAS, "monitor")
    assert nav.count('aria-current="page"') == 1 and '<a href="monitor.html" aria-current="page">Monitor</a>' in nav
    assert all(f'href="{a}"' in nav for a in ARQUIVOS)


@pytest.mark.parametrize("maturidade,esperado", [
    ("MVP experimental — em calibração com S2iD/TCE-PI",
     "<b>MVP experimental.</b> Em calibração com S2iD/TCE-PI. Não substitui alerta oficial"),
    ("MVP experimental", "<b>MVP experimental.</b> Não substitui alerta oficial"),
    ("mvp experimental - em teste.", "<b>MVP experimental.</b> Em teste. Não substitui alerta oficial"),
    ("MVP experimental — x<y", "<b>MVP experimental.</b> X&lt;y. Não substitui alerta oficial")])
def test_aviso_de_maturidade(maturidade, esperado):
    assert esperado in paginas.html_aviso(maturidade)


# ================================================================== textos derivados dos dados
MESES3 = ["2026-05", "2026-06", "2026-07"]


@pytest.mark.parametrize("t,esperado", [
    (2, "O monitor abre no mês mais recente, julho de 2026. A maioria dos municípios tem leitura confiável."),
    (1, "O monitor abre em junho de 2026, o último mês em que a maioria dos municípios tem leitura confiável. "
        "O mês seguinte, julho de 2026, está no monitor, mas 40% das leituras dele têm baixa confiança."),
    (0, "O monitor abre em maio de 2026, o último mês em que a maioria dos municípios tem leitura confiável. "
        "Os 2 meses seguintes, até julho de 2026, estão no monitor, mas 40% das leituras de julho de 2026 têm baixa confiança.")])
def test_frase_de_abertura_nos_tres_casos(t, esperado):
    assert paginas.frase_abertura({"abertura": t, "meses": MESES3, "pct_baixa_ultimo": 40}) == esperado


@pytest.mark.parametrize("regra,tem,nao_tem", [
    ({"q_max": 0.5, "media_min_mm_por_mes": 10.0}, ["abaixo de 10 mm por mês", "mais de 50% dos anos"], []),
    ({"q_max": 0.7, "media_min_mm_por_mes": 7.5}, ["abaixo de 7,5 mm por mês", "mais de 70% dos anos"], []),
    ({"q_max": 0.5, "media_min_mm_por_mes": 0.0}, ["mais de 50% dos anos"], ["média do período", " ou "])])
def test_regra_de_baixa_confianca_em_texto_segue_o_params(regra, tem, nao_tem):
    t = paginas.regra_baixa_confianca({"regra_baixa": regra})
    assert t.startswith("Baixa confiança:") and t.endswith("mudam muito o SPI.")
    assert all(x in t for x in tem) and not any(x in t for x in nao_tem)


def clima(meses, anos, pct=62.4):
    return {"clima": {"estacao": {"meses": meses, "largura": len(meses), "media_mm": 500.0, "anos": anos}, "pct_estacao": pct}}


ANOS = [{"ano": 2010, "total_mm": 350.0, "desvio_pct": -30.0}, {"ano": 2011, "total_mm": 450.0, "desvio_pct": -10.0},
        {"ano": 2012, "total_mm": 300.0, "desvio_pct": -40.0}, {"ano": 2013, "total_mm": 600.0, "desvio_pct": 20.0}]


def test_textos_do_clima_com_estacao_que_atravessa_o_ano():
    t = paginas.textos_clima(clima([12, 1, 2, 3], ANOS))
    assert t["frase_clima"] == "Os quatro meses mais chuvosos, de dezembro a março, trazem 62% da chuva do ano."
    assert t["frase_estacoes"] == ("As três estações mais secas foram as de 2012 (−40%), 2010 (−30%) e 2011 (−10%). "
                                   "A mais recente completa, a de 2013, ficou 20% acima da média. "
                                   "O ano indicado é o do início da estação.")
    assert t["estacao_meses_curto"] == "dez a mar"


def test_textos_do_clima_sem_atravessar_o_ano_nao_explica_o_ano_de_inicio():
    t = paginas.textos_clima(clima([1, 2, 3, 4], ANOS))
    assert "início da estação" not in t["frase_estacoes"] and t["estacao_meses_curto"] == "jan a abr"


@pytest.mark.parametrize("anos,abre", [(ANOS[:1], "A estação mais seca foi a de 2010 (−30%). "),
                                       (ANOS[:2], "As duas estações mais secas foram as de 2010 (−30%) e 2011 (−10%). ")])
def test_textos_do_clima_com_poucas_estacoes_nao_diz_tres(anos, abre):
    assert paginas.textos_clima(clima([1, 2, 3, 4], anos))["frase_estacoes"].startswith(abre)


def test_textos_do_clima_estacao_recente_na_media_e_abaixo():
    na_media = [{"ano": 2020, "total_mm": 500.0, "desvio_pct": 0.3}]
    assert paginas.textos_clima(clima([1, 2, 3, 4], na_media))["frase_estacoes"].endswith("a de 2020, ficou na média.")
    abaixo = [{"ano": 2020, "total_mm": 400.0, "desvio_pct": -20.0}]
    assert paginas.textos_clima(clima([1, 2, 3, 4], abaixo))["frase_estacoes"].endswith("ficou 20% abaixo da média.")


def test_textos_do_clima_sem_nenhuma_estacao_completa_e_erro():
    with pytest.raises(PaginaError, match="estação chuvosa completa"):
        paginas.textos_clima(clima([1, 2, 3, 4], []))


# ================================================================== blocos do El Niño
@pytest.fixture(scope="module")
def pacote(enso):
    return ce.para_pacote(enso)


def test_enso_ausente_mostra_indisponivel_em_todos_os_blocos():
    for f in (paginas.html_enso_resumo, paginas.html_enso_status):
        assert "Indisponível" in f(None) and "contexto_enso.json" in f(None)
    assert "Indisponível" in paginas.html_enso_leituras(None) and "Sem arquivo" in paginas.html_enso_limites(None)


def test_enso_resumo_traz_fonte_data_aviso_de_contexto_e_numeros_do_arquivo(pacote):
    r = paginas.html_enso_resumo(pacote)
    s = pacote["situacao"]
    cpc = next(f for f in pacote["fontes"] if f["id"] == s["fonte"])
    assert "Contexto, não previsão do nosso modelo." in r and "não uma previsão do nosso modelo" in r
    assert paginas.dia_br(cpc["emitido_em"]) in r and cpc["url"] in r and 'rel="noopener"' in r
    assert f"mais de {s['prob_forte_ou_muito_forte_min_pct']}%" in r and f"cerca de {s['prob_evento_historico_pct']}%" in r
    assert f"+{paginas.num_br(s['nino34_c'], 1)} °C" in r
    assert 'id="ensoAtualizacao" hidden' in r                 # o aviso de "saiu discussão nova" começa escondido


def test_enso_status_repete_os_numeros_e_a_fonte(pacote):
    r = paginas.html_enso_status(pacote)
    assert 'id="ensoAtualizacao" hidden' in r and "Contexto, não previsão do nosso modelo." in r
    assert str(pacote["situacao"]["prob_evento_historico_pct"]) in r


def test_enso_leituras_uma_por_item_com_link_do_documento_quando_existe(pacote):
    r = paginas.html_enso_leituras(pacote)
    assert r.count("<li>") == len(pacote["leituras"]) > 0
    por_id = {f["id"]: f for f in pacote["fontes"]}
    for x in pacote["leituras"]:
        f = por_id[x["fonte"]]
        assert html.escape(f.get("url_documento") or f["url"]) in r and html.escape(x["texto"]) in r


def test_enso_limites_listados(pacote):
    r = paginas.html_enso_limites(pacote)
    assert r.count("<li>") == len(pacote["limites"]) > 0


@pytest.mark.parametrize("campo", ["resumo_pt", "impactos_pt", "forca_pt", "periodo_forte_pt", "status_alerta"])
def test_enso_texto_do_arquivo_nunca_vira_html(pacote, campo):
    p = copy.deepcopy(pacote)
    p["situacao"][campo] = '<script>alert("x")</script>'
    for f in (paginas.html_enso_resumo, paginas.html_enso_status):
        r = f(p)
        assert "<script>" not in r and "&lt;script&gt;" in r


def test_enso_leitura_e_limite_do_arquivo_nunca_viram_html(pacote):
    p = copy.deepcopy(pacote)
    p["leituras"][0]["texto"] = "<img src=x onerror=alert(1)>"
    p["leituras"][0]["escopo_pt"] = "<b>"
    p["limites"][0] = "<i>"
    assert "<img" not in paginas.html_enso_leituras(p) and "<i>" not in paginas.html_enso_limites(p)


# ================================================================== a lista de páginas
def test_lista_de_paginas_e_coerente():
    assert len(set(ARQUIVOS)) == len(ARQUIVOS) == 6 and ARQUIVOS[0] == "index.html"
    assert len({p.id for p in paginas.PAGINAS}) == len(paginas.PAGINAS)
    for p in paginas.PAGINAS:
        assert (paginas.DIR_MODELOS / p.modelo).is_file() and p.descricao and p.rotulo and p.titulo
        assert all((ASSETS / f"{s}.js").is_file() for s in p.scripts)
        assert not re.search(r"\b(19|20)\d\d\b", p.descricao.replace("2026-2027", "")), p.descricao   # ano fixo vira mentira
    assert (paginas.DIR_MODELOS / "_base.html").is_file()


# ================================================================== a pasta do site: estrutura
@pytest.mark.parametrize("arquivo", ARQUIVOS)
def test_cada_pagina_tem_a_estrutura_basica(site, arquivo):
    h = site[arquivo]
    pagina = next(p for p in paginas.PAGINAS if p.arquivo == arquivo)
    assert h.startswith("<!doctype html>") and '<html lang="pt-BR"' in h and 'name="viewport"' in h
    assert f"<title>{html.escape(pagina.titulo)} · Piauí Resiliente</title>" in h
    assert len(tags(h, "h1")) == 1 and h.count("<main") == 1 and 'href="#conteudo"' in h and 'id="conteudo"' in h
    assert h.count('id="pr-gota"') == 1 and '<use href="#pr-gota"/>' in h and 'rel="icon"' in h
    assert "<b>Piauí</b><span>Resiliente</span>" in h                        # o lockup da marca no topo
    nav = re.search(r"<nav[^>]*>(.*?)</nav>", h, re.S).group(1)
    assert all(f'href="{a}"' in nav for a in ARQUIVOS) and nav.count("aria-current") == 1
    assert f'<a href="{arquivo}" aria-current="page">' in nav


@pytest.mark.parametrize("arquivo", ARQUIVOS)
def test_toda_pagina_leva_o_aviso_de_mvp_a_data_e_o_hash_da_verificacao(site, payload_enso, arquivo):
    h = site[arquivo]
    for texto in ("MVP experimental", "Em calibração com S2iD/TCE-PI", "Não substitui alerta oficial",
                  "arquivos conferidos por SHA-256", payload_enso["verificacao"]["dataset_sha256"], "19/11/2026 12:00",
                  "CHIRPS", "Equipe"):
        assert texto in h, (arquivo, texto)


@pytest.mark.parametrize("arquivo", ARQUIVOS)
def test_pagina_sem_marcador_sobrando_nem_valor_vazio(site, arquivo):
    h = site[arquivo]
    assert "{{" not in h
    assert not re.search(r"\b(None|nan|undefined|null)\b", re.sub(r"<script.*?</script>", "", h, flags=re.S)), arquivo
    assert not re.search(r"<(p|li|h[1-3])>\s*</\1>", h), "parágrafo, item ou título vazio"


@pytest.mark.parametrize("arquivo", ARQUIVOS)
def test_pagina_sem_pontuacao_dobrada(site, arquivo):
    """Pega o ponto duplicado de 'equipe V.I.G.I.A..' e similares; reticências de três pontos não contam."""
    texto = re.sub(r"<(script|style)\b.*?</\1>", "", site[arquivo], flags=re.S)
    texto = html.unescape(re.sub(r"<[^>]+>", " ", texto))
    assert not re.findall(r".{12}(?<!\.)\.\.(?!\.).{6}", texto), re.findall(r".{12}(?<!\.)\.\.(?!\.).{6}", texto)
    assert not re.findall(r".{10}[,;:]{2}.{6}", texto)


def test_nome_da_equipe_termina_com_um_ponto_so_em_fim_de_frase(payload):
    for equipe in ("V.I.G.I.A.", "V.I.G.I.A", "Equipe X. "):
        v = paginas.valores_base({**payload, "equipe": equipe}, {})
        assert v["equipe_final"].endswith(".") and not v["equipe_final"].endswith("..")


def test_todos_os_links_internos_e_assets_existem(site):
    existentes = set(site)
    for arquivo in ARQUIVOS:
        h = site[arquivo]
        for ref in re.findall(r'(?:href|src)="([^"#:]+)"', h):
            assert ref in existentes, (arquivo, ref)
        for ref in re.findall(r'href="([^"#:]+)#', h):
            assert ref in existentes, (arquivo, ref)
        for ancora in re.findall(r'href="#([^"]+)"', h):
            assert f'id="{ancora}"' in h, (arquivo, ancora)


def test_links_externos_so_para_as_fontes_do_arquivo_do_el_nino_e_com_noopener(site, payload_enso):
    permitidas = set()
    for f in payload_enso["enso"]["fontes"]:
        permitidas |= {f["url"], f.get("url_documento")} - {None}
    achados = set()
    for arquivo in ARQUIVOS:
        for tag in tags(site[arquivo], "a"):
            m = re.search(r'href="(https?://[^"]*)"', tag)
            if m:
                assert 'rel="noopener' in tag, (arquivo, tag)
                achados.add(html.unescape(m.group(1)))
    assert achados and achados <= permitidas, achados - permitidas


def test_pagina_do_el_nino_com_arquivo_e_sem_arquivo(site, site_sem_enso, payload_enso):
    h = site["el-nino.html"]
    assert "Contexto, não previsão do nosso modelo." in h and "Indisponível: o arquivo de fontes" not in h
    assert 'id="ensoAtualizacao"' in h and "Os anos de El Niño não estão marcados nos gráficos" in h
    sem = site_sem_enso["el-nino.html"]
    assert "Indisponível: o arquivo de fontes oficiais" in sem and "ensodisc" not in sem
    assert "Contexto, não previsão do nosso modelo." not in sem          # sem fonte não se afirma nada sobre o Pacífico
    assert "El Niño" in site["index.html"] and "Indisponível" in site_sem_enso["index.html"]


def test_pagina_inicial_diz_o_objetivo_a_contagem_e_a_regra_do_mes_de_abertura(site, payload_enso):
    h = site["index.html"]
    assert "Onde e quando a chuva faltou no semiárido do Piauí" in h
    assert "3 municípios, mês a mês desde 1981" in h
    assert html.escape(paginas.frase_abertura(payload_enso)) in h
    assert "Terracota é falta de chuva" in h and 'id="mapa"' in h and 'id="mapaMes"' in h


def test_metodo_declara_o_que_nao_existe_e_os_limites(site, payload_enso):
    h = site["metodo.html"]
    for texto in ("Previsão do próximo mês", "Antecedência em relação aos decretos", "Plano de Contingência",
                  "Indisponível", "O SPI mede falta de chuva", "não prova que um açude vai secar", "in-sample",
                  "fora de treino, validação e teste", "Continuam nas contagens do mapa", "pontilhado", "Baixa confiança:"):
        assert texto in h, texto
    assert h.count('class="sw"') == len(payload_enso["classes"]) == 7
    for c in payload_enso["classes"]:
        assert html.escape(c["rotulo"]) in h and html.escape(c["faixa"]) in h
    assert "Até 2015" in h and "depois de dezembro de 2025" in h             # o ano vem dos dados, não do texto


def test_nenhuma_pagina_ou_script_promete_previsao(site):
    proibido = ("probabilidade de seca", "vai haver seca", "haverá seca", "previsão para", "alerta de seca", "risco de seca")
    for caminho, texto in site.items():
        if caminho.endswith((".html", ".js")) and caminho != "assets/dados.js":
            baixo = texto.lower()
            assert not [x for x in proibido if x in baixo], (caminho, [x for x in proibido if x in baixo])


def test_monitor_conta_so_leituras_confiaveis_e_diz_isso(site):
    js = site["assets/monitor.js"]
    for texto in ("Meses em seca (SPI ≤ −1,0, só meses confiáveis):", "Mês mais seco (só meses confiáveis):",
                  "Meses de baixa confiança:"):
        assert texto in js, texto
    # o mês de baixa confiança sai do "mês mais seco" e das contagens de seca: o `continue` vem antes de elas serem tocadas
    i = js.index("if (bxAt(k, i, t)) { baixas++; continue; }")
    assert js.index("secos++", i) > i and js.index("pior = x", i) > i
    h = site["monitor.html"]                                  # a explicação mora em dois lugares: no HTML e no texto dinâmico do JS
    assert "pontilhado" in h and "Pouca chuva esperada nesta época do ano" in h and 'id="notaBc"' in h
    assert "esmaecido = baixa confiança" in js and "Baixa confiança: neste período do ano" in js
    assert "BX[k][m * NT + t] ? 105 : 255" in js                # leitura de baixa confiança sai mais fraca no mapa


# ================================================================== a pasta do site: offline
def test_site_e_100_por_cento_offline(site):
    for caminho, texto in site.items():
        if caminho == "assets/marca/gota.svg":
            continue
        assert not re.search(r"<script[^>]+\bsrc=\"https?:", texto), caminho
        assert not re.search(r"<link\b[^>]+href=\"https?:", texto), caminho
        assert not re.search(r"@import|url\(\s*['\"]?https?:", texto), caminho
        assert not re.search(r"\b(fetch|XMLHttpRequest|WebSocket|EventSource|importScripts)\b|navigator\.sendBeacon|\bimport\(", texto), caminho
        assert not re.search(r"<(img|iframe|embed|object|video|audio|source)\b[^>]*\bsrc=\"https?:", texto), caminho
    js = "".join(t for c, t in site.items() if c.endswith(".js") and c != "assets/dados.js")
    assert set(re.findall(r"https?://[^\s'\"`)]+", js)) <= {"http://www.w3.org/2000/svg"}


def test_regiao_rolavel_da_tabela_e_acessivel_por_teclado(site):
    """A tabela rola dentro de um contêiner de altura limitada: sem tabindex e nome, o teclado não chega nela."""
    tag = next(t for t in tags(site["municipios.html"], "div") if 'class="tabela-mun"' in t)
    assert 'tabindex="0"' in tag and 'role="region"' in tag and re.search(r'aria-label="[^"]{10,}"', tag)


def test_scripts_sao_classicos_e_funcionam_em_file(site):
    for arquivo in ARQUIVOS:
        h = site[arquivo]
        assert 'type="module"' not in h
        ordem = re.findall(r'<script src="assets/([a-z]+)\.js">', h)
        assert ordem[:2] == ["dados", "core"] and len(ordem) == len(set(ordem))      # dados antes do núcleo, sem repetir


def test_armazenamento_do_navegador_sempre_dentro_de_try(site):
    for caminho, texto in site.items():
        if caminho.endswith((".js", ".html")) and caminho != "assets/dados.js":
            for m in re.finditer(r"localStorage|sessionStorage", texto):
                antes = texto[max(0, m.start() - 160):m.start()]
                assert "try" in antes, (caminho, antes[-80:])


@pytest.mark.skipif(shutil.which("node") is None, reason="node não instalado")
def test_scripts_tem_sintaxe_valida_e_os_dados_carregam(site, tmp_path):
    for caminho, texto in site.items():
        if caminho.endswith(".js"):
            f = tmp_path / Path(caminho).name
            f.write_text(texto, encoding="utf-8")
            r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
            assert r.returncode == 0, (caminho, r.stderr)
    script = ("const vm=require('vm'),fs=require('fs');const ctx={window:{}};vm.createContext(ctx);"
              "vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),ctx);"
              "const d=ctx.window.PR_DADOS;console.log(JSON.stringify([d.municipios.length,d.meses.length,d.classes.length]))")
    r = subprocess.run(["node", "-e", script, str(tmp_path / "dados.js")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout) == [3, 549, 7]


@pytest.mark.skipif(shutil.which("node") is None, reason="node não instalado")
def test_arredondamento_do_js_e_o_do_python_batem(site):
    """O texto da página (Python) e o rótulo do gráfico (JS) mostram o mesmo número, inclusive nos meios exatos."""
    linha = re.search(r"const arred = [^\n]+;", site["assets/core.js"]).group(0)
    valores = [-38.5, -37.5, -36.5, -0.5, -0.4, 0.0, 0.4, 0.5, 1.49, 12.5, 13.5, 62.4, 99.5, -75.0]
    script = f"{linha}\nconsole.log(JSON.stringify({json.dumps(valores)}.map((v) => String(arred(v)))))"
    r = subprocess.run(["node", "-e", script], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    em_js = [x.replace("-", "−") for x in json.loads(r.stdout)]
    assert em_js == [paginas.num_br(v) for v in valores]


def test_ids_usados_pelo_script_existem_na_pagina(site):
    for p in paginas.PAGINAS:
        h = site[p.arquivo]
        js = site["assets/core.js"] + "".join(site[f"assets/{s}.js"] for s in p.scripts)
        usados = set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", js)) | set(re.findall(r"getElementById\('([A-Za-z0-9_]+)'\)", js))
        ids = set(re.findall(r"\bid=\"([A-Za-z0-9_-]+)\"", h))
        assert usados - ids == set(), (p.arquivo, sorted(usados - ids))


# ================================================================== a pasta do site: marca, cores e acessibilidade
def variaveis(css: str, padrao: str) -> dict[str, str]:
    m = re.search(padrao, css, re.S | re.M)
    assert m, padrao
    return dict(re.findall(r"--([a-z0-9-]+):([^;]+);", m.group(1)))


@pytest.fixture(scope="module")
def css(site):
    return site["assets/site.css"]


@pytest.fixture(scope="module")
def temas(css):
    claro = variaveis(css, r"^:root\{(.*?)\n\}")
    escuro = variaveis(css, r'^:root\[data-theme="dark"\]\{(.*?)\n\}')
    auto = variaveis(css, r':root:where\(:not\(\[data-theme="light"\]\)\)\{(.*?)\n  \}')
    return claro, escuro, auto


def test_css_traz_as_duas_fontes_embutidas_e_nenhum_marcador(css):
    assert css.count("data:font/woff2;base64,") == 2 and "__FONTE_" not in css
    assert 'font-family:"Atkinson Hyperlegible Next"' in css and 'font-family:"Outfit"' in css


def test_css_usa_a_paleta_da_marca(css, temas):
    claro, escuro, auto = temas
    assert claro["petroleo"] == "#0F6E7A" and claro["terracota"] == "#B4472A"
    assert claro["plane"] == "#F6F1E8" and claro["ink"] == "#12262B" and claro["gota-falta"] == "#B4472A"
    assert escuro["plane"] == "#0D1B1F" and escuro["surface"] == "#12262B"
    assert escuro == auto                                                   # tema escuro automático = tema escuro escolhido


def test_cores_das_classes_do_spi_no_css_sao_as_do_pacote(temas, payload_enso):
    claro, escuro, _ = temas
    for i, c in enumerate(bs.CLASSES):
        assert claro[f"c{i}"].lower() == c[3].lower() and escuro[f"c{i}"].lower() == c[4].lower(), i
    assert [c["rotulo"] for c in payload_enso["classes"]] == [c[1] for c in bs.CLASSES]


@pytest.mark.parametrize("tema", [0, 1])
def test_texto_e_elementos_graficos_tem_contraste_nos_dois_temas(temas, tema):
    v = temas[tema]
    for fundo in ("plane", "surface"):
        for texto in ("ink", "ink2", "muted", "link"):
            assert contraste(v[texto], v[fundo]) >= 4.5, (tema, texto, fundo)
        for grafico in ("gota-falta", "gota-ha", "dry", "wet", "foco"):
            assert contraste(v[grafico], v[fundo]) >= 3.0, (tema, grafico, fundo)
    assert contraste(v["acao-ink"], v["acao"]) >= 4.5 and contraste(v["tip-ink"], v["tip"]) >= 4.5


def test_css_respeita_foco_movimento_reduzido_e_nao_grita_em_caixa_alta(css):
    assert ":focus-visible" in css and "prefers-reduced-motion" in css
    assert "text-transform:uppercase" not in css.replace(" ", "")
    assert "@media print" in css


def test_lockup_da_marca_usa_outfit_com_pesos_700_e_300(css):
    assert re.search(r"\.nome\s*b\s*\{[^}]*font-weight:\s*700", css) and re.search(r"\.nome\s*span\s*\{[^}]*font-weight:\s*300", css)


def test_gota_isolada_e_a_mesma_do_modulo_da_marca(site):
    from src.web import marca
    assert site["assets/marca/gota.svg"] == marca.svg_isolado()
