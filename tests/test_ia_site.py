"""Página Planejamento (rascunhos de IA revisados) e a ligação com o gerador do site. Dados sintéticos só em tests/."""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dados_sinteticos import A, B, VERIFICADO, raw_sintetico
from src.ia import arquivo as aq
from src.ia import checklist as ck
from src.ia import fatos as ft
from src.ia import rascunhos as rs
from src.web import build_site as bs
from src.web import paginas
from test_build_site import extrair, lido, montar_proj

REVISAO = datetime(2026, 11, 20, 9, 0, 0, tzinfo=timezone.utc)
AGORA = datetime(2026, 11, 19, 12, 30, 45, tzinfo=timezone.utc)
ITENS = [{"id": "mobilizar_defesa_civil", "titulo": "Mobilizar a Defesa Civil",
          "descricao": "Reunir a equipe e revisar contatos.", "fonte": "Fonte de teste 1"},
         {"id": "revisar_abastecimento", "titulo": "Revisar o abastecimento de água",
          "descricao": "Conferir carros-pipa e pontos de distribuição.", "fonte": "Fonte de teste 2"}]
CHECKLIST = {"versao": "v7", "elaborado_por": "Equipe de teste", "atualizado_em": "2026-11-19", "itens": ITENS}
RESUMO = "O município pede atenção redobrada neste mês."


def rascunho(payload, cod=A, resumo=RESUMO):
    f = ft.fatos_municipio(payload, cod)
    return rs.Rascunho(cod_ibge=cod, nome=f.nome, mes_referencia=f.mes, escala=f.escala, ficha=ft.ficha_texto(f),
                       resumo=resumo, itens=(rs.ItemEscolhido(id="mobilizar_defesa_civil", porque="Prepara a equipe."),
                                             rs.ItemEscolhido(id="revisar_abastecimento", porque="Confere a água.")),
                       modelo="modelo-falso", gerado_em_utc="2026-11-19T12:30:45Z", tentativas=1)


@pytest.fixture
def lista(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps(CHECKLIST), encoding="utf-8")
    return ck.carregar(p)


@pytest.fixture
def pacote(payload, lista):
    arq = aq.ArquivoRascunhos(dataset_sha256="ab" * 32, checklist_sha256="cd" * 32)
    arq = aq.aprovar(aq.registrar(arq, rascunho(payload)), A, "Ana Souza", agora=REVISAO, checklist=lista)
    pac, _ = aq.para_pacote(arq, lista, dataset_sha256="ab" * 32, checklist_sha256="cd" * 32)
    return pac


@pytest.fixture
def com_plano(payload, pacote):
    return {**payload, "planejamento": pacote}


# ================================================================== páginas e HTML
def test_sem_rascunho_a_lista_de_paginas_nao_muda(payload):
    assert paginas.paginas_do_site(payload) == paginas.PAGINAS
    assert paginas.paginas_do_site({**payload, "planejamento": None}) == paginas.PAGINAS


def test_com_rascunho_planejamento_entra_antes_de_metodo(com_plano):
    ids = [p.id for p in paginas.paginas_do_site(com_plano)]
    assert ids == ["inicio", "monitor", "municipios", "el-nino", "planejamento", "metodo", "sobre"]


def test_pagina_de_planejamento_e_coerente():
    p = paginas.PAGINA_PLANEJAMENTO
    assert (paginas.DIR_MODELOS / p.modelo).is_file() and p.arquivo == "planejamento.html"
    assert not re.search(r"\b(19|20)\d\d\b", p.descricao)
    assert "sem valor jurídico" in p.descricao


def test_sem_rascunho_nao_gera_a_pagina_nem_o_link(payload):
    arquivos = bs.montar_arquivos(payload)
    assert "planejamento.html" not in arquivos
    assert all("planejamento.html" not in h for n, h in arquivos.items() if n.endswith(".html"))


def test_com_rascunho_gera_a_pagina_e_o_link_em_todas(com_plano):
    arquivos = bs.montar_arquivos(com_plano)
    assert "planejamento.html" in arquivos
    for nome, h in arquivos.items():
        if nome.endswith(".html"):
            nav = re.search(r"<nav[^>]*>(.*?)</nav>", h, re.S).group(1)
            assert 'href="planejamento.html"' in nav and nav.count("aria-current") == 1, nome


def test_conteudo_da_pagina(com_plano, pacote):
    h = bs.montar_arquivos(com_plano)["planejamento.html"]
    for texto in (rs.ROTULO, RESUMO, "Alfa do Norte", "Mobilizar a Defesa Civil", "Reunir a equipe e revisar contatos.",
                  "Prepara a equipe.", "Fonte do item: Fonte de teste 1", "Fonte do item: Fonte de teste 2",
                  "Revisado por Ana Souza em 20/11/2026 06:00",          # 09:00 UTC = 06:00 em Fortaleza
                  "Texto escrito por IA (modelo-falso)", "Base do checklist: versão v7, de Equipe de teste, atualizada em 19/11/2026.",
                  "MVP experimental", "Em calibração com S2iD/TCE-PI", com_plano["verificacao"]["dataset_sha256"]):
        assert texto in h, texto
    assert h.count("<h1") == 1 and 'id="m2200001"' in h and "{{" not in h
    assert f"SPI-{com_plano['escala_alvo']}" in h
    assert not re.search(r"\b(None|nan|undefined|null)\b", h.split("<script")[0])
    assert not re.search(r"<(p|li|h[1-3])>\s*</\1>", h)


def test_pagina_nao_chama_nada_de_fora(com_plano):
    h = bs.montar_arquivos(com_plano)["planejamento.html"]
    assert not re.search(r"\b(?:src|href)=[\"']https?:", h)
    assert "fetch(" not in h and "XMLHttpRequest" not in h and "type=\"module\"" not in h


def test_so_o_que_o_pacote_traz_aparece_e_texto_malicioso_e_escapado(com_plano):
    ruim = json.loads(json.dumps(com_plano["planejamento"]))
    ruim["rascunhos"][0]["resumo"] = '<script>alert(1)</script> "aspas" & mais'
    ruim["rascunhos"][0]["itens"][0]["fonte"] = '<img src=x onerror=1>'
    h = bs.montar_arquivos({**com_plano, "planejamento": ruim})["planejamento.html"]
    assert "<script>alert(1)</script>" not in h and "<img src=x" not in h
    assert "&lt;script&gt;alert(1)&lt;/script&gt; &quot;aspas&quot; &amp; mais" in h


def test_dados_js_nao_leva_os_rascunhos(com_plano):
    dados = bs.montar_arquivos(com_plano)["assets/dados.js"]
    assert "planejamento" not in dados and RESUMO not in dados and "Ana Souza" not in dados


def test_html_planejamento_vazio():
    assert paginas.html_planejamento(None) == "" and paginas.texto_checklist(None) == ""


# ================================================================== ponta a ponta
def sha_do_manifest(raiz: Path) -> str:
    return json.loads((raiz / "data" / "curated" / "manifest.json").read_text(encoding="utf-8"))["dataset_sha256"]


def preparar(cfg, raiz, *, aprovar=True, dataset_sha=None, checklist=True, cods=(A,)):
    """Grava checklist_base.json e rascunhos_ia.json em `raiz`, do jeito que a CLI faria."""
    pay = bs.montar_payload(bs.carregar_insumos(cfg), cfg, verificado_em=VERIFICADO)
    caminho_ck = raiz / ck.ARQUIVO_PADRAO
    caminho_ck.parent.mkdir(parents=True, exist_ok=True)
    caminho_ck.write_text(json.dumps(CHECKLIST), encoding="utf-8")
    lista = ck.carregar(caminho_ck)
    arq = aq.ArquivoRascunhos(dataset_sha256=dataset_sha or sha_do_manifest(raiz), checklist_sha256=ck.sha256(caminho_ck))
    for cod in cods:
        arq = aq.registrar(arq, rascunho(pay, cod))
        if aprovar:
            arq = aq.aprovar(arq, cod, "Ana Souza", agora=REVISAO, checklist=lista)
    aq.gravar(arq, raiz / aq.ARQUIVO_PADRAO)
    if not checklist:
        caminho_ck.unlink()


@pytest.fixture
def proj(tmp_path):
    return montar_proj(tmp_path, raw_sintetico())


def test_ponta_a_ponta_sem_arquivos_nada_muda(proj):
    cfg, raiz = proj
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["planejamento"] == 0 and r["avisos"] == []
    assert not (raiz / "site" / "planejamento.html").exists()


def test_ponta_a_ponta_rascunho_revisado_vira_pagina(proj):
    cfg, raiz = proj
    preparar(cfg, raiz, cods=(A, B))
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["planejamento"] == 2 and r["avisos"] == []
    h = (raiz / "site" / "planejamento.html").read_text(encoding="utf-8")
    assert "Alfa do Norte" in h and "Beta do Sul" in h and rs.ROTULO in h
    assert 'href="planejamento.html"' in (raiz / "site" / "index.html").read_text(encoding="utf-8")
    assert "planejamento" not in extrair(raiz / "site")


def test_ponta_a_ponta_pendente_nao_aparece_e_a_cli_avisa(proj):
    cfg, raiz = proj
    preparar(cfg, raiz, aprovar=False)
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["planejamento"] == 0 and r["avisos"] == ["1 rascunho(s) de IA aguardam revisão e não aparecem no site"]
    assert not (raiz / "site" / "planejamento.html").exists()


def test_ponta_a_ponta_rascunho_de_outro_dataset_fica_de_fora(proj):
    cfg, raiz = proj
    preparar(cfg, raiz, dataset_sha="ef" * 32)
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["planejamento"] == 0 and "outro dataset" in r["avisos"][0]
    assert not (raiz / "site" / "planejamento.html").exists()


def test_ponta_a_ponta_checklist_alterado_depois_da_revisao_deixa_de_fora(proj):
    cfg, raiz = proj
    preparar(cfg, raiz)
    p = raiz / ck.ARQUIVO_PADRAO
    p.write_text(json.dumps({**CHECKLIST, "versao": "v8"}), encoding="utf-8")
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["planejamento"] == 0 and "base do checklist mudou" in r["avisos"][0]


def test_ponta_a_ponta_sem_checklist_deixa_de_fora_e_avisa(proj):
    cfg, raiz = proj
    preparar(cfg, raiz, checklist=False)
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["planejamento"] == 0 and "não foi encontrada ou é inválida" in r["avisos"][0]


def test_ponta_a_ponta_arquivo_de_rascunhos_invalido_nao_derruba_o_site(proj):
    cfg, raiz = proj
    p = raiz / aq.ARQUIVO_PADRAO
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{quebrado", encoding="utf-8")
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert (raiz / "site" / "index.html").is_file() and r["planejamento"] == 0
    assert r["avisos"][0].startswith("rascunhos de IA ignorados:")


def test_ponta_a_ponta_gerar_duas_vezes_da_o_mesmo_conteudo(proj):
    cfg, raiz = proj
    preparar(cfg, raiz)
    bs.gerar_site(cfg, agora=VERIFICADO)
    a = lido(raiz / "site")
    bs.gerar_site(cfg, agora=VERIFICADO)
    assert lido(raiz / "site") == a and "planejamento.html" in a


def test_cli_do_site_imprime_os_avisos(proj, capsys):
    cfg, raiz = proj
    preparar(cfg, raiz, aprovar=False)
    assert bs.main(["--params", str(raiz / "params.yaml")]) == 0
    saida = capsys.readouterr().out
    assert "AVISO: 1 rascunho(s) de IA aguardam revisão" in saida and "página Planejamento" not in saida
    preparar(cfg, raiz)
    assert bs.main(["--params", str(raiz / "params.yaml")]) == 0
    assert "página Planejamento: 1 rascunho(s) revisado(s)" in capsys.readouterr().out
