"""Arquivo de rascunhos e revisão humana (dados sintéticos só aqui, em tests/)."""
import json
import os
import re
import stat
from datetime import datetime, timezone

import pytest

from dados_sinteticos import A, B
from src.ia import arquivo as aq
from src.ia import checklist as ck
from src.ia import fatos as ft
from src.ia import rascunhos as rs

AGORA = datetime(2026, 11, 19, 12, 30, 45, tzinfo=timezone.utc)
REVISAO = datetime(2026, 11, 20, 9, 0, 0, tzinfo=timezone.utc)
DS, CK = "ab" * 32, "cd" * 32


@pytest.fixture
def lista(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "versao": "v7", "elaborado_por": "Equipe de teste", "atualizado_em": "2026-11-19",
        "itens": [{"id": "mobilizar_defesa_civil", "titulo": "Mobilizar a Defesa Civil",
                   "descricao": "Reunir a equipe e revisar contatos.", "fonte": "Fonte de teste 1"},
                  {"id": "revisar_abastecimento", "titulo": "Revisar o abastecimento de água",
                   "descricao": "Conferir carros-pipa e pontos de distribuição.", "fonte": "Fonte de teste 2"}]}), encoding="utf-8")
    return ck.carregar(p)


def rascunho(payload, cod=A, resumo="O município pede atenção.", itens=(("mobilizar_defesa_civil", "Prepara a equipe."),
                                                                         ("revisar_abastecimento", "Confere a água."))):
    f = ft.fatos_municipio(payload, cod)
    return rs.Rascunho(cod_ibge=cod, nome=f.nome, mes_referencia=f.mes, escala=f.escala, ficha=ft.ficha_texto(f),
                       resumo=resumo, itens=tuple(rs.ItemEscolhido(id=i, porque=p) for i, p in itens),
                       modelo="modelo-falso", gerado_em_utc="2026-11-19T12:30:45Z", tentativas=1)


@pytest.fixture
def vazio():
    return aq.ArquivoRascunhos(dataset_sha256=DS, checklist_sha256=CK)


# ------------------------------------------------------------------ leitura e gravação
def test_arquivo_inexistente_e_none(tmp_path):
    assert aq.carregar(tmp_path / "nao.json") is None


def test_gravar_e_carregar_ida_e_volta(tmp_path, payload, vazio):
    arq = aq.registrar(aq.registrar(vazio, rascunho(payload, B)), rascunho(payload, A))
    p = aq.gravar(arq, tmp_path / "sub" / "r.json")
    assert aq.carregar(p) == arq
    bruto = p.read_bytes()
    assert b"\r" not in bruto and bruto.endswith(b"\n")
    assert list(json.loads(bruto)["rascunhos"]) == sorted([A, B])           # ordem por código
    if os.name != "nt":                                                    # o Windows não tem bits de permissão POSIX (sempre 0o666)
        assert stat.S_IMODE(p.stat().st_mode) == 0o644
    assert not list(p.parent.glob("*.tmp"))


def test_gravar_nao_deixa_temporario_quando_falha(tmp_path, vazio, monkeypatch):
    def quebra(*a, **k):
        raise OSError("disco cheio")
    monkeypatch.setattr(aq.os, "replace", quebra)
    with pytest.raises(OSError):
        aq.gravar(vazio, tmp_path / "r.json")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("conteudo", ["{nao é json", "[]", '{"versao_arquivo": 2}',
                                      json.dumps({"dataset_sha256": "x", "checklist_sha256": "y"})])
def test_arquivo_invalido_e_erro_e_nao_e_ignorado(tmp_path, conteudo):
    p = tmp_path / "r.json"
    p.write_text(conteudo, encoding="utf-8")
    with pytest.raises(aq.ArquivoError, match="inválido"):
        aq.carregar(p)


def test_chave_precisa_ser_o_codigo_do_rascunho(tmp_path, payload, vazio):
    d = json.loads(aq.gravar(aq.registrar(vazio, rascunho(payload)), tmp_path / "r.json").read_text(encoding="utf-8"))
    d["rascunhos"] = {B: d["rascunhos"][A]}
    (tmp_path / "r.json").write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(aq.ArquivoError, match="chave diferente"):
        aq.carregar(tmp_path / "r.json")


# ------------------------------------------------------------------ abrir_ou_criar
def test_abrir_ou_criar_cria_vazio_ou_reabre(tmp_path, payload):
    p = tmp_path / "r.json"
    novo = aq.abrir_ou_criar(p, DS, CK)
    assert novo.rascunhos == {} and (novo.dataset_sha256, novo.checklist_sha256) == (DS, CK)
    aq.gravar(aq.registrar(novo, rascunho(payload)), p)
    assert list(aq.abrir_ou_criar(p, DS, CK).rascunhos) == [A]


@pytest.mark.parametrize("ds, ckk, quem", [("ef" * 32, CK, "dataset"), (DS, "ef" * 32, "checklist")])
def test_outro_dataset_ou_checklist_exige_recomecar(tmp_path, payload, ds, ckk, quem):
    p = tmp_path / "r.json"
    aq.gravar(aq.registrar(aq.abrir_ou_criar(p, DS, CK), rascunho(payload)), p)
    with pytest.raises(aq.ArquivoError, match=f"outro {quem}.*--recomecar"):
        aq.abrir_ou_criar(p, ds, ckk)
    assert aq.abrir_ou_criar(p, ds, ckk, recomecar=True).rascunhos == {}


# ------------------------------------------------------------------ registrar
def test_registrar_nao_muda_o_original_e_substitui_pendente(payload, vazio):
    um = aq.registrar(vazio, rascunho(payload))
    assert vazio.rascunhos == {} and list(um.rascunhos) == [A]
    dois = aq.registrar(um, rascunho(payload, resumo="Outro texto."))
    assert dois.rascunhos[A].resumo == "Outro texto." and um.rascunhos[A].resumo != "Outro texto."


def test_nao_sobrescreve_revisado_sem_pedir(payload, vazio, lista):
    rev = aq.aprovar(aq.registrar(vazio, rascunho(payload)), A, "Ana", agora=REVISAO, checklist=lista)
    with pytest.raises(aq.ArquivoError, match="--sobrescrever"):
        aq.registrar(rev, rascunho(payload, resumo="Outro."))
    novo = aq.registrar(rev, rascunho(payload, resumo="Outro."), sobrescrever_revisado=True)
    assert novo.rascunhos[A].status == "pendente_revisao"


# ------------------------------------------------------------------ aprovar e rejeitar
def test_aprovar_grava_quem_e_quando(payload, vazio, lista):
    arq = aq.registrar(vazio, rascunho(payload))
    rev = aq.aprovar(arq, A, "  Ana Souza ", agora=REVISAO, checklist=lista)
    r = rev.rascunhos[A]
    assert (r.status, r.revisado_por, r.revisado_em_utc) == ("revisado", "Ana Souza", "2026-11-20T09:00:00Z")
    assert arq.rascunhos[A].status == "pendente_revisao"               # o original não muda


def test_aprovar_valida_de_novo_o_texto_editado_a_mao(payload, vazio, lista):
    ruim = aq.registrar(vazio, rascunho(payload, resumo="Choveu 999 mm."))
    with pytest.raises(aq.ArquivoError, match="não passa na validação.*número fora da ficha"):
        aq.aprovar(ruim, A, "Ana", agora=REVISAO, checklist=lista)
    inventado = aq.registrar(vazio, rascunho(payload, itens=(("inventado", "x"), ("mobilizar_defesa_civil", "y"))))
    with pytest.raises(aq.ArquivoError, match="não existe na base"):
        aq.aprovar(inventado, A, "Ana", agora=REVISAO, checklist=lista)


@pytest.mark.parametrize("char, codigo", [("\u200b", "U+200B"), ("\u00a0", "U+00A0")])
def test_aprovar_recusa_caractere_invisivel_colado_a_mao(payload, vazio, lista, char, codigo):
    ruim = aq.registrar(vazio, rascunho(payload, resumo=f"Texto com{char}problema."))
    with pytest.raises(aq.ArquivoError, match="não passa na validação.*" + re.escape(codigo)):
        aq.aprovar(ruim, A, "Ana", agora=REVISAO, checklist=lista)


@pytest.mark.parametrize("por", ["", " ", "A", "x" * 81, "Ana\nSouza", "<b>Ana</b>"])
def test_revisor_invalido(payload, vazio, lista, por):
    arq = aq.registrar(vazio, rascunho(payload))
    with pytest.raises(aq.ArquivoError, match="quem revisou"):
        aq.aprovar(arq, A, por, agora=REVISAO, checklist=lista)
    with pytest.raises(aq.ArquivoError, match="quem revisou"):
        aq.rejeitar(arq, A, por, agora=REVISAO)


def test_so_se_aprova_pendente_e_o_codigo_precisa_existir(payload, vazio, lista):
    rev = aq.aprovar(aq.registrar(vazio, rascunho(payload)), A, "Ana", agora=REVISAO, checklist=lista)
    with pytest.raises(aq.ArquivoError, match="só se aprova rascunho pendente"):
        aq.aprovar(rev, A, "Ana", agora=REVISAO, checklist=lista)
    with pytest.raises(aq.ArquivoError, match="não há rascunho"):
        aq.aprovar(vazio, A, "Ana", agora=REVISAO, checklist=lista)
    with pytest.raises(aq.ArquivoError, match="não há rascunho"):
        aq.rejeitar(vazio, A, "Ana", agora=REVISAO)


def test_rejeitar_e_uma_vez_so_e_pode_vir_depois_de_aprovado(payload, vazio, lista):
    rej = aq.rejeitar(aq.registrar(vazio, rascunho(payload)), A, "Ana", agora=REVISAO)
    assert rej.rascunhos[A].status == "rejeitado" and rej.rascunhos[A].revisado_por == "Ana"
    with pytest.raises(aq.ArquivoError, match="já está rejeitado"):
        aq.rejeitar(rej, A, "Ana", agora=REVISAO)
    rev = aq.aprovar(aq.registrar(vazio, rascunho(payload)), A, "Ana", agora=REVISAO, checklist=lista)
    assert aq.rejeitar(rev, A, "Bia", agora=REVISAO).rascunhos[A].status == "rejeitado"      # revisão posterior vale


def test_agora_sem_fuso_e_recusado(payload, vazio, lista):
    arq = aq.registrar(vazio, rascunho(payload))
    with pytest.raises(ValueError, match="fuso"):
        aq.aprovar(arq, A, "Ana", agora=datetime(2026, 1, 1), checklist=lista)


# ------------------------------------------------------------------ pacote do site
def _arq(payload, vazio, lista, *cods, aprovar_cods=()):
    arq = vazio
    for c in cods:
        arq = aq.registrar(arq, rascunho(payload, c))
    for c in aprovar_cods:
        arq = aq.aprovar(arq, c, "Ana", agora=REVISAO, checklist=lista)
    return arq


def test_pacote_leva_so_revisados_com_titulo_e_fonte_da_equipe(payload, vazio, lista):
    arq = _arq(payload, vazio, lista, A, B, aprovar_cods=(A,))
    pacote, avisos = aq.para_pacote(arq, lista, dataset_sha256=DS, checklist_sha256=CK)
    assert pacote["rotulo"] == rs.ROTULO
    assert pacote["checklist"] == {"versao": "v7", "elaborado_por": "Equipe de teste", "atualizado_em": "2026-11-19"}
    (item,) = pacote["rascunhos"]
    assert (item["cod"], item["nome"], item["revisado_por"], item["revisado_em"]) == (A, "Alfa do Norte", "Ana", "2026-11-20T09:00:00Z")
    assert item["itens"][0] == {"titulo": "Mobilizar a Defesa Civil", "descricao": "Reunir a equipe e revisar contatos.",
                                "fonte": "Fonte de teste 1", "porque": "Prepara a equipe."}
    assert avisos == ["1 rascunho(s) de IA aguardam revisão e não aparecem no site"]


def test_pacote_ordenado_por_nome(payload, vazio, lista):
    pacote, _ = aq.para_pacote(_arq(payload, vazio, lista, B, A, aprovar_cods=(B, A)), lista, dataset_sha256=DS,
                               checklist_sha256=CK)
    assert [r["nome"] for r in pacote["rascunhos"]] == ["Alfa do Norte", "Beta do Sul"]


def test_pacote_none_sem_arquivo_sem_revisado_ou_so_rejeitado(payload, vazio, lista):
    assert aq.para_pacote(None, lista, dataset_sha256=DS, checklist_sha256=CK) == (None, [])
    assert aq.para_pacote(vazio, lista, dataset_sha256=DS, checklist_sha256=CK) == (None, [])
    rej = aq.rejeitar(_arq(payload, vazio, lista, A), A, "Ana", agora=REVISAO)
    assert aq.para_pacote(rej, lista, dataset_sha256=DS, checklist_sha256=CK) == (None, [])


@pytest.mark.parametrize("ds, ckk, usar_lista, trecho", [
    ("ef" * 32, CK, True, "outro dataset"),
    (DS, "ef" * 32, True, "base do checklist mudou"),
    (DS, CK, False, "base do checklist não foi encontrada")])
def test_rascunho_desatualizado_fica_fora_do_site_e_avisa(payload, vazio, lista, ds, ckk, usar_lista, trecho):
    arq = _arq(payload, vazio, lista, A, aprovar_cods=(A,))
    pacote, avisos = aq.para_pacote(arq, lista if usar_lista else None, dataset_sha256=ds,
                                    checklist_sha256=ckk if usar_lista else None)
    assert pacote is None and len(avisos) == 1 and trecho in avisos[0]
