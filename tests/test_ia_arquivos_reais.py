"""Arquivos reais do plano (`checklist_base.json` e `rascunhos_ia.json`): válidos, coerentes entre si e sem lei não conferida."""
from pathlib import Path

from src.ia import arquivo as aq
from src.ia import checklist as ck
from src.ia import rascunhos as rs

RAIZ = Path(__file__).resolve().parents[1]
CHECKLIST = RAIZ / ck.ARQUIVO_PADRAO
RASCUNHOS = RAIZ / aq.ARQUIVO_PADRAO

# Citações cujo texto foi conferido em 2026-11-19 nas versões consolidadas (Planalto, Câmara, LegJur):
#   Lei 12.608/2012, art. 8º, II (coordenar o SINPDEC no âmbito local, com União e Estados), III (incorporar as
#   ações de proteção e defesa civil no planejamento municipal) e VI (declarar situação de emergência e estado de
#   calamidade pública); Lei 12.340/2010, art. 3º-A (Plano de Contingência de Proteção e Defesa Civil do Município).
# Para citar outra lei ou artigo: confira o texto-fonte, escreva a conferência acima e acrescente o início da citação aqui.
CITACOES_CONFERIDAS = (
    "Lei 12.608/2012, art. 8º, II e III (",
    "Lei 12.608/2012, art. 8º, II (",
    "Lei 12.608/2012, art. 8º, VI (",
    "Lei 12.340/2010, art. 3º-A (",
)
SEM_BASE = "Prática recomendada; sem base legal conferida pela equipe."


def test_checklist_real_e_valido_e_toda_fonte_e_lei_conferida_ou_diz_que_nao_tem():
    c = ck.carregar(CHECKLIST)
    assert len(c.itens) >= 2
    for i in c.itens:
        if i.fonte == SEM_BASE:
            continue
        assert i.fonte.startswith(CITACOES_CONFERIDAS), (
            f"{i.id}: a fonte {i.fonte!r} cita lei não conferida; confira o texto-fonte (contrato 1.2) antes de acrescentar")


def test_checklist_real_nao_promete_o_que_o_contrato_proibe():
    c = ck.carregar(CHECKLIST)
    texto = " ".join(f"{i.titulo} {i.descricao}" for i in c.itens).lower()
    for proibido in ("garant", "colapso", "mudança climática", "previs", "certeza"):
        assert proibido not in texto, proibido


def test_rascunhos_reais_batem_com_o_checklist_e_continuam_validos():
    arq = aq.carregar(RASCUNHOS)
    assert arq is not None and arq.rascunhos
    lista = ck.carregar(CHECKLIST)
    assert arq.checklist_sha256 == ck.sha256(CHECKLIST), "a base do checklist mudou depois destes rascunhos"
    for cod, r in arq.rascunhos.items():
        assert r.cod_ibge == cod
        assert set(i.id for i in r.itens) <= set(lista.por_id())
        assert rs.validar(r.resumo, [(i.id, i.porque) for i in r.itens], r.ficha, lista) == [], cod
        if r.status == "revisado":
            assert r.revisado_por and r.revisado_em_utc, cod
        else:
            assert r.status in ("pendente_revisao", "rejeitado")


def test_rascunho_gerado_na_sessao_diz_isso_no_modelo_e_nao_finge_ser_revisado():
    arq = aq.carregar(RASCUNHOS)
    for r in arq.rascunhos.values():
        if r.status == "pendente_revisao":
            assert r.revisado_por is None


def test_nenhum_rascunho_da_lista_traz_numero_que_a_ficha_nao_tem():
    arq = aq.carregar(RASCUNHOS)
    lista = ck.carregar(CHECKLIST)
    from src.ia import fatos as ft
    for cod, r in arq.rascunhos.items():
        permitidos = ft.numeros_permitidos(r.ficha, lista.texto_para_modelo())
        for texto in [r.resumo] + [i.porque for i in r.itens]:
            assert ft.numeros_fora(texto, permitidos) == [], (cod, texto)
