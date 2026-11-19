"""Base do checklist (conteúdo sintético só aqui, em tests/; a base real é escrita pela equipe)."""
import json

import pytest

from src.ia import checklist as ck


def item(**k):
    base = {"id": "item_um", "titulo": "Título do item", "descricao": "Descrição do item.", "fonte": "Fonte do item"}
    return {**base, **k}


def doc(*itens, **k):
    base = {"versao": "t1", "elaborado_por": "Equipe de teste", "atualizado_em": "2026-11-19",
            "itens": list(itens) or [item()]}
    return {**base, **k}


def gravar(tmp_path, d, nome="checklist_base.json", nl="\n"):
    p = tmp_path / nome
    p.write_bytes(json.dumps(d, ensure_ascii=False, indent=2).replace("\n", nl).encode("utf-8"))
    return p


def test_carrega_e_indexa_por_id(tmp_path):
    c = ck.carregar(gravar(tmp_path, doc(item(), item(id="item_dois", titulo="Outro"))))
    assert [i.id for i in c.itens] == ["item_um", "item_dois"]
    assert c.por_id()["item_dois"].titulo == "Outro"
    assert str(c.atualizado_em) == "2026-11-19"


def test_texto_para_modelo_nao_mostra_a_fonte(tmp_path):
    c = ck.carregar(gravar(tmp_path, doc(item(fonte="Lei X, art. Y"))))
    txt = c.texto_para_modelo()
    assert "item_um: Título do item. Descrição do item." in txt
    assert "Lei X" not in txt and "art." not in txt


def test_arquivo_ausente_explica_que_a_equipe_escreve(tmp_path):
    with pytest.raises(ck.ChecklistError, match="escrita pela equipe"):
        ck.carregar(tmp_path / "nao_existe.json")


def test_json_quebrado_e_erro_claro(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{nao é json", encoding="utf-8")
    with pytest.raises(ck.ChecklistError, match="não consegui ler"):
        ck.carregar(p)


@pytest.mark.parametrize("id_", ["Item", "item1", "1item", "ab", "a" * 41, "com-hifen", "com espaço", ""])
def test_id_so_letras_minusculas_e_sublinhado(tmp_path, id_):
    with pytest.raises(ck.ChecklistError, match="inválido"):
        ck.carregar(gravar(tmp_path, doc(item(id=id_))))


def test_ids_repetidos(tmp_path):
    with pytest.raises(ck.ChecklistError, match="ids repetidos: item_um"):
        ck.carregar(gravar(tmp_path, doc(item(), item(titulo="Outro"))))


@pytest.mark.parametrize("campo", ["titulo", "descricao", "fonte"])
def test_campos_obrigatorios_nao_podem_ser_vazios_nem_curtos(tmp_path, campo):
    with pytest.raises(ck.ChecklistError):
        ck.carregar(gravar(tmp_path, doc(item(**{campo: ""}))))


def test_fonte_e_obrigatoria(tmp_path):
    i = item()
    del i["fonte"]
    with pytest.raises(ck.ChecklistError, match="fonte"):
        ck.carregar(gravar(tmp_path, doc(i)))


@pytest.mark.parametrize("campo", ["titulo", "descricao", "fonte"])
def test_quebra_de_linha_e_espaco_nas_pontas_sao_recusados(tmp_path, campo):
    for v in ("linha 1\nlinha 2", " com espaço antes", "com espaço depois "):
        with pytest.raises(ck.ChecklistError, match="sem quebra"):
            ck.carregar(gravar(tmp_path, doc(item(**{campo: v}))))


def test_limites_de_tamanho(tmp_path):
    with pytest.raises(ck.ChecklistError):
        ck.carregar(gravar(tmp_path, doc(item(titulo="x" * 121))))
    with pytest.raises(ck.ChecklistError):
        ck.carregar(gravar(tmp_path, doc(item(descricao="x" * 401))))
    def ident(n):
        return "item_" + chr(97 + n // 26) + chr(97 + n % 26)
    ok = [item(id=ident(n)) for n in range(ck.MAX_ITENS)]
    assert len(ck.carregar(gravar(tmp_path, doc(*ok), "ok.json")).itens) == ck.MAX_ITENS
    with pytest.raises(ck.ChecklistError):
        ck.carregar(gravar(tmp_path, doc(*ok, item(id=ident(ck.MAX_ITENS))), "muitos.json"))


def test_lista_vazia_e_campo_extra_e_data_invalida(tmp_path):
    with pytest.raises(ck.ChecklistError):
        ck.carregar(gravar(tmp_path, {**doc(), "itens": []}))
    with pytest.raises(ck.ChecklistError):
        ck.carregar(gravar(tmp_path, doc(item(extra="x"))))
    with pytest.raises(ck.ChecklistError):
        ck.carregar(gravar(tmp_path, doc(atualizado_em="19/11/2026")))


def test_hash_nao_depende_da_quebra_de_linha(tmp_path):
    d = doc()
    a = ck.sha256(gravar(tmp_path, d, "a.json", "\n"))
    b = ck.sha256(gravar(tmp_path, d, "b.json", "\r\n"))
    assert a == b and len(a) == 64


def test_hash_muda_quando_um_item_muda(tmp_path):
    a = ck.sha256(gravar(tmp_path, doc(item()), "a.json"))
    b = ck.sha256(gravar(tmp_path, doc(item(descricao="Outra descrição.")), "b.json"))
    assert a != b
