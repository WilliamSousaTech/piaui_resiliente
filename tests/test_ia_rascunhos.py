"""Pedido, validação e registro do rascunho (cliente falso, dados sintéticos só aqui, em tests/)."""
import ast
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pytest

from dados_sinteticos import A, B
from src.ia import checklist as ck
from src.ia import fatos as ft
from src.ia import rascunhos as rs

AGORA = datetime(2026, 11, 19, 12, 30, 45, tzinfo=timezone.utc)


class ClienteFalso:
    """Devolve as respostas da fila, na ordem; guarda cada pedido."""
    modelo = "modelo-falso"

    def __init__(self, *respostas):
        self.respostas, self.pedidos = list(respostas), []

    def gerar(self, sistema, usuario, *, max_tokens=1200):
        self.pedidos.append((sistema, usuario, max_tokens))
        return self.respostas.pop(0)


@pytest.fixture
def lista(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "versao": "t", "elaborado_por": "Equipe de teste", "atualizado_em": "2026-11-19",
        "itens": [{"id": "mobilizar_defesa_civil", "titulo": "Mobilizar a Defesa Civil",
                   "descricao": "Reunir a equipe e revisar contatos.", "fonte": "Fonte de teste 1"},
                  {"id": "revisar_abastecimento", "titulo": "Revisar o abastecimento de água",
                   "descricao": "Conferir carros-pipa e pontos de distribuição.", "fonte": "Fonte de teste 2"},
                  {"id": "avisar_produtores", "titulo": "Avisar os produtores rurais",
                   "descricao": "Informar sobre a falta de chuva.", "fonte": "Fonte de teste 3"}]}), encoding="utf-8")
    return ck.carregar(p)


@pytest.fixture
def fatos(payload):
    return ft.fatos_municipio(payload, A)


@pytest.fixture
def ficha(fatos):
    return ft.ficha_texto(fatos)


def resp(resumo="O município pede atenção.", itens=(("mobilizar_defesa_civil", "Prepara a equipe."),
                                                     ("revisar_abastecimento", "Confere a água.")), cerca=False):
    s = json.dumps({"resumo": resumo, "itens": [{"id": i, "porque": p} for i, p in itens]}, ensure_ascii=False)
    return f"```json\n{s}\n```" if cerca else s


# ------------------------------------------------------------------ caminho feliz
def test_gera_rascunho_valido_na_primeira_tentativa(fatos, lista):
    c = ClienteFalso(resp())
    r = rs.gerar_rascunho(fatos, lista, c, agora=AGORA)
    assert (r.cod_ibge, r.nome, r.mes_referencia, r.escala) == (A, "Alfa do Norte", fatos.mes, 3)
    assert r.status == "pendente_revisao" and r.revisado_por is None and r.revisado_em_utc is None
    assert r.modelo == "modelo-falso" and r.tentativas == 1 and r.gerado_em_utc == "2026-11-19T12:30:45Z"
    assert [i.id for i in r.itens] == ["mobilizar_defesa_civil", "revisar_abastecimento"]
    assert r.ficha == ft.ficha_texto(fatos)


def test_o_pedido_leva_a_ficha_e_o_checklist_sem_fonte_e_o_sistema_com_as_regras(fatos, lista, ficha):
    c = ClienteFalso(resp())
    rs.gerar_rascunho(fatos, lista, c, agora=AGORA, max_tokens=777)
    (sistema, usuario, max_tokens), = c.pedidos
    assert sistema == rs.SISTEMA and max_tokens == 777
    assert ficha in usuario and "mobilizar_defesa_civil: Mobilizar a Defesa Civil." in usuario
    assert "Fonte de teste" not in usuario                      # a IA não vê a fonte
    for regra in ("Não cite lei", "Use só números da ficha", "baixa confiança", "somente com um objeto JSON",
                  "Seja proporcional"):
        assert regra.lower() in sistema.lower()


def test_aceita_resposta_dentro_de_cerca_de_codigo(fatos, lista):
    assert rs.gerar_rascunho(fatos, lista, ClienteFalso(resp(cerca=True)), agora=AGORA).tentativas == 1


def test_numero_da_propria_ficha_e_aceito(fatos, lista, ficha):
    spi = ficha.split("SPI-3 observado: ")[1].split(" ")[0]
    r = rs.gerar_rascunho(fatos, lista, ClienteFalso(resp(f"O SPI-3 ficou em {spi} em {ft.MESES[int(fatos.mes[5:]) - 1]}.")),
                          agora=AGORA)
    assert spi in r.resumo


# ------------------------------------------------------------------ recusas
@pytest.mark.parametrize("resumo, trecho", [
    ("Choveu 999 mm no mês.", "número fora da ficha (999)"),
    ("Cerca de 40% abaixo da média.", "número fora da ficha (40)"),
    ("Conforme a Lei 12.340, a equipe age.", "cita fonte ou norma ('Lei')"),
    ("Veja o art. 5 do plano.", "cita fonte ou norma ('art')"),
    ("Segundo o decreto municipal.", "cita fonte ou norma ('decreto')"),
    ("Dados do S2iD indicam.", "cita fonte ou norma ('S2iD')"),
    ("Certamente vai faltar água.", "linguagem de certeza ou de previsão"),
    ("O reservatório vai secar.", "linguagem de certeza ou de previsão"),
    ("Há risco de colapso hídrico.", "linguagem de certeza ou de previsão"),
    ("A previsão é de piora.", "linguagem de certeza ou de previsão"),
    ("Veja https://exemplo.com", "tem HTML, markdown, link"),
    ("**Atenção** à seca.", "tem HTML, markdown, link"),
    ("<b>Atenção</b>", "tem HTML, markdown, link"),
    ("Linha um.\nLinha dois.", "tem HTML, markdown, link"),
    ("", "vazio"),
    ("x" * (rs.MAX_RESUMO + 1), "o limite é"),
])
def test_resumo_recusado(fatos, lista, resumo, trecho):
    c = ClienteFalso(resp(resumo), resp(resumo), resp(resumo))
    with pytest.raises(rs.RascunhoRecusado, match="tentativa 3") as e:
        rs.gerar_rascunho(fatos, lista, c, agora=AGORA)
    assert trecho in " ".join(e.value.tentativas[0]) and len(e.value.tentativas) == 3


@pytest.mark.parametrize("texto", [" começa com espaço", "termina com espaço "])
def test_espaco_nas_pontas_e_recusado(fatos, lista, texto):
    with pytest.raises(rs.RascunhoRecusado, match="espaço nas pontas"):
        rs.gerar_rascunho(fatos, lista, ClienteFalso(*[resp(texto)] * 3), agora=AGORA)


def test_leitura_nao_e_confundida_com_lei_nem_arte_com_art(fatos, lista):
    texto = "A leitura do mês mostra atenção; a arte de planejar ajuda."
    assert rs.gerar_rascunho(fatos, lista, ClienteFalso(resp(texto)), agora=AGORA).resumo == texto


@pytest.mark.parametrize("itens, trecho", [
    ([("inventado", "x")], "não existe na base"),
    ([("mobilizar_defesa_civil", "a"), ("mobilizar_defesa_civil", "b")], "repetido"),
    ([("mobilizar_defesa_civil", "Só um item.")], "pelo menos 2"),
    ([("mobilizar_defesa_civil", "Choveu 31 mm.")] * 2, "número fora da ficha"),
    ([("mobilizar_defesa_civil", "Conforme a Lei X.")] * 2, "cita fonte"),
])
def test_itens_recusados(fatos, lista, itens, trecho):
    r = resp(itens=itens)
    with pytest.raises(rs.RascunhoRecusado) as e:
        rs.gerar_rascunho(fatos, lista, ClienteFalso(r, r, r), agora=AGORA)
    assert trecho in " ".join(e.value.tentativas[0])


def test_mais_itens_que_o_maximo(fatos, tmp_path):
    itens = [{"id": f"item_{chr(97 + n)}", "titulo": "Título", "descricao": "Descrição", "fonte": "Fonte"}
             for n in range(rs.MAX_ITENS_ESCOLHIDOS + 1)]
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"versao": "t", "elaborado_por": "Eq", "atualizado_em": "2026-11-19", "itens": itens}))
    grande = ck.carregar(p)
    r = resp(itens=[(i["id"], "Por quê.") for i in itens])
    with pytest.raises(rs.RascunhoRecusado) as e:
        rs.gerar_rascunho(fatos, grande, ClienteFalso(r, r, r), agora=AGORA)
    assert f"o máximo é {rs.MAX_ITENS_ESCOLHIDOS}" in " ".join(e.value.tentativas[0])


def test_checklist_com_um_so_item_aceita_um(fatos, tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"versao": "t", "elaborado_por": "Eq", "atualizado_em": "2026-11-19", "itens": [
        {"id": "unico_item", "titulo": "Único", "descricao": "Descrição", "fonte": "Fonte"}]}))
    um = ck.carregar(p)
    assert len(rs.gerar_rascunho(fatos, um, ClienteFalso(resp(itens=[("unico_item", "Por quê.")])), agora=AGORA).itens) == 1


@pytest.mark.parametrize("lixo", ["não é json", "[]", '{"resumo": "x"}', '{"resumo": "x", "itens": "y"}',
                                  '{"resumo": "x", "itens": [{"id": "a"}]}', '{"resumo": "x", "itens": [1]}',
                                  '{"resumo": "x", "itens": [{"id": "a", "porque": 3}]}',
                                  '{"resumo": "x", "itens": [], "extra": 1}'])
def test_forma_errada_e_recusada(fatos, lista, lixo):
    with pytest.raises(rs.RascunhoRecusado):
        rs.gerar_rascunho(fatos, lista, ClienteFalso(lixo, lixo, lixo), agora=AGORA)


def test_resumo_que_nao_e_texto_e_recusado(fatos, lista):
    r = json.dumps({"resumo": 5, "itens": [{"id": "mobilizar_defesa_civil", "porque": "ok"}] * 2})
    with pytest.raises(rs.RascunhoRecusado, match="vazio ou não é texto"):
        rs.gerar_rascunho(fatos, lista, ClienteFalso(r, r, r), agora=AGORA)


# ------------------------------------------------------------------ nova tentativa
def test_recusa_volta_ao_modelo_com_os_motivos_e_a_segunda_vale(fatos, lista):
    c = ClienteFalso(resp("Choveu 999 mm."), resp("Texto sem número."))
    r = rs.gerar_rascunho(fatos, lista, c, agora=AGORA)
    assert r.tentativas == 2 and r.resumo == "Texto sem número."
    primeiro, segundo = c.pedidos[0][1], c.pedidos[1][1]
    assert "recusada" not in primeiro
    assert "recusada" in segundo and "número fora da ficha (999)" in segundo
    assert "Choveu 999 mm." not in segundo                      # o texto recusado não é devolvido ao modelo


def test_esgota_as_tentativas(fatos, lista):
    c = ClienteFalso(*[resp("Choveu 999 mm.")] * 2)
    with pytest.raises(rs.RascunhoRecusado) as e:
        rs.gerar_rascunho(fatos, lista, c, agora=AGORA, tentativas=2)
    assert len(c.pedidos) == 2 and len(e.value.tentativas) == 2


def test_falha_do_cliente_sobe_sem_nova_tentativa(fatos, lista):
    class Quebrado(ClienteFalso):
        def gerar(self, *a, **k):
            self.pedidos.append(1)
            raise RuntimeError("rede caiu")
    c = Quebrado()
    with pytest.raises(RuntimeError):
        rs.gerar_rascunho(fatos, lista, c, agora=AGORA)
    assert len(c.pedidos) == 1


# ------------------------------------------------------------------ entradas
def test_municipio_sem_dado_nao_chama_o_modelo(payload, lista):
    f = ft.Fatos(cod_ibge=B, nome="Beta do Sul", mes="2020-03", escala=3, excluido=True, spi_cent=None, classe=None,
                 em_seca=None, baixa_confianca=False, chuva_mes_dec=None, chuva_12m_dec=None, meses_em_seca_12m=0,
                 leituras_confiaveis_12m=0)
    c = ClienteFalso()
    with pytest.raises(ft.FatosError):
        rs.gerar_rascunho(f, lista, c, agora=AGORA)
    assert c.pedidos == []


def test_agora_sem_fuso_e_tentativas_zero(fatos, lista):
    with pytest.raises(ValueError, match="fuso"):
        rs.gerar_rascunho(fatos, lista, ClienteFalso(resp()), agora=datetime(2026, 1, 1))
    with pytest.raises(ValueError, match="tentativas"):
        rs.gerar_rascunho(fatos, lista, ClienteFalso(resp()), agora=AGORA, tentativas=0)


# ------------------------------------------------------------------ modelo de dados
def _base(**k):
    d = dict(cod_ibge=A, nome="Alfa", mes_referencia="2026-07", escala=3, ficha="x" * 12, resumo="r",
             itens=(rs.ItemEscolhido(id="a", porque="b"),), modelo="m", gerado_em_utc="2026-11-19T12:30:45Z", tentativas=1)
    return {**d, **k}


def test_status_revisado_exige_quem_e_quando():
    with pytest.raises(ValueError, match="quem revisou"):
        rs.Rascunho(**_base(status="revisado"))
    assert rs.Rascunho(**_base(status="revisado", revisado_por="Ana", revisado_em_utc="2026-11-20T10:00:00Z"))


def test_pendente_nao_pode_ter_dados_de_revisao():
    with pytest.raises(ValueError, match="pendente"):
        rs.Rascunho(**_base(revisado_por="Ana"))


@pytest.mark.parametrize("campo, valor", [("cod_ibge", "123"), ("mes_referencia", "2026-7"), ("tentativas", 0),
                                          ("gerado_em_utc", "2026-11-19"), ("status", "aprovado"), ("extra", 1)])
def test_campos_invalidos(campo, valor):
    with pytest.raises(ValueError):
        rs.Rascunho(**_base(**{campo: valor}))


def test_rotulo_e_aviso_fixos_estao_no_codigo_e_nao_no_texto_do_modelo():
    assert rs.ROTULO == "Rascunho gerado por IA e revisado por humano, sem valor jurídico."
    assert rs.AVISO_MVP.endswith("em calibração com S2iD/TCE-PI.")


def test_rotulo_de_nao_revisado_e_fixo_e_nunca_diz_que_uma_pessoa_revisou():
    assert rs.ROTULO_NAO_REVISADO == "Rascunho gerado por IA e ainda não revisado por uma pessoa, sem valor jurídico."
    assert "revisado por humano" not in rs.ROTULO_NAO_REVISADO and "sem valor jurídico" in rs.ROTULO_NAO_REVISADO


@pytest.mark.parametrize("status, esperado", [("revisado", rs.ROTULO), ("pendente_revisao", rs.ROTULO_NAO_REVISADO),
                                              ("rejeitado", rs.ROTULO_NAO_REVISADO), ("qualquer", rs.ROTULO_NAO_REVISADO)])
def test_rotulo_do_status_so_afirma_revisao_para_revisado(status, esperado):
    assert rs.rotulo_do_status(status) == esperado


# ------------------------------------------------------------------ higiene
def test_modulo_sem_rede_arquivo_nem_framework():
    fonte = Path(rs.__file__).read_text(encoding="utf-8")
    mods = set()
    for n in ast.walk(ast.parse(fonte)):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    proibidos = ("requests", "httpx", "urllib", "socket", "http", "os", "pathlib", "src.web", "src.ia.cliente", "fastapi",
                 "flask", "sqlite3", "sqlalchemy", "torch")
    assert not [m for m in mods if any(m == p or m.startswith(p + ".") for p in proibidos)]


# ------------------------------------------------------------------ prompt de sistema
def test_o_sistema_tem_as_regras_e_os_limites_do_codigo():
    s = rs.SISTEMA
    for trecho in ("Seja proporcional", "LEITURA DE BAIXA CONFIANÇA", "CARACTERES", "SE FALTAR INFORMAÇÃO",
                   "ANTES DE RESPONDER, CONFIRA", f"até {rs.MAX_RESUMO} caracteres", f"até {rs.MAX_PORQUE} caracteres",
                   f"de 2 a {rs.MAX_ITENS_ESCOLHIDOS})"):
        assert trecho in s, trecho
    assert "{MAX_" not in s and "{{" not in s and "}}" not in s            # nenhum marcador ficou sem trocar
    # toda menção a limite diz o mesmo número que o validador usa (o prompt cita cada limite em mais de um lugar)
    assert set(re.findall(r"até (\d+) caracteres", s)) == {str(rs.MAX_RESUMO), str(rs.MAX_PORQUE)}
    assert set(re.findall(r"de 2 a (\d+)\)", s)) == {str(rs.MAX_ITENS_ESCOLHIDOS)}


def test_o_sistema_orienta_contra_garantir_e_contra_id_abreviado(checklist=None):
    """Nas rodadas reais, o modelo errou por escrever "garante/garantindo" e por abreviar ids longos. O prompt diz o que
    fazer, e as palavras sugeridas no lugar passam no validador."""
    s = rs.SISTEMA
    assert 'Nunca escreva "garante", "garantir" nem "garantindo"' in s and "letra por letra" in s
    for troca in ("assegurar", "ajudar a", "reforçar"):
        assert not rs._CERTEZA.search(troca) and not rs._CITACAO.search(troca), troca
    for proibida in ("garante", "garantir", "garantindo"):
        assert rs._CERTEZA.search(proibida), proibida


def test_o_formato_pedido_no_sistema_e_o_json_que_o_codigo_le():
    modelo = json.loads(rs.SISTEMA.strip().splitlines()[-1])
    assert set(modelo) == {"resumo", "itens"} and set(modelo["itens"][0]) == {"id", "porque"}
    resumo, itens = rs.interpretar(json.dumps(modelo))
    assert resumo == modelo["resumo"] and itens == [("id_da_acao", modelo["itens"][0]["porque"])]


# ------------------------------------------------------------------ caracteres invisíveis e especiais
ITENS_OK = [("mobilizar_defesa_civil", "Prepara a equipe."), ("revisar_abastecimento", "Confere a água.")]


@pytest.mark.parametrize("entrada, saida", [
    ("SPI\u20113", "SPI-3"), ("SPI\u20103", "SPI-3"),                      # hífen e hífen que não quebra linha
    ("a\u00a0b", "a b"), ("a\u202fb", "a b"), ("a\u2009b", "a b"), ("a\u3000b", "a b"),
    ("−0,30 – x ’ “é” ç", "−0,30 – x ’ “é” ç"),                          # menos, travessão, aspas e acentos ficam
    ("a\u200bb", "a\u200bb"),                                           # invisível não é limpo: é recusado
    ("a\tb", "a\tb"),
])
def test_limpar_texto(entrada, saida):
    assert rs.limpar_texto(entrada) == saida


def test_limpar_texto_devolve_o_que_nao_e_texto_como_veio():
    assert rs.limpar_texto(None) is None and rs.limpar_texto(5) == 5


@pytest.mark.parametrize("char, codigo", [
    ("\u200b", "U+200B"), ("\u200d", "U+200D"), ("\u00ad", "U+00AD"), ("\ufeff", "U+FEFF"), ("\t", "U+0009"),
    ("\x00", "U+0000"), ("\u2028", "U+2028"), ("\u2029", "U+2029"), ("\u00a0", "U+00A0"), ("\u202f", "U+202F"),
])
def test_resumo_com_caractere_invisivel_ou_especial_e_recusado(lista, ficha, char, codigo):
    motivos = rs.validar(f"O município pede{char}atenção.", ITENS_OK, ficha, lista)
    assert any(m.startswith("resumo:") and codigo in m and "invisível ou especial" in m for m in motivos), motivos


def test_porque_com_caractere_invisivel_e_recusado_no_item_certo(lista, ficha):
    itens = [("mobilizar_defesa_civil", "Prepara\u200ba equipe."), ("revisar_abastecimento", "Confere a água.")]
    motivos = rs.validar("Texto ok.", itens, ficha, lista)
    assert any(m.startswith("item 1") and "U+200B" in m for m in motivos)
    assert not any(m.startswith("item 2") or m.startswith("resumo") for m in motivos)


def test_acento_sinal_de_menos_travessao_e_aspas_tipograficas_sao_aceitos(lista, ficha):
    texto = "Ação, coração, lá, ç, ã: “aspas”, ’, sinal − e travessão – são aceitos."
    assert rs.validar(texto, ITENS_OK, ficha, lista) == []


def test_quebra_de_linha_tem_so_o_motivo_de_formato(lista, ficha):
    motivos = rs.validar("Linha um.\nLinha dois.", ITENS_OK, ficha, lista)
    assert any("quebra de linha" in m for m in motivos) and not any("invisível" in m for m in motivos)


def test_o_motivo_lista_no_maximo_cinco_codigos_em_ordem_e_sem_repetir(lista, ficha):
    ruins = "\ufeff\u2060\u200f\u200e\u200d\u200c\u200b\u200b"          # 7 diferentes, um repetido
    motivo = next(m for m in rs.validar(f"a{ruins}b", ITENS_OK, ficha, lista) if "invisível" in m)
    assert motivo.count("U+") == 5
    assert "U+200B, U+200C, U+200D, U+200E, U+200F" in motivo


def test_interpretar_limpa_hifen_e_espaco_especial_no_resumo_e_no_porque_mas_nao_no_id():
    bruto = json.dumps({"resumo": "SPI\u20113 em\u202fjulho.",
                        "itens": [{"id": "a\u00a0b", "porque": "Item\u00a0A e SPI\u20113."}]})
    resumo, itens = rs.interpretar(bruto)
    assert resumo == "SPI-3 em julho."
    assert itens == [("a\u00a0b", "Item A e SPI-3.")]


def test_interpretar_mantem_o_invisivel_para_a_validacao_recusar():
    resumo, itens = rs.interpretar(json.dumps({"resumo": "pode\u200bexagerar",
                                               "itens": [{"id": "x", "porque": "ok\u200b"}]}))
    assert "\u200b" in resumo and "\u200b" in itens[0][1]


def test_texto_com_caractere_invisivel_volta_ao_modelo_e_a_segunda_vale(fatos, lista):
    c = ClienteFalso(resp("pode\u200bexagerar a falta."), resp("pode exagerar a falta."))
    r = rs.gerar_rascunho(fatos, lista, c, agora=AGORA)
    assert r.tentativas == 2 and r.resumo == "pode exagerar a falta."
    assert "U+200B" in c.pedidos[1][1] and "invisível" in c.pedidos[1][1]


def test_hifen_especial_do_modelo_e_gravado_como_hifen_comum(fatos, lista):
    c = ClienteFalso(resp("O SPI\u20113 pede\u202fatenção."))
    r = rs.gerar_rascunho(fatos, lista, c, agora=AGORA)
    assert r.tentativas == 1 and r.resumo == "O SPI-3 pede atenção."

