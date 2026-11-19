"""O núcleo do plano ao vivo (src/web/api/_nucleo.js) tem de recusar e aceitar exatamente o que o validador em Python
(src/ia/rascunhos.py) recusa e aceita. Os dois lados rodam nos mesmos textos; o Node lê tudo de uma vez pela entrada
padrão. A única diferença permitida é a recusa de sinal trocado, só do lado JavaScript (testada à parte)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.ia import checklist as ck
from src.ia import fatos as ft
from src.ia import rascunhos as rs
from src.web import build_site as bs

NUCLEO = Path(bs.__file__).parent / "api" / "_nucleo.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node não instalado")

FICHA = "\n".join([
    "Município: Acauã (2200053)",
    "Mês de referência: julho de 2026",
    "SPI-3 observado: −0,30 (classe: próximo do normal)",
    "Leitura de baixa confiança: não",
    "Chuva no mês: 12,5 mm",
    "Chuva nos 12 meses até o mês de referência: 1.234,5 mm",
    "Meses em seca (SPI-3 de −1,0 ou menos) entre as leituras confiáveis dos últimos 12 meses: 3 de 12",
])


def _lista() -> ck.Checklist:
    itens = [{"id": i, "titulo": f"Título {i}", "descricao": f"Descrição do item {i} com 2 pessoas.",
              "fonte": f"Fonte de {i}"} for i in ("mobilizar_defesa_civil", "levantar_carros_pipa", "listar_reservatorios",
                                                   "avisar_comunidades", "revisar_contatos", "registrar_ocorrencias",
                                                   "acompanhar_chuva")]
    return ck.Checklist.model_validate({"versao": "t", "elaborado_por": "Equipe", "atualizado_em": "2026-11-19", "itens": itens})


def _pacote(lista: ck.Checklist) -> dict:
    return {
        "checklist_texto": lista.texto_para_modelo(),
        "checklist": [{"id": i.id, "titulo": i.titulo, "descricao": i.descricao, "fonte": i.fonte} for i in lista.itens],
        "limites": {"max_resumo": rs.MAX_RESUMO, "max_porque": rs.MAX_PORQUE, "max_itens": rs.MAX_ITENS_ESCOLHIDOS},
    }


def _node(operacao: str, entrada) -> list:
    roteiro = (
        "const n = require(process.argv[1]); const e = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
        "const saida = e.casos.map((c) => { try { return {ok: " + operacao + "}; } catch (x) { return {erro: x.message}; } });\n"
        "process.stdout.write(JSON.stringify(saida));\n")
    r = subprocess.run(["node", "-e", roteiro, str(NUCLEO)], input=json.dumps(entrada), capture_output=True, text=True,
                       encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


OK = "Em julho de 2026 o SPI-3 ficou em −0,30. Vale organizar a equipe e acompanhar a chuva."
ITENS_OK = [("mobilizar_defesa_civil", "A chuva do mês foi 12,5 mm; vale reunir a equipe."),
            ("acompanhar_chuva", "Acompanhar mês a mês, com 3 de 12 meses em seca.")]

CASOS_VALIDAR = [
    (OK, ITENS_OK),
    ("", ITENS_OK), ("   ", ITENS_OK), (OK + " ", ITENS_OK), (" " + OK, ITENS_OK),
    ("a" * 601, ITENS_OK), ("a" * 600, ITENS_OK), ("😀" * 601, ITENS_OK), ("😀" * 600, ITENS_OK),
    ("Texto com **negrito**.", ITENS_OK), ("Veja https://exemplo.org agora.", ITENS_OK), ("Veja www.exemplo.org.", ITENS_OK),
    ("Linha um\nlinha dois.", ITENS_OK), ("Lista # item", ITENS_OK), ("<b>oi</b>", ITENS_OK), ("use `código`", ITENS_OK),
    ("Colchete [um].", ITENS_OK),
    ("Palavra​colada.", ITENS_OK), ("Espaço duro.", ITENS_OK), ("Fino aqui.", ITENS_OK),
    ("Hífen‑duro.", ITENS_OK), ("Controle\u0007aqui.", ITENS_OK), ("Separador de linha.", ITENS_OK),
    ("Conforme a lei vigente.", ITENS_OK), ("Segundo o Decreto estadual.", ITENS_OK), ("Cf. art. 5 do texto.", ITENS_OK),
    ("O artigo diz algo.", ITENS_OK), ("Veja o § único.", ITENS_OK), ("Registrar no S2iD.", ITENS_OK),
    ("Pela Constituição.", ITENS_OK), ("Pela constituicao.", ITENS_OK), ("Resolução do órgão.", ITENS_OK),
    ("Conforme a NBR citada.", ITENS_OK), ("Decretos antigos.", ITENS_OK), ("Portarias novas.", ITENS_OK),
    ("A leitura do mês é calma.", ITENS_OK), ("Um partido político.", ITENS_OK), ("Trabalho artesanal.", ITENS_OK),
    ("Garantia de água.", ITENS_OK), ("Isso é garantido.", ITENS_OK), ("Certamente chove.", ITENS_OK),
    ("Sem dúvida, falta chuva.", ITENS_OK), ("Sem duvida, falta chuva.", ITENS_OK), ("O colapso do açude.", ITENS_OK),
    ("Colapsos locais.", ITENS_OK), ("Ele prevê chuva.", ITENS_OK), ("Ele PREVÊ chuva.", ITENS_OK),
    ("Preciso prever o mês.", ITENS_OK), ("Nossa previsão é calma.", ITENS_OK), ("NOSSA PREVISÃO.", ITENS_OK),
    ("Previsões para o mês.", ITENS_OK), ("Ele ãprever algo.", ITENS_OK), ("As éleis dizem.", ITENS_OK), ("Isso garantí.", ITENS_OK), ("Um colapsó local.", ITENS_OK), ("Um quadro previsível.", ITENS_OK), ("Mudança inevitável.", ITENS_OK),
    ("O açude vai secar.", ITENS_OK), ("Vai faltar água.", ITENS_OK), ("Irá piorar.", ITENS_OK), ("Ira piorar.", ITENS_OK),
    ("Vai chover pouco.", ITENS_OK), ("Um prognóstico ruim.", ITENS_OK), ("Definitivamente sim.", ITENS_OK),
    ("Choveu 7,0 mm.", ITENS_OK), ("Choveu 1.234,5 mm no ano.", ITENS_OK), ("Choveu 1234,5 mm no ano.", ITENS_OK),
    ("Choveu 1,234,5 mm.", ITENS_OK), ("Foram 99 pessoas.", ITENS_OK), ("Ficou em 12,50 mm.", ITENS_OK),
    ("O número ٣ aparece.", ITENS_OK), ("Mês 2026-07.", ITENS_OK), ("SPI-3 sem sinal.", ITENS_OK),
    (OK, ITENS_OK[:1]), (OK, []), (OK, ITENS_OK + [("levantar_carros_pipa", "Levantar os carros."),
                                                    ("listar_reservatorios", "Listar os açudes."),
                                                    ("avisar_comunidades", "Avisar as comunidades."),
                                                    ("revisar_contatos", "Revisar contatos."),
                                                    ("registrar_ocorrencias", "Registrar ocorrências.")]),
    (OK, [("id_que_nao_existe", "Texto."), ITENS_OK[1]]),
    (OK, [ITENS_OK[0], ("mobilizar_defesa_civil", "Outra vez.")]),
    (OK, [("mobilizar_defesa_civil", ""), ITENS_OK[1]]),
    (OK, [("mobilizar_defesa_civil", "x" * 301), ITENS_OK[1]]),
    (OK, [("mobilizar_defesa_civil", "Conforme a lei."), ("acompanhar_chuva", "Vai secar com 9,9 mm.")]),
    (OK, [("mobilizar_defesa_civil", 5), ITENS_OK[1]]),
    (None, ITENS_OK), (OK, None), (OK, "texto"),
]


def test_validar_igual_ao_python():
    lista = _lista()
    esperado = []
    for resumo, itens in CASOS_VALIDAR:
        esperado.append(rs.validar(resumo, itens, FICHA, lista))
    casos = [{"resumo": r, "itens": i} for r, i in CASOS_VALIDAR]
    js = _node("n.validar(c.resumo, c.itens, e.ficha, e.pacote)", {"casos": casos, "ficha": FICHA, "pacote": _pacote(lista)})
    assert len(js) == len(esperado)
    for (resumo, itens), py, r in zip(CASOS_VALIDAR, esperado, js):
        assert r.get("ok") == py, (resumo[:60] if isinstance(resumo, str) else resumo, itens, py, r)


def test_casos_cobrem_aceitos_e_recusados():
    lista = _lista()
    veredictos = [not rs.validar(r, i, FICHA, lista) for r, i in CASOS_VALIDAR]
    assert 5 <= sum(veredictos) <= 20 and sum(not v for v in veredictos) >= 55     # o teste não é trivial para nenhum lado


CASOS_INTERPRETAR = [
    '{"resumo": "Texto.", "itens": [{"id": "a", "porque": "Razão."}]}',
    '```json\n{"resumo": "Texto.", "itens": [{"id": "a", "porque": "Razão."}]}\n```',
    '```\n{"resumo": "T.", "itens": []}\n```',
    '  {"resumo": "T.", "itens": []}  ', 'texto antes {"resumo": "T.", "itens": []}',
    '{"resumo": "T.", "itens": [], "extra": 1}', '{"resumo": "T."}', '[]', 'null', '5', '"x"', 'não é json', '',
    '{"resumo": "T.", "itens": "x"}', '{"resumo": "T.", "itens": [1]}', '{"resumo": "T.", "itens": [{"id": "a"}]}',
    '{"resumo": "T.", "itens": [{"id": "a", "porque": "p", "x": 1}]}', '{"resumo": "T.", "itens": [{"id": 1, "porque": "p"}]}',
    '{"resumo": "T.", "itens": [{"id": "a", "porque": 2}]}', '{"resumo": 7, "itens": []}',
    '{"resumo": "SPI‑3 com espaço fino", "itens": [{"id": "a", "porque": "h‐x y"}]}',
]


def test_interpretar_igual_ao_python():
    esperado = []
    for t in CASOS_INTERPRETAR:
        try:
            resumo, itens = rs.interpretar(t)
            esperado.append({"ok": [resumo, [list(x) for x in itens]]})
        except rs.RascunhoError as e:
            esperado.append({"erro": str(e)})
    js = _node("n.interpretar(c)", {"casos": CASOS_INTERPRETAR})
    assert js == esperado


def test_limpar_texto_igual_ao_python():
    textos = ["SPI‑3", "a b", "a b", "a b", "a　b", "ok normal", "‐‑", "a b", "a​b"]
    esperado = [{"ok": rs.limpar_texto(t)} for t in textos]
    assert _node("n.limparTexto(c)", {"casos": textos}) == esperado


def test_pedido_igual_ao_python():
    lista = _lista()
    recusas = [None, [], ["resumo: vazio ou não é texto"], ["a", "b"]]
    esperado = [{"ok": rs.montar_pedido(FICHA, lista, r)} for r in recusas]
    assert _node("n.montarPedido(e.ficha, e.pacote, c)", {"casos": recusas, "ficha": FICHA, "pacote": _pacote(lista)}) == esperado


def test_numeros_iguais_aos_do_python():
    textos = ["1.234,5 e 1234,5", "SPI-3 e 2026-07", "0,30 / 12,5 / 7", "1.5 vira 1,5", "٣ e ٣,٤", "1.000.000,5", "sem número"]
    esperado = [{"ok": ft.numeros(t)} for t in textos]
    assert _node("n.numeros(c)", {"casos": textos}) == esperado


# ------------------------------------------------------------------ só do lado JavaScript: sinal trocado
def _sinais(texto: str, ficha: str = FICHA):
    return _node("n.sinaisDivergentes(c, e.ficha)", {"casos": [texto], "ficha": ficha})[0]["ok"]


def test_sinal_trocado_e_recusado_so_no_javascript():
    assert _sinais("O SPI-3 ficou em 0,30 neste mês.") == ["0,30"]                 # a ficha diz −0,30
    assert _sinais("O SPI-3 ficou em −0,30 neste mês.") == []
    assert _sinais("O SPI-3 ficou em -0,30 neste mês.") == []                      # hífen no lugar do menos: mesmo sinal
    assert _sinais("O SPI-3 ficou em –0,30 neste mês.") == []                      # meia-risca também conta como sinal
    assert _sinais("Choveu −12,5 mm.") == ["12,5"]                                  # a ficha diz 12,5 mm, sem sinal
    assert _sinais("Choveu 12,5 mm.") == []
    assert _sinais("SPI-3 e SPI-6 em 2026-07, com 3 de 12 meses.") == []           # hífen de rótulo e de data não é sinal
    assert _sinais("SPI-0,30 neste mês.") == ["0,30"]                                # hífen colado em letra não é sinal
    assert _sinais("Choveu 1.234,5 mm no ano.") == []
    assert _sinais("Choveu −1.234,5 mm no ano.") == ["1234,5"]
    assert _sinais("de 0,30 a −0,30", ficha="SPI −0,30 e chuva 0,30 mm") == []     # a ficha traz os dois sinais


def test_sinal_trocado_vira_motivo_na_validacao():
    lista = _lista()
    ruim = "Em julho de 2026 o SPI-3 ficou em 0,30. Vale acompanhar a chuva."
    assert rs.validar(ruim, ITENS_OK, FICHA, lista) == []                           # o Python não vê o problema
    js = _node("n.validar(c, e.itens, e.ficha, e.pacote)", {"casos": [ruim], "itens": ITENS_OK, "ficha": FICHA,
                                                              "pacote": _pacote(lista)})[0]["ok"]
    assert js == ["resumo: sinal diferente do da ficha (0,30)"]


# ------------------------------------------------------------------ o plano que o navegador recebe
def test_montar_plano_traz_titulo_e_fonte_da_equipe_e_status_pendente():
    lista = _lista()
    pacote = _pacote(lista) | {"municipios": {"2200053": {"nome": "Acauã", "ficha": FICHA}}, "mes_referencia": "2026-07",
                               "escala": 3, "rotulo_nao_revisado": rs.ROTULO_NAO_REVISADO, "aviso_mvp": rs.AVISO_MVP}
    saida = _node("n.montarPlano(e.pacote, '2200053', e.resumo, e.itens, {modelo: 'm', tentativas: 2, agora: new Date('2026-11-19T18:05:07.123Z')})",
                  {"casos": [0], "pacote": pacote, "resumo": OK, "itens": [list(x) for x in ITENS_OK]})[0]["ok"]
    assert saida["status"] == "pendente_revisao" and saida["rotulo"] == rs.ROTULO_NAO_REVISADO
    assert saida["gerado_em_utc"] == "2026-11-19T18:05:07Z" and saida["tentativas"] == 2 and saida["modelo"] == "m"
    assert saida["itens"][0] == {"id": "mobilizar_defesa_civil", "titulo": "Título mobilizar_defesa_civil",
                                 "descricao": "Descrição do item mobilizar_defesa_civil com 2 pessoas.",
                                 "fonte": "Fonte de mobilizar_defesa_civil", "porque": ITENS_OK[0][1]}
    assert saida["aviso"] == rs.AVISO_MVP and saida["ficha"] == FICHA and saida["nome"] == "Acauã"
