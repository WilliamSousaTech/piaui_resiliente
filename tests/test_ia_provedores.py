"""Escolha de provedor: ambiente falso e sessão falsa, nenhuma rede."""
import pytest

from src.ia import cliente as cl
from src.ia import cliente_compat_openai as co
from src.ia import cliente_gemini as cg
from src.ia import provedores as pv
from src.ia.cliente_reserva import ClienteComReserva

AMB_TODOS = {cl.VAR_CHAVE: "k-claude", cg.VAR_CHAVE: "k-gemini", co.VAR_CHAVE_GROQ: "k-groq"}


# ------------------------------------------------------------------ nomes
def test_nomes_normaliza_espacos_e_maiusculas_e_mantem_a_ordem():
    assert pv.nomes(" Gemini , GROQ ") == ["gemini", "groq"]
    assert pv.nomes("groq,gemini") == ["groq", "gemini"]


@pytest.mark.parametrize("texto", ["", "   ", ",", None])
def test_nomes_vazio_e_erro(texto):
    with pytest.raises(ValueError, match="ao menos um provedor"):
        pv.nomes(texto)


def test_nome_desconhecido_lista_os_validos():
    with pytest.raises(ValueError, match="desconhecido: grok .use anthropic, gemini, groq"):
        pv.nomes("gemini,grok")


def test_nome_repetido_e_erro():
    with pytest.raises(ValueError, match="repetido: gemini"):
        pv.nomes("gemini,groq,gemini")


# ------------------------------------------------------------------ um provedor
@pytest.mark.parametrize("nome, classe, modelo", [("anthropic", cl.ClienteAnthropic, cl.MODELO_PADRAO),
                                                  ("gemini", cg.ClienteGemini, cg.MODELO_PADRAO),
                                                  ("groq", co.ClienteCompatOpenAI, co.MODELO_GROQ_PADRAO)])
def test_um_provedor_devolve_o_cliente_dele_sem_reserva(nome, classe, modelo):
    c = pv.montar_cliente(nome, ambiente=AMB_TODOS)
    assert type(c) is classe and c.modelo == modelo


@pytest.mark.parametrize("nome", ["anthropic", "gemini", "groq"])
def test_um_provedor_repassa_a_sessao_para_o_cliente(nome):
    sessao = object()
    assert pv.montar_cliente(nome, ambiente=AMB_TODOS, sessao=sessao)._sessao is sessao


def test_um_provedor_usa_o_modelo_explicito():
    assert pv.montar_cliente("gemini", "gemini-teste", ambiente=AMB_TODOS).modelo == "gemini-teste"
    assert pv.montar_cliente("groq", "grupo/modelo-teste", ambiente=AMB_TODOS).modelo == "grupo/modelo-teste"


@pytest.mark.parametrize("nome, var", [("anthropic", "ANTHROPIC_API_KEY"), ("gemini", "GEMINI_API_KEY"),
                                       ("groq", "GROQ_API_KEY")])
def test_um_provedor_sem_chave_sobe_chaveausente_com_a_variavel_certa(nome, var):
    with pytest.raises(cl.ChaveAusente, match=var):
        pv.montar_cliente(nome, ambiente={})


# ------------------------------------------------------------------ lista com reserva
def test_lista_com_chaves_monta_reserva_na_ordem_pedida():
    c = pv.montar_cliente("groq,gemini", ambiente=AMB_TODOS)
    assert isinstance(c, ClienteComReserva) and c.modelos == (co.MODELO_GROQ_PADRAO, cg.MODELO_PADRAO)
    c2 = pv.montar_cliente("gemini,groq,anthropic", ambiente=AMB_TODOS)
    assert c2.modelos == (cg.MODELO_PADRAO, co.MODELO_GROQ_PADRAO, cl.MODELO_PADRAO)


def test_lista_usa_a_variavel_de_modelo_de_cada_provedor():
    amb = {**AMB_TODOS, cg.VAR_MODELO: "g-env", co.VAR_MODELO_GROQ: "grupo/q-env"}
    assert pv.montar_cliente("gemini,groq", ambiente=amb).modelos == ("g-env", "grupo/q-env")


def test_lista_ignora_provedor_sem_chave_e_avisa():
    avisos = []
    c = pv.montar_cliente("gemini,groq,anthropic", ambiente={cg.VAR_CHAVE: "k", cl.VAR_CHAVE: "k2"},
                          aviso=lambda n, m: avisos.append((n, m)))
    assert isinstance(c, ClienteComReserva) and c.modelos == (cg.MODELO_PADRAO, cl.MODELO_PADRAO)
    assert avisos == [("groq", "ignorado: falta a variável GROQ_API_KEY nesta máquina")]


def test_lista_com_so_um_provedor_com_chave_devolve_o_cliente_puro_sem_reserva():
    avisos = []
    c = pv.montar_cliente("gemini,groq", ambiente={cg.VAR_CHAVE: "k"}, aviso=lambda n, m: avisos.append(n))
    assert type(c) is cg.ClienteGemini and avisos == ["groq"]


def test_lista_sem_nenhuma_chave_explica_o_que_definir():
    avisos = []
    with pytest.raises(cl.ChaveAusente, match="GEMINI_API_KEY ou GROQ_API_KEY") as e:
        pv.montar_cliente("gemini,groq", ambiente={}, aviso=lambda n, m: avisos.append(n))
    assert "não vai para o projeto" in str(e.value) and avisos == ["gemini", "groq"]


def test_lista_sem_aviso_nao_quebra_quando_ignora():
    assert type(pv.montar_cliente("gemini,groq", ambiente={cg.VAR_CHAVE: "k"})) is cg.ClienteGemini


def test_modelo_explicito_com_varios_provedores_e_erro_e_nao_monta_nada():
    with pytest.raises(ValueError, match="--modelo vale para um provedor só"):
        pv.montar_cliente("gemini,groq", "x", ambiente=AMB_TODOS)


def test_aviso_de_troca_em_tempo_de_execucao_chega_ao_mesmo_callback():
    class Resp:
        def __init__(self, status, corpo):
            self.status_code, self._c = status, corpo

        def json(self):
            return self._c

    avisos = []

    class SessaoPorUrl:                                  # Gemini devolve 400 (sem repetir); o Groq responde
        def post(self, url, **kw):
            if "googleapis" in url:
                return Resp(400, {"error": {"message": "modelo ruim"}})
            return Resp(200, {"choices": [{"message": {"content": "texto do groq"}, "finish_reason": "stop"}]})

    c = pv.montar_cliente("gemini,groq", ambiente=AMB_TODOS, sessao=SessaoPorUrl(), aviso=lambda n, m: avisos.append((n, m)))
    assert c.gerar("s", "u") == "texto do groq" and c.modelo == co.MODELO_GROQ_PADRAO
    assert len(avisos) == 1 and avisos[0][0] == cg.MODELO_PADRAO and "HTTP 400" in avisos[0][1]
    assert c.usados == {co.MODELO_GROQ_PADRAO: 1}


def test_o_padrao_e_anthropic_para_nao_mudar_o_comportamento_antigo():
    assert pv.PADRAO == "anthropic"
