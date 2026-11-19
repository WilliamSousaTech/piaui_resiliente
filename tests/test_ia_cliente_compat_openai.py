"""Cliente no formato chat completions (Groq): sessão falsa, nenhuma chamada de rede e nenhuma espera real."""
import ast
import time
from pathlib import Path

import pytest
import requests

from src.ia import cliente as cl
from src.ia import cliente_compat_openai as co

CHAVE = "gsk_chave-de-teste-123"
URL = "https://api.exemplo.test/v1/chat/completions"


class Resp:
    def __init__(self, status=200, corpo=None, invalido=False):
        self.status_code, self._corpo, self._invalido = status, corpo, invalido

    def json(self):
        if self._invalido:
            raise ValueError("sem json")
        return self._corpo


class Sessao:
    """Devolve as respostas em ordem; a última se repete. `erro` faz toda chamada levantar."""

    def __init__(self, *resps, erro=None):
        self.resps, self.erro, self.chamadas = list(resps), erro, []

    def post(self, url, **kw):
        self.chamadas.append((url, kw))
        if self.erro:
            raise self.erro
        return self.resps[min(len(self.chamadas), len(self.resps)) - 1]


def ok(texto="olá", fim="stop", **extra):
    return Resp(200, {"choices": [{"index": 0, "message": {"role": "assistant", "content": texto}, "finish_reason": fim}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}, **extra})


def cliente(sessao, **kw):
    esperas = []
    kw.setdefault("dormir", esperas.append)
    kw.setdefault("url", URL)
    c = co.ClienteCompatOpenAI(CHAVE, "modelo-x", sessao=sessao, **kw)
    c.esperas_feitas = esperas
    return c


# ------------------------------------------------------------------ pedido
def test_pedido_tem_url_cabecalhos_e_corpo_de_chat_completions():
    s = Sessao(ok())
    assert cliente(s, timeout=12, folga_raciocinio=100).gerar("SIS", "USU", max_tokens=321) == "olá"
    (url, kw), = s.chamadas
    assert url == URL
    assert kw["headers"] == {"Authorization": f"Bearer {CHAVE}", "Content-Type": "application/json"}
    assert kw["json"] == {"model": "modelo-x", "max_completion_tokens": 421,
                          "messages": [{"role": "system", "content": "SIS"}, {"role": "user", "content": "USU"}]}
    assert kw["timeout"] == 12.0


def test_a_chave_vai_so_no_cabecalho_e_nunca_na_url_nem_no_corpo():
    s = Sessao(ok())
    cliente(s).gerar("SIS", "USU")
    (url, kw), = s.chamadas
    assert CHAVE not in url and CHAVE not in str(kw["json"])


def test_campo_do_limite_de_saida_e_configuravel():
    s = Sessao(ok())
    cliente(s, campo_max_tokens="max_tokens", folga_raciocinio=0).gerar("s", "u", max_tokens=50)
    corpo = s.chamadas[0][1]["json"]
    assert corpo["max_tokens"] == 50 and "max_completion_tokens" not in corpo


def test_folga_de_raciocinio_padrao_soma_ao_limite_pedido():
    s = Sessao(ok())
    co.ClienteCompatOpenAI(CHAVE, "m", url=URL, sessao=s).gerar("s", "u", max_tokens=1200)
    assert s.chamadas[0][1]["json"]["max_completion_tokens"] == 1200 + co.FOLGA_RACIOCINIO == 5296


@pytest.mark.parametrize("sistema", ["", "   ", None])
def test_sem_sistema_nao_manda_mensagem_de_sistema(sistema):
    s = Sessao(ok())
    cliente(s).gerar(sistema, "u")
    assert s.chamadas[0][1]["json"]["messages"] == [{"role": "user", "content": "u"}]


def test_guarda_o_uso_de_tokens():
    c = cliente(Sessao(ok()))
    c.gerar("s", "u")
    assert c.ultimo_uso == {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}


def test_usage_que_nao_e_dict_vira_none():
    c = cliente(Sessao(ok(usage=["x"])))
    c.gerar("s", "u")
    assert c.ultimo_uso is None


def test_ignora_o_campo_de_raciocinio_e_le_so_o_content():
    r = Resp(200, {"choices": [{"message": {"content": "resposta", "reasoning": "pensando"}, "finish_reason": "stop"}]})
    assert cliente(Sessao(r)).gerar("s", "u") == "resposta"


# ------------------------------------------------------------------ fim da geração
def test_texto_cortado_e_resposta_truncada_mesmo_sem_texto_visivel():
    with pytest.raises(cl.RespostaTruncada):
        cliente(Sessao(ok("pela metade", "length"))).gerar("s", "u")
    vazio = Resp(200, {"choices": [{"message": {"content": None}, "finish_reason": "length"}]})
    with pytest.raises(cl.RespostaTruncada):
        cliente(Sessao(vazio)).gerar("s", "u")


@pytest.mark.parametrize("fim", ["content_filter", "tool_calls", "function_call", "error", "", None])
def test_outra_parada_e_erro_mesmo_com_algum_texto(fim):
    with pytest.raises(cl.ErroLLM, match="finish_reason"):
        cliente(Sessao(ok("parcial", fim))).gerar("s", "u")


@pytest.mark.parametrize("resp", [Resp(200, invalido=True), Resp(200, {"sem": "choices"}), Resp(200, ["lista"]),
                                  Resp(200, {"choices": []}), Resp(200, {"choices": ["x"]}),
                                  Resp(200, {"choices": [{"finish_reason": "stop"}]}),
                                  Resp(200, {"choices": [{"message": "texto", "finish_reason": "stop"}]}),
                                  Resp(200, {"choices": [{"message": {"content": 5}, "finish_reason": "stop"}]}),
                                  Resp(200, {"choices": [{"message": {"content": "  "}, "finish_reason": "stop"}]})])
def test_resposta_ilegivel_ou_vazia_e_erro(resp):
    with pytest.raises(cl.ErroLLM):
        cliente(Sessao(resp)).gerar("s", "u")


# ------------------------------------------------------------------ falhas HTTP
@pytest.mark.parametrize("status", [401, 403, 500])
def test_http_diferente_de_200_mostra_status_nome_e_mensagem(status):
    r = Resp(status, {"error": {"message": "algo deu errado", "type": "x"}})
    with pytest.raises(cl.ErroLLM, match=f"HTTP {status} do provedor .provedor.: algo deu errado"):
        cliente(Sessao(r)).gerar("s", "u")


@pytest.mark.parametrize("status", [400, 404])
def test_400_e_404_apontam_para_a_troca_do_modelo(status):
    c = cliente(Sessao(Resp(status, {"error": {"message": "model not found"}})), var_modelo="PR_GROQ_MODELO")
    with pytest.raises(cl.ErroLLM, match="PR_GROQ_MODELO ou --modelo"):
        c.gerar("s", "u")


def test_http_com_corpo_que_nao_e_json_ainda_mostra_o_status():
    with pytest.raises(cl.ErroLLM, match="HTTP 502 do provedor"):
        cliente(Sessao(Resp(502, invalido=True))).gerar("s", "u")


def test_mensagem_do_provedor_e_cortada_em_300_caracteres():
    with pytest.raises(cl.ErroLLM) as e:
        cliente(Sessao(Resp(401, {"error": {"message": "x" * 1000}}))).gerar("s", "u")
    assert str(e.value).count("x") == 300


def test_a_chave_nunca_aparece_nas_mensagens_de_erro():
    r = Resp(401, {"error": {"message": f"Invalid API Key: {CHAVE}"}})
    with pytest.raises(cl.ErroLLM) as e:
        cliente(Sessao(r)).gerar("s", "u")
    assert CHAVE not in str(e.value) and "***" in str(e.value)


def test_falha_de_rede_vira_erro_sem_a_chave_e_sem_nova_tentativa():
    s = Sessao(erro=requests.ConnectionError(f"falhou com {CHAVE}"))
    c = cliente(s)
    with pytest.raises(cl.ErroLLM) as e:
        c.gerar("s", "u")
    assert "ConnectionError" in str(e.value) and CHAVE not in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__ is True    # `from None`: nada vaza
    assert len(s.chamadas) == 1 and c.esperas_feitas == []


# ------------------------------------------------------------------ limite de uso (camada gratuita)
def test_429_espera_e_tenta_de_novo():
    s = Sessao(Resp(429, {"error": {"message": "cota"}}), ok("depois"))
    c = cliente(s)
    assert c.gerar("s", "u") == "depois"
    assert len(s.chamadas) == 2 and c.esperas_feitas == [15.0]


def test_503_e_429_seguidos_esperam_15_e_30_segundos():
    s = Sessao(Resp(503, {"error": {"message": "ocupado"}}), Resp(429, {"error": {"message": "cota"}}), ok("enfim"))
    c = cliente(s)
    assert c.gerar("s", "u") == "enfim"
    assert len(s.chamadas) == 3 and c.esperas_feitas == [15.0, 30.0]


def test_429_persistente_sobe_erro_depois_de_tres_tentativas_e_explica():
    s = Sessao(Resp(429, {"error": {"message": "cota"}}))
    c = cliente(s)
    with pytest.raises(cl.ErroLLM, match="HTTP 429 do provedor .provedor.: cota .limite de uso da camada gratuita"):
        c.gerar("s", "u")
    assert len(s.chamadas) == 3 and c.esperas_feitas == [15.0, 30.0]


def test_503_persistente_tambem_para_depois_de_tres_tentativas():
    s = Sessao(Resp(503, {"error": {"message": "ocupado"}}))
    with pytest.raises(cl.ErroLLM, match="HTTP 503"):
        cliente(s).gerar("s", "u")
    assert len(s.chamadas) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500, 502])
def test_outros_erros_http_nao_sao_repetidos(status):
    s = Sessao(Resp(status, {"error": {"message": "x"}}))
    c = cliente(s)
    with pytest.raises(cl.ErroLLM):
        c.gerar("s", "u")
    assert len(s.chamadas) == 1 and c.esperas_feitas == []


def test_sem_esperas_configuradas_nao_repete_nem_dorme():
    s = Sessao(Resp(429, {"error": {"message": "cota"}}))
    c = cliente(s, esperas=())
    with pytest.raises(cl.ErroLLM, match="HTTP 429"):
        c.gerar("s", "u")
    assert len(s.chamadas) == 1 and c.esperas_feitas == []


def test_espera_padrao_e_a_do_modulo_e_dormir_padrao_e_time_sleep():
    c = co.ClienteCompatOpenAI(CHAVE, "m", url=URL)
    assert co.ESPERAS_S == (15.0, 30.0) and c._esperas == (15.0, 30.0) and c._dormir is time.sleep


# ------------------------------------------------------------------ construção e validação
@pytest.mark.parametrize("kw", [{"api_key": ""}, {"api_key": "   "}, {"modelo": ""}, {"modelo": "   "}, {"modelo": "x y"},
                                {"modelo": "-x"}, {"modelo": "x\ny"}, {"modelo": "x;y"}, {"url": "http://x"},
                                {"campo_max_tokens": "  "}, {"folga_raciocinio": -1}, {"esperas": (1.0, -2.0)}])
def test_construcao_invalida(kw):
    args = {"api_key": CHAVE, "modelo": "m", "url": URL, **kw}
    with pytest.raises(ValueError):
        co.ClienteCompatOpenAI(**args)


@pytest.mark.parametrize("nome", ["openai/gpt-oss-120b", "meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.3-70b-versatile",
                                  "modelo:v1.2_x"])
def test_aceita_nomes_de_modelo_com_barra_e_dois_pontos(nome):
    assert co.ClienteCompatOpenAI(CHAVE, nome, url=URL).modelo == nome


def test_max_tokens_e_usuario_precisam_ser_validos():
    c = cliente(Sessao(ok()))
    with pytest.raises(ValueError):
        c.gerar("s", "u", max_tokens=0)
    for ruim in ("", "   ", None):
        with pytest.raises(ValueError):
            c.gerar("s", ruim)


def test_repr_nao_mostra_a_chave():
    r = repr(cliente(Sessao(ok()), nome="groq"))
    assert CHAVE not in r and "modelo-x" in r and "groq" in r


def test_sessao_padrao_e_requests():
    assert isinstance(co.ClienteCompatOpenAI(CHAVE, "m", url=URL)._sessao, requests.Session)


def test_cumpre_o_protocolo():
    assert isinstance(cliente(Sessao(ok())), cl.ClienteLLM)
    assert cliente(Sessao(ok())).modelo == "modelo-x"


# ------------------------------------------------------------------ Groq pelo ambiente
def test_groq_usa_chave_e_modelo_do_ambiente_e_a_url_do_groq():
    c = co.groq_desde_ambiente(ambiente={co.VAR_CHAVE_GROQ: f" {CHAVE} ", co.VAR_MODELO_GROQ: "modelo-env"})
    assert c.modelo == "modelo-env" and c._chave == CHAVE and c._url == co.URL_GROQ
    assert co.URL_GROQ == "https://api.groq.com/openai/v1/chat/completions"


def test_groq_modelo_explicito_vence_e_padrao_nao_e_o_modelo_desligado():
    assert co.groq_desde_ambiente("outro", ambiente={co.VAR_CHAVE_GROQ: CHAVE, co.VAR_MODELO_GROQ: "env"}).modelo == "outro"
    padrao = co.groq_desde_ambiente(ambiente={co.VAR_CHAVE_GROQ: CHAVE}).modelo
    assert padrao == co.MODELO_GROQ_PADRAO == "openai/gpt-oss-120b" and "llama-3.3-70b" not in padrao


def test_modelos_de_outros_provedores_no_ambiente_nao_vazam_para_o_groq():
    amb = {co.VAR_CHAVE_GROQ: CHAVE, cl.VAR_MODELO: "claude-sonnet-5-5", "PR_GEMINI_MODELO": "gemini-x"}
    assert co.groq_desde_ambiente(ambiente=amb).modelo == co.MODELO_GROQ_PADRAO


@pytest.mark.parametrize("amb", [{}, {co.VAR_CHAVE_GROQ: ""}, {co.VAR_CHAVE_GROQ: "  "}])
def test_groq_sem_chave_explica_como_definir_sem_pedir_para_gravar_no_projeto(amb):
    with pytest.raises(cl.ChaveAusente, match="GROQ_API_KEY") as e:
        co.groq_desde_ambiente(ambiente=amb)
    assert "não vai para o projeto" in str(e.value)


def test_groq_le_o_ambiente_real_quando_nao_recebe_um(monkeypatch):
    monkeypatch.setenv(co.VAR_CHAVE_GROQ, CHAVE)
    monkeypatch.delenv(co.VAR_MODELO_GROQ, raising=False)
    assert co.groq_desde_ambiente().modelo == co.MODELO_GROQ_PADRAO
    monkeypatch.delenv(co.VAR_CHAVE_GROQ)
    with pytest.raises(cl.ChaveAusente):
        co.groq_desde_ambiente()


def test_o_erro_do_groq_aponta_para_a_variavel_do_modelo_do_groq():
    c = co.groq_desde_ambiente(ambiente={co.VAR_CHAVE_GROQ: CHAVE}, sessao=Sessao(Resp(404, {"error": {"message": "x"}})))
    with pytest.raises(cl.ErroLLM, match="PR_GROQ_MODELO"):
        c.gerar("s", "u")


# ------------------------------------------------------------------ higiene
def test_modulo_sem_chave_escrita_sem_arquivo_e_so_com_bibliotecas_esperadas():
    fonte = Path(co.__file__).read_text(encoding="utf-8")
    assert "gsk_" not in fonte
    arvore = ast.parse(fonte)
    mods = set()
    for n in ast.walk(arvore):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert {m for m in mods if m not in ("__future__", "os", "re", "time", "requests", "src.ia.cliente")} == set()
    assert not [n for n in ast.walk(arvore) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open"]


# ------------------------------------------------------------------ esforço de raciocínio
def test_sem_esforco_o_corpo_nao_leva_o_campo():
    s = Sessao(ok())
    cliente(s).gerar("SIS", "USU")
    assert "reasoning_effort" not in s.chamadas[0][1]["json"]


@pytest.mark.parametrize("nivel", ["low", "medium", "high"])
def test_esforco_vai_no_corpo_do_pedido(nivel):
    s = Sessao(ok())
    cliente(s, esforco_raciocinio=nivel).gerar("SIS", "USU", max_tokens=10)
    corpo = s.chamadas[0][1]["json"]
    assert corpo["reasoning_effort"] == nivel
    assert set(corpo) == {"model", "messages", "max_completion_tokens", "reasoning_effort"}


@pytest.mark.parametrize("nivel", ["", "LOW", "max", "none", "nenhum", " low"])
def test_esforco_invalido_e_recusado_na_construcao(nivel):
    with pytest.raises(ValueError, match="esforço de raciocínio"):
        cliente(Sessao(ok()), esforco_raciocinio=nivel)


def test_400_com_esforco_ensina_a_desligar_e_sem_esforco_nao_fala_disso():
    com = cliente(Sessao(Resp(400, {"error": {"message": "bad"}})), esforco_raciocinio="low", var_modelo="PR_GROQ_MODELO")
    with pytest.raises(cl.ErroLLM, match="PR_GROQ_ESFORCO=nenhum"):
        com.gerar("s", "u")
    sem = cliente(Sessao(Resp(400, {"error": {"message": "bad"}})), var_modelo="PR_GROQ_MODELO")
    with pytest.raises(cl.ErroLLM) as e:
        sem.gerar("s", "u")
    assert "PR_GROQ_ESFORCO" not in str(e.value)


def test_groq_pede_esforco_low_por_padrao():
    s = Sessao(ok())
    co.groq_desde_ambiente(ambiente={co.VAR_CHAVE_GROQ: CHAVE}, sessao=s).gerar("SIS", "USU")
    assert s.chamadas[0][1]["json"]["reasoning_effort"] == "low" == co.ESFORCO_GROQ_PADRAO


@pytest.mark.parametrize("valor, esperado", [("high", "high"), (" MEDIUM ", "medium"), ("Low", "low"),
                                             ("nenhum", None), (" NENHUM", None)])
def test_groq_esforco_pelo_ambiente(valor, esperado):
    s = Sessao(ok())
    co.groq_desde_ambiente(ambiente={co.VAR_CHAVE_GROQ: CHAVE, co.VAR_ESFORCO_GROQ: valor}, sessao=s).gerar("S", "U")
    corpo = s.chamadas[0][1]["json"]
    assert corpo.get("reasoning_effort") == esperado
    assert ("reasoning_effort" in corpo) == (esperado is not None)


def test_groq_esforco_invalido_no_ambiente_diz_qual_variavel():
    with pytest.raises(ValueError, match="PR_GROQ_ESFORCO inválida"):
        co.groq_desde_ambiente(ambiente={co.VAR_CHAVE_GROQ: CHAVE, co.VAR_ESFORCO_GROQ: "maximo"})


def test_esforco_vazio_no_ambiente_vira_o_padrao():
    s = Sessao(ok())
    co.groq_desde_ambiente(ambiente={co.VAR_CHAVE_GROQ: CHAVE, co.VAR_ESFORCO_GROQ: "  "}, sessao=s).gerar("S", "U")
    assert s.chamadas[0][1]["json"]["reasoning_effort"] == "low"

