"""Cliente de LLM: sessão falsa, nenhuma chamada de rede."""
import ast
from pathlib import Path

import pytest
import requests

from src.ia import cliente as cl

CHAVE = "chave-de-teste-123"


class Resp:
    def __init__(self, status=200, corpo=None, invalido=False):
        self.status_code, self._corpo, self._invalido = status, corpo, invalido

    def json(self):
        if self._invalido:
            raise ValueError("sem json")
        return self._corpo


class Sessao:
    def __init__(self, resp=None, erro=None):
        self.resp, self.erro, self.chamadas = resp, erro, []

    def post(self, url, **kw):
        self.chamadas.append((url, kw))
        if self.erro:
            raise self.erro
        return self.resp


def ok(texto="olá", **extra):
    return Resp(200, {"content": [{"type": "text", "text": texto}], "stop_reason": "end_turn",
                      "usage": {"input_tokens": 10, "output_tokens": 3}, **extra})


def cliente(sessao, **kw):
    return cl.ClienteAnthropic(CHAVE, "modelo-x", sessao=sessao, **kw)


# ------------------------------------------------------------------ pedido
def test_pedido_tem_url_cabecalhos_e_corpo_da_messages_api():
    s = Sessao(ok())
    assert cliente(s, timeout=12).gerar("SIS", "USU", max_tokens=321) == "olá"
    (url, kw), = s.chamadas
    assert url == "https://api.anthropic.com/v1/messages"
    assert kw["headers"] == {"x-api-key": CHAVE, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    assert kw["json"] == {"model": "modelo-x", "max_tokens": 321, "system": "SIS",
                          "messages": [{"role": "user", "content": "USU"}]}
    assert kw["timeout"] == 12.0


def test_junta_blocos_de_texto_e_ignora_os_outros():
    r = Resp(200, {"content": [{"type": "thinking", "thinking": "x"}, {"type": "text", "text": "a"},
                               {"type": "text", "text": "b"}, "lixo"], "stop_reason": "end_turn"})
    assert cliente(Sessao(r)).gerar("s", "u") == "ab"


def test_guarda_o_uso_de_tokens():
    c = cliente(Sessao(ok()))
    c.gerar("s", "u")
    assert c.ultimo_uso == {"input_tokens": 10, "output_tokens": 3}


# ------------------------------------------------------------------ falhas
def test_texto_cortado_por_limite_de_tokens_e_erro():
    r = Resp(200, {"content": [{"type": "text", "text": "pela metade"}], "stop_reason": "max_tokens"})
    with pytest.raises(cl.RespostaTruncada):
        cliente(Sessao(r)).gerar("s", "u")


def test_recusa_do_modelo_e_erro_mesmo_com_algum_texto():
    r = Resp(200, {"content": [{"type": "text", "text": "parcial"}], "stop_reason": "refusal"})
    with pytest.raises(cl.ErroLLM, match="recusou o pedido"):
        cliente(Sessao(r)).gerar("s", "u")


@pytest.mark.parametrize("status", [400, 401, 429, 500, 529])
def test_http_diferente_de_200_mostra_status_e_mensagem(status):
    r = Resp(status, {"error": {"type": "x", "message": "algo deu errado"}})
    with pytest.raises(cl.ErroLLM, match=f"HTTP {status} do provedor: algo deu errado"):
        cliente(Sessao(r)).gerar("s", "u")


def test_a_chave_nunca_aparece_nas_mensagens_de_erro():
    r = Resp(401, {"error": {"message": f"invalid x-api-key {CHAVE}"}})
    with pytest.raises(cl.ErroLLM) as e:
        cliente(Sessao(r)).gerar("s", "u")
    assert CHAVE not in str(e.value) and "***" in str(e.value)


def test_falha_de_rede_vira_erro_sem_a_chave():
    s = Sessao(erro=requests.ConnectionError(f"falhou com {CHAVE}"))
    with pytest.raises(cl.ErroLLM) as e:
        cliente(s).gerar("s", "u")
    assert "ConnectionError" in str(e.value) and CHAVE not in str(e.value)
    assert e.value.__cause__ is None                      # `from None`: nada da exceção original vaza


@pytest.mark.parametrize("resp", [Resp(200, invalido=True), Resp(200, {"sem": "content"}), Resp(200, ["lista"]),
                                  Resp(200, {"content": []}), Resp(200, {"content": [{"type": "text", "text": "  "}]})])
def test_resposta_ilegivel_ou_vazia_e_erro(resp):
    with pytest.raises(cl.ErroLLM):
        cliente(Sessao(resp)).gerar("s", "u")


def test_http_com_corpo_que_nao_e_json_ainda_mostra_o_status():
    with pytest.raises(cl.ErroLLM, match="HTTP 502 do provedor"):
        cliente(Sessao(Resp(502, invalido=True))).gerar("s", "u")


# ------------------------------------------------------------------ construção
@pytest.mark.parametrize("kw", [{"api_key": ""}, {"api_key": "   "}, {"modelo": ""}, {"url": "http://x"}])
def test_construcao_invalida(kw):
    args = {"api_key": CHAVE, "modelo": "m", "url": cl.URL_ANTHROPIC, **kw}
    with pytest.raises(ValueError):
        cl.ClienteAnthropic(**args)


def test_max_tokens_precisa_ser_positivo():
    with pytest.raises(ValueError):
        cliente(Sessao(ok())).gerar("s", "u", max_tokens=0)


def test_repr_nao_mostra_a_chave():
    assert CHAVE not in repr(cliente(Sessao(ok()))) and "modelo-x" in repr(cliente(Sessao(ok())))


def test_sessao_padrao_e_requests():
    assert isinstance(cl.ClienteAnthropic(CHAVE)._sessao, requests.Session)


def test_cumpre_o_protocolo():
    assert isinstance(cliente(Sessao(ok())), cl.ClienteLLM)
    assert cliente(Sessao(ok())).modelo == "modelo-x"


# ------------------------------------------------------------------ ambiente
def test_desde_ambiente_usa_chave_e_modelo_do_ambiente():
    c = cl.desde_ambiente(ambiente={cl.VAR_CHAVE: f" {CHAVE} ", cl.VAR_MODELO: "modelo-env"})
    assert c.modelo == "modelo-env" and c._chave == CHAVE


def test_desde_ambiente_modelo_explicito_vence_e_padrao_existe():
    assert cl.desde_ambiente("outro", ambiente={cl.VAR_CHAVE: CHAVE, cl.VAR_MODELO: "env"}).modelo == "outro"
    assert cl.desde_ambiente(ambiente={cl.VAR_CHAVE: CHAVE}).modelo == cl.MODELO_PADRAO == "claude-sonnet-5-5"


@pytest.mark.parametrize("amb", [{}, {cl.VAR_CHAVE: ""}, {cl.VAR_CHAVE: "  "}])
def test_sem_chave_explica_como_definir_sem_pedir_para_gravar_no_projeto(amb):
    with pytest.raises(cl.ChaveAusente, match="ANTHROPIC_API_KEY") as e:
        cl.desde_ambiente(ambiente=amb)
    assert "não vai para o projeto" in str(e.value)


def test_desde_ambiente_le_o_ambiente_real_quando_nao_recebe_um(monkeypatch):
    monkeypatch.setenv(cl.VAR_CHAVE, CHAVE)
    monkeypatch.delenv(cl.VAR_MODELO, raising=False)
    assert cl.desde_ambiente().modelo == cl.MODELO_PADRAO
    monkeypatch.delenv(cl.VAR_CHAVE)
    with pytest.raises(cl.ChaveAusente):
        cl.desde_ambiente()


# ------------------------------------------------------------------ higiene
def test_modulo_sem_chave_escrita_nem_leitura_de_arquivo_nem_outra_biblioteca_de_rede():
    fonte = Path(cl.__file__).read_text(encoding="utf-8")
    assert "sk-ant" not in fonte
    arvore = ast.parse(fonte)
    mods = set()
    for n in ast.walk(arvore):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert {m for m in mods if m not in ("__future__", "os", "typing", "requests")} == set()
    assert not [n for n in ast.walk(arvore) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open"]
