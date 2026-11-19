"""Cliente do Gemini: sessão falsa, nenhuma chamada de rede e nenhuma espera real."""
import ast
import time
from pathlib import Path

import pytest
import requests

from src.ia import cliente as cl
from src.ia import cliente_gemini as cg

CHAVE = "chave-gemini-de-teste-123"


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


def ok(texto="olá", fim="STOP", **extra):
    return Resp(200, {"candidates": [{"content": {"role": "model", "parts": [{"text": texto}]}, "finishReason": fim}],
                      "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": 0},
                      **extra})


def cliente(sessao, **kw):
    esperas = []
    kw.setdefault("dormir", esperas.append)
    c = cg.ClienteGemini(CHAVE, "modelo-x", sessao=sessao, **kw)
    c.esperas_feitas = esperas
    return c


# ------------------------------------------------------------------ pedido
def test_pedido_tem_url_cabecalhos_e_corpo_do_generate_content():
    s = Sessao(ok())
    assert cliente(s, timeout=12, folga_raciocinio=100).gerar("SIS", "USU", max_tokens=321) == "olá"
    (url, kw), = s.chamadas
    assert url == "https://generativelanguage.googleapis.com/v1beta/models/modelo-x:generateContent"
    assert kw["headers"] == {"x-goog-api-key": CHAVE, "content-type": "application/json"}
    assert kw["json"] == {"contents": [{"role": "user", "parts": [{"text": "USU"}]}],
                          "systemInstruction": {"parts": [{"text": "SIS"}]},
                          "generationConfig": {"maxOutputTokens": 421}}
    assert kw["timeout"] == 12.0


def test_a_chave_vai_no_cabecalho_e_nunca_na_url_nem_no_corpo():
    s = Sessao(ok())
    cliente(s).gerar("SIS", "USU")
    (url, kw), = s.chamadas
    assert CHAVE not in url and "key=" not in url and CHAVE not in str(kw["json"])


def test_folga_de_raciocinio_padrao_soma_ao_limite_pedido():
    s = Sessao(ok())
    cg.ClienteGemini(CHAVE, "m", sessao=s).gerar("s", "u", max_tokens=1200)
    assert s.chamadas[0][1]["json"]["generationConfig"]["maxOutputTokens"] == 1200 + cg.FOLGA_RACIOCINIO == 5296


@pytest.mark.parametrize("sistema", ["", "   ", None])
def test_sem_sistema_nao_manda_systeminstruction(sistema):
    s = Sessao(ok())
    cliente(s).gerar(sistema, "u")
    assert "systemInstruction" not in s.chamadas[0][1]["json"]


def test_junta_partes_de_texto_e_ignora_resumo_de_raciocinio_e_lixo():
    r = Resp(200, {"candidates": [{"content": {"parts": [{"text": "pensando", "thought": True}, {"text": "a"},
                                                         {"text": "b", "thought": False}, {"inlineData": {}}, "lixo",
                                                         {"text": 5}]}, "finishReason": "STOP"}]})
    assert cliente(Sessao(r)).gerar("s", "u") == "ab"


def test_guarda_usagemetadata():
    c = cliente(Sessao(ok()))
    c.gerar("s", "u")
    assert c.ultimo_uso == {"promptTokenCount": 10, "candidatesTokenCount": 3, "thoughtsTokenCount": 0}


def test_usagemetadata_que_nao_e_dict_vira_none():
    c = cliente(Sessao(ok(usageMetadata=["x"])))
    c.gerar("s", "u")
    assert c.ultimo_uso is None


# ------------------------------------------------------------------ fim da geração
@pytest.mark.parametrize("fim", ["MAX_TOKENS", "CONTINUATION"])
def test_texto_cortado_e_resposta_truncada_mesmo_sem_texto_visivel(fim):
    with pytest.raises(cl.RespostaTruncada):
        cliente(Sessao(ok("pela metade", fim))).gerar("s", "u")
    vazio = Resp(200, {"candidates": [{"content": {"parts": []}, "finishReason": fim}]})
    with pytest.raises(cl.RespostaTruncada):
        cliente(Sessao(vazio)).gerar("s", "u")


@pytest.mark.parametrize("fim", ["SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "OTHER", "LANGUAGE",
                                 "FINISH_REASON_UNSPECIFIED"])
def test_parada_por_filtro_e_erro_mesmo_com_algum_texto(fim):
    with pytest.raises(cl.ErroLLM, match=f"finishReason {fim}"):
        cliente(Sessao(ok("parcial", fim))).gerar("s", "u")


def test_sem_finishreason_e_erro():
    r = Resp(200, {"candidates": [{"content": {"parts": [{"text": "x"}]}}]})
    with pytest.raises(cl.ErroLLM, match="finishReason ausente"):
        cliente(Sessao(r)).gerar("s", "u")


def test_pedido_bloqueado_sem_candidatos_mostra_o_motivo():
    r = Resp(200, {"promptFeedback": {"blockReason": "SAFETY"}})
    with pytest.raises(cl.ErroLLM, match="bloqueou o pedido .blockReason SAFETY"):
        cliente(Sessao(r)).gerar("s", "u")


@pytest.mark.parametrize("resp", [Resp(200, invalido=True), Resp(200, {"sem": "candidates"}), Resp(200, ["lista"]),
                                  Resp(200, {"candidates": []}), Resp(200, {"candidates": ["x"]}),
                                  Resp(200, {"candidates": [{"finishReason": "STOP"}]}),
                                  Resp(200, {"candidates": [{"content": {"parts": [{"text": "  "}]}, "finishReason": "STOP"}]}),
                                  Resp(200, {"candidates": [], "promptFeedback": "texto"})])
def test_resposta_ilegivel_ou_vazia_e_erro(resp):
    with pytest.raises(cl.ErroLLM):
        cliente(Sessao(resp)).gerar("s", "u")


# ------------------------------------------------------------------ falhas HTTP
@pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
def test_http_diferente_de_200_mostra_status_e_mensagem(status):
    r = Resp(status, {"error": {"code": status, "message": "algo deu errado", "status": "X"}})
    with pytest.raises(cl.ErroLLM, match=f"HTTP {status} do provedor: algo deu errado"):
        cliente(Sessao(r)).gerar("s", "u")


def test_404_aponta_para_o_nome_do_modelo():
    with pytest.raises(cl.ErroLLM, match="PR_GEMINI_MODELO"):
        cliente(Sessao(Resp(404, {"error": {"message": "models/x is not found"}}))).gerar("s", "u")


def test_http_com_corpo_que_nao_e_json_ainda_mostra_o_status():
    with pytest.raises(cl.ErroLLM, match="HTTP 502 do provedor"):
        cliente(Sessao(Resp(502, invalido=True))).gerar("s", "u")


def test_mensagem_do_provedor_e_cortada_em_300_caracteres():
    with pytest.raises(cl.ErroLLM) as e:
        cliente(Sessao(Resp(400, {"error": {"message": "x" * 1000}}))).gerar("s", "u")
    assert str(e.value).count("x") == 300


def test_a_chave_nunca_aparece_nas_mensagens_de_erro():
    r = Resp(400, {"error": {"message": f"API key not valid: {CHAVE}"}})
    with pytest.raises(cl.ErroLLM) as e:
        cliente(Sessao(r)).gerar("s", "u")
    assert CHAVE not in str(e.value) and "***" in str(e.value)


def test_falha_de_rede_vira_erro_sem_a_chave_e_sem_nova_tentativa():
    s = Sessao(erro=requests.ConnectionError(f"falhou com {CHAVE}"))
    c = cliente(s)
    with pytest.raises(cl.ErroLLM) as e:
        c.gerar("s", "u")
    assert "ConnectionError" in str(e.value) and CHAVE not in str(e.value)
    assert e.value.__cause__ is None and e.value.__suppress_context__ is True    # `from None`: nada da exceção original vaza
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
    with pytest.raises(cl.ErroLLM, match="HTTP 429 do provedor: cota .limite de uso da camada gratuita"):
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
    assert cg.ESPERAS_S == (15.0, 30.0)
    assert cg.ClienteGemini(CHAVE)._dormir is time.sleep and cg.ClienteGemini(CHAVE)._esperas == (15.0, 30.0)


# ------------------------------------------------------------------ construção e validação
@pytest.mark.parametrize("kw", [{"api_key": ""}, {"api_key": "   "}, {"modelo": ""}, {"modelo": "   "},
                                {"modelo": "a/b"}, {"modelo": "../x"}, {"modelo": "x?key=1"}, {"modelo": "x:y"},
                                {"modelo": "-x"}, {"modelo": "x y"}, {"modelo": "models/"},
                                {"url_base": "http://x"}, {"folga_raciocinio": -1}, {"esperas": (1.0, -2.0)}])
def test_construcao_invalida(kw):
    args = {"api_key": CHAVE, "modelo": "m", **kw}
    with pytest.raises(ValueError):
        cg.ClienteGemini(**args)


def test_prefixo_models_barra_e_removido_do_nome():
    c = cg.ClienteGemini(CHAVE, " models/gemini-teste ")
    assert c.modelo == "gemini-teste" and c._url.endswith("/models/gemini-teste:generateContent")


def test_barra_final_da_url_base_e_ignorada():
    c = cg.ClienteGemini(CHAVE, "m", url_base="https://exemplo.test/v1beta/")
    assert c._url == "https://exemplo.test/v1beta/models/m:generateContent"


def test_max_tokens_e_usuario_precisam_ser_validos():
    c = cliente(Sessao(ok()))
    with pytest.raises(ValueError):
        c.gerar("s", "u", max_tokens=0)
    for ruim in ("", "   ", None):
        with pytest.raises(ValueError):
            c.gerar("s", ruim)


def test_repr_nao_mostra_a_chave():
    assert CHAVE not in repr(cliente(Sessao(ok()))) and "modelo-x" in repr(cliente(Sessao(ok())))


def test_sessao_padrao_e_requests():
    assert isinstance(cg.ClienteGemini(CHAVE)._sessao, requests.Session)


def test_cumpre_o_protocolo():
    assert isinstance(cliente(Sessao(ok())), cl.ClienteLLM)
    assert cliente(Sessao(ok())).modelo == "modelo-x"


# ------------------------------------------------------------------ ambiente
def test_desde_ambiente_usa_chave_e_modelo_do_ambiente():
    c = cg.desde_ambiente(ambiente={cg.VAR_CHAVE: f" {CHAVE} ", cg.VAR_MODELO: "modelo-env"})
    assert c.modelo == "modelo-env" and c._chave == CHAVE


def test_desde_ambiente_modelo_explicito_vence_e_padrao_existe():
    assert cg.desde_ambiente("outro", ambiente={cg.VAR_CHAVE: CHAVE, cg.VAR_MODELO: "env"}).modelo == "outro"
    assert cg.desde_ambiente(ambiente={cg.VAR_CHAVE: CHAVE}).modelo == cg.MODELO_PADRAO == "gemini-3.6-flash"


def test_o_modelo_do_claude_no_ambiente_nao_vaza_para_o_gemini():
    amb = {cg.VAR_CHAVE: CHAVE, cl.VAR_MODELO: "claude-sonnet-5-5"}
    assert cg.desde_ambiente(ambiente=amb).modelo == cg.MODELO_PADRAO


@pytest.mark.parametrize("amb", [{}, {cg.VAR_CHAVE: ""}, {cg.VAR_CHAVE: "  "}])
def test_sem_chave_explica_como_definir_sem_pedir_para_gravar_no_projeto(amb):
    with pytest.raises(cl.ChaveAusente, match="GEMINI_API_KEY") as e:
        cg.desde_ambiente(ambiente=amb)
    assert "não vai para o projeto" in str(e.value)


def test_desde_ambiente_le_o_ambiente_real_quando_nao_recebe_um(monkeypatch):
    monkeypatch.setenv(cg.VAR_CHAVE, CHAVE)
    monkeypatch.delenv(cg.VAR_MODELO, raising=False)
    assert cg.desde_ambiente().modelo == cg.MODELO_PADRAO
    monkeypatch.delenv(cg.VAR_CHAVE)
    with pytest.raises(cl.ChaveAusente):
        cg.desde_ambiente()


# ------------------------------------------------------------------ higiene
def test_modulo_sem_chave_escrita_sem_chave_na_url_sem_arquivo_e_so_com_bibliotecas_esperadas():
    fonte = Path(cg.__file__).read_text(encoding="utf-8")
    assert "AIza" not in fonte and "?key=" not in fonte and "&key=" not in fonte
    arvore = ast.parse(fonte)
    mods = set()
    for n in ast.walk(arvore):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    assert {m for m in mods if m not in ("__future__", "os", "re", "time", "requests", "src.ia.cliente")} == set()
    assert not [n for n in ast.walk(arvore) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "open"]
