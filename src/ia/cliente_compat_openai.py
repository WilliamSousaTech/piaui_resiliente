"""Cliente para APIs no formato "chat completions" da OpenAI, com a mesma interface de `cliente.ClienteLLM`.

Hoje serve ao Groq (reserva gratuita do Gemini). A mesma classe serve a outros provedores com o mesmo formato (o Grok da
xAI, por exemplo): troque `url`, `nome` e os nomes das variáveis de ambiente. Os testes usam sessão falsa e nunca tocam a
rede.

Chave de API: só em variável de ambiente (`GROQ_API_KEY` para o Groq) na máquina de quem roda a geração. Nunca em
arquivo do projeto, no manifest, no site, em log ou em mensagem de erro. A chave vai no cabeçalho `Authorization: Bearer`.

Protocolo do Groq (conferido na referência oficial em 2026-11-19, https://console.groq.com/docs/api-reference):
POST https://api.groq.com/openai/v1/chat/completions com corpo `{model, messages, max_completion_tokens}`. O campo
`max_tokens` está marcado como obsoleto em favor de `max_completion_tokens`, que é o que enviamos. A resposta traz
`choices[0].message.content` e `choices[0].finish_reason` (`stop` é o fim normal; `length` é texto cortado; os demais
valores do esquema, como `content_filter` e `tool_calls`, não servem para nós).

Modelo padrão: o changelog oficial do Groq anuncia o desligamento do `llama-3.3-70b-versatile` em 2026-08-16 e recomenda
`openai/gpt-oss-120b`. Esse é um modelo de raciocínio: os tokens de raciocínio podem contar dentro do limite de saída
(a documentação consultada não confirma), por isso `gerar` pede `max_tokens + folga_raciocinio` ao provedor, como no
cliente do Gemini. O nome do modelo muda com frequência: use `PR_GROQ_MODELO` ou `--modelo`, sem mexer no código.

Camada gratuita: tem limite por minuto e por dia. Em 429 e 503 o cliente espera e tenta de novo (`esperas`); se o limite
persistir, sobe `ErroLLM` com a explicação.
"""
from __future__ import annotations

import os
import re
import time

import requests

from src.ia.cliente import ChaveAusente, ErroLLM, RespostaTruncada

URL_GROQ = "https://api.groq.com/openai/v1/chat/completions"
VAR_CHAVE_GROQ = "GROQ_API_KEY"
VAR_MODELO_GROQ = "PR_GROQ_MODELO"
MODELO_GROQ_PADRAO = "openai/gpt-oss-120b"
TIMEOUT_S = 90.0
FOLGA_RACIOCINIO = 4096
ESPERAS_S = (15.0, 30.0)
_MODELO_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
_REPETIVEIS = frozenset({429, 503})


class ClienteCompatOpenAI:
    def __init__(self, api_key: str, modelo: str, *, url: str, nome: str = "provedor", var_modelo: str = "o modelo",
                 sessao: requests.Session | None = None, timeout: float = TIMEOUT_S,
                 folga_raciocinio: int = FOLGA_RACIOCINIO, esperas: tuple[float, ...] = ESPERAS_S, dormir=time.sleep,
                 campo_max_tokens: str = "max_completion_tokens"):
        if not api_key or not api_key.strip():
            raise ValueError("chave de API vazia")
        nome_modelo = (modelo or "").strip()
        if not _MODELO_OK.match(nome_modelo):
            raise ValueError("nome de modelo inválido (use letras, números, ponto, hífen, dois-pontos e barra)")
        if not url.startswith("https://"):
            raise ValueError("a URL da API precisa ser https://")
        if not campo_max_tokens.strip():
            raise ValueError("campo_max_tokens vazio")
        if folga_raciocinio < 0:
            raise ValueError("folga_raciocinio não pode ser negativa")
        if any(e < 0 for e in esperas):
            raise ValueError("esperas não podem ser negativas")
        self._chave = api_key.strip()
        self._modelo = nome_modelo
        self._url = url
        self._nome = nome
        self._var_modelo = var_modelo
        self._timeout = float(timeout)
        self._folga = int(folga_raciocinio)
        self._esperas = tuple(float(e) for e in esperas)
        self._dormir = dormir
        self._campo = campo_max_tokens.strip()
        self._sessao = sessao if sessao is not None else requests.Session()
        self.ultimo_uso: dict | None = None            # `usage` da última chamada, para o relatório da CLI

    @property
    def modelo(self) -> str:
        return self._modelo

    def __repr__(self) -> str:                         # nunca mostra a chave
        return f"ClienteCompatOpenAI(nome={self._nome!r}, modelo={self._modelo!r})"

    def _sem_chave(self, texto: str) -> str:
        return texto.replace(self._chave, "***")

    def _enviar(self, cabecalhos: dict, corpo: dict):
        try:
            return self._sessao.post(self._url, headers=cabecalhos, json=corpo, timeout=self._timeout)
        except requests.RequestException as e:
            raise ErroLLM(self._sem_chave(f"falha de rede ao chamar o modelo ({type(e).__name__})")) from None

    def gerar(self, sistema: str, usuario: str, *, max_tokens: int = 1200) -> str:
        if max_tokens < 1:
            raise ValueError("max_tokens precisa ser positivo")
        if not isinstance(usuario, str) or not usuario.strip():
            raise ValueError("o pedido do usuário está vazio")
        mensagens = []
        if isinstance(sistema, str) and sistema.strip():
            mensagens.append({"role": "system", "content": sistema})
        mensagens.append({"role": "user", "content": usuario})
        cabecalhos = {"Authorization": f"Bearer {self._chave}", "Content-Type": "application/json"}
        corpo = {"model": self._modelo, "messages": mensagens, self._campo: int(max_tokens) + self._folga}
        for tentativa in range(len(self._esperas) + 1):
            r = self._enviar(cabecalhos, corpo)
            if r.status_code in _REPETIVEIS and tentativa < len(self._esperas):
                self._dormir(self._esperas[tentativa])
                continue
            break
        try:
            dados = r.json()
        except ValueError:
            dados = None
        if r.status_code != 200:
            raise ErroLLM(self._sem_chave(self._texto_http(r.status_code, dados)))
        return self._texto_da_resposta(dados)

    def _texto_http(self, status: int, dados) -> str:
        msg = ""
        if isinstance(dados, dict) and isinstance(dados.get("error"), dict):
            msg = str(dados["error"].get("message", ""))[:300]
        texto = f"HTTP {status} do provedor ({self._nome})" + (f": {msg}" if msg else "")
        if status == 429:
            texto += " (limite de uso da camada gratuita por minuto ou por dia; espere e rode de novo)"
        elif status in (400, 404):
            texto += f" (se o modelo não existe mais, troque pelo nome atual: {self._var_modelo} ou --modelo)"
        return texto

    def _texto_da_resposta(self, dados) -> str:
        if not isinstance(dados, dict):
            raise ErroLLM("resposta do provedor ilegível")
        uso = dados.get("usage")
        self.ultimo_uso = uso if isinstance(uso, dict) else None
        escolhas = dados.get("choices")
        if not isinstance(escolhas, list) or not escolhas or not isinstance(escolhas[0], dict):
            raise ErroLLM("resposta do provedor sem choices")
        escolha = escolhas[0]
        mensagem = escolha.get("message")
        conteudo = mensagem.get("content") if isinstance(mensagem, dict) else None
        texto = conteudo.strip() if isinstance(conteudo, str) else ""
        fim = escolha.get("finish_reason")
        if fim == "length":
            raise RespostaTruncada("o modelo parou por limite de tokens; aumente max_tokens ou peça texto mais curto")
        if fim != "stop":
            raise ErroLLM(f"o modelo não concluiu a resposta (finish_reason {fim or 'ausente'})")
        if not texto:
            raise ErroLLM("resposta do provedor sem texto")
        return texto


def groq_desde_ambiente(modelo: str | None = None, *, ambiente=None,
                        sessao: requests.Session | None = None) -> ClienteCompatOpenAI:
    """Cliente do Groq com a chave de `GROQ_API_KEY` e o modelo de `modelo`, `PR_GROQ_MODELO` ou o padrão."""
    amb = os.environ if ambiente is None else ambiente
    chave = (amb.get(VAR_CHAVE_GROQ) or "").strip()
    if not chave:
        raise ChaveAusente(f"defina a variável de ambiente {VAR_CHAVE_GROQ} nesta máquina "
                           f"(PowerShell: $env:{VAR_CHAVE_GROQ}=\"sua-chave\"); ela não vai para o projeto nem para o Git")
    return ClienteCompatOpenAI(chave, modelo or (amb.get(VAR_MODELO_GROQ) or "").strip() or MODELO_GROQ_PADRAO,
                               url=URL_GROQ, nome="groq", var_modelo=VAR_MODELO_GROQ, sessao=sessao)
