"""Cliente de modelo de linguagem: um protocolo pequeno e uma implementação para a Messages API da Anthropic.

O resto do código só conhece `ClienteLLM` (`modelo` e `gerar`). Trocar de provedor é escrever outra classe com os
mesmos dois membros; os testes usam um cliente falso e nunca tocam a rede.

Chave de API: só em variável de ambiente (`ANTHROPIC_API_KEY`) na máquina de quem roda a geração. Nunca em arquivo do
projeto, em `params.yaml`, no manifest, no site, em log ou em mensagem de erro (o texto das falhas passa por
`_sem_chave`). O site gerado é estático e não chama IA: o texto já vem pronto e revisado.

Protocolo da API (https://docs.claude.com): POST /v1/messages com cabeçalhos `x-api-key`, `anthropic-version` e
`content-type`, e corpo `{model, max_tokens, system, messages}`; a resposta traz `content` (lista de blocos, os de
texto têm `type == "text"`) e `stop_reason` (`"max_tokens"` quando o texto foi cortado; `"refusal"` quando o modelo
recusou). Conferido na documentação oficial da API em 2026-11-19.
"""
from __future__ import annotations

import os
from typing import Protocol, runtime_checkable

import requests

URL_ANTHROPIC = "https://api.anthropic.com/v1/messages"
VERSAO_API = "2023-06-01"
VAR_CHAVE = "ANTHROPIC_API_KEY"
VAR_MODELO = "PR_LLM_MODELO"
MODELO_PADRAO = "claude-sonnet-5-5"
TIMEOUT_S = 90.0


class ErroLLM(RuntimeError):
    """Falha ao obter texto do modelo (rede, HTTP, resposta vazia ou ilegível)."""


class RespostaTruncada(ErroLLM):
    """O modelo parou por limite de tokens: o texto está cortado e não pode ser usado."""


class ChaveAusente(ErroLLM):
    """A variável de ambiente com a chave de API não está definida."""


@runtime_checkable
class ClienteLLM(Protocol):
    @property
    def modelo(self) -> str: ...

    def gerar(self, sistema: str, usuario: str, *, max_tokens: int = 1200) -> str: ...


class ClienteAnthropic:
    def __init__(self, api_key: str, modelo: str = MODELO_PADRAO, *, sessao: requests.Session | None = None,
                 url: str = URL_ANTHROPIC, timeout: float = TIMEOUT_S):
        if not api_key or not api_key.strip():
            raise ValueError("chave de API vazia")
        if not modelo or not modelo.strip():
            raise ValueError("modelo vazio")
        if not url.startswith("https://"):
            raise ValueError("a URL da API precisa ser https://")
        self._chave = api_key.strip()
        self._modelo = modelo.strip()
        self._url = url
        self._timeout = float(timeout)
        self._sessao = sessao if sessao is not None else requests.Session()
        self.ultimo_uso: dict | None = None            # tokens da última chamada, para o relatório da CLI

    @property
    def modelo(self) -> str:
        return self._modelo

    def __repr__(self) -> str:                         # nunca mostra a chave
        return f"ClienteAnthropic(modelo={self._modelo!r})"

    def _sem_chave(self, texto: str) -> str:
        return texto.replace(self._chave, "***")

    def gerar(self, sistema: str, usuario: str, *, max_tokens: int = 1200) -> str:
        if max_tokens < 1:
            raise ValueError("max_tokens precisa ser positivo")
        cabecalhos = {"x-api-key": self._chave, "anthropic-version": VERSAO_API, "content-type": "application/json"}
        corpo = {"model": self._modelo, "max_tokens": int(max_tokens), "system": sistema,
                 "messages": [{"role": "user", "content": usuario}]}
        try:
            r = self._sessao.post(self._url, headers=cabecalhos, json=corpo, timeout=self._timeout)
        except requests.RequestException as e:
            raise ErroLLM(self._sem_chave(f"falha de rede ao chamar o modelo ({type(e).__name__})")) from None
        try:
            dados = r.json()
        except ValueError:
            dados = None
        if r.status_code != 200:
            msg = ""
            if isinstance(dados, dict) and isinstance(dados.get("error"), dict):
                msg = str(dados["error"].get("message", ""))[:300]
            raise ErroLLM(self._sem_chave(f"HTTP {r.status_code} do provedor" + (f": {msg}" if msg else "")))
        if not isinstance(dados, dict) or not isinstance(dados.get("content"), list):
            raise ErroLLM("resposta do provedor sem o campo content")
        texto = "".join(b.get("text", "") for b in dados["content"]
                        if isinstance(b, dict) and b.get("type") == "text").strip()
        uso = dados.get("usage")
        self.ultimo_uso = uso if isinstance(uso, dict) else None
        if dados.get("stop_reason") == "max_tokens":
            raise RespostaTruncada("o modelo parou por limite de tokens; aumente max_tokens ou peça texto mais curto")
        if dados.get("stop_reason") == "refusal":
            raise ErroLLM("o modelo recusou o pedido (stop_reason refusal); revise a ficha e o checklist")
        if not texto:
            raise ErroLLM("resposta do provedor sem texto")
        return texto


def desde_ambiente(modelo: str | None = None, *, ambiente=None, sessao: requests.Session | None = None) -> ClienteAnthropic:
    """Monta o cliente com a chave de `ANTHROPIC_API_KEY` e o modelo de `modelo`, `PR_LLM_MODELO` ou o padrão."""
    amb = os.environ if ambiente is None else ambiente
    chave = (amb.get(VAR_CHAVE) or "").strip()
    if not chave:
        raise ChaveAusente(f"defina a variável de ambiente {VAR_CHAVE} nesta máquina "
                           f"(PowerShell: $env:{VAR_CHAVE}=\"sua-chave\"); ela não vai para o projeto nem para o Git")
    return ClienteAnthropic(chave, modelo or (amb.get(VAR_MODELO) or "").strip() or MODELO_PADRAO, sessao=sessao)
