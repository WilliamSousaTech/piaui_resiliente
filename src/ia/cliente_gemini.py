"""Cliente do Gemini (Google AI Studio, camada gratuita ou paga) com a mesma interface de `cliente.ClienteLLM`.

O resto do código só conhece `modelo` e `gerar`; este módulo é a segunda implementação do protocolo (a primeira é
`cliente.ClienteAnthropic`). Os testes usam sessão falsa e nunca tocam a rede.

Chave de API: só na variável de ambiente `GEMINI_API_KEY` da máquina de quem roda a geração. Nunca em arquivo do
projeto, no manifest, no site, na URL, em log ou em mensagem de erro. A chave vai no cabeçalho `x-goog-api-key`, não na
query string (a URL aparece em mensagens de erro de rede e em registros; o cabeçalho não).

Protocolo (conferido na documentação oficial em 2026-11-19, https://ai.google.dev/api/generate-content):
POST {base}/models/{modelo}:generateContent com corpo `{systemInstruction, contents, generationConfig}`; a resposta traz
`candidates[].content.parts[].text`, `candidates[].finishReason` (`STOP` é o fim normal; `MAX_TOKENS` e `CONTINUATION`
são texto cortado; `SAFETY`, `RECITATION`, `BLOCKLIST`, `PROHIBITED_CONTENT`, `SPII`, `OTHER` etc. são parada por filtro),
`promptFeedback.blockReason` (pedido bloqueado) e `usageMetadata` (com `thoughtsTokenCount`).

Pontos que a documentação consultada NÃO confirmou (por isso o código é defensivo e os testes cobrem os dois lados):
  - Se os tokens de raciocínio contam dentro de `maxOutputTokens`. Alguns modelos pensam antes de responder; se contarem,
    um limite curto corta a resposta sem texto visível. Por isso `gerar` pede `max_tokens + folga_raciocinio` ao provedor.
    A folga só aumenta o teto; quem decide o tamanho do texto continua sendo o pedido ("até N caracteres").
  - O formato exato do corpo de erro e o cabeçalho `Retry-After`: lemos `error.message` se existir e ignoramos o resto.

Camada gratuita: tem limite por minuto e por dia. Em 429 e 503 o cliente espera e tenta de novo (`esperas`, 2 repetições
por padrão); se o limite persistir, sobe `ErroLLM` com a explicação. O conteúdo enviado na camada gratuita pode ser usado
pelo Google para melhorar produtos (termos do Gemini API): a ficha só tem dado público, mas não envie dado pessoal.
"""
from __future__ import annotations

import os
import re
import time

import requests

from src.ia.cliente import ChaveAusente, ErroLLM, RespostaTruncada

URL_BASE = "https://generativelanguage.googleapis.com/v1beta"
VAR_CHAVE = "GEMINI_API_KEY"
VAR_MODELO = "PR_GEMINI_MODELO"
# Modelo que a página oficial de depreciações (2026-11-19) lista sem data de desligamento. Confirme no AI Studio se ele
# está liberado na sua camada; troque com PR_GEMINI_MODELO ou --modelo, sem mexer no código.
MODELO_PADRAO = "gemini-3.6-flash"
TIMEOUT_S = 90.0
FOLGA_RACIOCINIO = 4096
ESPERAS_S = (15.0, 30.0)
_MODELO_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_FIM_CORTADO = frozenset({"MAX_TOKENS", "CONTINUATION"})
_REPETIVEIS = frozenset({429, 503})


class ClienteGemini:
    def __init__(self, api_key: str, modelo: str = MODELO_PADRAO, *, sessao: requests.Session | None = None,
                 url_base: str = URL_BASE, timeout: float = TIMEOUT_S, folga_raciocinio: int = FOLGA_RACIOCINIO,
                 esperas: tuple[float, ...] = ESPERAS_S, dormir=time.sleep):
        if not api_key or not api_key.strip():
            raise ValueError("chave de API vazia")
        nome = (modelo or "").strip()
        if nome.startswith("models/"):
            nome = nome[len("models/"):]
        if not _MODELO_OK.match(nome):
            raise ValueError("nome de modelo inválido (use letras, números, ponto, hífen e sublinhado)")
        if not url_base.startswith("https://"):
            raise ValueError("a URL da API precisa ser https://")
        if folga_raciocinio < 0:
            raise ValueError("folga_raciocinio não pode ser negativa")
        if any(e < 0 for e in esperas):
            raise ValueError("esperas não podem ser negativas")
        self._chave = api_key.strip()
        self._modelo = nome
        self._url = f"{url_base.rstrip('/')}/models/{nome}:generateContent"
        self._timeout = float(timeout)
        self._folga = int(folga_raciocinio)
        self._esperas = tuple(float(e) for e in esperas)
        self._dormir = dormir
        self._sessao = sessao if sessao is not None else requests.Session()
        self.ultimo_uso: dict | None = None            # usageMetadata da última chamada, para o relatório da CLI

    @property
    def modelo(self) -> str:
        return self._modelo

    def __repr__(self) -> str:                         # nunca mostra a chave
        return f"ClienteGemini(modelo={self._modelo!r})"

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
        cabecalhos = {"x-goog-api-key": self._chave, "content-type": "application/json"}
        corpo: dict = {"contents": [{"role": "user", "parts": [{"text": usuario}]}],
                       "generationConfig": {"maxOutputTokens": int(max_tokens) + self._folga}}
        if isinstance(sistema, str) and sistema.strip():
            corpo["systemInstruction"] = {"parts": [{"text": sistema}]}
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

    @staticmethod
    def _texto_http(status: int, dados) -> str:
        msg = ""
        if isinstance(dados, dict) and isinstance(dados.get("error"), dict):
            msg = str(dados["error"].get("message", ""))[:300]
        texto = f"HTTP {status} do provedor" + (f": {msg}" if msg else "")
        if status == 429:
            texto += " (limite de uso da camada gratuita por minuto ou por dia; espere e rode de novo)"
        elif status == 404:
            texto += f" (confira o nome do modelo: variável {VAR_MODELO} ou --modelo)"
        return texto

    def _texto_da_resposta(self, dados) -> str:
        if not isinstance(dados, dict):
            raise ErroLLM("resposta do provedor ilegível")
        uso = dados.get("usageMetadata")
        self.ultimo_uso = uso if isinstance(uso, dict) else None
        candidatos = dados.get("candidates")
        if not isinstance(candidatos, list) or not candidatos or not isinstance(candidatos[0], dict):
            bloqueio = (dados.get("promptFeedback") or {}).get("blockReason") if isinstance(dados.get("promptFeedback"), dict) else None
            if bloqueio:
                raise ErroLLM(f"o Gemini bloqueou o pedido (blockReason {bloqueio}); revise a ficha e o checklist")
            raise ErroLLM("resposta do provedor sem candidatos")
        cand = candidatos[0]
        conteudo = cand.get("content")
        partes = conteudo.get("parts") if isinstance(conteudo, dict) else None
        texto = "".join(p["text"] for p in (partes if isinstance(partes, list) else [])
                        if isinstance(p, dict) and isinstance(p.get("text"), str) and p.get("thought") is not True).strip()
        fim = cand.get("finishReason")
        if fim in _FIM_CORTADO:
            raise RespostaTruncada("o modelo parou por limite de tokens; aumente max_tokens ou peça texto mais curto")
        if fim != "STOP":
            raise ErroLLM(f"o Gemini não concluiu a resposta (finishReason {fim or 'ausente'})")
        if not texto:
            raise ErroLLM("resposta do provedor sem texto")
        return texto


def desde_ambiente(modelo: str | None = None, *, ambiente=None, sessao: requests.Session | None = None) -> ClienteGemini:
    """Monta o cliente com a chave de `GEMINI_API_KEY` e o modelo de `modelo`, `PR_GEMINI_MODELO` ou o padrão."""
    amb = os.environ if ambiente is None else ambiente
    chave = (amb.get(VAR_CHAVE) or "").strip()
    if not chave:
        raise ChaveAusente(f"defina a variável de ambiente {VAR_CHAVE} nesta máquina "
                           f"(PowerShell: $env:{VAR_CHAVE}=\"sua-chave\"); ela não vai para o projeto nem para o Git")
    return ClienteGemini(chave, modelo or (amb.get(VAR_MODELO) or "").strip() or MODELO_PADRAO, sessao=sessao)
