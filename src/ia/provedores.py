"""Escolha do provedor de IA pelo nome (`--provedor` da CLI): um só, ou uma lista em que o seguinte é a reserva.

    anthropic   Claude (ANTHROPIC_API_KEY, modelo em PR_LLM_MODELO)       pago; saldo pré-pago no Console
    gemini      Gemini (GEMINI_API_KEY, modelo em PR_GEMINI_MODELO)        tem camada gratuita
    groq        Groq (GROQ_API_KEY, modelo em PR_GROQ_MODELO)              tem camada gratuita

Exemplos: `gemini` (só o Gemini); `gemini,groq` (Gemini primeiro; se falhar por rede, limite de uso ou resposta vazia, o
Groq). Em uma lista, provedor sem chave na máquina é ignorado com aviso; só é erro se nenhum puder ser montado.
`--modelo` vale só quando há um provedor; com vários, use as variáveis de ambiente de cada um.

Este módulo não faz rede: só monta os clientes (que recebem a `sessao` falsa nos testes).
"""
from __future__ import annotations

from collections.abc import Callable

from src.ia import cliente as cl
from src.ia import cliente_compat_openai as co
from src.ia import cliente_gemini as cg
from src.ia.cliente_reserva import ClienteComReserva

PADRAO = "anthropic"
FABRICAS: dict[str, Callable] = {"anthropic": cl.desde_ambiente, "gemini": cg.desde_ambiente, "groq": co.groq_desde_ambiente}
VAR_CHAVE: dict[str, str] = {"anthropic": cl.VAR_CHAVE, "gemini": cg.VAR_CHAVE, "groq": co.VAR_CHAVE_GROQ}


def nomes(texto: str) -> list[str]:
    """Lê `gemini,groq` em ['gemini', 'groq'] (minúsculas, sem espaços). Vazio, desconhecido ou repetido é erro."""
    lista = [n.strip().lower() for n in (texto or "").split(",") if n.strip()]
    if not lista:
        raise ValueError(f"informe ao menos um provedor ({', '.join(FABRICAS)})")
    desconhecidos = [n for n in lista if n not in FABRICAS]
    if desconhecidos:
        raise ValueError(f"provedor desconhecido: {', '.join(desconhecidos)} (use {', '.join(FABRICAS)})")
    repetidos = sorted({n for n in lista if lista.count(n) > 1})
    if repetidos:
        raise ValueError(f"provedor repetido: {', '.join(repetidos)}")
    return lista


def montar_cliente(texto: str, modelo: str | None = None, *, ambiente=None, sessao=None,
                   aviso: Callable[[str, str], None] | None = None):
    """Cliente do provedor (ou `ClienteComReserva` se a lista tiver mais de um com chave). `aviso(nome, motivo)` é
    chamado para cada provedor ignorado e, depois, a cada troca em tempo de execução."""
    lista = nomes(texto)
    if len(lista) == 1:
        return FABRICAS[lista[0]](modelo, ambiente=ambiente, sessao=sessao)
    if modelo is not None:
        raise ValueError("--modelo vale para um provedor só; com vários, use PR_LLM_MODELO, PR_GEMINI_MODELO e "
                         "PR_GROQ_MODELO")
    clientes, sem_chave = [], []
    for n in lista:
        try:
            clientes.append(FABRICAS[n](None, ambiente=ambiente, sessao=sessao))
        except cl.ChaveAusente:
            sem_chave.append(n)
            if aviso is not None:
                aviso(n, f"ignorado: falta a variável {VAR_CHAVE[n]} nesta máquina")
    if not clientes:
        raise cl.ChaveAusente("nenhum provedor da lista tem chave nesta máquina; defina "
                              + " ou ".join(VAR_CHAVE[n] for n in lista)
                              + " (PowerShell: $env:NOME=\"sua-chave\"); ela não vai para o projeto nem para o Git")
    return clientes[0] if len(clientes) == 1 else ClienteComReserva(clientes, aviso=aviso)
