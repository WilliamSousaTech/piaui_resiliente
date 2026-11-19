"""Cliente com reserva: tenta cada provedor da lista, em ordem, até um devolver texto.

Cumpre `cliente.ClienteLLM` (`modelo` e `gerar`), então o resto do código não sabe que há mais de um provedor.

Regras:
  - Só `ErroLLM` (rede, HTTP, limite de uso, texto cortado, chave ausente, resposta vazia ou bloqueada) passa a vez ao
    próximo. Erro de programação (`ValueError`, `TypeError`) sobe na hora: trocar de provedor não o resolve.
  - `modelo` é o do provedor que respondeu por último, e é isso que o rascunho grava (rastreabilidade: quem revisa vê qual
    modelo escreveu cada texto). Antes da primeira resposta, é o modelo do primeiro da lista.
  - Cada troca é anunciada em `aviso(modelo_que_falhou, motivo)`, se houver, e fica em `falhas` (só da última chamada).
    `usados` conta, por modelo, quantas respostas cada um deu desde que o cliente foi criado.
  - Se todos falham, sobe um `ErroLLM` que lista o motivo de cada um. As mensagens vêm dos clientes, que já escondem a chave.
  - Descanso (opcional, `descanso_s` > 0): o provedor que falha é pulado nas próximas chamadas até o descanso acabar, para
    não gastar 15 a 90 s de espera em cada município quando o limite do dia já estourou ou o serviço está fora. Pular não
    gera aviso nem entra em `falhas`; o provedor que volta a responder sai do descanso. Se TODOS estão em descanso, tenta
    todos na ordem (nunca trava). Com `descanso_s=0` (padrão) o comportamento é o de sempre: cada chamada tenta tudo.
  - O texto devolvido ainda passa por `rascunhos.validar`; a reserva não muda nenhuma regra do rascunho.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from src.ia.cliente import ClienteLLM, ErroLLM


class ClienteComReserva:
    def __init__(self, clientes: Sequence[ClienteLLM], *, aviso: Callable[[str, str], None] | None = None,
                 descanso_s: float = 0.0, relogio: Callable[[], float] = time.monotonic):
        if descanso_s < 0:
            raise ValueError("descanso_s não pode ser negativo")
        lista = tuple(clientes)
        if not lista:
            raise ValueError("a lista de clientes está vazia")
        for c in lista:
            if not isinstance(c, ClienteLLM):
                raise TypeError("todo cliente precisa ter `modelo` e `gerar` (ClienteLLM)")
        self._clientes = lista
        self._aviso = aviso
        self._descanso = float(descanso_s)
        self._relogio = relogio
        self._ate: dict[int, float] = {}                  # posição na lista -> instante em que o descanso acaba
        self._motivo: dict[int, str] = {}                 # posição na lista -> por que falhou da última vez
        self._ultimo: ClienteLLM = lista[0]
        self.falhas: list[tuple[str, str]] = []
        self.usados: dict[str, int] = {}
        self.ultimo_uso: dict | None = None

    @property
    def modelo(self) -> str:
        return self._ultimo.modelo

    @property
    def modelos(self) -> tuple[str, ...]:
        """Modelos da lista, na ordem em que são tentados."""
        return tuple(c.modelo for c in self._clientes)

    @property
    def descanso_s(self) -> float:
        return self._descanso

    @property
    def em_descanso(self) -> tuple[str, ...]:
        """Modelos que estão sendo pulados agora (falharam há menos de `descanso_s`), na ordem da lista."""
        agora = self._relogio()
        return tuple(c.modelo for i, c in enumerate(self._clientes) if self._ate.get(i, 0.0) > agora)

    def __repr__(self) -> str:
        return f"ClienteComReserva(modelos={list(self.modelos)!r})"

    def gerar(self, sistema: str, usuario: str, *, max_tokens: int = 1200) -> str:
        falhas: list[tuple[str, str]] = []
        agora = self._relogio()
        todos = list(enumerate(self._clientes))
        ativos = [(i, c) for i, c in todos if self._ate.get(i, 0.0) <= agora]
        tentar = ativos or todos                          # todos em descanso: tenta todos em vez de travar
        idx_tentados = {i for i, _ in tentar}
        puladas = [(i, c) for i, c in todos if i not in idx_tentados]
        for i, c in tentar:
            try:
                texto = c.gerar(sistema, usuario, max_tokens=max_tokens)
            except ErroLLM as e:
                falhas.append((c.modelo, str(e)))
                self._motivo[i] = str(e)
                self._ate[i] = self._relogio() + self._descanso       # com descanso 0, vence na hora: nunca pula
                if self._aviso is not None:
                    self._aviso(c.modelo, str(e))
                continue
            self._ate.pop(i, None)
            self._ultimo, self.falhas = c, falhas
            self.usados[c.modelo] = self.usados.get(c.modelo, 0) + 1
            uso = getattr(c, "ultimo_uso", None)
            self.ultimo_uso = uso if isinstance(uso, dict) else None
            return texto
        self.falhas = falhas
        motivos = [f"{m}: {msg}" for m, msg in falhas]
        motivos += [f"{c.modelo}: em descanso (falhou antes: {self._motivo.get(i, 'sem detalhe')})" for i, c in puladas]
        raise ErroLLM("nenhum provedor respondeu: " + " | ".join(motivos))
