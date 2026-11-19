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
  - O texto devolvido ainda passa por `rascunhos.validar`; a reserva não muda nenhuma regra do rascunho.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

from src.ia.cliente import ClienteLLM, ErroLLM


class ClienteComReserva:
    def __init__(self, clientes: Sequence[ClienteLLM], *, aviso: Callable[[str, str], None] | None = None):
        lista = tuple(clientes)
        if not lista:
            raise ValueError("a lista de clientes está vazia")
        for c in lista:
            if not isinstance(c, ClienteLLM):
                raise TypeError("todo cliente precisa ter `modelo` e `gerar` (ClienteLLM)")
        self._clientes = lista
        self._aviso = aviso
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

    def __repr__(self) -> str:
        return f"ClienteComReserva(modelos={list(self.modelos)!r})"

    def gerar(self, sistema: str, usuario: str, *, max_tokens: int = 1200) -> str:
        falhas: list[tuple[str, str]] = []
        for c in self._clientes:
            try:
                texto = c.gerar(sistema, usuario, max_tokens=max_tokens)
            except ErroLLM as e:
                falhas.append((c.modelo, str(e)))
                if self._aviso is not None:
                    self._aviso(c.modelo, str(e))
                continue
            self._ultimo, self.falhas = c, falhas
            self.usados[c.modelo] = self.usados.get(c.modelo, 0) + 1
            uso = getattr(c, "ultimo_uso", None)
            self.ultimo_uso = uso if isinstance(uso, dict) else None
            return texto
        self.falhas = falhas
        raise ErroLLM("nenhum provedor respondeu: " + " | ".join(f"{m}: {msg}" for m, msg in falhas))
