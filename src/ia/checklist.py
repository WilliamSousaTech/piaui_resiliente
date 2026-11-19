"""Base de itens do checklist de planejamento: escrita pela equipe, lida aqui, nunca inventada pela IA.

A IA só *escolhe e explica* itens desta base (pelo `id`). Ela não recebe o campo `fonte` e não pode citar lei ou
artigo: a fonte de cada item (lei, diretriz do TCE-PI ou do MIDR, já conferida no texto-fonte) é escrita pela equipe e
mostrada pelo site ao lado do item (contrato, seções 1.2 e 5.8).

Arquivo: `data/reference/checklist_base.json`
    {"versao": "...", "elaborado_por": "...", "atualizado_em": "AAAA-MM-DD",
     "itens": [{"id": "mobilizar_defesa_civil", "titulo": "...", "descricao": "...", "fonte": "..."}]}
`id` só tem letras minúsculas e "_" (sem dígitos: números do rascunho são conferidos contra a ficha do município).

Sem rede, sem ML, sem framework web.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.data.manifest import sha256_arquivo

ARQUIVO_PADRAO = Path("data") / "reference" / "checklist_base.json"
MAX_ITENS = 30


class ChecklistError(ValueError):
    """Arquivo ausente, ilegível ou fora do formato."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Item(_Base):
    id: str = Field(pattern=r"^[a-z][a-z_]{2,39}$")
    titulo: str = Field(min_length=3, max_length=120)
    descricao: str = Field(min_length=3, max_length=400)
    fonte: str = Field(min_length=3, max_length=300)

    @field_validator("titulo", "descricao", "fonte")
    @classmethod
    def _sem_quebra(cls, v: str) -> str:
        if "\n" in v or "\r" in v or v != v.strip():
            raise ValueError("sem quebra de linha nem espaço nas pontas")
        return v


class Checklist(_Base):
    versao: str = Field(min_length=1, max_length=40)
    elaborado_por: str = Field(min_length=2, max_length=120)
    atualizado_em: date
    itens: tuple[Item, ...] = Field(min_length=1, max_length=MAX_ITENS)

    @model_validator(mode="after")
    def _ids_unicos(self):
        ids = [i.id for i in self.itens]
        repetidos = sorted({x for x in ids if ids.count(x) > 1})
        if repetidos:
            raise ValueError(f"ids repetidos: {', '.join(repetidos)}")
        return self

    def por_id(self) -> dict[str, Item]:
        return {i.id: i for i in self.itens}

    def texto_para_modelo(self) -> str:
        """O que o modelo vê: id, título e descrição. Sem a fonte (a IA não cita lei)."""
        return "\n".join(f"- {i.id}: {i.titulo}. {i.descricao}" for i in self.itens)


def carregar(caminho: Path | str = ARQUIVO_PADRAO) -> Checklist:
    p = Path(caminho)
    try:
        bruto = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ChecklistError(
            f"base do checklist não encontrada: {p}. Ela é escrita pela equipe, com a fonte de cada item "
            f"(formato no topo de src/ia/checklist.py); a IA não inventa itens") from e
    except (OSError, ValueError) as e:
        raise ChecklistError(f"não consegui ler {p}: {e}") from e
    try:
        return Checklist.model_validate(bruto)
    except ValueError as e:                        # pydantic.ValidationError herda de ValueError
        raise ChecklistError(f"{p} inválido: {e}") from e


def sha256(caminho: Path | str = ARQUIVO_PADRAO) -> str:
    """SHA-256 do texto normalizado em LF (o mesmo critério do manifest)."""
    return sha256_arquivo(caminho)
