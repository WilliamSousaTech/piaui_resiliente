"""Contexto do El Niño: leitura e validação do arquivo transcrito de fontes oficiais.

Os números NÃO são calculados aqui nem pelo nosso modelo. O arquivo `data/reference/contexto_enso.json` guarda o
que o CPC/NOAA e o INMET publicaram (paráfrase + número copiado), com instituição, data de emissão e link. Este
módulo só confere a forma: referências cruzadas, datas, https, percentuais entre 0 e 100. Quem usa o dado
(o site) mostra a fonte e a data ao lado de cada número e avisa quando a próxima atualização já passou.

Sem rede, sem ML, sem framework web.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ARQUIVO_PADRAO = Path("data") / "reference" / "contexto_enso.json"


class ContextoEnsoError(ValueError):
    """Arquivo ausente, ilegível ou com forma inválida."""


def _https(url: str) -> str:
    if not url.startswith("https://") or " " in url:
        raise ValueError(f"o link precisa começar com https:// e não ter espaço: {url!r}")
    return url


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Fonte(_Base):
    id: str = Field(min_length=1, max_length=20, pattern=r"^[a-z][a-z0-9_]*$")
    instituicao: str = Field(min_length=3)
    titulo: str = Field(min_length=3)
    emitido_em: date
    url: str
    url_documento: str | None = None
    proxima_atualizacao: date | None = None

    _u = field_validator("url", "url_documento")(lambda v: v if v is None else _https(v))


class Situacao(_Base):
    fonte: str
    status_alerta: str = Field(min_length=3)
    resumo_pt: str = Field(min_length=3, max_length=300)
    forca_pt: str = Field(min_length=3)
    nino34_c: float = Field(ge=-6, le=6)
    prob_forte_ou_muito_forte_min_pct: int = Field(ge=0, le=100)
    periodo_forte_pt: str = Field(min_length=3)
    prob_evento_historico_pct: int = Field(ge=0, le=100)
    periodo_historico_pt: str = Field(min_length=3)
    impactos_pt: str = Field(min_length=3, max_length=300)


class Leitura(_Base):
    fonte: str
    escopo_pt: str = Field(min_length=3, max_length=120)
    texto: str = Field(min_length=10, max_length=400)


class ContextoEnso(_Base):
    versao: int = Field(ge=1)
    obtido_em: date
    transcrito_de: str = Field(min_length=10)
    fontes: tuple[Fonte, ...] = Field(min_length=1)
    situacao: Situacao
    leituras: tuple[Leitura, ...] = Field(min_length=1)
    limites: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _referencias(self) -> "ContextoEnso":
        ids = [f.id for f in self.fontes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"ids de fonte repetidos: {ids}")
        citadas = {self.situacao.fonte, *(x.fonte for x in self.leituras)}
        if not citadas <= set(ids):
            raise ValueError(f"fonte citada que não está em `fontes`: {sorted(citadas - set(ids))}")
        futuras = [f.id for f in self.fontes if f.emitido_em > self.obtido_em]
        if futuras:
            raise ValueError(f"fonte emitida depois de `obtido_em`: {futuras}")
        return self

    def fonte(self, id_: str) -> Fonte:
        return next(f for f in self.fontes if f.id == id_)

    def desatualizado(self, hoje: date) -> bool:
        """Verdadeiro se alguma fonte já tinha atualização marcada antes de `hoje`."""
        return any(f.proxima_atualizacao is not None and f.proxima_atualizacao < hoje for f in self.fontes)


def carregar(caminho: Path | str = ARQUIVO_PADRAO) -> ContextoEnso:
    p = Path(caminho)
    try:
        bruto = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ContextoEnsoError(f"arquivo do contexto El Niño não encontrado: {p}") from e
    except (OSError, ValueError) as e:
        raise ContextoEnsoError(f"não consegui ler {p}: {e}") from e
    try:
        return ContextoEnso.model_validate(bruto)
    except ValueError as e:                    # pydantic.ValidationError herda de ValueError
        raise ContextoEnsoError(f"{p} inválido: {e}") from e


def para_pacote(ctx: ContextoEnso) -> dict:
    """Dicionário JSON puro para o pacote do site (datas em ISO)."""
    return json.loads(ctx.model_dump_json())
