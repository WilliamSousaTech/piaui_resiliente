"""Previsão do próximo mês: leitura e conferência do arquivo `data/reference/previsao_proximo_mes.json`.

Quem GERA o arquivo é `src/model/prever.py` (rede treinada do zero). Este módulo só LÊ e confere a forma e a
coerência interna, sem torch, sem rede e sem framework web, para o site poder importá-lo. Confere:
  * o aviso de MVP está no arquivo;
  * o mês previsto é a origem + 1 mês e o horizonte é 1;
  * cada município aparece uma vez, em ordem de código, com os dois SPI (spi3 e spi6);
  * o sinal de alerta de cada município bate com o número previsto e o limiar da cabeça (ninguém "edita" um alerta);
  * as contagens do resumo batem com a lista;
  * `dataset_sha256` bate com o dos dados do site (`conferir_dataset`), para nunca misturar previsão de um conjunto
    de dados com o monitor de outro.

Nada aqui calcula previsão.
"""
from __future__ import annotations

import json
from datetime import date
from typing import Literal
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

VERSAO = 1
ARQUIVO_PADRAO = Path("data") / "reference" / "previsao_proximo_mes.json"
IDS = ("spi3", "spi6")
SPI_MAX = 5.0                                  # |SPI| <= ~4,75 pela regra de H em [EPS_H, 1 - EPS_H]


class PrevisaoError(ValueError):
    """Arquivo ausente, ilegível, inválido ou de outro conjunto de dados."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Prob = Field(ge=0, le=1)
Veredito = Literal["superou", "indistinguivel", "pior", "indefinido"]      # comparação de MAE com bootstrap de meses (avaliar.py)
Baseline = Literal["persistencia", "climatologia", "parcial_climatologica", "ridge", "floresta_aleatoria"]


class Cabeca(_Base):
    id: str = Field(pattern=r"^spi\d+$")
    escala: int = Field(ge=1, le=24)
    nome_alvo: str = Field(pattern=r"^alvo_spi\d+_h\d+$")
    limiar_alerta: float = Field(ge=-SPI_MAX, le=0)
    margem: float = Field(gt=0, le=SPI_MAX)
    margem_q: float = Field(gt=0, lt=1)
    margem_cobertura_validacao: float = Prob
    margem_cobertura_2a_metade: float = Prob
    margem_estavel: bool
    n_linhas_validacao: int = Field(gt=0)
    mae: float = Field(ge=0)
    mae_persistencia: float = Field(ge=0)
    mae_parcial_climatologica: float = Field(ge=0)
    mae_mes_alvo: float = Field(ge=0)
    mae_mes_alvo_persistencia: float = Field(ge=0)
    veredito_persistencia: Veredito
    melhor_baseline: Baseline
    veredito_melhor_baseline: Veredito
    acerto_classe: float = Prob
    acerto_sempre_normal: float = Prob
    secas_reais: int = Field(ge=0)
    secas_pegas: int = Field(ge=0)
    alarmes_falsos: int = Field(ge=0)
    revocacao: float = Prob
    precisao: float = Prob

    @model_validator(mode="after")
    def _coerente(self) -> "Cabeca":
        if self.id != f"spi{self.escala}" or self.nome_alvo != f"alvo_spi{self.escala}_h1":
            raise ValueError(f"cabeça incoerente: id={self.id!r}, escala={self.escala}, nome_alvo={self.nome_alvo!r}")
        if self.secas_pegas > self.secas_reais:
            raise ValueError(f"{self.id}: secas pegas ({self.secas_pegas}) maior que as reais ({self.secas_reais})")
        return self


class Celula(_Base):
    previsto: float = Field(ge=-SPI_MAX, le=SPI_MAX)
    dispersao: float = Field(ge=0)
    baixa_conf: bool
    alerta: bool


class Municipio(_Base):
    cod_ibge: str = Field(pattern=r"^\d{7}$")
    spi3: Celula
    spi6: Celula


class Resumo(_Base):
    n_municipios: int = Field(gt=0)
    n_alerta_algum: int = Field(ge=0)
    n_alerta_spi3: int = Field(ge=0)
    n_alerta_spi6: int = Field(ge=0)
    n_baixa_conf_spi3: int = Field(ge=0)
    n_baixa_conf_spi6: int = Field(ge=0)


class Modelo(_Base):
    pasta: str = Field(min_length=1)
    n_membros: int = Field(ge=1)
    receita_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def _mes(s: str) -> pd.Period:
    return pd.Period(s, freq="M")


class Previsao(_Base):
    versao: int
    aviso: str = Field(min_length=20)
    gerado_em: date
    origem: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    alvo: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    horizonte_meses: int
    corte_b: str = Field(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")
    alvo_ja_passou: bool
    teste_congelado_lido: Literal[False]            # a previsão nunca roda com o teste liberado; se isto mudar, o leitor recusa
    dataset_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    modelo: Modelo
    cabecas: tuple[Cabeca, ...]
    resumo: Resumo
    municipios: tuple[Municipio, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _coerente(self) -> "Previsao":
        if self.versao != VERSAO:
            raise ValueError(f"versão {self.versao} do arquivo; esta leitura entende a {VERSAO}")
        if "MVP" not in self.aviso:
            raise ValueError("o aviso de maturidade (MVP experimental) precisa estar no arquivo")
        if self.horizonte_meses != 1 or _mes(self.alvo) != _mes(self.origem) + 1:
            raise ValueError(f"o mês previsto ({self.alvo}) deve ser a origem ({self.origem}) mais 1 mês")
        if tuple(c.id for c in self.cabecas) != IDS:
            raise ValueError(f"cabeças esperadas {IDS}, recebi {tuple(c.id for c in self.cabecas)}")
        cods = [m.cod_ibge for m in self.municipios]
        if cods != sorted(set(cods)):
            raise ValueError("municípios repetidos ou fora da ordem de código")
        for m in self.municipios:
            for h in self.cabecas:
                c = getattr(m, h.id)
                if c.alerta != (c.previsto <= h.limiar_alerta):
                    raise ValueError(f"{m.cod_ibge} {h.id}: o alerta ({c.alerta}) não bate com previsto {c.previsto} "
                                     f"e limiar {h.limiar_alerta}")
        esperado = Resumo(
            n_municipios=len(self.municipios),
            n_alerta_algum=sum(m.spi3.alerta or m.spi6.alerta for m in self.municipios),
            n_alerta_spi3=sum(m.spi3.alerta for m in self.municipios),
            n_alerta_spi6=sum(m.spi6.alerta for m in self.municipios),
            n_baixa_conf_spi3=sum(m.spi3.baixa_conf for m in self.municipios),
            n_baixa_conf_spi6=sum(m.spi6.baixa_conf for m in self.municipios))
        if self.resumo != esperado:
            raise ValueError(f"o resumo {self.resumo.model_dump()} não bate com a lista ({esperado.model_dump()})")
        return self

    def cabeca(self, id_: str) -> Cabeca:
        return next(h for h in self.cabecas if h.id == id_)


def carregar(caminho: Path | str = ARQUIVO_PADRAO) -> Previsao:
    p = Path(caminho)
    try:
        bruto = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise PrevisaoError(f"arquivo da previsão não encontrado: {p}") from e
    except (OSError, ValueError) as e:
        raise PrevisaoError(f"não consegui ler {p}: {e}") from e
    try:
        return Previsao.model_validate(bruto)
    except ValueError as e:                    # pydantic.ValidationError herda de ValueError
        raise PrevisaoError(f"{p} inválido: {e}") from e


def conferir_dataset(prev: Previsao, dataset_sha256: str, cods) -> None:
    """A previsão só vale para os dados do site se vem do mesmo `dataset_sha256` e cobre os mesmos municípios."""
    if prev.dataset_sha256 != dataset_sha256:
        raise PrevisaoError(f"a previsão é do dataset {prev.dataset_sha256[:12]}…, e os dados do site são do "
                            f"{dataset_sha256[:12]}…; gere a previsão de novo (python -m src.model.prever)")
    meus, do_site = {m.cod_ibge for m in prev.municipios}, {str(c) for c in cods}
    if meus != do_site:
        faltam, sobram = sorted(do_site - meus), sorted(meus - do_site)
        raise PrevisaoError(f"a previsão e o site não têm os mesmos municípios (faltam {faltam[:5]}, sobram {sobram[:5]})")


def para_pacote(prev: Previsao) -> dict:
    """Dicionário JSON puro para o pacote do site."""
    return json.loads(prev.model_dump_json())
