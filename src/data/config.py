"""Configuração validada do pipeline de dados (Fase 1).

Lê o params.yaml e devolve um objeto imutável. Lógica pura: sem rede, sem GEE.
As invariantes anti-vazamento do contrato de dados são verificadas aqui.

Limitações conhecidas: o PyYAML aceita chaves duplicadas e fica com a última
(não detectado), e comentários do YAML (ex.: "VERIFICAR" após #) não são vistos
por pendencias()/exigir_resolvido(); só valores.
"""
import re
from pathlib import Path
from typing import Annotated, Literal, Union

import pandas as pd
import yaml
from pydantic import (AfterValidator, BaseModel, ConfigDict, Field, PlainValidator,
                      PrivateAttr, StrictBool, StrictInt, StrictStr, ValidationError,
                      field_validator, model_validator)

PENDENTE = "VERIFICAR"
VERSAO_MINIMA = (0, 2, 0)
CHIRPS_INICIO = pd.Period("1981-01", freq="M")


class ConfigError(ValueError):
    """params.yaml ausente, inválido ou violando o contrato de dados."""


class PendenciaError(ConfigError):
    """Chave exigida pelo módulo ainda está marcada como VERIFICAR."""


def _mes(v):
    if not (isinstance(v, str) and re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", v)):
        raise ValueError('use "AAAA-MM" entre aspas (ex.: "2015-12")')
    return pd.Period(v, freq="M")


def _caminho(v: str) -> str:
    if (not v or v.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", v)
            or ".." in re.split(r"[\\/]", v)):
        raise ValueError("use caminho relativo ao repositório, sem '..'")
    return v


def _int_ou_pendente(v):
    if isinstance(v, int) and not isinstance(v, bool) and v > 0:
        return v
    if isinstance(v, str) and PENDENTE in v:
        return v
    raise ValueError(f"use inteiro positivo ou texto contendo '{PENDENTE}'")


def _idx(p: pd.Period) -> int:
    return p.year * 12 + p.month


Mes = Annotated[pd.Period, PlainValidator(_mes)]
Caminho = Annotated[StrictStr, AfterValidator(_caminho)]
IntOuPendente = Annotated[Union[int, str], PlainValidator(_int_ou_pendente)]
Texto = Annotated[StrictStr, Field(min_length=1)]
Fracao = Annotated[float, Field(gt=0, le=1)]


class _Base(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)


class Projeto(_Base):
    nome: Texto
    equipe: Texto
    versao_params: StrictStr = Field(pattern=r"^\d+\.\d+\.\d+$")
    maturidade: Texto
    seed: StrictInt = Field(ge=0)

    @field_validator("versao_params")
    @classmethod
    def _versao(cls, v):
        if tuple(int(x) for x in v.split(".")) < VERSAO_MINIMA:
            raise ValueError("versão antiga: aplique o diff do contrato (mínimo "
                             + ".".join(map(str, VERSAO_MINIMA)) + ")")
        return v

    @field_validator("maturidade")
    @classmethod
    def _maturidade(cls, v):
        if "experimental" not in v.lower():
            raise ValueError("deve declarar o aviso de MVP experimental")
        return v


class Piloto(_Base):
    nome: Texto
    polo_cod_ibge: StrictStr = Field(pattern=r"^22\d{5}$")  # str de 7 dígitos, UF 22 (PI)
    arquivo: Caminho
    composicao_status: Texto
    usos: tuple[Texto, ...] = Field(min_length=1)


class Escopo(_Base):
    modo: Literal["semiarido", "piloto"]
    arquivo_semiarido: Caminho
    fonte_semiarido: Texto
    malha_municipal_ano: IntOuPendente
    crs_origem: StrictStr = Field(pattern=r"^EPSG:\d+$")
    crs_gee: StrictStr = Field(pattern=r"^EPSG:\d+$")
    incluir_transicao: StrictBool  # municípios da regra de transição entram no escopo?
    simplificacao_graus: Annotated[float, Field(strict=True, gt=0, le=0.01)]  # tolerância p/ o GeoJSON do GEE
    piloto: Piloto


class Janela(_Base):
    inicio: Mes
    fim: Mes


class Alvo(_Base):
    tipo: Literal["spi"]
    escala_spi_meses: StrictInt = Field(ge=1, le=24)
    horizonte_h: StrictInt = Field(ge=1, le=12)


class Split(_Base):
    por: Literal["mes_alvo"]  # invariante anti-vazamento
    treino_fim: Mes
    val_fim: Mes
    teste_fim: Mes
    embargo_meses: StrictInt = Field(ge=0)
    teste_liberado: StrictBool


class Spi(_Base):
    distribuicao: Literal["gama"]
    ajuste: Literal["mle_floc0"]
    zero_inflado: StrictBool
    convencao_zero: Texto  # vocabulário validado no Item 5 (spi.py)
    ajustar_somente_em: Literal["treino"]  # invariante anti-vazamento
    agrupar_por: tuple[Texto, ...] = Field(min_length=1)
    min_anos_treino: StrictInt = Field(ge=1)
    min_positivos_ajuste: StrictInt = Field(ge=2)
    q_baixa_confianca: float = Field(gt=0, lt=1)


class Qualidade(_Base):
    min_cobertura_dias_mes: Fracao
    min_cobertura_pixels: Fracao
    imputacao_precip: Literal["mediana_mensal_treino"]
    max_meses_imputados_consecutivos: StrictInt = Field(ge=0)
    min_fracao_meses_validos_municipio: Fracao


class Features(_Base):
    max_lag_meses: StrictInt = Field(ge=0)


class Chirps(_Base):
    gee_id: Texto
    banda: Texto
    unidade: Texto
    agregacao_mensal: Literal["soma"]
    reducer_espacial: Literal["media"]
    escala_m: StrictInt = Field(gt=0)
    limiar_dia_chuva_mm: Annotated[float, Field(strict=True, gt=0, le=50)]  # dia de chuva: >= limiar


class Inmet(_Base):
    estacoes: Texto
    uso: Texto
    sentinela_faltante: StrictInt


class Mapbiomas(_Base):
    colecao: IntOuPendente
    defasagem_anos: StrictInt = Field(ge=1)  # invariante: mapa de Y-1 para prever em Y


class S2id(_Base):
    uso: Literal["auditoria"]
    entra_como_feature: StrictBool
    cobrade: tuple[Texto, ...] = Field(min_length=1)

    @field_validator("entra_como_feature")
    @classmethod
    def _so_auditoria(cls, v):
        if v:
            raise ValueError("deve ser false: S2iD é só auditoria (contrato, seção 9)")
        return v


class Fontes(_Base):
    chirps: Chirps
    inmet: Inmet
    mapbiomas: Mapbiomas
    s2id: S2id


class Gee(_Base):
    projeto_env: StrictStr = Field(pattern=r"^[A-Z][A-Z0-9_]*$")  # só o NOME da variável
    max_linhas_por_lote: StrictInt = Field(gt=0, le=5000)


class Caminhos(_Base):
    raw: Caminho
    processed: Caminho
    curated: Caminho
    reference: Caminho
    artifacts: Caminho
    docs: Caminho

    @model_validator(mode="after")
    def _camadas_distintas(self):
        norm = {c.replace("\\", "/").strip("/") for c in (self.raw, self.processed, self.curated)}
        if len(norm) < 3:
            raise ValueError("raw, processed e curated devem ser pastas distintas")
        return self


class Manifest(_Base):
    algoritmo_hash: Literal["sha256"]
    arquivo: Caminho


def _pendente(v) -> bool:
    if isinstance(v, str):
        return PENDENTE in v
    if isinstance(v, BaseModel):
        return any(True for _ in _percorrer(v))
    return isinstance(v, tuple) and any(_pendente(x) for x in v)


def _percorrer(modelo: BaseModel, prefixo: str = ""):
    for nome in type(modelo).model_fields:
        v = getattr(modelo, nome)
        if isinstance(v, BaseModel):
            yield from _percorrer(v, f"{prefixo}{nome}.")
        elif _pendente(v):
            yield f"{prefixo}{nome}"


class Config(_Base):
    projeto: Projeto
    escopo: Escopo
    janela: Janela
    alvo: Alvo
    split: Split
    spi: Spi
    qualidade: Qualidade
    features: Features
    fontes: Fontes
    gee: Gee
    caminhos: Caminhos
    manifest: Manifest
    _raiz: Path | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _coerencia(self):
        j, s = self.janela, self.split
        if j.inicio < CHIRPS_INICIO:
            raise ValueError("janela.inicio anterior a 1981-01 (início do CHIRPS)")
        if not (j.inicio < s.treino_fim < s.val_fim < s.teste_fim <= j.fim):
            raise ValueError("datas fora de ordem: inicio < treino_fim < val_fim < teste_fim <= fim")
        if s.embargo_meses < self.alvo.escala_spi_meses - 1:
            raise ValueError("split.embargo_meses menor que escala_spi_meses - 1 (janela do SPI se sobrepõe)")
        if (_idx(s.val_fim) - _idx(s.treino_fim) <= s.embargo_meses
                or _idx(s.teste_fim) - _idx(s.val_fim) <= s.embargo_meses):
            raise ValueError("o embargo consome todo o bloco de validação ou de teste")
        anos = (_idx(s.treino_fim) - _idx(j.inicio) + 1) // 12
        if anos < self.spi.min_anos_treino:
            raise ValueError(f"treino com {anos} anos, abaixo de spi.min_anos_treino={self.spi.min_anos_treino}")
        return self

    @property
    def raiz(self) -> Path:
        if self._raiz is None:
            raise ConfigError("Config não veio de load_config(); raiz desconhecida")
        return self._raiz

    def caminho(self, nome: str) -> Path:
        """Caminho absoluto de uma pasta de `caminhos` (ex.: 'raw')."""
        if nome not in Caminhos.model_fields:
            raise ConfigError(f"caminho desconhecido: {nome}")
        return self.raiz / getattr(self.caminhos, nome)

    def pendencias(self) -> list[str]:
        """Chaves (com pontos) cujo valor ainda contém VERIFICAR."""
        return list(_percorrer(self))


def exigir_resolvido(cfg: Config, *chaves: str) -> None:
    """Gate por módulo: falha se alguma das chaves ainda estiver como VERIFICAR."""
    abertas = []
    for chave in chaves:
        atual = cfg
        for parte in chave.split("."):
            if not isinstance(atual, BaseModel) or parte not in type(atual).model_fields:
                raise ConfigError(f"chave inexistente: {chave}")
            atual = getattr(atual, parte)
        if _pendente(atual):
            abertas.append(chave)
    if abertas:
        raise PendenciaError(f"pendências {PENDENTE} em: " + ", ".join(abertas))


_MSG = {"extra_forbidden": "chave não reconhecida (erro de digitação?)",
        "missing": "chave obrigatória ausente"}


def _formatar(erro: ValidationError) -> str:
    linhas = ["params.yaml inválido:"]
    for e in erro.errors():
        onde = ".".join(str(x) for x in e["loc"]) or "(geral)"
        msg = _MSG.get(e["type"], e["msg"].removeprefix("Value error, "))
        linhas.append(f"  - {onde}: {msg}")
    return "\n".join(linhas)


def load_config(caminho: str | Path = "params.yaml") -> Config:
    """Lê (UTF-8, tolera BOM), valida e devolve a Config imutável."""
    p = Path(caminho)
    if not p.is_file():
        raise FileNotFoundError(f"params.yaml não encontrado: {p.resolve()}")
    try:
        texto = p.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        raise ConfigError("params.yaml fora de UTF-8: salve o arquivo como UTF-8") from e
    try:
        dados = yaml.safe_load(texto)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML inválido: {e}") from e
    if not isinstance(dados, dict):
        raise ConfigError("params.yaml vazio ou sem mapeamento no topo")
    try:
        cfg = Config.model_validate(dados)
    except ValidationError as e:
        raise ConfigError(_formatar(e)) from e
    cfg._raiz = p.resolve().parent
    return cfg
