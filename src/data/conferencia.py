"""Conferência indicativa do mês previsto: leitura e validação do arquivo gerado por `conferencia_outubro.py`.

O arquivo `data/reference/conferencia_outubro.json` traz a chuva do mês previsto vinda de OUTRA fonte (Open-Meteo,
reanálise) e a posição dessa chuva no histórico do CHIRPS. Nada é calculado aqui nem no site: este módulo só confere
que o arquivo é coerente (contas que fecham, fonte e licença presentes, mesmo dataset e mesmos municípios da previsão)
para que o site possa mostrá-lo com fonte, data e aviso. Arquivo incoerente => a conferência não aparece no site.

Sem rede, sem ML, sem framework web.
"""
from __future__ import annotations

import json
import re
import statistics
from datetime import date
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ARQUIVO_PADRAO = Path("data") / "reference" / "conferencia_outubro.json"
_MES = r"^\d{4}-(0[1-9]|1[0-2])$"
_SHA = r"^[0-9a-f]{64}$"

Real = Annotated[float, Field(allow_inf_nan=False)]
Mm = Annotated[float, Field(ge=0, le=5000, allow_inf_nan=False)]
Pct = Annotated[float, Field(ge=0, le=100, allow_inf_nan=False)]


class ConferenciaError(ValueError):
    """Arquivo ausente, ilegível, incoerente, ou de outro conjunto de dados que o da previsão."""


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Fonte(_Base):
    nome: str = Field(min_length=3)
    url: str
    atribuicao: str = Field(min_length=10)
    licenca: str = Field(pattern=r"^CC BY 4\.0$")
    variavel: str = Field(pattern=r"^precipitation_sum$")
    periodo: tuple[date, date]
    ponto: str = Field(min_length=3)
    valores_diarios_invalidos: int = Field(ge=0)

    @field_validator("url")
    @classmethod
    def _https(cls, v: str) -> str:
        if not v.startswith("https://") or " " in v:
            raise ValueError(f"o link precisa começar com https:// e não ter espaço: {v!r}")
        return v

    @field_validator("atribuicao")
    @classmethod
    def _atribuicao(cls, v: str) -> str:
        if "Open-Meteo" not in v:                       # a licença CC BY 4.0 exige a atribuição
            raise ValueError("a atribuição precisa citar a Open-Meteo")
        return v

    @model_validator(mode="after")
    def _ordem(self) -> "Fonte":
        if self.periodo[0] > self.periodo[1]:
            raise ValueError("período com início depois do fim")
        return self


class Historico(_Base):
    fonte: str = Field(min_length=3)
    periodo: str = Field(pattern=r"^\d{4}-\d{2} a \d{4}-\d{2}$")
    normal: str = Field(min_length=3)
    minimo_normal_mm: float = Field(gt=0, allow_inf_nan=False)
    minimo_anos_percentil: int = Field(ge=1)
    limiar_seco_percentil: float = Field(gt=0, lt=100, allow_inf_nan=False)


class ParFontes(_Base):
    n: int = Field(ge=0)
    vies_medio_mm: Optional[Real] = None
    erro_medio_abs_mm: Optional[Mm] = None
    correlacao: Optional[Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]] = None
    open_meteo_medio_mm: Optional[Mm] = None
    chirps_medio_mm: Optional[Mm] = None

    @model_validator(mode="after")
    def _vazio(self) -> "ParFontes":
        tem = [self.vies_medio_mm, self.erro_medio_abs_mm, self.open_meteo_medio_mm, self.chirps_medio_mm]
        if (self.n == 0) != all(x is None for x in tem):
            raise ValueError("comparação com n = 0 não pode ter números, e com n > 0 precisa tê-los")
        if self.n == 0 and self.correlacao is not None:
            raise ValueError("correlação sem pares")
        return self


class Grupo(_Base):
    n: int = Field(ge=0)
    n_com_dado: int = Field(ge=0)
    n_secos: int = Field(ge=0)
    fracao_secos: Optional[Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]] = None
    mediana_percentil: Optional[Pct] = None
    mediana_anomalia_mm: Optional[Real] = None
    n_abaixo_da_normal: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordem(self) -> "Grupo":
        if not (self.n_secos <= self.n_com_dado <= self.n) or self.n_abaixo_da_normal > self.n:
            raise ValueError("contagens do grupo fora de ordem")
        return self


class Grupos(_Base):
    com_sinal: Grupo
    sem_sinal: Grupo


class Registro(_Base):
    cod_ibge: str = Field(pattern=r"^\d{7}$")
    chuva_mm: dict[str, Optional[Mm]]
    dias_validos_alvo: Optional[int] = Field(default=None, ge=0, le=31)
    chirps_mm: dict[str, Optional[Mm]]
    normal_mm: Optional[Mm] = None
    anomalia_mm: Optional[Real] = None
    pct_normal: Optional[Annotated[float, Field(ge=0, le=100000, allow_inf_nan=False)]] = None
    percentil: Optional[Pct] = None
    completo: bool
    sinal_modelo: bool

    @field_validator("chuva_mm", "chirps_mm")
    @classmethod
    def _chaves_mes(cls, d: dict) -> dict:
        ruins = [k for k in d if not re.fullmatch(_MES, k)]
        if ruins:
            raise ValueError(f"chave que não é AAAA-MM: {ruins}")
        return d


class Conferencia(_Base):
    versao: int = Field(ge=1, le=1)
    aviso: str = Field(min_length=20)
    avisos: tuple[str, ...] = Field(min_length=1)
    gerado_em: date
    mes_alvo: str = Field(pattern=_MES)
    previsao_dataset_sha256: str = Field(pattern=_SHA)
    fonte_conferencia: Fonte
    base_historica: Historico
    comparacao_fontes: dict[str, ParFontes]
    grupos: Grupos
    n_municipios: int = Field(gt=0)
    n_incompletos: int = Field(ge=0)
    municipios: tuple[Registro, ...]

    @model_validator(mode="after")
    def _coerencia(self) -> "Conferencia":
        regs, alvo = self.municipios, self.mes_alvo
        cods = [r.cod_ibge for r in regs]
        if len(regs) != self.n_municipios or len(set(cods)) != len(cods):
            raise ValueError("n_municipios não bate com a lista, ou há município repetido")
        if self.fonte_conferencia.periodo[1].strftime("%Y-%m") != alvo or self.gerado_em < self.fonte_conferencia.periodo[1]:
            raise ValueError("o período buscado precisa terminar no mês-alvo e ser anterior à data de geração")
        if alvo in self.comparacao_fontes:
            raise ValueError("a comparação entre fontes é só dos meses em que as duas existem, nunca do mês-alvo")
        if any(not re.fullmatch(_MES, k) for k in self.comparacao_fontes):
            raise ValueError("chave da comparação entre fontes que não é AAAA-MM")
        for r in regs:
            if alvo not in r.chuva_mm:
                raise ValueError(f"{r.cod_ibge}: sem chuva do mês-alvo {alvo}")
            if r.completo != (r.chuva_mm[alvo] is not None):
                raise ValueError(f"{r.cod_ibge}: `completo` não bate com a chuva do mês-alvo")
            if not r.completo and (r.percentil, r.anomalia_mm, r.pct_normal) != (None, None, None):
                raise ValueError(f"{r.cod_ibge}: mês incompleto não pode ter percentil, desvio nem % da normal")
            c, n = r.chuva_mm[alvo], r.normal_mm
            if r.anomalia_mm is not None and (c is None or n is None or abs(r.anomalia_mm - (c - n)) > 0.11):
                raise ValueError(f"{r.cod_ibge}: o desvio não bate com chuva − normal")
            if r.pct_normal is not None and (c is None or not n or n < self.base_historica.minimo_normal_mm
                                              or abs(r.pct_normal - 100 * c / n) > 1 + 0.011 * r.pct_normal):
                raise ValueError(f"{r.cod_ibge}: a % da normal não bate com chuva ÷ normal")
        if self.n_incompletos != sum(1 for r in regs if not r.completo):
            raise ValueError("n_incompletos não bate com a lista")
        for sinal, g in ((True, self.grupos.com_sinal), (False, self.grupos.sem_sinal)):
            self._conferir_grupo([r for r in regs if r.sinal_modelo == sinal], g, "com_sinal" if sinal else "sem_sinal")
        return self

    def _conferir_grupo(self, regs: list[Registro], g: Grupo, nome: str) -> None:
        perc = [r.percentil for r in regs if r.percentil is not None]
        anom = [r.anomalia_mm for r in regs if r.anomalia_mm is not None]
        limiar = self.base_historica.limiar_seco_percentil
        secos = sum(1 for p in perc if p <= limiar)
        esperado = (len(regs), len(perc), secos, sum(1 for a in anom if a < 0))
        if (g.n, g.n_com_dado, g.n_secos, g.n_abaixo_da_normal) != esperado:
            raise ValueError(f"grupo {nome}: as contagens não batem com a lista de municípios")
        if (g.fracao_secos is None) != (not perc) or (perc and abs(g.fracao_secos - secos / len(perc)) > 0.0006):
            raise ValueError(f"grupo {nome}: fração de municípios secos não bate")
        for valor, lista, rotulo in ((g.mediana_percentil, perc, "percentil"), (g.mediana_anomalia_mm, anom, "desvio")):
            if (valor is None) != (not lista) or (lista and abs(valor - statistics.median(lista)) > 0.051):
                raise ValueError(f"grupo {nome}: mediana do {rotulo} não bate")

    @property
    def com_sinal(self) -> tuple[Registro, ...]:
        return tuple(r for r in self.municipios if r.sinal_modelo)


def carregar(caminho: Path | str = ARQUIVO_PADRAO) -> Conferencia:
    p = Path(caminho)
    try:
        bruto = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise ConferenciaError(f"arquivo da conferência não encontrado: {p}") from e
    except (OSError, ValueError) as e:
        raise ConferenciaError(f"não consegui ler {p}: {e}") from e
    try:
        return Conferencia.model_validate(bruto)
    except ValueError as e:                    # pydantic.ValidationError herda de ValueError
        raise ConferenciaError(f"{p} inválido: {e}") from e


def conferir_previsao(conf: Conferencia, dataset_sha256: str, alvo: str, cods_com_sinal: set[str], cods) -> None:
    """A conferência só vale para a previsão do site se for do mesmo dataset, do mesmo mês-alvo e dos mesmos municípios,
    e se o grupo "com sinal" for exatamente o que a previsão marcou."""
    if conf.previsao_dataset_sha256 != dataset_sha256:
        raise ConferenciaError(f"a conferência é do dataset {conf.previsao_dataset_sha256[:12]}…, e a previsão do "
                               f"{dataset_sha256[:12]}…; rode `python -m src.data.conferencia_outubro` de novo")
    if conf.mes_alvo != alvo:
        raise ConferenciaError(f"a conferência é de {conf.mes_alvo} e a previsão, de {alvo}")
    meus, do_site = {r.cod_ibge for r in conf.municipios}, {str(c) for c in cods}
    if meus != do_site:
        raise ConferenciaError(f"a conferência e o site não têm os mesmos municípios (faltam {sorted(do_site - meus)[:5]}, "
                               f"sobram {sorted(meus - do_site)[:5]})")
    marcados = {r.cod_ibge for r in conf.com_sinal}
    if marcados != set(cods_com_sinal):
        raise ConferenciaError("os municípios com sinal na conferência não são os que a previsão marcou; "
                               "rode `python -m src.data.conferencia_outubro` de novo")


def para_pacote(conf: Conferencia) -> dict:
    """Dicionário JSON puro para o pacote do site (datas em ISO)."""
    return json.loads(conf.model_dump_json())
