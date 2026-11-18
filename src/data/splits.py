"""Splits cronológicos por mês-alvo (Item 4). Sem I/O e sem rede.

Regras do contrato (seção 4): o split é definido por `mes_alvo`; nunca há shuffle; há embargo
entre os blocos; o teste só é lido com `split.teste_liberado: true`. Este módulo apenas rotula e
filtra. Quem ajusta scalers, parâmetros Gama ou qualquer estatística deve usar só as linhas `train`.

Blocos para um `mes_alvo` m, com embargo e (meses):
    train    : inicio <= m <= treino_fim
    embargo  : treino_fim < m <= treino_fim + e
    val      : treino_fim + e < m <= val_fim
    embargo  : val_fim < m <= val_fim + e
    test     : val_fim + e < m <= teste_fim

Embargo mínimo sem sobreposição das janelas do alvo: maior escala do SPI - 1 (SPI-3: 2; SPI-6: 5).
O início real do treino depende das features (Item 9) e de NaN; aqui `train` é todo mês-alvo
dentro da janela e até `treino_fim`, e o descarte de linhas sem histórico fica para o `features.py`.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.config import Config

BLOCOS = ("train", "val", "test", "embargo")   # ordem do contrato (seção 5.2)


class SplitError(ValueError):
    """Configuração ou entrada inconsistente com os splits."""


class TesteBloqueadoError(PermissionError):
    """Leitura do teste congelado sem `split.teste_liberado: true`."""


def _ym(p: pd.Period) -> int:
    return p.year * 12 + p.month


@dataclass(frozen=True)
class Cortes:
    inicio: pd.Period
    treino_fim: pd.Period
    val_ini: pd.Period
    val_fim: pd.Period
    teste_ini: pd.Period
    teste_fim: pd.Period
    embargo_meses: int

    def para_manifest(self) -> dict:
        """Cortes e embargo em JSON puro, para o manifest."""
        return {"inicio": str(self.inicio), "treino_fim": str(self.treino_fim),
                "val_ini": str(self.val_ini), "val_fim": str(self.val_fim),
                "teste_ini": str(self.teste_ini), "teste_fim": str(self.teste_fim),
                "embargo_meses": self.embargo_meses}


def cortes(cfg: Config, escalas: Sequence[int] | None = None) -> Cortes:
    """Cortes por mês-alvo. `escalas` = janelas do SPI de todas as cabeças (padrão: `spi.janelas_meses`)."""
    esc = tuple(escalas) if escalas is not None else tuple(cfg.spi.janelas_meses)
    if not esc or any(isinstance(k, bool) or not isinstance(k, int) or k < 1 for k in esc):
        raise SplitError(f"escalas do SPI inválidas: {esc!r} (use inteiros >= 1)")
    s, e = cfg.split, cfg.split.embargo_meses
    minimo = max(esc) - 1
    if e < minimo:
        raise SplitError(f"split.embargo_meses={e} < {minimo} (SPI-{max(esc)}): a janela do alvo de "
                         "validação se sobrepõe ao treino; aumente o embargo no params.yaml")
    return Cortes(inicio=cfg.janela.inicio, treino_fim=s.treino_fim, val_ini=s.treino_fim + e + 1,
                  val_fim=s.val_fim, teste_ini=s.val_fim + e + 1, teste_fim=s.teste_fim, embargo_meses=e)


def rotular(meses_alvo, cfg: Config, escalas: Sequence[int] | None = None) -> pd.Series:
    """Bloco de cada mês-alvo (Series categórica `split`, categorias = BLOCOS). Erro se fora da janela ou NaT."""
    c = cortes(cfg, escalas)
    s = meses_alvo if isinstance(meses_alvo, pd.Series) else pd.Series(meses_alvo)
    dt = s if isinstance(s.dtype, pd.PeriodDtype) else pd.to_datetime(s)
    if dt.isna().any():
        raise SplitError(f"mes_alvo ausente (NaT) em {int(dt.isna().sum())} linha(s)")
    ym = dt.dt.year.astype("int64") * 12 + dt.dt.month.astype("int64")
    fora = (ym < _ym(c.inicio)) | (ym > _ym(c.teste_fim))
    if fora.any():
        ex = [str(x)[:10] for x in s[fora].head(3)]
        raise SplitError(f"{int(fora.sum())} mes_alvo fora da janela [{c.inicio}, {c.teste_fim}], ex.: {ex}")
    e = c.embargo_meses
    cod = np.select([ym <= _ym(c.treino_fim), ym <= _ym(c.treino_fim) + e, ym <= _ym(c.val_fim),
                     ym <= _ym(c.val_fim) + e],
                    [BLOCOS.index("train"), BLOCOS.index("embargo"), BLOCOS.index("val"),
                     BLOCOS.index("embargo")], default=BLOCOS.index("test"))
    return pd.Series(pd.Categorical.from_codes(cod, categories=list(BLOCOS)), index=s.index, name="split")


def exigir_teste_liberado(cfg: Config) -> None:
    if not cfg.split.teste_liberado:
        raise TesteBloqueadoError("o teste congelado só pode ser lido com split.teste_liberado: true "
                                  "(avaliação final da Fase 2)")


def bloco(df: pd.DataFrame, nome: str, cfg: Config, *, col: str = "mes_alvo",
          escalas: Sequence[int] | None = None) -> pd.DataFrame:
    """Linhas de `df` do bloco `nome`. O bloco `test` recusa antes de qualquer cálculo se não liberado."""
    if nome not in BLOCOS:
        raise SplitError(f"bloco {nome!r} desconhecido; use um de {BLOCOS}")
    if nome == "test":
        exigir_teste_liberado(cfg)
    if col not in df.columns:
        raise SplitError(f"coluna {col!r} ausente")
    return df.loc[(rotular(df[col], cfg, escalas) == nome).to_numpy()].copy()
