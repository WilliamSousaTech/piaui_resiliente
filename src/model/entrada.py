"""Leitura verificada das entradas do modelo (Item 9). Não calcula SPI, não treina e não grava nada.

Só abre o que o manifest lista e só depois de conferir o SHA-256 de todos os arquivos, inclusive o
`params.yaml` (`manifest.verificar_manifest`). A conferência lê o `params.yaml` do disco: mudar o
`teste_liberado` na memória (para a avaliação final) não altera o `dataset_sha256`.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data import chirps_raw, manifest as mf, quality
from src.data.config import Config
from src.model import features as fe

RAW_PREFIXO = "raw_chirps/"


class EntradaError(RuntimeError):
    """O manifest não traz o que o modelo precisa."""


@dataclass(frozen=True)
class Insumos:
    demo: pd.DataFrame             # demo_observado: SPI e flags
    serie_imputada: pd.DataFrame   # quality.avaliar_qualidade(raw).serie_imputada
    climatologia: pd.DataFrame     # climatologia_precip (só treino)
    municipios: pd.DataFrame       # qualidade_municipio
    params_gama: pd.DataFrame      # spi_gamma_params (ajustados só no treino; usados pelo baseline parcial)
    dataset_sha256: str


def carregar_insumos(cfg: Config) -> Insumos:
    """Confere o manifest inteiro e lê só os arquivos que ele lista."""
    m = mf.verificar_manifest(cfg)
    lotes = sorted(n for n in m["arquivos"] if n.startswith(RAW_PREFIXO))
    if not lotes:
        raise EntradaError("o manifest não lista nenhum lote do RAW")
    for nome in ("demo_observado", "climatologia_precip", "qualidade_municipio", "spi_gamma_params"):
        if nome not in m["arquivos"]:
            raise EntradaError(f"o manifest não lista {nome}")
    ler = lambda nome: pd.read_parquet(mf.caminho_verificado(cfg, nome), engine="pyarrow")   # noqa: E731
    raw = pd.concat([chirps_raw.ler_lote(mf.caminho_verificado(cfg, n)) for n in lotes], ignore_index=True)
    q = quality.avaliar_qualidade(raw, cfg)
    return Insumos(demo=ler("demo_observado"), serie_imputada=q.serie_imputada,
                   climatologia=ler("climatologia_precip"), municipios=ler("qualidade_municipio"),
                   params_gama=ler("spi_gamma_params"), dataset_sha256=m["dataset_sha256"])


def montar_conjunto(cfg: Config) -> tuple[fe.Conjunto, str]:
    """Features e alvos de todos os municípios utilizáveis, mais o `dataset_sha256` de onde vieram."""
    ins = carregar_insumos(cfg)
    return fe.construir(ins.demo, ins.serie_imputada, ins.climatologia, ins.municipios, cfg), ins.dataset_sha256
