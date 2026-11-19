"""Fixtures compartilhadas pelos testes do site: insumos sintéticos e o pacote de dados montado com eles."""
import base64

import numpy as np
import pytest

from dados_sinteticos import PARAMS_REAL, VERIFICADO, malha_tres, raw_sintetico
from src.data import build_demo as bd
from src.data.config import load_config
from src.web import build_site as bs


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def raw_base():
    return raw_sintetico()


@pytest.fixture(scope="module")
def insumos(cfg, raw_base):
    r = bd.montar_demo(raw_base, cfg)
    manifest = {"fontes": {"chirps": {"gee_id": "UCSB-CHG/CHIRPS/DAILY"}}, "projeto": {"versao_params": "0.3.1"},
                "escopo": {"malha_municipal_ano": 2022}, "janela": {"inicio": "1981-01", "fim": "2026-09"},
                "split": {"treino_fim": "2015-12", "teste_fim": "2025-12"},
                "extras": {"demo": {"spi_calibrado_ate": "2015-12"}},
                "dataset_sha256": "ab" * 32, "gerado_em_utc": "2026-11-19T12:30:45Z", "arquivos": {"x": {}, "y": {}}}
    return bs.Insumos(demo=r.demo_observado, qualidade=r.qualidade_municipio, raw=raw_base, malha=malha_tres(),
                      manifest=manifest, extraido_em=("2026-11-18T23:10:00Z", "2026-11-19T00:40:00Z"))


@pytest.fixture(scope="module")
def payload(insumos, cfg):
    return bs.montar_payload(insumos, cfg, verificado_em=VERIFICADO)
