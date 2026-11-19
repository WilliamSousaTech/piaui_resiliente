"""Testes de src/model/entrada.py (Item 9). Projeto de mentira em tmp_path; nada de rede."""
import hashlib

import numpy as np
import pytest

from dados_sinteticos import A, B, C, raw_sintetico
from src.data import manifest as mf
from src.data.config import load_config
from src.model import entrada as en
from src.model import features as fe
from test_build_site import montar_proj


@pytest.fixture
def proj(tmp_path):
    return montar_proj(tmp_path, raw_sintetico())


def hashes(raiz):
    return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(raiz.rglob("*")) if p.is_file()}


def test_conjunto_do_projeto_confere_com_o_manifest(proj):
    cfg, raiz = proj
    antes = hashes(raiz)
    c, sha = en.montar_conjunto(cfg)
    assert sha == mf.verificar_manifest(cfg)["dataset_sha256"] and len(sha) == 64
    assert set(c.meta["cod_ibge"]) == {A, B, C} and len(c) == 3 * (549 - 12 + 1)
    assert c.X.dtype == np.float32 and c.Y.dtype == np.float32
    assert not c.M[(c.meta["split"] == "test").to_numpy()].any()          # teste congelado também aqui
    assert hashes(raiz) == antes                                           # só leitura: nenhum arquivo mudou
    ins = en.carregar_insumos(cfg)
    assert {"cod_ibge", "mes_calendario", "escala_meses", "q", "shape", "scale", "valido"} <= set(ins.params_gama.columns)
    assert len(ins.params_gama) == 3 * 12 * 2 and ins.dataset_sha256 == sha


def test_mesmo_resultado_que_a_construcao_direta(proj):
    cfg, _ = proj
    ins = en.carregar_insumos(cfg)
    direto = fe.construir(ins.demo, ins.serie_imputada, ins.climatologia, ins.municipios, cfg)
    c, _ = en.montar_conjunto(cfg)
    np.testing.assert_array_equal(c.X, direto.X)
    np.testing.assert_array_equal(c.Y, direto.Y)


@pytest.mark.parametrize("arquivo", ["data/curated/demo_observado.parquet",
                                     "data/raw/chirps/chirps_mensal_198101_200012.parquet"])
def test_arquivo_adulterado_derruba_a_leitura(proj, arquivo):
    cfg, raiz = proj
    alvo = raiz / arquivo
    assert alvo.is_file()
    alvo.write_bytes(alvo.read_bytes() + b"\0")
    with pytest.raises(mf.ManifestIntegridadeError):
        en.montar_conjunto(cfg)


def test_params_alterado_no_disco_derruba_a_leitura(proj):
    cfg, raiz = proj
    p = raiz / "params.yaml"
    p.write_text(p.read_text(encoding="utf-8") + "\n# mudou\n", encoding="utf-8")
    with pytest.raises(mf.ManifestIntegridadeError):
        en.montar_conjunto(load_config(p))


def test_liberar_o_teste_so_na_memoria_nao_muda_o_hash(proj):
    cfg, raiz = proj
    c0, sha0 = en.montar_conjunto(cfg)
    cfg2 = cfg.model_copy(update={"split": cfg.split.model_copy(update={"teste_liberado": True})})
    c1, sha1 = en.montar_conjunto(cfg2)
    assert sha0 == sha1 and "teste_liberado: false" in (raiz / "params.yaml").read_text(encoding="utf-8")
    teste = (c1.meta["split"] == "test").to_numpy()
    assert c1.M[teste].any() and not c0.M[teste].any()


def test_manifest_sem_raw_ou_sem_tabela_e_recusado(proj, monkeypatch):
    cfg, _ = proj
    real = mf.verificar_manifest(cfg)
    sem_raw = {**real, "arquivos": {k: v for k, v in real["arquivos"].items() if not k.startswith("raw_chirps/")}}
    monkeypatch.setattr(mf, "verificar_manifest", lambda c: sem_raw)
    with pytest.raises(en.EntradaError, match="RAW"):
        en.carregar_insumos(cfg)
    sem_demo = {**real, "arquivos": {k: v for k, v in real["arquivos"].items() if k != "demo_observado"}}
    monkeypatch.setattr(mf, "verificar_manifest", lambda c: sem_demo)
    with pytest.raises(en.EntradaError, match="demo_observado"):
        en.carregar_insumos(cfg)
