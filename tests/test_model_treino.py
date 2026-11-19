"""Testes de src/model/treino.py (Item 9). Dados sintéticos só aqui; o alvo plantado é uma função conhecida
das features, para provar que o laço forward -> perda -> backward -> otimizador aprende de verdade."""
import dataclasses
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import torch

from dados_sinteticos import PARAMS_REAL, raw_sintetico
from src.data import build_demo as bd
from src.data import manifest as mf
from src.data import quality
from src.data.config import load_config
from src.model import features as fe
from src.model import metricas as mt
from src.model import rede as rd
from src.model import treino as tn
from test_build_site import montar_proj

HP = tn.Hiperparametros(ocultas=(16,), dropout=0.0, lr=1e-2, lote=128, epocas_max=40, paciencia=40)


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def dados(cfg):
    raw = raw_sintetico()
    r = bd.montar_demo(raw, cfg)
    q = quality.avaliar_qualidade(raw, cfg)
    c = fe.construir(r.demo_observado, q.serie_imputada, r.climatologia_precip, r.qualidade_municipio, cfg)
    tr, va = fe.selecionar(c, "train", cfg, so_com_rotulo=True), fe.selecionar(c, "val", cfg, so_com_rotulo=True)
    p = fe.Padronizador.ajustar(tr)
    # alvo plantado: combinação linear mais um degrau suave das features padronizadas, com desvio 1
    Z = p.transformar(c)
    w = np.linspace(-1.0, 1.0, Z.shape[1])
    y = Z @ w + 1.5 * np.tanh(Z[:, 0])
    y = (y - y[(c.meta["split"] == "train").to_numpy()].mean()) / y[(c.meta["split"] == "train").to_numpy()].std()
    plantado = dataclasses.replace(c, Y=np.where(c.M, np.column_stack([y, 0.5 * y]), np.nan).astype("float32"))
    tr2, va2 = fe.selecionar(plantado, "train", cfg, so_com_rotulo=True), fe.selecionar(plantado, "val", cfg, so_com_rotulo=True)
    return {"c": c, "tr": tr2, "va": va2, "p": p}


def hist(res, chave):
    return [h[chave] for h in res.historico]


# ------------------------------------------------------------------ aprende de verdade
def test_treina_do_zero_e_aprende_o_alvo_plantado(dados):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    res = tn.treinar(tr, va, p, seed=42, hp=HP)
    assert hist(res, "perda_treino")[-1] < 0.5 * hist(res, "perda_treino")[0]
    prev = tn.RedePreditor(res.rede, p).prever(va)
    e = mt.erros(va.Y[:, 0], prev[:, 0], va.M[:, 0])
    zero = mt.erros(va.Y[:, 0], np.zeros(len(va), dtype="float32"), va.M[:, 0])
    assert e["mae"] < 0.5 * zero["mae"] and res.melhor_epoca >= 5
    assert not res.rede.training and res.alvos == tr.alvos and res.colunas == tr.colunas


def test_deterministico_com_a_mesma_semente_e_diferente_com_outra(dados):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    hp = dataclasses.replace(HP, epocas_max=6, paciencia=6)
    a, b, c = (tn.treinar(tr, va, p, seed=s, hp=hp) for s in (7, 7, 8))
    for (k, va_), vb in zip(a.rede.state_dict().items(), b.rede.state_dict().values()):
        assert torch.equal(va_, vb), k
    assert a.historico == b.historico
    assert a.historico != c.historico


def test_pesos_iniciais_vem_da_semente_informada(dados):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    hp = tn.Hiperparametros(ocultas=(8,), dropout=0.0, lr=1e-12, weight_decay=0.0, epocas_max=1, paciencia=1)
    res = tn.treinar(tr, va, p, seed=123, hp=hp)                              # lr minúsculo: pesos ~ iniciais
    esperado = rd.criar(rd.Arquitetura(len(tr.colunas), len(tr.alvos), (8,), 0.0), 123)
    for k, v in esperado.state_dict().items():
        assert torch.allclose(res.rede.state_dict()[k], v, atol=1e-6), k
    outra = rd.criar(rd.Arquitetura(len(tr.colunas), len(tr.alvos), (8,), 0.0), 124)
    assert not torch.allclose(res.rede.state_dict()["tronco.0.weight"], outra.state_dict()["tronco.0.weight"], atol=1e-3)


def test_validacao_so_observa_nunca_gera_gradiente(dados):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    hp = dataclasses.replace(HP, epocas_max=5, paciencia=100)               # sem parada antecipada: 5 épocas completas
    lixo = dataclasses.replace(va, Y=np.where(va.M, 50.0, np.nan).astype("float32"))
    a = tn.treinar(tr, va, p, seed=3, hp=hp)
    b = tn.treinar(tr, lixo, p, seed=3, hp=hp)
    assert hist(a, "perda_treino") == hist(b, "perda_treino")                # o treino não enxergou a validação
    assert hist(a, "perda_val") != hist(b, "perda_val")                      # ela só mede


@pytest.mark.parametrize("dropout", [0.0, 0.3])
def test_parada_antecipada_restaura_a_melhor_epoca(dados, dropout):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    pequeno = tr.fatiar(np.arange(len(tr)) < 40)                              # pouquíssimo treino: sobreajuste garantido
    hp = tn.Hiperparametros(ocultas=(64, 64), dropout=dropout, lr=2e-2, lote=40, epocas_max=200, paciencia=5)
    res = tn.treinar(pequeno, va, p, seed=1, hp=hp)
    pv = hist(res, "perda_val")
    assert res.melhor_epoca == int(np.argmin(pv)) + 1
    assert len(pv) == res.melhor_epoca + hp.paciencia and len(pv) < hp.epocas_max
    P = torch.from_numpy(tn.RedePreditor(res.rede, p).prever(va))
    restaurada = float(rd.perda_mascarada(P, torch.from_numpy(va.Y), torch.from_numpy(va.M)))
    assert restaurada == pytest.approx(min(pv), rel=1e-4)                     # pesos da melhor época, não da última
    assert pv[-1] > min(pv)


def test_callback_recebe_cada_epoca_na_ordem(dados):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    vistos = []
    res = tn.treinar(tr, va, p, seed=1, hp=dataclasses.replace(HP, epocas_max=4, paciencia=4), ao_fim_da_epoca=vistos.append)
    assert [v["epoca"] for v in vistos] == [1, 2, 3, 4] and tuple(vistos) == res.historico


# ------------------------------------------------------------------ guardas
def test_divergencia_vira_erro_claro_e_threads_sao_restauradas(dados):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    antes = torch.get_num_threads()
    with pytest.raises(tn.TreinoError, match="perda não finita na época 1"):
        tn.treinar(tr, va, p, seed=1, hp=dataclasses.replace(HP, lr=1e30))
    assert torch.get_num_threads() == antes
    tn.treinar(tr, va, p, seed=1, hp=dataclasses.replace(HP, epocas_max=2, n_threads=2))
    assert torch.get_num_threads() == antes


@pytest.mark.parametrize("kw", [dict(lr=0.0), dict(lr=float("nan")), dict(lr=float("inf")), dict(weight_decay=-1.0),
                                dict(lote=0), dict(epocas_max=True), dict(paciencia=0), dict(n_threads=1.5),
                                dict(ocultas=()), dict(dropout=1.0)])
def test_hiperparametros_invalidos(kw):
    with pytest.raises((tn.TreinoError, rd.RedeError)):
        tn.Hiperparametros(**kw)


def test_blocos_colunas_e_rotulos_sao_conferidos(dados, cfg):
    tr, va, p, c = dados["tr"], dados["va"], dados["p"], dados["c"]
    with pytest.raises(tn.TreinoError, match="'train'"):
        tn.treinar(va, va, p, seed=1, hp=HP)
    with pytest.raises(tn.TreinoError, match="'val'"):
        tn.treinar(tr, tr, p, seed=1, hp=HP)
    with pytest.raises(tn.TreinoError, match="'train'"):
        tn.treinar(c, va, p, seed=1, hp=HP)                                   # conjunto inteiro: mistura de blocos
    sem = dataclasses.replace(va, M=np.column_stack([va.M[:, 0], np.zeros(len(va), dtype=bool)]))
    with pytest.raises(tn.TreinoError, match="sem rótulo"):
        tn.treinar(tr, sem, p, seed=1, hp=HP)
    outras = dataclasses.replace(va, colunas=tuple(reversed(va.colunas)))
    with pytest.raises(tn.TreinoError, match="diferentes"):
        tn.treinar(tr, outras, p, seed=1, hp=HP)


# ------------------------------------------------------------------ preditor
def test_preditor_formato_modo_eval_e_guardas(dados):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    res = tn.treinar(tr, va, p, seed=2, hp=dataclasses.replace(HP, epocas_max=3, dropout=0.5))
    pr = tn.RedePreditor(res.rede, p)
    res.rede.train()
    P1, P2 = pr.prever(va), pr.prever(va, lote=17)
    assert P1.shape == (len(va), 2) and P1.dtype == np.float32 and np.isfinite(P1).all()
    np.testing.assert_allclose(P1, P2, atol=1e-5)                              # sem dropout na previsão, qualquer lote
    assert res.rede.training                                                   # devolve o modo em que encontrou
    assert pr.prever(va.fatiar(np.zeros(len(va), dtype=bool))).shape == (0, 2)
    with pytest.raises(tn.TreinoError, match="colunas"):
        pr.prever(dataclasses.replace(va, colunas=tuple(reversed(va.colunas))))
    with torch.no_grad():
        res.rede.cabecas[0].bias.fill_(float("nan"))
    with pytest.raises(tn.TreinoError, match="não finita"):
        pr.prever(va)


# ------------------------------------------------------------------ salvar e carregar
def test_ida_e_volta_do_resultado(dados, tmp_path):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    res = tn.treinar(tr, va, p, seed=5, hp=dataclasses.replace(HP, epocas_max=3))
    sha = tn.salvar_resultado(res, tmp_path, "ab" * 32)
    pr, meta = tn.carregar_preditor(tmp_path)
    np.testing.assert_array_equal(pr.prever(va), tn.RedePreditor(res.rede, p).prever(va))
    assert meta["dataset_sha256"] == "ab" * 32 and meta["seed"] == 5 and meta["melhor_epoca"] == res.melhor_epoca
    assert meta["alvos"] == list(res.alvos) and meta["historico"] == list(res.historico) and "MVP" in meta["aviso"]
    assert json.loads((tmp_path / "rede.json").read_text(encoding="utf-8"))["pesos_sha256"] == sha


def test_padronizador_que_nao_combina_com_a_rede_e_recusado(dados, tmp_path):
    tr, va, p = dados["tr"], dados["va"], dados["p"]
    res = tn.treinar(tr, va, p, seed=5, hp=dataclasses.replace(HP, epocas_max=2))
    tn.salvar_resultado(res, tmp_path, "ab" * 32)
    j = tmp_path / "rede.json"
    doc = json.loads(j.read_text(encoding="utf-8"))
    doc["meta"]["colunas"] = list(reversed(doc["meta"]["colunas"]))
    j.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(tn.TreinoError, match="não combina"):
        tn.carregar_preditor(tmp_path)
    del doc["meta"]["padronizador"]
    j.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(tn.TreinoError, match="inválido"):
        tn.carregar_preditor(tmp_path)


# ------------------------------------------------------------------ linha de comando, ponta a ponta
@pytest.fixture
def proj(tmp_path):
    return montar_proj(tmp_path, raw_sintetico())


def hashes(raiz, excluir=()):
    return {str(p.relative_to(raiz)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(raiz.rglob("*")) if p.is_file() and not any(x in p.parts for x in excluir)}


def test_cli_treina_grava_e_nao_mexe_nos_dados(proj):
    cfg, raiz = proj
    antes = hashes(raiz)
    linhas = []
    cod = tn.main(["--params", str(raiz / "params.yaml"), "--epocas", "3", "--ocultas", "8"], saida=linhas.append)
    txt = "\n".join(linhas)
    assert cod == 0 and "AVISO: MVP experimental" in txt and "época   3" in txt and "Próximo passo" in txt
    assert "teste congelado não foi lido" in txt
    pasta = raiz / "artifacts" / "modelo"
    assert (pasta / "rede.pt").is_file() and (pasta / "rede.json").is_file()
    meta = json.loads((pasta / "rede.json").read_text(encoding="utf-8"))["meta"]
    assert meta["dataset_sha256"] == mf.verificar_manifest(cfg)["dataset_sha256"] and meta["seed"] == cfg.projeto.seed
    assert hashes(raiz, excluir=("modelo",)) == antes                          # só leitura dos dados: nada mudou


def test_cli_recusa_teste_liberado_arquivo_adulterado_e_argumento_ruim(proj):
    cfg, raiz = proj
    linhas = []
    liberado = cfg.model_copy(update={"split": cfg.split.model_copy(update={"teste_liberado": True})})
    assert tn.main(["--params", str(raiz / "params.yaml")], saida=linhas.append, carregar_cfg=lambda _: liberado) == 1
    assert "ERRO" in linhas[-1] and "teste_liberado" in linhas[-1]
    assert not (raiz / "artifacts" / "modelo").exists()
    alvo = raiz / "data" / "curated" / "demo_observado.parquet"
    alvo.write_bytes(alvo.read_bytes() + b"\0")
    linhas.clear()
    assert tn.main(["--params", str(raiz / "params.yaml"), "--epocas", "2"], saida=linhas.append) == 1
    assert "ERRO" in linhas[-1] and not (raiz / "artifacts" / "modelo").exists()
    assert tn.main(["--params", str(raiz / "nao_existe.yaml")], saida=linhas.append) == 1
    with pytest.raises(SystemExit):
        tn.main(["--ocultas", "a,b"])
    linhas.clear()
    assert tn.main(["--params", str(raiz / "params.yaml"), "--lr", "0"], saida=linhas.append) == 1
    assert "lr" in linhas[-1]
