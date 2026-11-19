"""Avaliação na validação (Item 9): núcleo puro com previsões plantadas e linha de comando ponta a ponta.

Só dados sintéticos (tests/dados_sinteticos.py); nenhum número aqui é medida do projeto."""
import dataclasses
import hashlib
import json
import math

import numpy as np
import pytest

from dados_sinteticos import PARAMS_REAL, raw_sintetico
from src.data import build_demo as bd
from src.data import manifest as mf
from src.data import quality
from src.data.config import load_config
from src.model import avaliar as av
from src.model import features as fe
from src.model import metricas as mt
from src.model import treino as tn
from test_build_site import montar_proj


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def conj(cfg):
    raw = raw_sintetico()
    r = bd.montar_demo(raw, cfg)
    q = quality.avaliar_qualidade(raw, cfg)
    return fe.construir(r.demo_observado, q.serie_imputada, r.climatologia_precip, r.qualidade_municipio, cfg)


@pytest.fixture(scope="module")
def va(conj, cfg):
    return fe.selecionar(conj, "val", cfg, so_com_rotulo=True)


def com_erro(va, erro):
    """Previsão = rótulo + erro (nas linhas sem rótulo: 0 + erro, só para ser finita)."""
    return (np.where(va.M, va.Y, 0.0) + erro).astype("float32")


def ruido(va, escala, seed):
    return com_erro(va, np.random.default_rng(seed).normal(0.0, escala, size=va.Y.shape))


def quase_uniforme(va):
    """Erros em [0, 1) bem espalhados (sequência de baixa discrepância): as duas metades têm a mesma distribuição."""
    u = np.modf(np.arange(len(va)) * 0.6180339887498949)[0]
    return np.column_stack([u, np.roll(u, 7)])


def prevs(va):
    return {"persistencia": ruido(va, 0.8, 1), "climatologia": ruido(va, 1.0, 2), "ridge": ruido(va, 0.5, 3),
            "rede": ruido(va, 0.2, 4)}


# ------------------------------------------------------------------ pontuação
def test_pontua_todos_nas_mesmas_linhas_e_a_rede_melhor_supera_com_intervalo(va):
    r = av.pontuar(va, prevs(va), n_boot=300, seed=1)
    assert list(r["modelos"]) == ["persistencia", "climatologia", "ridge", "rede"]          # ordem do relatório
    for alvo in va.alvos:
        ns = {n: r["modelos"][n][alvo]["todas"]["n"] for n in r["modelos"]}
        assert set(ns.values()) == {165}                                                    # mesmas linhas para todos
        maes = {n: r["modelos"][n][alvo]["todas"]["mae"] for n in r["modelos"]}
        assert min(maes, key=maes.get) == "rede" and max(maes, key=maes.get) == "climatologia"
        c = r["comparacao"][alvo]
        assert c["persistencia"]["contra"] == "persistencia" and c["melhor_baseline"]["contra"] == "ridge"
        assert c["persistencia"]["veredito"] == "superou" and c["persistencia"]["ic_sup"] < 0
        assert c["persistencia"]["n_grupos"] == 55
    assert r["rede_supera_persistencia_em_todas_as_cabecas"] and r["rede_supera_melhor_baseline_em_todas_as_cabecas"]


def test_ganho_so_numa_cabeca_nao_vale_para_todas_e_a_semente_muda_o_intervalo(va):
    misto = np.column_stack([ruido(va, 0.2, 4)[:, 0], ruido(va, 1.5, 5)[:, 1]])             # bom no SPI-3, ruim no SPI-6
    r = av.pontuar(va, prevs(va) | {"rede": misto}, n_boot=300, seed=1)
    assert r["comparacao"]["alvo_spi3_h1"]["persistencia"]["veredito"] == "superou"
    assert r["comparacao"]["alvo_spi6_h1"]["persistencia"]["veredito"] == "pior"
    assert not r["rede_supera_persistencia_em_todas_as_cabecas"] and not r["rede_supera_melhor_baseline_em_todas_as_cabecas"]
    a = av.pontuar(va, prevs(va), n_boot=300, seed=1)["comparacao"]["alvo_spi3_h1"]["persistencia"]
    b = av.pontuar(va, prevs(va), n_boot=300, seed=2)["comparacao"]["alvo_spi3_h1"]["persistencia"]
    assert a["dif"] == b["dif"] and a["ic_inf"] != b["ic_inf"]                              # a semente só mexe no intervalo


def test_predicao_perfeita_tem_erro_zero_e_acerta_toda_seca(va):
    p = prevs(va) | {"rede": com_erro(va, 0.0)}
    t = av.pontuar(va, p, n_boot=200, seed=1)["modelos"]["rede"]["alvo_spi3_h1"]["todas"]
    assert t["mae"] == t["rmse"] == t["vies"] == 0.0
    assert t["seca"]["fp"] == t["seca"]["fn"] == 0 and t["seca"]["tp"] == t["seca"]["secas_reais"]


def test_veredito_pior_e_indistinguivel_e_a_funcao_isolada(va):
    pior = av.pontuar(va, prevs(va) | {"rede": ruido(va, 1.5, 9)}, n_boot=300, seed=1)
    assert pior["comparacao"]["alvo_spi3_h1"]["persistencia"]["veredito"] == "pior"
    assert not pior["rede_supera_persistencia_em_todas_as_cabecas"]
    igual = prevs(va)
    igual["rede"] = igual["persistencia"].copy()
    emp = av.pontuar(va, igual, n_boot=300, seed=1)["comparacao"]["alvo_spi3_h1"]["persistencia"]
    assert emp["dif"] == 0.0 and emp["veredito"] == "indistinguivel"
    assert av._veredito({"ic_inf": -0.2, "ic_sup": -0.01}) == "superou"
    assert av._veredito({"ic_inf": 0.01, "ic_sup": 0.2}) == "pior"
    assert av._veredito({"ic_inf": -0.1, "ic_sup": 0.0}) == "indistinguivel"                # tocar o zero não é ganho
    assert av._veredito({"ic_inf": float("nan"), "ic_sup": float("nan")}) == "indefinido"


def test_alvos_de_baixa_confianca_ficam_fora_so_da_coluna_confiaveis(va):
    baixa = np.zeros(len(va), dtype=bool)
    baixa[::4] = True
    meta = va.meta.copy()
    meta["alvo_spi3_baixa_conf"] = baixa
    va2 = dataclasses.replace(va, meta=meta)
    # erro 1,0 nas linhas de baixa confiança e 0,0 nas demais: o "confiável" tem de dar 0 e o "todas" 0,25
    erro = np.zeros(va.Y.shape)
    erro[baixa, 0] = 1.0
    p = prevs(va2) | {"rede": com_erro(va2, erro)}
    t = av.pontuar(va2, p, n_boot=200, seed=1)["modelos"]["rede"]["alvo_spi3_h1"]
    assert t["confiaveis"]["n"] == int((~baixa).sum()) and t["confiaveis"]["mae"] == 0.0
    assert t["todas"]["n"] == len(va) and t["todas"]["mae"] == pytest.approx(baixa.mean())
    t6 = av.pontuar(va2, p, n_boot=200, seed=1)["modelos"]["rede"]["alvo_spi6_h1"]
    assert t6["confiaveis"]["n"] == len(va)                                                  # SPI-6 usa a flag do SPI-6


def test_mae_por_mes_do_calendario_e_meses_sem_linha_ficam_vazios(va):
    mes = va.meta["mes_alvo"].dt.month.to_numpy()
    erro = np.zeros(va.Y.shape)
    erro[mes == 3, :] = 1.0
    p = prevs(va) | {"rede": com_erro(va, erro)}
    por_mes = av.pontuar(va, p, n_boot=200, seed=1)["modelos"]["rede"]["alvo_spi3_h1"]["por_mes_calendario"]
    assert por_mes["3"] == pytest.approx(1.0, abs=1e-5)                                      # float32 no rótulo
    assert all(por_mes[str(c)] == 0.0 for c in range(1, 13) if c != 3)
    primeiro_semestre = va.fatiar(mes <= 6)
    pm = av.pontuar(primeiro_semestre, prevs(primeiro_semestre), n_boot=200, seed=1)
    vazio = pm["modelos"]["rede"]["alvo_spi3_h1"]["por_mes_calendario"]
    assert all(vazio[str(c)] is None for c in range(7, 13)) and all(vazio[str(c)] is not None for c in range(1, 7))


def test_margem_e_o_quantil_do_erro_e_acusa_instabilidade_entre_metades(va):
    g = (va.meta["mes_alvo"].dt.year * 12 + va.meta["mes_alvo"].dt.month).to_numpy()
    meses = np.unique(g)
    primeira = g < meses[len(meses) // 2]
    rng = np.random.default_rng(0)
    d = rng.uniform(0.0, 1.0, size=va.Y.shape)
    d[primeira] *= 0.5                                      # 1ª metade: erros pequenos; 2ª: o dobro
    r = av.pontuar(va, prevs(va) | {"rede": com_erro(va, d)}, n_boot=200, seed=1)["margem"]["alvo_spi3_h1"]
    assert r["valor"] == pytest.approx(np.quantile(d[:, 0], 0.8))
    assert r["cobertura_validacao"] >= 0.8 - 1e-9
    assert r["margem_1a_metade"] == pytest.approx(np.quantile(d[primeira, 0], 0.8))
    assert r["cobertura_2a_metade"] < 0.6 and r["estavel"] is False
    # erro estacionário: a margem da 1ª metade segura a 2ª
    ok = av.pontuar(va, prevs(va) | {"rede": com_erro(va, quase_uniforme(va))}, n_boot=200, seed=1)["margem"]["alvo_spi3_h1"]
    assert ok["estavel"] is True and ok["cobertura_2a_metade"] > 0.7


def test_entradas_incoerentes_sao_recusadas(conj, cfg, va):
    p = prevs(va)
    treino = fe.selecionar(conj, "train", cfg, so_com_rotulo=True)
    with pytest.raises(av.AvaliarError, match="bloco 'val'"):
        av.pontuar(treino, {"persistencia": np.zeros(treino.Y.shape), "rede": np.zeros(treino.Y.shape)}, seed=1)
    with pytest.raises(av.AvaliarError, match="bloco 'val'"):
        av.pontuar(va.fatiar(np.zeros(len(va), dtype=bool)), {n: x[:0] for n, x in p.items()}, seed=1)
    with pytest.raises(av.AvaliarError, match="persistencia"):
        av.pontuar(va, {"rede": p["rede"]}, seed=1)
    with pytest.raises(av.AvaliarError, match="'rede'"):
        av.pontuar(va, {"persistencia": p["persistencia"]}, seed=1)
    with pytest.raises(av.AvaliarError, match="shape"):
        av.pontuar(va, p | {"ridge": p["ridge"][:, :1]}, seed=1)
    ruim = p["rede"].copy()
    ruim[np.flatnonzero(va.M[:, 0])[0], 0] = np.nan
    with pytest.raises(mt.MetricasError, match="previsão"):
        av.pontuar(va, p | {"rede": ruim}, n_boot=200, seed=1)
    M = va.M.copy()
    M[:, 0] = False
    with pytest.raises(av.AvaliarError, match="alvo_spi3_h1"):
        av.pontuar(dataclasses.replace(va, M=M), p, n_boot=200, seed=1)


# ------------------------------------------------------------------ relatório
def relatorio(va, p=None, **kw):
    corpo = av.pontuar(va, p or prevs(va), n_boot=200, seed=1, **kw)
    return {"aviso": av.AVISO, "bloco": "val", "dataset_sha256": "ab" * 32, "seed": 42, "alvos": list(va.alvos),
            "horizonte": 1, "janela_validacao": ["2016-06", "2020-12"],
            "rede": {"pesos_sha256": "cd" * 32, "melhor_epoca": 5, "hiperparametros": {}}, **corpo}


def test_limpar_deixa_o_json_estrito():
    sujo = {"a": np.float32("nan"), "b": np.int64(3), "c": (1.5, np.bool_(True), float("inf")), "d": {"e": np.float64(2.5)}}
    limpo = av._limpar(sujo)
    assert limpo == {"a": None, "b": 3, "c": [1.5, True, None], "d": {"e": 2.5}}
    assert type(limpo["b"]) is int and type(limpo["c"][1]) is bool and type(limpo["d"]["e"]) is float
    json.dumps(limpo, allow_nan=False)


def test_texto_do_relatorio_tem_aviso_virgula_decimal_e_nao_inventa_ganho(va):
    txt = av.texto_relatorio(av._limpar(relatorio(va)))
    assert txt.count(av.AVISO) == 2 and "o teste congelado não foi lido" in txt
    assert "## alvo_spi3_h1" in txt and "## alvo_spi6_h1" in txt and "Rede PyTorch (do zero)" in txt
    assert "a rede erra menos" in txt and "supera a persistência em todas as cabeças: sim" in txt
    assert "0." not in txt.split("| Modelo |", 1)[1].split("\n\n")[0]                      # tabela com vírgula decimal
    igual = prevs(va)
    igual["rede"] = igual["persistencia"].copy()
    emp = av.texto_relatorio(av._limpar(relatorio(va, igual)))
    assert "indistinguível de ruído" in emp and "a rede erra menos" not in emp
    assert "supera a persistência em todas as cabeças: não" in emp
    assert "—" in av.texto_relatorio(av._limpar(relatorio(va.fatiar(va.meta["mes_alvo"].dt.month.to_numpy() <= 6))))  # mês vazio


def test_relatorio_avisa_quando_a_margem_nao_e_estavel(va):
    g = (va.meta["mes_alvo"].dt.year * 12 + va.meta["mes_alvo"].dt.month).to_numpy()
    primeira = g < np.unique(g)[len(np.unique(g)) // 2]
    d = np.random.default_rng(0).uniform(0.0, 1.0, size=va.Y.shape)
    d[primeira] *= 0.5
    txt = av.texto_relatorio(av._limpar(relatorio(va, prevs(va) | {"rede": com_erro(va, d)})))
    assert "NÃO se manteve entre as metades da validação em alvo_spi3_h1" in txt and "garantida" in txt
    estavel = av.texto_relatorio(av._limpar(relatorio(va, prevs(va) | {"rede": com_erro(va, quase_uniforme(va))})))
    assert "NÃO se manteve" not in estavel and "dentro da tolerância" in estavel


def test_gravar_cria_json_estrito_e_markdown_sem_deixar_temporarios(va, tmp_path):
    rel = relatorio(va)
    rel["modelos"]["rede"]["alvo_spi3_h1"]["todas"]["seca"]["f1"] = float("nan")             # NaN legítimo (sem denominador)
    pasta = tmp_path / "novo" / "modelo"
    j, m = av.gravar(rel, pasta)
    doc = json.loads(j.read_text(encoding="utf-8"), parse_constant=lambda x: pytest.fail(f"constante inválida {x}"))
    assert doc["modelos"]["rede"]["alvo_spi3_h1"]["todas"]["seca"]["f1"] is None and doc["aviso"] == av.AVISO
    assert m.read_text(encoding="utf-8").startswith("# Avaliação na validação") and sorted(p.name for p in pasta.iterdir()) == [av.ARQ_JSON, av.ARQ_MD]


# ------------------------------------------------------------------ linha de comando, ponta a ponta
@pytest.fixture
def proj(tmp_path):
    return montar_proj(tmp_path, raw_sintetico())


def hashes(raiz, excluir=()):
    return {str(p.relative_to(raiz)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(raiz.rglob("*")) if p.is_file() and not any(x in p.parts for x in excluir)}


def treinar(raiz):
    assert tn.main(["--params", str(raiz / "params.yaml"), "--epocas", "3", "--ocultas", "8"], saida=lambda _: None) == 0


ARGS = ["--n-arvores", "5", "--n-boot", "200"]


def test_cli_avalia_grava_nao_le_o_teste_nem_mexe_nos_dados_e_e_deterministica(proj, monkeypatch):
    cfg, raiz = proj
    treinar(raiz)
    antes = hashes(raiz, excluir=("modelo",))
    blocos = []
    original = fe.selecionar
    monkeypatch.setattr(fe, "selecionar", lambda c, nome, *a, **k: (blocos.append(nome), original(c, nome, *a, **k))[1])
    linhas = []
    assert av.main(["--params", str(raiz / "params.yaml"), *ARGS], saida=linhas.append) == 0
    txt = "\n".join(linhas)
    assert "AVISO: MVP experimental" in txt and "teste congelado não foi lido" in txt and "Próximo passo" in txt
    assert "alvo_spi3_h1: rede MAE" in txt and "alvo_spi6_h1: rede MAE" in txt
    assert set(blocos) == {"train", "val"}                                                   # o bloco "test" nunca foi pedido
    pasta = raiz / "artifacts" / "modelo"
    doc = json.loads((pasta / av.ARQ_JSON).read_text(encoding="utf-8"))
    assert doc["dataset_sha256"] == mf.verificar_manifest(cfg)["dataset_sha256"] and doc["bloco"] == "val"
    assert set(doc["modelos"]) == {"persistencia", "climatologia", "parcial_climatologica", "ridge", "floresta_aleatoria", "rede"}
    assert doc["rede"]["melhor_epoca"] >= 1 and len(doc["rede"]["pesos_sha256"]) == 64 and doc["n_linhas"] > 0
    assert (pasta / av.ARQ_MD).read_text(encoding="utf-8").count(av.AVISO) == 2
    assert hashes(raiz, excluir=("modelo",)) == antes
    primeiro = (pasta / av.ARQ_JSON).read_bytes()
    assert av.main(["--params", str(raiz / "params.yaml"), *ARGS], saida=lambda _: None) == 0
    assert (pasta / av.ARQ_JSON).read_bytes() == primeiro                                    # mesma semente, mesmo relatório


def test_cli_passa_semente_e_numero_de_arvores_aos_baselines(proj, monkeypatch):
    cfg, raiz = proj
    treinar(raiz)
    visto = {}
    original = av.bl.ajustar_todos

    def espia(*a, **k):
        visto.update(k)
        return original(*a, **k)
    monkeypatch.setattr(av.bl, "ajustar_todos", espia)
    assert av.main(["--params", str(raiz / "params.yaml"), "--n-arvores", "7", "--n-boot", "100"], saida=lambda _: None) == 0
    assert visto == {"seed": cfg.projeto.seed, "n_arvores": 7} and cfg.projeto.seed != 0


def test_cli_recusa_teste_liberado_sem_modelo_dado_adulterado_e_argumento_ruim(proj):
    cfg, raiz = proj
    p = ["--params", str(raiz / "params.yaml"), *ARGS]
    linhas = []
    assert av.main(p, saida=linhas.append) == 1 and "ERRO" in linhas[-1]                      # sem rede treinada
    treinar(raiz)
    liberado = cfg.model_copy(update={"split": cfg.split.model_copy(update={"teste_liberado": True})})
    linhas.clear()
    assert av.main(p, saida=linhas.append, carregar_cfg=lambda _: liberado) == 1
    assert "ERRO" in linhas[-1] and "teste_liberado" in linhas[-1]
    linhas.clear()
    assert av.main(["--params", str(raiz / "params.yaml"), "--n-arvores", "5", "--n-boot", "10"], saida=linhas.append) == 1
    assert "ERRO" in linhas[-1] and "n_boot" in linhas[-1]
    assert not (raiz / "artifacts" / "modelo" / av.ARQ_JSON).exists()                         # recusa não deixa relatório
    assert av.main(["--params", str(raiz / "nao_existe.yaml")], saida=linhas.append) == 1
    alvo = raiz / "data" / "curated" / "demo_observado.parquet"
    alvo.write_bytes(alvo.read_bytes() + b"\0")
    linhas.clear()
    assert av.main(p, saida=linhas.append) == 1 and "ERRO" in linhas[-1]


def test_cli_recusa_rede_treinada_com_outros_dados_ou_com_outro_padronizador(proj, monkeypatch):
    cfg, raiz = proj
    treinar(raiz)
    p = ["--params", str(raiz / "params.yaml"), *ARGS]
    arq = raiz / "artifacts" / "modelo" / "rede.json"
    doc = json.loads(arq.read_text(encoding="utf-8"))
    doc["meta"]["dataset_sha256"] = "0" * 64
    arq.write_text(json.dumps(doc), encoding="utf-8")
    linhas = []
    assert av.main(p, saida=linhas.append) == 1 and "outro dataset_sha256" in linhas[-1]
    doc["meta"]["dataset_sha256"] = mf.verificar_manifest(cfg)["dataset_sha256"]
    doc["meta"]["alvos"] = doc["meta"]["alvos"][::-1]                                         # só o avaliar confere os alvos
    arq.write_text(json.dumps(doc), encoding="utf-8")
    linhas.clear()
    assert av.main(p, saida=linhas.append) == 1 and "alvos" in linhas[-1]
    doc["meta"]["alvos"] = doc["meta"]["alvos"][::-1]
    doc["meta"]["colunas"] = doc["meta"]["colunas"][::-1]
    arq.write_text(json.dumps(doc), encoding="utf-8")
    linhas.clear()
    assert av.main(p, saida=linhas.append) == 1 and ("colunas" in linhas[-1] or "padronizador" in linhas[-1])
    doc["meta"]["colunas"] = doc["meta"]["colunas"][::-1]
    arq.write_text(json.dumps(doc), encoding="utf-8")
    carregar = tn.carregar_preditor

    def deslocado(pasta):
        pre, meta = carregar(pasta)
        pre.padronizador = dataclasses.replace(pre.padronizador, media=pre.padronizador.media + 0.5)
        return pre, meta
    monkeypatch.setattr(tn, "carregar_preditor", deslocado)
    linhas.clear()
    assert av.main(p, saida=linhas.append) == 1 and "padronizador" in linhas[-1]
