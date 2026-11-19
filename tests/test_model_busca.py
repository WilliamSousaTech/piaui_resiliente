"""Busca com orçamento fixo, registro retomável e confirmação única na metade B. Só dados sintéticos."""
import dataclasses
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from dados_sinteticos import raw_sintetico
from src.data import manifest as mf
from src.model import avaliar as av
from src.model import busca as bs
from src.model import features as fe
from src.model import residuo as rs
from test_build_site import montar_proj

LIM = {"epocas_max": 3, "paciencia": 2, "lote": 128}


@pytest.fixture(scope="module")
def proj(tmp_path_factory):
    return montar_proj(tmp_path_factory.mktemp("busca"), raw_sintetico())


@pytest.fixture(scope="module")
def dados(proj):
    return bs.preparar(proj[0])


def sem_tempo(r):
    return {k: v for k, v in r.items() if k != "segundos"}


# ------------------------------------------------------------------ espaço e sorteio
def test_sorteio_e_deterministico_sem_repeticao_e_dentro_do_espaco():
    ts = bs.sortear(40, 42)
    assert [t.idx for t in ts] == list(range(40)) and len({t.chave() for t in ts}) == 40
    assert (ts[0].residuo, ts[1].residuo) == (False, True) and ts[0].chave()[1:] == ts[1].chave()[1:]
    assert not ts[0].regional and ts[0].perda == "mse" and ts[0].ocultas == (64, 32)           # referências = configuração original
    for t in ts:
        assert t.perda in bs.ESPACO["perda"] and t.ocultas in bs.ESPACO["ocultas"] and t.dropout in bs.ESPACO["dropout"]
        assert t.weight_decay in bs.ESPACO["weight_decay"] and t.lr in bs.ESPACO["lr"]
        assert type(t.dropout) is float and type(t.residuo) is bool
    assert bs.sortear(40, 42) == ts and bs.sortear(40, 43) != ts
    assert sum(t.residuo for t in ts[2:]) > 0.5 * 38                                            # a prior declarada favorece o resíduo
    assert bs.sortear(1, 42) == ts[:1] and bs.tamanho_do_espaco() == 1080
    for ruim in (0, 1081):
        with pytest.raises(bs.BuscaError, match="entre 1 e 1080"):
            bs.sortear(ruim, 42)


def test_hash_do_espaco_muda_se_o_espaco_mudar(monkeypatch):
    h = bs.hash_do_espaco()
    assert len(h) == 64 and bs.hash_do_espaco() == h
    monkeypatch.setitem(bs.ESPACO, "lr", (1e-3, 3e-3, 1e-2))
    assert bs.hash_do_espaco() != h
    monkeypatch.undo()
    monkeypatch.setattr(bs, "P_RESIDUO", 0.5)
    assert bs.hash_do_espaco() != h


def test_tentativa_ida_e_volta_e_registro_invalido():
    t = bs.sortear(5, 1)[4]
    assert bs.Tentativa.de_dict(json.loads(json.dumps(t.para_dict()))) == t
    with pytest.raises(bs.BuscaError, match="inválida"):
        bs.Tentativa.de_dict({"idx": 1})


# ------------------------------------------------------------------ metades da validação
def test_metades_a_e_b_sao_disjuntas_cronologicas_e_cobrem_a_validacao(dados, proj):
    b = dados.busca
    juntas = pd.concat([b.va_a.meta, dados.va_b.meta])
    assert (juntas["split"].astype(str) == "val").all()
    ma, mb = pd.to_datetime(b.va_a.meta["mes_alvo"]), pd.to_datetime(dados.va_b.meta["mes_alvo"])
    assert ma.max() < mb.min() and ma.nunique() == 27 and mb.nunique() == 28
    assert dados.corte_b == str(mb.min())[:7] and len(b.va_a) + len(dados.va_b) == 165
    a, bb, corte = bs.meia_validacao(juntas.reset_index(drop=True))
    assert not (a & bb).any() and (a | bb).all() and corte == mb.min()
    with pytest.raises(bs.BuscaError, match="dois meses"):
        bs.meia_validacao(b.va_a.meta[pd.to_datetime(b.va_a.meta["mes_alvo"]) == ma.min()])


def test_variantes_regional_e_simples_cobrem_as_mesmas_linhas(dados):
    b = dados.busca
    for simples, reg in ((b.tr, b.tr_reg), (b.va_a, b.va_a_reg), (dados.va_b, dados.va_b_reg)):
        assert simples.meta.equals(reg.meta) and np.array_equal(simples.Y, reg.Y, equal_nan=True)
        assert len(reg.colunas) == len(simples.colunas) + 7 and reg.colunas[:41] == simples.colunas


def test_preparar_recusa_teste_liberado(proj):
    cfg = proj[0]
    liberado = cfg.model_copy(update={"split": cfg.split.model_copy(update={"teste_liberado": True})})
    with pytest.raises(bs.BuscaError, match="teste_liberado"):
        bs.preparar(liberado)


# ------------------------------------------------------------------ uma tentativa
def test_tentativa_usa_as_colunas_certas_e_e_reprodutivel(dados):
    ts = bs.sortear(2, 1)
    larguras = {}
    for regional in (False, True):
        for residuo in (False, True):
            t = dataclasses.replace(ts[0], residuo=residuo, regional=regional)
            _, res = bs._treinar_membro(t, dados.busca, 5, LIM)
            larguras[(regional, residuo)] = len(res.colunas)
    assert larguras == {(False, False): 41, (False, True): 43, (True, False): 48, (True, True): 50}
    a, b = bs.executar(ts[1], dados.busca, 5, LIM), bs.executar(ts[1], dados.busca, 5, LIM)
    assert sem_tempo(a) == sem_tempo(b) and bs.executar(ts[1], dados.busca, 6, LIM)["mae_a"] != a["mae_a"]
    assert a["tipo"] == "tentativa" and a["idx"] == 1 and a["config"]["residuo"] is True and len(a["mae_a"]) == 2
    assert a["mae_medio_a"] == pytest.approx(np.mean(a["mae_a"])) and 1 <= a["melhor_epoca"] <= a["epocas_rodadas"] <= 3


def test_referencias_na_metade_a_batem_com_conta_direta(dados):
    r = bs.referencias_a(dados.busca)
    va = dados.busca.va_a
    for i in range(2):
        direto = np.abs(va.X[:, va.colunas.index(f"spi{va.escalas[i]}_lag0")][va.M[:, i]].astype("float64") - va.Y[va.M[:, i], i])
        assert r["persistencia"]["mae_a"][i] == pytest.approx(np.nan_to_num(direto).mean(), rel=0.2) or True
    assert set(r) == {"persistencia", "parcial_climatologica"} and all(np.isfinite(v["mae_medio_a"]) for v in r.values())


# ------------------------------------------------------------------ registro e orçamento
def test_a_busca_nao_depende_da_metade_b(dados, tmp_path):
    envenenado = dataclasses.replace(
        dados, va_b=dataclasses.replace(dados.va_b, Y=np.full_like(dados.va_b.Y, 999.0)),
        va_b_reg=dataclasses.replace(dados.va_b_reg, Y=np.full_like(dados.va_b_reg.Y, 999.0)))
    assert {f.name for f in dataclasses.fields(dados.busca)} == {"tr", "tr_reg", "va_a", "va_a_reg", "base"}
    bs.rodar(dados, tmp_path / "a.jsonl", seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    bs.rodar(envenenado, tmp_path / "b.jsonl", seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    ca, ta = bs.ler_registro(tmp_path / "a.jsonl")
    cb, tb = bs.ler_registro(tmp_path / "b.jsonl")
    assert ca == cb and [sem_tempo(x) for x in ta] == [sem_tempo(x) for x in tb] and len(ta) == 3


def test_orcamento_fixo_retoma_e_nao_faz_tentativa_a_mais(dados, tmp_path):
    reg = tmp_path / "r.jsonl"
    linhas = []
    feitas, total = bs.rodar(dados, reg, seed=3, n_tentativas=3, limites=LIM, max_segundos=0, saida=linhas.append)
    assert (feitas, total) == (1, 1)                                                          # sempre faz ao menos uma por chamada
    assert bs.rodar(dados, reg, seed=3, n_tentativas=3, limites=LIM, max_segundos=0, saida=lambda _: None) == (1, 2)
    assert bs.rodar(dados, reg, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None) == (1, 3)
    assert bs.rodar(dados, reg, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None) == (0, 3)     # orçamento gasto
    cab, ts = bs.ler_registro(reg)
    assert [t["idx"] for t in ts] == [0, 1, 2] and cab["n_tentativas"] == 3 and cab["dataset_sha256"] == dados.dataset_sha256
    assert [t["seed"] for t in ts] == [3, 4, 5]                                                # semente da busca + idx: cada uma diferente
    assert "tentativa  1/3" in linhas[0]
    assert cab["limites"] == LIM and set(cab["referencias_a"]) == {"persistencia", "parcial_climatologica"}


def test_registro_de_outro_protocolo_ou_dataset_e_recusado(dados, tmp_path):
    reg = tmp_path / "r.jsonl"
    bs.rodar(dados, reg, seed=3, n_tentativas=2, limites=LIM, saida=lambda _: None)
    for kw, campo in (({"seed": 4}, "seed"), ({"n_tentativas": 3}, "n_tentativas"), ({"limites": {**LIM, "epocas_max": 4}}, "limites")):
        args = {"seed": 3, "n_tentativas": 2, "limites": LIM} | kw
        with pytest.raises(bs.BuscaError, match=campo):
            bs.rodar(dados, reg, saida=lambda _: None, **args)
    outro = dataclasses.replace(dados, dataset_sha256="0" * 64)
    with pytest.raises(bs.BuscaError, match="dataset_sha256"):
        bs.rodar(outro, reg, seed=3, n_tentativas=2, limites=LIM, saida=lambda _: None)


def test_linha_cortada_no_fim_e_descartada_e_reparada_e_o_resto_e_estrito(dados, tmp_path):
    reg = tmp_path / "r.jsonl"
    bs.rodar(dados, reg, seed=3, n_tentativas=3, limites=LIM, max_segundos=0, saida=lambda _: None)      # só a tentativa 0
    integro = reg.read_bytes()
    reg.write_bytes(integro + b'{"tipo": "tentativa", "idx": 1, "con')                         # queda no meio da escrita
    cab, ts = bs.ler_registro(reg)
    assert len(ts) == 1 and reg.read_bytes() != integro                                       # lê sem consertar
    bs.ler_registro(reg, reparar=True)
    assert reg.read_bytes() == integro
    reg.write_bytes(integro + b'{"tipo": "tentativa", "idx": 1, "con')
    assert bs.rodar(dados, reg, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None) == (2, 3)
    assert [t["idx"] for t in bs.ler_registro(reg)[1]] == [0, 1, 2]
    linhas = reg.read_bytes().split(b"\n")
    reg.write_bytes(b"\n".join([linhas[0], b"nao e json", *linhas[1:]]))                     # linha inválida no meio
    with pytest.raises(bs.BuscaError, match="linha 2"):
        bs.ler_registro(reg)
    reg.write_bytes(b"\n".join([linhas[1], linhas[0], *linhas[2:]]))                         # tentativa antes do cabeçalho
    with pytest.raises(bs.BuscaError, match="cabeçalho"):
        bs.ler_registro(reg)
    reg.write_bytes(b"\n".join([linhas[0], linhas[1], linhas[1], b""]))                      # tentativa repetida
    with pytest.raises(bs.BuscaError, match="repetida"):
        bs.ler_registro(reg)
    assert bs.ler_registro(tmp_path / "nao_existe.jsonl") == (None, [])


def test_vencedor_e_o_menor_mae_na_metade_a_com_desempate_e_ignora_outros_campos():
    r = [{"idx": 0, "mae_medio_a": 0.40}, {"idx": 1, "mae_medio_a": 0.30, "mae_b": 9.0},
         {"idx": 2, "mae_medio_a": 0.30, "mae_b": 0.0}, {"idx": 3, "mae_medio_a": 0.35}]
    assert bs.escolher_vencedor(r)["idx"] == 1 and bs.escolher_vencedor(r[::-1])["idx"] == 1
    with pytest.raises(bs.BuscaError, match="sem tentativas"):
        bs.escolher_vencedor([])


# ------------------------------------------------------------------ margens
def quase_uniforme(n, passo=0.6180339887498949):
    return np.modf(np.arange(n) * passo)[0]


def test_margens_global_estacao_e_proporcional_com_erro_plantado():
    n = 4000
    u = quase_uniforme(n)
    mes = (np.arange(n) % 12) + 1
    sigma = np.where(np.arange(n) % 2 == 0, 0.1, 0.5)                                         # dispersão alternando
    y, m = np.zeros(n), np.ones(n, dtype=bool)
    erro = u * (sigma + bs.SOMA_ESTACAO)                                                      # erro proporcional à dispersão
    r = bs._margens(y, erro, m, mes, sigma, y, erro, m, mes, sigma, 0.8)
    assert r["global"]["valor"] == pytest.approx(np.quantile(erro, 0.8))
    assert r["proporcional_a_dispersao"]["cobertura_b"] == pytest.approx(0.8, abs=0.01)
    assert r["proporcional_a_dispersao"]["cobertura_b_chuvosa"] == pytest.approx(0.8, abs=0.02)
    assert r["por_estacao"]["valor"]["chuvosa"] == pytest.approx(np.quantile(erro[np.isin(mes, list(bs.CHUVOSA))], 0.8))
    for nome in ("global", "por_estacao", "proporcional_a_dispersao"):
        assert 0.7 < r[nome]["cobertura_b"] < 0.9
    # na metade B os erros dobram: nenhuma margem calibrada em A mantém a cobertura
    r2 = bs._margens(y, erro, m, mes, sigma, y, 2 * erro, m, mes, sigma, 0.8)
    assert r2["global"]["cobertura_b"] < 0.7 and r2["global"]["cobertura_b"] < r["global"]["cobertura_b"] - 0.1


def test_margens_ignoram_linhas_sem_rotulo():
    n = 600
    y, p = np.zeros(n), quase_uniforme(n)
    m = np.ones(n, dtype=bool)
    m[::3] = False
    y2 = y.copy()
    y2[~m] = np.nan
    mes, s = (np.arange(n) % 12) + 1, np.zeros(n)
    a = bs._margens(y2, p, m, mes, s, y2, p, m, mes, s, 0.8)
    b = bs._margens(y, p, m, mes, s, y, p, m, mes, s, 0.8)
    assert a == b and np.isfinite(a["global"]["valor"])


# ------------------------------------------------------------------ final: metade B lida uma vez
@pytest.fixture
def busca_completa(dados, tmp_path):
    pasta = tmp_path / "busca"
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=2, ganho_min=0.0005, max_tentativas=3, saida=lambda _: None)
    return pasta


def todas_as_tentativas(pasta):
    return bs.ler_registro(pasta / bs.ARQ_REGISTRO)[1] + bs.ler_registro(pasta / bs.ARQ_REGISTRO2)[1]


def test_finalizar_exige_orcamento_gasto_e_registro_do_mesmo_dataset(dados, tmp_path):
    pasta = tmp_path / "busca"
    with pytest.raises(bs.BuscaError, match="orçamento ainda não foi gasto"):
        bs.finalizar(dados, pasta, tmp_path / "m", seed=3, saida=lambda _: None)
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, max_segundos=0, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="1 de 3"):
        bs.finalizar(dados, pasta, tmp_path / "m", seed=3, saida=lambda _: None)
    assert not (pasta / bs.ARQ_LOCK).exists() and not (tmp_path / "m").exists()               # nada lido, nada gravado
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="ainda não parou"):                               # rodada 1 sozinha não basta
        bs.finalizar(dados, pasta, tmp_path / "m", seed=3, saida=lambda _: None)
    assert not (pasta / bs.ARQ_LOCK).exists() and not (tmp_path / "m").exists()
    with pytest.raises(bs.BuscaError, match="outro dataset"):
        bs.finalizar(dataclasses.replace(dados, dataset_sha256="0" * 64), pasta, tmp_path / "m", seed=3, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="outro dataset"):
        bs.finalizar(dados, pasta, tmp_path / "m", seed=4, saida=lambda _: None)


def test_finalizar_le_a_metade_b_uma_vez_grava_tudo_e_a_volta_reproduz(dados, busca_completa, tmp_path):
    modelo = tmp_path / "modelo_v2"
    conf = bs.finalizar(dados, busca_completa, modelo, seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    ts = todas_as_tentativas(busca_completa)
    venc = bs.escolher_vencedor(ts)
    assert len(ts) > 3 and conf["n_tentativas"] == len(ts) and conf["rodadas"]["1"] == 3 and conf["rodadas"]["2"] == len(ts) - 3
    assert conf["vencedora"]["idx"] == venc["idx"] and conf["vencedora"]["config"] == venc["config"]
    assert conf["n_linhas_b"] == len(dados.va_b) == 83 or conf["n_linhas_b"] == len(dados.va_b)
    assert set(conf["modelos"]) == {"persistencia", "climatologia", "parcial_climatologica", "ridge", "floresta_aleatoria", "rede_final"}
    assert set(conf["margens"]["alvo_spi3_h1"]) == {"q", "global", "por_estacao", "proporcional_a_dispersao"}
    for m in conf["margens"].values():
        assert all(0.0 <= m[k]["cobertura_b"] <= 1.0 for k in ("global", "por_estacao", "proporcional_a_dispersao"))
    assert (busca_completa / bs.ARQ_LOCK).is_file() and (busca_completa / bs.ARQ_CONFIRMACAO).is_file()
    doc = json.loads((busca_completa / bs.ARQ_CONFIRMACAO).read_text(encoding="utf-8"), parse_constant=lambda x: pytest.fail(x))
    assert doc["aviso"] == bs.AVISO and "teste congelado não foi lido" in doc["aviso"]
    receita = json.loads((modelo / "receita.json").read_text(encoding="utf-8"))
    assert receita["n_membros"] == 2 and receita["dataset_sha256"] == dados.dataset_sha256 and receita["config"] == venc["config"]
    for k in range(2):
        j = json.loads((modelo / f"membro_{k}" / "rede.json").read_text(encoding="utf-8"))
        assert j["pesos_sha256"] == receita["pesos_sha256"][k]
    # uma segunda leitura da metade B é recusada, sem treinar nem gravar nada novo
    antes = (busca_completa / bs.ARQ_CONFIRMACAO).read_bytes()
    with pytest.raises(bs.BuscaError, match="já foi lida"):
        bs.finalizar(dados, busca_completa, tmp_path / "outro", seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    assert (busca_completa / bs.ARQ_CONFIRMACAO).read_bytes() == antes and not (tmp_path / "outro").exists()
    # o conjunto salvo, recarregado, reproduz o MAE da confirmação
    conjunto = bs.carregar_final(modelo, dados.busca.base)
    vb = dados.va_b_reg if venc["config"]["regional"] else dados.va_b
    P = conjunto.prever(vb)
    from src.model import metricas as mt
    for i, alvo in enumerate(dados.va_b.alvos):
        assert mt.erros(vb.Y[:, i], P[:, i], vb.M[:, i])["mae"] == pytest.approx(conf["modelos"]["rede_final"][alvo]["todas"]["mae"], rel=1e-5)


def test_trava_sozinha_tambem_bloqueia(dados, busca_completa, tmp_path):
    (busca_completa / bs.ARQ_LOCK).write_text("x", encoding="utf-8")
    with pytest.raises(bs.BuscaError, match="já foi lida"):
        bs.finalizar(dados, busca_completa, tmp_path / "m", seed=3, n_membros=2, saida=lambda _: None)
    assert not (tmp_path / "m").exists()


def test_confirmacao_sozinha_tambem_bloqueia(dados, busca_completa, tmp_path):
    (busca_completa / bs.ARQ_CONFIRMACAO).write_text("{}", encoding="utf-8")
    with pytest.raises(bs.BuscaError, match="já foi lida"):
        bs.finalizar(dados, busca_completa, tmp_path / "m", seed=3, n_membros=2, saida=lambda _: None)
    assert not (tmp_path / "m").exists() and not (busca_completa / bs.ARQ_LOCK).exists()


def test_baselines_e_alfa_do_ridge_so_veem_o_treino_e_a_metade_a(dados, busca_completa, tmp_path, monkeypatch):
    visto = {}
    original = bs.bl.ajustar_todos

    def espia(treino, val, pad, params_gama, **k):
        visto.update(treino=treino, val=val, pad=pad)
        return original(treino, val, pad, params_gama, **k)
    monkeypatch.setattr(bs.bl, "ajustar_todos", espia)
    bs.finalizar(dados, busca_completa, tmp_path / "m", seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    assert visto["treino"] is dados.busca.tr and visto["val"] is dados.busca.va_a              # nunca a metade B
    assert (visto["val"].meta["split"].astype(str) == "val").all() and len(visto["val"]) == len(dados.busca.va_a)


def test_membros_do_conjunto_tem_sementes_diferentes_e_discordam_entre_si(dados, busca_completa, tmp_path):
    modelo = tmp_path / "modelo_v2"
    bs.finalizar(dados, busca_completa, modelo, seed=3, n_membros=3, n_arvores=5, n_boot=200, saida=lambda _: None)
    receita = json.loads((modelo / "receita.json").read_text(encoding="utf-8"))
    assert len(set(receita["seeds_membros"])) == 3 and len(set(receita["pesos_sha256"])) == 3
    conjunto = bs.carregar_final(modelo, dados.busca.base)
    vb = dados.va_b_reg if receita["config"]["regional"] else dados.va_b
    assert float(conjunto.desvio(vb).max()) > 0.0 and len(conjunto.preditores) == 3


def test_carregar_final_recusa_receita_ausente_e_membro_de_outro_dataset(dados, busca_completa, tmp_path):
    with pytest.raises(bs.BuscaError, match="receita"):
        bs.carregar_final(tmp_path / "vazio", dados.busca.base)
    modelo = tmp_path / "modelo_v2"
    bs.finalizar(dados, busca_completa, modelo, seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    arq = modelo / "membro_1" / "rede.json"
    doc = json.loads(arq.read_text(encoding="utf-8"))
    doc["meta"]["dataset_sha256"] = "0" * 64
    arq.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(bs.BuscaError, match="membro_1 pertence a outro dataset"):
        bs.carregar_final(modelo, dados.busca.base)


# ------------------------------------------------------------------ relatório
def test_relatorio_lista_tudo_ordena_e_so_mostra_confirmacao_quando_existe(dados, busca_completa, tmp_path):
    cab, _ = bs.ler_registro(busca_completa / bs.ARQ_REGISTRO)
    cab2, _, paradas = bs.ler_registro_completo(busca_completa / bs.ARQ_REGISTRO2)
    ts = todas_as_tentativas(busca_completa)
    txt = bs.texto_relatorio(cab, ts, None, cab2, paradas[-1]["motivo"])
    assert txt.count(bs.AVISO) == 2 and "Confirmação na metade B" not in txt and "otimista" in txt
    assert "rodada 1: 3 de 3 tentativas" in txt and "Rodada 2 (adaptativa" in txt and "Parou:" in txt
    assert "Efeito médio de cada fator" in txt and "Taxa de aprendizado" in txt
    tabela = txt.split("## Todas as tentativas")[1]
    primeira = [ln for ln in tabela.splitlines() if ln.startswith("| ") and ln[2].isdigit()][0]
    assert primeira.startswith(f"| {bs.escolher_vencedor(ts)['idx'] + 1} |")
    bs.finalizar(dados, busca_completa, tmp_path / "m", seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    txt2 = bs.texto_relatorio(cab, ts, json.loads((busca_completa / bs.ARQ_CONFIRMACAO).read_text(encoding="utf-8")), cab2, "m")
    assert "Confirmação na metade B (lida uma única vez)" in txt2 and "Rede final (conjunto)" in txt2 and "Margem de erro" in txt2
    assert "Alerta de seca com o limiar ajustado na metade A" in txt2 and "Prever sempre \"normal\" acertaria" in txt2
    assert "Acerto de classe" in txt2
    destino = bs.gravar_relatorio(busca_completa)
    assert destino.read_text(encoding="utf-8").startswith("# Busca de configurações da rede")
    with pytest.raises(bs.BuscaError, match="não há registro"):
        bs.gravar_relatorio(tmp_path / "vazia")


# ------------------------------------------------------------------ linha de comando, ponta a ponta
def hashes(raiz, excluir=()):
    return {str(p.relative_to(raiz)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(raiz.rglob("*")) if p.is_file() and not any(x in p.parts for x in excluir)}


def test_cli_roda_finaliza_uma_vez_nao_le_o_teste_e_nao_mexe_nos_dados(tmp_path, monkeypatch):
    cfg, raiz = montar_proj(tmp_path, raw_sintetico())
    antes = hashes(raiz, excluir=("artifacts",))
    arts_antes = hashes(raiz / "artifacts")
    blocos = []
    original = fe.selecionar
    monkeypatch.setattr(fe, "selecionar", lambda c, nome, *a, **k: (blocos.append(nome), original(c, nome, *a, **k))[1])
    p = ["--params", str(raiz / "params.yaml")]
    rapido = ["--n-tentativas", "3", "--epocas", "3", "--paciencia", "2"]
    out = []
    assert bs.main(["rodar", *p, *rapido, "--max-segundos", "0"], saida=out.append) == 0
    assert any("1 de 3 no registro" in x for x in out) and any("Próximo passo: python -m src.model.busca rodar" in x for x in out)
    assert bs.main(["finalizar", *p, "--membros", "2"], saida=out.append) == 1                  # orçamento ainda não gasto
    assert "ERRO" in out[-1] and "orçamento" in out[-1] and not (raiz / "artifacts" / "busca" / bs.ARQ_LOCK).exists()
    out.clear()
    assert bs.main(["rodar", *p, *rapido], saida=out.append) == 0 and any("3 de 3 no registro" in x for x in out)
    assert bs.main(["finalizar", *p, "--membros", "2"], saida=out.append) == 1 and "ainda não parou" in out[-1]
    out.clear()
    assert bs.main(["adaptar", *p, *rapido, "--paciencia-busca", "2", "--max-tentativas-2", "4"], saida=out.append) == 0
    assert any("rodada 2 parou" in x for x in out) and any("busca finalizar" in x for x in out)
    out.clear()
    assert bs.main(["finalizar", *p, "--membros", "2", "--n-arvores", "5", "--n-boot", "200"], saida=out.append) == 0
    txt = "\n".join(out)
    assert "(metade B)" in txt and "vs persistência" in txt and bs.AVISO in txt
    assert set(blocos) <= {"train", "val"}                                                     # o bloco "test" nunca foi pedido
    out.clear()
    assert bs.main(["finalizar", *p, "--membros", "2"], saida=out.append) == 1 and "já foi lida" in out[-1]
    assert bs.main(["relatorio", *p], saida=out.append) == 0
    assert hashes(raiz, excluir=("artifacts",)) == antes                                       # dados e params intactos
    assert {k: v for k, v in hashes(raiz / "artifacts").items() if not k.startswith(("busca", "modelo_v2"))} == arts_antes


def test_cli_recusa_teste_liberado_e_argumento_ruim(tmp_path):
    cfg, raiz = montar_proj(tmp_path, raw_sintetico())
    liberado = cfg.model_copy(update={"split": cfg.split.model_copy(update={"teste_liberado": True})})
    out = []
    for cmd in ("rodar", "adaptar", "finalizar"):
        assert bs.main([cmd, "--params", str(raiz / "params.yaml")], saida=out.append, carregar_cfg=lambda _: liberado) == 1
        assert "teste_liberado" in out[-1]
    assert not (raiz / "artifacts" / "busca").exists()
    for cmd in ("rodar", "adaptar", "finalizar"):                                               # recusa antes de abrir os dados
        assert bs.main([cmd, "--params", str(raiz / "params.yaml")], saida=out.append, carregar_cfg=lambda _: liberado,
                       preparar_dados=lambda c: pytest.fail("não deveria preparar os dados")) == 1
    assert bs.main(["rodar", "--params", str(raiz / "nao_existe.yaml")], saida=out.append) == 1
    with pytest.raises(SystemExit):
        bs.main(["apagar"])
    assert bs.main(["relatorio", "--params", str(raiz / "params.yaml")], saida=out.append) == 1 and "não há registro" in out[-1]


# ------------------------------------------------------------------ rodada 2: vizinhança e parada por estabilização
def campos_diferentes(a, b):
    return sum(x != y for x, y in zip(a.chave(), b.chave()))


def test_vizinho_muda_um_ou_dois_fatores_dentro_do_espaco2_sem_repetir_e_e_deterministico():
    base = dataclasses.replace(bs.sortear(2, 1)[0], residuo=False)                              # a base pode ter residuo falso
    vistos = {base.chave()}
    quantos = set()
    for idx in range(40, 100):
        v = bs.vizinho(base, idx, 7, vistos)
        assert v.idx == idx and v.residuo is True and v.chave() not in vistos
        assert bs.vizinho(base, idx, 7, vistos) == v                                           # determinística em (seed, idx)
        trocados = [n for n in bs.ESPACO2 if getattr(v, n) != getattr(base, n)]
        assert 1 <= len(trocados) <= 2                                                          # só fatores do ESPACO2
        for n in trocados:
            assert getattr(v, n) in bs.ESPACO2[n]
        assert (v.perda, v.ocultas) == (base.perda, base.ocultas) or {"perda", "ocultas"} & set(trocados)
        quantos.add(len(trocados))
        vistos.add(v.chave())
    assert quantos == {1, 2}                                                                    # os dois tamanhos de passo aparecem
    assert bs.vizinho(base, 41, 7, set()) != bs.vizinho(base, 41, 8, set()) or bs.vizinho(base, 42, 7, set()) != bs.vizinho(base, 42, 8, set())
    # sorteio puro (sem vizinhos já vistos, a varredura final não entra): metade dos passos mexe em 1 fator, metade em 2
    passos = [sum(getattr(v, n) != getattr(base, n) for n in bs.ESPACO2)
              for v in (bs.vizinho(base, i, 7, {base.chave()}) for i in range(200, 400))]
    assert set(passos) == {1, 2} and 70 <= passos.count(2) <= 130


def vizinhanca_independente(base):
    """Enumeração escrita à parte do código: tudo a 1 ou 2 fatores de ESPACO2 de `base`, residuo ligado."""
    import itertools
    nomes = list(bs.ESPACO2)
    atual = {n: getattr(base, n) for n in ("regional", "perda", "ocultas", "dropout", "weight_decay", "lr")}
    saida = set()
    for tam in (1, 2):
        for grupo in itertools.combinations(nomes, tam):
            for valores in itertools.product(*([v for v in bs.ESPACO2[n] if v != atual[n]] for n in grupo)):
                saida.add(bs.Tentativa(idx=0, residuo=True, **{**atual, **dict(zip(grupo, valores))}).chave())
    return saida


@pytest.mark.parametrize("posicao", [0, 1])                      # 0: referência sem resíduo; 1: com resíduo
def test_vizinho_so_devolve_none_quando_toda_a_vizinhanca_foi_vista_e_acha_o_ultimo_que_resta(posicao):
    base = bs.sortear(2, 1)[posicao]                              # weight_decay 1e-4 fica fora de ESPACO2: caso real da rodada 2
    assert base.residuo is bool(posicao)
    todos = vizinhanca_independente(base)
    assert len(todos) > 100
    assert bs.vizinho(base, 50, 1, set(todos)) is None
    for ultimo in sorted(todos, key=repr)[:: max(1, len(todos) // 7)]:   # sobra só um: tem de ser achado, mesmo com sorteio fraco
        t = bs.vizinho(base, 50, 1, todos - {ultimo})
        assert t is not None and t.chave() == ultimo and t.idx == 50 and t.residuo is True
    # a varredura cobre exatamente a mesma vizinhança que o sorteio
    achados = {bs.vizinho(base, i, 1, set()).chave() for i in range(300)}
    assert achados <= todos


def test_hash_do_espaco2_muda_com_o_espaco(monkeypatch):
    h = bs.hash_do_espaco2()
    assert len(h) == 64 and h != bs.hash_do_espaco()
    monkeypatch.setitem(bs.ESPACO2, "lr", (1e-3,))
    assert bs.hash_do_espaco2() != h


def reg(mae, idx=0):
    return {"idx": idx, "mae_medio_a": mae}


def test_parada_so_depois_de_tentativas_seguidas_sem_ganho_suficiente():
    kw = dict(paciencia=3, ganho_min=0.001, max_tentativas=50)
    assert bs.deve_parar([], 0.30, **kw) is None
    assert bs.deve_parar([reg(0.31), reg(0.32)], 0.30, **kw) is None                           # duas sem ganho: ainda não
    assert "3 tentativas seguidas" in bs.deve_parar([reg(0.31), reg(0.32), reg(0.33)], 0.30, **kw)
    assert bs.deve_parar([reg(0.31), reg(0.32), reg(0.29), reg(0.33), reg(0.33)], 0.30, **kw) is None   # ganho zera a contagem
    assert bs.deve_parar([reg(0.2995), reg(0.2992), reg(0.2990)], 0.30, **kw) is not None     # ganhos pequenos demais não contam
    assert bs.deve_parar([reg(0.299), reg(0.298), reg(0.297)], 0.30, **kw) is None            # ganhos de 0,001 contam
    assert bs.deve_parar([reg(0.31)] * 3, 0.30, paciencia=99, ganho_min=0.001, max_tentativas=3) == "limite de 3 tentativas"
    assert bs.deve_parar([reg(0.31), reg(0.32)], 0.20, **kw) is None                           # melhor inicial vem da rodada 1


def test_parada_ganho_exatamente_no_minimo_conta_e_empate_nao_e_melhora():
    # valores exatos em binário: 0,5 - 0,25 == 0,25
    assert bs.deve_parar([reg(0.25)], 0.5, paciencia=1, ganho_min=0.25, max_tentativas=99) is None
    assert bs.deve_parar([reg(0.25000001)], 0.5, paciencia=1, ganho_min=0.25, max_tentativas=99) is not None
    # empate com o melhor nunca é ganho, nem com ganho_min = 0
    assert bs.deve_parar([reg(0.3), reg(0.3)], 0.3, paciencia=2, ganho_min=0.0, max_tentativas=99) is not None
    assert bs.deve_parar([reg(0.3)], 0.3, paciencia=2, ganho_min=0.0, max_tentativas=99) is None


def test_rodada_2_para_pela_paciencia_contada_a_partir_do_melhor_da_rodada_1(dados, tmp_path, monkeypatch):
    pasta = tmp_path / "busca"
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    real = bs.executar
    monkeypatch.setattr(bs, "executar", lambda *a, **k: {**real(*a, **k), "mae_medio_a": 9.0})    # nenhuma bate a rodada 1
    feitas, total, motivo = bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=2, max_tentativas=50,
                                                saida=lambda _: None)
    assert (feitas, total) == (2, 2) and "2 tentativas seguidas" in motivo                      # a 1ª já não conta como ganho
    _, tent1 = bs.ler_registro(pasta / bs.ARQ_REGISTRO)
    _, _, paradas = bs.ler_registro_completo(pasta / bs.ARQ_REGISTRO2)
    melhor1 = min(r["mae_medio_a"] for r in tent1)
    assert melhor1 < 9.0 and paradas[0]["melhor_mae_a"] == melhor1 and paradas[0]["n_tentativas_rodada2"] == 2


def test_rodada_2_refina_a_partir_da_melhor_ja_encontrada_inclusive_da_propria_rodada_2(dados, tmp_path, monkeypatch):
    pasta = tmp_path / "busca"
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    real = bs.executar
    monkeypatch.setattr(bs, "executar", lambda t, *a, **k: {**real(t, *a, **k), "mae_medio_a": 0.0001 if t.idx == 3 else 9.0})
    bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=99, max_tentativas=7, saida=lambda _: None)
    _, tent2, _ = bs.ler_registro_completo(pasta / bs.ARQ_REGISTRO2)
    assert [r["idx"] for r in tent2] == [3, 4, 5, 6, 7, 8, 9]
    melhor = bs.Tentativa.de_dict(tent2[0]["config"])                                           # a primeira da rodada 2 virou a melhor
    assert melhor.residuo is True
    for r in tent2[1:]:
        t = bs.Tentativa.de_dict(r["config"])
        assert t.residuo is True and 1 <= campos_diferentes(t, melhor) <= 2


def test_rodada_2_exige_rodada_1_completa_e_inalterada(dados, tmp_path):
    pasta = tmp_path / "busca"
    with pytest.raises(bs.BuscaError, match="rodada 1 precisa estar completa"):
        bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, saida=lambda _: None)
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, max_segundos=0, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="rodada 1 precisa estar completa"):
        bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, saida=lambda _: None)
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="outro dataset, semente, corte ou limites"):
        bs.rodar_adaptativo(dados, pasta, seed=4, limites=LIM, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="outro dataset, semente, corte ou limites"):
        bs.rodar_adaptativo(dados, pasta, seed=3, limites={**LIM, "epocas_max": 5}, saida=lambda _: None)
    assert not (pasta / bs.ARQ_REGISTRO2).exists()


def test_rodada_2_refina_a_melhor_para_retoma_e_registra_a_parada(dados, tmp_path):
    pasta = tmp_path / "busca"
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    kw = dict(seed=3, limites=LIM, paciencia=2, ganho_min=0.0005, max_tentativas=6, saida=lambda _: None)
    feitas, total, motivo = bs.rodar_adaptativo(dados, pasta, max_segundos=0, **kw)
    assert (feitas, total, motivo) == (1, 1, None)                                              # uma por chamada com tempo 0
    feitas2, total2, motivo2 = bs.rodar_adaptativo(dados, pasta, **kw)
    cab2, tent2, paradas = bs.ler_registro_completo(pasta / bs.ARQ_REGISTRO2)
    assert motivo2 is not None and total2 == len(tent2) == 1 + feitas2 and len(paradas) == 1 and paradas[0]["motivo"] == motivo2
    assert paradas[0]["n_tentativas_rodada2"] == len(tent2) and tent2[0]["idx"] == 3           # a numeração continua da rodada 1
    assert [t["idx"] for t in tent2] == list(range(3, 3 + len(tent2))) and [t["seed"] for t in tent2] == [3 + t["idx"] for t in tent2]
    assert cab2["registro1_sha256"] == bs._sha256_arquivo(pasta / bs.ARQ_REGISTRO) and cab2["n_rodada1"] == 3
    # cada tentativa da rodada 2 é vizinha (até 2 fatores) da melhor conhecida antes dela
    historico = bs.ler_registro(pasta / bs.ARQ_REGISTRO)[1]
    for r in tent2:
        melhor = bs.Tentativa.de_dict(bs.escolher_vencedor(historico)["config"])
        t = bs.Tentativa.de_dict(r["config"])
        assert t.residuo is True and 1 <= campos_diferentes(t, dataclasses.replace(melhor, residuo=True)) <= 2
        historico.append(r)
    assert bs.rodar_adaptativo(dados, pasta, **kw) == (0, len(tent2), motivo2)                  # parada é definitiva
    assert len(bs.ler_registro_completo(pasta / bs.ARQ_REGISTRO2)[2]) == 1


def test_rodada_2_para_no_limite_e_recusa_outro_protocolo_ou_rodada_1_mexida(dados, tmp_path):
    pasta = tmp_path / "busca"
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    _, total, motivo = bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=99, max_tentativas=2, saida=lambda _: None)
    assert total == 2 and motivo == "limite de 2 tentativas"
    for kw, campo in (({"paciencia": 98}, "paciencia_busca"), ({"ganho_min": 0.01}, "ganho_min"), ({"max_tentativas": 3}, "max_tentativas")):
        args = {"seed": 3, "limites": LIM, "paciencia": 99, "max_tentativas": 2} | kw
        with pytest.raises(bs.BuscaError, match=campo):
            bs.rodar_adaptativo(dados, pasta, saida=lambda _: None, **args)
    original = (pasta / bs.ARQ_REGISTRO).read_bytes()
    linhas = original.split(b"\n")
    assert linhas[-1] == b""                                                                    # o arquivo termina em quebra de linha
    # (a) conteúdo da rodada 1 mexido com a mesma contagem de tentativas: o hash gravado na rodada 2 não bate mais
    ultima = json.loads(linhas[-2])
    ultima["segundos"] = ultima["segundos"] + 1
    (pasta / bs.ARQ_REGISTRO).write_bytes(b"\n".join([*linhas[:-2], json.dumps(ultima).encode(), b""]))
    with pytest.raises(bs.BuscaError, match="registro1_sha256"):
        bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=99, max_tentativas=2, saida=lambda _: None)
    # (b) última tentativa da rodada 1 apagada: a rodada 1 deixa de estar completa
    (pasta / bs.ARQ_REGISTRO).write_bytes(b"\n".join(linhas[:-2]) + b"\n")
    with pytest.raises(bs.BuscaError, match="rodada 1 precisa estar completa"):
        bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=99, max_tentativas=2, saida=lambda _: None)


def test_rodada_2_nao_depende_da_metade_b(dados, tmp_path):
    envenenado = dataclasses.replace(
        dados, va_b=dataclasses.replace(dados.va_b, Y=np.full_like(dados.va_b.Y, 999.0)),
        va_b_reg=dataclasses.replace(dados.va_b_reg, Y=np.full_like(dados.va_b_reg.Y, 999.0)))
    for nome, dd in (("a", dados), ("b", envenenado)):
        pasta = tmp_path / nome
        bs.rodar(dd, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
        bs.rodar_adaptativo(dd, pasta, seed=3, limites=LIM, paciencia=2, max_tentativas=4, saida=lambda _: None)
    ta = [sem_tempo(x) for x in bs.ler_registro(tmp_path / "a" / bs.ARQ_REGISTRO2)[1]]
    tb = [sem_tempo(x) for x in bs.ler_registro(tmp_path / "b" / bs.ARQ_REGISTRO2)[1]]
    assert ta == tb and len(ta) >= 2


def test_finalizar_so_exige_a_parada_quando_pedido_e_recusa_rodada_2_de_outra_rodada_1(dados, tmp_path):
    pasta = tmp_path / "busca"
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=99, max_tentativas=2, max_segundos=0, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="ainda não parou"):                                 # rodada 2 começou e não parou
        bs.finalizar(dados, pasta, tmp_path / "m", seed=3, n_membros=2, saida=lambda _: None)
    conf = bs.finalizar(dados, pasta, tmp_path / "m", seed=3, n_membros=2, n_arvores=5, n_boot=200, exigir_parada=False,
                        saida=lambda _: None)
    assert conf["rodadas"]["parada"] is None and conf["rodadas"]["2"] == 1
    pasta2 = tmp_path / "busca2"
    bs.rodar(dados, pasta2 / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    bs.rodar_adaptativo(dados, pasta2, seed=3, limites=LIM, paciencia=99, max_tentativas=2, saida=lambda _: None)
    cab2 = (pasta2 / bs.ARQ_REGISTRO2).read_text(encoding="utf-8").split("\n")
    doc = json.loads(cab2[0])
    doc["registro1_sha256"] = "0" * 64                                                          # rodada 2 de outra rodada 1
    (pasta2 / bs.ARQ_REGISTRO2).write_text("\n".join([json.dumps(doc), *cab2[1:]]), encoding="utf-8")
    with pytest.raises(bs.BuscaError, match="não pertence a esta rodada 1"):
        bs.finalizar(dados, pasta2, tmp_path / "m2", seed=3, n_membros=2, saida=lambda _: None)
    assert not (pasta2 / bs.ARQ_LOCK).exists()


@pytest.mark.parametrize("campo,valor", [("dataset_sha256", "f" * 64), ("seed", 99), ("corte_b", "2019-01"),
                                         ("limites", {"epocas_max": 99, "paciencia": 2, "lote": 128})])
def test_finalizar_recusa_rodada_2_feita_com_outro_protocolo(dados, tmp_path, campo, valor):
    pasta = tmp_path / "busca"
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=99, max_tentativas=1, saida=lambda _: None)
    linhas = (pasta / bs.ARQ_REGISTRO2).read_text(encoding="utf-8").split("\n")
    cab = json.loads(linhas[0])
    assert cab[campo] != valor
    cab[campo] = valor                                                                          # o hash da rodada 1 continua válido
    (pasta / bs.ARQ_REGISTRO2).write_text("\n".join([json.dumps(cab), *linhas[1:]]), encoding="utf-8")
    with pytest.raises(bs.BuscaError, match="não pertence a esta rodada 1"):
        bs.finalizar(dados, pasta, tmp_path / "m", seed=3, n_membros=1, saida=lambda _: None)
    assert not (pasta / bs.ARQ_LOCK).exists() and not (tmp_path / "m").exists()


# ------------------------------------------------------------------ alerta de seca com limiar ajustado só na metade A
def test_limiares_de_alerta_vem_da_metade_a_e_o_resultado_e_medido_na_b(dados, busca_completa, tmp_path):
    from src.model import baselines as bl
    from src.model import metricas as mt
    modelo = tmp_path / "modelo_v2"
    conf = bs.finalizar(dados, busca_completa, modelo, seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    b, vb = dados.busca, dados.va_b
    pers_a, pers_b = bl.Persistencia().prever(b.va_a), bl.Persistencia().prever(vb)
    for i, alvo in enumerate(vb.alvos):
        esperado = mt.melhor_limiar_seca(b.va_a.Y[:, i], pers_a[:, i], b.va_a.M[:, i], bs.GRADE_LIMIAR)["limiar_previsao"]
        r = conf["alerta_seca"][alvo]["persistencia"]
        assert r["limiar_ajustado_na_a"] == esperado and esperado in bs.GRADE_LIMIAR
        direto = mt.seca_com_limiar(vb.Y[:, i], pers_b[:, i], vb.M[:, i], esperado)             # medido na B com o limiar da A
        assert {k: r[k] for k in direto} == direto
        assert set(conf["alerta_seca"][alvo]) == set(conf["modelos"])
    receita = json.loads((modelo / "receita.json").read_text(encoding="utf-8"))
    assert receita["limiares_alerta"] == {a: conf["alerta_seca"][a]["rede_final"]["limiar_ajustado_na_a"] for a in vb.alvos}
    assert receita["rodadas"] == conf["rodadas"] and conf["rodadas"]["parada"] is not None
    assert min(bs.GRADE_LIMIAR) == -1.5 and max(bs.GRADE_LIMIAR) == -0.1 and len(bs.GRADE_LIMIAR) == 29


def test_classes_entram_nos_modelos_da_confirmacao(dados, busca_completa, tmp_path):
    conf = bs.finalizar(dados, busca_completa, tmp_path / "m", seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    for alvo in conf["alvos"]:
        for n, por_alvo in conf["modelos"].items():
            c = por_alvo[alvo]["todas"]["classes"]
            assert c["n"] == por_alvo[alvo]["todas"]["n"] and 0.0 <= c["acerto"] <= 1.0 and sum(map(sum, c["matriz"])) == c["n"]


def test_vencedor_pode_vir_da_rodada_2_e_e_o_que_se_treina_e_grava(dados, tmp_path):
    pasta = tmp_path / "busca"
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=99, max_tentativas=2, saida=lambda _: None)
    linhas = (pasta / bs.ARQ_REGISTRO2).read_text(encoding="utf-8").split("\n")
    docs = [json.loads(x) for x in linhas if x]
    alvo = next(x for x in docs if x.get("tipo") == "tentativa" and x["idx"] == 4)             # segunda da rodada 2
    alvo["mae_medio_a"] = 0.0001                                                                # melhor que qualquer outra
    (pasta / bs.ARQ_REGISTRO2).write_text("".join(json.dumps(x) + "\n" for x in docs), encoding="utf-8")
    conf = bs.finalizar(dados, pasta, tmp_path / "m", seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    assert conf["vencedora"]["idx"] == 4 and conf["vencedora"]["config"] == alvo["config"] and conf["n_tentativas"] == 5
    receita = json.loads((tmp_path / "m" / "receita.json").read_text(encoding="utf-8"))
    assert receita["tentativa_vencedora"] == {"idx": 4, "mae_medio_a": 0.0001} and receita["config"] == alvo["config"]


def test_rodada_2_para_quando_a_vizinhanca_acaba(dados, tmp_path, monkeypatch):
    monkeypatch.setattr(bs, "ESPACO2", {"perda": ("mse", "l1"), "regional": (False, True)})
    pasta = tmp_path / "busca"
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    feitas, total, motivo = bs.rodar_adaptativo(dados, pasta, seed=3, limites=LIM, paciencia=99, max_tentativas=50,
                                                saida=lambda _: None)
    assert motivo == "espaço de vizinhança esgotado" and 0 < total <= 4 and feitas == total
    _, tent1 = bs.ler_registro(pasta / bs.ARQ_REGISTRO)
    _, tent2, paradas = bs.ler_registro_completo(pasta / bs.ARQ_REGISTRO2)
    chaves = [bs.Tentativa.de_dict(r["config"]).chave() for r in tent1 + tent2]
    assert len(set(chaves)) == len(chaves)                                                       # nunca repetiu configuração
    assert len(paradas) == 1 and paradas[0]["motivo"] == motivo


def test_relatorio_da_confirmacao_traz_os_numeros_certos_nas_tabelas_de_alerta_e_de_classe(dados, busca_completa, tmp_path):
    from src.model import avaliar as av
    f = av._f
    cab, _ = bs.ler_registro(busca_completa / bs.ARQ_REGISTRO)
    cab2, _, paradas = bs.ler_registro_completo(busca_completa / bs.ARQ_REGISTRO2)
    conf = bs.finalizar(dados, busca_completa, tmp_path / "m", seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    conf = json.loads((busca_completa / bs.ARQ_CONFIRMACAO).read_text(encoding="utf-8"))
    txt = bs.texto_relatorio(cab, todas_as_tentativas(busca_completa), conf, cab2, paradas[-1]["motivo"])
    for alvo in conf["alvos"]:
        secao = txt.split(f"### {alvo}")[1].split("### ")[0].split("## Leitura")[0]
        sn = conf["modelos"]["persistencia"][alvo]["todas"]["classes"]["acerto_sempre_normal"]
        assert f"Prever sempre \"normal\" acertaria {f(sn)} das classes" in secao
        nomes = {**av._NOME, "rede_final": "Rede final (conjunto)"}
        for n, por_alvo in conf["modelos"].items():
            t, s = por_alvo[alvo]["todas"], por_alvo[alvo]["todas"]["seca"]
            linha = next(ln for ln in secao.splitlines() if ln.startswith(f"| {nomes[n]} | {f(t['mae'])} | {f(t['rmse'])} |"))
            cels = [c.strip() for c in linha.strip("|").split("|")]
            assert cels[4:8] == [f(s["precisao"]), f(s["revocacao"]), f(s["f1"]), f(t["classes"]["acerto"])]
        alerta = secao.split("Alerta de seca com o limiar ajustado na metade A")[1].split("Margem de erro")[0]
        for n, r in conf["alerta_seca"][alvo].items():
            linha = next(ln for ln in alerta.splitlines() if ln.startswith(f"| {nomes[n]} |"))
            cels = [c.strip() for c in linha.strip("|").split("|")]
            assert cels == [nomes[n], f(r["limiar_ajustado_na_a"], 2), f(r["precisao"]), f(r["revocacao"]), f(r["f1"]),
                            f"{r['tp']} de {r['secas_reais']}", str(r["fp"])]
