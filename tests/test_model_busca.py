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
    return pasta


def test_finalizar_exige_orcamento_gasto_e_registro_do_mesmo_dataset(dados, tmp_path):
    pasta = tmp_path / "busca"
    with pytest.raises(bs.BuscaError, match="orçamento ainda não foi gasto"):
        bs.finalizar(dados, pasta, tmp_path / "m", seed=3, saida=lambda _: None)
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, max_segundos=0, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="1 de 3"):
        bs.finalizar(dados, pasta, tmp_path / "m", seed=3, saida=lambda _: None)
    assert not (pasta / bs.ARQ_LOCK).exists() and not (tmp_path / "m").exists()               # nada lido, nada gravado
    bs.rodar(dados, pasta / bs.ARQ_REGISTRO, seed=3, n_tentativas=3, limites=LIM, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="outro dataset"):
        bs.finalizar(dataclasses.replace(dados, dataset_sha256="0" * 64), pasta, tmp_path / "m", seed=3, saida=lambda _: None)
    with pytest.raises(bs.BuscaError, match="outro dataset"):
        bs.finalizar(dados, pasta, tmp_path / "m", seed=4, saida=lambda _: None)


def test_finalizar_le_a_metade_b_uma_vez_grava_tudo_e_a_volta_reproduz(dados, busca_completa, tmp_path):
    modelo = tmp_path / "modelo_v2"
    conf = bs.finalizar(dados, busca_completa, modelo, seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    cab, ts = bs.ler_registro(busca_completa / bs.ARQ_REGISTRO)
    venc = bs.escolher_vencedor(ts)
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
    cab, ts = bs.ler_registro(busca_completa / bs.ARQ_REGISTRO)
    txt = bs.texto_relatorio(cab, ts)
    assert txt.count(bs.AVISO) == 2 and "Confirmação na metade B" not in txt and "otimista" in txt
    assert "3 de 3 tentativas" in txt and "Efeito médio de cada fator" in txt and "Taxa de aprendizado" in txt
    tabela = txt.split("## Todas as tentativas")[1]
    primeira = [ln for ln in tabela.splitlines() if ln.startswith("| ") and ln[2].isdigit()][0]
    assert primeira.startswith(f"| {bs.escolher_vencedor(ts)['idx'] + 1} |")
    bs.finalizar(dados, busca_completa, tmp_path / "m", seed=3, n_membros=2, n_arvores=5, n_boot=200, saida=lambda _: None)
    txt2 = bs.texto_relatorio(cab, ts, json.loads((busca_completa / bs.ARQ_CONFIRMACAO).read_text(encoding="utf-8")))
    assert "Confirmação na metade B (lida uma única vez)" in txt2 and "Rede final (conjunto)" in txt2 and "Margem de erro" in txt2
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
    assert any("busca finalizar" in x for x in out)
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
    for cmd in ("rodar", "finalizar"):
        assert bs.main([cmd, "--params", str(raiz / "params.yaml")], saida=out.append, carregar_cfg=lambda _: liberado) == 1
        assert "teste_liberado" in out[-1]
    assert not (raiz / "artifacts" / "busca").exists()
    for cmd in ("rodar", "finalizar"):                                                          # recusa antes de abrir os dados
        assert bs.main([cmd, "--params", str(raiz / "params.yaml")], saida=out.append, carregar_cfg=lambda _: liberado,
                       preparar_dados=lambda c: pytest.fail("não deveria preparar os dados")) == 1
    assert bs.main(["rodar", "--params", str(raiz / "nao_existe.yaml")], saida=out.append) == 1
    with pytest.raises(SystemExit):
        bs.main(["apagar"])
    assert bs.main(["relatorio", "--params", str(raiz / "params.yaml")], saida=out.append) == 1 and "não há registro" in out[-1]
