"""Busca de configurações da rede com orçamento fixo, registro e confirmação única (Item 9).

PROTOCOLO (declarado antes de rodar; mudar qualquer item invalida a comparação com o registro):
 1. A validação é dividida pelo mês-alvo em duas metades: A (escolha) e B (confirmação).
 2. A busca tem orçamento FIXO de `N_TENTATIVAS`: duas referências e o restante sorteado com semente fixa
    sobre `ESPACO`. Não para na primeira configuração que parecer boa e não aceita tentativa a mais.
 3. Cada tentativa treina do zero no treino, escolhe a época na metade A e é pontuada na metade A. A metade B
    não existe para a função da busca: `DadosBusca` não a contém.
 4. Vencedor = menor MAE médio das cabeças na metade A. Desempate: menor `idx`.
 5. Só com as N tentativas no registro, `finalizar` treina um conjunto de redes (sementes diferentes) com a
    configuração vencedora e lê a metade B UMA vez, contra todos os baselines (também ajustados com o treino e
    com a metade A). O arquivo `confirmacao_B.lock` marca a leitura: uma segunda é recusada.
 6. O teste congelado nunca é aberto aqui.

Por que assim: com dezenas de tentativas avaliadas nos mesmos dados, a melhor delas parece boa por acaso. A
metade B não participou de nenhuma escolha, então mostra quanto do ganho sobrevive.

Uso:
    python -m src.model.busca rodar [--max-segundos 480]
    python -m src.model.busca finalizar
    python -m src.model.busca relatorio
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import Config, load_config
from src.model import avaliar as av
from src.model import baselines as bl
from src.model import entrada
from src.model import features as fe
from src.model import metricas as mt
from src.model import residuo as rs
from src.model import treino as tn

PROTOCOLO = 1
N_TENTATIVAS = 40
N_MEMBROS = 5
ESPACO = {"perda": ("mse", "l1", "huber"),
          "ocultas": ((16,), (32,), (64, 32), (128, 64), (32, 16)),
          "dropout": (0.0, 0.1, 0.3),
          "weight_decay": (1e-4, 1e-3, 1e-2),
          "lr": (1e-3, 3e-3)}
P_RESIDUO, P_REGIONAL = 0.75, 0.5          # chance de cada sorteio usar a correção sobre o baseline / o contexto regional
LIMITES_PADRAO = {"epocas_max": 80, "paciencia": 10, "lote": 512}
Q_MARGEM = 0.80
SOMA_ESTACAO = 0.05                         # piso do desvio entre as redes ao normalizar a margem (evita dividir por ~0)
CHUVOSA = frozenset({11, 12, 1, 2, 3, 4})   # meses-alvo da estação chuvosa do semiárido, para a margem por estação
ARQ_REGISTRO = "registro.jsonl"
ARQ_LOCK = "confirmacao_B.lock"
ARQ_CONFIRMACAO = "confirmacao_B.json"
ARQ_RELATORIO = "relatorio.md"
AVISO = ("MVP experimental, em calibração com S2iD/TCE-PI. Métricas de validação; o teste congelado não foi lido. "
         "A metade B da validação foi lida uma única vez, ao final.")


class BuscaError(RuntimeError):
    """Protocolo violado, registro inválido ou dados incompatíveis com a busca."""


# ---------------------------------------------------------------- espaço e sorteio
@dataclass(frozen=True)
class Tentativa:
    idx: int
    residuo: bool
    regional: bool
    perda: str
    ocultas: tuple[int, ...]
    dropout: float
    weight_decay: float
    lr: float

    def chave(self) -> tuple:
        return tuple(getattr(self, f.name) for f in fields(self) if f.name != "idx")

    def para_dict(self) -> dict:
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        d["ocultas"] = list(self.ocultas)
        return d

    @classmethod
    def de_dict(cls, d: dict) -> "Tentativa":
        try:
            return cls(idx=int(d["idx"]), residuo=bool(d["residuo"]), regional=bool(d["regional"]), perda=str(d["perda"]),
                       ocultas=tuple(int(x) for x in d["ocultas"]), dropout=float(d["dropout"]),
                       weight_decay=float(d["weight_decay"]), lr=float(d["lr"]))
        except (KeyError, TypeError, ValueError) as e:
            raise BuscaError(f"tentativa inválida no registro: {e}") from e

    def hiperparametros(self, limites: dict) -> tn.Hiperparametros:
        return tn.Hiperparametros(ocultas=self.ocultas, dropout=self.dropout, lr=self.lr, weight_decay=self.weight_decay,
                                  perda=self.perda, **limites)


def _referencias() -> list[Tentativa]:
    padrao = dict(regional=False, perda="mse", ocultas=(64, 32), dropout=0.1, weight_decay=1e-4, lr=1e-3)
    return [Tentativa(idx=0, residuo=False, **padrao), Tentativa(idx=1, residuo=True, **padrao)]


def tamanho_do_espaco() -> int:
    n = 4                                                           # residuo x regional
    for v in ESPACO.values():
        n *= len(v)
    return n


def sortear(n: int, seed: int) -> list[Tentativa]:
    """As `n` configurações da busca: duas referências e o resto sorteado (sem repetir) com `seed`."""
    if not (1 <= n <= tamanho_do_espaco()):
        raise BuscaError(f"n deve estar entre 1 e {tamanho_do_espaco()} (recebi {n})")
    saida = _referencias()[:n]
    vistos = {t.chave() for t in saida}
    rng = np.random.default_rng(seed)
    while len(saida) < n:
        t = Tentativa(idx=len(saida), residuo=bool(rng.random() < P_RESIDUO), regional=bool(rng.random() < P_REGIONAL),
                      perda=ESPACO["perda"][int(rng.integers(len(ESPACO["perda"])))],
                      ocultas=tuple(ESPACO["ocultas"][int(rng.integers(len(ESPACO["ocultas"])))]),
                      dropout=float(ESPACO["dropout"][int(rng.integers(len(ESPACO["dropout"])))]),
                      weight_decay=float(ESPACO["weight_decay"][int(rng.integers(len(ESPACO["weight_decay"])))]),
                      lr=float(ESPACO["lr"][int(rng.integers(len(ESPACO["lr"])))]))
        if t.chave() not in vistos:
            vistos.add(t.chave())
            saida.append(t)
    return saida


def hash_do_espaco() -> str:
    doc = {"espaco": {k: [list(x) if isinstance(x, tuple) else x for x in v] for k, v in ESPACO.items()},
           "p_residuo": P_RESIDUO, "p_regional": P_REGIONAL, "referencias": [t.para_dict() for t in _referencias()]}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- dados e metades da validação
@dataclass(frozen=True, eq=False)
class DadosBusca:
    """Tudo o que a busca enxerga: treino, metade A e o baseline parcial. A metade B não está aqui."""
    tr: fe.Conjunto
    tr_reg: fe.Conjunto
    va_a: fe.Conjunto
    va_a_reg: fe.Conjunto
    base: bl.ParcialClimatologica


@dataclass(frozen=True, eq=False)
class Dados:
    busca: DadosBusca
    va_b: fe.Conjunto
    va_b_reg: fe.Conjunto
    params_gama: pd.DataFrame
    dataset_sha256: str
    corte_b: str                       # primeiro mês-alvo da metade B, "AAAA-MM"


def meia_validacao(meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.Timestamp]:
    """Máscaras (A, B) e o primeiro mês da metade B. A = meses-alvo antes da mediana dos meses; B = a partir dela."""
    mes = pd.to_datetime(meta["mes_alvo"])
    g = (mes.dt.year * 12 + mes.dt.month).to_numpy()
    u = np.unique(g)
    if len(u) < 2:
        raise BuscaError("a validação precisa de pelo menos dois meses-alvo para ter metades A e B")
    corte = u[len(u) // 2]
    b = g >= corte
    return ~b, b, mes[b].min()


def preparar(cfg: Config) -> Dados:
    """Lê os dados verificados e separa treino, metade A e metade B. Recusa `teste_liberado: true`."""
    if cfg.split.teste_liberado:
        raise BuscaError("a busca recusa rodar com split.teste_liberado: true")
    ins = entrada.carregar_insumos(cfg)
    conj = fe.construir(ins.demo, ins.serie_imputada, ins.climatologia, ins.municipios, cfg)
    conj_reg = rs.adicionar_regional(conj)
    sel = lambda c, b: fe.selecionar(c, b, cfg, so_com_rotulo=True)          # noqa: E731
    val, val_reg = sel(conj, "val"), sel(conj_reg, "val")
    a, b, corte = meia_validacao(val.meta)
    busca = DadosBusca(tr=sel(conj, "train"), tr_reg=sel(conj_reg, "train"), va_a=val.fatiar(a), va_a_reg=val_reg.fatiar(a),
                       base=bl.ParcialClimatologica(ins.params_gama))
    return Dados(busca=busca, va_b=val.fatiar(b), va_b_reg=val_reg.fatiar(b), params_gama=ins.params_gama,
                 dataset_sha256=ins.dataset_sha256, corte_b=str(corte)[:7])


# ---------------------------------------------------------------- uma tentativa
def _treinar_membro(t: Tentativa, d: DadosBusca, seed: int, limites: dict):
    """Treina do zero uma rede com a configuração `t`; devolve (preditor completo, resultado do treino)."""
    tr, va = (d.tr_reg, d.va_a_reg) if t.regional else (d.tr, d.va_a)
    if t.residuo:
        tr_in, va_in = rs.com_base(tr, d.base.prever(tr)), rs.com_base(va, d.base.prever(va))
    else:
        tr_in, va_in = tr, va
    pad = fe.Padronizador.ajustar(tr_in)
    res = tn.treinar(tr_in, va_in, pad, seed=seed, hp=t.hiperparametros(limites))
    pred = tn.RedePreditor(res.rede, pad)
    return (rs.ResiduoPreditor(d.base, pred) if t.residuo else pred), res


def _erros_por_cabeca(c: fe.Conjunto, P: np.ndarray) -> list[dict]:
    return [mt.erros(c.Y[:, i], P[:, i], c.M[:, i]) for i in range(len(c.alvos))]


def executar(t: Tentativa, d: DadosBusca, seed: int, limites: dict | None = None) -> dict:
    """Treina e pontua UMA tentativa na metade A. Devolve o registro (sem a metade B)."""
    limites = limites or LIMITES_PADRAO
    t0 = time.perf_counter()
    pred, res = _treinar_membro(t, d, seed, limites)
    va = d.va_a_reg if t.regional else d.va_a
    e = _erros_por_cabeca(va, pred.prever(va))
    maes = [x["mae"] for x in e]
    return {"tipo": "tentativa", "idx": t.idx, "config": t.para_dict(), "seed": seed, "melhor_epoca": res.melhor_epoca,
            "epocas_rodadas": len(res.historico), "mae_a": maes, "rmse_a": [x["rmse"] for x in e],
            "vies_a": [x["vies"] for x in e], "mae_medio_a": float(np.mean(maes)),
            "segundos": round(time.perf_counter() - t0, 2)}


def referencias_a(d: DadosBusca) -> dict:
    """MAE na metade A da persistência e do baseline parcial: o piso que a busca quer bater."""
    saida = {}
    for nome, b in (("persistencia", bl.Persistencia()), ("parcial_climatologica", d.base)):
        maes = [x["mae"] for x in _erros_por_cabeca(d.va_a, b.prever(d.va_a))]
        saida[nome] = {"mae_a": maes, "mae_medio_a": float(np.mean(maes))}
    return saida


# ---------------------------------------------------------------- registro (JSONL, retomável)
def cabecalho(d: Dados | DadosBusca, dataset_sha256: str, corte_b: str, seed: int, n_tentativas: int, limites: dict) -> dict:
    busca = d.busca if isinstance(d, Dados) else d
    return {"tipo": "cabecalho", "protocolo": PROTOCOLO, "dataset_sha256": dataset_sha256, "seed": seed,
            "n_tentativas": n_tentativas, "espaco_sha256": hash_do_espaco(), "limites": dict(limites),
            "corte_b": corte_b, "n_treino": len(busca.tr), "n_a": len(busca.va_a), "referencias_a": referencias_a(busca)}


_CAMPOS_FIXOS = ("protocolo", "dataset_sha256", "seed", "n_tentativas", "espaco_sha256", "limites", "corte_b")


def ler_registro(caminho, *, reparar: bool = False) -> tuple[dict | None, list[dict]]:
    """(cabeçalho, tentativas). Uma última linha cortada (queda no meio da escrita) é descartada; com `reparar`
    ela também é removida do arquivo. Qualquer outra linha inválida é erro."""
    p = Path(caminho)
    if not p.exists():
        return None, []
    bruto = p.read_bytes()
    linhas = bruto.split(b"\n")
    cortada = linhas[-1] != b""                                  # sem \n final: a última linha pode estar pela metade
    docs: list[dict] = []
    for i, ln in enumerate(linhas):
        if ln == b"":
            continue
        try:
            docs.append(json.loads(ln.decode("utf-8")))
        except (ValueError, UnicodeDecodeError) as e:
            if cortada and i == len(linhas) - 1:
                if reparar:
                    p.write_bytes(bruto[: len(bruto) - len(ln)])
                continue
            raise BuscaError(f"linha {i + 1} do registro inválida: {e}") from e
    cab = [x for x in docs if x.get("tipo") == "cabecalho"]
    tent = [x for x in docs if x.get("tipo") == "tentativa"]
    if len(cab) > 1 or (tent and not cab) or (docs and docs[0].get("tipo") != "cabecalho"):
        raise BuscaError("registro sem cabeçalho único na primeira linha")
    idxs = [x["idx"] for x in tent]
    if len(set(idxs)) != len(idxs):
        raise BuscaError("tentativa repetida no registro")
    return (cab[0] if cab else None), tent


def _anexar(caminho: Path, doc: dict) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("ab") as f:
        f.write((json.dumps(av._limpar(doc), ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def rodar(d: Dados, caminho, *, seed: int, n_tentativas: int = N_TENTATIVAS, limites: dict | None = None,
          max_segundos: float | None = None, saida: Callable[[str], None] = print) -> tuple[int, int]:
    """Executa as tentativas que faltam. Devolve (feitas agora, total no registro). Retoma de onde parou."""
    limites = dict(limites or LIMITES_PADRAO)
    caminho = Path(caminho)
    novo = cabecalho(d, d.dataset_sha256, d.corte_b, seed, n_tentativas, limites)
    cab, feitas = ler_registro(caminho, reparar=True)
    if cab is None:
        _anexar(caminho, novo)
    else:
        diff = [k for k in _CAMPOS_FIXOS if cab.get(k) != novo[k]]
        if diff:
            raise BuscaError(f"o registro existente foi feito com outro protocolo ou dataset ({diff}); "
                             "apague-o conscientemente para recomeçar")
    ja = {x["idx"] for x in feitas}
    inicio, agora = time.perf_counter(), 0
    for t in sortear(n_tentativas, seed):
        if t.idx in ja:
            continue
        if max_segundos is not None and agora > 0 and time.perf_counter() - inicio >= max_segundos:
            break
        reg = executar(t, d.busca, seed + t.idx, limites)
        _anexar(caminho, reg)
        agora += 1
        saida(f"tentativa {t.idx + 1:>2}/{n_tentativas} | MAE médio na metade A {reg['mae_medio_a']:.4f} "
              f"(SPI-3 {reg['mae_a'][0]:.3f}, SPI-6 {reg['mae_a'][1]:.3f}) | época {reg['melhor_epoca']} | {reg['segundos']} s")
    return agora, len(ja) + agora


def escolher_vencedor(tentativas: list[dict]) -> dict:
    """Menor MAE médio na metade A; empate resolvido pelo menor idx. Só olha campos da metade A."""
    if not tentativas:
        raise BuscaError("registro sem tentativas")
    return min(tentativas, key=lambda r: (r["mae_medio_a"], r["idx"]))


# ---------------------------------------------------------------- final: conjunto de redes e metade B
def _margens(yA, pA, mA, mesA, sA, yB, pB, mB, mesB, sB, q: float) -> dict:
    """Margens calibradas na metade A e verificadas na B, três jeitos declarados: global, por estação e
    proporcional à dispersão entre as redes. Devolve valores e a cobertura obtida na B."""
    def cobertura(margem_b, m=mB):
        return float((np.abs(pB - yB)[m] <= margem_b[m]).mean()) if m.any() else float("nan")
    erro_a, erro_b = np.abs(pA - yA), np.abs(pB - yB)
    glob = float(np.quantile(erro_a[mA], q))
    chuvosa_a, chuvosa_b = np.isin(mesA, list(CHUVOSA)), np.isin(mesB, list(CHUVOSA))
    est = {}
    for nome, sel_a in (("chuvosa", chuvosa_a), ("seca", ~chuvosa_a)):
        est[nome] = float(np.quantile(erro_a[mA & sel_a], q)) if (mA & sel_a).any() else glob
    margem_est_b = np.where(chuvosa_b, est["chuvosa"], est["seca"])
    razao = float(np.quantile((erro_a / (sA + SOMA_ESTACAO))[mA], q))
    margem_prop_b = razao * (sB + SOMA_ESTACAO)
    return {"q": q,
            "global": {"valor": glob, "cobertura_b": cobertura(np.full(len(pB), glob)),
                       "cobertura_b_chuvosa": cobertura(np.full(len(pB), glob), mB & chuvosa_b),
                       "cobertura_b_seca": cobertura(np.full(len(pB), glob), mB & ~chuvosa_b)},
            "por_estacao": {"valor": est, "cobertura_b": cobertura(margem_est_b),
                            "cobertura_b_chuvosa": cobertura(margem_est_b, mB & chuvosa_b),
                            "cobertura_b_seca": cobertura(margem_est_b, mB & ~chuvosa_b)},
            "proporcional_a_dispersao": {"razao": razao, "cobertura_b": cobertura(margem_prop_b),
                                         "cobertura_b_chuvosa": cobertura(margem_prop_b, mB & chuvosa_b),
                                         "cobertura_b_seca": cobertura(margem_prop_b, mB & ~chuvosa_b),
                                         "margem_media_b": float(margem_prop_b[mB].mean()) if mB.any() else float("nan")}}


def finalizar(d: Dados, pasta_busca, pasta_modelo, *, seed: int, n_membros: int = N_MEMBROS, n_arvores: int = 100,
              n_boot: int = 2000, saida: Callable[[str], None] = print) -> dict:
    """Treina o conjunto do vencedor, lê a metade B uma única vez e grava pesos, receita e confirmação."""
    pasta_busca, pasta_modelo = Path(pasta_busca), Path(pasta_modelo)
    cab, tentativas = ler_registro(pasta_busca / ARQ_REGISTRO)
    if cab is None or len(tentativas) != cab["n_tentativas"]:
        raise BuscaError(f"o orçamento ainda não foi gasto: {len(tentativas)} de {cab['n_tentativas'] if cab else '?'} "
                         "tentativas no registro (rode `busca rodar` até completar)")
    if (cab["dataset_sha256"], cab["seed"], cab["corte_b"]) != (d.dataset_sha256, seed, d.corte_b):
        raise BuscaError("o registro foi feito com outro dataset, semente ou corte da validação")
    if (pasta_busca / ARQ_LOCK).exists() or (pasta_busca / ARQ_CONFIRMACAO).exists():
        raise BuscaError("a metade B já foi lida (confirmacao_B.lock/json existe): uma segunda leitura invalidaria "
                         "a confirmação")
    venc = escolher_vencedor(tentativas)
    t = Tentativa.de_dict(venc["config"])
    limites = cab["limites"]
    busca = d.busca
    saida(f"vencedor da busca: tentativa {t.idx + 1} (MAE médio na metade A {venc['mae_medio_a']:.4f}); "
          f"treinando {n_membros} redes")
    preditores, resultados = [], []
    for k in range(n_membros):
        p, r = _treinar_membro(t, busca, seed + 10_000 + k, limites)
        preditores.append(p)
        resultados.append(r)
    conjunto = rs.MediaDePreditores(preditores)
    va_a = busca.va_a_reg if t.regional else busca.va_a
    pad_plain = fe.Padronizador.ajustar(busca.tr)
    bases = bl.ajustar_todos(busca.tr, busca.va_a, pad_plain, d.params_gama, seed=seed, n_arvores=n_arvores)
    pA, sA = conjunto.prever(va_a), conjunto.desvio(va_a)
    # --- a partir daqui a metade B é aberta, uma única vez
    pasta_busca.mkdir(parents=True, exist_ok=True)
    (pasta_busca / ARQ_LOCK).write_text(f"metade B lida em {time.strftime('%Y-%m-%d %H:%M:%S')}\n", encoding="utf-8")
    va_b = d.va_b_reg if t.regional else d.va_b
    prev = {n: b.prever(d.va_b) for n, b in bases.items()}
    pB, sB = conjunto.prever(va_b), conjunto.desvio(va_b)
    prev["rede_final"] = pB
    corpo = av.pontuar(d.va_b, prev, principal="rede_final", n_boot=n_boot, seed=seed)
    mes_a = pd.to_datetime(va_a.meta["mes_alvo"]).dt.month.to_numpy()
    mes_b = pd.to_datetime(va_b.meta["mes_alvo"]).dt.month.to_numpy()
    margens = {alvo: _margens(va_a.Y[:, i], pA[:, i], va_a.M[:, i], mes_a, sA[:, i], d.va_b.Y[:, i], pB[:, i],
                              d.va_b.M[:, i], mes_b, sB[:, i], Q_MARGEM) for i, alvo in enumerate(d.va_b.alvos)}
    shas = []
    for k, r in enumerate(resultados):
        shas.append(tn.salvar_resultado(r, pasta_modelo / f"membro_{k}", d.dataset_sha256))
    receita = {"aviso": AVISO, "protocolo": PROTOCOLO, "dataset_sha256": d.dataset_sha256, "seed": seed,
               "config": t.para_dict(), "limites": limites, "n_membros": n_membros,
               "seeds_membros": [seed + 10_000 + k for k in range(n_membros)], "pesos_sha256": shas,
               "colunas": list(resultados[0].colunas), "alvos": list(resultados[0].alvos), "corte_b": d.corte_b,
               "margens_a": {alvo: m["global"]["valor"] for alvo, m in margens.items()},
               "margem_por_estacao_a": {alvo: m["por_estacao"]["valor"] for alvo, m in margens.items()},
               "razao_dispersao_a": {alvo: m["proporcional_a_dispersao"]["razao"] for alvo, m in margens.items()},
               "tentativa_vencedora": {"idx": venc["idx"], "mae_medio_a": venc["mae_medio_a"]}}
    pasta_modelo.mkdir(parents=True, exist_ok=True)
    (pasta_modelo / "receita.json").write_text(json.dumps(av._limpar(receita), ensure_ascii=False, indent=2, allow_nan=False)
                                               + "\n", encoding="utf-8")
    conf = {"aviso": AVISO, "dataset_sha256": d.dataset_sha256, "seed": seed, "corte_b": d.corte_b, "n_linhas_b": len(d.va_b),
            "vencedora": {"idx": venc["idx"], "config": venc["config"], "mae_medio_a": venc["mae_medio_a"]},
            "n_tentativas": len(tentativas), "n_membros": n_membros, "alvos": list(d.va_b.alvos), "margens": margens,
            "baselines": {"n_arvores": n_arvores}, "bootstrap": {"n_boot": n_boot, "nivel": 0.95}, **corpo}
    arq = pasta_busca / ARQ_CONFIRMACAO
    arq.write_text(json.dumps(av._limpar(conf), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return conf


def carregar_final(pasta_modelo, base: bl.ParcialClimatologica) -> rs.MediaDePreditores:
    """Reconstrói o conjunto salvo. Para configurações com contexto regional, preveja sobre o conjunto com
    `residuo.adicionar_regional` aplicado."""
    pasta = Path(pasta_modelo)
    try:
        receita = json.loads((pasta / "receita.json").read_text(encoding="utf-8"))
        n, residuo = int(receita["n_membros"]), bool(receita["config"]["residuo"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise BuscaError(f"receita.json ausente ou inválida: {e}") from e
    membros = []
    for k in range(n):
        p, meta = tn.carregar_preditor(pasta / f"membro_{k}")
        if meta.get("dataset_sha256") != receita.get("dataset_sha256"):
            raise BuscaError(f"membro_{k} pertence a outro dataset_sha256")
        membros.append(rs.ResiduoPreditor(base, p) if residuo else p)
    return rs.MediaDePreditores(membros)


# ---------------------------------------------------------------- relatório
_FATORES = (("residuo", "Correção sobre o baseline"), ("regional", "Contexto regional"), ("perda", "Perda"),
            ("ocultas", "Camadas ocultas"), ("dropout", "Dropout"), ("weight_decay", "Weight decay"), ("lr", "Taxa de aprendizado"))


def _rot(v) -> str:
    return "sim" if v is True else "não" if v is False else str(tuple(v) if isinstance(v, list) else v)


def texto_relatorio(cab: dict, tentativas: list[dict], conf: dict | None = None) -> str:
    f = av._f
    ref = cab["referencias_a"]
    L = ["# Busca de configurações da rede", "", f"> {AVISO}", "",
         f"- Protocolo {cab['protocolo']}: {len(tentativas)} de {cab['n_tentativas']} tentativas (orçamento fixo), sorteio com semente "
         f"{cab['seed']}; escolha só na metade A da validação ({cab['n_a']} linhas, meses-alvo antes de {cab['corte_b']}).",
         f"- dataset_sha256 `{cab['dataset_sha256'][:16]}…`. Limites do treino: {cab['limites']}.",
         f"- Piso na metade A (MAE médio das cabeças): persistência {f(ref['persistencia']['mae_medio_a'])}, "
         f"baseline parcial + climatologia {f(ref['parcial_climatologica']['mae_medio_a'])}.", ""]
    if tentativas:
        venc = escolher_vencedor(tentativas)
        c = venc["config"]
        L += [f"## Vencedora na metade A: tentativa {venc['idx'] + 1}", "",
              f"MAE médio {f(venc['mae_medio_a'], 4)} (SPI-3 {f(venc['mae_a'][0], 4)}, SPI-6 {f(venc['mae_a'][1], 4)}). "
              f"Correção sobre o baseline: {_rot(c['residuo'])}; contexto regional: {_rot(c['regional'])}; perda {c['perda']}; "
              f"camadas {_rot(c['ocultas'])}; dropout {c['dropout']}; weight decay {c['weight_decay']}; lr {c['lr']}.", "",
              "A tentativa vencedora foi escolhida entre várias nos mesmos dados: o MAE dela na metade A é otimista. "
              "O número que vale é o da metade B.", "", "## Efeito médio de cada fator (MAE médio na metade A)", "",
              "| Fator | Valor | Tentativas | MAE médio | Melhor |", "|---|---|---|---|---|"]
        for chave, nome in _FATORES:
            grupos = defaultdict(list)
            for r in tentativas:
                grupos[_rot(r["config"][chave])].append(r["mae_medio_a"])
            for val, xs in sorted(grupos.items()):
                L.append(f"| {nome} | {val} | {len(xs)} | {f(float(np.mean(xs)))} | {f(min(xs))} |")
        L += ["", "## Todas as tentativas (da melhor para a pior na metade A)", "",
              "| # | Resíduo | Regional | Perda | Camadas | Dropout | WD | lr | Época | MAE SPI-3 | MAE SPI-6 | MAE médio |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(tentativas, key=lambda r: (r["mae_medio_a"], r["idx"])):
            c = r["config"]
            L.append(f"| {r['idx'] + 1} | {_rot(c['residuo'])} | {_rot(c['regional'])} | {c['perda']} | {_rot(c['ocultas'])} | "
                     f"{c['dropout']} | {c['weight_decay']} | {c['lr']} | {r['melhor_epoca']} | {f(r['mae_a'][0])} | "
                     f"{f(r['mae_a'][1])} | {f(r['mae_medio_a'], 4)} |")
        L.append("")
    if conf is not None:
        L += ["## Confirmação na metade B (lida uma única vez)", "",
              f"{conf['n_linhas_b']} linhas, meses-alvo a partir de {conf['corte_b']}. Conjunto de {conf['n_membros']} redes da "
              f"configuração vencedora. Baselines ajustados no treino (Ridge com alfa escolhido na metade A).", ""]
        for alvo in conf["alvos"]:
            L += [f"### {alvo}", "", "| Modelo | MAE | RMSE | Viés | Seca: precisão | Seca: revocação | Seca: F1 |", "|---|---|---|---|---|---|---|"]
            nomes = {**av._NOME, "rede_final": "Rede final (conjunto)"}
            for n, por_alvo in conf["modelos"].items():
                t = por_alvo[alvo]["todas"]
                s = t["seca"]
                L.append(f"| {nomes.get(n, n)} | {f(t['mae'])} | {f(t['rmse'])} | {f(t['vies'])} | {f(s['precisao'])} | "
                         f"{f(s['revocacao'])} | {f(s['f1'])} |")
            L.append("")
            for rot, titulo in (("persistencia", "persistência"), ("melhor_baseline", "melhor baseline")):
                r = conf["comparacao"][alvo][rot]
                L.append(f"- Rede final − {nomes.get(r['contra'], r['contra'])} ({titulo}): diferença de MAE {f(r['dif'], 4)} "
                         f"[{f(r['ic_inf'], 4)}; {f(r['ic_sup'], 4)}] (95%, bootstrap de {r['n_grupos']} meses): "
                         f"{av._VEREDITO[r['veredito']]}.")
            m = conf["margens"][alvo]
            L += ["", f"Margem de erro ({int(round(m['q'] * 100))}% dos erros na metade A), cobertura obtida na metade B:", "",
                  "| Margem | Valor | Cobertura B | Estação chuvosa | Estação seca |", "|---|---|---|---|---|",
                  f"| Global | ±{f(m['global']['valor'])} | {f(m['global']['cobertura_b'])} | {f(m['global']['cobertura_b_chuvosa'])} | "
                  f"{f(m['global']['cobertura_b_seca'])} |",
                  f"| Por estação | chuvosa ±{f(m['por_estacao']['valor']['chuvosa'])}, seca ±{f(m['por_estacao']['valor']['seca'])} | "
                  f"{f(m['por_estacao']['cobertura_b'])} | {f(m['por_estacao']['cobertura_b_chuvosa'])} | "
                  f"{f(m['por_estacao']['cobertura_b_seca'])} |",
                  f"| Proporcional à dispersão do conjunto | ±{f(m['proporcional_a_dispersao']['margem_media_b'])} em média | "
                  f"{f(m['proporcional_a_dispersao']['cobertura_b'])} | {f(m['proporcional_a_dispersao']['cobertura_b_chuvosa'])} | "
                  f"{f(m['proporcional_a_dispersao']['cobertura_b_seca'])} |", ""]
        L += ["## Leitura", "",
              f"- Rede final supera a persistência em todas as cabeças: {'sim' if conf['rede_supera_persistencia_em_todas_as_cabecas'] else 'não'}.",
              f"- Rede final supera o melhor baseline em todas as cabeças: {'sim' if conf['rede_supera_melhor_baseline_em_todas_as_cabecas'] else 'não'}.",
              "- A metade B não participou de escolha alguma (nem de baseline). Ela é uma medida honesta, mas de 28 meses de uma só região: "
              "o teste congelado continua sendo a medida final.", ""]
    L += [AVISO, ""]
    return "\n".join(L)


def gravar_relatorio(pasta_busca) -> Path:
    pasta = Path(pasta_busca)
    cab, tentativas = ler_registro(pasta / ARQ_REGISTRO)
    if cab is None:
        raise BuscaError("não há registro da busca")
    conf = None
    if (pasta / ARQ_CONFIRMACAO).exists():
        conf = json.loads((pasta / ARQ_CONFIRMACAO).read_text(encoding="utf-8"))
    destino = pasta / ARQ_RELATORIO
    tmp = destino.with_name(destino.name + ".tmp")
    tmp.write_text(texto_relatorio(cab, tentativas, conf), encoding="utf-8")
    os.replace(tmp, destino)
    return destino


# ---------------------------------------------------------------- linha de comando
def main(argv: list[str] | None = None, *, saida: Callable[[str], None] = print,
         carregar_cfg: Callable[[str], Config] = load_config, preparar_dados: Callable[[Config], Dados] = preparar) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.model.busca", description=__doc__.split("\n\n")[0])
    ap.add_argument("comando", choices=("rodar", "finalizar", "relatorio"))
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--max-segundos", type=float, default=None, help="não começa tentativa nova depois deste tempo")
    ap.add_argument("--n-tentativas", type=int, default=N_TENTATIVAS)
    ap.add_argument("--membros", type=int, default=N_MEMBROS)
    ap.add_argument("--n-arvores", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--epocas", type=int, default=LIMITES_PADRAO["epocas_max"])
    ap.add_argument("--paciencia", type=int, default=LIMITES_PADRAO["paciencia"])
    a = ap.parse_args(argv)
    try:
        cfg = carregar_cfg(a.params)
        pasta_busca = cfg.caminho("artifacts") / "busca"
        saida(f"AVISO: {AVISO}")
        limites = {**LIMITES_PADRAO, "epocas_max": a.epocas, "paciencia": a.paciencia}
        if a.comando == "relatorio":
            saida(f"Relatório em {gravar_relatorio(pasta_busca)}")
            return 0
        if cfg.split.teste_liberado:
            raise BuscaError("a busca recusa rodar com split.teste_liberado: true")
        d = preparar_dados(cfg)
        if a.comando == "rodar":
            feitas, total = rodar(d, pasta_busca / ARQ_REGISTRO, seed=cfg.projeto.seed, n_tentativas=a.n_tentativas,
                                  limites=limites, max_segundos=a.max_segundos, saida=saida)
            saida(f"{feitas} tentativa(s) agora; {total} de {a.n_tentativas} no registro")
            saida(f"Relatório em {gravar_relatorio(pasta_busca)}")
            saida("Próximo passo: " + ("python -m src.model.busca finalizar" if total >= a.n_tentativas
                                       else "python -m src.model.busca rodar (retoma de onde parou)"))
        else:
            conf = finalizar(d, pasta_busca, cfg.caminho("artifacts") / "modelo_v2", seed=cfg.projeto.seed, n_membros=a.membros,
                             n_arvores=a.n_arvores, n_boot=a.n_boot, saida=saida)
            for alvo in conf["alvos"]:
                c = conf["comparacao"][alvo]
                m = conf["modelos"]["rede_final"][alvo]["todas"]
                saida(f"{alvo} (metade B): rede final MAE {av._f(m['mae'])} | vs persistência {av._f(c['persistencia']['dif'], 4)} "
                      f"[{av._f(c['persistencia']['ic_inf'], 4)}; {av._f(c['persistencia']['ic_sup'], 4)}] ({c['persistencia']['veredito']}) | "
                      f"vs melhor baseline {av._f(c['melhor_baseline']['dif'], 4)} ({c['melhor_baseline']['veredito']})")
            saida(f"Relatório em {gravar_relatorio(pasta_busca)}")
    except (ValueError, RuntimeError, OSError) as e:      # BuscaError, TreinoError, AvaliarError, MetricasError, manifest, arquivo
        saida(f"ERRO: {e}")
        return 1
    saida(AVISO)
    return 0


if __name__ == "__main__":
    sys.exit(main())
