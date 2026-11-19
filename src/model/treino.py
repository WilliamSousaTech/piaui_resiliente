"""Treino da rede (Item 9): forward -> perda mascarada -> backward -> otimizador, em CPU.

Anti-vazamento: o treino usa só linhas do bloco `train`; a validação só observa (escolhe a época e restaura
os melhores pesos), nunca gera gradiente. O teste congelado não passa por aqui. A padronização vem de um
`Padronizador` ajustado no treino. Reprodutível: semente do params.yaml, `n_threads = 1` e embaralhamento
com gerador próprio (embaralhar linhas DENTRO do treino não quebra o split cronológico).

Uso:
    python -m src.model.treino [--params params.yaml] [--saida artifacts/modelo] [--epocas 80]
"""
from __future__ import annotations

import argparse
import copy
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.data.config import Config, load_config
from src.model import entrada
from src.model import features as fe
from src.model import rede as rd

AVISO = "MVP experimental, em calibração com S2iD/TCE-PI. Métricas de validação; o teste congelado não foi lido."
PASTA_PADRAO = "artifacts/modelo"
ARQUIVO_PESOS = "rede.pt"


class TreinoError(ValueError):
    """Dados, hiperparâmetros ou resultado incompatíveis com o treino."""


@dataclass(frozen=True)
class Hiperparametros:
    ocultas: tuple[int, ...] = (64, 32)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lote: int = 512
    epocas_max: int = 80
    paciencia: int = 10
    n_threads: int = 1
    perda: str = "mse"                # mse, l1 ou huber (rede.PERDAS); vale para o treino e para a escolha da época

    def __post_init__(self):
        if self.perda not in rd.PERDAS:
            raise TreinoError(f"perda deve ser uma de {rd.PERDAS} (recebi {self.perda!r})")
        if not (self.lr > 0 and np.isfinite(self.lr)) or not (self.weight_decay >= 0 and np.isfinite(self.weight_decay)):
            raise TreinoError("lr deve ser > 0 e weight_decay >= 0, ambos finitos")
        for nome in ("lote", "epocas_max", "paciencia", "n_threads"):
            v = getattr(self, nome)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise TreinoError(f"{nome} deve ser um inteiro >= 1 (recebi {v!r})")
        rd.Arquitetura(n_entradas=1, n_cabecas=1, ocultas=tuple(self.ocultas), dropout=self.dropout)   # valida

    def para_dict(self) -> dict:
        d = asdict(self)
        d["ocultas"] = list(self.ocultas)
        return d


@dataclass(frozen=True, eq=False)
class Resultado:
    rede: rd.RedeSeca              # pesos da melhor época, em modo eval
    padronizador: fe.Padronizador
    historico: tuple[dict, ...]    # {"epoca", "perda_treino", "perda_val"} por época treinada
    melhor_epoca: int              # 1-based
    hiperparametros: Hiperparametros
    seed: int
    colunas: tuple[str, ...]
    alvos: tuple[str, ...]


class RedePreditor:
    """Rede + padronizador com a mesma interface dos baselines: `prever(Conjunto) -> (n, cabeças) float32`."""
    nome = "rede"

    def __init__(self, rede: rd.RedeSeca, padronizador: fe.Padronizador):
        self.rede, self.padronizador = rede, padronizador

    def prever(self, c: fe.Conjunto, *, lote: int = 65536) -> np.ndarray:
        if c.colunas != self.padronizador.colunas:
            raise TreinoError("as colunas do conjunto não batem com as do padronizador da rede")
        Z = torch.from_numpy(self.padronizador.transformar(c))
        estava_treinando = self.rede.training
        self.rede.eval()
        try:
            with torch.no_grad():
                saida = [self.rede(Z[i:i + lote]).numpy() for i in range(0, len(Z), lote)]
        finally:
            self.rede.train(estava_treinando)
        P = np.concatenate(saida, axis=0) if saida else np.zeros((0, len(c.alvos)), dtype="float32")
        if not np.isfinite(P).all():
            raise TreinoError("a rede produziu previsão não finita")
        return P.astype("float32")


def _tensores(c: fe.Conjunto, p: fe.Padronizador):
    Z = torch.from_numpy(p.transformar(c))
    Y = torch.from_numpy(np.ascontiguousarray(c.Y, dtype="float32"))
    M = torch.from_numpy(np.ascontiguousarray(c.M, dtype=bool))
    return Z, Y, M


def _exigir(c: fe.Conjunto, bloco: str) -> None:
    if len(c) == 0 or not (c.meta["split"].astype(str) == bloco).all():
        raise TreinoError(f"use só linhas do bloco {bloco!r} (e pelo menos uma)")
    sem = [c.alvos[i] for i in range(len(c.alvos)) if not c.M[:, i].any()]
    if sem:
        raise TreinoError(f"bloco {bloco!r} sem rótulo para a(s) cabeça(s) {sem}")


def treinar(treino: fe.Conjunto, val: fe.Conjunto, padronizador: fe.Padronizador, *, seed: int,
            hp: Hiperparametros | None = None,
            ao_fim_da_epoca: Callable[[dict], None] | None = None) -> Resultado:
    """Treina do zero (pesos aleatórios) e devolve os pesos da época com menor perda de validação."""
    hp = hp or Hiperparametros()
    _exigir(treino, "train")
    _exigir(val, "val")
    if treino.colunas != val.colunas or treino.colunas != padronizador.colunas or treino.alvos != val.alvos:
        raise TreinoError("colunas ou alvos diferentes entre treino, validação e padronizador")
    util = treino.M.any(axis=1)                                   # linha sem rótulo nenhum não ensina nada
    Ztr, Ytr, Mtr = _tensores(treino.fatiar(util), padronizador)
    Zva, Yva, Mva = _tensores(val, padronizador)
    n = len(Ztr)
    arq = rd.Arquitetura(n_entradas=Ztr.shape[1], n_cabecas=Ytr.shape[1], ocultas=tuple(hp.ocultas), dropout=hp.dropout)
    antes = torch.get_num_threads()
    torch.set_num_threads(hp.n_threads)
    try:
        rede = rd.criar(arq, seed)
        opt = torch.optim.AdamW(rede.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        g = torch.Generator().manual_seed(seed)
        hist: list[dict] = []
        melhor_perda, melhor_epoca, melhor_estado, sem_ganho = float("inf"), 0, None, 0
        for epoca in range(1, hp.epocas_max + 1):
            rede.train()
            perm = torch.randperm(n, generator=g)
            soma = 0.0
            for i in range(0, n, hp.lote):
                idx = perm[i:i + hp.lote]
                opt.zero_grad()
                perda = rd.perda_mascarada(rede(Ztr[idx]), Ytr[idx], Mtr[idx], hp.perda)
                if not torch.isfinite(perda):
                    raise TreinoError(f"perda não finita na época {epoca}: reduza lr ou revise os dados")
                perda.backward()
                opt.step()
                soma += float(perda.detach()) * len(idx)
            rede.eval()
            with torch.no_grad():
                perda_val = float(rd.perda_mascarada(rede(Zva), Yva, Mva, hp.perda))
            if not np.isfinite(perda_val):
                raise TreinoError(f"perda de validação não finita na época {epoca}")
            reg = {"epoca": epoca, "perda_treino": soma / n, "perda_val": perda_val}
            hist.append(reg)
            if ao_fim_da_epoca is not None:
                ao_fim_da_epoca(reg)
            if perda_val < melhor_perda:
                melhor_perda, melhor_epoca, sem_ganho = perda_val, epoca, 0
                melhor_estado = copy.deepcopy(rede.state_dict())
            else:
                sem_ganho += 1
                if sem_ganho >= hp.paciencia:
                    break
        rede.load_state_dict(melhor_estado)
        rede.eval()
    finally:
        torch.set_num_threads(antes)
    return Resultado(rede=rede, padronizador=padronizador, historico=tuple(hist), melhor_epoca=melhor_epoca,
                     hiperparametros=hp, seed=seed, colunas=treino.colunas, alvos=treino.alvos)


# ---------------------------------------------------------------- persistência
def salvar_resultado(res: Resultado, pasta, dataset_sha256: str) -> str:
    """Grava `rede.pt` e `rede.json` em `pasta`. Devolve o SHA-256 dos pesos."""
    meta = {"dataset_sha256": dataset_sha256, "seed": res.seed, "colunas": list(res.colunas), "alvos": list(res.alvos),
            "padronizador": res.padronizador.para_dict(), "hiperparametros": res.hiperparametros.para_dict(),
            "melhor_epoca": res.melhor_epoca, "historico": list(res.historico),
            "aviso": "MVP experimental, em calibração com S2iD/TCE-PI"}
    return rd.salvar(res.rede, Path(pasta) / ARQUIVO_PESOS, meta)


def carregar_preditor(pasta) -> tuple[RedePreditor, dict]:
    """Lê `rede.pt`/`rede.json` de `pasta`. Recusa pesos cujo padronizador não combina com a arquitetura."""
    rede, meta = rd.carregar(Path(pasta) / ARQUIVO_PESOS)
    try:
        p = fe.Padronizador.de_dict(meta["padronizador"])
        colunas = tuple(meta["colunas"])
    except (KeyError, TypeError, fe.FeaturesError) as e:
        raise TreinoError(f"meta da rede inválido: {e}") from e
    if p.colunas != colunas or len(colunas) != rede.arq.n_entradas:
        raise TreinoError("o padronizador salvo não combina com a arquitetura da rede")
    return RedePreditor(rede, p), meta


# ---------------------------------------------------------------- linha de comando
def _inteiros(txt: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in txt.split(",") if x.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"use inteiros separados por vírgula, ex.: 64,32 (recebi {txt!r})") from e


def main(argv: list[str] | None = None, *, saida: Callable[[str], None] = print,
         carregar_cfg: Callable[[str], Config] = load_config) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.model.treino", description=__doc__.split("\n\n")[0])
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--saida", default=None, help=f"pasta de saída (padrão: <artifacts>/modelo = {PASTA_PADRAO})")
    ap.add_argument("--epocas", type=int, default=Hiperparametros.epocas_max)
    ap.add_argument("--paciencia", type=int, default=Hiperparametros.paciencia)
    ap.add_argument("--lr", type=float, default=Hiperparametros.lr)
    ap.add_argument("--ocultas", type=_inteiros, default=Hiperparametros.ocultas)
    ap.add_argument("--dropout", type=float, default=Hiperparametros.dropout)
    ap.add_argument("--perda", choices=rd.PERDAS, default=Hiperparametros.perda)
    a = ap.parse_args(argv)
    try:
        hp = Hiperparametros(ocultas=a.ocultas, dropout=a.dropout, lr=a.lr, epocas_max=a.epocas, paciencia=a.paciencia,
                             perda=a.perda)
        cfg = carregar_cfg(a.params)
        if cfg.split.teste_liberado:
            raise TreinoError("o treino recusa rodar com split.teste_liberado: true (o teste só entra na avaliação final)")
        saida(f"AVISO: {AVISO}")
        conj, sha = entrada.montar_conjunto(cfg)
        tr, va = fe.selecionar(conj, "train", cfg, so_com_rotulo=True), fe.selecionar(conj, "val", cfg, so_com_rotulo=True)
        pad = fe.Padronizador.ajustar(tr)
        saida(f"dataset_sha256 {sha[:16]}… | treino {len(tr)} linhas, validação {len(va)} | seed {cfg.projeto.seed}")
        res = treinar(tr, va, pad, seed=cfg.projeto.seed, hp=hp,
                      ao_fim_da_epoca=lambda r: saida(f"época {r['epoca']:>3} | perda treino {r['perda_treino']:.4f} | "
                                                      f"perda validação {r['perda_val']:.4f}"))
        pasta = a.saida or str(cfg.caminho("artifacts") / "modelo")
        sha_pesos = salvar_resultado(res, pasta, sha)
    except (ValueError, RuntimeError, OSError) as e:       # TreinoError, RedeError, FeaturesError, manifest, params ausente
        saida(f"ERRO: {e}")
        return 1
    saida(f"melhor época {res.melhor_epoca} (perda de validação {res.historico[res.melhor_epoca - 1]['perda_val']:.4f}); "
          f"pesos em {pasta} (sha256 {sha_pesos[:16]}…)")
    saida(f"Próximo passo: python -m src.model.avaliar. {AVISO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
