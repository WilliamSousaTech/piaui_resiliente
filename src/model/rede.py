"""Rede neural do Piauí Resiliente: o núcleo preditivo (Item 9). PyTorch, treinada do zero.

Zero pesos pré-treinados: `criar` semeia o gerador e deixa o PyTorch inicializar os pesos; nada é lido de
disco para construir a rede. O treino (forward -> perda -> backward -> otimizador) vive em `treino.py`.

Arquitetura (contrato, seção 3.1): tronco compartilhado e uma cabeça linear por alvo (SPI-3 e SPI-6 em t+h).
Lógica de domínio e de ML apenas: nada aqui depende de framework web nem de SQL.

Persistência: os pesos vão em um `.pt` que contém só tensores (carregado com `weights_only=True`, sem
pickle de objetos) e os metadados em um `.json` ao lado, que guarda o SHA-256 do `.pt`.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

VERSAO_FORMATO = 1


class RedeError(ValueError):
    """Entrada, arquivo ou configuração incompatível com a rede."""


@dataclass(frozen=True)
class Arquitetura:
    n_entradas: int
    n_cabecas: int
    ocultas: tuple[int, ...] = (64, 32)
    dropout: float = 0.1

    def __post_init__(self):
        if self.n_entradas < 1 or self.n_cabecas < 1:
            raise RedeError("n_entradas e n_cabecas devem ser >= 1")
        if not self.ocultas or any((not isinstance(h, int)) or isinstance(h, bool) or h < 1 for h in self.ocultas):
            raise RedeError(f"ocultas deve ser uma lista de inteiros >= 1 (recebi {self.ocultas!r})")
        if not (0.0 <= self.dropout < 1.0):
            raise RedeError(f"dropout deve estar em [0, 1) (recebi {self.dropout!r})")

    def para_dict(self) -> dict:
        return {"n_entradas": self.n_entradas, "n_cabecas": self.n_cabecas, "ocultas": list(self.ocultas),
                "dropout": self.dropout}

    @classmethod
    def de_dict(cls, d: dict) -> "Arquitetura":
        try:
            return cls(n_entradas=int(d["n_entradas"]), n_cabecas=int(d["n_cabecas"]),
                       ocultas=tuple(int(h) for h in d["ocultas"]), dropout=float(d["dropout"]))
        except (KeyError, TypeError, ValueError) as e:
            raise RedeError(f"arquitetura inválida: {e}") from e


class RedeSeca(nn.Module):
    """MLP com tronco compartilhado e uma cabeça linear por alvo. Entrada (n, d) float32, saída (n, cabeças)."""

    def __init__(self, arq: Arquitetura):
        super().__init__()
        self.arq = arq
        camadas: list[nn.Module] = []
        d = arq.n_entradas
        for h in arq.ocultas:
            camadas += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(arq.dropout)]
            d = h
        self.tronco = nn.Sequential(*camadas)
        self.cabecas = nn.ModuleList([nn.Linear(d, 1) for _ in range(arq.n_cabecas)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.arq.n_entradas:
            raise RedeError(f"entrada deve ter shape (n, {self.arq.n_entradas}); recebi {tuple(x.shape)}")
        if x.dtype != torch.float32:
            raise RedeError(f"entrada deve ser float32; recebi {x.dtype}")
        z = self.tronco(x)
        return torch.cat([c(z) for c in self.cabecas], dim=1)


def criar(arq: Arquitetura, seed: int) -> RedeSeca:
    """Rede nova com pesos aleatórios determinados pela semente (zero pesos pré-treinados)."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RedeError("seed deve ser inteiro")
    torch.manual_seed(seed)
    return RedeSeca(arq)


def perda_mascarada(pred: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """Erro quadrático médio só nas posições com rótulo (`m` verdadeiro), média simples entre as cabeças.

    Posição sem rótulo não entra na perda nem no gradiente, e o valor dela (NaN) é trocado antes da conta:
    NaN * 0 continuaria NaN. Cabeça sem nenhum rótulo no lote é ignorada; lote sem rótulo nenhum dá perda 0."""
    if pred.dim() != 2 or pred.shape != y.shape or pred.shape != m.shape:
        raise RedeError(f"pred, y e m devem ter o mesmo shape (n, cabeças): {tuple(pred.shape)}, "
                        f"{tuple(y.shape)}, {tuple(m.shape)}")
    if m.dtype != torch.bool:
        raise RedeError("a máscara m deve ser bool")
    if not bool(torch.isfinite(y[m]).all()):
        raise RedeError("rótulo não finito em posição com máscara verdadeira")
    y0 = torch.where(m, y, torch.zeros_like(y))
    quad = torch.where(m, (pred - y0) ** 2, torch.zeros_like(pred))
    n = m.sum(dim=0)
    por_cabeca = quad.sum(dim=0) / n.clamp(min=1).to(pred.dtype)
    ativa = (n > 0).to(pred.dtype)
    return (por_cabeca * ativa).sum() / ativa.sum().clamp(min=1.0)


# ---------------------------------------------------------------- persistência
def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def _json_ao_lado(pesos: Path) -> Path:
    return pesos.with_suffix(".json")


def salvar(rede: RedeSeca, pesos: str | Path, meta: dict) -> str:
    """Grava `pesos` (.pt, só tensores) e o `.json` ao lado (arquitetura, SHA-256 dos pesos e `meta`).

    `meta` deve ser JSON puro (padronizador, dataset_sha256, semente, histórico...). Devolve o SHA-256 dos pesos."""
    pesos = Path(pesos)
    if pesos.suffix != ".pt":
        raise RedeError("o arquivo de pesos deve terminar em .pt")
    try:
        json.dumps(meta, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise RedeError(f"meta não é JSON puro: {e}") from e
    pesos.parent.mkdir(parents=True, exist_ok=True)
    tmp = pesos.with_name(pesos.name + ".tmp")
    torch.save({k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}, tmp)
    sha = _sha256(tmp)
    os.replace(tmp, pesos)
    doc = {"versao_formato": VERSAO_FORMATO, "arquitetura": rede.arq.para_dict(), "pesos_sha256": sha, "meta": meta}
    alvo_json = _json_ao_lado(pesos)
    tmp_json = alvo_json.with_name(alvo_json.name + ".tmp")
    tmp_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp_json, alvo_json)
    return sha


def carregar(pesos: str | Path) -> tuple[RedeSeca, dict]:
    """Lê `.pt` e `.json`. Confere o SHA-256 antes de abrir os pesos e abre só tensores (`weights_only=True`)."""
    pesos = Path(pesos)
    j = _json_ao_lado(pesos)
    if not pesos.is_file() or not j.is_file():
        raise RedeError(f"faltam {pesos.name} e/ou {j.name}")
    try:
        doc = json.loads(j.read_text(encoding="utf-8"))
        esperado, arq, meta = doc["pesos_sha256"], Arquitetura.de_dict(doc["arquitetura"]), doc["meta"]
        versao = doc["versao_formato"]
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise RedeError(f"{j.name} inválido: {e}") from e
    if versao != VERSAO_FORMATO:
        raise RedeError(f"versão de formato {versao!r} não suportada (esperada {VERSAO_FORMATO})")
    if _sha256(pesos) != esperado:
        raise RedeError(f"{pesos.name} não confere com o SHA-256 registrado em {j.name}")
    try:
        estado = torch.load(pesos, map_location="cpu", weights_only=True)
        rede = RedeSeca(arq)
        rede.load_state_dict(estado, strict=True)
    except Exception as e:                       # pickle recusado, chaves ou shapes diferentes: tudo vira RedeError
        raise RedeError(f"não foi possível carregar {pesos.name}: {type(e).__name__}: {e}") from e
    rede.eval()
    return rede, meta
