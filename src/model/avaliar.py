"""Avaliação da rede e dos baselines na VALIDAÇÃO (Item 9). O teste congelado não é aberto aqui.

Todo modelo é pontuado nas MESMAS linhas (as que têm rótulo) e nada é recalculado "só para os fáceis".
Entrega, por cabeça (SPI-3 e SPI-6): MAE, RMSE, viés, acerto de "em seca" (SPI <= -1), erro só nos alvos
de confiança normal, MAE por mês do calendário, a margem de erro que o site pode mostrar e a comparação
da rede com a persistência e com o melhor baseline (diferença de MAE com intervalo por bootstrap de MESES).

A rede só é avaliada se foi treinada com ESTES dados (mesmo `dataset_sha256`, mesmas colunas e o mesmo
padronizador de treino). Nada é gravado fora da pasta do modelo.

Uso:
    python -m src.model.avaliar [--params params.yaml] [--modelo artifacts/modelo] [--n-arvores 100]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.config import Config, load_config
from src.model import baselines as bl
from src.model import entrada
from src.model import features as fe
from src.model import metricas as mt
from src.model import treino as tn

AVISO = ("MVP experimental, em calibração com S2iD/TCE-PI. Métricas de validação, não de previsão operacional; "
         "o teste congelado não foi lido.")
Q_MARGEM = 0.80                         # a margem cobre 80% dos erros absolutos da validação
TOL_COBERTURA = 0.05                    # margem da 1ª metade que cobre menos que q - 0,05 da 2ª metade não é estável
ARQ_JSON = "avaliacao_validacao.json"
ARQ_MD = "avaliacao_validacao.md"
ORDEM = ("persistencia", "climatologia", "parcial_climatologica", "ridge", "floresta_aleatoria", "rede")
N_BOOT = 2000


class AvaliarError(ValueError):
    """A avaliação não pode ser feita de forma honesta com o que foi entregue."""


# ---------------------------------------------------------------- núcleo puro (sem arquivos)
def _veredito(r: dict) -> str:
    """`superou` só se o intervalo inteiro da diferença de MAE (rede - baseline) é negativo."""
    if not (math.isfinite(r["ic_inf"]) and math.isfinite(r["ic_sup"])):
        return "indefinido"
    if r["ic_sup"] < 0:
        return "superou"
    if r["ic_inf"] > 0:
        return "pior"
    return "indistinguivel"


def pontuar(va: fe.Conjunto, prev: dict[str, np.ndarray], *, principal: str = "rede", q_margem: float = Q_MARGEM,
            n_boot: int = N_BOOT, seed: int) -> dict:
    """Métricas de cada modelo em `prev` sobre a validação `va`. Recusa qualquer bloco que não seja `val`."""
    if len(va) == 0 or not (va.meta["split"].astype(str) == "val").all():
        raise AvaliarError("a avaliação usa só linhas do bloco 'val' (e pelo menos uma)")
    for nome in (principal, "persistencia"):
        if nome not in prev:
            raise AvaliarError(f"falta a previsão de {nome!r}")
    for nome, P in prev.items():
        if np.asarray(P).shape != (len(va), len(va.alvos)):
            raise AvaliarError(f"previsão de {nome!r} com shape {np.asarray(P).shape}; esperado {(len(va), len(va.alvos))}")
    nomes = [n for n in ORDEM if n in prev] + sorted(n for n in prev if n not in ORDEM)
    mes_alvo = pd.to_datetime(va.meta["mes_alvo"])
    mes_cal = mes_alvo.dt.month.to_numpy()
    grupo = (mes_alvo.dt.year * 12 + mes_alvo.dt.month).to_numpy()
    corte = np.sort(np.unique(grupo))[len(np.unique(grupo)) // 2]       # divide a validação em duas metades
    modelos: dict = {n: {} for n in nomes}
    margem, comparacao = {}, {}
    for i, alvo in enumerate(va.alvos):
        k = va.escalas[i]
        y, m = va.Y[:, i], va.M[:, i]
        if not m.any():
            raise AvaliarError(f"a validação não tem rótulo para {alvo}")
        baixa = va.meta[f"alvo_spi{k}_baixa_conf"].to_numpy(dtype=bool)
        for n in nomes:
            p = np.asarray(prev[n])[:, i]
            modelos[n][alvo] = {
                "todas": {**mt.erros(y, p, m), "seca": mt.seca(y, p, m), "classes": mt.acerto_classes(y, p, m)},
                "confiaveis": mt.erros(y, p, m & ~baixa),
                "por_mes_calendario": {str(c): (mt.erros(y, p, m & (mes_cal == c))["mae"] if (m & (mes_cal == c)).any()
                                                else None) for c in range(1, 13)},
            }
        p = np.asarray(prev[principal])[:, i]
        primeira, segunda = m & (grupo < corte), m & (grupo >= corte)
        valor = mt.quantil_erro_abs(y, p, m, q_margem)
        v1 = mt.quantil_erro_abs(y, p, primeira, q_margem) if primeira.any() else float("nan")
        cob2 = mt.cobertura_da_margem(y, p, segunda, v1) if segunda.any() and math.isfinite(v1) else float("nan")
        margem[alvo] = {"q": q_margem, "valor": valor, "cobertura_validacao": mt.cobertura_da_margem(y, p, m, valor),
                        "margem_1a_metade": v1, "cobertura_2a_metade": cob2,
                        "estavel": (cob2 >= q_margem - TOL_COBERTURA) if math.isfinite(cob2) else None}
        outros = [n for n in nomes if n != principal]
        melhor = min((n for n in outros), key=lambda n: modelos[n][alvo]["todas"]["mae"])
        comparacao[alvo] = {}
        for rotulo, contra in (("persistencia", "persistencia"), ("melhor_baseline", melhor)):
            r = mt.ic_bootstrap_dif_mae(y, p, np.asarray(prev[contra])[:, i], m, grupo, n_boot=n_boot, seed=seed)
            comparacao[alvo][rotulo] = {"contra": contra, **r, "veredito": _veredito(r)}
    return {
        "principal": principal, "n_linhas": len(va), "modelos": modelos, "margem": margem, "comparacao": comparacao,
        "rede_supera_persistencia_em_todas_as_cabecas":
            all(c["persistencia"]["veredito"] == "superou" for c in comparacao.values()),
        "rede_supera_melhor_baseline_em_todas_as_cabecas":
            all(c["melhor_baseline"]["veredito"] == "superou" for c in comparacao.values()),
    }


# ---------------------------------------------------------------- relatório
def _limpar(x):
    """NaN/inf viram None e tipos numpy viram nativos: o JSON sai estrito (sem `NaN`)."""
    if isinstance(x, dict):
        return {str(k): _limpar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_limpar(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def _f(x, casas: int = 3) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{casas}f}".replace(".", ",")


_NOME = {"persistencia": "Persistência", "climatologia": "Climatologia", "parcial_climatologica": "Parcial + climatologia",
         "ridge": "Ridge", "floresta_aleatoria": "Floresta aleatória", "rede": "Rede PyTorch (do zero)"}
_VEREDITO = {"superou": "a rede erra menos (intervalo inteiro abaixo de zero)",
             "pior": "a rede erra mais (intervalo inteiro acima de zero)",
             "indistinguivel": "diferença indistinguível de ruído (o intervalo contém zero)",
             "indefinido": "intervalo não calculável"}


def texto_relatorio(rel: dict) -> str:
    """Relatório em markdown, em português. Não afirma ganho que o intervalo não sustenta."""
    L = ["# Avaliação na validação: rede PyTorch contra baselines", "", f"> {rel['aviso']}", "",
         f"- dataset_sha256: `{rel['dataset_sha256'][:16]}…` | seed {rel['seed']} | {rel['n_linhas']} linhas de validação "
         f"({rel['janela_validacao'][0]} a {rel['janela_validacao'][1]}) | horizonte h={rel['horizonte']} mês",
         f"- Rede: melhor época {rel['rede']['melhor_epoca']}, pesos `{str(rel['rede']['pesos_sha256'])[:16]}…`",
         "- Erros em unidades de SPI. \"Em seca\" é SPI <= -1,0. Os baselines (persistência, climatologia, Ridge, floresta) "
         "existem só para comparar; o núcleo é a rede.", ""]
    for alvo in rel["alvos"]:
        L += [f"## {alvo}", "",
              "| Modelo | MAE | RMSE | Viés | MAE (alvos confiáveis) | Seca: precisão | Seca: revocação | Seca: F1 | Acerto de classe |",
              "|---|---|---|---|---|---|---|---|---|"]
        for n, por_alvo in rel["modelos"].items():
            t, c = por_alvo[alvo]["todas"], por_alvo[alvo]["confiaveis"]
            s = t["seca"]
            L.append(f"| {_NOME.get(n, n)} | {_f(t['mae'])} | {_f(t['rmse'])} | {_f(t['vies'])} | {_f(c['mae'])} | "
                     f"{_f(s['precisao'])} | {_f(s['revocacao'])} | {_f(s['f1'])} | {_f(t['classes']['acerto'])} |")
        ref = next(iter(rel["modelos"].values()))[alvo]["todas"]
        L += ["", f"Linhas em seca real: {ref['seca']['secas_reais']} de {ref['n']}. Acerto de classe = seca (SPI <= -1), normal ou "
                  f"úmido (SPI >= 1); prever sempre \"normal\" já acerta {_f(ref['classes']['acerto_sempre_normal'])}.", ""]
        for rot, titulo in (("persistencia", "persistência"), ("melhor_baseline", "melhor baseline")):
            r = rel["comparacao"][alvo][rot]
            L.append(f"- Rede − {_NOME.get(r['contra'], r['contra'])} ({titulo}): diferença de MAE {_f(r['dif'], 4)} "
                     f"[{_f(r['ic_inf'], 4)}; {_f(r['ic_sup'], 4)}] (95%, bootstrap de {r['n_grupos']} meses): "
                     f"{_VEREDITO[r['veredito']]}.")
        g = rel["margem"][alvo]
        L += ["", f"Margem de erro ({int(round(g['q'] * 100))}%): ±{_f(g['valor'])} de SPI; cobertura na validação "
                  f"{_f(g['cobertura_validacao'])}. Conferência: margem da 1ª metade ({_f(g['margem_1a_metade'])}) cobre "
                  f"{_f(g['cobertura_2a_metade'])} da 2ª metade.", "",
              "MAE por mês do calendário (mês do alvo):", "",
              "| Modelo | " + " | ".join(str(c) for c in range(1, 13)) + " |", "|---|" + "---|" * 12]
        for n, por_alvo in rel["modelos"].items():
            mes = por_alvo[alvo]["por_mes_calendario"]
            L.append(f"| {_NOME.get(n, n)} | " + " | ".join(_f(mes[str(c)], 2) for c in range(1, 13)) + " |")
        L.append("")
    instaveis = [a for a in rel["alvos"] if rel["margem"][a].get("estavel") is False]
    L += ["## Leitura", "",
          ("- A margem de erro NÃO se manteve entre as metades da validação em " + ", ".join(instaveis) + ": não apresentá-la "
           f"como cobertura de {int(round(Q_MARGEM * 100))}% garantida." if instaveis else
           "- A margem de erro da 1ª metade da validação cobriu a 2ª metade dentro da tolerância."),
          f"- Rede supera a persistência em todas as cabeças: {'sim' if rel['rede_supera_persistencia_em_todas_as_cabecas'] else 'não'}.",
          f"- Rede supera o melhor baseline em todas as cabeças: {'sim' if rel['rede_supera_melhor_baseline_em_todas_as_cabecas'] else 'não'}.",
          "- Validação serviu para escolher a época e o alfa do Ridge; ela é otimista para quem foi ajustado nela. "
          "Só o teste congelado, lido uma vez, dá a medida final.", "", AVISO, ""]
    return "\n".join(L)


def gravar(rel: dict, pasta) -> tuple[Path, Path]:
    """Grava JSON estrito e markdown em `pasta`, por arquivo temporário (nunca deixa relatório pela metade)."""
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    destinos = (pasta / ARQ_JSON, pasta / ARQ_MD)
    conteudos = (json.dumps(_limpar(rel), ensure_ascii=False, indent=2, allow_nan=False) + "\n", texto_relatorio(_limpar(rel)))
    for destino, txt in zip(destinos, conteudos):
        tmp = destino.with_name(destino.name + ".tmp")
        tmp.write_text(txt, encoding="utf-8")
        os.replace(tmp, destino)
    return destinos


# ---------------------------------------------------------------- avaliação com os arquivos do projeto
def avaliar(cfg: Config, pasta_modelo, *, n_arvores: int = 100, n_boot: int = N_BOOT) -> dict:
    """Carrega dados verificados e a rede treinada, ajusta os baselines no treino e pontua tudo na validação."""
    if cfg.split.teste_liberado:
        raise AvaliarError("esta avaliação é só de validação: recuse rodar com split.teste_liberado: true")
    ins = entrada.carregar_insumos(cfg)
    conj = fe.construir(ins.demo, ins.serie_imputada, ins.climatologia, ins.municipios, cfg)
    preditor, meta = tn.carregar_preditor(pasta_modelo)
    if meta.get("dataset_sha256") != ins.dataset_sha256:
        raise AvaliarError("a rede foi treinada com outro dataset_sha256; treine de novo (python -m src.model.treino)")
    if tuple(meta.get("colunas", ())) != conj.colunas or tuple(meta.get("alvos", ())) != conj.alvos:
        raise AvaliarError("as colunas ou os alvos da rede não batem com as features atuais; treine de novo")
    tr = fe.selecionar(conj, "train", cfg, so_com_rotulo=True)
    va = fe.selecionar(conj, "val", cfg, so_com_rotulo=True)
    pad = fe.Padronizador.ajustar(tr)
    if not (np.allclose(pad.media, preditor.padronizador.media, rtol=1e-7, atol=1e-9)
            and np.allclose(pad.desvio, preditor.padronizador.desvio, rtol=1e-7, atol=1e-9)):
        raise AvaliarError("o padronizador da rede difere do ajustado no treino atual; treine de novo")
    prev = {n: b.prever(va) for n, b in bl.ajustar_todos(tr, va, pad, ins.params_gama, seed=cfg.projeto.seed,
                                                       n_arvores=n_arvores).items()}
    prev[preditor.nome] = preditor.prever(va)
    corpo = pontuar(va, prev, principal=preditor.nome, n_boot=n_boot, seed=cfg.projeto.seed)
    mes = pd.to_datetime(va.meta["mes_alvo"])
    pesos = json.loads((Path(pasta_modelo) / "rede.json").read_text(encoding="utf-8")).get("pesos_sha256")
    return {"aviso": AVISO, "bloco": "val", "dataset_sha256": ins.dataset_sha256, "seed": cfg.projeto.seed,
            "alvos": list(va.alvos), "horizonte": va.horizonte,
            "janela_validacao": [str(mes.min())[:7], str(mes.max())[:7]],
            "rede": {"pesos_sha256": pesos, "melhor_epoca": meta.get("melhor_epoca"),
                     "hiperparametros": meta.get("hiperparametros")},
            "baselines": {"n_arvores": n_arvores}, "bootstrap": {"n_boot": n_boot, "nivel": 0.95}, **corpo}


# ---------------------------------------------------------------- linha de comando
def main(argv: list[str] | None = None, *, saida: Callable[[str], None] = print,
         carregar_cfg: Callable[[str], Config] = load_config) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.model.avaliar", description=__doc__.split("\n\n")[0])
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--modelo", default=None, help=f"pasta com rede.pt/rede.json (padrão: <artifacts>/modelo = {tn.PASTA_PADRAO})")
    ap.add_argument("--n-arvores", type=int, default=100)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args(argv)
    try:
        cfg = carregar_cfg(a.params)
        pasta = a.modelo or str(cfg.caminho("artifacts") / "modelo")
        saida(f"AVISO: {AVISO}")
        rel = avaliar(cfg, pasta, n_arvores=a.n_arvores, n_boot=a.n_boot)
        destinos = gravar(rel, pasta)
    except (ValueError, RuntimeError, OSError) as e:     # AvaliarError, TreinoError, MetricasError, manifest, arquivo ausente
        saida(f"ERRO: {e}")
        return 1
    for alvo in rel["alvos"]:
        r = rel["modelos"]["rede"][alvo]["todas"]
        p = rel["modelos"]["persistencia"][alvo]["todas"]
        c = rel["comparacao"][alvo]["persistencia"]
        saida(f"{alvo}: rede MAE {_f(r['mae'])} / RMSE {_f(r['rmse'])} | persistência MAE {_f(p['mae'])} / RMSE {_f(p['rmse'])} "
              f"| rede - persistência {_f(c['dif'], 4)} [{_f(c['ic_inf'], 4)}; {_f(c['ic_sup'], 4)}] ({c['veredito']})")
    saida(f"Relatório em {destinos[1]} (dados em {destinos[0]})")
    saida(f"Próximo passo: decidir o que o site mostra e se o teste congelado será lido uma única vez. {AVISO}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
