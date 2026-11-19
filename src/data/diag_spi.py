"""Diagnóstico do SPI em um mês-alvo (somente leitura, sem rede, sem gravar nada).

Pergunta que responde: os municípios "úmidos" de um mês estão úmidos porque choveu muito, porque a
base climatológica da janela é quase zero (poucos mm já dão SPI alto), porque a Gama se ajustou mal
ou porque o dado recente é suspeito? Três hipóteses, três conjuntos de números:

  H1 base baixa      -> mm da janela vs. média histórica; mm necessários para SPI >= +1.
  H2 ajuste ruim     -> fração dos anos do TREINO (in-sample) com SPI >= +1; esperado ~16%.
  H3 dado recente    -> chuva mediana do estado nos 12 meses até o alvo vs. a mediana do treino.

Lê SÓ o que o manifest lista e confere o SHA-256 (`manifest.verificar_manifest`). A lógica de cálculo
(`calcular_diagnostico`) é pura: recebe DataFrames, devolve dict. Não é previsão e não treina nada.
"""
from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.data import chirps_raw, manifest as mf, spi
from src.data.config import Config, load_config

LIMIAR = 1.0                  # SPI >= +1,0 é "úmido"; SPI <= -1,0 é "seca" (mesma regra do site)
GRUPOS = ("umido", "normal", "seca")
LIMITES_CALIBRACAO = (5.0, 10.0, 15.0, 20.0, 30.0)   # mm por mês da janela, para escolher `spi.media_min_mm_por_mes`


class DiagError(ValueError):
    """Entrada incompatível com o diagnóstico (mês fora da janela, escala sem coluna, etc.)."""


# ------------------------------------------------------------------ cálculo (puro)
def _matriz(demo: pd.DataFrame, coluna: str, indice: pd.PeriodIndex, cods: list[str]) -> pd.DataFrame:
    d = pd.DataFrame({"cod": demo["cod_ibge"].astype(str).to_numpy(),
                      "mes": pd.to_datetime(demo["mes"]).dt.to_period("M").to_numpy(),
                      "v": demo[coluna].to_numpy()})
    if d.duplicated(["cod", "mes"]).any():
        raise DiagError(f"demo_observado com duplicatas (cod_ibge, mes) na coluna {coluna}")
    return d.pivot(index="mes", columns="cod", values="v").reindex(index=indice, columns=cods)


def _med(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def _limiar_mm(q: np.ndarray, forma: np.ndarray, escala: np.ndarray, valido: np.ndarray, alvo_h: float, *,
               superior: bool) -> np.ndarray:
    """mm da janela em que H(x) = alvo_h, com H(0) = q e H crescente em x.

    q <= alvo_h: há solução, x = ppf da Gama (q == alvo_h dá 0 mm). q > alvo_h: H(x) >= q > alvo_h para todo x >= 0.
      superior (SPI >= +1): já vale com zero mm, então o limiar é 0 ("qualquer chuva basta").
      inferior (SPI <= -1): nenhuma chuva chega lá, então NaN ("inalcançável")."""
    out = np.full(q.shape, np.nan)
    ok = valido & np.isfinite(q) & np.isfinite(forma) & np.isfinite(escala)
    gama = ok & (q <= alvo_h)
    out[gama] = stats.gamma.ppf((alvo_h - q[gama]) / (1.0 - q[gama]), forma[gama], scale=escala[gama])
    if superior:
        out[ok & (q > alvo_h)] = 0.0
    return out


def _calibracao(pk: pd.DataFrame, k: int, q_baixa: float) -> list[dict]:
    """Por mês-calendário: mediana da média da janela por mês (mm) e % de células abaixo de cada limite.
    Só usa os parâmetros do treino (spi_gamma_params), então vale antes de qualquer rebuild."""
    linhas = []
    for m in range(1, 13):
        x = pk[(pk["mes_calendario"] == m) & pk["valido"].fillna(False).astype(bool)]
        media = ((1.0 - x["q"]) * x["shape"] * x["scale"] / k).to_numpy(dtype="float64")
        n = int(media.size)
        linhas.append({"mes": m, "n": n, "mediana": _med(media), "q_acima": int((x["q"] > q_baixa).sum()),
                       "pct": {lim: (float((media < lim).mean() * 100) if n else float("nan")) for lim in LIMITES_CALIBRACAO}})
    return linhas


def calcular_diagnostico(raw: pd.DataFrame, demo: pd.DataFrame, params: pd.DataFrame, *,
                         treino_fim: pd.Period, mes: pd.Period, k: int, q_baixa: float = 0.5,
                         media_min_mm_mes: float = 0.0) -> dict:
    M = spi.para_matriz(raw)
    if mes not in M.index:
        raise DiagError(f"mês {mes} fora da janela do RAW ({M.index[0]}..{M.index[-1]})")
    col = f"spi{k}"
    for c in (col, f"{col}_invalid", f"{col}_baixa_conf"):
        if c not in demo.columns:
            raise DiagError(f"demo_observado sem a coluna {c!r}: escala SPI-{k} não calculada")
    S = spi.acumular(M, k)
    cods = [str(c) for c in S.columns]
    spi_m = _matriz(demo, col, S.index, cods).to_numpy(dtype="float64")
    baixa_m = _matriz(demo, f"{col}_baixa_conf", S.index, cods).to_numpy(dtype="float64") == 1.0
    pk = params[params["escala_meses"] == k]
    if pk.empty:
        raise DiagError(f"spi_gamma_params sem a escala SPI-{k}")
    pc = pk[pk["mes_calendario"] == mes.month].assign(cod_ibge=lambda d: d["cod_ibge"].astype(str))
    pc = pc.drop_duplicates("cod_ibge").set_index("cod_ibge").reindex(cods)
    q = pc["q"].to_numpy(dtype="float64")
    forma, escala = pc["shape"].to_numpy(dtype="float64"), pc["scale"].to_numpy(dtype="float64")
    valido = pc["valido"].fillna(False).to_numpy(dtype=bool)

    idx = S.index
    i_t = int(idx.get_loc(mes))
    s_t = S.to_numpy(dtype="float64")[i_t]
    cal_igual = np.asarray(idx.month) == mes.month
    m_treino = cal_igual & np.asarray(idx <= treino_fim)
    m_antes = cal_igual & (np.arange(len(idx)) < i_t)
    Sv = S.to_numpy(dtype="float64")
    tr, antes = Sv[m_treino], Sv[m_antes]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        clim = np.nanmean(tr, axis=0) if len(tr) else np.full(len(cods), np.nan)
        p90 = np.nanpercentile(tr, 90, axis=0) if len(tr) else np.full(len(cods), np.nan)
        maximo = np.nanmax(antes, axis=0) if len(antes) else np.full(len(cods), np.nan)
    media_mes = np.where(valido, (1.0 - q) * forma * escala / k, np.nan)       # mm por mês da janela (treino)
    pouca_chuva = np.isfinite(media_mes) & (media_mes < media_min_mm_mes)
    mm_mais1 = _limiar_mm(q, forma, escala, valido, float(stats.norm.cdf(LIMIAR)), superior=True)
    mm_seca = _limiar_mm(q, forma, escala, valido, float(stats.norm.cdf(-LIMIAR)), superior=False)

    spi_t = np.round(spi_m[i_t], 2)
    fin = np.isfinite(spi_t)
    umido, seca = fin & (spi_t >= LIMIAR), fin & (spi_t <= -LIMIAR)
    normal = fin & ~umido & ~seca
    trs = np.round(spi_m[m_treino], 2)
    n_tr = np.isfinite(trs).sum(axis=0)
    with np.errstate(all="ignore"):
        ins_umido = np.where(n_tr > 0, (trs >= LIMIAR).sum(axis=0) / n_tr, np.nan)
        ins_seca = np.where(n_tr > 0, (trs <= -LIMIAR).sum(axis=0) / n_tr, np.nan)
        razao = np.where(np.isfinite(clim) & (clim > 0), s_t / clim, np.nan)
    recorde = np.isfinite(s_t) & np.isfinite(maximo) & (s_t > maximo)
    acima_p90 = np.isfinite(s_t) & np.isfinite(p90) & (s_t > p90)

    detalhe = pd.DataFrame({
        "cod_ibge": cods, "grupo": np.select([umido, seca, normal], ["umido", "seca", "normal"], default="sem_spi"),
        "spi": spi_t, "soma_mm": s_t, "media_hist_mm": clim, "p90_mm": p90, "razao": razao, "q": q,
        "mm_para_umido": mm_mais1, "mm_para_seca": mm_seca, "insample_umido": ins_umido,
        "insample_seca": ins_seca, "acima_p90": acima_p90, "recorde": recorde, "baixa_conf": baixa_m[i_t],
        "media_mm_mes": media_mes, "pouca_chuva": pouca_chuva})
    grupos = {}
    for nome, m in zip(GRUPOS, (umido, normal, seca)):
        grupos[nome] = {
            "n": int(m.sum()), "soma_mm": _med(s_t[m]), "media_hist_mm": _med(clim[m]), "razao": _med(razao[m]),
            "mm_para_umido": _med(mm_mais1[m]), "mm_para_seca": _med(mm_seca[m]),
            "insample_umido": _med(ins_umido[m]), "insample_seca": _med(ins_seca[m]),
            "acima_p90": int(acima_p90[m].sum()), "recorde": int(recorde[m].sum()),
            "baixa_conf": int(baixa_m[i_t][m].sum()), "q_acima_limiar": int((q[m] > q_baixa).sum()),
            "pouca_chuva": int(pouca_chuva[m].sum()),
            "q_mediana": _med(q[m]),
        }
    # mês-calendário em que cada município tem o SPI mínimo (o "mês mais seco" da ficha)
    com = np.isfinite(spi_m).any(axis=0)
    amin = np.nanargmin(np.where(np.isfinite(spi_m[:, com]), spi_m[:, com], np.inf), axis=0)
    cal_min = np.bincount(np.asarray(idx.month)[amin], minlength=13)[1:]
    # chuva mediana do estado, 12 meses até o alvo, contra a mediana do treino no mesmo mês-calendário
    Mv = M.to_numpy(dtype="float64")
    treino_linhas = np.asarray(M.index <= treino_fim)
    serie = []
    pos_t = int(M.index.get_loc(mes))
    for j in range(max(0, pos_t - 11), pos_t + 1):
        p = M.index[j]
        base = Mv[treino_linhas & (np.asarray(M.index.month) == p.month)]
        serie.append({"mes": str(p), "estado_mm": _med(Mv[j]), "clim_mm": _med(base.ravel()) if base.size else float("nan")})
    return {
        "mes": str(mes), "escala": k, "treino_fim": str(treino_fim), "mes_no_treino": bool(mes <= treino_fim),
        "n_municipios": len(cods), "n_validos": int(fin.sum()), "n_invalidos": int((~fin).sum()),
        "grupos": grupos, "detalhe": detalhe, "media_min_mm_mes": media_min_mm_mes,
        "calibracao": _calibracao(pk, k, q_baixa),
        "q": {"mediana": _med(q), "maximo": float(np.nanmax(q)) if np.isfinite(q).any() else float("nan"),
              "n_acima_limiar": int((q > q_baixa).sum()), "limiar": q_baixa},
        "media_hist_mm": {"min": float(np.nanmin(clim)) if np.isfinite(clim).any() else float("nan"),
                          "mediana": _med(clim),
                          "max": float(np.nanmax(clim)) if np.isfinite(clim).any() else float("nan")},
        "esperado_insample": float(stats.norm.sf(LIMIAR)),
        "cal_do_spi_minimo": [int(x) for x in cal_min],
        "serie_estado": serie,
    }


# ------------------------------------------------------------------ texto
def _f(x: float, casas: int = 1, sufixo: str = "") -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{casas}f}{sufixo}"


def formatar(r: dict, aviso: str) -> str:
    k = r["escala"]
    L = [f"AVISO: {aviso}",
         f"DIAGNÓSTICO SPI-{k} em {r['mes']} | municípios={r['n_municipios']} com SPI={r['n_validos']} sem SPI={r['n_invalidos']}"
         + (" | ATENÇÃO: mês dentro do treino (in-sample)" if r["mes_no_treino"] else ""),
         f"treino até {r['treino_fim']}; esperado in-sample com SPI>=+1: {r['esperado_insample'] * 100:.1f}%", ""]
    nomes = {"umido": "úmidos (SPI>=+1)", "normal": "normais", "seca": "em seca (SPI<=-1)"}
    L.append("grupo | n | soma da janela mm | média hist. mm | razão | mm p/ SPI>=+1 | in-sample SPI>=+1 | >p90 | recorde | baixa conf. | q>lim. | "
             f"pouca chuva (<{_f(r['media_min_mm_mes'])} mm/mês)")
    for g in GRUPOS:
        d = r["grupos"][g]
        L.append(f"{nomes[g]} | {d['n']} | {_f(d['soma_mm'])} | {_f(d['media_hist_mm'])} | {_f(d['razao'], 2)} | "
                 f"{_f(d['mm_para_umido'])} | {_f(d['insample_umido'] * 100, 1, '%')} | "
                 f"{d['acima_p90']} | {d['recorde']} | {d['baixa_conf']} | {d['q_acima_limiar']} | {d['pouca_chuva']}")
    L.append("(medianas por grupo; razão = soma / média histórica; recorde = maior valor desse mês-calendário na série)")
    top = r["detalhe"].sort_values("spi", ascending=False).head(5)
    L += ["", "5 maiores SPI (cod_ibge | SPI | soma mm | média hist. mm | mm p/ SPI>=+1 | in-sample SPI>=+1):"]
    for _, t in top.iterrows():
        L.append(f"{t['cod_ibge']} | {_f(t['spi'], 2)} | {_f(t['soma_mm'])} | {_f(t['media_hist_mm'])} | "
                 f"{_f(t['mm_para_umido'])} | {_f(t['insample_umido'] * 100, 1, '%')}")
    L += ["", f"q (fração de zeros no treino): mediana={_f(r['q']['mediana'], 3)} máx={_f(r['q']['maximo'], 3)} "
              f"municípios com q>{r['q']['limiar']}: {r['q']['n_acima_limiar']}",
          f"média histórica da janela (mm): mín={_f(r['media_hist_mm']['min'])} mediana={_f(r['media_hist_mm']['mediana'])} "
          f"máx={_f(r['media_hist_mm']['max'])}", ""]
    L.append("mês-calendário do SPI mínimo de cada município (jan..dez): " + " ".join(str(x) for x in r["cal_do_spi_minimo"]))
    L += ["", f"calibração do limite (SPI-{k}): por mês do ano, mediana da média da janela por mês (mm) e % de municípios abaixo de cada limite (mm/mês)",
          "mês | n | mediana mm/mês | " + " | ".join(f"<{lim:g}" for lim in LIMITES_CALIBRACAO) + " | q>lim."]
    for c in r["calibracao"]:
        L.append(f"{c['mes']:02d} | {c['n']} | {_f(c['mediana'])} | " + " | ".join(_f(c['pct'][lim], 0, '%') for lim in LIMITES_CALIBRACAO)
                 + f" | {c['q_acima']}")
    L += ["", "chuva mediana do estado (mm) nos 12 meses até o alvo | mediana do treino no mesmo mês | razão"]
    for s in r["serie_estado"]:
        razao = s["estado_mm"] / s["clim_mm"] if np.isfinite(s["clim_mm"]) and s["clim_mm"] >= 1.0 else float("nan")
        L.append(f"{s['mes']} | {_f(s['estado_mm'])} | {_f(s['clim_mm'])} | {_f(razao, 2)}")
    return "\n".join(L)


# ------------------------------------------------------------------ entrada (I/O)
def carregar(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """Confere os ARQUIVOS do manifest e lê RAW, demo_observado e parâmetros da Gama só pelo que ele lista.

    O `params.yaml` pode ter mudado desde o manifest: é o caso de quem testa um limiar novo ANTES de regenerar
    o demo, que é exatamente para o que este diagnóstico serve. Então o hash do `params.yaml` não é exigido
    aqui (o site e o build_demo continuam exigindo). O que o diagnóstico usa do `params.yaml` é só a regra de
    baixa confiança; o que ele usa do ajuste (treino, janela, escalas) tem de bater com o manifest, senão
    recusa. Devolve também se o `params.yaml` mudou, para o relatório avisar."""
    m = mf.verificar_manifest(cfg, conferir_params=False)
    arq = cfg.raiz / mf.PARAMS
    mudou = (not arq.is_file()) or mf.sha256_arquivo(arq) != m["params_sha256"]
    if mudou:
        esperado = {"treino_fim": str(cfg.split.treino_fim), "janela_inicio": str(cfg.janela.inicio),
                    "janela_fim": str(cfg.janela.fim), "janelas_spi": list(cfg.spi.janelas_meses)}
        no_manifest = {"treino_fim": m["split"]["treino_fim"], "janela_inicio": m["janela"]["inicio"],
                       "janela_fim": m["janela"]["fim"], "janelas_spi": list(m["alvo"]["janelas_spi"])}
        difere = sorted(k for k in esperado if esperado[k] != no_manifest[k])
        if difere:
            raise DiagError(f"params.yaml mudou em {difere}, que afetam o ajuste da Gama; regenere o demo "
                            "(python -m src.data.build_demo --substituir-manifest) antes de diagnosticar")
    lotes = sorted(n for n in m["arquivos"] if n.startswith("raw_chirps/"))
    faltam = [n for n in ("demo_observado", "spi_gamma_params") if n not in m["arquivos"]]
    if not lotes or faltam:
        raise DiagError(f"o manifest não lista {faltam or 'lotes do RAW'}")
    ler = lambda nome: pd.read_parquet(mf.caminho_verificado(cfg, nome), engine="pyarrow")   # noqa: E731
    raw = pd.concat([chirps_raw.ler_lote(mf.caminho_verificado(cfg, n)) for n in lotes], ignore_index=True)
    return raw, ler("demo_observado"), ler("spi_gamma_params"), mudou


def main(argv: list[str] | None = None) -> int:       # python -m src.data.diag_spi
    ap = argparse.ArgumentParser(prog="python -m src.data.diag_spi",
                                 description="Diagnóstico somente leitura do SPI em um mês (hipóteses H1/H2/H3).")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--mes", default=None, help="AAAA-MM; padrão: janela.fim")
    ap.add_argument("--escala", type=int, default=None, help="padrão: alvo.escala_spi_meses")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.params)
        mes = pd.Period(a.mes, freq="M") if a.mes else cfg.janela.fim
        k = a.escala if a.escala is not None else cfg.alvo.escala_spi_meses
        if k not in cfg.spi.janelas_meses:
            raise DiagError(f"escala {k} fora de spi.janelas_meses={list(cfg.spi.janelas_meses)}")
        raw, demo, params, params_mudou = carregar(cfg)
        r = calcular_diagnostico(raw, demo, params, treino_fim=cfg.split.treino_fim, mes=mes, k=k,
                                 q_baixa=cfg.spi.q_baixa_confianca, media_min_mm_mes=cfg.spi.media_min_mm_por_mes)
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    print(formatar(r, cfg.projeto.maturidade))
    if params_mudou:
        print("\nNOTA: o params.yaml mudou desde o manifest (esperado ao testar um limiar novo). Os dados vêm do "
              "manifest íntegro; só a regra de baixa confiança usa o params.yaml atual. Para valer no site: "
              "python -m src.data.build_demo --substituir-manifest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
