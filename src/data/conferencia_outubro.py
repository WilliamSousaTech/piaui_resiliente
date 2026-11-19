"""Conferência INDICATIVA do mês previsto com chuva de outra fonte (Open-Meteo), sem tocar nos dados do modelo.

Por que existe: a previsão de 2026-10 (origem 2026-09) já é passado, mas o CHIRPS que treinou o modelo só vai até
2026-09. Este comando roda na máquina da equipe (o ambiente de desenvolvimento não alcança a API), busca a chuva
diária do centro de cada município na API de histórico da Open-Meteo (reanálise) e grava UM arquivo de referência,
`data/reference/conferencia_outubro.json`, que o site mostrará com data, fonte e aviso.

O que NÃO é: não é o acerto do modelo. A fonte é outra (reanálise, não CHIRPS), o ponto é o centro do município (não a
média dele) e a previsão do modelo é um SPI (índice), não milímetros. Por isso o arquivo traz:
  * `comparacao_fontes`: chuva das duas fontes em ago e set (meses em que as duas existem), para medir o viés;
  * `percentil` e `pct_normal` de outubro contra os outubros de 1981–2015 do CHIRPS (a normal do projeto);
  * `grupos`: municípios COM sinal de seca do modelo x SEM sinal. O viés da fonte pesa igual nos dois grupos, então a
    diferença entre eles é mais confiável que o valor absoluto de cada um.

Garantias: não grava Parquet, não altera o manifest nem o `dataset_sha256`; a previsão só é aceita se for do mesmo
dataset dos dados (mesma checagem do site). Dia sem dado nunca vira zero: o mês fica "incompleto" e sem número.
Sem ML e sem framework web. Uso (na raiz do projeto, com internet):
    python -m src.data.conferencia_outubro
"""
from __future__ import annotations

import argparse
import json
import math
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from shapely.geometry import shape

from src.data import chirps_raw
from src.data import manifest as mf
from src.data import previsao as prev_mod
from src.data.config import Config, ConfigError, load_config
from src.data.spi import SpiError, para_matriz

URL_API = "https://archive-api.open-meteo.com/v1/archive"
ATRIBUICAO = "Weather data by Open-Meteo.com (CC BY 4.0)"
SITE_FONTE = "https://open-meteo.com/"
VARIAVEL = "precipitation_sum"
FUSO = "America/Fortaleza"
ATRASO_DIAS = 5                       # docs da Open-Meteo: reanálise diária com 5 dias de atraso
GEOJSON = "referencia/malha_pi_2022_gee.geojson"      # chave no manifest (mesma que o site usa)
SAIDA_PADRAO = Path("data") / "reference" / "conferencia_outubro.json"

MINIMO_NORMAL_MM = 5.0                # abaixo disto a "% da normal" explode com poucos mm: só mostramos o desvio em mm
MINIMO_ANOS = 20                      # percentil só com pelo menos 20 anos de histórico
LIMIAR_SECO_PERCENTIL = 25.0          # "mês seco" = percentil <= 25
MAX_DIA_MM = 500.0                    # acima disto o valor diário é tratado como inválido
LAT_PI, LON_PI = (-11.0, -2.5), (-46.2, -40.2)        # caixa do Piauí: pega coordenada trocada ou fora do estado
USER_AGENT = "piaui-resiliente/conferencia (projeto academico; uso nao comercial)"

AVISOS = (
    "Conferência indicativa: a chuva vem de outra fonte (reanálise da Open-Meteo), não do CHIRPS usado no treino, "
    "e é de um ponto (o centro do município), não da média do município.",
    "A previsão do modelo é um índice (SPI) e esta conferência é chuva em milímetros. Por isso comparamos grupos "
    "(com sinal x sem sinal): o viés da fonte pesa igual nos dois, e a diferença entre eles é mais confiável.",
    "Não substitui a verificação com o CHIRPS do mês, quando ele estiver disponível. O teste congelado não foi aberto.",
)

Abrir = Callable[[str, float], "tuple[int, bytes]"]


class ConferenciaError(ValueError):
    """Entrada inválida, API recusou, resposta fora do formato ou previsão incompatível com os dados."""


@dataclass(frozen=True)
class Ponto:
    cod_ibge: str
    nome: str
    lat: float
    lon: float


# ---------------------------------------------------------------- pontos e período
def pontos_da_malha(malha: dict) -> list[Ponto]:
    """Um ponto por município: o centroide; se ele cair fora do polígono (forma côncava), um ponto interno."""
    if not isinstance(malha, dict) or not isinstance(malha.get("features"), list) or not malha["features"]:
        raise ConferenciaError("malha sem 'features'")
    pontos, vistos = [], set()
    for f in malha["features"]:
        try:
            cod = str(f["properties"]["cod_ibge"])
            nome = str(f["properties"].get("nome", cod))
            geom = shape(f["geometry"])
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise ConferenciaError(f"feição da malha inválida: {e!r}") from e
        if geom.is_empty or not geom.is_valid:
            raise ConferenciaError(f"geometria vazia ou inválida em {cod}")
        c = geom.centroid
        if not geom.contains(c):
            c = geom.representative_point()
        if not (LAT_PI[0] <= c.y <= LAT_PI[1] and LON_PI[0] <= c.x <= LON_PI[1]):
            raise ConferenciaError(f"{cod}: ponto (lat {c.y:.3f}, lon {c.x:.3f}) fora do Piauí; confira a ordem lon/lat")
        if cod in vistos:
            raise ConferenciaError(f"cod_ibge repetido na malha: {cod}")
        vistos.add(cod)
        pontos.append(Ponto(cod, nome, round(c.y, 4), round(c.x, 4)))
    return sorted(pontos, key=lambda p: p.cod_ibge)


def periodo_busca(alvo: pd.Period, hoje: date) -> tuple[date, date]:
    """Do 1º dia do 2º mês antes do alvo (ago, p/ out) ao último dia do alvo. Recusa se a reanálise ainda não chegou."""
    ini = (alvo - 2).start_time.date()
    fim = alvo.end_time.date()
    if fim > hoje - timedelta(days=ATRASO_DIAS):
        raise ConferenciaError(f"{alvo} termina em {fim} e a reanálise tem {ATRASO_DIAS} dias de atraso: "
                               f"tente a partir de {fim + timedelta(days=ATRASO_DIAS)}")
    return ini, fim


def datas_do_periodo(ini: date, fim: date) -> list[str]:
    return [(ini + timedelta(days=i)).isoformat() for i in range((fim - ini).days + 1)]


# ---------------------------------------------------------------- API
def montar_url(pontos: list[Ponto], ini: date, fim: date) -> str:
    if not pontos:
        raise ConferenciaError("nenhum ponto para consultar")
    q = {"latitude": ",".join(f"{p.lat:.4f}" for p in pontos), "longitude": ",".join(f"{p.lon:.4f}" for p in pontos),
         "start_date": ini.isoformat(), "end_date": fim.isoformat(), "daily": VARIAVEL, "timezone": FUSO}
    return URL_API + "?" + urllib.parse.urlencode(q, safe=",/")


def _numero(x):
    if x is None:
        return np.nan
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ConferenciaError(f"valor diário não numérico: {x!r}")
    return float(x)


def ler_resposta(corpo: bytes, n_pontos: int, datas: list[str]) -> tuple[list[np.ndarray], int]:
    """Valida a resposta e devolve, por ponto, o vetor diário (NaN onde faltou). Também conta valores inválidos.

    Um ponto => objeto; vários => lista (docs da Open-Meteo). As datas devolvidas têm de ser exatamente as pedidas.
    Negativo, infinito ou > MAX_DIA_MM vira NaN (e é contado); `null` vira NaN sem contar."""
    try:
        obj = json.loads(corpo)
    except ValueError as e:
        raise ConferenciaError("a resposta da Open-Meteo não é JSON") from e
    if isinstance(obj, dict) and obj.get("error"):
        raise ConferenciaError(f"a Open-Meteo recusou o pedido: {obj.get('reason', 'sem motivo informado')}")
    itens = obj if isinstance(obj, list) else [obj]
    if len(itens) != n_pontos:
        raise ConferenciaError(f"pedi {n_pontos} ponto(s) e vieram {len(itens)}")
    series, invalidos = [], 0
    for i, it in enumerate(itens):
        d = it.get("daily") if isinstance(it, dict) else None
        if not isinstance(d, dict) or d.get("time") != datas:
            raise ConferenciaError(f"ponto {i}: as datas devolvidas não são as pedidas")
        v = d.get(VARIAVEL)
        if not isinstance(v, list) or len(v) != len(datas):
            raise ConferenciaError(f"ponto {i}: '{VARIAVEL}' ausente ou com tamanho errado")
        arr = np.array([_numero(x) for x in v], dtype="float64")
        ruim = ~np.isnan(arr) & (~np.isfinite(arr) | (arr < 0) | (arr > MAX_DIA_MM))
        invalidos += int(ruim.sum())
        arr[ruim] = np.nan
        series.append(arr)
    return series, invalidos


def _abrir_urllib(url: str, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:      # verificação de certificado: padrão, ligada
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read()
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            raise ConferenciaError("o certificado HTTPS da Open-Meteo não foi aceito. Não desligue a verificação: "
                                   "copie esta mensagem e envie ao chat") from e
        raise ConferenciaError(f"sem conexão com a Open-Meteo: {e.reason}") from e
    except TimeoutError as e:
        raise ConferenciaError("a Open-Meteo não respondeu a tempo") from e


def _motivo(corpo: bytes) -> str:
    try:
        obj = json.loads(corpo)
        if isinstance(obj, dict) and obj.get("reason"):
            return str(obj["reason"])
    except ValueError:
        pass
    return corpo[:120].decode("utf-8", "replace")


def _buscar_lote(pontos, ini, fim, datas, abrir, timeout, tentativas, pausa, dormir):
    url = montar_url(pontos, ini, fim)
    ultimo = "sem tentativa"
    for n in range(1, tentativas + 1):
        status, corpo = abrir(url, timeout)
        if status == 200:
            return ler_resposta(corpo, len(pontos), datas)
        ultimo = f"HTTP {status}: {_motivo(corpo)}"
        if status == 429:                       # limite por minuto: espera um minuto cheio, cada vez mais
            if n < tentativas:
                dormir(60.0 * n)
        elif status >= 500:
            if n < tentativas:
                dormir(pausa * 2 ** n)
        else:                                   # 400 etc.: pedido errado, repetir não adianta
            raise ConferenciaError(f"a Open-Meteo recusou o pedido ({ultimo})")
    raise ConferenciaError(f"a Open-Meteo não respondeu após {tentativas} tentativas ({ultimo}); "
                           f"espere alguns minutos e rode de novo")


def buscar(pontos: list[Ponto], ini: date, fim: date, *, abrir: Abrir = _abrir_urllib, lote: int = 20,
           pausa: float = 10.0, timeout: float = 60.0, tentativas: int = 4,
           dormir: Callable[[float], None] = time.sleep) -> tuple[dict[str, np.ndarray], int]:
    """Chuva diária por município, em lotes (o peso exato por chamada não está nos docs: espaçamos e repetimos após 429)."""
    if lote < 1 or tentativas < 1:
        raise ConferenciaError("lote e tentativas precisam ser >= 1")
    datas = datas_do_periodo(ini, fim)
    saida, invalidos = {}, 0
    for k in range(0, len(pontos), lote):
        if k:
            dormir(pausa)
        grupo = pontos[k:k + lote]
        series, inv = _buscar_lote(grupo, ini, fim, datas, abrir, timeout, tentativas, pausa, dormir)
        invalidos += inv
        for p, s in zip(grupo, series):
            saida[p.cod_ibge] = s
    return saida, invalidos


# ---------------------------------------------------------------- números
def soma_mensal(datas: list[str], valores: np.ndarray) -> dict[str, dict]:
    """Soma por mês. Só vale (`mm` não nulo) se TODOS os dias do mês têm valor: dia faltante nunca vira zero."""
    if len(datas) != len(valores):
        raise ConferenciaError("datas e valores com tamanhos diferentes")
    meses = pd.PeriodIndex(pd.DatetimeIndex(datas), freq="M")
    saida = {}
    for m in meses.unique():
        sel = np.asarray(meses == m)
        v = valores[sel]
        ok = int(np.isfinite(v).sum())
        saida[str(m)] = {"mm": round(float(v.sum()), 1) if ok == len(v) else None, "dias_validos": ok,
                         "dias_no_mes": int(len(v))}
    return saida


def percentil_historico(valor: float | None, historico: np.ndarray) -> float | None:
    """Posição de `valor` entre os anos do histórico (0–100; baixo = seco). Empate conta metade. Poucos anos => None."""
    h = np.asarray(historico, dtype="float64")
    h = h[np.isfinite(h)]
    if valor is None or not math.isfinite(valor) or len(h) < MINIMO_ANOS:
        return None
    return round(100.0 * (float((h < valor).sum()) + 0.5 * float((h == valor).sum())) / len(h), 1)


def _mediana(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 1) if xs else None


def comparar_fontes(registros: list[dict], meses: list[str]) -> dict:
    """Viés da Open-Meteo contra o CHIRPS nos meses em que as duas existem (município a município)."""
    saida = {}
    for m in meses:
        pares = [(r["chuva_mm"].get(m), r["chirps_mm"].get(m)) for r in registros]
        pares = [(a, b) for a, b in pares if a is not None and b is not None]
        if not pares:
            saida[m] = {"n": 0, "vies_medio_mm": None, "erro_medio_abs_mm": None, "correlacao": None,
                        "open_meteo_medio_mm": None, "chirps_medio_mm": None}
            continue
        a, b = np.array([p[0] for p in pares]), np.array([p[1] for p in pares])
        corr = float(np.corrcoef(a, b)[0, 1]) if len(a) >= 3 and a.std() > 0 and b.std() > 0 else None
        saida[m] = {"n": len(pares), "vies_medio_mm": round(float((a - b).mean()), 1),
                    "erro_medio_abs_mm": round(float(np.abs(a - b).mean()), 1),
                    "correlacao": None if corr is None else round(corr, 2),
                    "open_meteo_medio_mm": round(float(a.mean()), 1), "chirps_medio_mm": round(float(b.mean()), 1)}
    return saida


def resumir_grupo(regs: list[dict]) -> dict:
    com_perc = [r["percentil"] for r in regs if r["percentil"] is not None]
    secos = sum(1 for p in com_perc if p <= LIMIAR_SECO_PERCENTIL)
    anom = [r["anomalia_mm"] for r in regs if r["anomalia_mm"] is not None]
    return {"n": len(regs), "n_com_dado": len(com_perc), "n_secos": secos,
            "fracao_secos": round(secos / len(com_perc), 3) if com_perc else None,
            "mediana_percentil": _mediana(com_perc), "mediana_anomalia_mm": _mediana(anom),
            "n_abaixo_da_normal": sum(1 for x in anom if x < 0)}


def montar_registros(cods: list[str], om_mensal: dict[str, dict[str, dict]], chirps: pd.DataFrame, alvo: pd.Period,
                     normal: pd.Series, historico: dict[str, np.ndarray], com_sinal: set[str]) -> list[dict]:
    """Um registro por município. `chirps` é a matriz (mês x cod) do projeto; `normal` = mediana do mês no treino."""
    meses_ref = [alvo - 2, alvo - 1]
    regs = []
    for cod in cods:
        om = om_mensal[cod]
        mm_alvo = om.get(str(alvo), {}).get("mm")
        nrm = normal.get(cod)
        nrm = None if nrm is None or not math.isfinite(float(nrm)) else float(nrm)
        ch = {}
        for m in meses_ref:
            v = chirps.at[m, cod] if (m in chirps.index and cod in chirps.columns) else np.nan
            ch[str(m)] = round(float(v), 1) if math.isfinite(float(v)) else None
        regs.append({
            "cod_ibge": cod,
            "chuva_mm": {m: d["mm"] for m, d in om.items()},
            "dias_validos_alvo": om.get(str(alvo), {}).get("dias_validos"),
            "chirps_mm": ch,
            "normal_mm": None if nrm is None else round(nrm, 1),
            "anomalia_mm": None if mm_alvo is None or nrm is None else round(mm_alvo - nrm, 1),
            "pct_normal": (round(100.0 * mm_alvo / nrm, 0) if mm_alvo is not None and nrm is not None
                           and nrm >= MINIMO_NORMAL_MM else None),
            "percentil": percentil_historico(mm_alvo, historico.get(cod, np.array([]))),
            "completo": mm_alvo is not None,
            "sinal_modelo": cod in com_sinal,
        })
    return regs


def montar_documento(regs: list[dict], *, alvo: pd.Period, ini: date, fim: date, hoje: date, dataset_sha256: str,
                     janela_historica: str, n_invalidos: int) -> dict:
    meses_ref = [str(alvo - 2), str(alvo - 1)]
    com = [r for r in regs if r["sinal_modelo"]]
    sem = [r for r in regs if not r["sinal_modelo"]]
    return {
        "versao": 1,
        "aviso": AVISOS[0],
        "avisos": list(AVISOS),
        "gerado_em": hoje.isoformat(),
        "mes_alvo": str(alvo),
        "previsao_dataset_sha256": dataset_sha256,
        "fonte_conferencia": {"nome": "Open-Meteo Historical Weather API (reanálise)", "url": SITE_FONTE,
                              "atribuicao": ATRIBUICAO, "licenca": "CC BY 4.0", "variavel": VARIAVEL,
                              "periodo": [ini.isoformat(), fim.isoformat()], "ponto": "centro do município",
                              "valores_diarios_invalidos": n_invalidos},
        "base_historica": {"fonte": "CHIRPS v2.0 (dados do projeto)", "periodo": janela_historica,
                           "normal": "mediana do mês-calendário no treino", "minimo_normal_mm": MINIMO_NORMAL_MM,
                           "minimo_anos_percentil": MINIMO_ANOS, "limiar_seco_percentil": LIMIAR_SECO_PERCENTIL},
        "comparacao_fontes": comparar_fontes(regs, meses_ref),
        "grupos": {"com_sinal": resumir_grupo(com), "sem_sinal": resumir_grupo(sem)},
        "n_municipios": len(regs),
        "n_incompletos": sum(1 for r in regs if not r["completo"]),
        "municipios": regs,
    }


def gravar_atomico(destino: Path, doc: dict) -> None:
    texto = json.dumps(doc, ensure_ascii=False, indent=1, allow_nan=False) + "\n"      # allow_nan=False: NaN seria erro
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    tmp.write_text(texto, encoding="utf-8")
    os.replace(tmp, destino)


# ---------------------------------------------------------------- leitura dos dados do projeto
def _carregar_projeto(cfg: Config, previsao_json: Path):
    """Confere o manifest inteiro, a previsão (mesmo dataset e mesmos municípios) e lê só o que ele lista."""
    m = mf.verificar_manifest(cfg)
    malha = json.loads(mf.caminho_verificado(cfg, GEOJSON).read_text(encoding="utf-8"))
    prev = prev_mod.carregar(previsao_json)
    prev_mod.conferir_dataset(prev, m["dataset_sha256"], [f["properties"]["cod_ibge"] for f in malha["features"]])
    lotes = sorted(n for n in m["arquivos"] if n.startswith("raw_chirps/"))
    if not lotes or "climatologia_precip" not in m["arquivos"]:
        raise ConferenciaError("o manifest não lista o RAW do CHIRPS ou a climatologia")
    raw = pd.concat([chirps_raw.ler_lote(mf.caminho_verificado(cfg, n)) for n in lotes], ignore_index=True)
    clim = pd.read_parquet(mf.caminho_verificado(cfg, "climatologia_precip"), engine="pyarrow")
    return m, malha, prev, raw, clim


_ERROS = (ConferenciaError, prev_mod.PrevisaoError, mf.ManifestError, ConfigError, SpiError, FileNotFoundError, OSError)


def main(argv: list[str] | None = None, *, saida: Callable[[str], None] = print, abrir: Abrir = _abrir_urllib,
         dormir: Callable[[float], None] = time.sleep, hoje: date | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.data.conferencia_outubro", description=__doc__.split("\n\n")[0])
    ap.add_argument("--params", type=Path, default=Path("params.yaml"))
    ap.add_argument("--previsao", type=Path, default=prev_mod.ARQUIVO_PADRAO)
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    ap.add_argument("--lote", type=int, default=20, help="municípios por pedido")
    ap.add_argument("--pausa", type=float, default=10.0, help="segundos entre pedidos")
    ap.add_argument("--tentativas", type=int, default=4)
    a = ap.parse_args(argv)
    try:
        hoje = hoje or datetime.now(ZoneInfo(FUSO)).date()
        cfg = load_config(a.params)
        _, malha, prev, raw, clim = _carregar_projeto(cfg, a.previsao)
        alvo = pd.Period(prev.alvo, freq="M")
        ini, fim = periodo_busca(alvo, hoje)
        pontos = pontos_da_malha(malha)
        saida(f"Buscando chuva diária de {ini} a {fim} para {len(pontos)} municípios (Open-Meteo)…")
        series, invalidos = buscar(pontos, ini, fim, abrir=abrir, lote=a.lote, pausa=a.pausa, tentativas=a.tentativas,
                                   dormir=dormir)
        datas = datas_do_periodo(ini, fim)
        om_mensal = {c: soma_mensal(datas, s) for c, s in series.items()}
        matriz = para_matriz(raw[["cod_ibge", "mes", "precip_mm"]])
        cods = [p.cod_ibge for p in pontos]
        hist_idx = [m for m in matriz.index if m.month == alvo.month and m <= cfg.split.treino_fim]
        historico = {c: matriz.loc[hist_idx, c].dropna().to_numpy() for c in cods if c in matriz.columns}
        normal = clim[clim["mes_calendario"] == alvo.month].set_index("cod_ibge")["mediana_mm"]
        com_sinal = {m.cod_ibge for m in prev.municipios if m.spi3.alerta or m.spi6.alerta}
        regs = montar_registros(cods, om_mensal, matriz, alvo, normal, historico, com_sinal)
        doc = montar_documento(regs, alvo=alvo, ini=ini, fim=fim, hoje=hoje, dataset_sha256=prev.dataset_sha256,
                               janela_historica=f"{cfg.janela.inicio} a {cfg.split.treino_fim}", n_invalidos=invalidos)
        gravar_atomico(a.saida, doc)
    except _ERROS as e:
        saida(f"ERRO: {e}")
        return 2
    g = doc["grupos"]
    saida(f"Pronto: {a.saida} ({doc['n_municipios']} municípios, {doc['n_incompletos']} com dia faltante).")
    for nome, rot in (("com_sinal", "com sinal de seca"), ("sem_sinal", "sem sinal")):
        x = g[nome]
        saida(f"  {rot}: {x['n']} municípios, {x['n_secos']} de {x['n_com_dado']} com outubro seco "
              f"(percentil <= {LIMIAR_SECO_PERCENTIL:.0f}); percentil mediano {x['mediana_percentil']}")
    for m, x in doc["comparacao_fontes"].items():
        saida(f"  {m}: viés Open-Meteo − CHIRPS = {x['vies_medio_mm']} mm (n={x['n']}, correlação {x['correlacao']})")
    saida("Envie o arquivo ao chat ou faça: git add data/reference/conferencia_outubro.json && git commit && git push")
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
