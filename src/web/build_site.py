"""Gerador do site (Item 11): uma pasta de páginas estáticas que funciona offline, a partir dos artefatos do manifest.

Só abre o que o manifest lista e confere o SHA-256 antes (`manifest.verificar_manifest`). Não calcula
SPI, não treina e não lê o teste congelado: apenas traduz `demo_observado`, `qualidade_municipio`, a
chuva do RAW e a malha do GEE num pacote de dados (SPI e chuva em inteiros de 16 bits, classes em
bytes), mais o contexto do El Niño (`contexto_enso.json`, transcrito de fontes oficiais), e monta as páginas.
Lógica de domínio e ML não dependem deste módulo.

Dados sintéticos existem só em `tests/`. As páginas carregam o aviso de MVP, as limitações e o
`dataset_sha256`; camadas sem dado oficial validado aparecem como "Indisponível".

Uso (PowerShell, depois de `python -m src.data.build_demo`):
    python -m src.web.build_site                      # grava a pasta site/ (abra site/index.html)
    python -m src.web.build_site --saida outra_pasta
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import chirps_raw
from src.data import contexto_enso as ce
from src.data import manifest as mf
from src.data.config import Config, load_config
from src.web import destaques, marca, paginas

NA16 = -32768                 # "sem dado" nos inteiros de 16 bits
SEM_DADO = 255                # "sem dado" nas classes
ESCALA_SPI = 100              # SPI x 100: as duas casas que a interface mostra
ESCALA_CHUVA = 10             # mm x 10
LARGURA_MAPA = 1000.0         # unidades do viewBox do SVG
GEOJSON = "referencia/malha_pi_2022_gee.geojson"
RAIZ_WEB = Path(__file__).resolve().parent
ASSETS = RAIZ_WEB / "assets"
FONTE_CORPO = ASSETS / "AtkinsonHyperlegibleNext-latin-wght.woff2"
FONTE_MARCA = ASSETS / "Outfit-latin-wght.woff2"
SAIDA_PADRAO = Path("site")
JS_NUCLEO = ("core",)

# Classes do SPI (McKee et al. 1993; a mesma divisão que a OMM usa). "Em seca" = SPI <= -1,0.
# (id, rótulo, faixa mostrada, cor no tema claro, cor no tema escuro, é seca)
# Duas rampas da marca: terracota para a falta de chuva, azul-petróleo para o excesso, com o neutro no meio.
# O SPI-3 mais extremo é o mais escuro no tema claro e o mais claro no escuro (validado com o guia de visualização).
CLASSES = (
    ("seca_extrema", "Seca extrema", "−2,0 ou menos", "#6E1D0B", "#FF9068", True),
    ("seca_severa", "Seca severa", "−1,99 a −1,5", marca.TERRACOTA, "#D0603E", True),
    ("seca_moderada", "Seca moderada", "−1,49 a −1,0", "#E39A78", "#8A4A38", True),
    ("normal", "Próximo do normal", "−0,99 a 0,99", "#E8E1D3", "#2A444B", False),
    ("umido_moderado", "Úmido moderado", "1,0 a 1,49", "#78B6BF", "#1F5A63", False),
    ("muito_umido", "Muito úmido", "1,5 a 1,99", marca.PETROLEO, "#2E93A0", False),
    ("umido_extremo", "Extremamente úmido", "2,0 ou mais", "#073F47", "#74D4E0", False),
)
LIMITES = (-2.0, -1.5, -1.0, 1.0, 1.5, 2.0)   # seca: SPI <= limite; úmido: SPI >= limite


class SiteError(ValueError):
    """Entrada que não permite gerar o site (grade incompleta, malha inválida, modelo sem marcador)."""


# ------------------------------------------------------------------ classes e codificação
def classificar(spi) -> np.ndarray:
    """Classe 0..6 (seca extrema .. úmido extremo) ou 255 (sem dado), a partir do SPI sem arredondar."""
    a = np.asarray(spi, dtype="float64")
    out = np.full(a.shape, SEM_DADO, dtype="uint8")
    ok = ~np.isnan(a)
    v = a[ok]
    out[ok] = np.select([v <= -2.0, v <= -1.5, v <= -1.0, v < 1.0, v < 1.5, v < 2.0],
                        [0, 1, 2, 3, 4, 5], default=6).astype("uint8")
    return out


def inteiros16(valores, escala: int) -> np.ndarray:
    """Valor x escala arredondado em int16 (NaN vira NA16). Estoura = erro, nunca satura em silêncio."""
    a = np.asarray(valores, dtype="float64")
    ok = ~np.isnan(a)
    r = np.full(a.shape, NA16, dtype="int16")
    q = np.round(a[ok] * escala)
    if q.size and (q.min() <= NA16 or q.max() > np.iinfo("int16").max):
        raise SiteError(f"valor fora do intervalo de int16 com escala {escala}")
    r[ok] = q.astype("int16")
    return r


def em_base64(a: np.ndarray) -> str:
    """Bytes little-endian em base64 (é o que `Int16Array`/`Uint8Array` leem nos navegadores)."""
    return base64.b64encode(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<"), copy=False).tobytes()).decode("ascii")


# ------------------------------------------------------------------ malha
def _aneis(geom: dict):
    t = geom.get("type")
    if t == "Polygon":
        return [geom["coordinates"]]
    if t == "MultiPolygon":
        return list(geom["coordinates"])
    raise SiteError(f"geometria não suportada: {t!r}")


def _numero(v: float) -> str:
    s = f"{v:.1f}"
    return s[:-2] if s.endswith(".0") else s


def projetar_malha(geojson: dict, largura: float = LARGURA_MAPA) -> tuple[list[dict], tuple[float, float]]:
    """Projeção equiretangular (corrigida pelo cosseno da latitude central) para um viewBox de `largura`.

    Devolve (municípios com `cod`, `nome`, caminho SVG `d` e centro lon/lat), (largura, altura)."""
    feats = geojson.get("features") or []
    if not feats:
        raise SiteError("malha sem municípios")
    pts = [p for f in feats for poly in _aneis(f["geometry"]) for anel in poly for p in anel]
    lons, lats = [p[0] for p in pts], [p[1] for p in pts]
    lon_min, lon_max, lat_min, lat_max = min(lons), max(lons), min(lats), max(lats)
    k = math.cos(math.radians((lat_min + lat_max) / 2))
    esc = largura / ((lon_max - lon_min) * k)
    altura = (lat_max - lat_min) * esc

    def xy(p):
        return ((p[0] - lon_min) * k * esc, (lat_max - p[1]) * esc)

    saida = []
    for f in feats:
        cod = str(f["properties"]["cod_ibge"])
        partes, cx, cy = [], [], []
        for poly in _aneis(f["geometry"]):
            for anel in poly:
                q, anterior = [], None
                for p in anel:
                    a, b = _numero(xy(p)[0]), _numero(xy(p)[1])
                    if (a, b) != anterior:
                        q.append((a, b))
                    anterior = (a, b)
                if len(q) > 1 and q[0] == q[-1]:
                    q.pop()
                if len(q) >= 3:
                    partes.append("M" + q[0][0] + " " + q[0][1] + "L" + " ".join(f"{a} {b}" for a, b in q[1:]) + "Z")
            cx += [p[0] for p in poly[0]]
            cy += [p[1] for p in poly[0]]
        if not partes:
            raise SiteError(f"município {cod} sem polígono válido")
        saida.append({"cod": cod, "nome": str(f["properties"].get("nome", cod)), "d": "".join(partes),
                      "lon": (min(cx) + max(cx)) / 2, "lat": (min(cy) + max(cy)) / 2})
    return saida, (round(float(largura), 1), round(float(altura), 1))


# ------------------------------------------------------------------ insumos e pacote de dados
@dataclass(frozen=True)
class Insumos:
    demo: pd.DataFrame
    qualidade: pd.DataFrame
    raw: pd.DataFrame
    malha: dict
    manifest: dict
    extraido_em: tuple[str, str] | None
    enso: ce.ContextoEnso | None = None


def _grade(df: pd.DataFrame, cods: list[str], meses: pd.PeriodIndex, colunas: list[str], onde: str) -> pd.DataFrame:
    """Reindexa para (município x mês) e exige a grade inteira: sem linha faltando, sobrando ou repetida."""
    alvo = pd.MultiIndex.from_product([cods, meses.to_timestamp()], names=["cod_ibge", "mes"])
    x = df.set_index(["cod_ibge", "mes"])
    if len(x) != len(alvo) or x.index.has_duplicates or not x.index.isin(alvo).all():
        raise SiteError(f"{onde}: a grade município x mês não bate com a malha e a janela "
                        f"({len(x)} linhas, esperado {len(alvo)})")
    return x.reindex(alvo)[colunas]


def montar_payload(ins: Insumos, cfg: Config, *, verificado_em: datetime) -> dict:
    """Pacote JSON puro que o template desenha. Nada é inventado: tudo vem dos insumos conferidos."""
    municipios, (larg, alt) = projetar_malha(ins.malha)
    municipios.sort(key=lambda m: m["cod"])
    cods = [m["cod"] for m in municipios]
    if len(set(cods)) != len(cods):
        raise SiteError("códigos IBGE repetidos na malha")
    meses = pd.period_range(cfg.janela.inicio, cfg.janela.fim, freq="M")
    nm = len(cods)
    escalas = [int(k) for k in cfg.spi.janelas_meses]

    cols = [c for k in escalas for c in (f"spi{k}", f"spi{k}_invalid", f"spi{k}_baixa_conf")]
    demo = _grade(ins.demo, cods, meses, cols, "demo_observado")
    chuva = _grade(ins.raw, cods, meses, ["precip_mm"], "RAW do CHIRPS")["precip_mm"].to_numpy("float64")

    q = ins.qualidade.set_index("cod_ibge")
    if set(q.index) != set(cods):
        raise SiteError("qualidade_municipio e a malha não têm os mesmos municípios")
    for m in municipios:
        row = q.loc[m["cod"]]
        m["exc"] = bool(row["excluded"])
        m["motivo"] = str(row["motivo"]) if m["exc"] else ""

    spi, cls, baixa = {}, {}, {}
    cls_mat, baixa_mat = {}, {}
    for k in escalas:
        v = demo[f"spi{k}"].to_numpy("float64")
        v = np.where(demo[f"spi{k}_invalid"].to_numpy(bool), np.nan, v)
        q = inteiros16(v, ESCALA_SPI)
        # a classe é a do número que a tela mostra (2 casas): nunca "−1,00" junto de "próximo do normal"
        c = classificar(np.where(q == NA16, np.nan, q / ESCALA_SPI))
        b = (demo[f"spi{k}_baixa_conf"].to_numpy(bool) & ~np.isnan(v)).astype("uint8")
        cls[str(k)], spi[str(k)], baixa[str(k)] = em_base64(c), em_base64(q), em_base64(b)
        cls_mat[k], baixa_mat[k] = c.reshape(nm, len(meses)), b.reshape(nm, len(meses))

    ordem = sorted(range(nm), key=lambda i: (-municipios[i]["lat"], municipios[i]["lon"]))   # norte -> sul
    for m in municipios:
        del m["lon"], m["lat"]

    # destaques do site: só leituras observadas e confiáveis (ver destaques.py)
    alvo = int(cfg.alvo.escala_spi_meses)
    if alvo not in cls_mat:
        raise SiteError(f"escala_alvo {alvo} não está nas escalas {escalas}")
    t_abertura = destaques.mes_abertura(cls_mat[alvo], baixa_mat[alvo])
    maiores = [{**x, "mes": str(meses[x["t"]])} for x in destaques.maiores_secas(cls_mat[alvo], baixa_mat[alvo])]
    ultimo_valido = cls_mat[alvo][:, -1] != SEM_DADO
    pct_baixa_ultimo = int(round(100 * baixa_mat[alvo][:, -1][ultimo_valido].sum() / max(1, ultimo_valido.sum())))
    incluir = ~np.array([x["exc"] for x in municipios])
    chuva_mat = chuva.reshape(nm, len(meses))
    clima_mensal = destaques.climatologia_mensal(chuva_mat, meses, cfg.split.treino_fim, incluir)
    estacao = destaques.estacao_chuvosa(chuva_mat, meses, cfg.split.treino_fim, incluir)
    pct_estacao = round(100 * sum(clima_mensal[mm - 1] for mm in estacao["meses"]) / sum(clima_mensal), 1)

    m = ins.manifest
    demo_extras = (m.get("extras") or {}).get("demo", {})
    return {
        "n_municipios": nm,
        "abertura": int(t_abertura), "pct_baixa_ultimo": pct_baixa_ultimo, "maiores_secas": maiores,
        "clima": {"climatologia": clima_mensal, "estacao": estacao, "pct_estacao": pct_estacao,
                  "treino_fim": str(cfg.split.treino_fim)},
        "enso": ce.para_pacote(ins.enso) if ins.enso is not None else None,
        "titulo": cfg.projeto.nome,
        "equipe": cfg.projeto.equipe,
        "maturidade": cfg.projeto.maturidade,
        "viewBox": [larg, alt],
        "escalas": escalas,
        "escala_alvo": int(cfg.alvo.escala_spi_meses),
        "limites": list(LIMITES),
        "classes": [{"id": i, "rotulo": r, "faixa": f, "claro": cl, "escuro": ce, "seca": s}
                    for i, r, f, cl, ce, s in CLASSES],
        "meses": [str(p) for p in meses],
        "municipios": municipios,
        "ordem": ordem,
        "spi": spi, "cls": cls, "baixa": baixa,
        "regra_baixa": {"q_max": float(cfg.spi.q_baixa_confianca),
                        "media_min_mm_por_mes": float(cfg.spi.media_min_mm_por_mes)},
        "precip": em_base64(inteiros16(chuva, ESCALA_CHUVA)),
        "escala_spi": ESCALA_SPI, "escala_chuva": ESCALA_CHUVA,
        "fonte": {"chirps": m["fontes"]["chirps"].get("gee_id"), "versao_params": m["projeto"]["versao_params"],
                  "malha_ano": m["escopo"]["malha_municipal_ano"], "extraido_em": list(ins.extraido_em or [])},
        "periodo": {"inicio": m["janela"]["inicio"], "fim": m["janela"]["fim"],
                    "treino_fim": str(demo_extras.get("spi_calibrado_ate", m["split"].get("treino_fim", ""))),
                    "teste_fim": str(m["split"].get("teste_fim", ""))},
        "n_excluidos": int(sum(x["exc"] for x in municipios)),
        "verificacao": {"dataset_sha256": m["dataset_sha256"], "manifest_gerado_em": m["gerado_em_utc"],
                        "verificado_em": verificado_em.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "arquivos": len(m["arquivos"])},
    }


# ------------------------------------------------------------------ leitura (via manifest) e HTML
def _extracao(cfg: Config, nomes: list[str]) -> tuple[str, str] | None:
    datas = []
    for n in nomes:
        p = mf.caminho_verificado(cfg, n)
        selo = p.with_name(p.name + ".sha256.json")      # mesma convenção de `chirps_raw`
        if selo.is_file():
            d = (json.loads(selo.read_text(encoding="utf-8")).get("meta") or {}).get("extraido_em_utc")
            if d:
                datas.append(str(d))
    return (min(datas), max(datas)) if datas else None


def carregar_insumos(cfg: Config) -> Insumos:
    """Confere o manifest inteiro (SHA-256 de tudo, inclusive `params.yaml`) e lê só o que ele lista."""
    m = mf.verificar_manifest(cfg)
    lotes = sorted(n for n in m["arquivos"] if n.startswith("raw_chirps/"))
    if not lotes:
        raise SiteError("o manifest não lista nenhum lote do RAW")
    ler = lambda nome: pd.read_parquet(mf.caminho_verificado(cfg, nome), engine="pyarrow")   # noqa: E731
    if GEOJSON not in m["arquivos"]:
        raise SiteError(f"o manifest não lista {GEOJSON}")
    malha = json.loads(mf.caminho_verificado(cfg, GEOJSON).read_text(encoding="utf-8"))
    raw = pd.concat([chirps_raw.ler_lote(mf.caminho_verificado(cfg, n)) for n in lotes], ignore_index=True)
    arq_enso = cfg.raiz / ce.ARQUIVO_PADRAO          # fora do manifest: contexto de fonte externa, com a própria data
    enso = ce.carregar(arq_enso) if arq_enso.is_file() else None
    return Insumos(demo=ler("demo_observado"), qualidade=ler("qualidade_municipio"), raw=raw, malha=malha,
                   manifest=m, extraido_em=_extracao(cfg, lotes), enso=enso)


# ------------------------------------------------------------------ montagem da pasta do site
def _dados_js(payload: dict) -> str:
    j = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return "window.PR_DADOS=" + j.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029") + ";\n"


def _css(fonte_corpo: bytes, fonte_marca: bytes) -> str:
    css = (ASSETS / "site.css").read_text(encoding="utf-8")
    for marcador, valor in (("__FONTE_CORPO__", base64.b64encode(fonte_corpo).decode("ascii")),
                            ("__FONTE_MARCA__", base64.b64encode(fonte_marca).decode("ascii"))):
        if css.count(marcador) != 1:
            raise SiteError(f"site.css precisa ter {marcador} exatamente uma vez")
        css = css.replace(marcador, valor)
    return css


def montar_arquivos(payload: dict, lista: tuple[paginas.Pagina, ...] = paginas.PAGINAS) -> dict[str, str]:
    """Caminho relativo -> conteúdo de texto de tudo o que a pasta do site leva. Sem escrever nada em disco."""
    arquivos: dict[str, str] = {
        "assets/dados.js": _dados_js(payload),
        "assets/site.css": _css(FONTE_CORPO.read_bytes(), FONTE_MARCA.read_bytes()),
        "assets/marca/gota.svg": marca.svg_isolado(),
    }
    scripts = {*JS_NUCLEO, *(s for p in lista for s in p.scripts)}
    for s in sorted(scripts):
        arquivos[f"assets/{s}.js"] = (ASSETS / f"{s}.js").read_text(encoding="utf-8")
    for p in lista:
        arquivos[p.arquivo] = paginas.renderizar_pagina(p, lista, payload)
    return arquivos


def _gravar_atomico(destino: Path, texto: str) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=destino.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(texto)
        os.chmod(tmp, 0o644)
        os.replace(tmp, destino)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def gerar_site(cfg: Config, saida: Path | None = None, *, agora: datetime | None = None) -> dict:
    """Confere o manifest, monta o pacote e grava a pasta do site. Devolve um resumo (JSON puro)."""
    agora = agora or datetime.now(timezone.utc)
    ins = carregar_insumos(cfg)
    payload = montar_payload(ins, cfg, verificado_em=agora)
    arquivos = montar_arquivos(payload)                          # tudo montado na memória antes de gravar qualquer coisa
    destino = Path(saida) if saida else cfg.raiz / SAIDA_PADRAO
    if destino.suffix.lower() in (".html", ".htm"):
        raise SiteError(f"a saída agora é uma pasta com várias páginas, não um arquivo: {destino}")
    for rel, texto in arquivos.items():
        _gravar_atomico(destino / rel, texto)
    return {"pasta": str(destino), "inicio": str(destino / "index.html"), "arquivos": len(arquivos),
            "bytes": sum(len(t.encode("utf-8")) for t in arquivos.values()), "municipios": len(payload["municipios"]),
            "meses": len(payload["meses"]), "excluidos": payload["n_excluidos"],
            "dataset_sha256": payload["verificacao"]["dataset_sha256"], "maturity": payload["maturidade"],
            "enso": payload["enso"] is not None}


def main(argv: list[str] | None = None) -> int:       # python -m src.web.build_site
    ap = argparse.ArgumentParser(prog="python -m src.web.build_site",
                                 description="Gera o site (pasta de páginas estáticas, offline) a partir dos artefatos do manifest.")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--saida", default=None, help=f"pasta de saída; padrão: {SAIDA_PADRAO}")
    a = ap.parse_args(argv)
    try:
        r = gerar_site(load_config(a.params), Path(a.saida) if a.saida else None)
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    print(f"AVISO: {r['maturity']}")
    print(f"site gravado em {r['pasta']} ({r['arquivos']} arquivos, {r['bytes'] / 1e6:.2f} MB) | municípios={r['municipios']} "
          f"meses={r['meses']} excluídos={r['excluidos']} | dataset_sha256={r['dataset_sha256'][:16]}…")
    print(f"abra: {r['inicio']}" + ("" if r["enso"] else " | AVISO: sem data/reference/contexto_enso.json; a camada do El Niño aparece como indisponível"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
