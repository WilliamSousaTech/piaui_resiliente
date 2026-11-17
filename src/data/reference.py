"""Item 2b — referência territorial: municípios do escopo + malha IBGE 2022.

Lê SÓ arquivos locais oficiais (data/reference/fonte/), sem rede:
  - PI_Municipios_2022.zip              (IBGE, EPSG:4674, 224 municípios do PI)
  - semiarido_res150_2021.csv           (anexo da Res. CONDEL/SUDENE 150/2021: 1.427)
  - semiarido_transicao_res176_2024.csv (municípios mantidos pela Res. 176/2024)

Classifica cada município do PI em res150 | transicao | fora, marca `no_escopo`
e grava (atômico, imutável, com selo SHA-256):
  - <escopo.arquivo_semiarido>  CSV só do escopo (cod_ibge lido como TEXTO)
  - malha_pi_2022.parquet       GeoParquet, EPSG:4674, todos os 224
  - malha_pi_2022_gee.geojson   só o escopo, EPSG:4326, simplificado (payload do GEE)
  - referencia.sha256.json      selo do conjunto

Domínio puro: sem rede, sem GEE, sem framework web.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.data.chirps_raw import sha256_arquivo
from src.data.config import Config, exigir_resolvido

ZIP_MALHA = "PI_Municipios_2022.zip"
CSV_ANEXO = "semiarido_res150_2021.csv"
CSV_TRANSICAO = "semiarido_transicao_res176_2024.csv"
PARQUET = "malha_pi_2022.parquet"
GEOJSON = "malha_pi_2022_gee.geojson"
SELO = "referencia.sha256.json"

N_ANEXO, N_ANEXO_PI, N_MUN_PI = 1427, 215, 224
SITUACOES = ("res150", "transicao", "fora")
COLS_MALHA = ("CD_MUN", "NM_MUN", "SIGLA_UF", "AREA_KM2")
COLS_CSV_ESCOPO = ("cod_ibge", "nome", "uf", "situacao", "area_km2")
VAR_AREA_MAX = 0.01  # |Δárea| relativa máxima por município após simplificar
_CRS_AREA = "EPSG:5880"  # SIRGAS 2000 / Brazil Polyconic (área em m²)


class RefSchemaError(ValueError):
    """Fonte ou saída fora do formato esperado."""


class RefImutavelError(RuntimeError):
    """Já existe referência gerada com entradas/parâmetros diferentes."""


class RefIntegridadeError(RuntimeError):
    """Cruzamento entre fontes inconsistente, ou arquivo gravado não bate com o selo."""


@dataclass(frozen=True)
class ResultadoReferencia:
    status: str  # "gerado" | "ja_existente"
    contagens: dict
    conteudo_sha256: str
    simplificacao: dict


def _falhar(exc: type[Exception], msg: str, exemplos=()) -> None:
    ex = list(exemplos)[:3]
    raise exc(msg + (f" (ex.: {', '.join(map(str, ex))})" if ex else ""))


def normalizar_nome(nome: str) -> str:
    """Minúsculas, sem acento e só [a-z0-9]: compara 'ÁGUA BRANCA' com 'Água Branca'."""
    s = unicodedata.normalize("NFKD", nome)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---------------------------------------------------------------- leitura
def ler_malha_ibge(zip_path: Path, crs_esperado: str) -> gpd.GeoDataFrame:
    """Malha IBGE → colunas cod_ibge, nome, uf, area_km2, geometry (validada)."""
    p = Path(zip_path)
    if not p.is_file():
        raise FileNotFoundError(f"malha IBGE não encontrada: {p}")
    g = gpd.read_file(str(p))
    faltam = [c for c in COLS_MALHA if c not in g.columns]
    if faltam:
        _falhar(RefSchemaError, f"malha sem colunas {faltam}; tem {list(g.columns)}")
    if g.crs is None or g.crs.to_string() != crs_esperado:
        _falhar(RefSchemaError, f"CRS da malha {g.crs} != {crs_esperado}")
    out = gpd.GeoDataFrame(
        {
            "cod_ibge": g["CD_MUN"].astype(str),
            "nome": g["NM_MUN"].astype(str),
            "uf": g["SIGLA_UF"].astype(str),
            "area_km2": pd.to_numeric(g["AREA_KM2"], errors="coerce").astype("float64"),
        },
        geometry=g.geometry.values,
        crs=g.crs,
    )
    mal = ~out["cod_ibge"].str.fullmatch(r"22\d{5}")
    if mal.any():
        _falhar(RefSchemaError, "CD_MUN deve ter 7 dígitos começando em 22", out.loc[mal, "cod_ibge"])
    if out["cod_ibge"].duplicated().any():
        _falhar(RefSchemaError, "CD_MUN duplicado", out.loc[out["cod_ibge"].duplicated(), "cod_ibge"])
    if len(out) != N_MUN_PI:
        _falhar(RefSchemaError, f"malha tem {len(out)} municípios; esperado {N_MUN_PI} (PI)")
    if (out["uf"] != "PI").any():
        _falhar(RefSchemaError, "SIGLA_UF diferente de PI")
    if not (out["area_km2"].notna() & (out["area_km2"] > 0)).all():
        _falhar(RefSchemaError, "AREA_KM2 ausente ou <= 0", out.loc[~(out["area_km2"] > 0), "cod_ibge"])
    geom = out.geometry
    if geom.isna().any() or geom.is_empty.any():
        _falhar(RefSchemaError, "geometria nula ou vazia", out.loc[geom.isna() | geom.is_empty, "cod_ibge"])
    if not geom.is_valid.all():
        _falhar(RefSchemaError, "geometria inválida", out.loc[~geom.is_valid, "cod_ibge"])
    if not geom.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        _falhar(RefSchemaError, "geometria que não é (Multi)Polygon")
    return out


def ler_lista(caminho: Path, extras: tuple[str, ...] = ()) -> pd.DataFrame:
    """CSV UTF-8 com colunas cod_ibge, uf, nome (+extras). Tudo como texto (BOM é tolerado)."""
    p = Path(caminho)
    if not p.is_file():
        raise FileNotFoundError(f"lista não encontrada: {p}")
    cols = ["cod_ibge", "uf", "nome", *extras]
    try:
        df = pd.read_csv(p, dtype=str, encoding="utf-8", keep_default_na=False)
    except UnicodeDecodeError as e:
        raise RefSchemaError(f"{p.name} fora de UTF-8") from e
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        raise RefSchemaError(f"{p.name} vazio ou malformado: {e}") from e
    if list(df.columns) != cols:
        _falhar(RefSchemaError, f"{p.name}: colunas {list(df.columns)} != {cols}")
    if not df["cod_ibge"].str.fullmatch(r"\d{7}").all():
        _falhar(RefSchemaError, f"{p.name}: cod_ibge deve ter 7 dígitos", df.loc[~df["cod_ibge"].str.fullmatch(r"\d{7}"), "cod_ibge"])
    if df["cod_ibge"].duplicated().any():
        _falhar(RefSchemaError, f"{p.name}: cod_ibge duplicado", df.loc[df["cod_ibge"].duplicated(), "cod_ibge"])
    if not df["uf"].str.fullmatch(r"[A-Z]{2}").all() or (df["nome"].str.strip() == "").any():
        _falhar(RefSchemaError, f"{p.name}: uf inválida ou nome vazio")
    incoerente = (df["cod_ibge"].str[:2] == "22") != (df["uf"] == "PI")
    if incoerente.any():
        _falhar(RefSchemaError, f"{p.name}: prefixo 22 incoerente com uf=PI", df.loc[incoerente, "cod_ibge"])
    return df


# ---------------------------------------------------------- classificação
def classificar(malha: gpd.GeoDataFrame, anexo: pd.DataFrame, transicao: pd.DataFrame,
                incluir_transicao: bool, polo: str) -> gpd.GeoDataFrame:
    """Acrescenta `situacao` (res150|transicao|fora) e `no_escopo`; cruza as fontes."""
    if len(anexo) != N_ANEXO:
        _falhar(RefIntegridadeError, f"anexo Res. 150 tem {len(anexo)} linhas; esperado {N_ANEXO}")
    anexo_pi = anexo[anexo["uf"] == "PI"]
    if len(anexo_pi) != N_ANEXO_PI:
        _falhar(RefIntegridadeError, f"anexo tem {len(anexo_pi)} do PI; esperado {N_ANEXO_PI}")
    repetidos = set(anexo["cod_ibge"]) & set(transicao["cod_ibge"])
    if repetidos:
        _falhar(RefIntegridadeError, "código de transição já está no anexo da Res. 150", sorted(repetidos))
    comp = pd.concat([anexo_pi.assign(situacao="res150"), transicao.assign(situacao="transicao")])
    nome_malha = dict(zip(malha["cod_ibge"], malha["nome"]))
    ausentes = comp.loc[~comp["cod_ibge"].isin(nome_malha), "cod_ibge"]
    if len(ausentes):
        _falhar(RefIntegridadeError, "código das listas ausente na malha IBGE 2022", ausentes)
    diverge = [
        f"{c}: lista='{n}' malha='{nome_malha[c]}'"
        for c, n in zip(comp["cod_ibge"], comp["nome"])
        if normalizar_nome(n) != normalizar_nome(nome_malha[c])
    ]
    if diverge:
        _falhar(RefIntegridadeError, f"{len(diverge)} nome(s) divergem entre lista e malha", diverge)
    out = malha.copy()
    out["situacao"] = out["cod_ibge"].map(dict(zip(comp["cod_ibge"], comp["situacao"]))).fillna("fora").astype(str)
    out["no_escopo"] = out["situacao"].eq("res150") | (incluir_transicao & out["situacao"].eq("transicao"))
    if polo not in set(out.loc[out["no_escopo"], "cod_ibge"]):
        _falhar(RefIntegridadeError, f"polo do piloto {polo} fora do escopo")
    return out


def simplificar_gee(escopo: gpd.GeoDataFrame, crs_gee: str, tolerancia_graus: float):
    """Reprojeta p/ crs_gee e simplifica (topologia preservada). Devolve (gdf, métricas)."""
    if not (0 < tolerancia_graus <= 0.01):
        _falhar(RefSchemaError, f"tolerancia_graus fora de (0, 0.01]: {tolerancia_graus}")
    g0 = escopo.to_crs(crs_gee)
    g1 = g0.copy()
    g1["geometry"] = g0.geometry.simplify(tolerancia_graus, preserve_topology=True)
    geom = g1.geometry
    if geom.isna().any() or geom.is_empty.any() or not geom.is_valid.all():
        _falhar(RefIntegridadeError, "simplificação gerou geometria vazia ou inválida")
    if not geom.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        _falhar(RefIntegridadeError, "simplificação mudou o tipo da geometria")
    a0 = g0.geometry.to_crs(_CRS_AREA).area
    var = ((g1.geometry.to_crs(_CRS_AREA).area - a0).abs() / a0)
    if var.max() > VAR_AREA_MAX:
        _falhar(RefIntegridadeError, f"simplificação alterou área > {VAR_AREA_MAX:.0%}", g1.loc[var > VAR_AREA_MAX, "cod_ibge"])
    v0, v1 = int(g0.geometry.count_coordinates().sum()), int(geom.count_coordinates().sum())
    if v1 >= v0:
        _falhar(RefIntegridadeError, f"simplificação não reduziu vértices ({v0} -> {v1})")
    return g1, {"vertices_antes": v0, "vertices_depois": v1, "var_area_max": float(var.max())}


def hash_conteudo(g: gpd.GeoDataFrame) -> str:
    """SHA-256 canônico (cod_ibge, situacao, WKB little-endian 2D), independente de formato."""
    g = g.sort_values("cod_ibge")
    h = hashlib.sha256()
    wkbs = g.geometry.to_wkb(byte_order=1, output_dimension=2)
    for cod, sit, wkb in zip(g["cod_ibge"], g["situacao"], wkbs):
        h.update(f"{cod}|{sit}|".encode())
        h.update(wkb)
    return h.hexdigest()


# --------------------------------------------------------------- gravação
def _destinos(cfg: Config) -> dict[str, Path]:
    d = cfg.caminho("reference")
    csv = cfg.raiz / cfg.escopo.arquivo_semiarido
    return {csv.name: csv, PARQUET: d / PARQUET, GEOJSON: d / GEOJSON, SELO: d / SELO}


def _atomico(destino: Path, escrever) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    try:
        escrever(tmp)
        os.replace(tmp, destino)
    finally:
        if tmp.exists():
            tmp.unlink()


def _verificar_arquivos(selo: dict, destinos: dict[str, Path]) -> None:
    for nome, esperado in selo["arquivos"].items():
        p = destinos.get(nome)
        if p is None or not p.is_file():
            _falhar(RefIntegridadeError, f"arquivo do selo ausente: {nome}")
        if sha256_arquivo(p) != esperado:
            _falhar(RefIntegridadeError, f"{nome}: SHA-256 difere do selo (arquivo alterado)")


def _ler_selo(destinos: dict[str, Path]) -> dict:
    try:
        selo = json.loads(destinos[SELO].read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RefIntegridadeError(f"selo ilegível: {e}") from e
    if not isinstance(selo.get("arquivos"), dict):
        raise RefIntegridadeError("selo sem campo 'arquivos'")
    return selo


def gerar_referencia(cfg: Config, *, incluir_transicao: bool | None = None,
                     tolerancia_graus: float | None = None) -> ResultadoReferencia:
    """Gera (ou confirma) a referência territorial. Idempotente; nunca sobrescreve.

    Sem argumentos, `incluir_transicao` e `tolerancia_graus` vêm de `cfg.escopo`
    (params.yaml); argumento explícito prevalece.
    """
    if incluir_transicao is None:
        incluir_transicao = cfg.escopo.incluir_transicao
    if tolerancia_graus is None:
        tolerancia_graus = cfg.escopo.simplificacao_graus
    exigir_resolvido(cfg, "escopo.malha_municipal_ano")
    if cfg.escopo.malha_municipal_ano != 2022:
        _falhar(RefSchemaError, f"malha_municipal_ano={cfg.escopo.malha_municipal_ano}; este módulo lê 2022")
    fonte = cfg.caminho("reference") / "fonte"
    entradas = {n: sha256_arquivo(fonte / n) for n in (ZIP_MALHA, CSV_ANEXO, CSV_TRANSICAO) if (fonte / n).is_file()}
    faltam = [n for n in (ZIP_MALHA, CSV_ANEXO, CSV_TRANSICAO) if n not in entradas]
    if faltam:
        raise FileNotFoundError(f"faltam em {fonte}: {faltam}")
    params = {"incluir_transicao": bool(incluir_transicao), "tolerancia_graus": float(tolerancia_graus),
              "malha_ano": 2022, "crs_origem": cfg.escopo.crs_origem, "crs_gee": cfg.escopo.crs_gee,
              "polo": cfg.escopo.piloto.polo_cod_ibge}

    destinos = _destinos(cfg)
    existentes = [n for n, p in destinos.items() if p.exists()]
    malha = ler_malha_ibge(fonte / ZIP_MALHA, cfg.escopo.crs_origem)
    cls = classificar(malha, ler_lista(fonte / CSV_ANEXO), ler_lista(fonte / CSV_TRANSICAO, ("fonte",)),
                      incluir_transicao, cfg.escopo.piloto.polo_cod_ibge)
    esc = cls[cls["no_escopo"]]
    gee, simp = simplificar_gee(esc[["cod_ibge", "nome", "situacao", "geometry"]], cfg.escopo.crs_gee, tolerancia_graus)
    contagens = {"malha": len(cls), "escopo": len(esc), **{s: int((cls["situacao"] == s).sum()) for s in SITUACOES}}
    conteudo = hash_conteudo(cls)

    if existentes:
        if len(existentes) != len(destinos):
            _falhar(RefIntegridadeError, f"estado parcial em reference/: existem {existentes}, faltam {sorted(set(destinos) - set(existentes))}")
        selo = _ler_selo(destinos)
        _verificar_arquivos(selo, destinos)
        if selo.get("entradas") != entradas or selo.get("parametros") != params:
            raise RefImutavelError("referência já gravada com entradas/parâmetros diferentes; "
                                   "apague data/reference/ conscientemente e regenere")
        return ResultadoReferencia("ja_existente", selo["contagens"], selo["conteudo_sha256"], selo["simplificacao"])

    csv = pd.DataFrame(esc[list(COLS_CSV_ESCOPO)]).sort_values("cod_ibge")
    csv["area_km2"] = csv["area_km2"].round(3)
    nome_csv = next(n for n in destinos if n not in (PARQUET, GEOJSON, SELO))
    _atomico(destinos[nome_csv], lambda t: csv.to_csv(t, index=False, encoding="utf-8", lineterminator="\n"))
    _atomico(destinos[PARQUET], lambda t: cls.to_parquet(t, index=False))
    _atomico(destinos[GEOJSON], lambda t: gee.to_file(t, driver="GeoJSON"))
    selo = {
        "versao": 1, "entradas": entradas, "parametros": params, "contagens": contagens,
        "simplificacao": simp, "conteudo_sha256": conteudo,
        "arquivos": {n: sha256_arquivo(destinos[n]) for n in (nome_csv, PARQUET, GEOJSON)},
        "bibliotecas": {k: version(k) for k in ("geopandas", "shapely", "pyogrio", "pyarrow", "pandas")},
    }
    _atomico(destinos[SELO], lambda t: t.write_text(json.dumps(selo, indent=2, sort_keys=True) + "\n", encoding="utf-8"))
    return ResultadoReferencia("gerado", contagens, conteudo, simp)


# ---------------------------------------------------------------- leitores
def verificar_integridade(cfg: Config) -> dict:
    """Confere os arquivos gravados contra o selo; devolve o selo."""
    destinos = _destinos(cfg)
    if not destinos[SELO].is_file():
        raise FileNotFoundError(f"selo ausente: {destinos[SELO]} (rode gerar_referencia)")
    selo = _ler_selo(destinos)
    _verificar_arquivos(selo, destinos)
    return selo


def ler_municipios_escopo(cfg: Config) -> pd.DataFrame:
    """Municípios do escopo (cod_ibge:str de 7 dígitos), após checar o selo."""
    verificar_integridade(cfg)
    df = pd.read_csv(cfg.raiz / cfg.escopo.arquivo_semiarido, dtype={"cod_ibge": str}, encoding="utf-8")
    if tuple(df.columns) != COLS_CSV_ESCOPO:
        _falhar(RefSchemaError, f"colunas {list(df.columns)} != {list(COLS_CSV_ESCOPO)}")
    return df


def ler_geojson_gee(cfg: Config) -> dict:
    """FeatureCollection (dict) do escopo, EPSG:4326, após checar o selo — entrada do 2c."""
    verificar_integridade(cfg)
    return json.loads((cfg.caminho("reference") / GEOJSON).read_text(encoding="utf-8"))


def main() -> None:  # python -m src.data.reference
    from src.data.config import load_config

    r = gerar_referencia(load_config())
    print(f"[{r.status}] contagens={r.contagens}")
    print(f"simplificação={r.simplificacao}")
    print(f"conteudo_sha256={r.conteudo_sha256[:16]}…")


if __name__ == "__main__":
    main()
