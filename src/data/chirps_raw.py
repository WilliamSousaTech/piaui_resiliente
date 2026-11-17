"""RAW do CHIRPS mensal por município (Item 2a). Parte pura: sem `ee`, sem rede.

Responsabilidades: meses da janela, divisão em lotes (< limite de elementos do getInfo),
padronização e validação do schema (contrato 5.1, v0.2.0), gravação IMUTÁVEL em Parquet com selo
SHA-256 (mais `meta` informativo opcional, ex.: nº de imagens por mês), leitura com checagem de
integridade e de completude (município x mês).

Fora do escopo (ficam no 2c): chamadas ao GEE. Quem chama deve entregar `cobertura` numérica
(0 quando o GEE devolver null): aqui `dias_validos` e `pixels_validos_frac` nunca são NaN.
`precip_mm` pode ser NaN (município sem pixel válido, contrato 6.6); NaN nunca vira 0.
Quando `precip_mm` é NaN, `precip_max_dia_mm` e `dias_chuva` também são (e só nesse caso).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.data.config import Config

COLUNAS = ["cod_ibge", "mes", "precip_mm", "precip_max_dia_mm", "dias_chuva", "dias_validos",
           "dias_no_mes", "pixels_validos_frac"]
DTYPES = {"mes": "datetime64[ns]", "precip_mm": "float32", "precip_max_dia_mm": "float32",
          "dias_chuva": "float32", "dias_validos": "int16", "dias_no_mes": "int16",
          "pixels_validos_frac": "float32"}
SUBDIR = "chirps"
PRECIP_MAX_MM_MES = 2000.0   # teto de sanidade (semiárido); acima disso é erro de unidade/soma
TOL_FLOAT32 = 1e-3           # folga para comparar médias gravadas em float32 (mm ou dias)
_COD = re.compile(r"\d{7}")
_PREFIXO = "chirps_mensal_"


class RawSchemaError(ValueError):
    """Schema, tipos, valores ou completude fora do contrato."""


class RawImutavelError(RuntimeError):
    """Tentativa de alterar um lote RAW já gravado."""


class RawIntegridadeError(RuntimeError):
    """Arquivo sem selo, selo órfão ou SHA-256 divergente."""


@dataclass(frozen=True)
class ResultadoLote:
    caminho: Path
    status: str          # "gravado" | "pulado"
    sha256: str
    n_linhas: int
    n_precip_nan: int


# ---------------------------------------------------------------- meses e lotes
def meses_janela(cfg: Config) -> list[pd.Period]:
    return list(pd.period_range(cfg.janela.inicio, cfg.janela.fim, freq="M"))


def diretorio_raw(cfg: Config) -> Path:
    return cfg.caminho("raw") / SUBDIR


def meses_por_lote(n_municipios: int, max_linhas: int) -> int:
    if n_municipios < 1 or max_linhas < 1:
        raise ValueError("n_municipios e max_linhas devem ser >= 1")
    k = max_linhas // n_municipios
    if k < 1:
        raise ValueError(f"{n_municipios} municípios não cabem em um mês de {max_linhas} linhas")
    return k


def dividir_em_lotes(meses: list[pd.Period], n_municipios: int,
                     max_linhas: int) -> list[tuple[pd.Period, pd.Period]]:
    """Blocos consecutivos de meses; cada um gera <= max_linhas linhas (N x meses)."""
    if not meses:
        raise ValueError("lista de meses vazia")
    if any(m.freqstr != "M" for m in meses) or any(
            (b - a).n != 1 for a, b in zip(meses, meses[1:])):
        raise ValueError("meses devem ser Periods mensais, consecutivos e crescentes")
    k = meses_por_lote(n_municipios, max_linhas)
    return [(meses[i], meses[min(i + k, len(meses)) - 1]) for i in range(0, len(meses), k)]


# ---------------------------------------------------------------- schema
def _checar_municipios(municipios: Iterable[str]) -> list[str]:
    cods = list(municipios)
    if not cods:
        raise RawSchemaError("lista de municípios vazia")
    ruins = [c for c in cods if not (isinstance(c, str) and _COD.fullmatch(c))]
    if ruins:
        raise RawSchemaError(f"cod_ibge fora do padrão de 7 dígitos (str): {ruins[:5]}")
    if len(set(cods)) != len(cods):
        raise RawSchemaError("lista de municípios com códigos repetidos")
    return sorted(cods)


def _inteiro16(s: pd.Series, nome: str) -> pd.Series:
    s = pd.to_numeric(s, errors="raise")
    if s.isna().any() or ((s % 1) != 0).any():
        raise RawSchemaError(f"{nome}: precisa ser inteiro, sem NaN")
    if s.abs().max() > 32767:
        raise RawSchemaError(f"{nome}: valor fora do intervalo de int16")
    return s.astype("int16")


def padronizar(df: pd.DataFrame) -> pd.DataFrame:
    """Converte para os dtypes do contrato, fixa a ordem das colunas e ordena (cod_ibge, mes)."""
    if not isinstance(df, pd.DataFrame):
        raise RawSchemaError("esperado um DataFrame")
    if set(df.columns) != set(COLUNAS) or len(df.columns) != len(COLUNAS):
        raise RawSchemaError(f"colunas {list(df.columns)}; esperado {COLUNAS}")
    out = df[COLUNAS].copy()
    try:
        out["cod_ibge"] = out["cod_ibge"].astype(str)
        out["mes"] = pd.to_datetime(out["mes"]).astype("datetime64[ns]")
        out["precip_mm"] = pd.to_numeric(out["precip_mm"], errors="raise").astype("float32")
        for c in ("precip_max_dia_mm", "dias_chuva", "pixels_validos_frac"):
            out[c] = pd.to_numeric(out[c], errors="raise").astype("float32")
    except (ValueError, TypeError) as e:
        raise RawSchemaError(f"conversão de tipos falhou: {e}") from e
    out["dias_validos"] = _inteiro16(out["dias_validos"], "dias_validos")
    out["dias_no_mes"] = _inteiro16(out["dias_no_mes"], "dias_no_mes")
    return out.sort_values(["cod_ibge", "mes"], kind="stable").reset_index(drop=True)


def _mes_int(s: pd.Series) -> pd.Series:
    return (s.dt.year * 12 + s.dt.month).astype(int)


def _fmt(cod: str, k: int) -> str:
    a, m = divmod(k - 1, 12)
    return f"{cod}@{a}-{m + 1:02d}"


def validar(df: pd.DataFrame, municipios: Iterable[str], meses: list[pd.Period]) -> None:
    """Levanta RawSchemaError (com TODAS as violações) se `df` não for exatamente municípios x meses."""
    cods = _checar_municipios(municipios)
    if list(df.columns) != COLUNAS:
        raise RawSchemaError(f"colunas {list(df.columns)}; esperado {COLUNAS}")
    tipos = [f"{c}: {df[c].dtype}, esperado {t}" for c, t in DTYPES.items()
             if str(df[c].dtype) != t]
    if not pd.api.types.is_string_dtype(df["cod_ibge"]):
        tipos.append(f"cod_ibge: {df['cod_ibge'].dtype}, esperado str")
    if tipos:
        raise RawSchemaError("dtypes fora do contrato -> " + "; ".join(tipos))

    erros: list[str] = []

    def regra(mascara: pd.Series, msg: str) -> None:
        n = int(mascara.sum())
        if n:
            sub = df.loc[mascara, ["cod_ibge", "mes"]].head(3)
            ex = [f"{c}@{m}" for c, m in zip(sub["cod_ibge"], sub["mes"])]
            erros.append(f"{msg}: {n} linha(s), ex.: {ex}")

    mes = df["mes"]
    regra(~df["cod_ibge"].astype(str).str.fullmatch(_COD.pattern).fillna(False), "cod_ibge inválido")
    regra(mes.isna() | (mes != mes.dt.normalize()) | ~mes.dt.is_month_start,
          "mes deve ser o dia 1 às 00:00")
    regra(df["dias_no_mes"] != mes.dt.days_in_month, "dias_no_mes diferente do calendário")
    regra((df["dias_validos"] < 0) | (df["dias_validos"] > df["dias_no_mes"]),
          "dias_validos fora de [0, dias_no_mes]")
    regra(~df["pixels_validos_frac"].between(0, 1), "pixels_validos_frac fora de [0,1] ou NaN")
    p = df["precip_mm"]
    tem = p.notna()
    regra(tem & (~np.isfinite(p) | (p < 0) | (p > PRECIP_MAX_MM_MES)),
          f"precip_mm fora de [0, {PRECIP_MAX_MM_MES:g}] ou infinito")
    regra(tem & ((df["dias_validos"] == 0) | (df["pixels_validos_frac"] == 0)),
          "precip_mm preenchido sem dia/pixel válido")
    pm, dc = df["precip_max_dia_mm"], df["dias_chuva"]
    regra((p.isna() != pm.isna()) | (p.isna() != dc.isna()),
          "precip_mm, precip_max_dia_mm e dias_chuva devem ser NaN juntos")
    regra(pm.notna() & (~np.isfinite(pm) | (pm < 0)), "precip_max_dia_mm negativo ou infinito")
    regra(tem & pm.notna() & (pm > p + TOL_FLOAT32), "precip_max_dia_mm maior que precip_mm")
    regra(dc.notna() & (~np.isfinite(dc) | (dc < 0) | (dc > df["dias_no_mes"] + TOL_FLOAT32)),
          "dias_chuva fora de [0, dias_no_mes] ou infinito")
    regra(tem & pm.notna() & dc.notna() & (p == 0) & ((pm > 0) | (dc > 0)),
          "precip_mm = 0 com chuva diária (precip_max_dia_mm ou dias_chuva > 0)")

    d = df[mes.notna()]
    chaves = list(zip(d["cod_ibge"], _mes_int(d["mes"])))
    if d.duplicated(["cod_ibge", "mes"]).any():
        dup = d[d.duplicated(["cod_ibge", "mes"], keep=False)].head(3)
        erros.append("duplicatas (cod_ibge, mes), ex.: "
                     + str([_fmt(c, k) for c, k in zip(dup["cod_ibge"], _mes_int(dup["mes"]))]))
    obs = set(chaves)
    esp = {(c, m.year * 12 + m.month) for c in cods for m in meses}
    if esp - obs:
        erros.append(f"faltam {len(esp - obs)} linha(s), ex.: "
                     + str([_fmt(*t) for t in sorted(esp - obs)[:3]]))
    if obs - esp:
        erros.append(f"{len(obs - esp)} linha(s) fora do esperado, ex.: "
                     + str([_fmt(*t) for t in sorted(obs - esp)[:3]]))
    if erros:
        raise RawSchemaError("RAW CHIRPS inválido -> " + " | ".join(erros))


# ---------------------------------------------------------------- arquivos
def sha256_arquivo(caminho: Path) -> str:
    h = hashlib.sha256()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()


def caminho_lote(dir_raw: Path, ini: pd.Period, fim: pd.Period) -> Path:
    return Path(dir_raw) / f"{_PREFIXO}{ini.strftime('%Y%m')}_{fim.strftime('%Y%m')}.parquet"


def _selo(parquet: Path) -> Path:
    return parquet.with_name(parquet.name + ".sha256.json")


def verificar_integridade(parquet: Path) -> str:
    """Compara o SHA-256 do arquivo com o selo. Devolve o hash ou levanta RawIntegridadeError."""
    selo = _selo(parquet)
    if not parquet.is_file() or not selo.is_file():
        raise RawIntegridadeError(f"lote sem par arquivo+selo: {parquet.name}")
    esperado = json.loads(selo.read_text(encoding="utf-8")).get("sha256")
    atual = sha256_arquivo(parquet)
    if atual != esperado:
        raise RawIntegridadeError(f"SHA-256 divergente em {parquet.name}: RAW foi alterado")
    return atual


def lote_gravado(dir_raw: Path, ini: pd.Period, fim: pd.Period) -> bool:
    """True se o lote existe com selo. Meio-estado (só um dos dois) é erro, nunca 'falso'."""
    arq = caminho_lote(dir_raw, ini, fim)
    if arq.exists() != _selo(arq).exists():
        raise RawIntegridadeError(f"lote em meio-estado (arquivo/selo): {arq.name}; "
                                  "apague os dois manualmente após conferir")
    return arq.exists()


def ler_lote(parquet: Path) -> pd.DataFrame:
    try:
        return padronizar(pd.read_parquet(parquet, engine="pyarrow"))
    except ImportError as e:
        raise ImportError("RAW em Parquet exige pyarrow: pip install pyarrow") from e


def _escreve_atomico(destino: Path, escreve) -> None:
    tmp = destino.with_name(destino.name + ".tmp")
    escreve(tmp)
    os.replace(tmp, destino)


def gravar_lote(df: pd.DataFrame, dir_raw: Path, municipios: Iterable[str],
                ini: pd.Period, fim: pd.Period, meta: dict | None = None) -> ResultadoLote:
    """Valida e grava um lote. Existente e idêntico -> 'pulado'; existente e diferente -> erro.

    `meta` (opcional, JSON puro) vai só para o selo, como registro de auditoria (ex.: nº de
    imagens do GEE por mês). Não entra na comparação de conteúdo nem no SHA-256 do Parquet.
    """
    cods = _checar_municipios(municipios)
    novo = padronizar(df)
    validar(novo, cods, list(pd.period_range(ini, fim, freq="M")))
    if meta is not None:
        try:
            json.dumps(meta, allow_nan=False)
        except (TypeError, ValueError) as e:
            raise RawSchemaError(f"meta precisa ser JSON puro (sem NaN): {e}") from e
    destino = caminho_lote(dir_raw, ini, fim)
    n, n_nan = len(novo), int(novo["precip_mm"].isna().sum())
    if lote_gravado(dir_raw, ini, fim):
        sha = verificar_integridade(destino)
        try:
            pd.testing.assert_frame_equal(ler_lote(destino), novo, check_exact=True)
        except AssertionError as e:
            raise RawImutavelError(f"{destino.name} já existe com conteúdo diferente; "
                                   "RAW é imutável (nova versão = nova pasta)") from e
        return ResultadoLote(destino, "pulado", sha, n, n_nan)
    Path(dir_raw).mkdir(parents=True, exist_ok=True)
    try:
        _escreve_atomico(destino, lambda t: novo.to_parquet(t, engine="pyarrow", index=False))
    except ImportError as e:
        raise ImportError("RAW em Parquet exige pyarrow: pip install pyarrow") from e
    sha = sha256_arquivo(destino)
    selo = {"arquivo": destino.name, "sha256": sha, "n_linhas": n,
            "mes_inicio": str(ini), "mes_fim": str(fim)}
    if meta is not None:
        selo["meta"] = meta
    _escreve_atomico(_selo(destino), lambda t: t.write_text(
        json.dumps(selo, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"))
    return ResultadoLote(destino, "gravado", sha, n, n_nan)


def ler_raw(dir_raw: Path, municipios: Iterable[str], meses: list[pd.Period]) -> pd.DataFrame:
    """Lê todos os lotes (checando integridade) e exige o conjunto completo município x mês."""
    arquivos = sorted(Path(dir_raw).glob(f"{_PREFIXO}*.parquet"))
    if not arquivos:
        raise RawIntegridadeError(f"nenhum lote RAW em {dir_raw}")
    partes = []
    for a in arquivos:
        verificar_integridade(a)
        partes.append(ler_lote(a))
    todo = padronizar(pd.concat(partes, ignore_index=True))
    validar(todo, municipios, meses)
    return todo
