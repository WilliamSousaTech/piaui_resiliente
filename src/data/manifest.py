"""Manifest do dataset (contrato 5.3): rastreabilidade de cada arquivo por SHA-256.

Cada arquivo do dataset (Parquet curado, referência territorial, artefatos de treino...) entra com o
caminho RELATIVO à raiz do projeto, o SHA-256 e, para Parquet, linhas, colunas e tipos (Arrow). Regras:

  - Hash de texto é calculado sobre o conteúdo normalizado em LF (CRLF vira LF), para não depender do
    Git/Windows (complementa o `.gitattributes`). Binários usam os bytes brutos.
  - Caminho absoluto, caminho fora da raiz (inclusive por link simbólico) e `..` são recusados.
  - `dataset_sha256` identifica o conjunto: hash dos pares (nome, SHA-256) mais o SHA-256 do `params.yaml`.
  - O manifest não se sobrescreve: com `dataset_sha256` igual é "ja_existente"; diferente exige
    `substituir=True`.
  - `verificar_manifest` e `caminho_verificado` recalculam os hashes: é o que a interface chama ao carregar
    (contrato 1.6, item 5). Arquivo alterado, ausente ou fora da raiz levanta ManifestIntegridadeError.

Domínio puro: sem rede, sem GEE, sem framework web.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
from collections.abc import Mapping
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import Any

from src.data import splits
from src.data.config import Config

VERSAO_MANIFEST = 1
PARAMS = "params.yaml"
SELO_REFERENCIA = "referencia.sha256.json"
EXTENSOES_TEXTO = frozenset({".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".geojson", ".csv",
                             ".html", ".css", ".js"})
BIBLIOTECAS = ("numpy", "pandas", "pyarrow", "scipy", "pydantic", "PyYAML", "torch")
_BLOCO = 1 << 20
_HEX64 = re.compile(r"[0-9a-f]{64}")
_UNIDADE_WIN = re.compile(r"^[A-Za-z]:")


class ManifestError(ValueError):
    """Entrada ou configuração incompatível com a geração do manifest."""


class ManifestImutavelError(RuntimeError):
    """Já existe um manifest de outro conjunto de dados; use `substituir=True` conscientemente."""


class ManifestIntegridadeError(RuntimeError):
    """Manifest ilegível ou arquivo listado ausente, alterado ou fora da raiz."""


# ---------------------------------------------------------------- hash
def sha256_arquivo(caminho: str | Path) -> str:
    """SHA-256 do arquivo. Texto (EXTENSOES_TEXTO) é normalizado para LF antes do hash."""
    p = Path(caminho)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        if p.suffix.lower() not in EXTENSOES_TEXTO:
            for bloco in iter(lambda: f.read(_BLOCO), b""):
                h.update(bloco)
            return h.hexdigest()
        resto = b""                     # um \r no fim do bloco pode ser o início de um \r\n
        for bloco in iter(lambda: f.read(_BLOCO), b""):
            dados = resto + bloco
            resto = b"\r" if dados.endswith(b"\r") else b""
            h.update((dados[:-1] if resto else dados).replace(b"\r\n", b"\n"))
        h.update(resto)                 # \r isolado no fim do arquivo fica como está
    return h.hexdigest()


# ---------------------------------------------------------------- caminhos
def _raiz(cfg: Config) -> Path:
    return cfg.raiz.resolve()


def _relativo(raiz: Path, caminho: str | Path) -> tuple[Path, str]:
    """(absoluto resolvido, relativo em POSIX). Recusa o que sai da raiz (inclusive por link simbólico)."""
    p = Path(caminho)
    absoluto = (p if p.is_absolute() else raiz / p).resolve()
    try:
        rel = absoluto.relative_to(raiz)
    except ValueError:
        raise ManifestError(f"fora da raiz do projeto: {caminho}") from None
    return absoluto, rel.as_posix()


def _resolver_armazenado(raiz: Path, caminho: Any) -> Path | None:
    """Caminho vindo do JSON do manifest -> absoluto dentro da raiz; None se for inseguro."""
    if not isinstance(caminho, str) or not caminho or "\\" in caminho or _UNIDADE_WIN.match(caminho):
        return None
    pp = PurePosixPath(caminho)
    if pp.is_absolute() or ".." in pp.parts:
        return None
    absoluto = (raiz / pp).resolve()
    try:
        absoluto.relative_to(raiz)
    except ValueError:
        return None
    return absoluto


def _destino(cfg: Config) -> Path:
    return cfg.raiz / cfg.manifest.arquivo


# ---------------------------------------------------------------- descrição dos arquivos
def _descrever_parquet(p: Path) -> dict:
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError("manifest de Parquet exige pyarrow: pip install pyarrow") from e
    try:
        pf = pq.ParquetFile(p)
        esquema = pf.schema_arrow
        return {"linhas": int(pf.metadata.num_rows), "colunas": list(esquema.names),
                "dtypes": {n: str(t) for n, t in zip(esquema.names, esquema.types)}}
    except (OSError, ValueError) as e:
        raise ManifestError(f"Parquet ilegível: {p.name}: {e}") from e


def _descrever(absoluto: Path, rel: str) -> dict:
    d = {"caminho": rel, "sha256": sha256_arquivo(absoluto)}
    if absoluto.suffix.lower() == ".parquet":
        d.update(_descrever_parquet(absoluto))
    return d


def _ler_selo_referencia(cfg: Config) -> dict:
    p = cfg.caminho("reference") / SELO_REFERENCIA
    if not p.is_file():
        raise ManifestError(f"selo da referência ausente: {p} (rode gerar_referencia)")
    try:
        selo = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ManifestError(f"selo da referência ilegível: {e}") from e
    if (not isinstance(selo, dict) or not isinstance(selo.get("conteudo_sha256"), str)
            or not _HEX64.fullmatch(selo["conteudo_sha256"]) or not isinstance(selo.get("contagens"), dict)):
        raise ManifestError("selo da referência sem 'conteudo_sha256' (64 hex) ou 'contagens'")
    return {"conteudo_sha256": selo["conteudo_sha256"], "contagens": selo["contagens"],
            "entradas": selo.get("entradas", {}), "parametros": selo.get("parametros", {})}


def _versoes() -> dict:
    out = {}
    for nome in BIBLIOTECAS:
        try:
            out[nome] = version(nome)
        except PackageNotFoundError:
            continue
    return out


def _dataset_sha256(hashes: Mapping[str, str], params_sha256: str) -> str:
    canonico = json.dumps({"arquivos": dict(hashes), "params_sha256": params_sha256},
                          sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonico.encode("utf-8")).hexdigest()


def _json_puro(valor: Any, onde: str) -> Any:
    try:
        json.dumps(valor, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ManifestError(f"{onde} não é JSON puro: {e}") from e
    return valor


# ---------------------------------------------------------------- construção
def construir_manifest(cfg: Config, arquivos: Mapping[str, str | Path], *,
                       linhas_por_bloco: Mapping[str, int] | None = None,
                       extras: Mapping[str, Any] | None = None,
                       agora: datetime | None = None) -> dict:
    """Manifest em JSON puro. `arquivos`: nome lógico -> caminho (relativo à raiz ou absoluto dentro dela).

    `linhas_por_bloco` (train/val/test/embargo) e `extras` (ex.: edição e ano da camada de preparo)
    são opcionais. `agora` (com fuso) torna a data injetável nos testes."""
    if not isinstance(arquivos, Mapping) or not arquivos:
        raise ManifestError("informe ao menos um arquivo")
    raiz = _raiz(cfg)
    params = raiz / PARAMS
    if not params.is_file():
        raise ManifestError(f"{PARAMS} não encontrado em {raiz}")
    manifest_abs = _destino(cfg).resolve()
    entradas, vistos = {}, {}
    for nome, caminho in arquivos.items():
        if not isinstance(nome, str) or not nome.strip():
            raise ManifestError(f"nome lógico inválido: {nome!r}")
        absoluto, rel = _relativo(raiz, caminho)
        if not absoluto.is_file():
            raise ManifestError(f"arquivo inexistente: {rel}")
        if absoluto == manifest_abs:
            raise ManifestError("o manifest não pode listar a si mesmo")
        if rel in vistos:
            raise ManifestError(f"mesmo arquivo em dois nomes ({vistos[rel]!r} e {nome!r}): {rel}")
        vistos[rel] = nome
        entradas[nome] = _descrever(absoluto, rel)

    if linhas_por_bloco is not None:
        for bloco, n in linhas_por_bloco.items():
            if bloco not in splits.BLOCOS:
                raise ManifestError(f"bloco desconhecido em linhas_por_bloco: {bloco!r}")
            if isinstance(n, bool) or not isinstance(n, int) or n < 0:
                raise ManifestError(f"linhas_por_bloco[{bloco!r}] deve ser inteiro >= 0")
    if agora is None:
        agora = datetime.now(timezone.utc)
    elif agora.tzinfo is None:
        raise ManifestError("`agora` deve ter fuso horário")
    ref = _ler_selo_referencia(cfg)
    params_sha = sha256_arquivo(params)
    c = cfg.fontes
    manifest = {
        "versao_manifest": VERSAO_MANIFEST,
        "algoritmo_hash": cfg.manifest.algoritmo_hash,
        "gerado_em_utc": agora.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "maturity": cfg.projeto.maturidade,
        "projeto": {"nome": cfg.projeto.nome, "equipe": cfg.projeto.equipe,
                    "versao_params": cfg.projeto.versao_params, "seed": cfg.projeto.seed},
        "params_sha256": params_sha,
        "dataset_sha256": _dataset_sha256({n: e["sha256"] for n, e in entradas.items()}, params_sha),
        "ambiente": {"python": platform.python_version(), "bibliotecas": _versoes()},
        "janela": {"inicio": str(cfg.janela.inicio), "fim": str(cfg.janela.fim)},
        "split": {**splits.cortes(cfg).para_manifest(), "teste_liberado": cfg.split.teste_liberado},
        "alvo": {"escala_spi_meses": cfg.alvo.escala_spi_meses, "horizonte_h": cfg.alvo.horizonte_h,
                 "janelas_spi": list(cfg.spi.janelas_meses)},
        "escopo": {"modo": cfg.escopo.modo, "incluir_transicao": cfg.escopo.incluir_transicao,
                   "fonte_semiarido": cfg.escopo.fonte_semiarido,
                   "malha_municipal_ano": cfg.escopo.malha_municipal_ano},
        "referencia": ref,
        "fontes": {"chirps": c.chirps.model_dump(mode="json"), "mapbiomas": c.mapbiomas.model_dump(mode="json"),
                   "s2id": c.s2id.model_dump(mode="json")},
        "arquivos": dict(sorted(entradas.items())),
    }
    if linhas_por_bloco is not None:
        manifest["linhas_por_bloco"] = dict(linhas_por_bloco)
    if extras is not None:
        manifest["extras"] = _json_puro(dict(extras), "extras")
    return manifest


# ---------------------------------------------------------------- validação do JSON
def _validar_estrutura(m: Any, exc: type[Exception]) -> None:
    def falhar(msg):
        raise exc(f"manifest inválido: {msg}")

    if not isinstance(m, dict):
        falhar("esperado um objeto JSON")
    if m.get("versao_manifest") != VERSAO_MANIFEST:
        falhar(f"versao_manifest={m.get('versao_manifest')!r}; esperado {VERSAO_MANIFEST}")
    if m.get("algoritmo_hash") != "sha256":
        falhar("algoritmo_hash deve ser 'sha256'")
    mat = m.get("maturity")
    if not isinstance(mat, str) or "experimental" not in mat.lower():
        falhar("falta o aviso de maturidade ('maturity') de MVP experimental")
    arq = m.get("arquivos")
    if not isinstance(arq, dict) or not arq:
        falhar("'arquivos' ausente ou vazio")
    for nome, e in arq.items():
        if not isinstance(e, dict) or not isinstance(e.get("caminho"), str):
            falhar(f"entrada sem 'caminho': {nome!r}")
        if not isinstance(e.get("sha256"), str) or not _HEX64.fullmatch(e["sha256"]):
            falhar(f"sha256 inválido em {nome!r}")
    p = m.get("params_sha256")
    if not isinstance(p, str) or not _HEX64.fullmatch(p):
        falhar("params_sha256 inválido")
    esperado = _dataset_sha256({n: e["sha256"] for n, e in arq.items()}, p)
    if m.get("dataset_sha256") != esperado:
        falhar("dataset_sha256 não confere com os hashes listados (manifest adulterado?)")


# ---------------------------------------------------------------- gravação
def gravar_manifest(cfg: Config, manifest: Mapping, *, substituir: bool = False) -> str:
    """Grava `manifest.arquivo` de forma atômica. Devolve 'gerado', 'ja_existente' ou 'substituido'."""
    try:
        m = json.loads(json.dumps(manifest, allow_nan=False))
    except (TypeError, ValueError) as e:
        raise ManifestError(f"manifest não é JSON puro: {e}") from e
    _validar_estrutura(m, ManifestError)
    destino = _destino(cfg)
    status = "gerado"
    if destino.exists():
        try:
            atual = json.loads(destino.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ManifestIntegridadeError(f"manifest existente ilegível: {e}") from e
        if isinstance(atual, dict) and atual.get("dataset_sha256") == m["dataset_sha256"]:
            return "ja_existente"
        if not substituir:
            raise ManifestImutavelError("já existe um manifest de outro conjunto de dados; "
                                        "use substituir=True conscientemente")
        status = "substituido"
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_name(destino.name + ".tmp")
    try:
        tmp.write_text(json.dumps(m, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                       encoding="utf-8", newline="\n")
        os.replace(tmp, destino)
    finally:
        if tmp.exists():
            tmp.unlink()
    return status


# ---------------------------------------------------------------- verificação
def _ler_manifest(cfg: Config) -> dict:
    destino = _destino(cfg)
    if not destino.is_file():
        raise ManifestIntegridadeError(f"manifest ausente: {destino}")
    try:
        m = json.loads(destino.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ManifestIntegridadeError(f"manifest ilegível: {e}") from e
    _validar_estrutura(m, ManifestIntegridadeError)
    return m


def _problema(raiz: Path, nome: str, entrada: Mapping) -> str | None:
    p = _resolver_armazenado(raiz, entrada["caminho"])
    if p is None:
        return f"{nome}: caminho inseguro ou fora da raiz ({entrada['caminho']!r})"
    if not p.is_file():
        return f"{nome}: arquivo ausente ({entrada['caminho']})"
    if sha256_arquivo(p) != entrada["sha256"]:
        return f"{nome}: SHA-256 difere do manifest (arquivo alterado: {entrada['caminho']})"
    return None


def verificar_manifest(cfg: Config, *, conferir_params: bool = True) -> dict:
    """Recalcula todos os hashes. Devolve o manifest ou levanta ManifestIntegridadeError (lista tudo)."""
    m = _ler_manifest(cfg)
    raiz = _raiz(cfg)
    problemas = [x for n, e in m["arquivos"].items() if (x := _problema(raiz, n, e))]
    if conferir_params:
        params = raiz / PARAMS
        if not params.is_file():
            problemas.append(f"{PARAMS}: ausente")
        elif sha256_arquivo(params) != m["params_sha256"]:
            problemas.append(f"{PARAMS}: mudou desde a geração do manifest")
    if problemas:
        raise ManifestIntegridadeError("; ".join(problemas))
    return m


def caminho_verificado(cfg: Config, nome: str) -> Path:
    """Caminho absoluto do arquivo `nome` do manifest, depois de conferir o SHA-256 dele.

    É a porta de entrada da interface: só se abre o que está no manifest e íntegro."""
    m = _ler_manifest(cfg)
    entrada = m["arquivos"].get(nome)
    if entrada is None:
        raise ManifestError(f"{nome!r} não consta no manifest")
    problema = _problema(_raiz(cfg), nome, entrada)
    if problema:
        raise ManifestIntegridadeError(problema)
    return _resolver_armazenado(_raiz(cfg), entrada["caminho"])
