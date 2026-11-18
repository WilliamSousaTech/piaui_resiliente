"""Demonstração observada (Nível 1): RAW do CHIRPS -> qualidade -> SPI -> `demo_observado` + manifest.

Fluxo (contrato 1.5, 4.1, 6 e 7):
  1. `quality.avaliar_qualidade` marca os meses válidos e as lacunas (D9: o SPI usa a série VÁLIDA,
     nunca a imputada).
  2. `spi.calcular_spi` ajusta a Gama só no treino (<= `split.treino_fim`) e aplica ao período todo
     (1981-2025). O SPI das linhas de treino é in-sample (limitação registrada no contrato).
  3. D11: município `excluded` pela cobertura do treino fica com SPI NaN, `invalid = True` e
     `baixa_conf = False` em todos os meses ("sem dado" na interface, contrato 1.4). Isso se faz
     marcando `valido = False` nos parâmetros do município, não editando o SPI depois.
  4. Grava, de forma atômica e imutável, `demo_observado` (curated), `cobertura_mensal` e
     `qualidade_municipio` (processed), `spi_gamma_params` e `climatologia_precip` (artifacts).
  5. Gera o manifest (SHA-256 dos artefatos, dos lotes do RAW e da referência territorial).

CA-12: este módulo não lê o arquivo do teste congelado e não escreve alvo, split nem métrica.
Domínio puro: sem rede, sem GEE, sem framework web.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data import chirps_raw, manifest, quality, spi
from src.data.config import Config, load_config

# nome lógico -> (pasta em `caminhos`, arquivo)
ARTEFATOS = {
    "demo_observado": ("curated", "demo_observado.parquet"),
    "cobertura_mensal": ("processed", "cobertura_mensal.parquet"),
    "qualidade_municipio": ("processed", "qualidade_municipio.parquet"),
    "spi_gamma_params": ("artifacts", "spi_gamma_params.parquet"),
    "climatologia_precip": ("artifacts", "climatologia_precip.parquet"),
}
# Nomes da referência territorial: iguais a `reference.PARQUET/GEOJSON` (conferido nos testes). Duplicados
# aqui para não importar geopandas (pesado) só para listar arquivos.
PARQUET_REFERENCIA = "malha_pi_2022.parquet"
GEOJSON_REFERENCIA = "malha_pi_2022_gee.geojson"


class DemoError(ValueError):
    """Resultado inconsistente com o contrato."""


class DemoImutavelError(RuntimeError):
    """Já existe um artefato com conteúdo diferente; apague conscientemente e regenere."""


@dataclass(frozen=True)
class ResultadoDemo:
    demo_observado: pd.DataFrame
    cobertura_mensal: pd.DataFrame
    qualidade_municipio: pd.DataFrame
    spi_gamma_params: pd.DataFrame
    climatologia_precip: pd.DataFrame


# ---------------------------------------------------------------- montagem (sem I/O)
def montar_demo(raw: pd.DataFrame, cfg: Config) -> ResultadoDemo:
    """Constrói as cinco tabelas a partir do RAW. Não lê nem grava arquivos.

    D11: município `excluded` tem as células de `spi_gamma_params` marcadas `valido = False` e o SPI é
    reaplicado com esses parâmetros. Assim a regra mora num lugar só: `spi.calcular_spi(serie, cfg,
    parametros=resultado.spi_gamma_params)` reproduz `demo_observado` exatamente."""
    q = quality.avaliar_qualidade(raw, cfg)
    serie = q.serie_valida[["cod_ibge", "mes", "precip_mm"]]
    r = spi.calcular_spi(serie, cfg)
    params = r.parametros.reset_index(drop=True)
    excluidos = set(q.municipios.loc[q.municipios["excluded"], "cod_ibge"])
    if excluidos:
        params.loc[params["cod_ibge"].isin(excluidos), "valido"] = False
        r = spi.calcular_spi(serie, cfg, parametros=params)
        params = r.parametros.reset_index(drop=True)
    demo = r.spi.reset_index(drop=True)
    chaves_demo = pd.MultiIndex.from_frame(demo[["cod_ibge", "mes"]])
    chaves_cob = pd.MultiIndex.from_frame(q.cobertura[["cod_ibge", "mes"]])
    if not chaves_demo.equals(chaves_cob):
        raise DemoError("o SPI não cobre exatamente a grade município x mês da cobertura")
    return ResultadoDemo(demo_observado=demo, cobertura_mensal=q.cobertura, qualidade_municipio=q.municipios,
                         spi_gamma_params=params, climatologia_precip=q.climatologia)


# ---------------------------------------------------------------- gravação
def destino(cfg: Config, nome: str) -> Path:
    pasta, arquivo = ARTEFATOS[nome]
    return cfg.caminho(pasta) / arquivo


def _gravar_atomico(df: pd.DataFrame, alvo: Path) -> None:
    alvo.parent.mkdir(parents=True, exist_ok=True)
    tmp = alvo.with_name(alvo.name + ".tmp")
    try:
        try:
            df.to_parquet(tmp, engine="pyarrow", index=False)
        except ImportError as e:
            raise ImportError("Parquet exige pyarrow: pip install pyarrow") from e
        os.replace(tmp, alvo)
    finally:
        if tmp.exists():
            tmp.unlink()


def gravar_demo(cfg: Config, resultado: ResultadoDemo) -> dict[str, str]:
    """Grava os artefatos que faltam e confere os que já existem. Devolve 'gerado' ou 'ja_existente' por nome.

    Todo-ou-nada quanto a conflito: se QUALQUER artefato existente difere do novo, levanta
    DemoImutavelError antes de gravar o que for."""
    status: dict[str, str] = {}
    a_gravar: list[str] = []
    for nome in ARTEFATOS:
        df, alvo = getattr(resultado, nome), destino(cfg, nome)
        if not alvo.exists():
            a_gravar.append(nome)
            continue
        try:
            atual = pd.read_parquet(alvo, engine="pyarrow")
            pd.testing.assert_frame_equal(atual, df, check_exact=True)
        except (OSError, ValueError, AssertionError) as e:
            raise DemoImutavelError(f"{alvo.name} já existe com conteúdo diferente ou ilegível; "
                                    "apague conscientemente e regenere") from e
        status[nome] = "ja_existente"
    for nome in a_gravar:
        _gravar_atomico(getattr(resultado, nome), destino(cfg, nome))
        status[nome] = "gerado"
    return {n: status[n] for n in ARTEFATOS}


# ---------------------------------------------------------------- orquestração (com I/O)
def _arquivos_do_manifest(cfg: Config) -> dict[str, Path]:
    arquivos: dict[str, Path] = {nome: destino(cfg, nome) for nome in ARTEFATOS}
    # mesmo padrão de nome que `chirps_raw.ler_raw` usa para achar os lotes
    for lote in sorted(chirps_raw.diretorio_raw(cfg).glob(f"{chirps_raw._PREFIXO}*.parquet")):
        arquivos[f"raw_chirps/{lote.name}"] = lote
    csv = cfg.raiz / cfg.escopo.arquivo_semiarido
    ref = cfg.caminho("reference")
    for nome, p in ((csv.name, csv), (PARQUET_REFERENCIA, ref / PARQUET_REFERENCIA),
                    (GEOJSON_REFERENCIA, ref / GEOJSON_REFERENCIA),
                    (manifest.SELO_REFERENCIA, ref / manifest.SELO_REFERENCIA)):
        if p.is_file():
            arquivos[f"referencia/{nome}"] = p
    return arquivos


def construir_demo(cfg: Config, *, municipios: Iterable[str] | None = None, agora: datetime | None = None,
                   substituir_manifest: bool = False) -> dict:
    """Lê o RAW, monta, grava os artefatos e o manifest. Devolve um resumo (JSON puro).

    `municipios` vem da referência territorial (selo conferido) quando omitido. Nada é gravado se o
    RAW estiver incompleto ou inválido (a leitura valida a grade município x mês inteira). Se o
    manifest falhar depois (ex.: selo da referência ausente), rodar de novo é seguro: os artefatos
    já gravados são só conferidos."""
    if municipios is None:
        from src.data import reference        # importação tardia: geopandas é pesado e só esta rota o usa
        municipios = reference.ler_municipios_escopo(cfg)["cod_ibge"]
    cods = list(municipios)
    raw = chirps_raw.ler_raw(chirps_raw.diretorio_raw(cfg), cods, chirps_raw.meses_janela(cfg))
    r = montar_demo(raw, cfg)
    status = gravar_demo(cfg, r)
    demo = r.demo_observado
    n_excl = int(r.qualidade_municipio["excluded"].sum())
    col_alvo = f"spi{cfg.alvo.escala_spi_meses}_invalid"
    resumo_demo = {"n_municipios": len(cods), "n_meses": int(demo["mes"].nunique()), "n_excluidos": n_excl,
                   "escalas_spi": list(cfg.spi.janelas_meses),
                   "spi_calibrado_ate": str(cfg.split.treino_fim), "spi_treino_in_sample": True}
    m = manifest.construir_manifest(cfg, _arquivos_do_manifest(cfg), extras={"demo": resumo_demo}, agora=agora)
    st_manifest = manifest.gravar_manifest(cfg, m, substituir=substituir_manifest)
    return {"artefatos": status, "manifest": st_manifest, "dataset_sha256": m["dataset_sha256"],
            "maturity": cfg.projeto.maturidade,
            "frac_alvo_invalido": float(demo[col_alvo].mean()), **resumo_demo}


def main(argv: list[str] | None = None) -> int:      # python -m src.data.build_demo
    ap = argparse.ArgumentParser(prog="python -m src.data.build_demo",
                                 description="Gera o demo_observado (SPI 1981-2025) e o manifest a partir do RAW do CHIRPS.")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--substituir-manifest", action="store_true",
                    help="troca o manifest existente de outro conjunto de dados")
    a = ap.parse_args(argv)
    try:
        cfg = load_config(a.params)
        r = construir_demo(cfg, substituir_manifest=a.substituir_manifest)
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    print(f"AVISO: {r['maturity']}")
    for nome, st in r["artefatos"].items():
        print(f"[{st}] {nome}")
    print(f"[{r['manifest']}] manifest  dataset_sha256={r['dataset_sha256'][:16]}…")
    print(f"municípios={r['n_municipios']} meses={r['n_meses']} excluídos={r['n_excluidos']} "
          f"SPI-alvo inválido={r['frac_alvo_invalido']:.1%} (SPI de treino é in-sample)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
