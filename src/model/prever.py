"""Previsão do próximo mês com o modelo final (Item 9). Sem rede, sem teste congelado, sem treino.

Pega a ÚLTIMA origem do conjunto (o último mês com chuva observada), roda o conjunto de redes salvo em
`artifacts/modelo_v2` e grava `data/reference/previsao_proximo_mes.json`: SPI-3 e SPI-6 previstos para o mês
seguinte em cada município, a margem de erro, o sinal de alerta de seca e os números da validação que o site
precisa mostrar junto. O site só lê esse arquivo; nada aqui depende de framework web.

Garantias (cada uma tem teste):
  * a última origem tem de cair no bloco `pos_teste` e não pode ter rótulo: não lê o teste congelado e não
    chama "previsão" algo que já tem resposta;
  * a rede e os números da validação têm de ser do mesmo `dataset_sha256` dos dados lidos agora;
  * a marca de baixa confiança do mês previsto é calculada dos parâmetros Gama do treino (a coluna
    `alvo_spiK_baixa_conf` do conjunto vem toda falsa nesse mês, porque o SPI futuro não existe);
  * nenhum NaN ou infinito vira número no arquivo, o arquivo só aparece inteiro (escrita atômica) e só é gravado se
    passar na mesma conferência que o site faz ao ler (`src/data/previsao.py`).

Uso:
    python -m src.model.prever [--params params.yaml] [--modelo artifacts/modelo_v2]
                               [--validacao artifacts/busca/confirmacao_B.json]
                               [--saida data/reference/previsao_proximo_mes.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import previsao as pr
from src.data.config import Config, load_config
from src.model import baselines as bl
from src.model import busca as bu
from src.model import entrada
from src.model import features as fe
from src.model import residuo as rs

VERSAO = pr.VERSAO
ARQUIVO_PADRAO = pr.ARQUIVO_PADRAO
PASTA_MODELO_PADRAO = Path("artifacts") / "modelo_v2"
VALIDACAO_PADRAO = Path("artifacts") / "busca" / "confirmacao_B.json"
AVISO = ("MVP experimental, em calibração com S2iD/TCE-PI. É uma previsão com margem de erro, não uma certeza: "
         "na validação o modelo errou, em média, bem menos que a persistência, mas pegou só uma parte das secas reais.")
CASAS = 2                                   # SPI com duas casas; o sinal de alerta vale para o número arredondado
CASAS_METRICA = 6                           # MAE, margem e taxas da validação; arredondar cedo causaria arredondamento duplo na página


class PreverError(ValueError):
    """A previsão não pode ser feita ou publicada de forma honesta com o que foi entregue."""


# ---------------------------------------------------------------- pequenos auxiliares
def _arredondar(x: float, casas: int = CASAS) -> float:
    v = round(float(x), casas)
    return 0.0 if v == 0 else v                                   # nunca grava -0.0


def _pegar(d: dict, *chaves, onde: str):
    """`d[chaves[0]][chaves[1]]...` com erro claro quando falta (o arquivo da validação é de outra época)."""
    atual = d
    for k in chaves:
        if not isinstance(atual, dict) or k not in atual:
            raise PreverError(f"{onde} não traz {'/'.join(map(str, chaves))}")
        atual = atual[k]
    return atual


def _finito(x, nome: str) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise PreverError(f"{nome} deve ser um número finito (recebi {x!r})")
    return float(x)


def _mes(t: pd.Timestamp) -> str:
    return f"{t.year:04d}-{t.month:02d}"


# ---------------------------------------------------------------- confiança do mês previsto
def referencia_do_mes(params_gama: pd.DataFrame, cfg: Config, cods, escalas, mes_calendario: int) -> tuple[np.ndarray, np.ndarray]:
    """`(invalido, baixa_conf)`, ambos (n municípios, n escalas) bool, para o mês-calendário do alvo.

    Mesma regra de `spi._aplicar`: sem parâmetro válido o SPI não existe (`invalido`); com parâmetro válido, é
    `baixa_conf` quando q > `spi.q_baixa_confianca` ou a média do treino por mês da janela
    ((1 - q) * shape * scale / k) fica abaixo de `spi.media_min_mm_por_mes`. Só usa parâmetros do treino."""
    faltam = [c for c in ("escala_meses", "cod_ibge", "mes_calendario", "q", "shape", "scale", "valido") if c not in params_gama.columns]
    if faltam:
        raise PreverError(f"params_gama sem as colunas {faltam}")
    p = params_gama.set_index(["escala_meses", "cod_ibge", "mes_calendario"])[["q", "shape", "scale", "valido"]]
    if p.index.duplicated().any():
        raise PreverError("params_gama com chave (escala, cod_ibge, mês) repetida")
    cods = [str(c) for c in cods]
    invalido, baixa = [], []
    for K in escalas:
        q = p.reindex(pd.MultiIndex.from_arrays([np.full(len(cods), K), cods, np.full(len(cods), mes_calendario)]))
        qq, a, esc = (q[k].to_numpy(dtype="float64") for k in ("q", "shape", "scale"))
        ok = q["valido"].fillna(False).to_numpy(dtype=bool) & np.isfinite(qq) & np.isfinite(a) & np.isfinite(esc)
        with np.errstate(invalid="ignore"):
            media = (1.0 - qq) * a * esc / K
            pouca = (qq > cfg.spi.q_baixa_confianca) | (media < cfg.spi.media_min_mm_por_mes)
        invalido.append(~ok)
        baixa.append(ok & pouca)
    return np.column_stack(invalido), np.column_stack(baixa)


# ---------------------------------------------------------------- números da validação, por cabeça
def _cabeca(alvo: str, escala: int, receita: dict, val: dict, mes_alvo: pd.Timestamp) -> dict:
    onde = "a validação (confirmacao_B.json)"
    principal = val.get("principal", "rede_final")
    rede = _pegar(val, "modelos", principal, alvo, onde=onde)
    pers = _pegar(val, "modelos", "persistencia", alvo, "todas", onde=onde)
    parc = _pegar(val, "modelos", "parcial_climatologica", alvo, "todas", onde=onde)
    seca = _pegar(val, "alerta_seca", alvo, principal, onde=onde)       # alarme no MESMO limiar que marca o município
    classes = _pegar(rede, "todas", "classes", onde=onde)
    marg = _pegar(val, "margem", alvo, onde=onde)
    comp_pers = _pegar(val, "comparacao", alvo, "persistencia", "veredito", onde=onde)
    comp_base = _pegar(val, "comparacao", alvo, "melhor_baseline", onde=onde)
    limiar = _finito(_pegar(receita, "limiares_alerta", alvo, onde="a receita do modelo"), f"limiar de {alvo}")
    limiar_val = _pegar(seca, "limiar_previsao", onde=onde)
    if not math.isclose(limiar, _finito(limiar_val, "limiar da validação")):
        raise PreverError(f"o limiar de alerta de {alvo} difere entre a receita ({limiar}) e a validação ({limiar_val})")
    por_mes = _pegar(rede, "por_mes_calendario", onde=onde)
    pers_por_mes = _pegar(val, "modelos", "persistencia", alvo, "por_mes_calendario", onde=onde)
    chave_mes = str(mes_alvo.month)
    if chave_mes not in por_mes or chave_mes not in pers_por_mes:
        raise PreverError(f"a validação não traz o erro do mês-calendário {chave_mes}")
    return {
        "id": f"spi{escala}", "escala": escala, "nome_alvo": alvo,
        "limiar_alerta": _arredondar(limiar),
        "margem": _arredondar(_finito(marg["valor"], "margem"), CASAS_METRICA),
        "margem_q": _finito(marg["q"], "q da margem"),
        "margem_cobertura_validacao": _arredondar(_finito(marg["cobertura_validacao"], "cobertura da margem"), CASAS_METRICA),
        "margem_cobertura_2a_metade": _arredondar(_finito(marg["cobertura_2a_metade"], "cobertura da margem na 2ª metade"), CASAS_METRICA),
        "margem_estavel": bool(marg["estavel"]),
        "n_linhas_validacao": int(_finito(rede["todas"]["n"], "n da validação")),
        "mae": _arredondar(_finito(rede["todas"]["mae"], "mae"), CASAS_METRICA),
        "mae_persistencia": _arredondar(_finito(pers["mae"], "mae da persistência"), CASAS_METRICA),
        "mae_parcial_climatologica": _arredondar(_finito(parc["mae"], "mae do baseline parcial"), CASAS_METRICA),
        "mae_mes_alvo": _arredondar(_finito(por_mes[chave_mes], "mae do mês-alvo"), CASAS_METRICA),
        "mae_mes_alvo_persistencia": _arredondar(_finito(pers_por_mes[chave_mes], "mae do mês-alvo (persistência)"), CASAS_METRICA),
        "veredito_persistencia": str(comp_pers),
        "melhor_baseline": str(_pegar(comp_base, "contra", onde=onde)),
        "veredito_melhor_baseline": str(_pegar(comp_base, "veredito", onde=onde)),
        "acerto_classe": _arredondar(_finito(classes["acerto"], "acerto de classe"), CASAS_METRICA),
        "acerto_sempre_normal": _arredondar(_finito(classes["acerto_sempre_normal"], "acerto de 'sempre normal'"), CASAS_METRICA),
        "secas_reais": int(seca["secas_reais"]), "secas_pegas": int(seca["tp"]), "alarmes_falsos": int(seca["fp"]),
        "revocacao": _arredondar(_finito(seca["revocacao"], "revocação"), CASAS_METRICA),
        "precisao": _arredondar(_finito(seca["precisao"], "precisão"), CASAS_METRICA),
    }


# ---------------------------------------------------------------- núcleo (sem arquivos)
def prever_proximo_mes(c: fe.Conjunto, preditor, *, params_gama: pd.DataFrame, cfg: Config, receita: dict, validacao: dict,
                       gerado_em: date) -> dict:
    """Previsão da última origem de `c`. `preditor.prever(Conjunto) -> (n, cabeças)` e `.desvio(Conjunto)`."""
    if len(c) == 0:
        raise PreverError("conjunto vazio")
    origem = c.meta["mes_origem"].max()
    cu = c.fatiar((c.meta["mes_origem"] == origem).to_numpy())
    blocos = {str(b) for b in cu.meta["split"].unique()}
    if blocos != {fe.POS_TESTE}:
        raise PreverError(f"a última origem ({_mes(origem)}) cai em {sorted(blocos)}; só aceito {fe.POS_TESTE!r}, "
                          "para nunca ler o teste congelado")
    if cu.M.any():
        raise PreverError("o mês-alvo da última origem já tem rótulo; isto não seria uma previsão do futuro")
    alvos = {t for t in cu.meta["mes_alvo"].unique()}
    mes_alvo = origem + pd.offsets.MonthBegin(cu.horizonte)
    if alvos != {mes_alvo}:
        raise PreverError(f"o mês-alvo ({sorted(map(str, alvos))}) não é a origem {_mes(origem)} + {cu.horizonte} mês(es)")
    if list(cu.alvos) != list(receita.get("alvos", [])):
        raise PreverError(f"cabeças do conjunto {list(cu.alvos)} diferem das da receita {receita.get('alvos')}")
    if cu.horizonte != 1:
        raise PreverError("a página de previsão do próximo mês só existe para h = 1")

    n, e = len(cu), len(cu.alvos)
    P = np.asarray(preditor.prever(cu), dtype="float64")
    D = np.asarray(preditor.desvio(cu), dtype="float64")
    if P.shape != (n, e) or D.shape != (n, e):
        raise PreverError(f"o preditor devolveu shapes {P.shape} e {D.shape}; esperado {(n, e)}")
    if not (np.isfinite(P).all() and np.isfinite(D).all()):
        raise PreverError("o preditor devolveu NaN ou infinito; nenhuma previsão parcial é publicada")

    cods = cu.meta["cod_ibge"].astype(str).tolist()
    if len(set(cods)) != n:
        raise PreverError("município repetido na última origem")
    invalido, baixa = referencia_do_mes(params_gama, cfg, cods, cu.escalas, mes_alvo.month)
    if invalido.any():
        ruins = sorted({cods[i] for i in np.where(invalido.any(axis=1))[0]})
        raise PreverError(f"sem parâmetro Gama válido para o mês-alvo em {len(ruins)} município(s): {ruins[:5]}")

    cabecas = [_cabeca(a, k, receita, validacao, mes_alvo) for a, k in zip(cu.alvos, cu.escalas)]
    municipios = []
    for i in np.argsort(np.asarray(cods)):
        linha: dict = {"cod_ibge": cods[i]}
        for j, h in enumerate(cabecas):
            v = _arredondar(P[i, j])
            linha[h["id"]] = {"previsto": v, "dispersao": _arredondar(D[i, j]), "baixa_conf": bool(baixa[i, j]),
                              "alerta": bool(v <= h["limiar_alerta"])}
        municipios.append(linha)
    ids = [h["id"] for h in cabecas]
    resumo = {"n_municipios": n, "n_alerta_algum": sum(any(m[k]["alerta"] for k in ids) for m in municipios),
              **{f"n_alerta_{k}": sum(m[k]["alerta"] for m in municipios) for k in ids},
              **{f"n_baixa_conf_{k}": sum(m[k]["baixa_conf"] for m in municipios) for k in ids}}
    fim_do_alvo = mes_alvo + pd.offsets.MonthBegin(1)
    return {
        "versao": VERSAO, "aviso": AVISO, "gerado_em": gerado_em.isoformat(),
        "origem": _mes(origem), "alvo": _mes(mes_alvo), "horizonte_meses": 1,
        "corte_b": str(_pegar(validacao, "corte_b", onde="a validação (confirmacao_B.json)")),
        "alvo_ja_passou": bool(pd.Timestamp(gerado_em) >= fim_do_alvo),
        "teste_congelado_lido": False,
        "dataset_sha256": str(validacao.get("dataset_sha256")),
        "cabecas": cabecas, "resumo": resumo, "municipios": municipios,
    }


# ---------------------------------------------------------------- arquivos
def gravar(doc: dict, destino) -> Path:
    """Escrita atômica; recusa NaN/infinito (`allow_nan=False`)."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    texto = json.dumps(doc, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    fd, tmp = tempfile.mkstemp(dir=destino.parent, prefix=destino.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(texto)
        os.chmod(tmp, 0o644)
        os.replace(tmp, destino)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return destino


def _ler_json(caminho: Path, o_que: str) -> dict:
    try:
        d = json.loads(caminho.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        raise PreverError(f"{o_que} ausente ou inválido ({caminho}): {err}") from err
    if not isinstance(d, dict):
        raise PreverError(f"{o_que} ({caminho}) não é um objeto JSON")
    return d


def gerar(cfg: Config, *, pasta_modelo=PASTA_MODELO_PADRAO, arquivo_validacao=VALIDACAO_PADRAO, destino=ARQUIVO_PADRAO,
          gerado_em: date | None = None) -> dict:
    """Lê dados, modelo e validação; confere que são do mesmo dataset; grava a previsão e devolve o documento."""
    if cfg.split.teste_liberado:
        raise PreverError("a previsão recusa rodar com split.teste_liberado: true (o teste congelado só entra na avaliação final)")
    pasta_modelo = Path(pasta_modelo)
    receita_txt = (pasta_modelo / "receita.json")
    receita = _ler_json(receita_txt, "a receita do modelo")
    validacao = _ler_json(Path(arquivo_validacao), "a validação")
    ins = entrada.carregar_insumos(cfg)
    for nome, d in (("a receita", receita), ("a validação", validacao)):
        if d.get("dataset_sha256") != ins.dataset_sha256:
            raise PreverError(f"{nome} é de outro dataset_sha256 ({d.get('dataset_sha256')!s:.12}…) que o dos dados "
                              f"lidos agora ({ins.dataset_sha256[:12]}…); retreine ou rode a avaliação de novo")
    c = fe.construir(ins.demo, ins.serie_imputada, ins.climatologia, ins.municipios, cfg)
    if receita.get("config", {}).get("regional"):
        c = rs.adicionar_regional(c)
    esperadas = [x for x in receita.get("colunas", []) if not x.startswith(rs.PREFIXO_BASE)]   # a base entra dentro do preditor
    if list(c.colunas) != esperadas:
        raise PreverError("as colunas das features diferem das usadas no treino do modelo salvo")
    modelo = bu.carregar_final(pasta_modelo, bl.ParcialClimatologica(ins.params_gama))
    doc = prever_proximo_mes(c, modelo, params_gama=ins.params_gama, cfg=cfg, receita=receita, validacao=validacao,
                             gerado_em=gerado_em or datetime.now(timezone.utc).date())
    doc["dataset_sha256"] = ins.dataset_sha256
    doc["modelo"] = {"pasta": pasta_modelo.as_posix(), "n_membros": int(receita["n_membros"]),
                     "receita_sha256": hashlib.sha256(receita_txt.read_bytes()).hexdigest()}
    try:
        pr.Previsao.model_validate(doc)                       # nunca grava um arquivo que o site recusaria
    except ValueError as err:
        raise PreverError(f"a previsão gerada não passa na conferência do site: {err}") from err
    gravar(doc, destino)
    return doc


def main(argv: list[str] | None = None, *, saida: Callable[[str], None] = print) -> int:
    ap = argparse.ArgumentParser(description="Gera data/reference/previsao_proximo_mes.json com o modelo final.")
    ap.add_argument("--params", default="params.yaml")
    ap.add_argument("--modelo", default=str(PASTA_MODELO_PADRAO))
    ap.add_argument("--validacao", default=str(VALIDACAO_PADRAO))
    ap.add_argument("--saida", default=str(ARQUIVO_PADRAO))
    a = ap.parse_args(argv)
    try:
        doc = gerar(load_config(a.params), pasta_modelo=a.modelo, arquivo_validacao=a.validacao, destino=a.saida)
    except (PreverError, entrada.EntradaError, bu.BuscaError) as err:
        saida(f"ERRO: {err}")
        return 1
    r = doc["resumo"]
    saida(f"previsão de {doc['alvo']} (origem {doc['origem']}) gravada em {a.saida}")
    saida(f"{r['n_municipios']} municípios; em alerta de seca: SPI-3 {r['n_alerta_spi3']}, SPI-6 {r['n_alerta_spi6']}, "
          f"algum dos dois {r['n_alerta_algum']}")
    if doc["alvo_ja_passou"]:
        saida(f"AVISO: {doc['alvo']} já passou; a série observada termina em {doc['origem']}")
    saida(doc["aviso"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
