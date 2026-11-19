"""Linha de comando dos rascunhos de IA: gerar, ver, listar, aprovar e rejeitar.

É a camada que junta as peças: confere o manifest, monta o pacote do site (para ter a mesma ficha que o monitor
mostra), chama o modelo e grava `data/reference/rascunhos_ia.json`. Por isso é o único módulo de `src/ia` que conhece
`src.web.build_site`; o domínio (`fatos`, `checklist`, `rascunhos`, `arquivo`, `cliente`) não depende dele.

Uso (PowerShell, depois de `python -m src.data.build_demo`):
    $env:ANTHROPIC_API_KEY="sua-chave"                     # só nesta máquina; nunca no projeto
    python -m src.ia.cli gerar --municipios 2207801,2205706   # cria rascunhos pendentes (cada chamada ao modelo custa)
    # Provedor gratuito, com reserva (cada chave só nesta máquina):
    $env:GEMINI_API_KEY="sua-chave"; $env:GROQ_API_KEY="sua-chave"
    python -m src.ia.cli gerar --provedor gemini,groq --municipios 2207801   # Gemini; se falhar, Groq
    python -m src.ia.cli ver 2207801                          # leia o texto e a ficha antes de decidir
    python -m src.ia.cli aprovar 2207801 --por "Seu Nome"     # só uma pessoa aprova; o texto é validado de novo
    python -m src.ia.cli rejeitar 2207801 --por "Seu Nome"
    python -m src.ia.cli listar
    python -m src.web.build_site                              # a página Planejamento aparece se houver revisados

A base do checklist (`data/reference/checklist_base.json`) é escrita pela equipe (formato em `checklist.py`).
Aprovar é ato humano: nenhum comando daqui aprova sozinho.
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.data import manifest as mf
from src.data.config import Config, load_config
from src.ia import arquivo as aq
from src.ia import checklist as ck
from src.ia import cliente as cl
from src.ia import cliente_compat_openai as co
from src.ia import cliente_gemini as cg
from src.ia import fatos as ft
from src.ia import provedores as pv
from src.ia import rascunhos as rs
from src.web import build_site as bs

MAX_PADRAO = 10
_COD = re.compile(r"^\d{7}$")


class CliError(ValueError):
    """Pedido que não faz sentido (código malformado, limite estourado, arquivo de outro dataset)."""


def _municipios(texto: str, maximo: int) -> list[str]:
    cods = list(dict.fromkeys(c.strip() for c in texto.split(",") if c.strip()))
    if not cods:
        raise CliError("informe ao menos um código IBGE em --municipios")
    ruins = [c for c in cods if not _COD.match(c)]
    if ruins:
        raise CliError(f"código IBGE precisa ter 7 dígitos: {', '.join(ruins)}")
    if len(cods) > maximo:
        raise CliError(f"{len(cods)} municípios passam do limite de {maximo} (cada um é uma chamada paga ao modelo); "
                       f"aumente com --max se for de propósito")
    return cods


def _checklist(cfg: Config):
    caminho = cfg.raiz / ck.ARQUIVO_PADRAO
    return ck.carregar(caminho), ck.sha256(caminho)


def _dataset_atual(cfg: Config) -> str:
    return mf.verificar_manifest(cfg)["dataset_sha256"]


# ------------------------------------------------------------------ comandos (sem print; devolvem dados)
def executar_gerar(cfg: Config, municipios: list[str], cliente, *, mes: str | None = None, escala: int | None = None,
                   tentativas: int = rs.TENTATIVAS_PADRAO, sobrescrever: bool = False, recomecar: bool = False,
                   agora: datetime | None = None) -> dict:
    """Gera e grava um rascunho por município. Grava a cada sucesso, para uma falha no meio não perder o que já foi pago."""
    agora = agora or datetime.now(timezone.utc)
    lista, sha_ck = _checklist(cfg)
    payload = bs.montar_payload(bs.carregar_insumos(cfg), cfg, verificado_em=agora)
    caminho = cfg.raiz / aq.ARQUIVO_PADRAO
    arq = aq.abrir_ou_criar(caminho, payload["verificacao"]["dataset_sha256"], sha_ck, recomecar=recomecar)
    resultado: dict[str, dict] = {}
    for cod in municipios:
        f = ft.fatos_municipio(payload, cod, mes, escala)
        if not f.utilizavel:
            resultado[cod] = {"nome": f.nome, "situacao": "sem_dado"}
            continue
        antigo = arq.rascunhos.get(cod)
        if antigo is not None and antigo.status == "revisado" and not sobrescrever:
            resultado[cod] = {"nome": f.nome, "situacao": "ja_revisado"}
            continue
        try:
            r = rs.gerar_rascunho(f, lista, cliente, agora=agora, tentativas=tentativas)
        except rs.RascunhoRecusado as e:
            resultado[cod] = {"nome": f.nome, "situacao": "recusado", "motivos": e.tentativas}
            continue
        arq = aq.registrar(arq, r, sobrescrever_revisado=sobrescrever)
        aq.gravar(arq, caminho)
        resultado[cod] = {"nome": f.nome, "situacao": "pendente_revisao", "tentativas": r.tentativas}
    return {"arquivo": str(caminho), "modelo": cliente.modelo, "municipios": resultado}


def _carregar_arquivo(cfg: Config) -> tuple[aq.ArquivoRascunhos, Path]:
    caminho = cfg.raiz / aq.ARQUIVO_PADRAO
    arq = aq.carregar(caminho)
    if arq is None:
        raise CliError(f"ainda não há rascunhos ({caminho}); rode `gerar` primeiro")
    return arq, caminho


def executar_revisao(cfg: Config, acao: str, cod: str, por: str, *, agora: datetime | None = None) -> aq.ArquivoRascunhos:
    agora = agora or datetime.now(timezone.utc)
    arq, caminho = _carregar_arquivo(cfg)
    if acao == "aprovar":
        lista, sha_ck = _checklist(cfg)
        if arq.checklist_sha256 != sha_ck:
            raise CliError("a base do checklist mudou depois destes rascunhos; gere de novo (--recomecar)")
        if arq.dataset_sha256 != _dataset_atual(cfg):
            raise CliError("os rascunhos são de outro dataset; gere de novo (--recomecar)")
        novo = aq.aprovar(arq, cod, por, agora=agora, checklist=lista)
    elif acao == "rejeitar":
        novo = aq.rejeitar(arq, cod, por, agora=agora)
    else:
        raise CliError(f"ação desconhecida: {acao}")
    aq.gravar(novo, caminho)
    return novo


def texto_do_rascunho(r: rs.Rascunho, lista: ck.Checklist | None) -> str:
    itens = lista.por_id() if lista else {}
    linhas = [f"{r.nome} ({r.cod_ibge}) | {ft.mes_por_extenso(r.mes_referencia)} | SPI-{r.escala} | situação: {r.status}",
              f"gerado em {r.gerado_em_utc} por {r.modelo} ({r.tentativas} tentativa(s))", "", "FICHA (os únicos números permitidos)",
              r.ficha, "", "RESUMO", r.resumo, "", "ITENS ESCOLHIDOS"]
    for i in r.itens:
        t = itens.get(i.id)
        linhas += [f"- {t.titulo if t else i.id} [{i.id}]", f"  por quê: {i.porque}"]
    if r.revisado_por:
        linhas += ["", f"revisado por {r.revisado_por} em {r.revisado_em_utc}"]
    return "\n".join(linhas)


# ------------------------------------------------------------------ linha de comando
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.ia.cli", description="Rascunhos de planejamento escritos por IA, "
                                 "sempre revisados por uma pessoa antes de irem ao site.")
    ap.add_argument("--params", default="params.yaml")
    sub = ap.add_subparsers(dest="comando", required=True)
    g = sub.add_parser("gerar", help="pede rascunhos ao modelo (chamada paga) e grava como pendentes")
    g.add_argument("--municipios", required=True, help="códigos IBGE separados por vírgula")
    g.add_argument("--mes", default=None, help="AAAA-MM; padrão: o mês de abertura do site")
    g.add_argument("--escala", type=int, default=None, help="escala do SPI; padrão: a do alvo")
    g.add_argument("--provedor", default=pv.PADRAO,
                   help=f"{', '.join(pv.FABRICAS)}; vários separados por vírgula formam uma lista com reserva "
                        f"(ex.: gemini,groq); padrão: {pv.PADRAO}")
    g.add_argument("--modelo", default=None, help=f"só com um provedor; padrão: ${cl.VAR_MODELO} ou {cl.MODELO_PADRAO} "
                   f"(Claude), ${cg.VAR_MODELO} (Gemini), ${co.VAR_MODELO_GROQ} (Groq)")
    g.add_argument("--tentativas", type=int, default=rs.TENTATIVAS_PADRAO)
    g.add_argument("--max", type=int, default=MAX_PADRAO, help="limite de municípios por execução")
    g.add_argument("--sobrescrever", action="store_true", help="refaz também os já revisados")
    g.add_argument("--recomecar", action="store_true", help="descarta rascunhos de outro dataset ou checklist")
    sub.add_parser("listar", help="mostra todos os rascunhos e a situação de cada um")
    v = sub.add_parser("ver", help="mostra um rascunho inteiro, com a ficha")
    v.add_argument("cod_ibge")
    for nome, ajuda in (("aprovar", "marca como revisado (valida o texto de novo)"), ("rejeitar", "marca como rejeitado")):
        p = sub.add_parser(nome, help=ajuda)
        p.add_argument("cod_ibge")
        p.add_argument("--por", required=True, help="nome de quem revisou")
    return ap


def main(argv: list[str] | None = None, *, fabrica_cliente=cl.desde_ambiente, fabrica_provedores=pv.montar_cliente,
         saida=print) -> int:
    a = _parser().parse_args(argv)
    try:
        cfg = load_config(a.params)
        saida(f"AVISO: {cfg.projeto.maturidade}")
        if a.comando == "gerar":
            cods = _municipios(a.municipios, a.max)
            lista, _ = _checklist(cfg)                       # falha cedo, antes de gastar com o modelo
            if a.provedor.strip().lower() == pv.PADRAO:
                cliente = fabrica_cliente(a.modelo)                  # comportamento de sempre (Claude)
            else:
                cliente = fabrica_provedores(a.provedor, a.modelo,
                                             aviso=lambda quem, motivo: saida(f"AVISO: {quem}: {motivo}"))
            r = executar_gerar(cfg, cods, cliente, mes=a.mes, escala=a.escala, tentativas=a.tentativas,
                               sobrescrever=a.sobrescrever, recomecar=a.recomecar)
            saida(f"modelo: {r['modelo']} | arquivo: {r['arquivo']} | checklist com {len(lista.itens)} itens")
            usados = getattr(cliente, "usados", None)
            if usados:                                               # lista com reserva: quem escreveu o quê
                saida("respostas por modelo: " + ", ".join(f"{m} = {n}" for m, n in usados.items()))
            for cod, x in r["municipios"].items():
                extra = {"pendente_revisao": f"pendente de revisão ({x.get('tentativas')} tentativa(s))",
                         "sem_dado": "sem SPI no mês de referência; nada a escrever",
                         "ja_revisado": "já revisado; use --sobrescrever para refazer",
                         "recusado": "recusado em todas as tentativas: " + " | ".join("; ".join(m) for m in x.get("motivos", []))}[x["situacao"]]
                saida(f"  {cod} {x['nome']}: {extra}")
            if any(x["situacao"] == "pendente_revisao" for x in r["municipios"].values()):
                saida("Próximo passo: leia com `ver <código>` e decida com `aprovar` ou `rejeitar`. Nada vai ao site antes disso.")
        elif a.comando == "listar":
            arq, _ = _carregar_arquivo(cfg)
            vale = arq.dataset_sha256 == _dataset_atual(cfg)
            saida(f"dataset {arq.dataset_sha256[:16]}… ({'o atual' if vale else 'OUTRO: não vale para o site'}) | "
                  f"{len(arq.rascunhos)} rascunho(s)")
            for cod, r in sorted(arq.rascunhos.items()):
                quem = f" | {r.revisado_por}" if r.revisado_por else ""
                saida(f"  {cod} {r.nome} | {r.mes_referencia} | {r.status}{quem}")
        elif a.comando == "ver":
            arq, _ = _carregar_arquivo(cfg)
            if a.cod_ibge not in arq.rascunhos:
                raise CliError(f"não há rascunho para {a.cod_ibge}")
            try:
                lista, _ = _checklist(cfg)
            except ck.ChecklistError:
                lista = None
            saida(texto_do_rascunho(arq.rascunhos[a.cod_ibge], lista))
            saida("\n" + rs.ROTULO)
        else:
            novo = executar_revisao(cfg, a.comando, a.cod_ibge, a.por)
            r = novo.rascunhos[a.cod_ibge]
            saida(f"{r.nome} ({r.cod_ibge}): {r.status} por {r.revisado_por} em {r.revisado_em_utc}")
            if a.comando == "aprovar":
                saida("Rode `python -m src.web.build_site` para a página Planejamento incluir este rascunho.")
        saida(rs.AVISO_MVP)
    except (ValueError, RuntimeError, OSError, ImportError) as e:
        print(f"ERRO: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
