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
    # Todos os municípios (continua de onde parou; nunca refaz quem já tem rascunho):
    python -m src.ia.cli gerar --provedor groq,gemini --todos --so-faltantes --ordem seca --max 60
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
MAX_FALHAS_SEGUIDAS = 3                  # falhas do provedor em sequência que interrompem a execução
ORDENS = ("codigo", "seca")
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


def _tokens_validos(*valores) -> bool:
    return all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in valores)


def tokens_do_uso(uso) -> int | None:
    """Total de tokens de uma chamada, a partir do `usage` de qualquer provedor; None se não der para saber."""
    if not isinstance(uso, dict):
        return None
    for chave in ("total_tokens", "totalTokenCount"):                  # OpenAI/Groq e Gemini já trazem o total
        if chave in uso:
            return uso[chave] if _tokens_validos(uso[chave]) else None
    for entrada, saida in (("prompt_tokens", "completion_tokens"), ("input_tokens", "output_tokens")):
        if entrada in uso and saida in uso:
            return uso[entrada] + uso[saida] if _tokens_validos(uso[entrada], uso[saida]) else None
    return None


class _ContaUso:
    """Cumpre `ClienteLLM` por cima de outro cliente e soma os tokens de cada chamada (inclusive as recusadas)."""

    def __init__(self, cliente):
        self._cliente = cliente
        self.total = 0
        self.visto = False                                              # algum provedor informou o uso?

    @property
    def modelo(self) -> str:
        return self._cliente.modelo

    def gerar(self, sistema: str, usuario: str, *, max_tokens: int = 1200) -> str:
        texto = self._cliente.gerar(sistema, usuario, max_tokens=max_tokens)
        n = tokens_do_uso(getattr(self._cliente, "ultimo_uso", None))
        if n is not None:
            self.total += n
            self.visto = True
        return texto


def _motivo_de_pular(antigo: rs.Rascunho | None, *, sobrescrever: bool, so_faltantes: bool) -> str | None:
    if antigo is not None and antigo.status == "revisado" and not sobrescrever:
        return "ja_revisado"
    if antigo is not None and so_faltantes:
        return "ja_tem"
    return None


# ------------------------------------------------------------------ comandos (sem print; devolvem dados)
def executar_gerar(cfg: Config, municipios: list[str], cliente, *, mes: str | None = None, escala: int | None = None,
                   tentativas: int = rs.TENTATIVAS_PADRAO, sobrescrever: bool = False, recomecar: bool = False,
                   agora: datetime | None = None, todos: bool = False, so_faltantes: bool = False,
                   ordem: str = "codigo", limite: int | None = None, progresso=None,
                   max_falhas_seguidas: int = MAX_FALHAS_SEGUIDAS) -> dict:
    """Gera e grava um rascunho por município. Grava a cada sucesso, para uma falha no meio não perder o que já foi pago.

    `todos`: usa todos os municípios do pacote (ignora `municipios`). `so_faltantes`: não chama o modelo para quem já tem
    rascunho de qualquer status (pendente, revisado ou rejeitado), para continuar de onde parou. `ordem="seca"` começa
    pelo SPI mais baixo. `limite`: máximo de municípios que chamam o modelo nesta execução (pulados não contam).
    `progresso(posição, total, cod, info)` é chamado a cada município resolvido. Falha do provedor (`ErroLLM`) num
    município o marca "falhou" e segue; `max_falhas_seguidas` falhas em sequência interrompem a execução."""
    if so_faltantes and sobrescrever:
        raise CliError("--so-faltantes e --sobrescrever se contradizem: um só refaz quem falta, o outro refaz o que existe")
    if ordem not in ORDENS:
        raise CliError(f"ordem desconhecida: {ordem!r}; use {', '.join(ORDENS)}")
    if limite is not None and limite < 1:
        raise CliError("--max precisa ser pelo menos 1")
    if max_falhas_seguidas < 1:
        raise CliError("max_falhas_seguidas precisa ser pelo menos 1")
    agora = agora or datetime.now(timezone.utc)
    lista, sha_ck = _checklist(cfg)
    payload = bs.montar_payload(bs.carregar_insumos(cfg), cfg, verificado_em=agora)
    codigos = [m["cod"] for m in payload["municipios"]] if todos else list(municipios)
    fatos = {cod: ft.fatos_municipio(payload, cod, mes, escala) for cod in codigos}     # município inválido falha ANTES de gastar
    if ordem == "seca":
        codigos.sort(key=lambda c: (not fatos[c].utilizavel, fatos[c].spi_cent if fatos[c].utilizavel else 0, c))
    caminho = cfg.raiz / aq.ARQUIVO_PADRAO
    arq = aq.abrir_ou_criar(caminho, payload["verificacao"]["dataset_sha256"], sha_ck, recomecar=recomecar)
    resultado: dict[str, dict] = {}
    conta = _ContaUso(cliente)
    chamadas = restantes = falhas_seguidas = 0
    interrompido: str | None = None
    for pos, cod in enumerate(codigos, 1):
        f = fatos[cod]
        if not f.utilizavel:
            info = {"nome": f.nome, "situacao": "sem_dado"}
        elif (pulo := _motivo_de_pular(arq.rascunhos.get(cod), sobrescrever=sobrescrever, so_faltantes=so_faltantes)):
            info = {"nome": f.nome, "situacao": pulo}
            if pulo == "ja_tem":
                info["status"] = arq.rascunhos[cod].status
        elif interrompido is not None or (limite is not None and chamadas >= limite):
            restantes += 1
            continue
        else:
            chamadas += 1
            antes = conta.total
            try:
                r = rs.gerar_rascunho(f, lista, conta, agora=agora, tentativas=tentativas)
            except rs.RascunhoRecusado as e:
                falhas_seguidas = 0                                     # o provedor respondeu; o texto é que foi recusado
                info = {"nome": f.nome, "situacao": "recusado", "motivos": e.tentativas}
            except cl.ErroLLM as e:
                falhas_seguidas += 1
                info = {"nome": f.nome, "situacao": "falhou", "motivo": str(e)}
                if falhas_seguidas >= max_falhas_seguidas:
                    interrompido = f"{falhas_seguidas} falhas seguidas do provedor; a última foi: {e}"
            else:
                falhas_seguidas = 0
                arq = aq.registrar(arq, r, sobrescrever_revisado=sobrescrever)
                aq.gravar(arq, caminho)
                info = {"nome": f.nome, "situacao": "pendente_revisao", "tentativas": r.tentativas}
            if conta.visto:
                info["tokens"] = conta.total - antes
        resultado[cod] = info
        if progresso is not None:
            progresso(pos, len(codigos), cod, info)
    return {"arquivo": str(caminho), "modelo": cliente.modelo, "municipios": resultado, "restantes": restantes,
            "interrompido": interrompido, "tokens": conta.total if conta.visto else None}


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


# ------------------------------------------------------------------ texto da linha de comando
_SILENCIOSAS_COM_TODOS = frozenset({"ja_tem", "ja_revisado", "sem_dado"})     # com --todos só entram no resumo
_ROTULOS_RESUMO = (("pendente_revisao", "novos"), ("recusado", "recusados"), ("falhou", "falharam"),
                   ("sem_dado", "sem dado"), ("ja_tem", "já tinham rascunho"), ("ja_revisado", "já revisados"))


def _descricao(x: dict) -> str:
    return {"pendente_revisao": lambda: f"pendente de revisão ({x.get('tentativas')} tentativa(s))",
            "sem_dado": lambda: "sem SPI no mês de referência; nada a escrever",
            "ja_revisado": lambda: "já revisado; use --sobrescrever para refazer",
            "ja_tem": lambda: f"já tem rascunho ({x.get('status')}); nada a fazer",
            "falhou": lambda: f"falhou: {x.get('motivo')}",
            "recusado": lambda: "recusado em todas as tentativas: " + " | ".join("; ".join(m) for m in x.get("motivos", []))
            }[x["situacao"]]()


def _resumo(r: dict) -> str:
    contagem: dict[str, int] = {}
    for x in r["municipios"].values():
        contagem[x["situacao"]] = contagem.get(x["situacao"], 0) + 1
    partes = [f"{rotulo}: {contagem[sit]}" for sit, rotulo in _ROTULOS_RESUMO if contagem.get(sit)]
    if r["tokens"] is not None:
        partes.append(f"tokens no total: {r['tokens']}")
    return "Resumo: " + (" | ".join(partes) if partes else "nenhum município processado")


# ------------------------------------------------------------------ linha de comando
def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m src.ia.cli", description="Rascunhos de planejamento escritos por IA, "
                                 "sempre revisados por uma pessoa antes de irem ao site.")
    ap.add_argument("--params", default="params.yaml")
    sub = ap.add_subparsers(dest="comando", required=True)
    g = sub.add_parser("gerar", help="pede rascunhos ao modelo (chamada paga) e grava como pendentes")
    alvo = g.add_mutually_exclusive_group(required=True)
    alvo.add_argument("--municipios", help="códigos IBGE separados por vírgula")
    alvo.add_argument("--todos", action="store_true", help="todos os municípios do pacote que têm SPI no mês de referência")
    g.add_argument("--so-faltantes", action="store_true",
                   help="não chama o modelo para quem já tem rascunho (de qualquer status); serve para continuar de onde parou")
    g.add_argument("--ordem", choices=ORDENS, default="codigo",
                   help="codigo (padrão) ou seca (do SPI mais baixo ao mais alto; útil com --max no plano grátis)")
    g.add_argument("--mes", default=None, help="AAAA-MM; padrão: o mês de abertura do site")
    g.add_argument("--escala", type=int, default=None, help="escala do SPI; padrão: a do alvo")
    g.add_argument("--provedor", default=pv.PADRAO,
                   help=f"{', '.join(pv.FABRICAS)}; vários separados por vírgula formam uma lista com reserva "
                        f"(ex.: gemini,groq); padrão: {pv.PADRAO}")
    g.add_argument("--modelo", default=None, help=f"só com um provedor; padrão: ${cl.VAR_MODELO} ou {cl.MODELO_PADRAO} "
                   f"(Claude), ${cg.VAR_MODELO} (Gemini), ${co.VAR_MODELO_GROQ} (Groq)")
    g.add_argument("--tentativas", type=int, default=rs.TENTATIVAS_PADRAO)
    g.add_argument("--max", type=int, default=None,
                   help=f"com --municipios: quantos códigos aceita (padrão {MAX_PADRAO}); com --todos: quantos municípios "
                        f"chamam o modelo nesta execução (padrão: sem limite; quem foi pulado não conta)")
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
            cods = [] if a.todos else _municipios(a.municipios, a.max if a.max is not None else MAX_PADRAO)
            lista, _ = _checklist(cfg)                       # falha cedo, antes de gastar com o modelo
            if a.provedor.strip().lower() == pv.PADRAO:
                cliente = fabrica_cliente(a.modelo)                  # comportamento de sempre (Claude)
            else:
                cliente = fabrica_provedores(a.provedor, a.modelo,
                                             aviso=lambda quem, motivo: saida(f"AVISO: {quem}: {motivo}"))

            def andamento(pos, total, cod, x):                       # uma linha por município, na hora em que termina
                if a.todos and x["situacao"] in _SILENCIOSAS_COM_TODOS:
                    return
                tokens = f" | {x['tokens']} tokens" if x.get("tokens") else ""
                saida(f"[{pos}/{total}] {cod} {x['nome']}: {_descricao(x)}{tokens}")
            r = executar_gerar(cfg, cods, cliente, mes=a.mes, escala=a.escala, tentativas=a.tentativas,
                               sobrescrever=a.sobrescrever, recomecar=a.recomecar, todos=a.todos,
                               so_faltantes=a.so_faltantes, ordem=a.ordem, limite=a.max if a.todos else None,
                               progresso=andamento)
            saida(f"modelo: {r['modelo']} | arquivo: {r['arquivo']} | checklist com {len(lista.itens)} itens")
            usados = getattr(cliente, "usados", None)
            if usados:                                               # lista com reserva: quem escreveu o quê
                saida("respostas por modelo: " + ", ".join(f"{m} = {n}" for m, n in usados.items()))
            saida(_resumo(r))
            if r["interrompido"]:
                saida(f"Parei: {r['interrompido']}. O que já foi gerado está salvo; quando o provedor voltar, "
                      f"rode de novo com --so-faltantes.")
            if r["restantes"]:
                saida(f"Faltam {r['restantes']} município(s) desta lista; rode de novo com --so-faltantes para continuar.")
            situacoes = [x["situacao"] for x in r["municipios"].values()]
            if "falhou" in situacoes:
                saida("Alguns falharam por erro do provedor (veja os AVISO acima); rode de novo com --so-faltantes "
                      "para tentar só os que faltam.")
            if "pendente_revisao" in situacoes:
                saida("Próximo passo: leia com `ver <código>` e decida com `aprovar` ou `rejeitar`. Nada vai ao site antes disso.")
            if "falhou" in situacoes:                                 # inclui toda interrupção: ela só nasce de falhas
                saida(rs.AVISO_MVP)
                return 1
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
            saida("\n" + rs.rotulo_do_status(arq.rascunhos[a.cod_ibge].status))
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
