"""Rascunho técnico de planejamento: pedido ao modelo, validação do que volta e o registro do rascunho.

O modelo recebe a ficha do município (`fatos.ficha_texto`) e a lista de itens da base do checklist (`checklist`) e
devolve um JSON `{"resumo": ..., "itens": [{"id": ..., "porque": ...}]}`. Nada do que volta é aceito sem passar por
`validar`:
  - os `id` existem na base da equipe (a IA escolhe, não cria itens);
  - todo número do texto está na ficha ou na base (a IA não calcula nem estima);
  - sem citação de lei, artigo, decreto, portaria ou S2iD (a fonte de cada item é da equipe, não da IA);
  - sem linguagem de certeza, de previsão ou de colapso (o site não prevê nada: contrato, seção 1.4);
  - texto simples, de um parágrafo, sem HTML, markdown ou link.
Um rascunho sai como `pendente_revisao`. Só vira `revisado` por uma pessoa (ver `arquivo.py`), e só o revisado vai ao site.

Este módulo não faz rede nem lê arquivo: recebe o cliente já pronto (`gerar(sistema, usuario, max_tokens=...)`, ver
`cliente.ClienteLLM`) e devolve dados.
"""
from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ia import fatos as ft
from src.ia.checklist import Checklist

MAX_RESUMO = 600
MAX_PORQUE = 300
MAX_ITENS_ESCOLHIDOS = 6
TENTATIVAS_PADRAO = 3
MAX_TOKENS = 1200
ROTULO = "Rascunho gerado por IA e revisado por humano, sem valor jurídico."
ROTULO_NAO_REVISADO = "Rascunho gerado por IA e ainda não revisado por uma pessoa, sem valor jurídico."
AVISO_MVP = "O site continua em MVP experimental, em calibração com S2iD/TCE-PI."


def rotulo_do_status(status: str) -> str:
    """O rótulo fixo que acompanha o texto. Só `revisado` pode dizer que uma pessoa revisou; o resto, nunca."""
    return ROTULO if status == "revisado" else ROTULO_NAO_REVISADO


class RascunhoError(ValueError):
    """Rascunho fora das regras, ou registro inconsistente."""


class RascunhoRecusado(RascunhoError):
    """O modelo não entregou um rascunho válido em todas as tentativas. `tentativas` guarda os motivos de cada uma."""

    def __init__(self, tentativas: list[list[str]]):
        self.tentativas = tentativas
        super().__init__("o modelo não entregou um rascunho válido: "
                         + " | ".join(f"tentativa {n}: {'; '.join(m)}" for n, m in enumerate(tentativas, 1)))


SISTEMA = f"""PAPEL
Você é analista técnico de apoio à Defesa Civil municipal no semiárido do Piauí. Escreve rascunhos de planejamento que um técnico da prefeitura vai ler, conferir e só então usar. Você não decide, não prevê e não substitui o plano do município: organiza o que os dados já mostram e aponta quais ações do checklist merecem atenção primeiro.

PÚBLICO E TOM
O leitor é coordenador ou técnico de Defesa Civil de um município pequeno, com pouco tempo e sem formação em estatística. Escreva em português do Brasil, com frases curtas, verbos de ação e sem jargão. Na primeira vez que usar "SPI", diga em poucas palavras que é um índice que compara a chuva dos últimos meses com o normal do local.

O QUE VOCÊ RECEBE
1. A FICHA DO MUNICÍPIO: fatos observados e já calculados pela equipe (município, mês de referência, SPI e sua classe, se a leitura é de baixa confiança, chuva do mês, chuva dos 12 meses e quantos meses recentes ficaram em seca). Se a ficha trouxer outros campos, trate-os do mesmo jeito: vale só o que está escrito.
2. O CHECKLIST: a lista de ações escrita pela equipe, cada uma com um id.

COMO ESCREVER O RESUMO (um parágrafo, até {MAX_RESUMO} caracteres)
Primeira frase: a situação do mês (município, mês, classe, SPI e chuva do mês). Segunda frase: o contexto dos 12 meses e quantos meses ficaram em seca. Última frase: o que isso pede em organização e acompanhamento, sem dizer o que vai acontecer depois.

COMO ESCOLHER AS AÇÕES (de 2 a {MAX_ITENS_ESCOLHIDOS})
- Escolha pelo id exato, só entre os da lista. Não crie, não renomeie e não reescreva ações.
- Seja proporcional. Classe normal ou úmida: ações de preparo e de acompanhamento. Seca moderada: ações de organização e de levantamento. Seca severa ou extrema, ou vários meses em seca nos últimos 12: inclua também ações de resposta.
- Nunca sugira medida pesada (documento de emergência, acionar o governo estadual) com base em uma única leitura ou em leitura de baixa confiança. Nesses casos, recomende acompanhar.

COMO ESCREVER O "porque" DE CADA AÇÃO (1 ou 2 frases, até {MAX_PORQUE} caracteres)
Ligue a ação a um fato da ficha e diga por que ela importa agora. Não repita o título da ação. Diga o que fazer com palavras simples, sem nomear atos legais.

NÚMEROS
Use só números da ficha, copiados exatamente (mesma vírgula, mesmo sinal, mesma unidade). Não calcule, estime, arredonde, some, subtraia, compare nem converta. Não escreva percentual, média, nem comparação com outro município ou outro ano. Não escreva número que não esteja na ficha, nem por extenso.

LINGUAGEM PROIBIDA
Não cite lei, decreto, portaria, resolução, norma, artigo, inciso, parágrafo nem S2iD: a base legal de cada ação é da equipe. Não faça previsão nem prognóstico e não diga que algo vai acontecer ("vai secar", "vai faltar", "colapso"). Não use palavras de certeza ("certamente", "garantido", "sem dúvida"). Evite até as palavras "previsão", "prever", "garantir" e "colapso", mesmo em frase negativa.

CARACTERES
Use só letras, números, pontuação e espaço comuns. Não use hífen que não quebra linha, espaço fino ou não separável, nem caracteres invisíveis. Separe todas as palavras com um espaço comum.

LEITURA DE BAIXA CONFIANÇA
Se a ficha disser que a leitura é de baixa confiança, diga no resumo que o SPI do mês pode exagerar, porque a chuva esperada na janela é pouca, e priorize acompanhar antes de agir.

SE FALTAR INFORMAÇÃO
Se a ficha não trouxer um dado, não invente e não deduza. Não preencha lacunas com conhecimento geral sobre o município, a região, o clima ou os açudes.

ANTES DE RESPONDER, CONFIRA
(a) cada número está na ficha, igual ao original; (b) cada id existe na lista e não se repete; (c) nenhuma palavra da linguagem proibida; (d) resumo com até {MAX_RESUMO} caracteres e cada "porque" com até {MAX_PORQUE}; (e) texto simples: sem lista, markdown, HTML, link ou quebra de linha dentro dos textos; (f) nada que soe como previsão.

FORMATO
Responda somente com um objeto JSON, sem texto antes ou depois:
{{"resumo": "até {MAX_RESUMO} caracteres", "itens": [{{"id": "id_da_acao", "porque": "até {MAX_PORQUE} caracteres"}}]}}"""


# ------------------------------------------------------------------ modelo de dados
class ItemEscolhido(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    id: str
    porque: str


class Rascunho(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    cod_ibge: str = Field(pattern=r"^\d{7}$")
    nome: str
    mes_referencia: str = Field(pattern=r"^\d{4}-\d{2}$")
    escala: int = Field(ge=1, le=24)
    ficha: str = Field(min_length=10)
    resumo: str
    itens: tuple[ItemEscolhido, ...] = Field(min_length=1, max_length=MAX_ITENS_ESCOLHIDOS)
    modelo: str = Field(min_length=1)
    gerado_em_utc: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    tentativas: int = Field(ge=1, le=20)
    status: Literal["pendente_revisao", "revisado", "rejeitado"] = "pendente_revisao"
    revisado_por: str | None = None
    revisado_em_utc: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

    @model_validator(mode="after")
    def _revisao_coerente(self):
        if self.status == "revisado" and not (self.revisado_por and self.revisado_em_utc):
            raise ValueError("rascunho revisado precisa de quem revisou e de quando")
        if self.status == "pendente_revisao" and (self.revisado_por or self.revisado_em_utc):
            raise ValueError("rascunho pendente não pode ter dados de revisão")
        return self


# ------------------------------------------------------------------ validação do texto
_CITACAO = re.compile(r"\b(?:leis?|decretos?|portarias?|resolu[cç][aã]o|resolu[cç][oõ]es|arts?|artigos?|incisos?|"
                      r"par[aá]grafos?|nbr|cobrade|s2id|constitui[cç][aã]o)\b|§", re.IGNORECASE)
_CERTEZA = re.compile(r"\b(?:certamente|com certeza|sem d[uú]vida|garant\w+|definitivamente|inevit[aá]ve\w+|"
                      r"prev[eê]|prever|prevemos|previs[aã]o|previs[oõ]es|prognóstico|colaps\w+|"
                      r"vai (?:secar|faltar|piorar|chover|acabar)|ir[aá] (?:secar|faltar|piorar|chover|acabar))\b",
                      re.IGNORECASE)
_FORMATO = re.compile(r"[<>`*#\[\]]|https?:|www\.|\n|\r", re.IGNORECASE)
_HIFENS_ESPECIAIS = frozenset("\u2010\u2011")      # hífen e hífen que não quebra linha: viram "-"


def limpar_texto(texto):
    """Troca hífens e espaços especiais (ex.: "SPI‑3", espaço não separável) pelos comuns. Só muda a aparência.

    Caractere invisível ou de controle NÃO é limpo aqui: `validar` recusa, para o modelo refazer o texto."""
    if not isinstance(texto, str):
        return texto
    return "".join("-" if c in _HIFENS_ESPECIAIS
                   else " " if c != " " and unicodedata.category(c) == "Zs"
                   else c for c in texto)


def _estranhos(texto: str) -> list[int]:
    """Pontos de código de controle, invisíveis (largura zero, hífen suave) ou espaços que não são o comum.
    Quebra de linha fica para `_FORMATO`."""
    return sorted({ord(c) for c in texto
                   if c not in "\n\r" and ((cat := unicodedata.category(c))[0] == "C" or (cat[0] == "Z" and c != " "))})


def _texto_ok(rotulo: str, texto, limite: int, permitidos: frozenset[str]) -> list[str]:
    if not isinstance(texto, str) or not texto.strip():
        return [f"{rotulo}: vazio ou não é texto"]
    motivos = []
    if len(texto) > limite:
        motivos.append(f"{rotulo}: {len(texto)} caracteres, o limite é {limite}")
    if texto != texto.strip():
        motivos.append(f"{rotulo}: espaço nas pontas")
    if _FORMATO.search(texto):
        motivos.append(f"{rotulo}: tem HTML, markdown, link ou quebra de linha")
    if (estranhos := _estranhos(texto)):
        cods = ", ".join(f"U+{n:04X}" for n in estranhos[:5])
        motivos.append(f"{rotulo}: tem caractere invisível ou especial ({cods}); use só letras, números, "
                       "pontuação e espaço comuns, com um espaço entre as palavras")
    if (c := _CITACAO.search(texto)):
        motivos.append(f"{rotulo}: cita fonte ou norma ({c.group(0)!r}); isso é da equipe")
    if (c := _CERTEZA.search(texto)):
        motivos.append(f"{rotulo}: linguagem de certeza ou de previsão ({c.group(0)!r})")
    fora = ft.numeros_fora(texto, permitidos)
    if fora:
        motivos.append(f"{rotulo}: número fora da ficha ({', '.join(fora)})")
    return motivos


def validar(resumo, itens, ficha: str, checklist: Checklist) -> list[str]:
    """Motivos pelos quais o texto não pode ser aceito (lista vazia = válido). `itens` é uma lista de (id, porque)."""
    permitidos = ft.numeros_permitidos(ficha, checklist.texto_para_modelo())
    motivos = _texto_ok("resumo", resumo, MAX_RESUMO, permitidos)
    if not isinstance(itens, (list, tuple)) or not itens:
        return motivos + ["itens: lista vazia"]
    if len(itens) > MAX_ITENS_ESCOLHIDOS:
        motivos.append(f"itens: {len(itens)}, o máximo é {MAX_ITENS_ESCOLHIDOS}")
    if len(itens) < min(2, len(checklist.itens)):
        motivos.append("itens: escolha pelo menos 2")
    conhecidos, vistos = checklist.por_id(), []
    for n, (id_, porque) in enumerate(itens, 1):
        if id_ not in conhecidos:
            motivos.append(f"item {n}: id {id_!r} não existe na base do checklist")
        elif id_ in vistos:
            motivos.append(f"item {n}: id {id_!r} repetido")
        vistos.append(id_)
        motivos += _texto_ok(f"item {n} ({id_})", porque, MAX_PORQUE, permitidos)
    return motivos


def interpretar(resposta: str) -> tuple[str, list[tuple[str, str]]]:
    """Lê o JSON do modelo (aceita cercas ``` em volta). Forma errada = RascunhoError."""
    t = resposta.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", t).strip()
    try:
        d = json.loads(t)
    except ValueError as e:
        raise RascunhoError("a resposta não é um JSON válido") from e
    if not isinstance(d, dict) or set(d) != {"resumo", "itens"}:
        raise RascunhoError('o JSON precisa ter exatamente as chaves "resumo" e "itens"')
    if not isinstance(d["itens"], list):
        raise RascunhoError('"itens" precisa ser uma lista')
    itens = []
    for n, x in enumerate(d["itens"], 1):
        if not isinstance(x, dict) or set(x) != {"id", "porque"} or not all(isinstance(v, str) for v in x.values()):
            raise RascunhoError(f'item {n}: precisa ser {{"id": texto, "porque": texto}}')
        itens.append((x["id"], limpar_texto(x["porque"])))
    return limpar_texto(d["resumo"]), itens


# ------------------------------------------------------------------ pedido
def montar_pedido(ficha: str, checklist: Checklist, recusas: list[str] | None = None) -> str:
    partes = [f"FICHA DO MUNICÍPIO\n{ficha}", f"CHECKLIST (escolha só destes ids)\n{checklist.texto_para_modelo()}"]
    if recusas:
        partes.append("A resposta anterior foi recusada pelos motivos abaixo. Corrija e responda de novo, só com o JSON.\n"
                      + "\n".join(f"- {m}" for m in recusas))
    partes.append("Escreva o rascunho agora.")
    return "\n\n".join(partes)


def gerar_rascunho(f: ft.Fatos, checklist: Checklist, cliente, *, agora: datetime,
                   tentativas: int = TENTATIVAS_PADRAO, max_tokens: int = MAX_TOKENS) -> Rascunho:
    """Pede o rascunho ao `cliente` até `tentativas` vezes, recusando o que não passa em `validar`.

    Falha de transporte (rede, HTTP, texto cortado) sobe sem nova tentativa: não adianta repetir o mesmo pedido."""
    if tentativas < 1:
        raise ValueError("tentativas precisa ser pelo menos 1")
    if agora.tzinfo is None:
        raise ValueError("`agora` precisa ter fuso (UTC)")
    ficha = ft.ficha_texto(f)                                       # recusa município sem SPI
    recusas: list[str] = []
    historico: list[list[str]] = []
    for n in range(1, tentativas + 1):
        resposta = cliente.gerar(SISTEMA, montar_pedido(ficha, checklist, recusas), max_tokens=max_tokens)
        try:
            resumo, itens = interpretar(resposta)
            motivos = validar(resumo, itens, ficha, checklist)
        except RascunhoError as e:
            motivos = [str(e)]
        if not motivos:
            return Rascunho(cod_ibge=f.cod_ibge, nome=f.nome, mes_referencia=f.mes, escala=f.escala, ficha=ficha,
                            resumo=resumo, itens=tuple(ItemEscolhido(id=i, porque=p) for i, p in itens),
                            modelo=cliente.modelo, gerado_em_utc=agora.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                            tentativas=n)
        historico.append(motivos)
        recusas = motivos
    raise RascunhoRecusado(historico)
