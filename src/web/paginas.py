"""Páginas do site: lista, navegação, textos derivados dos dados e preenchimento dos modelos HTML.

Só monta texto e HTML a partir do pacote (`payload`) já conferido; não lê o manifest, não calcula SPI e não usa
rede. Cada `{{nome}}` dos modelos precisa existir em `valores`: marcador sem valor é erro, nunca HTML pela metade.
Texto simples é escapado; fragmento já montado (`Html`) entra como está.
"""
from __future__ import annotations

import html as _html
import math
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

from src.web import marca

DIR_MODELOS = Path(__file__).resolve().parent / "paginas"
FUSO = ZoneInfo("America/Fortaleza")
MESES = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
         "novembro", "dezembro")
ABREV = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")


class PaginaError(ValueError):
    """Modelo com marcador sem valor, ou dado que não permite montar a página."""


class Html(str):
    """Fragmento HTML já pronto (não é escapado de novo)."""


@dataclass(frozen=True)
class Pagina:
    id: str
    arquivo: str
    rotulo: str
    titulo: str
    descricao: str
    scripts: tuple[str, ...] = ()
    modelo: str = ""


PAGINAS = (
    Pagina("inicio", "index.html", "Início", "Monitor de seca do semiárido piauiense",
           "Onde e quando a chuva faltou no semiárido do Piauí, mês a mês, com a confiança de cada leitura.",
           ("inicio",), "inicio.html"),
    Pagina("monitor", "monitor.html", "Monitor", "Monitor",
           "Mapa, linha do tempo e série de cada município do semiárido piauiense, com a classe do SPI e a confiança da leitura.",
           ("monitor",), "monitor.html"),
    Pagina("municipios", "municipios.html", "Municípios", "Municípios",
           "Tabela dos municípios do semiárido piauiense em um mês: SPI, classe, chuva e confiança da leitura.",
           ("municipios",), "municipios.html"),
    Pagina("el-nino", "el-nino.html", "El Niño", "El Niño e a seca no semiárido",
           "O que o CPC/NOAA e o INMET dizem sobre o El Niño de 2026-2027 e como foram as estações chuvosas do Piauí.",
           ("clima",), "el-nino.html"),
    Pagina("metodo", "metodo.html", "Método", "Método e limites",
           "Como o SPI é calculado, quando a leitura tem baixa confiança e o que o site ainda não mostra.", (), "metodo.html"),
    Pagina("sobre", "sobre.html", "O projeto", "O projeto",
           "O Piauí Resiliente, o que já existe, o que está em preparação e a marca.", (), "sobre.html"),
)

# Só entra na lista quando há rascunho de IA para mostrar (ver `paginas_do_site`).
PAGINA_PLANEJAMENTO = Pagina(
    "planejamento", "planejamento.html", "Planejamento", "Rascunhos de planejamento",
    "Rascunhos técnicos de apoio à Defesa Civil, escritos com IA e sem valor jurídico; cada um diz se uma pessoa já o revisou.",
    ("planejamento",), "planejamento.html")


def paginas_do_site(payload: dict) -> tuple[Pagina, ...]:
    """As seis páginas fixas, mais Planejamento (antes de Método) se o pacote traz rascunhos de IA."""
    if not payload.get("planejamento"):
        return PAGINAS
    i = next(n for n, p in enumerate(PAGINAS) if p.id == "metodo")
    return PAGINAS[:i] + (PAGINA_PLANEJAMENTO,) + PAGINAS[i:]


# ------------------------------------------------------------------ pequenos formatadores
def esc(s) -> str:
    return _html.escape(str(s), quote=True)


def num_br(x: float, casas: int = 0) -> str:
    """Número no padrão brasileiro. Arredonda meio para longe do zero, como `arred` do core.js: o texto da página e o
    rótulo do gráfico não podem divergir (o `format` do Python arredondaria −37,5 para −38 e o Math.round do JS, para −37)."""
    if not math.isfinite(x):
        raise PaginaError(f"número não finito: {x!r}")
    d = Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-casas), rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)                                           # nada de "−0"
    s = f"{d:,.{casas}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    return s.replace("-", "−")


def mes_extenso(iso_mes: str) -> str:
    a, m = iso_mes[:7].split("-")
    return f"{MESES[int(m) - 1]} de {a}"


def dia_br(iso: str) -> str:
    a, m, d = str(iso)[:10].split("-")
    return f"{d}/{m}/{a}"


def data_hora_br(iso_utc: str) -> str:
    """Instante ISO com fuso (Z ou +00:00) no horário de Fortaleza. Sem fuso é erro: a conta dependeria da máquina."""
    d = datetime.fromisoformat(str(iso_utc).replace("Z", "+00:00"))
    if d.tzinfo is None:
        raise PaginaError(f"data e hora sem fuso horário: {iso_utc!r}")
    return d.astimezone(FUSO).strftime("%d/%m/%Y %H:%M")


def lista_pt(itens: list[str], conj: str = "ou") -> str:
    itens = [str(i) for i in itens]
    if not itens:
        return ""
    return itens[0] if len(itens) == 1 else ", ".join(itens[:-1]) + f" {conj} " + itens[-1]


def intervalo_meses(meses: list[int]) -> str:
    return f"{MESES[meses[0] - 1]} a {MESES[meses[-1] - 1]}"


# ------------------------------------------------------------------ preenchimento
_MARCADOR = re.compile(r"\{\{([a-z_0-9]+)\}\}")


def preencher(modelo: str, valores: dict, onde: str) -> str:
    def troca(m: re.Match) -> str:
        chave = m.group(1)
        if chave not in valores:
            raise PaginaError(f"{onde}: o modelo pede {{{{{chave}}}}} e não há valor")
        v = valores[chave]
        return v if isinstance(v, Html) else esc(v)
    return _MARCADOR.sub(troca, modelo)


def ler_modelo(nome: str) -> str:
    return (DIR_MODELOS / nome).read_text(encoding="utf-8")


def html_nav(paginas: tuple[Pagina, ...], atual: str) -> Html:
    return Html("".join(f'<a href="{esc(p.arquivo)}"' + (' aria-current="page"' if p.id == atual else "") + f">{esc(p.rotulo)}</a>"
                        for p in paginas))


def html_aviso(maturidade: str) -> Html:
    resto = re.sub(r"^MVP experimental\s*[—-]?\s*", "", maturidade, flags=re.I).strip()
    resto = (resto[:1].upper() + resto[1:]).rstrip(".") + ". " if resto else ""
    return Html(f"<b>MVP experimental.</b> {esc(resto)}Não substitui alerta oficial: mostra a chuva já observada, não uma previsão.")


# ------------------------------------------------------------------ El Niño
def _enso_indisponivel() -> Html:
    return Html('<section class="secao"><h2>El Niño agora</h2><p class="sub">Indisponível: o arquivo de fontes oficiais '
                '(data/reference/contexto_enso.json) não foi encontrado.</p></section>')


def _fatos(s: dict, cpc: dict) -> str:
    d = dia_br(cpc["emitido_em"])
    return ('<div class="fatos">'
            f'<div><strong>+{num_br(s["nino34_c"], 1)} °C</strong><span>Índice Niño-3.4, segundo o CPC em {d}.</span></div>'
            f'<div><strong>mais de {s["prob_forte_ou_muito_forte_min_pct"]}%</strong><span>Chance de El Niño {esc(s["forca_pt"])} {esc(s["periodo_forte_pt"])}.</span></div>'
            f'<div><strong>cerca de {s["prob_evento_historico_pct"]}%</strong><span>Chance de um evento histórico (RONI de +2,5 °C ou mais) em {esc(s["periodo_historico_pt"])}.</span></div>'
            '</div>')


def _proveniencia(fonte: dict) -> str:
    prox = f" · próxima discussão em {dia_br(fonte['proxima_atualizacao'])}" if fonte.get("proxima_atualizacao") else ""
    return (f'{esc(fonte["instituicao"])}, {esc(fonte["titulo"])}, emitido em {dia_br(fonte["emitido_em"])}{prox}. '
            f'<a href="{esc(fonte["url"])}" rel="noopener">Abrir a fonte</a>')


def _estado(s: dict) -> str:
    return (f'<div class="estado"><strong>{esc(s["status_alerta"])}</strong><span class="selo atencao">{esc(s["forca_pt"])}</span></div>'
            f'<p style="margin-top:6px">{esc(s["resumo_pt"])}</p>')


def html_enso_resumo(enso: dict | None) -> Html:
    if not enso:
        return _enso_indisponivel()
    s = enso["situacao"]
    cpc = next(f for f in enso["fontes"] if f["id"] == s["fonte"])
    return Html(
        '<section class="secao" aria-labelledby="h-enso"><div class="bloco-enso"><div>'
        '<h2 id="h-enso">El Niño agora</h2>'
        '<p class="sub">O que o Pacífico equatorial indica, segundo o CPC/NOAA. É contexto de fonte oficial, não uma previsão do nosso modelo.</p>'
        f'<p class="proveniencia">{_proveniencia(cpc)}.</p>'
        '<p class="proveniencia"><span class="selo atencao" id="ensoAtualizacao" hidden>A próxima discussão já saiu: confira a fonte</span></p>'
        '<div class="acoes"><a class="btn" href="el-nino.html">Ver o contexto completo</a></div></div>'
        f'<div>{_estado(s)}{_fatos(s, cpc)}'
        f'<p class="aviso-contexto"><b>Contexto, não previsão do nosso modelo.</b> {esc(s["impactos_pt"])}</p></div></div></section>')


def html_enso_status(enso: dict | None) -> Html:
    if not enso:
        return _enso_indisponivel()
    s = enso["situacao"]
    cpc = next(f for f in enso["fontes"] if f["id"] == s["fonte"])
    return Html(
        '<section class="secao" aria-labelledby="h-status"><h2 id="h-status">O que o Pacífico indica agora</h2>'
        f'{_estado(s)}{_fatos(s, cpc)}'
        f'<p class="aviso-contexto"><b>Contexto, não previsão do nosso modelo.</b> {esc(s["impactos_pt"])}</p>'
        f'<p class="proveniencia">{_proveniencia(cpc)}.</p>'
        '<p class="proveniencia"><span class="selo atencao" id="ensoAtualizacao" hidden>A próxima discussão já saiu: confira a fonte</span></p></section>')


def html_enso_leituras(enso: dict | None) -> Html:
    if not enso:
        return Html('<li>Indisponível: sem arquivo de fontes oficiais.</li>')
    por_id = {f["id"]: f for f in enso["fontes"]}
    itens = []
    for x in enso["leituras"]:
        f = por_id[x["fonte"]]
        link = f["url"] if not f.get("url_documento") else f["url_documento"]
        itens.append(f'<li><span class="escopo">{esc(x["escopo_pt"])}</span><span>{esc(x["texto"])}</span>'
                     f'<span class="de">{esc(f["instituicao"])}, {esc(f["titulo"])}, {dia_br(f["emitido_em"])}. '
                     f'<a href="{esc(link)}" rel="noopener">Abrir a fonte</a></span></li>')
    return Html("".join(itens))


def html_enso_limites(enso: dict | None) -> Html:
    return Html("".join(f"<li>{esc(x)}</li>" for x in (enso["limites"] if enso else ("Sem arquivo de fontes oficiais.",))))


# ------------------------------------------------------------------ planejamento (rascunhos de IA)
def _selo_revisao(r: dict) -> str:
    if r["revisado"]:
        return f'<span class="selo">Revisado por {esc(r["revisado_por"])} em {esc(data_hora_br(r["revisado_em"]))}</span>'
    return '<span class="selo atencao">Ainda não revisado por uma pessoa</span>'


def html_planejamento(pac: dict | None) -> Html:
    """Um bloco recolhível por município. Todo texto vem do pacote e é escapado; a fonte de cada item é a escrita pela
    equipe. Cada bloco carrega o rótulo fixo que diz se uma pessoa revisou: o texto de quem não foi revisado nunca
    aparece sem o rótulo de não revisado."""
    if not pac:
        return Html("")
    blocos = []
    for r in pac["rascunhos"]:
        itens = "".join(
            f'<li><b>{esc(i["titulo"])}</b><span>{esc(i["descricao"])}</span><p>{esc(i["porque"])}</p>'
            f'<small>Fonte do item: {esc(i["fonte"])}</small></li>' for i in r["itens"])
        rotulo = pac["rotulo"] if r["revisado"] else pac["rotulo_nao_revisado"]
        fecho = ("Quem revisou responde pelo conteúdo." if r["revisado"]
                 else "Nenhuma pessoa leu este texto ainda: confira antes de usar.")
        blocos.append(
            f'<details class="rascunho" id="m{esc(r["cod"])}" data-rev="{1 if r["revisado"] else 0}">'
            f'<summary><h3>{esc(r["nome"])}</h3><span class="selos"><span class="selo">{esc(mes_extenso(r["mes"]))}</span>'
            f'<span class="selo">SPI-{int(r["escala"])}</span>{_selo_revisao(r)}</span></summary>'
            f'<p class="rotulo-plano">{esc(rotulo)}</p>'
            f'<p class="corpo">{esc(r["resumo"])}</p><ul class="itens-plano">{itens}</ul>'
            f'<p class="nota">Texto escrito por IA ({esc(r["modelo"])}) a partir da ficha do monitor e de itens escritos pela '
            f'equipe. {fecho}</p></details>')
    return Html("".join(blocos))


def _contagem_planos(pac: dict) -> tuple[int, int]:
    """(planos, revisados por uma pessoa)."""
    return len(pac["rascunhos"]), sum(1 for r in pac["rascunhos"] if r["revisado"])


def sobre_planejamento(pac: dict | None) -> str:
    """O que a página O projeto diz sobre o cartão do município e o rascunho técnico: só o que existe de fato."""
    if not pac:
        return ("Em preparação. O rascunho para a Defesa Civil é um documento de apoio, escrito com IA e sem valor jurídico. "
                "Ainda não há rascunho nesta versão do site.")
    n, rev = _contagem_planos(pac)
    return (f"Disponível como MVP na página Planejamento: {n} {'rascunho' if n == 1 else 'rascunhos'} escritos com IA a partir "
            f"da ficha do município, {rev} revisado{'' if rev == 1 else 's'} por uma pessoa. Os outros mostram o selo “Ainda não "
            f"revisado”. É um documento de apoio, sem valor jurídico, e não substitui o plano do município.")


def resumo_planejamento(pac: dict | None) -> str:
    """Quantos planos há e quantos uma pessoa já revisou, em uma frase."""
    if not pac:
        return ""
    n, rev = _contagem_planos(pac)
    return (f"Esta página tem {n} {'plano' if n == 1 else 'planos'}: {rev} "
            f"{'revisado' if rev == 1 else 'revisados'} por uma pessoa e {n - rev} "
            f"ainda não {'revisado' if n - rev == 1 else 'revisados'}.")


def texto_checklist(pac: dict | None) -> str:
    if not pac:
        return ""
    c = pac["checklist"]
    return f"Base do checklist: versão {c['versao']}, de {c['elaborado_por']}, atualizada em {dia_br(c['atualizado_em'])}."


# ------------------------------------------------------------------ textos derivados do pacote
def frase_abertura(payload: dict) -> str:
    t, ultimo = payload["abertura"], len(payload["meses"]) - 1
    mes, fim = mes_extenso(payload["meses"][t]), mes_extenso(payload["meses"][ultimo])
    base = f"O monitor abre em {mes}, o último mês em que a maioria dos municípios tem leitura confiável."
    if t == ultimo:
        return f"O monitor abre no mês mais recente, {mes}. A maioria dos municípios tem leitura confiável."
    n, pct = ultimo - t, payload["pct_baixa_ultimo"]
    resto = (f"O mês seguinte, {fim}, está no monitor, mas {pct}% das leituras dele têm baixa confiança." if n == 1 else
             f"Os {n} meses seguintes, até {fim}, estão no monitor, mas {pct}% das leituras de {fim} têm baixa confiança.")
    return f"{base} {resto}"


def regra_baixa_confianca(payload: dict) -> str:
    r = payload["regra_baixa"]
    motivos = []
    if r["media_min_mm_por_mes"] > 0:
        motivos.append(f"média do período de ajuste abaixo de {num_br(r['media_min_mm_por_mes'], 0 if float(r['media_min_mm_por_mes']).is_integer() else 1)} mm por mês da janela")
    motivos.append(f"mais de {round(r['q_max'] * 100)}% dos anos de ajuste sem chuva na janela")
    return f"Baixa confiança: meses em que a chuva esperada é muito baixa ({' ou '.join(motivos)}), quando poucos milímetros mudam muito o SPI."


def tabela_classes(payload: dict) -> Html:
    return Html("".join(f'<tr><td><span class="sw" style="background:var(--c{i})"></span>{esc(c["rotulo"])}</td><td>{esc(c["faixa"])}</td></tr>'
                        for i, c in enumerate(payload["classes"])))


def textos_clima(payload: dict) -> dict:
    c = payload["clima"]
    est = c["estacao"]
    meses = est["meses"]
    quebra = any(b < a for a, b in zip(meses, meses[1:]))             # a estação passa de dezembro para janeiro
    n_txt = {3: "três", 4: "quatro", 5: "cinco", 6: "seis"}.get(est["largura"], str(est["largura"]))
    frase_clima = f"Os {n_txt} meses mais chuvosos, de {intervalo_meses(meses)}, trazem {num_br(c['pct_estacao'])}% da chuva do ano."
    anos = est["anos"]
    if not anos:
        raise PaginaError("não há nenhuma estação chuvosa completa na série: o texto do clima não pode ser montado")
    secas = sorted(anos, key=lambda x: x["desvio_pct"])[:3]
    secas_txt = lista_pt([f"{x['ano']} ({num_br(x['desvio_pct'])}%)" for x in secas], "e")
    ult = anos[-1]
    if round(abs(ult["desvio_pct"])) == 0:
        situacao = "ficou na média"
    else:
        situacao = f"ficou {num_br(abs(ult['desvio_pct']))}% {'abaixo' if ult['desvio_pct'] < 0 else 'acima'} da média"
    abre = {1: "A estação mais seca foi a de", 2: "As duas estações mais secas foram as de"}.get(
        len(secas), "As três estações mais secas foram as de")
    frase_est = f"{abre} {secas_txt}. A mais recente completa, a de {ult['ano']}, {situacao}."
    if quebra:
        frase_est += " O ano indicado é o do início da estação."
    curto = f"{ABREV[meses[0] - 1]} a {ABREV[meses[-1] - 1]}"
    return {"frase_clima": frase_clima, "frase_estacoes": frase_est, "estacao_meses_curto": curto}


def valores_base(payload: dict, ctx: dict) -> dict:
    """Valores comuns a todas as páginas. `ctx` traz o que não está no pacote: sha, datas de extração e verificação."""
    escalas = payload["escalas"]
    ini, fim = payload["periodo"]["inicio"], payload["periodo"]["fim"]
    sha = payload["verificacao"]["dataset_sha256"]
    ext = payload["fonte"].get("extraido_em") or []
    extraido = ""
    if ext:
        a, b = data_hora_br(ext[0]), data_hora_br(ext[-1])
        extraido = f", extraído em {a}" + (f" a {b}" if b != a else "")
    v = {
        "n_mun": payload["n_municipios"], "ano_ini": ini[:4], "mes_fim": mes_extenso(fim),
        "periodo_ini": mes_extenso(ini), "periodo_fim": mes_extenso(fim), "extraido": extraido,
        "treino_fim_ano": payload["periodo"]["treino_fim"][:4], "teste_fim": mes_extenso(payload["periodo"]["teste_fim"]),
        "escala": payload["escala_alvo"], "escalas_txt": lista_pt([str(k) for k in escalas]),
        "equipe": payload["equipe"], "equipe_final": payload["equipe"].rstrip().rstrip(".") + ".",
        "n_excluidos": payload["n_excluidos"],
        "gee_id": payload["fonte"].get("chirps") or "CHIRPS", "malha_ano": payload["fonte"]["malha_ano"],
        "arquivos": payload["verificacao"]["arquivos"], "verificado_em": data_hora_br(payload["verificacao"]["verificado_em"]),
        "sha": sha, "sha_curto": sha[:16],
        "abertura_mes": mes_extenso(payload["meses"][payload["abertura"]]),
        "frase_abertura": frase_abertura(payload), "regra_bc": regra_baixa_confianca(payload),
        "tabela_classes": tabela_classes(payload),
        "enso_resumo": html_enso_resumo(payload.get("enso")), "enso_status": html_enso_status(payload.get("enso")),
        "enso_leituras": html_enso_leituras(payload.get("enso")), "enso_limites": html_enso_limites(payload.get("enso")),
        "planejamento_conteudo": html_planejamento(payload.get("planejamento")),
        "rotulo_ia": (payload.get("planejamento") or {}).get("rotulo", ""),
        "rotulo_nao_revisado": (payload.get("planejamento") or {}).get("rotulo_nao_revisado", ""),
        "planejamento_resumo": resumo_planejamento(payload.get("planejamento")),
        "sobre_planejamento": sobre_planejamento(payload.get("planejamento")),
        "checklist_info": texto_checklist(payload.get("planejamento")),
    }
    v.update(textos_clima(payload))
    v.update(ctx)
    return v


def renderizar_pagina(pagina: Pagina, paginas: tuple[Pagina, ...], payload: dict, ctx: dict | None = None) -> str:
    """HTML completo de uma página. Os marcadores de topo e rodapé vêm do modelo base."""
    v = valores_base(payload, ctx or {})
    conteudo = preencher(ler_modelo(pagina.modelo), v, pagina.arquivo)
    scripts = "".join(f'<script src="assets/{esc(s)}.js"></script>\n' for s in pagina.scripts)
    v.update({
        "pagina": pagina.id, "titulo_pagina": pagina.titulo, "descricao": pagina.descricao,
        "favicon": Html(marca.favicon_data_uri()), "simbolo": Html(marca.simbolo()),
        "nav": html_nav(paginas, pagina.id), "aviso": html_aviso(payload["maturidade"]),
        "conteudo": Html(conteudo), "scripts": Html(scripts),
    })
    return preencher(ler_modelo("_base.html"), v, f"{pagina.arquivo} (base)")
