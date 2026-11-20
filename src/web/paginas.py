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
           ("aovivo", "monitor"), "monitor.html"),
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


# Só entra na lista quando há previsão conferida (ver `paginas_do_site`).
PAGINA_PREVISAO = Pagina(
    "previsao", "previsao.html", "Previsão", "Previsão do próximo mês",
    "Estimativa experimental do SPI do próximo mês em cada município do semiárido, com margem de erro e o que a validação mostrou.",
    ("previsao",), "previsao.html")


def paginas_do_site(payload: dict) -> tuple[Pagina, ...]:
    """As seis páginas fixas, mais Previsão (antes de El Niño) se o pacote traz a previsão conferida e Planejamento
    (antes de Método) se traz rascunhos de IA."""
    lista = list(PAGINAS)
    if payload.get("previsao"):
        lista.insert(next(n for n, p in enumerate(lista) if p.id == "el-nino"), PAGINA_PREVISAO)
    if payload.get("planejamento"):
        lista.insert(next(n for n, p in enumerate(lista) if p.id == "metodo"), PAGINA_PLANEJAMENTO)
    return tuple(lista)


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


def html_aviso(maturidade: str, com_previsao: bool = False) -> Html:
    resto = re.sub(r"^MVP experimental\s*[—-]?\s*", "", maturidade, flags=re.I).strip()
    resto = (resto[:1].upper() + resto[1:]).rstrip(".") + ". " if resto else ""
    fecho = ("Não substitui alerta oficial: o SPI e o mapa mostram a chuva já observada; a previsão do modelo está na página "
             "Previsão, com margem de erro, e a ficha do Monitor traz a previsão de chuva da Open-Meteo, outra fonte."
             if com_previsao else
             "Não substitui alerta oficial: o SPI e o mapa mostram a chuva já observada, não uma previsão; a ficha do Monitor "
             "traz a previsão de chuva da Open-Meteo, outra fonte.")
    return Html(f"<b>MVP experimental.</b> {esc(resto)}{fecho}")


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


# ------------------------------------------------------------------ previsão do próximo mês
FAIXA_MIN, FAIXA_MAX = -2.5, 2.5                    # a faixa desenhada cobre SPI de −2,5 a +2,5; o que passar disso é cortado
NOME_BASELINE = {"parcial_climatologica": "a conta simples com a chuva mediana do mês", "persistencia": "repetir o mês anterior",
                 "climatologia": "a chuva mediana de sempre", "ridge": "a regressão linear (Ridge)",
                 "floresta_aleatoria": "a floresta aleatória"}
INDISPONIVEL_PREVISAO = ("Indisponível: o modelo está em calibração. Só aparece depois de uma avaliação auditada, "
                         "com margem de erro.")


def _spi_br(x: float) -> str:
    return num_br(x, 2)


def _n_municipios(n: int) -> str:
    return f"{n} município" if n == 1 else f"{n} municípios"


def _pct_br(x: float) -> str:
    return num_br(x * 100, 0)


def _pos_faixa(x: float) -> float:
    return (min(max(x, FAIXA_MIN), FAIXA_MAX) - FAIXA_MIN) / (FAIXA_MAX - FAIXA_MIN) * 100.0


def _faixa(cel: dict, cab: dict) -> str:
    """Desenho de uma previsão: ponto no SPI previsto, barra com a margem de erro e a zona de alerta de seca atrás."""
    lo, hi = cel["previsto"] - cab["margem"], cel["previsto"] + cab["margem"]
    a, b = _pos_faixa(lo), _pos_faixa(hi)
    return (f'<span class="faixa" aria-hidden="true"><i class="zona" style="width:{_pos_faixa(cab["limiar_alerta"]):.1f}%"></i>'
            f'<i class="zero"></i><i class="mg" style="left:{a:.1f}%;width:{b - a:.1f}%"></i>'
            f'<i class="pt" style="left:{_pos_faixa(cel["previsto"]):.1f}%"></i></span>')


def _celula_previsao(cel: dict, cab: dict) -> str:
    lo, hi = cel["previsto"] - cab["margem"], cel["previsto"] + cab["margem"]
    bc = '<small class="bc">baixa confiança</small>' if cel["baixa_conf"] else ""
    return (f'<td class="celula-prev">{_faixa(cel, cab)}<span class="v">{_spi_br(cel["previsto"])}</span>'
            f'<span class="sr"> (faixa provável de {_spi_br(lo)} a {_spi_br(hi)})</span>{bc}</td>')


def html_previsao_tabela(pac: dict | None, municipios: list[dict]) -> Html:
    """Linhas dos municípios, do SPI previsto mais baixo para o mais alto (vale o menor dos dois SPI)."""
    if not pac:
        return Html("")
    nomes = {m["cod"]: m["nome"] for m in municipios}
    cab = {c["id"]: c for c in pac["cabecas"]}
    faltam = [m["cod_ibge"] for m in pac["municipios"] if m["cod_ibge"] not in nomes]
    if faltam:
        raise PaginaError(f"previsão de município que não está no site: {faltam[:5]}")
    ordem = sorted(pac["municipios"], key=lambda m: (min(m["spi3"]["previsto"], m["spi6"]["previsto"]), nomes[m["cod_ibge"]]))
    linhas = []
    for m in ordem:
        alvos = [f"SPI-{cab[i]['escala']}" for i in ("spi3", "spi6") if m[i]["alerta"]]
        sinal = (f'<span class="selo atencao">Sinal de seca: {lista_pt(alvos, "e")}</span>' if alvos
                 else '<span class="sem-sinal">Sem sinal</span>')
        linhas.append(
            f'<tr data-sinal="{1 if alvos else 0}"><th scope="row"><a href="monitor.html#m={esc(m["cod_ibge"])}">'
            f'{esc(nomes[m["cod_ibge"]])}</a></th>{_celula_previsao(m["spi3"], cab["spi3"])}'
            f'{_celula_previsao(m["spi6"], cab["spi6"])}<td>{sinal}</td></tr>')
    return Html("".join(linhas))


def html_previsao_aviso(pac: dict | None, payload: dict) -> Html:
    if not pac:
        return Html("")
    alvo, origem = mes_extenso(pac["alvo"]), mes_extenso(pac["origem"])
    if pac["alvo_ja_passou"]:
        data = (f"O mês previsto, {esc(alvo)}, já passou: a chuva medida que o site tem termina em {esc(origem)}. Esta página mostra "
                f"o que o modelo previu com o último mês disponível e não serve para decidir hoje. Quando a chuva de {esc(alvo)} "
                f"entrar nos dados, ela será a primeira conferência real da previsão, fora da validação.")
        if tem_conferencia(payload.get("conferencia"), pac):
            data += (f" Enquanto isso, a seção “O que choveu em {esc(MESES[int(pac['alvo'][5:7]) - 1])}” traz uma conferência "
                     f"indicativa, feita com a chuva de outra fonte.")
    else:
        data = (f"O mês previsto, {esc(alvo)}, ainda não tem chuva medida no site, que termina em {esc(origem)}. Quando ela entrar "
                f"nos dados, será a primeira conferência real da previsão, fora da validação.")
    return Html(f'<div class="aviso-previsao"><p><b>Experimento, não alerta.</b> {esc(pac["aviso"])}</p><p>{data}</p></div>')


def html_previsao_fatos(pac: dict | None) -> Html:
    if not pac:
        return Html("")
    c3, c6 = (next(c for c in pac["cabecas"] if c["id"] == i) for i in ("spi3", "spi6"))
    r = pac["resumo"]
    return Html(
        f'<div class="fatos">'
        f'<div><strong>{r["n_alerta_algum"]} de {r["n_municipios"]}</strong><span>municípios com sinal de seca em '
        f'{esc(mes_extenso(pac["alvo"]))}: {r["n_alerta_spi3"]} pelo SPI-3 e {r["n_alerta_spi6"]} pelo SPI-6. O sinal é um alarme '
        f'experimental, não uma certeza.</span></div>'
        f'<div><strong>±{_spi_br(c3["margem"])} e ±{_spi_br(c6["margem"])}</strong><span>margem de erro do SPI-3 e do SPI-6. Na validação, '
        f'{_pct_br(c3["margem_q"])}% dos erros ficaram dentro dela.</span></div>'
        f'<div><strong>{_pct_br(c3["revocacao"])}% e {_pct_br(c6["revocacao"])}%</strong><span>das secas reais da validação que o alarme '
        f'pegou, no SPI-3 e no SPI-6. Quando ele soou, acertou {_pct_br(c3["precisao"])}% e {_pct_br(c6["precisao"])}% das vezes.</span></div>'
        f'</div>')


def _frase_baseline(c: dict) -> str:
    nome = NOME_BASELINE.get(c["melhor_baseline"], c["melhor_baseline"])
    n = f" ({num_br(c['mae_parcial_climatologica'], 2)})" if c["melhor_baseline"] == "parcial_climatologica" else ""
    return {"superou": f"{nome}{n} errou um pouco mais que a previsão", "pior": f"{nome}{n} errou menos que a previsão",
            "indistinguivel": f"entre a previsão e {nome}{n} não há diferença clara",
            "indefinido": f"a comparação com {nome} não é conclusiva"}[c["veredito_melhor_baseline"]]


def html_previsao_confianca(pac: dict | None, payload: dict) -> Html:
    """O que a validação mostrou, em frases com os números do arquivo. Nada aqui é calculado: só formatado."""
    if not pac:
        return Html("")
    c = {x["id"]: x for x in pac["cabecas"]}
    c3, c6, r = c["spi3"], c["spi6"], pac["resumo"]
    mes_alvo = MESES[int(pac["alvo"][5:7]) - 1]
    itens = []
    for h in (c3, c6):
        pers = {"superou": "errou menos que", "pior": "errou mais que", "indistinguivel": "ficou sem diferença clara de",
                "indefinido": "não deu para comparar com"}[h["veredito_persistencia"]]
        itens.append(
            f'<li><b>Erro médio no SPI-{h["escala"]}: {_spi_br(h["mae"])}</b><span>Na validação, a previsão {pers} repetir o mês '
            f'anterior ({_spi_br(h["mae_persistencia"])}); {_frase_baseline(h)}.</span></li>')
    itens.append(
        f'<li><b>No mês desta previsão o erro foi outro</b><span>Só nos meses de {esc(mes_alvo)} da validação, o erro médio foi '
        f'{_spi_br(c3["mae_mes_alvo"])} no SPI-3 e {_spi_br(c6["mae_mes_alvo"])} no SPI-6, contra {_spi_br(c3["mae"])} e '
        f'{_spi_br(c6["mae"])} em todos os meses.</span></li>')
    itens.append(
        f'<li><b>O alarme de seca pega pouco</b><span>Pegou {c3["secas_pegas"]} de {c3["secas_reais"]} secas reais no SPI-3 e '
        f'{c6["secas_pegas"]} de {c6["secas_reais"]} no SPI-6, com {c3["alarmes_falsos"]} e {c6["alarmes_falsos"]} alarmes falsos. '
        f'Sem sinal de seca não quer dizer sem seca.</span></li>')
    instaveis = [f"SPI-{h['escala']}" for h in (c3, c6) if not h["margem_estavel"]]
    estab = (f" No {lista_pt(instaveis, 'e')} ela não se manteve estável entre as duas metades: trate a margem como aproximada."
             if instaveis else "")
    itens.append(
        f'<li><b>A margem de erro é aproximada</b><span>Calibrada para cobrir {_pct_br(c3["margem_q"])}% dos erros. Calibrada só na '
        f'primeira metade da validação, ela cobriu {_pct_br(c3["margem_cobertura_2a_metade"])}% dos erros da segunda (SPI-3) e '
        f'{_pct_br(c6["margem_cobertura_2a_metade"])}% (SPI-6). A segunda metade começa em {esc(mes_extenso(pac["corte_b"]))}.{esc(estab)}</span></li>')
    itens.append(
        f'<li><b>Baixa confiança</b><span>Em {esc(mes_alvo)} se espera pouca chuva em {_n_municipios(r["n_baixa_conf_spi3"])} no SPI-3 e '
        f'{r["n_baixa_conf_spi6"]} no SPI-6. Neles, poucos milímetros mudam muito o índice; eles aparecem marcados na tabela.</span></li>')
    itens.append(
        f'<li><b>O teste final ainda não foi aberto</b><span>Em {esc(dia_br(pac["gerado_em"]))}, os dados reservados para a medida '
        f'definitiva (até {esc(mes_extenso(payload["periodo"]["teste_fim"]))}) continuavam sem uso. Todos os números acima são de validação, '
        f'onde a própria equipe escolheu configurações.</span></li>')
    return Html(f'<ul class="indisp-grande">{"".join(itens)}</ul>')


def html_previsao_legenda(pac: dict | None) -> Html:
    if not pac:
        return Html("")
    c = pac["cabecas"][0]
    exemplo = {"previsto": -0.4, "dispersao": 0, "baixa_conf": False, "alerta": False}
    return Html(f'<div class="leg-faixa">{_faixa(exemplo, {**c, "margem": 0.7})}<ul>'
                f'<li><b>Ponto</b>: o SPI previsto.</li>'
                f'<li><b>Barra</b>: a faixa provável, o previsto mais e menos a margem de erro.</li>'
                f'<li><b>Fundo terracota</b>: zona de seca (SPI no limiar do alarme ou abaixo dele; {_spi_br(c["limiar_alerta"])} no SPI-3).</li>'
                f'<li><b>Linha fina</b>: SPI zero, a chuva normal.</li></ul></div>')


def _sinal_br(x: float, casas: int = 1) -> str:
    """Desvio com sinal explícito: +0,7 / −1,4 / 0,0."""
    return f"+{num_br(x, casas)}" if round(x, casas) > 0 else num_br(x, casas)


def tem_conferencia(conf: dict | None, pac: dict | None) -> bool:
    """A conferência só aparece (e só é citada nos outros textos) quando há previsão e pelo menos um município com sinal e dado."""
    return bool(conf and pac and conf["grupos"]["com_sinal"]["n_com_dado"])


def html_previsao_conferencia(conf: dict | None, pac: dict | None, municipios: list[dict]) -> Html:
    """Seção "O que choveu em <mês>": a chuva do mês previsto vista por OUTRA fonte (Open-Meteo). Só formata o arquivo já
    validado (src/data/conferencia.py): nenhum número é calculado aqui, e a seção diz que é indicativa."""
    if not tem_conferencia(conf, pac):
        return Html("")
    nomes = {m["cod"]: m["nome"] for m in municipios}
    faltam = [r["cod_ibge"] for r in conf["municipios"] if r["cod_ibge"] not in nomes]
    if faltam:
        raise PaginaError(f"conferência de município que não está no site: {faltam[:5]}")
    g1, g2 = conf["grupos"]["com_sinal"], conf["grupos"]["sem_sinal"]
    base, fonte = conf["base_historica"], conf["fonte_conferencia"]
    mes = MESES[int(conf["mes_alvo"][5:7]) - 1]
    ano_ini, ano_fim = base["periodo"][:4], base["periodo"][-7:-3]
    lim = num_br(base["limiar_seco_percentil"], 0)
    resto = (f" Nos outros {g2['n_com_dado']}, foram {g2['n_secos']} ({_pct_br(g2['fracao_secos'])}%)." if g2["n_com_dado"] else "")
    fatos = [f'<div><strong>{g1["n_secos"]} de {g1["n_com_dado"]}</strong><span>municípios com sinal de seca tiveram um {mes} entre '
             f'os {lim}% mais secos de {ano_ini} a {ano_fim}.{resto}</span></div>']
    if g1["mediana_anomalia_mm"] is not None:
        outros = (f" Nos outros {g2['n']}, {_sinal_br(g2['mediana_anomalia_mm'])} mm." if g2["mediana_anomalia_mm"] is not None else "")
        fatos.append(f'<div><strong>{_sinal_br(g1["mediana_anomalia_mm"])} mm</strong><span>desvio mediano da chuva de {mes} em relação '
                     f'ao normal do mês nos municípios com sinal; {g1["n_abaixo_da_normal"]} de {g1["n"]} ficaram abaixo do normal.{outros}</span></div>')
    meses_ok = [(k, v) for k, v in sorted(conf["comparacao_fontes"].items()) if v["n"] and v["vies_medio_mm"] is not None]
    if meses_ok:
        nomes_m = lista_pt([MESES[int(k[5:7]) - 1] for k, _ in meses_ok], "e")
        vies = lista_pt([_sinal_br(v["vies_medio_mm"]) for _, v in meses_ok], "e")
        corrs = list(dict.fromkeys(num_br(v["correlacao"], 2) for _, v in meses_ok if v["correlacao"] is not None))
        fraca = (f" A correlação entre elas, município a município, é {lista_pt(corrs, 'e')}: por isso a leitura é por grupo."
                 if corrs else "")
        fatos.append(f'<div><strong>{vies} mm</strong><span>diferença média entre as duas fontes em {esc(nomes_m)} (Open-Meteo menos CHIRPS), '
                     f'os meses em que ambas existem.{esc(fraca)}</span></div>')
    linhas = []
    marcados = sorted((r for r in conf["municipios"] if r["sinal_modelo"]),
                      key=lambda r: (r["percentil"] if r["percentil"] is not None else 1e9, nomes[r["cod_ibge"]]))
    for r in marcados:
        chuva = r["chuva_mm"][conf["mes_alvo"]]
        nome = f'<th scope="row"><a href="monitor.html#m={esc(r["cod_ibge"])}">{esc(nomes[r["cod_ibge"]])}</a></th>'
        if chuva is None:
            linhas.append(f'<tr>{nome}<td colspan="6"><span class="sem-sinal">dias sem dado na outra fonte: sem conferência</span></td></tr>')
            continue
        pct = (f'{num_br(r["pct_normal"], 0)}%' if r["pct_normal"] is not None
               else '<span aria-hidden="true">—</span><span class="sr"> (normal baixa demais para uma porcentagem)</span>')
        pos = f'{num_br(100 - r["percentil"], 0)}%' if r["percentil"] is not None else "—"
        seco = r["percentil"] is not None and r["percentil"] <= base["limiar_seco_percentil"]
        selo = (f'<span class="selo atencao">{esc(mes.capitalize())} seco</span>' if seco
                else '<span class="sem-sinal">Fora da faixa seca</span>')
        linhas.append(f'<tr>{nome}<td>{num_br(chuva, 1)}</td><td>{num_br(r["normal_mm"], 1) if r["normal_mm"] is not None else "—"}</td>'
                      f'<td>{_sinal_br(r["anomalia_mm"]) if r["anomalia_mm"] is not None else "—"}</td><td>{pct}</td><td>{pos}</td><td>{selo}</td></tr>')
    avisos = "".join(f"<li>{esc(a)}</li>" for a in conf["avisos"])
    return Html(
        f'<section class="secao"><h2>O que choveu em {esc(mes)} nos municípios com sinal</h2>'
        f'<p class="sub"><b>Conferência indicativa, não o acerto oficial.</b> A chuva vem de outra fonte (reanálise da Open-Meteo), não do '
        f'CHIRPS que treinou o modelo. Mostra se os municípios que a IA marcou tiveram mesmo um {esc(mes)} seco.</p>'
        f'<div class="fatos">{"".join(fatos)}</div>'
        f'<div class="tabela-mun" style="margin-top:18px" tabindex="0" role="region" '
        f'aria-label="Chuva de {esc(mes)} nos municípios com sinal de seca. Use as setas para rolar."><table><thead><tr>'
        f'<th scope="col">Município</th><th scope="col">Choveu em {esc(mes)} (mm)</th><th scope="col">Normal do mês (mm)</th>'
        f'<th scope="col">Desvio (mm)</th><th scope="col">% do normal</th>'
        f'<th scope="col">Mais seco que (% dos {esc(mes)}s de {ano_ini} a {ano_fim})</th><th scope="col">Leitura</th>'
        f'</tr></thead><tbody>{"".join(linhas)}</tbody></table></div>'
        f'<div class="leitura" style="margin-top:14px"><ul>{avisos}</ul>'
        f'<p>Chuva diária de {esc(dia_br(fonte["periodo"][0]))} a {esc(dia_br(fonte["periodo"][1]))}, buscada em {esc(dia_br(conf["gerado_em"]))}, '
        f'no {esc(fonte["ponto"])}. <a href="{esc(fonte["url"])}" rel="noopener noreferrer">{esc(fonte["atribuicao"])}</a>. '
        f'Normal e histórico: {esc(base["fonte"])}, {esc(base["normal"])} ({esc(base["periodo"])}).</p></div></section>')


def html_metodo_conferencia(conf: dict | None, pac: dict | None) -> Html:
    """Parágrafo do Método sobre a segunda fonte de chuva: o que ela é, o que não é e por que não mexe nos dados."""
    if not tem_conferencia(conf, pac):
        return Html("")
    f, base = conf["fonte_conferencia"], conf["base_historica"]
    mes, mes_nome = mes_extenso(conf["mes_alvo"]), MESES[int(conf["mes_alvo"][5:7]) - 1]
    return Html(
        f'<section class="secao leitura"><h2>Segunda fonte de chuva, só para conferir</h2>'
        f'<p style="margin-top:12px">O CHIRPS, que alimenta o modelo, termina em {esc(mes_extenso(pac["origem"]))} e por isso ainda não mostra '
        f'como foi {esc(mes)}. O site tem duas camadas, e elas não se misturam:</p><ul style="margin-top:12px">'
        f'<li><b>Núcleo auditável.</b> A chuva do CHIRPS e a rede neural PyTorch treinada do zero. Tudo é gerado offline e conferido por '
        f'SHA-256 (veja o rodapé).</li>'
        f'<li><b>Conferência com outra fonte.</b> A chuva diária de {esc(dia_br(f["periodo"][0]))} a {esc(dia_br(f["periodo"][1]))} foi buscada '
        f'em {esc(dia_br(conf["gerado_em"]))} na API de histórico da Open-Meteo (reanálise), no {esc(f["ponto"])}, e guardada num arquivo com fonte, '
        f'licença e data. Ela não entra no treino, não prevê nada e não altera os dados nem o SHA-256.</li>'
        f'<li><b>Como ler.</b> A chuva de {esc(mes)} de cada município é posta ao lado dos {esc(mes_nome)}s de '
        f'{esc(base["periodo"][:4])} a {esc(base["periodo"][-7:-3])} do CHIRPS: o desvio em relação ao normal (a mediana do mês) e a posição '
        f'no histórico. Como as fontes são diferentes, a leitura é por grupo, municípios com sinal contra sem sinal.</li>'
        f'<li><b>Limite.</b> Não é o acerto oficial da previsão. A conferência com o CHIRPS fica para quando ele cobrir {esc(mes)}, e o '
        f'teste final continua fechado.</li></ul></section>')


def valores_previsao(payload: dict) -> dict:
    """Valores dos modelos que falam da previsão. Sem previsão no pacote, o site continua dizendo que ela está indisponível."""
    pac = payload.get("previsao")
    n_mun = pac["resumo"]["n_municipios"] if pac else 0
    if pac:
        alvo = mes_extenso(pac["alvo"])
        sobre = (f"Disponível só como experimento, na página Previsão: o SPI-3 e o SPI-6 de {alvo} em cada município, com margem de "
                 f"erro. Foi avaliada apenas na validação; o teste final ainda não foi aberto.")
        indisp = (f"Só como experimento: a página Previsão mostra o SPI de {alvo} com margem de erro. O teste final ainda não foi "
                  f"aberto e o alarme de seca pega só uma parte das secas reais.")
        if tem_conferencia(payload.get("conferencia"), pac):
            indisp += f" A página Previsão traz também uma conferência indicativa de {alvo} com outra fonte de chuva."
    else:
        sobre = "Indisponível. A rede neural de previsão está em calibração e só aparece depois de uma avaliação auditada."
        indisp = INDISPONIVEL_PREVISAO
    return {
        "previsao_alvo": mes_extenso(pac["alvo"]) if pac else "", "previsao_origem": mes_extenso(pac["origem"]) if pac else "",
        "previsao_n_mun": n_mun,
        "previsao_aviso": html_previsao_aviso(pac, payload), "previsao_fatos": html_previsao_fatos(pac),
        "previsao_tabela": html_previsao_tabela(pac, payload["municipios"]), "previsao_confianca": html_previsao_confianca(pac, payload),
        "previsao_legenda": html_previsao_legenda(pac),
        "previsao_conferencia": html_previsao_conferencia(payload.get("conferencia"), pac, payload["municipios"]),
        "metodo_conferencia": html_metodo_conferencia(payload.get("conferencia"), pac),
        "previsao_membros": pac["modelo"]["n_membros"] if pac else 0,
        "previsao_sobre": sobre, "previsao_metodo": indisp,
    }


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
        **valores_previsao(payload),
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
        "nav": html_nav(paginas, pagina.id),
        "aviso": html_aviso(payload["maturidade"], bool(payload.get("previsao"))),
        "conteudo": Html(conteudo), "scripts": Html(scripts),
    })
    return preencher(ler_modelo("_base.html"), v, f"{pagina.arquivo} (base)")
