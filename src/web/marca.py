"""Marca Piauí Resiliente em código: paleta e a gota (SVG) com a geometria da folha de construção.

Construção (folha "Construção" da marca):
- Gota: círculo de raio r com tangentes a 60° (triângulo equilátero circunscrito).
- Traço: r ÷ 4. O vazio entre o contorno e o fundo é de um traço.
- Nível: a linha do centro. Acima, o que falta (terracota). Abaixo, o que há (azul-petróleo).

Tudo é derivado de R_CIRCULO; o único número "solto" é o do viewBox, que sai das mesmas contas. Este módulo
não lê arquivo, não usa rede e não depende do resto do projeto.
"""
from __future__ import annotations

import math
import urllib.parse

# Paleta da marca (folha "Construção"). Não mudar sem refazer o contraste das classes do SPI.
PETROLEO = "#0F6E7A"      # o que há
TERRACOTA = "#B4472A"     # o que falta
TINTA = "#12262B"
CREME = "#F6F1E8"

R_CIRCULO = 64.0                      # r
TRACO = R_CIRCULO / 4                 # r ÷ 4
RAIO_EXTERNO = R_CIRCULO + TRACO / 2  # o contorno e o fundo chegam a r + traço/2 do centro
CX = RAIO_EXTERNO                     # centro, em x
CY = 2 * R_CIRCULO + TRACO / 2        # centro (o nível), em y: o topo do traço fica em y = 0
LARGURA = 2 * RAIO_EXTERNO            # 144
ALTURA = CY + RAIO_EXTERNO            # 208: do topo do contorno à base do fundo
NIVEL_CONTORNO = CY - TRACO / 2       # o contorno termina meio traço acima do nível
NIVEL_FUNDO = CY + TRACO / 2          # o fundo começa meio traço abaixo: vazio = um traço
ID_SIMBOLO = "pr-gota"


def _n(v: float) -> str:
    s = f"{v:.3f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def caminho_contorno() -> str:
    """Eixo do traço: arco do nível até a tangente, reta até o vértice (ângulo de 60°) e o lado oposto."""
    r = R_CIRCULO
    tx, ty = r * math.sin(math.radians(60)), r * math.cos(math.radians(60))    # ponto de tangência
    return (f"M{_n(CX - r)} {_n(CY)}A{_n(r)} {_n(r)} 0 0 1 {_n(CX - tx)} {_n(CY - ty)}"
            f"L{_n(CX)} {_n(CY - 2 * r)}L{_n(CX + tx)} {_n(CY - ty)}"
            f"A{_n(r)} {_n(r)} 0 0 1 {_n(CX + r)} {_n(CY)}")


def caminho_fundo() -> str:
    """Calota do disco de raio r + traço/2 abaixo de NIVEL_FUNDO."""
    meia_corda = math.sqrt(RAIO_EXTERNO ** 2 - (TRACO / 2) ** 2)
    return (f"M{_n(CX - meia_corda)} {_n(NIVEL_FUNDO)}A{_n(RAIO_EXTERNO)} {_n(RAIO_EXTERNO)} 0 0 0 "
            f"{_n(CX + meia_corda)} {_n(NIVEL_FUNDO)}Z")


def _miolo(cor_falta: str, cor_ha: str, id_corte: str) -> str:
    return (f'<clipPath id="{id_corte}"><rect x="0" y="0" width="{_n(LARGURA)}" height="{_n(NIVEL_CONTORNO)}"/></clipPath>'
            f'<path d="{caminho_contorno()}" fill="none" stroke="{cor_falta}" stroke-width="{_n(TRACO)}" '
            f'stroke-linejoin="round" clip-path="url(#{id_corte})"/>'
            f'<path d="{caminho_fundo()}" fill="{cor_ha}"/>')


def simbolo() -> str:
    """`<symbol>` para uma vez por página: `<use href="#pr-gota">`. As cores vêm das variáveis CSS
    `--gota-falta` e `--gota-ha`, que mudam com o tema."""
    return (f'<symbol id="{ID_SIMBOLO}" viewBox="0 0 {_n(LARGURA)} {_n(ALTURA)}">'
            f'{_miolo("var(--gota-falta)", "var(--gota-ha)", "pr-gota-corte")}</symbol>')


def svg_isolado(cor_falta: str = TERRACOTA, cor_ha: str = PETROLEO) -> str:
    """A gota como arquivo SVG completo (favicon, impressão, apresentação)."""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_n(LARGURA)} {_n(ALTURA)}" '
            f'width="{_n(LARGURA)}" height="{_n(ALTURA)}" role="img" aria-label="Piauí Resiliente">'
            f'{_miolo(cor_falta, cor_ha, "corte")}</svg>')


def favicon_data_uri() -> str:
    return "data:image/svg+xml," + urllib.parse.quote(svg_isolado(), safe="/:=,() ")
