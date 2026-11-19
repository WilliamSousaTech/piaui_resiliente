"""Testes de src/web/marca.py: a gota segue a folha de construção (r, traço r÷4, tangentes a 60°, nível no centro)."""
import math
import re
import urllib.parse
import xml.etree.ElementTree as ET

import pytest

from src.web import marca
from util_cor import contraste

NS = "{http://www.w3.org/2000/svg}"


def numeros(d: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", d)]


def test_proporcoes_da_folha_de_construcao():
    assert marca.TRACO == marca.R_CIRCULO / 4
    assert marca.RAIO_EXTERNO == marca.R_CIRCULO + marca.TRACO / 2
    assert marca.NIVEL_FUNDO - marca.NIVEL_CONTORNO == marca.TRACO              # o vazio entre contorno e fundo é um traço
    assert (marca.NIVEL_FUNDO + marca.NIVEL_CONTORNO) / 2 == marca.CY           # o nível é a linha do centro
    assert marca.CY - 2 * marca.R_CIRCULO - marca.TRACO / 2 == 0                # a junta redonda do vértice encosta em y = 0
    assert marca.LARGURA == 2 * marca.RAIO_EXTERNO == 144 and marca.ALTURA == marca.CY + marca.RAIO_EXTERNO == 208


def test_contorno_tem_tangentes_a_60_graus_que_tocam_o_circulo():
    (x0, y0, rx, ry, _, _, _, x1, y1, xa, ya, x2, y2, rx2, ry2, _, _, _, x3, y3) = numeros(marca.caminho_contorno())
    r, cx, cy = marca.R_CIRCULO, marca.CX, marca.CY
    assert (x0, y0, x3, y3) == (cx - r, cy, cx + r, cy)                      # o arco parte e chega no nível
    assert rx == ry == rx2 == ry2 == r
    for tx, ty in ((x1, y1), (x2, y2)):
        assert math.hypot(tx - cx, ty - cy) == pytest.approx(r, abs=0.01)    # ponto de tangência está no círculo
        # a reta até o vértice é perpendicular ao raio nesse ponto: é mesmo uma tangente
        assert ((xa - tx) * (tx - cx) + (ya - ty) * (ty - cy)) == pytest.approx(0, abs=0.5)
        assert math.hypot(xa - tx, ya - ty) == pytest.approx(r * math.sqrt(3), abs=0.01)
    assert (xa, ya) == (cx, cy - 2 * r)                                      # vértice a 2r do centro (triângulo equilátero)
    v1, v2 = (x1 - xa, y1 - ya), (x2 - xa, y2 - ya)
    ang = math.degrees(math.acos((v1[0] * v2[0] + v1[1] * v2[1]) / (math.hypot(*v1) * math.hypot(*v2))))
    assert ang == pytest.approx(60, abs=0.05)


def test_fundo_e_a_calota_do_disco_externo_abaixo_do_nivel():
    x0, y0, rx, ry, _, _, _, x1, y1 = numeros(marca.caminho_fundo())
    assert y0 == y1 == marca.NIVEL_FUNDO and rx == ry == marca.RAIO_EXTERNO
    for x in (x0, x1):
        assert math.hypot(x - marca.CX, y0 - marca.CY) == pytest.approx(marca.RAIO_EXTERNO, abs=0.01)
    assert x0 + x1 == pytest.approx(2 * marca.CX, abs=0.01) and x0 < x1


def test_svg_isolado_e_xml_valido_com_as_cores_da_marca_e_sem_recurso_externo():
    svg = marca.svg_isolado()
    raiz = ET.fromstring(svg)
    assert raiz.tag == NS + "svg" and raiz.get("viewBox") == "0 0 144 208" and raiz.get("role") == "img"
    assert marca.TERRACOTA in svg and marca.PETROLEO in svg
    assert "http" not in svg.replace("http://www.w3.org/2000/svg", "")
    clip = raiz.find(f".//{NS}clipPath/{NS}rect")
    assert float(clip.get("height")) == marca.NIVEL_CONTORNO                  # só o que está acima do nível leva o contorno
    contorno = next(p for p in raiz.iter(NS + "path") if p.get("fill") == "none")
    assert contorno.get("stroke") == marca.TERRACOTA and float(contorno.get("stroke-width")) == marca.TRACO
    assert contorno.get("stroke-linejoin") == "round"
    fundo = next(p for p in raiz.iter(NS + "path") if p.get("fill") == marca.PETROLEO)
    assert fundo.get("stroke") is None


def test_svg_isolado_aceita_outras_cores():
    svg = marca.svg_isolado("#111111", "#222222")
    assert "#111111" in svg and "#222222" in svg and marca.TERRACOTA not in svg


def test_simbolo_usa_variaveis_do_tema_e_id_estavel():
    s = marca.simbolo()
    ET.fromstring(f'<svg xmlns="http://www.w3.org/2000/svg">{s}</svg>')
    assert f'id="{marca.ID_SIMBOLO}"' == 'id="pr-gota"' and "var(--gota-falta)" in s and "var(--gota-ha)" in s
    assert marca.TERRACOTA not in s and marca.PETROLEO not in s              # a cor vem do tema, não fica fixa no símbolo


def test_favicon_data_uri_e_seguro_para_atributo_html():
    uri = marca.favicon_data_uri()
    assert uri.startswith("data:image/svg+xml,")
    corpo = uri.split(",", 1)[1]
    assert not set('<>"#') & set(corpo)
    assert urllib.parse.unquote(corpo) == marca.svg_isolado()


@pytest.mark.parametrize("v,esperado", [(64.0, "64"), (7.5, "7.5"), (-0.0001, "0"), (0.0, "0"), (1 / 3, "0.333"), (-12.0, "-12")])
def test_numero_curto_sem_zeros_a_toa(v, esperado):
    assert marca._n(v) == esperado


def test_paleta_e_texto_legivel_sobre_o_creme():
    for cor in (marca.TINTA, marca.PETROLEO, marca.TERRACOTA):
        assert re.fullmatch(r"#[0-9A-F]{6}", cor)
        assert contraste(cor, marca.CREME) >= 4.5, cor
    assert (marca.PETROLEO, marca.TERRACOTA, marca.TINTA, marca.CREME) == ("#0F6E7A", "#B4472A", "#12262B", "#F6F1E8")
