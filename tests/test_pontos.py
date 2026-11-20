"""Pontos por município (src/data/pontos.py): centro dentro do polígono, ordem, recusa de malha ruim."""
import ast
from pathlib import Path

import pytest
from shapely.geometry import Point, shape

from src.data import conferencia_outubro as co
from src.data import pontos as pt

RAIZ = Path(__file__).resolve().parents[1]


def _poligono(coords):
    return {"type": "Polygon", "coordinates": [coords]}


QUADRADO = _poligono([[-42, -6], [-41, -6], [-41, -7], [-42, -7], [-42, -6]])          # lon, lat
FORMA_C = _poligono([[-42, -5], [-40.5, -5], [-40.5, -5.2], [-41.8, -5.2], [-41.8, -6.8], [-40.5, -6.8],
                     [-40.5, -7], [-42, -7], [-42, -5]])


def _feicao(cod, geom, nome="X"):
    return {"type": "Feature", "properties": {"cod_ibge": cod, "nome": nome}, "geometry": geom}


def test_ordenado_por_codigo_e_dentro_do_estado():
    malha = {"features": [_feicao("2200002", QUADRADO, "B"), _feicao("2200001", QUADRADO, "A")]}
    p = pt.pontos_da_malha(malha)
    assert [x.cod_ibge for x in p] == ["2200001", "2200002"]
    assert [x.nome for x in p] == ["A", "B"]
    assert (p[0].lat, p[0].lon) == (-6.5, -41.5)
    assert all(pt.LAT_PI[0] <= x.lat <= pt.LAT_PI[1] and pt.LON_PI[0] <= x.lon <= pt.LON_PI[1] for x in p)


def test_centroide_fora_do_poligono_usa_ponto_interno():
    p = pt.pontos_da_malha({"features": [_feicao("2200001", FORMA_C)]})[0]
    assert not shape(FORMA_C).contains(shape(FORMA_C).centroid)       # a forma realmente "engana" o centroide
    assert shape(FORMA_C).contains(Point(p.lon, p.lat))


def test_coordenadas_arredondadas_a_4_casas():
    g = _poligono([[-42.123456, -6.123456], [-41.0, -6.123456], [-41.0, -7.0], [-42.123456, -7.0], [-42.123456, -6.123456]])
    p = pt.pontos_da_malha({"features": [_feicao("1", g)]})[0]
    assert p.lat == round(p.lat, 4) and p.lon == round(p.lon, 4)


@pytest.mark.parametrize("malha, trecho", [
    ({}, "sem 'features'"),
    ([], "sem 'features'"),
    ({"features": []}, "sem 'features'"),
    ({"features": [_feicao("1", QUADRADO), _feicao("1", QUADRADO)]}, "repetido"),
    ({"features": [{"properties": {}, "geometry": QUADRADO}]}, "inválida"),
    ({"features": [_feicao("1", _poligono([[-6, -42], [-6, -41], [-7, -41], [-7, -42], [-6, -42]]))]}, "fora do Piauí"),
    ({"features": [_feicao("1", _poligono([[-42, -6], [-41, -7], [-41, -6], [-42, -7], [-42, -6]]))]}, "inválida"),
])
def test_recusa_malha_ruim(malha, trecho):
    with pytest.raises(pt.PontosError, match=trecho):
        pt.pontos_da_malha(malha)


def test_conferencia_continua_levantando_o_proprio_erro_e_reexporta_ponto():
    assert co.Ponto is pt.Ponto
    with pytest.raises(co.ConferenciaError, match="repetido"):
        co.pontos_da_malha({"features": [_feicao("1", QUADRADO), _feicao("1", QUADRADO)]})


def test_modulo_e_puro():
    """Sem rede, ML, web nem SQL: o build do site importa este módulo."""
    src = (RAIZ / "src" / "data" / "pontos.py").read_text(encoding="utf-8")
    importados = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            importados |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            importados.add(n.module)
    proibidos = {"fastapi", "flask", "sqlite3", "sqlalchemy", "torch", "sklearn", "requests", "httpx", "urllib",
                 "urllib3", "http", "socket", "ssl"}
    assert not [m for m in importados if m.split(".")[0] in proibidos], importados


def test_malha_real_do_projeto_gera_216_pontos_distintos_dentro_do_poligono():
    import json
    caminho = RAIZ / "data" / "reference" / "malha_pi_2022_gee.geojson"
    if not caminho.exists():
        pytest.skip("malha real não está no checkout")
    malha = json.loads(caminho.read_text(encoding="utf-8"))
    p = pt.pontos_da_malha(malha)
    por_cod = {f["properties"]["cod_ibge"]: shape(f["geometry"]) for f in malha["features"]}
    assert len(p) == len(por_cod) and len({(x.lat, x.lon) for x in p}) == len(p)
    assert all(por_cod[x.cod_ibge].buffer(1e-3).contains(Point(x.lon, x.lat)) for x in p)
