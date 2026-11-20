"""Um ponto (lat, lon) por município, calculado da malha. Módulo puro: só geometria, sem rede, sem ML, sem web.

Usado por dois lados que precisam da MESMA coordenada para falar do mesmo lugar:
- `conferencia_outubro` (script do PC da equipe, busca a chuva do mês na Open-Meteo);
- `build_site` (grava a coordenada no pacote de dados, para o painel ao vivo do Monitor).
"""
from __future__ import annotations

from dataclasses import dataclass

from shapely.geometry import shape

LAT_PI, LON_PI = (-11.0, -2.5), (-46.2, -40.2)        # caixa do Piauí: pega coordenada trocada ou fora do estado


class PontosError(ValueError):
    """Malha ausente, feição inválida, código repetido ou ponto fora do Piauí."""


@dataclass(frozen=True)
class Ponto:
    cod_ibge: str
    nome: str
    lat: float
    lon: float


def pontos_da_malha(malha: dict) -> list[Ponto]:
    """Um ponto por município: o centroide; se ele cair fora do polígono (forma côncava), um ponto interno.

    Ordenado por `cod_ibge`. Coordenadas arredondadas a 4 casas (~11 m), o bastante para uma grade de ~25 km.
    """
    if not isinstance(malha, dict) or not isinstance(malha.get("features"), list) or not malha["features"]:
        raise PontosError("malha sem 'features'")
    pontos, vistos = [], set()
    for f in malha["features"]:
        try:
            cod = str(f["properties"]["cod_ibge"])
            nome = str(f["properties"].get("nome", cod))
            geom = shape(f["geometry"])
        except (KeyError, TypeError, ValueError, AttributeError) as e:
            raise PontosError(f"feição da malha inválida: {e!r}") from e
        if geom.is_empty or not geom.is_valid:
            raise PontosError(f"geometria vazia ou inválida em {cod}")
        c = geom.centroid
        if not geom.contains(c):
            c = geom.representative_point()
        if not (LAT_PI[0] <= c.y <= LAT_PI[1] and LON_PI[0] <= c.x <= LON_PI[1]):
            raise PontosError(f"{cod}: ponto (lat {c.y:.3f}, lon {c.x:.3f}) fora do Piauí; confira a ordem lon/lat")
        if cod in vistos:
            raise PontosError(f"cod_ibge repetido na malha: {cod}")
        vistos.add(cod)
        pontos.append(Ponto(cod, nome, round(c.y, 4), round(c.x, 4)))
    return sorted(pontos, key=lambda p: p.cod_ibge)
