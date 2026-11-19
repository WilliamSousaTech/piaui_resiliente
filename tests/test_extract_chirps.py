"""Testes de src/data/extract_chirps.py (Item 2c). Sem internet e sem GEE.

O `ee` é injetado (`rodar(cfg, ee=...)`). O falso `EEFalso`:
  - rejeita banda inexistente (KeyError) e exige a máscara (`selfMask`) em `dias_validos`
    na redução de mínimo (D4: mínimo só entre pixels com dado);
  - devolve valores DISTINTOS por município e por banda, para que troca de coluna apareça;
  - registra as chamadas de `reduceRegions`, os limiares passados a `gte` e as geometrias.
Dados sintéticos existem só aqui (testes); nada disto entra em data/ nem na interface.
"""
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import chirps_raw as raw
from src.data import extract_chirps as m
from src.data.config import PendenciaError, load_config

PARAMS_REAL = Path(__file__).resolve().parents[1] / "params.yaml"
LIMIAR = 2.5                      # diferente do 1.0 do params real: pega limiar fixo no código
COD = ["2200053", "2200251", "2200277", "2205706", "2207801"]   # ordem ordenada (rodar ordena)
IDX = {c: i for i, c in enumerate(COD)}
MAX_LINHAS = 500                  # 5 municípios x 100 meses por lote -> 549 meses = 6 lotes (o último com 49)
MESES_POR_LOTE = 100
N_MESES = 549                     # 1981-01 a 2026-09
N_LOTES = 6


# ------------------------------------------------------------------ malha sintética
def quad(x, y, d):
    return {"type": "Polygon", "coordinates": [[[x, y], [x + d, y], [x + d, y + d], [x, y + d], [x, y]]]}


def feat(cod, geom):
    return {"type": "Feature", "properties": {"cod_ibge": cod}, "geometry": geom}


GEO = {"type": "FeatureCollection", "features": [          # fora de ordem de propósito
    feat("2207801", quad(-41.0, -8.0, 0.50)),               # polo (Paulistana)
    feat("2205706", quad(-41.7, -2.9, 0.30)),               # Luís Correia (litoral)
    feat("2200053", quad(-42.0, -6.0, 0.05)),               # menor caixa envolvente
    feat("2200251", quad(-42.5, -6.5, 0.40)),
    feat("2200277", quad(-43.0, -7.0, 0.35)),
]}


# ------------------------------------------------------------------ ee falso
class EEErro(Exception):
    """Faz o papel de ee.EEException."""


class Num:
    def __init__(self, v):
        self.v = v

    def getInfo(self):
        return self.v


class Reducer:
    def __init__(self, tipo):
        self.tipo = tipo


class Proj:
    def crs(self):
        return Num("EPSG:4326")

    def nominalScale(self):
        return Num(5565.97)


def dias_do_mes(mes: str) -> int:
    return pd.Period(mes, freq="M").days_in_month


class Col:
    def __init__(self, ee, ini=None, fim=None):
        self.ee, self.ini, self.fim = ee, ini, fim

    def select(self, banda):
        return Col(self.ee, self.ini, self.fim)

    def filterDate(self, ini, fim):
        return Col(self.ee, ini, fim)

    def map(self, f):
        f(Img(self.ee, ["precipitation"]))     # executa o lambda: registra o limiar passado a gte
        return Col(self.ee, self.ini, self.fim)

    def first(self):
        return Img(self.ee, ["precipitation"])

    def sum(self):
        return Img(self.ee, ["precipitation"])

    max = min = count = sum

    def size(self):
        if not self.ini:
            return Num(self.ee.total_imagens)
        mes = self.ini[:7]
        return Num(self.ee.n_imagens.get(mes, dias_do_mes(mes)))

    def aggregate_min(self, prop):
        return Num(pd.Timestamp("1981-01-01").value // 10**6)

    def aggregate_max(self, prop):
        return Num(pd.Timestamp("2026-09-30").value // 10**6)


class Img:
    def __init__(self, ee, bandas, mascaradas=()):
        self.ee, self.bandas, self.mascaradas = ee, list(bandas), set(mascaradas)

    def select(self, b):
        b = b if isinstance(b, list) else [b]
        faltam = [x for x in b if x not in self.bandas]
        if faltam:
            raise KeyError(f"banda inexistente: {faltam} (há {self.bandas})")
        return Img(self.ee, b, self.mascaradas & set(b))

    def addBands(self, outra):
        return Img(self.ee, self.bandas + outra.bandas, self.mascaradas | outra.mascaradas)

    def rename(self, nome):
        return Img(self.ee, [nome], {nome} if self.mascaradas else ())

    def selfMask(self):
        return Img(self.ee, self.bandas, self.bandas)

    def gte(self, v):
        self.ee.limiares.append(v)
        return Img(self.ee, self.bandas)

    def setDefaultProjection(self, proj):
        return Img(self.ee, self.bandas, self.mascaradas)

    def projection(self):
        return Proj()

    def bandNames(self):
        return Num(["precipitation"])

    def propertyNames(self):
        return Num(["system:index", "system:time_start"])

    def reduceRegions(self, collection, reducer, **kw):
        ee = self.ee
        ee.chamadas.append((reducer.tipo, tuple(self.bandas)))
        if ee.falhar_apos is not None and len(ee.chamadas) > ee.falhar_apos:
            raise EEErro("falha permanente simulada")
        if ee.travar > 0:                                # chamada HTTP presa, sem erro e sem resposta
            ee.travar -= 1
            ee.liberar.wait(30)
        if ee.falhas > 0:
            ee.falhas -= 1
            raise EEErro("Too many concurrent aggregations")
        return Resultado(ee, collection.cods, self.bandas, reducer.tipo, self.mascaradas)


MEDIA = {"soma": lambda i: 100.0 + i, "max_dia": lambda i: 20.0 + i, "min_dia": lambda i: 0.0,
         "dias_chuva": lambda i: 5.0 + i / 2, "dias_validos": lambda i: 28.0}


class Resultado:
    def __init__(self, ee, cods, bandas, tipo, mascaradas):
        self.ee, self.cods, self.bandas, self.tipo, self.mascaradas = ee, cods, bandas, tipo, mascaradas

    def _valores(self, i):
        if self.tipo == "mean":
            return {b: MEDIA[b](i) for b in self.bandas}
        if self.tipo == "min":
            assert "dias_validos" in self.mascaradas, "dias_validos sem selfMask (D4)"
            return {b: ({"dias_validos": 27.0 if i == 2 else 28.0, "um": 1.0})[b] for b in self.bandas}
        return {b: ({"soma": 3.0 if i == 1 else 4.0, "um": 4.0})[b] for b in self.bandas}

    def getInfo(self):
        feats = []
        for cod in self.cods:
            if cod in self.ee.ausentes:                 # GEE não devolve a feature
                continue
            p = {"cod_ibge": cod}
            if cod not in self.ee.sem_dado:             # GEE devolve a feature sem as propriedades
                p.update(self._valores(IDX[cod]))
            feats.append({"properties": p})
        return {"features": feats}


class EEFalso:
    __version__ = "falso-1.0"
    EEException = EEErro

    def __init__(self, *, falhas=0, falhar_apos=None, sem_dado=(), ausentes=(), n_imagens=None,
                 total_imagens=12345, travar=0):
        self.falhas, self.falhar_apos = falhas, falhar_apos
        self.travar, self.liberar = travar, threading.Event()   # `travar` chamadas ficam presas até `liberar`
        self.sem_dado, self.ausentes = set(sem_dado), set(ausentes)
        self.n_imagens, self.total_imagens = dict(n_imagens or {}), total_imagens
        self.chamadas, self.limiares, self.colecoes, self.geometrias = [], [], [], []
        self.Reducer = SimpleNamespace(mean=lambda: Reducer("mean"), min=lambda: Reducer("min"),
                                       count=lambda: Reducer("count"))
        self.Image = SimpleNamespace(constant=lambda v: Img(self, ["constant"]))
        self.Feature = lambda geom, props: SimpleNamespace(cod=props["cod_ibge"])
        self.FeatureCollection = lambda feats: SimpleNamespace(cods=[f.cod for f in feats])
        self.Date = lambda ms: SimpleNamespace(
            format=lambda fmt: Num(pd.Timestamp(ms.getInfo(), unit="ms").strftime("%Y-%m-%d")))

    def ImageCollection(self, gee_id):
        self.colecoes.append(gee_id)
        return Col(self)

    def Geometry(self, geom, proj=None, geodesic=None):
        self.geometrias.append((geom, proj, geodesic))
        return SimpleNamespace(geom=geom)


# ------------------------------------------------------------------ fábricas
def cfg_tmp(tmp_path: Path, *, ajustes: dict | None = None):
    """params real + lote pequeno + limiar 2.5; `ajustes` = {("secao","chave",...): valor}."""
    d = yaml.safe_load(PARAMS_REAL.read_text(encoding="utf-8"))
    d["gee"]["max_linhas_por_lote"] = MAX_LINHAS
    d["fontes"]["chirps"]["limiar_dia_chuva_mm"] = LIMIAR
    for caminho, valor in (ajustes or {}).items():
        alvo = d
        for k in caminho[:-1]:
            alvo = alvo[k]
        alvo[caminho[-1]] = valor
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(d, allow_unicode=True, sort_keys=False),
                                          encoding="utf-8")
    return load_config(tmp_path / "params.yaml")


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ler_geojson_gee", lambda c: GEO)
    return cfg_tmp(tmp_path)


def executar(cfg, ee, **kw):
    saidas, esperas = [], []
    m.rodar(cfg, ee, dormir=esperas.append,
            saida=lambda *a, sep=" ": saidas.append(sep.join(map(str, a))), **kw)
    return saidas, esperas


def parquets(cfg):
    pasta = raw.diretorio_raw(cfg)
    return sorted(pasta.glob("*.parquet")) if pasta.exists() else []


def selo_de(parquet: Path) -> dict:
    return json.loads(parquet.with_name(parquet.name + ".sha256.json").read_text(encoding="utf-8"))


def esperado(cod):
    i = IDX[cod]
    return {"precip_mm": 100.0 + i, "precip_max_dia_mm": 20.0 + i, "dias_chuva": 5.0 + i / 2,
            "dias_validos": 27 if i == 2 else 28, "pixels_validos_frac": 0.75 if i == 1 else 1.0}


# ================================================================== helpers puros
@pytest.mark.parametrize("mes, esperado_", [
    ("2020-03", ("2020-03-01", "2020-04-01")),
    ("2020-12", ("2020-12-01", "2021-01-01")),     # vira o ano
    ("2020-02", ("2020-02-01", "2020-03-01")),     # bissexto: o fim é exclusivo, sem dia 29/30 fixo
    ("1981-01", ("1981-01-01", "1981-02-01")),
])
def test_intervalo_mes(mes, esperado_):
    assert m.intervalo_mes(mes) == esperado_


def test_area_bbox_poligono():
    assert m.area_bbox(quad(0, 0, 2)) == pytest.approx(4.0)
    assert m.area_bbox({"type": "Polygon", "coordinates": [[[0, 0], [2, 0], [2, 3], [0, 3], [0, 0]]]}) == 6.0


def test_area_bbox_multipoligono_com_buraco_usa_todos_os_pontos():
    geom = {"type": "MultiPolygon", "coordinates": [
        [[[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]], [[1, 1], [2, 1], [2, 1.5], [1, 1]]],
        [[[10, 5], [11, 5], [11, 6], [10, 5]]]]}
    assert m.area_bbox(geom) == pytest.approx(11 * 6)


def test_escolher_amostra_tres_motivos():
    assert m.escolher_amostra(GEO, "2207801") == {
        "2207801": "polo", "2200053": "menor bbox", "2205706": "litoral"}


def test_escolher_amostra_polo_que_e_o_menor_nao_duplica():
    geo = {"features": [feat("2207801", quad(0, 0, 0.01)), feat("2200251", quad(0, 0, 1.0)),
                        feat("2205706", quad(5, 5, 0.3))]}
    assert m.escolher_amostra(geo, "2207801") == {"2207801": "polo", "2205706": "litoral"}


def test_escolher_amostra_luis_correia_menor_fica_como_menor_bbox():
    geo = {"features": [feat("2205706", quad(0, 0, 0.01)), feat("2207801", quad(0, 0, 1.0))]}
    assert m.escolher_amostra(geo, "2207801") == {"2207801": "polo", "2205706": "menor bbox"}


def test_escolher_amostra_sem_polo_nem_litoral_no_escopo():
    geo = {"features": [feat("2200251", quad(0, 0, 1.0)), feat("2200277", quad(0, 0, 0.1))]}
    assert m.escolher_amostra(geo, "2207801") == {"2200277": "menor bbox"}


# ================================================================== montar_linhas
MES = pd.Period("2020-02", "M")                       # bissexto: 29 dias
MEDIA_A = {"A": {"soma": 120.0, "max_dia": 30.0, "dias_chuva": 8.5}}
MINIMO_A = {"A": {"dias_validos": 27.6, "um": 1.0}}
CONTAGEM_A = {"A": {"soma": 3.0, "um": 4.0}}


def uma(media=MEDIA_A, minimo=MINIMO_A, contagem=CONTAGEM_A):
    linhas = m.montar_linhas(MES, ["A"], media, minimo, contagem)
    assert len(linhas) == 1
    return linhas[0]


def test_montar_linhas_caso_normal():
    l = uma()
    assert l["cod_ibge"] == "A" and l["mes"] == pd.Timestamp("2020-02-01")
    assert (l["precip_mm"], l["precip_max_dia_mm"], l["dias_chuva"]) == (120.0, 30.0, 8.5)
    assert l["dias_validos"] == 28 and isinstance(l["dias_validos"], int)    # 27.6 arredonda
    assert l["dias_no_mes"] == 29
    assert l["pixels_validos_frac"] == pytest.approx(0.75)


def test_montar_linhas_zero_real_nao_vira_nan():
    media = {"A": {"soma": 0.0, "max_dia": 0.0, "dias_chuva": 0.0}}
    l = uma(media=media, contagem={"A": {"soma": 4.0, "um": 4.0}})
    assert l["precip_mm"] == 0.0 and l["precip_max_dia_mm"] == 0.0 and l["dias_chuva"] == 0.0
    assert not any(np.isnan(l[k]) for k in ("precip_mm", "precip_max_dia_mm", "dias_chuva"))


def test_montar_linhas_fracao_limitada_a_um():
    assert uma(contagem={"A": {"soma": 5.0, "um": 4.0}})["pixels_validos_frac"] == 1.0


@pytest.mark.parametrize("contagem", [
    {"A": {"soma": 0.0, "um": 4.0}},          # nenhum pixel com dado
    {"A": {"soma": None, "um": 4.0}},         # GEE devolveu null
    {"A": {}},                                # propriedades ausentes
    {},                                       # município ausente do retorno
])
def test_montar_linhas_sem_pixel_valido_vira_nan_triplo_e_cobertura_zero(contagem):
    l = uma(contagem=contagem)
    assert all(np.isnan(l[k]) for k in ("precip_mm", "precip_max_dia_mm", "dias_chuva"))
    assert l["pixels_validos_frac"] == 0.0 and l["dias_validos"] == 0


def test_montar_linhas_sem_pixel_valido_ignora_minimo_residual():
    l = uma(contagem={"A": {"soma": 0.0, "um": 4.0}}, minimo={"A": {"dias_validos": 31.0, "um": 1.0}})
    assert l["dias_validos"] == 0


def test_montar_linhas_total_zero_da_fracao_zero():
    l = uma(contagem={"A": {"soma": 0.0, "um": 0.0}})
    assert l["pixels_validos_frac"] == 0.0


@pytest.mark.parametrize("chave", ["soma", "max_dia", "dias_chuva"])
def test_montar_linhas_propriedade_nula_zera_o_trio_inteiro(chave):
    media = {"A": {"soma": 10.0, "max_dia": 5.0, "dias_chuva": 3.0, chave: None}}
    l = uma(media=media)
    assert all(np.isnan(l[k]) for k in ("precip_mm", "precip_max_dia_mm", "dias_chuva"))
    assert l["pixels_validos_frac"] == pytest.approx(0.75)


def test_montar_linhas_preserva_ordem_e_um_registro_por_municipio():
    cods = ["B", "A", "C"]
    media = {c: {"soma": 10.0 + i, "max_dia": 5.0, "dias_chuva": 3.0} for i, c in enumerate(cods)}
    linhas = m.montar_linhas(MES, cods, media, {}, {c: {"soma": 4.0, "um": 4.0} for c in cods})
    assert [l["cod_ibge"] for l in linhas] == cods
    assert [l["precip_mm"] for l in linhas] == [10.0, 11.0, 12.0]


def test_montar_linhas_passa_no_validar_do_raw():
    cods = ["2200053", "2200251"]
    media = {c: {"soma": 50.0, "max_dia": 20.0, "dias_chuva": 4.0} for c in cods}
    minimo = {c: {"dias_validos": 29.0, "um": 1.0} for c in cods}
    contagem = {c: {"soma": 4.0, "um": 4.0} for c in cods}
    df = raw.padronizar(pd.DataFrame(m.montar_linhas(MES, cods, media, minimo, contagem),
                                     columns=raw.COLUNAS))
    raw.validar(df, cods, [MES])


# ================================================================== _com_retry
def test_com_retry_sucesso_direto_nao_dorme():
    esperas = []
    assert m._com_retry(lambda: 42, EEFalso(), esperas.append) == 42
    assert esperas == []


@pytest.mark.parametrize("exc", [EEErro("transitório"), OSError("rede caiu")])
def test_com_retry_recupera_apos_duas_falhas_com_espera_dobrando(exc):
    tentativas = []

    def fn():
        tentativas.append(1)
        if len(tentativas) < 3:
            raise exc
        return "ok"

    esperas = []
    assert m._com_retry(fn, EEFalso(), esperas.append) == "ok"
    assert len(tentativas) == 3
    assert esperas == [m.ESPERA_S, m.ESPERA_S * 2] and esperas[0] < esperas[1]


def test_com_retry_esgota_tentativas_e_encadeia_o_erro_original():
    tentativas, esperas = [], []

    def fn():
        tentativas.append(1)
        raise EEErro("sempre falha")

    with pytest.raises(m.ExtracaoError, match="3 tentativas") as e:
        m._com_retry(fn, EEFalso(), esperas.append)
    assert len(tentativas) == 3
    assert esperas == [m.ESPERA_S, m.ESPERA_S * 2]          # não dorme depois da última
    assert isinstance(e.value.__cause__, EEErro)


def test_chamar_com_limite_devolve_o_valor():
    assert m._chamar_com_limite(lambda: 7, 5.0) == 7


@pytest.mark.parametrize("limite", [None, 0])
def test_chamar_com_limite_sem_limite_roda_na_propria_thread(limite):
    assert m._chamar_com_limite(threading.get_ident, limite) == threading.get_ident()


def test_chamar_com_limite_com_limite_roda_em_thread_daemon():
    info = m._chamar_com_limite(lambda: (threading.get_ident(), threading.current_thread().daemon), 5.0)
    assert info[0] != threading.get_ident() and info[1] is True


def test_chamar_com_limite_repassa_a_excecao_original_intacta():
    erro = EEErro("falha do gee")

    def fn():
        raise erro

    with pytest.raises(EEErro) as e:
        m._chamar_com_limite(fn, 5.0)
    assert e.value is erro


def test_chamar_com_limite_levanta_timeout_quando_a_chamada_prende():
    solta = threading.Event()
    try:
        with pytest.raises(TimeoutError, match="sem resposta"):
            m._chamar_com_limite(lambda: solta.wait(30), 0.05)
    finally:
        solta.set()


def test_timeout_e_um_oserror_logo_o_retry_repete():
    assert issubclass(TimeoutError, OSError)


def test_com_retry_repete_a_chamada_presa_e_recupera():
    solta, chamadas, esperas = threading.Event(), [], []

    def fn():
        chamadas.append(1)
        if len(chamadas) == 1:
            solta.wait(30)                      # 1ª tentativa trava
        return "ok"

    try:
        assert m._com_retry(fn, EEFalso(), esperas.append, limite_s=0.05) == "ok"
    finally:
        solta.set()
    assert len(chamadas) == 2 and esperas == [m.ESPERA_S]


def test_com_retry_presa_em_todas_as_tentativas_esgota_com_causa_timeout():
    solta, esperas = threading.Event(), []
    try:
        with pytest.raises(m.ExtracaoError, match="3 tentativas") as e:
            m._com_retry(lambda: solta.wait(30), EEFalso(), esperas.append, limite_s=0.02)
    finally:
        solta.set()
    assert isinstance(e.value.__cause__, TimeoutError)
    assert esperas == [m.ESPERA_S, m.ESPERA_S * 2]


def test_com_retry_erro_de_programacao_nao_e_repetido():
    tentativas, esperas = [], []

    def fn():
        tentativas.append(1)
        raise ValueError("bug")

    with pytest.raises(ValueError):
        m._com_retry(fn, EEFalso(), esperas.append)
    assert len(tentativas) == 1 and esperas == []


# ================================================================== _carregar_ee / _colecao
def test_carregar_ee_exige_variavel_de_ambiente(cfg, monkeypatch):
    monkeypatch.delenv(cfg.gee.projeto_env, raising=False)
    with pytest.raises(RuntimeError, match=cfg.gee.projeto_env):
        m._carregar_ee(cfg)


def test_carregar_ee_inicializa_com_o_projeto_do_ambiente(cfg, monkeypatch):
    iniciou = []
    falso = SimpleNamespace(Initialize=lambda project: iniciou.append(project))
    monkeypatch.setitem(sys.modules, "ee", falso)
    monkeypatch.setenv(cfg.gee.projeto_env, "meu-projeto")
    assert m._carregar_ee(cfg) is falso
    assert iniciou == ["meu-projeto"]


def test_colecao_filtra_municipios_e_usa_geometria_planar():
    ee = EEFalso()
    fc = m._colecao(ee, GEO, {"2200053", "2207801"})
    assert sorted(fc.cods) == ["2200053", "2207801"]
    assert len(ee.geometrias) == 2
    assert all(proj is None and geodesic is False for _, proj, geodesic in ee.geometrias)


# ================================================================== rodar
def test_lote_teste_grava_so_o_primeiro_lote_com_tres_reducoes_por_mes(cfg):
    ee = EEFalso()
    saidas, esperas = executar(cfg, ee, so_primeiro_lote=True)
    arqs = parquets(cfg)
    assert len(arqs) == 1 and esperas == []
    assert len(ee.chamadas) == MESES_POR_LOTE * 3
    assert set(ee.chamadas[:3]) == {("mean", ("soma", "max_dia", "dias_chuva")),
                                    ("min", ("dias_validos", "um")),
                                    ("count", ("soma", "um"))}
    assert "lotes=1" in saidas[0] and "a_extrair=1" in saidas[0]
    assert not any("RAW completo" in s for s in saidas)        # lote-teste não confere a janela inteira


def test_valores_chegam_na_coluna_certa_por_municipio(cfg):
    executar(cfg, EEFalso(), so_primeiro_lote=True)
    df = raw.ler_lote(parquets(cfg)[0])
    assert sorted(df["cod_ibge"].unique()) == COD
    for cod in COD:
        e = esperado(cod)
        x = df[df["cod_ibge"] == cod]
        assert len(x) == MESES_POR_LOTE
        for col in ("precip_mm", "precip_max_dia_mm", "dias_chuva", "pixels_validos_frac"):
            assert x[col].to_numpy() == pytest.approx(e[col]), (cod, col)
        assert (x["dias_validos"] == e["dias_validos"]).all(), cod


def test_limiar_colecao_e_geometria_vem_da_config(cfg):
    ee = EEFalso()
    executar(cfg, ee, so_primeiro_lote=True)
    assert set(ee.limiares) == {LIMIAR}
    assert set(ee.colecoes) == {cfg.fontes.chirps.gee_id}
    assert len(ee.geometrias) == len(COD) and all(g[2] is False for g in ee.geometrias)


def test_selo_guarda_proveniencia_e_imagens_por_mes(cfg):
    executar(cfg, EEFalso(n_imagens={"1981-02": 27}), so_primeiro_lote=True)
    meta = selo_de(parquets(cfg)[0])["meta"]
    assert meta["gee_id"] == cfg.fontes.chirps.gee_id and meta["banda"] == cfg.fontes.chirps.banda
    assert meta["limiar_dia_chuva_mm"] == LIMIAR and meta["ee_version"] == "falso-1.0"
    assert meta["extraido_em_utc"].endswith("+00:00")
    assert meta["n_imagens_por_mes"]["1981-01"] == 31
    assert meta["n_imagens_por_mes"]["1981-02"] == 27
    assert len(meta["n_imagens_por_mes"]) == MESES_POR_LOTE


def test_mes_com_lacuna_de_imagens_gera_aviso(cfg):
    saidas, _ = executar(cfg, EEFalso(n_imagens={"1981-02": 27}), so_primeiro_lote=True)
    assert any("meses_com_lacuna=['1981-02']" in s for s in saidas)


def test_janela_completa_grava_todos_os_lotes_e_confere_o_raw(cfg):
    ee = EEFalso()
    saidas, _ = executar(cfg, ee)
    assert len(parquets(cfg)) == N_LOTES
    assert len(ee.chamadas) == N_MESES * 3
    assert any("RAW completo" in s and f"linhas={N_MESES * len(COD)}" in s for s in saidas)
    meses = raw.meses_janela(cfg)
    assert len(raw.ler_raw(raw.diretorio_raw(cfg), COD, meses)) == N_MESES * len(COD)


def test_retoma_so_os_lotes_que_faltam_sem_tocar_nos_gravados(cfg):
    executar(cfg, EEFalso(), so_primeiro_lote=True)
    primeiro = parquets(cfg)[0]
    antes = (primeiro.read_bytes(), primeiro.stat().st_mtime_ns)
    ee = EEFalso()
    executar(cfg, ee)
    assert len(parquets(cfg)) == N_LOTES
    assert len(ee.chamadas) == (N_MESES - MESES_POR_LOTE) * 3          # o 1º lote não é reextraído
    assert (primeiro.read_bytes(), primeiro.stat().st_mtime_ns) == antes


def test_reexecucao_com_tudo_gravado_nao_chama_o_gee(cfg, monkeypatch):
    executar(cfg, EEFalso())
    monkeypatch.setattr(m, "_carregar_ee", lambda c: pytest.fail("não deveria carregar o ee"))
    saidas = []
    m.rodar(cfg, None, saida=lambda *a, sep=" ": saidas.append(sep.join(map(str, a))))
    assert "a_extrair=0" in saidas[0] and any("RAW completo" in s for s in saidas)


def test_sem_ee_injetado_usa_carregar_ee(cfg, monkeypatch):
    ee = EEFalso()
    monkeypatch.setattr(m, "_carregar_ee", lambda c: ee)
    m.rodar(cfg, None, so_primeiro_lote=True, dormir=lambda s: None, saida=lambda *a, sep=" ": None)
    assert len(ee.chamadas) == MESES_POR_LOTE * 3


def test_lote_adulterado_e_detectado_na_reexecucao(cfg):
    executar(cfg, EEFalso(), so_primeiro_lote=True)
    with open(parquets(cfg)[0], "ab") as f:
        f.write(b"x")
    with pytest.raises(raw.RawIntegridadeError):
        m.rodar(cfg, None, so_primeiro_lote=True, saida=lambda *a, sep=" ": None)


def test_falha_transitoria_e_repetida_e_o_lote_e_gravado(cfg):
    ee = EEFalso(falhas=2)
    _, esperas = executar(cfg, ee, so_primeiro_lote=True)
    assert esperas == [m.ESPERA_S, m.ESPERA_S * 2]
    assert len(parquets(cfg)) == 1


def test_falha_permanente_levanta_e_nao_grava_nada(cfg):
    ee = EEFalso(falhas=999)
    with pytest.raises(m.ExtracaoError):
        executar(cfg, ee, so_primeiro_lote=True)
    assert parquets(cfg) == []
    assert not list(raw.diretorio_raw(cfg).glob("*")) if raw.diretorio_raw(cfg).exists() else True


def test_falha_no_meio_preserva_lotes_prontos_e_a_retomada_completa(cfg):
    with pytest.raises(m.ExtracaoError):
        executar(cfg, EEFalso(falhar_apos=MESES_POR_LOTE * 3))         # 1º lote ok, 2º falha
    assert len(parquets(cfg)) == 1
    ee = EEFalso()
    saidas, _ = executar(cfg, ee)
    assert len(parquets(cfg)) == N_LOTES
    assert len(ee.chamadas) == (N_MESES - MESES_POR_LOTE) * 3
    assert any("RAW completo" in s for s in saidas)


def test_chamada_presa_no_gee_e_repetida_e_o_lote_e_gravado(cfg):
    ee = EEFalso(travar=1)
    try:
        _, esperas = executar(cfg, ee, so_primeiro_lote=True, limite_s=0.2)
    finally:
        ee.liberar.set()
    assert esperas == [m.ESPERA_S]                                  # uma única repetição
    assert len(parquets(cfg)) == 1
    df = raw.ler_lote(parquets(cfg)[0])
    assert len(df) == MESES_POR_LOTE * len(COD) and df["precip_mm"].notna().all()


def test_chamada_sempre_presa_levanta_e_nao_grava_nada(cfg):
    ee = EEFalso(travar=99)
    try:
        with pytest.raises(m.ExtracaoError, match="3 tentativas"):
            executar(cfg, ee, so_primeiro_lote=True, limite_s=0.02)
    finally:
        ee.liberar.set()
    assert parquets(cfg) == []


def test_progresso_uma_linha_por_mes_com_indice_global_do_lote(cfg):
    saidas, _ = executar(cfg, EEFalso(), lotes=(2, 2))
    por_mes = [s for s in saidas if s.startswith(f"RODAR | lote 2/{N_LOTES} | mes ")]
    assert len(por_mes) == MESES_POR_LOTE
    assert por_mes[0].startswith(f"RODAR | lote 2/{N_LOTES} | mes 1989-05 | ")      # lote 2 começa 100 meses após 1981-01
    assert por_mes[0].endswith("s") and por_mes[0].split(" | ")[-1][:-1].isdigit()


def test_progresso_mede_o_tempo_de_cada_mes_com_o_relogio_injetado(cfg):
    ticks = iter(range(0, 10_000, 7))
    saidas = []
    m.rodar(cfg, EEFalso(), so_primeiro_lote=True, dormir=lambda s: None, relogio=lambda: next(ticks),
            saida=lambda *a, sep=" ": saidas.append(sep.join(map(str, a))))
    por_mes = [s for s in saidas if " | mes " in s]
    assert len(por_mes) == MESES_POR_LOTE and all(s.endswith("| 7s") for s in por_mes)


def test_lotes_extrai_so_a_faixa_pedida_e_nao_confere_a_janela(cfg):
    ee = EEFalso()
    saidas, _ = executar(cfg, ee, lotes=(2, 3))
    assert len(ee.chamadas) == 2 * MESES_POR_LOTE * 3
    assert [p.name for p in parquets(cfg)] == sorted(
        raw.caminho_lote(raw.diretorio_raw(cfg), *l).name for l in raw.dividir_em_lotes(
            raw.meses_janela(cfg), len(COD), MAX_LINHAS)[1:3])
    assert f"selecao=2:3/{N_LOTES}" in saidas[0] and "lotes=2" in saidas[0] and "a_extrair=2" in saidas[0]
    assert not any("RAW completo" in s for s in saidas)


def test_lotes_a_b_iguais_extrai_um_lote(cfg):
    ee = EEFalso()
    executar(cfg, ee, lotes=(N_LOTES, N_LOTES))
    assert len(parquets(cfg)) == 1 and len(ee.chamadas) == (N_MESES - 5 * MESES_POR_LOTE) * 3


def test_tres_faixas_em_execucoes_separadas_completam_o_raw_sem_reextrair(cfg, monkeypatch):
    for faixa in ((1, 2), (3, 4), (5, N_LOTES)):
        executar(cfg, EEFalso(), lotes=faixa)
    assert len(parquets(cfg)) == N_LOTES
    monkeypatch.setattr(m, "_carregar_ee", lambda c: pytest.fail("não deveria carregar o ee"))
    saidas = []
    m.rodar(cfg, None, saida=lambda *a, sep=" ": saidas.append(sep.join(map(str, a))))
    assert "a_extrair=0" in saidas[0]
    assert any("RAW completo" in s and f"linhas={N_MESES * len(COD)}" in s for s in saidas)


def test_faixas_sobrepostas_nao_reextraem_o_que_ja_esta_gravado(cfg):
    executar(cfg, EEFalso(), lotes=(1, 3))
    ee = EEFalso()
    saidas, _ = executar(cfg, ee, lotes=(2, 4))
    assert len(ee.chamadas) == MESES_POR_LOTE * 3                  # só o lote 4
    assert "ja_gravados=2" in saidas[0] and "a_extrair=1" in saidas[0]


def test_lotes_com_lote_adulterado_dentro_da_faixa_e_detectado(cfg):
    executar(cfg, EEFalso(), lotes=(1, 1))
    with open(parquets(cfg)[0], "ab") as f:
        f.write(b"x")
    with pytest.raises(raw.RawIntegridadeError):
        m.rodar(cfg, None, lotes=(1, 2), saida=lambda *a, sep=" ": None)


@pytest.mark.parametrize("faixa", [(0, 2), (3, 2), (1, N_LOTES + 1), (N_LOTES + 1, N_LOTES + 1)])
def test_lotes_fora_do_intervalo_e_recusado_antes_de_chamar_o_gee(cfg, monkeypatch, faixa):
    monkeypatch.setattr(m, "_carregar_ee", lambda c: pytest.fail("não deveria carregar o ee"))
    with pytest.raises(m.ExtracaoError, match="fora de"):
        m.rodar(cfg, None, lotes=faixa, saida=lambda *a, sep=" ": None)
    assert parquets(cfg) == []


def test_lote_teste_e_lotes_juntos_sao_recusados(cfg):
    with pytest.raises(m.ExtracaoError, match="só um entre"):
        executar(cfg, EEFalso(), so_primeiro_lote=True, lotes=(1, 2))


@pytest.mark.parametrize("quem", ["sem_dado", "ausentes"])
def test_municipio_sem_dado_vira_nan_triplo_e_nao_contamina_os_outros(cfg, quem):
    ruim = "2205706"
    executar(cfg, EEFalso(**{quem: {ruim}}), so_primeiro_lote=True)
    df = raw.ler_lote(parquets(cfg)[0])
    x = df[df["cod_ibge"] == ruim]
    assert len(x) == MESES_POR_LOTE
    assert x["precip_mm"].isna().all() and x["precip_max_dia_mm"].isna().all() and x["dias_chuva"].isna().all()
    assert (x["pixels_validos_frac"] == 0).all() and (x["dias_validos"] == 0).all()
    outros = df[df["cod_ibge"] != ruim]
    assert outros["precip_mm"].notna().all()
    assert outros.loc[outros["cod_ibge"] == "2200053", "precip_mm"].to_numpy() == pytest.approx(100.0)


def test_modo_piloto_e_recusado(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ler_geojson_gee", lambda c: GEO)
    cfg_p = cfg_tmp(tmp_path, ajustes={("escopo", "modo"): "piloto"})
    ee = EEFalso()
    with pytest.raises(m.ExtracaoError, match="piloto"):
        m.rodar(cfg_p, ee)
    assert ee.chamadas == [] and parquets(cfg_p) == []


def test_pendencia_na_config_do_chirps_bloqueia(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ler_geojson_gee", lambda c: GEO)
    cfg_v = cfg_tmp(tmp_path, ajustes={("fontes", "chirps", "unidade"): "VERIFICAR"})
    ee = EEFalso()
    with pytest.raises(PendenciaError):
        m.rodar(cfg_v, ee)
    assert ee.chamadas == []


# ================================================================== sonda
def test_sonda_imprime_fatos_e_nao_grava_raw(cfg, capsys):
    ee = EEFalso()
    m.sonda(cfg, ee)
    linhas = [l for l in capsys.readouterr().out.splitlines() if l.startswith("SONDA")]
    texto = "\n".join(linhas)
    assert "colecao" in texto and cfg.fontes.chirps.gee_id in texto and "12345" in texto
    assert "1981-01-01" in texto and "2026-09-30" in texto and str(cfg.janela.fim) in texto
    assert "nominalScale" in texto and "5565.97" in texto and str(cfg.fontes.chirps.escala_m) in texto
    for mes in m.MESES_SONDA:
        assert f"mes | {mes}" in texto
    for cod, motivo in (("2207801", "polo"), ("2200053", "menor bbox"), ("2205706", "litoral")):
        assert f"{cod} | {motivo}" in texto
    assert set(ee.limiares) == {LIMIAR}
    assert not raw.diretorio_raw(cfg).exists()


# ================================================================== CLI
def test_cli_lote_teste_so_vale_para_rodar():
    with pytest.raises(SystemExit) as e:
        m.main(["sonda", "--lote-teste"])
    assert e.value.code == 2


def test_cli_encaminha_para_sonda_e_rodar(monkeypatch):
    chamadas = []
    monkeypatch.setattr(m, "load_config", lambda: "CFG")
    monkeypatch.setattr(m, "sonda", lambda c: chamadas.append(("sonda", c)))
    monkeypatch.setattr(m, "rodar", lambda c, so_primeiro_lote, lotes: chamadas.append(("rodar", c, so_primeiro_lote, lotes)))
    m.main(["sonda"])
    m.main(["rodar"])
    m.main(["rodar", "--lote-teste"])
    m.main(["rodar", "--lotes", "1:10"])
    m.main(["rodar", "--lotes", "21:30"])
    m.main(["rodar", "--lotes", "7:7"])
    assert chamadas == [("sonda", "CFG"), ("rodar", "CFG", False, None), ("rodar", "CFG", True, None),
                        ("rodar", "CFG", False, (1, 10)), ("rodar", "CFG", False, (21, 30)),
                        ("rodar", "CFG", False, (7, 7))]


@pytest.mark.parametrize("argv", [
    ["sonda", "--lotes", "1:10"],
    ["rodar", "--lote-teste", "--lotes", "1:10"],
    ["rodar", "--lotes", "abc"],
    ["rodar", "--lotes", "3"],
    ["rodar", "--lotes", "1:2:3"],
    ["rodar", "--lotes", "0:5"],
    ["rodar", "--lotes", "5:2"],
    ["rodar", "--lotes", "-1:4"],
])
def test_cli_lotes_invalido_sai_com_codigo_2_antes_de_carregar_a_config(monkeypatch, argv):
    monkeypatch.setattr(m, "load_config", lambda: pytest.fail("não deveria carregar a config"))
    monkeypatch.setattr(m, "rodar", lambda *a, **k: pytest.fail("não deveria rodar"))
    monkeypatch.setattr(m, "sonda", lambda *a, **k: pytest.fail("não deveria sondar"))
    with pytest.raises(SystemExit) as e:
        m.main(argv)
    assert e.value.code == 2
