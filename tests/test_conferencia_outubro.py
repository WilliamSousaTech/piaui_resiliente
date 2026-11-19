"""Conferência de outubro com a Open-Meteo: nunca usa a rede nos testes (a API é simulada por uma função)."""
import hashlib
import json
import urllib.parse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import conferencia_outubro as co

RAIZ = Path(__file__).resolve().parents[1]
ALVO = pd.Period("2026-10", freq="M")
INI, FIM = date(2026, 8, 1), date(2026, 10, 31)
DATAS = co.datas_do_periodo(INI, FIM)
HOJE = date(2026, 11, 19)


def _poligono(coords):
    return {"type": "Polygon", "coordinates": [coords]}


def _feicao(cod, geom, nome="X"):
    return {"type": "Feature", "properties": {"cod_ibge": cod, "nome": nome}, "geometry": geom}


QUADRADO = _poligono([[-42.0, -6.0], [-41.0, -6.0], [-41.0, -7.0], [-42.0, -7.0], [-42.0, -6.0]])
# "C" aberto para a direita: o centroide cai no vão (fora do polígono)
FORMA_C = _poligono([[-42.0, -6.0], [-41.0, -6.0], [-41.0, -6.2], [-41.8, -6.2], [-41.8, -6.8], [-41.0, -6.8],
                     [-41.0, -7.0], [-42.0, -7.0], [-42.0, -6.0]])


def _resposta(n, valores_por_ponto=None, datas=DATAS, como_lista=None):
    itens = []
    for i in range(n):
        v = valores_por_ponto[i] if valores_por_ponto else [0.0] * len(datas)
        itens.append({"latitude": -6.0, "longitude": -41.0, "daily": {"time": list(datas), "precipitation_sum": v}})
    lista = como_lista if como_lista is not None else n != 1
    return json.dumps(itens if lista else itens[0]).encode()


# ------------------------------------------------------------------ pontos
def test_pontos_ordenados_e_dentro_do_estado():
    malha = {"features": [_feicao("2200002", QUADRADO), _feicao("2200001", QUADRADO)]}
    p = co.pontos_da_malha(malha)
    assert [x.cod_ibge for x in p] == ["2200001", "2200002"]
    assert (p[0].lat, p[0].lon) == (-6.5, -41.5)


def test_centroide_fora_do_poligono_usa_ponto_interno():
    p = co.pontos_da_malha({"features": [_feicao("2200001", FORMA_C)]})[0]
    from shapely.geometry import Point, shape
    assert shape(FORMA_C).contains(Point(p.lon, p.lat))


@pytest.mark.parametrize("malha, trecho", [
    ({}, "sem 'features'"),
    ({"features": []}, "sem 'features'"),
    ({"features": [_feicao("1", QUADRADO), _feicao("1", QUADRADO)]}, "repetido"),
    ({"features": [{"properties": {}, "geometry": QUADRADO}]}, "inválida"),
    ({"features": [_feicao("1", _poligono([[-6, -42], [-6, -41], [-7, -41], [-7, -42], [-6, -42]]))]}, "fora do Piauí"),
    ({"features": [_feicao("1", _poligono([[-42, -6], [-41, -7], [-41, -6], [-42, -7], [-42, -6]]))]}, "inválida"),
])
def test_pontos_recusa_malha_ruim(malha, trecho):
    with pytest.raises(co.ConferenciaError, match=trecho):
        co.pontos_da_malha(malha)


# ------------------------------------------------------------------ período
def test_periodo_de_outubro():
    assert co.periodo_busca(ALVO, HOJE) == (INI, FIM)


def test_periodo_limite_do_atraso():
    assert co.periodo_busca(ALVO, date(2026, 11, 5)) == (INI, FIM)             # 31/out + 5 dias: pode
    with pytest.raises(co.ConferenciaError, match="atraso"):
        co.periodo_busca(ALVO, date(2026, 11, 4))                                # um dia antes: recusa


def test_datas_inclui_os_dois_extremos():
    assert len(DATAS) == 92 and DATAS[0] == "2026-08-01" and DATAS[-1] == "2026-10-31"


# ------------------------------------------------------------------ URL
def test_url_tem_os_parametros_documentados():
    pts = [co.Ponto("1", "A", -5.1, -41.2), co.Ponto("2", "B", -6.25, -42.0)]
    u = co.montar_url(pts, INI, FIM)
    q = urllib.parse.parse_qs(urllib.parse.urlparse(u).query)
    assert u.startswith("https://archive-api.open-meteo.com/v1/archive?")
    assert q["latitude"] == ["-5.1000,-6.2500"] and q["longitude"] == ["-41.2000,-42.0000"]
    assert q["start_date"] == ["2026-08-01"] and q["end_date"] == ["2026-10-31"]
    assert q["daily"] == ["precipitation_sum"] and q["timezone"] == ["America/Fortaleza"]
    assert " " not in u


def test_url_sem_pontos():
    with pytest.raises(co.ConferenciaError):
        co.montar_url([], INI, FIM)


# ------------------------------------------------------------------ leitura da resposta
def test_resposta_de_um_ponto_e_de_varios():
    s1, inv1 = co.ler_resposta(_resposta(1), 1, DATAS)
    s3, inv3 = co.ler_resposta(_resposta(3), 3, DATAS)
    assert len(s1) == 1 and len(s3) == 3 and inv1 == inv3 == 0 and s3[2].shape == (92,)


def test_resposta_lista_com_um_ponto_tambem_vale():
    s, _ = co.ler_resposta(_resposta(1, como_lista=True), 1, DATAS)
    assert len(s) == 1


def test_null_vira_nan_e_valor_absurdo_vira_nan_contado():
    v = [1.0] * 92
    v[0], v[1], v[2], v[3] = None, -0.5, 9999.0, 2.5
    (s,), inv = co.ler_resposta(_resposta(1, [v]), 1, DATAS)
    assert np.isnan(s[:3]).all() and s[3] == 2.5 and inv == 2                      # null não conta; -0.5 e 9999 contam


@pytest.mark.parametrize("corpo, trecho", [
    (b"<html>", "não é JSON"),
    (json.dumps({"error": True, "reason": "Parameter 'timezone' obrigatório"}).encode(), "recusou"),
    (_resposta(2), "pedi 3"),
    (_resposta(3, datas=DATAS[:-1]), "datas"),
    (json.dumps([{"daily": {"time": DATAS}}] * 3).encode(), "ausente"),
    (json.dumps([{"daily": {"time": DATAS, "precipitation_sum": [0.0] * 5}}] * 3).encode(), "tamanho"),
    (json.dumps([{"daily": {"time": DATAS, "precipitation_sum": ["a"] * 92}}] * 3).encode(), "não numérico"),
    (json.dumps([{"daily": {"time": DATAS, "precipitation_sum": [True] * 92}}] * 3).encode(), "não numérico"),
    (json.dumps([{}] * 3).encode(), "datas"),
])
def test_resposta_ruim_e_recusada(corpo, trecho):
    with pytest.raises(co.ConferenciaError, match=trecho):
        co.ler_resposta(corpo, 3, DATAS)


# ------------------------------------------------------------------ soma mensal
def test_soma_mensal_completa_e_incompleta():
    v = np.ones(92)
    v[40] = np.nan                                                                   # 10/set
    s = co.soma_mensal(DATAS, v)
    assert s["2026-08"] == {"mm": 31.0, "dias_validos": 31, "dias_no_mes": 31}
    assert s["2026-09"] == {"mm": None, "dias_validos": 29, "dias_no_mes": 30}     # dia faltante NUNCA vira zero
    assert s["2026-10"]["mm"] == 31.0


def test_soma_mensal_tamanhos_diferentes():
    with pytest.raises(co.ConferenciaError):
        co.soma_mensal(DATAS, np.ones(3))


# ------------------------------------------------------------------ percentil e agrupamento
def test_percentil_historico():
    h = np.arange(30, dtype=float)                                                   # 0..29
    assert co.percentil_historico(-1.0, h) == 0.0
    assert co.percentil_historico(100.0, h) == 100.0
    assert co.percentil_historico(10.0, h) == pytest.approx(100 * (10 + 0.5) / 30, abs=0.1)


def test_percentil_empate_conta_metade_e_poucos_anos_ou_nan_dao_none():
    h = np.zeros(30)
    assert co.percentil_historico(0.0, h) == 50.0
    assert co.percentil_historico(5.0, np.arange(19.0)) is None                    # só 19 anos
    assert co.percentil_historico(None, h) is None
    assert co.percentil_historico(float("nan"), h) is None
    assert co.percentil_historico(1.0, np.array([np.nan] * 40)) is None


def _reg(om, ch, perc=None, anom=None, sinal=False):
    return {"chuva_mm": om, "chirps_mm": ch, "percentil": perc, "anomalia_mm": anom, "sinal_modelo": sinal}


def test_comparar_fontes_concordancia_perfeita_e_pares_faltando():
    regs = [_reg({"2026-08": 1.0}, {"2026-08": 1.0}), _reg({"2026-08": 4.0}, {"2026-08": 4.0}),
            _reg({"2026-08": 9.0}, {"2026-08": 9.0}), _reg({"2026-08": None}, {"2026-08": 5.0})]
    r = co.comparar_fontes(regs, ["2026-08", "2026-09"])
    assert r["2026-08"]["n"] == 3 and r["2026-08"]["vies_medio_mm"] == 0.0 and r["2026-08"]["correlacao"] == 1.0
    assert r["2026-09"] == {"n": 0, "vies_medio_mm": None, "erro_medio_abs_mm": None, "correlacao": None,
                            "open_meteo_medio_mm": None, "chirps_medio_mm": None}


def test_comparar_fontes_vies_e_correlacao_indefinida():
    regs = [_reg({"m": 3.0}, {"m": 1.0}), _reg({"m": 3.0}, {"m": 2.0}), _reg({"m": 3.0}, {"m": 3.0})]
    r = co.comparar_fontes(regs, ["m"])["m"]
    assert r["vies_medio_mm"] == 1.0 and r["erro_medio_abs_mm"] == 1.0 and r["correlacao"] is None   # OM constante


def test_resumir_grupo():
    regs = [_reg({}, {}, 10.0, -5.0), _reg({}, {}, 25.0, -1.0), _reg({}, {}, 60.0, 4.0), _reg({}, {}, None, None)]
    g = co.resumir_grupo(regs)
    assert g == {"n": 4, "n_com_dado": 3, "n_secos": 2, "fracao_secos": 0.667, "mediana_percentil": 25.0,
                 "mediana_anomalia_mm": -1.0, "n_abaixo_da_normal": 2}
    vazio = co.resumir_grupo([])
    assert vazio["n"] == 0 and vazio["fracao_secos"] is None and vazio["mediana_percentil"] is None


# ------------------------------------------------------------------ registros e documento
def _montar(om_out, normal_mm, hist, chirps=None):
    cod = "2200001"
    om = {"2026-08": {"mm": 0.0, "dias_validos": 31, "dias_no_mes": 31},
          "2026-09": {"mm": 1.0, "dias_validos": 30, "dias_no_mes": 30},
          "2026-10": {"mm": om_out, "dias_validos": 31 if om_out is not None else 30, "dias_no_mes": 31}}
    mat = chirps if chirps is not None else pd.DataFrame(
        {cod: [0.5, 2.0]}, index=pd.period_range("2026-08", "2026-09", freq="M"))
    return co.montar_registros([cod], {cod: om}, mat, ALVO, pd.Series({cod: normal_mm}), {cod: hist}, {cod})[0]


def test_registro_pct_normal_e_percentil():
    r = _montar(10.0, 20.0, np.arange(35, dtype=float))
    assert r["pct_normal"] == 50.0 and r["anomalia_mm"] == -10.0 and r["normal_mm"] == 20.0
    assert r["percentil"] == pytest.approx(100 * 10.5 / 35, abs=0.1)
    assert r["completo"] and r["sinal_modelo"] and r["chirps_mm"] == {"2026-08": 0.5, "2026-09": 2.0}


def test_registro_normal_baixa_nao_calcula_porcentagem():
    r = _montar(15.0, 3.0, np.arange(35, dtype=float))
    assert r["pct_normal"] is None and r["anomalia_mm"] == 12.0                    # 500% de 3 mm enganaria


def test_registro_mes_incompleto_fica_sem_numero():
    r = _montar(None, 20.0, np.arange(35, dtype=float))
    assert not r["completo"] and r["pct_normal"] is None and r["percentil"] is None and r["anomalia_mm"] is None
    assert r["dias_validos_alvo"] == 30


def test_registro_sem_chirps_do_mes_fica_none():
    mat = pd.DataFrame({"outro": [1.0]}, index=pd.period_range("2026-09", "2026-09", freq="M"))
    r = _montar(10.0, 20.0, np.arange(35, dtype=float), chirps=mat)
    assert r["chirps_mm"] == {"2026-08": None, "2026-09": None}


def test_documento_nao_aceita_nan_no_json(tmp_path):
    r = _montar(10.0, 20.0, np.arange(35, dtype=float))
    doc = co.montar_documento([r], alvo=ALVO, ini=INI, fim=FIM, hoje=HOJE, dataset_sha256="a" * 64,
                              janela_historica="1981-01 a 2015-12", n_invalidos=0)
    destino = tmp_path / "sub" / "c.json"
    co.gravar_atomico(destino, doc)
    lido = json.loads(destino.read_text(encoding="utf-8"))
    assert lido["mes_alvo"] == "2026-10" and lido["grupos"]["com_sinal"]["n"] == 1 and lido["grupos"]["sem_sinal"]["n"] == 0
    assert lido["fonte_conferencia"]["licenca"] == "CC BY 4.0" and len(lido["avisos"]) == 3
    assert not list(destino.parent.glob("*.tmp"))
    doc["municipios"][0]["normal_mm"] = float("nan")
    with pytest.raises(ValueError):
        co.gravar_atomico(tmp_path / "nan.json", doc)


# ------------------------------------------------------------------ API simulada: lotes, repetição e erro
class Api:
    """Fingida: responde 200 com chuva zero, ou a sequência de status dada, e guarda o que foi pedido."""

    def __init__(self, status=(200,)):
        self.status, self.urls = list(status), []

    def __call__(self, url, timeout):
        self.urls.append(url)
        st = self.status.pop(0) if len(self.status) > 1 else self.status[0]
        if st != 200:
            return st, json.dumps({"error": True, "reason": f"motivo {st}"}).encode()
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        n = len(q["latitude"][0].split(","))
        return 200, _resposta(n, datas=co.datas_do_periodo(date.fromisoformat(q["start_date"][0]),
                                                           date.fromisoformat(q["end_date"][0])))


PONTOS = [co.Ponto(f"{2200000 + i}", "M", -5.0 - i / 10, -41.0) for i in range(5)]


def test_buscar_divide_em_lotes_e_espaca_os_pedidos():
    api, pausas = Api(), []
    saida, inv = co.buscar(PONTOS, INI, FIM, abrir=api, lote=2, pausa=7.0, dormir=pausas.append)
    assert len(api.urls) == 3 and pausas == [7.0, 7.0] and set(saida) == {p.cod_ibge for p in PONTOS} and inv == 0
    assert all(s.shape == (92,) for s in saida.values())


def test_buscar_repete_apos_429_esperando_um_minuto():
    api, pausas = Api([429, 429, 200]), []
    co.buscar(PONTOS[:1], INI, FIM, abrir=api, dormir=pausas.append)
    assert len(api.urls) == 3 and pausas == [60.0, 120.0]


def test_buscar_repete_apos_500_com_espera_crescente():
    api, pausas = Api([503, 200]), []
    co.buscar(PONTOS[:1], INI, FIM, abrir=api, pausa=2.0, dormir=pausas.append)
    assert pausas == [4.0]


def test_buscar_nao_repete_pedido_errado():
    api = Api([400])
    with pytest.raises(co.ConferenciaError, match="recusou.*motivo 400"):
        co.buscar(PONTOS[:1], INI, FIM, abrir=api, dormir=lambda s: None)
    assert len(api.urls) == 1


def test_buscar_desiste_depois_das_tentativas():
    api = Api([429])
    with pytest.raises(co.ConferenciaError, match="3 tentativas.*429"):
        co.buscar(PONTOS[:1], INI, FIM, abrir=api, tentativas=3, dormir=lambda s: None)
    assert len(api.urls) == 3


@pytest.mark.parametrize("kw", [{"lote": 0}, {"tentativas": 0}])
def test_buscar_parametros_invalidos(kw):
    with pytest.raises(co.ConferenciaError):
        co.buscar(PONTOS, INI, FIM, abrir=Api(), **kw)


# ------------------------------------------------------------------ ponta a ponta com os dados reais do repositório
def _api_outubro_chuvoso(faltando_primeiro=False):
    """Agosto e setembro secos, outubro com 1 mm por dia; opcionalmente 1 dia nulo no 1º ponto de cada pedido."""
    def abrir(url, timeout):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        n = len(q["latitude"][0].split(","))
        datas = co.datas_do_periodo(date.fromisoformat(q["start_date"][0]), date.fromisoformat(q["end_date"][0]))
        v = [1.0 if d.startswith("2026-10") else 0.0 for d in datas]
        pts = [list(v) for _ in range(n)]
        if faltando_primeiro:
            pts[0][datas.index("2026-10-15")] = None
        return 200, _resposta(n, pts, datas)
    return abrir


pytestmark_real = pytest.mark.skipif(not (RAIZ / "data" / "curated" / "manifest.json").is_file(),
                                     reason="dados do projeto ausentes")


def _args(tmp_path, destino="c.json"):
    return ["--params", str(RAIZ / "params.yaml"), "--previsao", str(RAIZ / "data/reference/previsao_proximo_mes.json"),
            "--saida", str(tmp_path / destino), "--pausa", "0", "--lote", "100"]


@pytestmark_real
def test_main_com_dados_reais_gera_arquivo_e_nao_altera_o_manifest(tmp_path):
    manifest = RAIZ / "data" / "curated" / "manifest.json"
    antes = hashlib.sha256(manifest.read_bytes()).hexdigest()
    previsao = json.loads((RAIZ / "data/reference/previsao_proximo_mes.json").read_text(encoding="utf-8"))
    linhas = []
    rc = co.main(_args(tmp_path), saida=linhas.append, abrir=_api_outubro_chuvoso(True), dormir=lambda s: None,
                 hoje=HOJE)
    assert rc == 0, linhas
    texto = (tmp_path / "c.json").read_text(encoding="utf-8")
    assert "NaN" not in texto and "Infinity" not in texto
    doc = json.loads(texto)
    assert doc["mes_alvo"] == "2026-10" and doc["n_municipios"] == 216 and len(doc["municipios"]) == 216
    assert doc["previsao_dataset_sha256"] == previsao["dataset_sha256"]
    assert doc["grupos"]["com_sinal"]["n"] == previsao["resumo"]["n_alerta_algum"]
    assert doc["grupos"]["com_sinal"]["n"] + doc["grupos"]["sem_sinal"]["n"] == 216
    assert doc["n_incompletos"] == 3                                               # 1º ponto de cada um dos 3 pedidos (lote 100)
    inc = [m for m in doc["municipios"] if not m["completo"]]
    assert all(m["chuva_mm"]["2026-10"] is None and m["percentil"] is None for m in inc)
    ok = next(m for m in doc["municipios"] if m["completo"])
    assert ok["chuva_mm"]["2026-10"] == 31.0 and ok["chuva_mm"]["2026-08"] == 0.0 and ok["normal_mm"] is not None
    assert set(doc["comparacao_fontes"]) == {"2026-08", "2026-09"}                 # só os meses em que as duas fontes existem
    assert doc["comparacao_fontes"]["2026-09"]["n"] > 200
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == antes             # o manifest não mudou
    assert any("Pronto" in x for x in linhas)


@pytestmark_real
def test_main_recusa_quando_a_reanalise_ainda_nao_chegou(tmp_path):
    linhas = []
    rc = co.main(_args(tmp_path), saida=linhas.append, abrir=Api(), dormir=lambda s: None, hoje=date(2026, 11, 2))
    assert rc == 2 and "atraso" in linhas[-1] and not (tmp_path / "c.json").exists()


@pytestmark_real
def test_main_erro_da_api_vira_codigo_2_sem_gravar(tmp_path):
    linhas = []
    rc = co.main(_args(tmp_path), saida=linhas.append, abrir=Api([400]), dormir=lambda s: None, hoje=HOJE)
    assert rc == 2 and "ERRO" in linhas[-1] and not (tmp_path / "c.json").exists()


def test_main_sem_params_da_erro_amigavel(tmp_path):
    linhas = []
    rc = co.main(["--params", str(tmp_path / "nao_existe.yaml"), "--saida", str(tmp_path / "c.json")],
                 saida=linhas.append, abrir=Api(), hoje=HOJE)
    assert rc == 2 and "params.yaml" in linhas[-1]
