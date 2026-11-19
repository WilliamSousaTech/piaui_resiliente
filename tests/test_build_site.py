"""Testes de src/web/build_site.py (Item 11). Sem internet; dados sintéticos só aqui (nunca em data/ nem no site).

Janela real do params.yaml: 1981-01..2026-09 (549 meses). Três municípios de mentira (A, B, C).
"""
import ast
import base64
import json
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import build_demo as bd
from src.data import chirps_raw, manifest as mf
from src.data.config import load_config
from src.web import build_site as bs

RAIZ = Path(__file__).resolve().parents[1]
PARAMS_REAL = RAIZ / "params.yaml"
TEMPLATE = RAIZ / "src" / "web" / "template.html"
MESES = pd.period_range("1981-01", "2026-09", freq="M")
NT = len(MESES)
A, B, C = "2200001", "2200002", "2200003"
AGORA = datetime(2026, 11, 19, 12, 30, 45, tzinfo=timezone.utc)
VERIFICADO = datetime(2026, 11, 19, 15, 0, 0, tzinfo=timezone.utc)


# ------------------------------------------------------------------ dados sintéticos
def quadrado(lon, lat, lado=0.5):
    return [[lon, lat], [lon + lado, lat], [lon + lado, lat - lado], [lon, lat - lado], [lon, lat]]


def feature(cod, nome, geom):
    return {"type": "Feature", "properties": {"cod_ibge": cod, "nome": nome, "situacao": "res150"}, "geometry": geom}


def malha_tres():
    """A no norte (lat maior), C no meio, B no sul; C é um MultiPolygon com um furo no primeiro polígono."""
    furo = quadrado(-44.9, -6.6, 0.1)[::-1]
    return {"type": "FeatureCollection", "features": [
        feature(B, "Beta do Sul", {"type": "Polygon", "coordinates": [quadrado(-44.0, -8.0)]}),
        feature(A, "Alfa do Norte", {"type": "Polygon", "coordinates": [quadrado(-44.0, -4.0)]}),
        feature(C, "Tanque do Piauí", {"type": "MultiPolygon", "coordinates": [
            [quadrado(-45.0, -6.0, 0.8), furo], [quadrado(-43.0, -6.0)]]}),
    ]}


def raw_sintetico(cods=(A, B, C), seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    quadros = []
    for cod in cods:
        p = rng.gamma(2.0, 60.0, NT).astype("float32")
        p[rng.random(NT) < 0.1] = 0.0
        quadros.append(pd.DataFrame({
            "cod_ibge": cod, "mes": MESES.to_timestamp(), "precip_mm": p,
            "precip_max_dia_mm": (p * 0.4).astype("float32"),
            "dias_chuva": np.minimum(MESES.days_in_month, np.floor(p / 8)).astype("float32"),
            "dias_validos": MESES.days_in_month.astype("int16"),
            "dias_no_mes": MESES.days_in_month.astype("int16"),
            "pixels_validos_frac": np.float32(1.0)}))
    return pd.concat(quadros, ignore_index=True)


def escrever_raw(raiz: Path, raw: pd.DataFrame, cods) -> None:
    cfg = load_config(raiz / "params.yaml")
    d = chirps_raw.diretorio_raw(cfg)
    lotes = (("1981-01", "2000-12", "2026-11-19T00:40:00+00:00"), ("2001-01", "2026-09", "2026-11-18T23:10:00+00:00"))
    for ini, fim, quando in lotes:
        i, f = pd.Period(ini, "M"), pd.Period(fim, "M")
        m = (raw["mes"] >= i.to_timestamp()) & (raw["mes"] <= f.to_timestamp())
        chirps_raw.gravar_lote(raw[m].reset_index(drop=True), d, cods, i, f, meta={"extraido_em_utc": quando})


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def raw_base():
    return raw_sintetico()


@pytest.fixture(scope="module")
def insumos(cfg, raw_base):
    r = bd.montar_demo(raw_base, cfg)
    manifest = {"fontes": {"chirps": {"gee_id": "UCSB-CHG/CHIRPS/DAILY"}}, "projeto": {"versao_params": "0.3.1"},
                "escopo": {"malha_municipal_ano": 2022}, "janela": {"inicio": "1981-01", "fim": "2026-09"},
                "split": {"treino_fim": "2015-12", "teste_fim": "2025-12"},
                "extras": {"demo": {"spi_calibrado_ate": "2015-12"}},
                "dataset_sha256": "ab" * 32, "gerado_em_utc": "2026-11-19T12:30:45Z", "arquivos": {"x": {}, "y": {}}}
    return bs.Insumos(demo=r.demo_observado, qualidade=r.qualidade_municipio, raw=raw_base, malha=malha_tres(),
                      manifest=manifest, extraido_em=("2026-11-18T23:10:00Z", "2026-11-19T00:40:00Z"))


@pytest.fixture(scope="module")
def payload(insumos, cfg):
    return bs.montar_payload(insumos, cfg, verificado_em=VERIFICADO)


def decodificar(b64: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


# ================================================================== classes
@pytest.mark.parametrize("spi,classe", [
    (-3.0, 0), (-2.0, 0), (-1.99, 1), (-1.5, 1), (-1.49, 2), (-1.0, 2), (-0.99, 3), (0.0, 3),
    (0.99, 3), (1.0, 4), (1.49, 4), (1.5, 5), (1.99, 5), (2.0, 6), (3.4, 6)])
def test_classificar_limites(spi, classe):
    assert bs.classificar([spi])[0] == classe


def test_classificar_nan_vira_sem_dado_e_mantem_forma():
    r = bs.classificar(np.array([[np.nan, -1.2], [0.3, np.nan]]))
    assert r.dtype == np.uint8 and r.shape == (2, 2)
    assert r.tolist() == [[255, 2], [3, 255]]


def test_em_seca_e_spi_menor_ou_igual_a_menos_um():
    seca = {i for i, c in enumerate(bs.CLASSES) if c[5]}
    assert seca == {0, 1, 2}
    assert not bs.CLASSES[3][5] and bs.CLASSES[3][1] == "Próximo do normal"


def test_classes_tem_sete_entradas_com_cores_distintas_nos_dois_temas():
    assert len(bs.CLASSES) == 7
    assert len({c[3] for c in bs.CLASSES}) == 7 and len({c[4] for c in bs.CLASSES}) == 7
    for c in bs.CLASSES:
        assert re.fullmatch(r"#[0-9a-f]{6}", c[3]) and re.fullmatch(r"#[0-9a-f]{6}", c[4])


# ================================================================== inteiros e base64
def test_inteiros16_ida_e_volta_com_duas_casas():
    v = np.array([-3.456, 0.0, 1.005, 2.999, np.nan])
    q = bs.inteiros16(v, 100)
    assert q.dtype == np.int16
    assert q.tolist() == [-346, 0, 100, 300, bs.NA16] or q.tolist() == [-346, 0, 101, 300, bs.NA16]
    ok = ~np.isnan(v)
    assert np.allclose(q[ok] / 100, v[ok], atol=0.006)


def test_inteiros16_nan_vira_sentinela():
    assert bs.inteiros16([np.nan, np.nan], 10).tolist() == [bs.NA16, bs.NA16]


@pytest.mark.parametrize("valor", [400.0, -400.0, -327.68])
def test_inteiros16_estouro_ou_sentinela_e_erro(valor):
    with pytest.raises(bs.SiteError, match="int16"):
        bs.inteiros16([0.0, valor], 100)


def test_inteiros16_aceita_os_extremos_validos():
    assert bs.inteiros16([327.67, -327.67], 100).tolist() == [32767, -32767]


def test_em_base64_little_endian():
    a = np.array([1, -2, 300], dtype="int16")
    assert base64.b64decode(bs.em_base64(a)) == b"\x01\x00\xfe\xff\x2c\x01"
    assert decodificar(bs.em_base64(np.array([0, 255], dtype="uint8")), "uint8").tolist() == [0, 255]


# ================================================================== malha
def test_projetar_malha_largura_altura_e_norte_para_cima():
    ms, (w, h) = bs.projetar_malha(malha_tres())
    assert w == 1000.0
    # lon -45,0..-42,5 (2,5 graus) e lat -8,5..-4,0 (4,5 graus); a largura em km encolhe com cos(lat central)
    k = np.cos(np.radians((-8.5 + -4.0) / 2))
    assert h == pytest.approx(4.5 * 1000.0 / (2.5 * k), rel=1e-4)
    por = {m["cod"]: m for m in ms}
    assert por[A]["lat"] > por[C]["lat"] > por[B]["lat"]
    ys = {c: float(re.findall(r"-?\d+(?:\.\d+)? (-?\d+(?:\.\d+)?)", por[c]["d"])[0]) for c in (A, B)}
    assert ys[A] < ys[B]                       # y cresce para baixo: o norte tem y menor


def test_projetar_malha_caminho_compacto_e_fechado():
    ms, _ = bs.projetar_malha(malha_tres())
    a = next(m for m in ms if m["cod"] == A)["d"]
    assert a.startswith("M") and a.endswith("Z") and a.count("M") == 1 and a.count("L") == 1
    assert len(re.findall(r"-?\d+(?:\.\d+)? -?\d+(?:\.\d+)?", a)) == 4      # o ponto de fechamento repetido saiu


def test_projetar_malha_multipoligono_com_furo_tem_tres_aneis():
    ms, _ = bs.projetar_malha(malha_tres())
    c = next(m for m in ms if m["cod"] == C)
    assert c["d"].count("M") == 3 and c["nome"] == "Tanque do Piauí"


def test_projetar_malha_ignora_pontos_repetidos():
    anel = [[0, 0], [0, 0], [1, 0], [1, 0], [1, 1], [0, 1], [0, 0]]
    g = {"type": "FeatureCollection", "features": [feature(A, "x", {"type": "Polygon", "coordinates": [anel]})]}
    d = bs.projetar_malha(g)[0][0]["d"]
    assert len(re.findall(r"-?\d+(?:\.\d+)? -?\d+(?:\.\d+)?", d)) == 4


def test_projetar_malha_vazia_ou_geometria_ruim_e_erro():
    with pytest.raises(bs.SiteError, match="sem municípios"):
        bs.projetar_malha({"type": "FeatureCollection", "features": []})
    ponto = {"type": "FeatureCollection", "features": [feature(A, "x", {"type": "Point", "coordinates": [0, 0]})]}
    with pytest.raises(bs.SiteError, match="não suportada"):
        bs.projetar_malha(ponto)
    linha = {"type": "FeatureCollection", "features": [
        feature(A, "x", {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [0, 0]]]}),
        feature(B, "y", {"type": "Polygon", "coordinates": [quadrado(0, 2)]})]}
    with pytest.raises(bs.SiteError, match=f"{A} sem polígono válido"):
        bs.projetar_malha(linha)


# ================================================================== grade
def test_grade_exata_ok_e_reindexa(insumos):
    g = bs._grade(insumos.raw, [A, B, C], MESES, ["precip_mm"], "RAW")
    assert len(g) == 3 * NT and g.index.names == ["cod_ibge", "mes"]
    assert g.index.get_level_values(0)[0] == A


def test_grade_linha_faltando_sobrando_ou_repetida_e_erro(insumos):
    raw = insumos.raw
    with pytest.raises(bs.SiteError, match="grade"):
        bs._grade(raw.iloc[1:], [A, B, C], MESES, ["precip_mm"], "RAW")
    with pytest.raises(bs.SiteError, match="grade"):
        bs._grade(pd.concat([raw, raw.iloc[:1]]), [A, B, C], MESES, ["precip_mm"], "RAW")
    with pytest.raises(bs.SiteError, match="grade"):
        bs._grade(raw, [A, B], MESES, ["precip_mm"], "RAW")             # município a mais no insumo


# ================================================================== pacote de dados
def test_chaves_e_tamanhos_do_pacote(payload, cfg):
    esperadas = {"titulo", "equipe", "maturidade", "viewBox", "escalas", "escala_alvo", "limites", "classes", "meses",
                 "municipios", "ordem", "spi", "cls", "baixa", "precip", "escala_spi", "escala_chuva", "fonte",
                 "periodo", "n_excluidos", "verificacao", "regra_baixa"}
    assert set(payload) == esperadas
    assert payload["escalas"] == [3, 6] and payload["escala_alvo"] == 3
    assert len(payload["meses"]) == NT and payload["meses"][0] == "1981-01" and payload["meses"][-1] == "2026-09"
    assert [m["cod"] for m in payload["municipios"]] == [A, B, C]
    for k in ("3", "6"):
        assert len(decodificar(payload["spi"][k], "<i2")) == 3 * NT
        assert len(decodificar(payload["cls"][k], "uint8")) == 3 * NT
        assert len(decodificar(payload["baixa"][k], "uint8")) == 3 * NT
    assert len(decodificar(payload["precip"], "<i2")) == 3 * NT
    assert payload["maturidade"] == cfg.projeto.maturidade
    json.dumps(payload, allow_nan=False)                                  # JSON puro, sem NaN


def test_ordem_e_norte_para_sul(payload):
    ordem = payload["ordem"]
    assert sorted(ordem) == [0, 1, 2]
    assert [payload["municipios"][i]["cod"] for i in ordem] == [A, C, B]


def test_spi_decodificado_bate_com_o_demo(payload, insumos):
    d = insumos.demo.set_index(["cod_ibge", "mes"]).sort_index()
    q = decodificar(payload["spi"]["3"], "<i2").reshape(3, NT)
    for i, cod in enumerate([A, B, C]):
        sub = d.loc[cod]
        esperado = np.where(sub["spi3_invalid"].to_numpy(), np.nan, sub["spi3"].to_numpy("float64"))
        ok = ~np.isnan(esperado)
        assert (q[i][~ok] == bs.NA16).all() and (q[i][ok] != bs.NA16).all()
        assert np.allclose(q[i][ok] / 100, esperado[ok], atol=0.0051)


def test_classe_bate_com_o_numero_mostrado(payload):
    q = decodificar(payload["spi"]["3"], "<i2")
    c = decodificar(payload["cls"]["3"], "uint8")
    sem = q == bs.NA16
    assert (c[sem] == bs.SEM_DADO).all()
    assert (c[~sem] == bs.classificar(q[~sem] / 100)).all()
    assert ((q[~sem] <= -100) == np.isin(c[~sem], [0, 1, 2])).all()      # seca <=> SPI exibido <= -1,00


def test_chuva_decodificada_bate_com_o_raw(payload, insumos):
    r = insumos.raw.set_index(["cod_ibge", "mes"]).sort_index()["precip_mm"].to_numpy("float64")
    q = decodificar(payload["precip"], "<i2")
    assert np.allclose(q / 10, r, atol=0.051)


def test_sem_spi_nos_meses_iniciais_da_janela(payload):
    q = decodificar(payload["spi"]["6"], "<i2").reshape(3, NT)
    assert (q[:, :5] == bs.NA16).all() and (q[:, 5:6] != bs.NA16).any()   # SPI-6 só existe a partir do 6º mês


def test_periodo_e_verificacao(payload):
    assert payload["periodo"] == {"inicio": "1981-01", "fim": "2026-09", "treino_fim": "2015-12", "teste_fim": "2025-12"}
    assert payload["verificacao"] == {"dataset_sha256": "ab" * 32, "manifest_gerado_em": "2026-11-19T12:30:45Z",
                                      "verificado_em": "2026-11-19T15:00:00Z", "arquivos": 2}
    assert payload["fonte"] == {"chirps": "UCSB-CHG/CHIRPS/DAILY", "versao_params": "0.3.1", "malha_ano": 2022,
                                "extraido_em": ["2026-11-18T23:10:00Z", "2026-11-19T00:40:00Z"]}


def test_municipio_excluido_fica_sem_spi_com_motivo(insumos, cfg):
    q = insumos.qualidade.copy()
    q.loc[q["cod_ibge"] == B, ["excluded", "motivo"]] = [True, "cobertura no treino 0.500 < 0.9"]
    demo = insumos.demo.copy()
    for k in (3, 6):
        m = demo["cod_ibge"] == B
        demo.loc[m, f"spi{k}"] = np.nan
        demo.loc[m, f"spi{k}_invalid"] = True
    p = bs.montar_payload(bs.Insumos(demo, q, insumos.raw, insumos.malha, insumos.manifest, None), cfg, verificado_em=VERIFICADO)
    b = next(m for m in p["municipios"] if m["cod"] == B)
    assert b["exc"] is True and "cobertura" in b["motivo"] and p["n_excluidos"] == 1
    assert (decodificar(p["cls"]["3"], "uint8").reshape(3, NT)[1] == bs.SEM_DADO).all()
    assert p["fonte"]["extraido_em"] == []


def test_baixa_confianca_nao_vale_onde_nao_ha_spi(payload):
    b = decodificar(payload["baixa"]["3"], "uint8")
    c = decodificar(payload["cls"]["3"], "uint8")
    assert (b[c == bs.SEM_DADO] == 0).all()


def com_demo(insumos, demo=None, qualidade=None):
    return bs.Insumos(insumos.demo if demo is None else demo, insumos.qualidade if qualidade is None else qualidade,
                      insumos.raw, insumos.malha, insumos.manifest, None)


def test_spi_marcado_como_invalido_vira_sem_dado_mesmo_com_valor(insumos, cfg):
    demo = insumos.demo.copy()
    alvo = (demo["cod_ibge"] == A) & (demo["mes"] == pd.Timestamp("2010-06-01"))
    assert alvo.sum() == 1 and not np.isnan(demo.loc[alvo, "spi3"].iloc[0])
    demo.loc[alvo, "spi3_invalid"] = True
    p = bs.montar_payload(com_demo(insumos, demo), cfg, verificado_em=VERIFICADO)
    t = MESES.get_loc(pd.Period("2010-06", "M"))
    assert decodificar(p["spi"]["3"], "<i2").reshape(3, NT)[0, t] == bs.NA16
    assert decodificar(p["cls"]["3"], "uint8").reshape(3, NT)[0, t] == bs.SEM_DADO


def test_baixa_confianca_marcada_em_mes_sem_spi_e_descartada(insumos, cfg):
    demo = insumos.demo.copy()
    alvo = (demo["cod_ibge"] == A) & (demo["mes"] == pd.Timestamp("1981-01-01"))
    assert np.isnan(demo.loc[alvo, "spi3"].iloc[0])
    demo.loc[alvo, "spi3_baixa_conf"] = True
    p = bs.montar_payload(com_demo(insumos, demo), cfg, verificado_em=VERIFICADO)
    assert decodificar(p["baixa"]["3"], "uint8").reshape(3, NT)[0, 0] == 0


def test_qualidade_com_outros_municipios_e_erro(insumos, cfg):
    q = insumos.qualidade
    with pytest.raises(bs.SiteError, match="qualidade_municipio"):
        bs.montar_payload(com_demo(insumos, qualidade=q[q["cod_ibge"] != C]), cfg, verificado_em=VERIFICADO)


def test_demo_com_linha_duplicada_ou_municipio_de_fora_e_erro_mesmo_com_o_mesmo_tamanho(insumos, cfg):
    demo = insumos.demo.copy()
    dup = demo.copy()
    dup.loc[dup.index[1], "mes"] = dup.loc[dup.index[0], "mes"]           # mesmo tamanho, mês repetido
    with pytest.raises(bs.SiteError, match="grade"):
        bs.montar_payload(com_demo(insumos, dup), cfg, verificado_em=VERIFICADO)
    fora = demo.copy()
    fora.loc[fora.index[-1], "cod_ibge"] = "2299999"                        # mesmo tamanho, município estranho
    with pytest.raises(bs.SiteError, match="grade"):
        bs.montar_payload(com_demo(insumos, fora), cfg, verificado_em=VERIFICADO)


def test_pacote_com_malha_e_demo_diferentes_e_erro(insumos, cfg):
    malha = malha_tres()
    malha["features"].append(feature("2200004", "Gama", {"type": "Polygon", "coordinates": [quadrado(-42, -5)]}))
    with pytest.raises(bs.SiteError):
        bs.montar_payload(bs.Insumos(insumos.demo, insumos.qualidade, insumos.raw, malha, insumos.manifest, None),
                          cfg, verificado_em=VERIFICADO)


def test_codigos_repetidos_na_malha_e_erro(insumos, cfg):
    malha = malha_tres()
    malha["features"].append(feature(A, "de novo", {"type": "Polygon", "coordinates": [quadrado(-42, -5)]}))
    with pytest.raises(bs.SiteError, match="repetidos"):
        bs.montar_payload(bs.Insumos(insumos.demo, insumos.qualidade, insumos.raw, malha, insumos.manifest, None),
                          cfg, verificado_em=VERIFICADO)


# ================================================================== template e HTML
def test_template_tem_os_tres_marcadores_uma_vez():
    t = TEMPLATE.read_text(encoding="utf-8")
    for m in ("__DADOS__", "__FONTE__", "__TITULO__"):
        assert t.count(m) == 1


def test_template_sem_recurso_externo():
    t = TEMPLATE.read_text(encoding="utf-8")
    urls = set(re.findall(r"https?://[^\s\"')<]+", t))
    assert urls <= {"http://www.w3.org/2000/svg"}, urls
    assert not re.search(r"<script[^>]+\bsrc=|<link\b|@import|\bfetch\(|XMLHttpRequest|navigator\.sendBeacon", t)
    assert "localStorage" not in t and "sessionStorage" not in t


def test_template_leva_os_avisos_e_as_camadas_indisponiveis():
    t = TEMPLATE.read_text(encoding="utf-8")
    for texto in ("MVP experimental", "Não substitui alerta oficial", "Previsão do próximo mês", "Indisponível",
                  "Antecedência em relação aos decretos", "Plano de Contingência", "in-sample",
                  "fora de treino, validação e teste", "Mede falta de chuva, não prova que um açude vai secar"):
        assert texto in t, texto


def test_regra_de_baixa_confianca_vai_para_o_pacote_como_esta_no_params(payload, cfg):
    assert payload["regra_baixa"] == {"q_max": 0.5, "media_min_mm_por_mes": 10.0}
    assert payload["regra_baixa"] == {"q_max": cfg.spi.q_baixa_confianca, "media_min_mm_por_mes": cfg.spi.media_min_mm_por_mes}


def test_regra_de_baixa_confianca_segue_o_params_alterado(insumos, cfg):
    outro = cfg.model_copy(update={"spi": cfg.spi.model_copy(update={"media_min_mm_por_mes": 0.0, "q_baixa_confianca": 0.7})})
    p = bs.montar_payload(insumos, outro, verificado_em=VERIFICADO)
    assert p["regra_baixa"] == {"q_max": 0.7, "media_min_mm_por_mes": 0.0}


def test_template_sinaliza_baixa_confianca_no_mapa_na_matriz_e_na_ficha():
    t = TEMPLATE.read_text(encoding="utf-8")
    for texto in ("pontilhado", "esmaecido = baixa confiança", "Meses de baixa confiança",
                  "Baixa confiança: pouca chuva esperada nesta época do ano",
                  "Pouca chuva esperada nesta época do ano: poucos milímetros mudam muito o SPI, que informa pouco.",
                  "Esses meses continuam nas contagens do mapa"):
        assert texto in t, texto
    for antigo in ("muitos zeros no histórico", "mais da metade dos anos de referência teve chuva zero"):
        assert antigo not in t, antigo
    assert 'id="pontilhado"' in t and 'id="gBc"' in t and "BX[k][m * NT + t] ? 105 : 255" in t


def test_template_nao_promete_previsao():
    t = TEMPLATE.read_text(encoding="utf-8").lower()
    assert "probabilidade de seca" not in t and "vai haver seca" not in t and "previsão para" not in t


def test_ids_usados_pelo_script_existem_no_html():
    t = TEMPLATE.read_text(encoding="utf-8")
    usados = set(re.findall(r"\$\('([A-Za-z0-9_]+)'\)", t)) | set(re.findall(r"getElementById\('([A-Za-z0-9_]+)'\)", t))
    ids = set(re.findall(r"\bid=\"([A-Za-z0-9_]+)\"", t.split("<script>")[0]))
    assert usados - ids == set(), usados - ids


@pytest.mark.skipif(shutil.which("node") is None, reason="node não instalado")
def test_script_do_template_tem_sintaxe_valida(tmp_path):
    t = TEMPLATE.read_text(encoding="utf-8")
    js = t.rsplit("<script>", 1)[1].split("</script>")[0]
    f = tmp_path / "painel.js"
    f.write_text(js, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_renderizar_troca_marcadores_e_escapa_fim_de_script(payload):
    p = dict(payload)
    p["titulo"] = "Piauí </script><b>"
    t = "<title>__TITULO__</title><style>@font-face{src:url(data:font/woff2;base64,__FONTE__)}</style><script>__DADOS__</script>"
    html = bs.renderizar(p, t, b"\x00\x01\x02")
    assert "AAEC" in html and "__" not in html.replace("__proto__", "")
    corpo = html.split("<script>")[1]
    assert corpo.count("</script>") == 1                     # o "</" de dentro do JSON saiu escapado
    assert json.loads(corpo.split("</script>")[0])["titulo"] == "Piauí </script><b>"


@pytest.mark.parametrize("quebrar", ["__DADOS__", "__FONTE__", "__TITULO__"])
def test_renderizar_sem_marcador_ou_com_marcador_repetido_e_erro(payload, quebrar):
    base = "__DADOS__ __FONTE__ __TITULO__"
    with pytest.raises(bs.SiteError, match=quebrar):
        bs.renderizar(payload, base.replace(quebrar, ""), b"x")
    with pytest.raises(bs.SiteError, match=quebrar):
        bs.renderizar(payload, base + " " + quebrar, b"x")


def test_renderizar_recusa_nan(payload):
    p = dict(payload)
    p["viewBox"] = [1000.0, float("nan")]
    with pytest.raises(ValueError):
        bs.renderizar(p, "__DADOS__ __FONTE__ __TITULO__", b"x")


def test_fonte_embutida_e_woff2_com_licenca_ao_lado():
    assert bs.FONTE.read_bytes()[:4] == b"wOF2"
    assert (bs.FONTE.parent / "AtkinsonHyperlegibleNext-OFL.txt").is_file()


# ================================================================== higiene do módulo
def test_modulo_nao_depende_de_web_sql_nem_de_ml_e_nao_inventa_dados():
    src = (RAIZ / "src" / "web" / "build_site.py").read_text(encoding="utf-8")
    raiz_imports = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            raiz_imports |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            raiz_imports.add(n.module.split(".")[0])
    assert not raiz_imports & {"fastapi", "flask", "sqlite3", "sqlalchemy", "torch", "sklearn", "requests", "urllib"}
    assert "default_rng" not in src and "np.random" not in src and "random." not in src


# ================================================================== ponta a ponta (projeto de mentira)
def montar_proj(tmp_path, raw):
    shutil.copy(PARAMS_REAL, tmp_path / "params.yaml")
    ref = tmp_path / "data" / "reference"
    ref.mkdir(parents=True)
    (ref / "referencia.sha256.json").write_text(json.dumps(
        {"conteudo_sha256": "ab" * 32, "contagens": {"malha": 224, "escopo": 3},
         "entradas": {"PI_Municipios_2022.zip": "cd" * 32}, "parametros": {"incluir_transicao": True}}), encoding="utf-8")
    (ref / "municipios_semiarido.csv").write_bytes(
        "cod_ibge,nome,uf,situacao,area_km2\n2200001,Aaa,PI,res150,10.0\n".encode("utf-8"))
    pd.DataFrame({"CD_MUN": [A], "geometry": [b"wkb"]}).to_parquet(ref / "malha_pi_2022.parquet", index=False)
    (ref / "malha_pi_2022_gee.geojson").write_text(json.dumps(malha_tres()), encoding="utf-8")
    cfg = load_config(tmp_path / "params.yaml")
    escrever_raw(tmp_path, raw, (A, B, C))
    bd.construir_demo(cfg, municipios=(A, B, C), agora=AGORA)
    return cfg, tmp_path


@pytest.fixture
def proj(tmp_path):
    return montar_proj(tmp_path, raw_sintetico())


@pytest.fixture
def proj_com_excluido(tmp_path):
    """B sem nenhum dado de satélite de 1981 a 2012: pela regra de qualidade do treino, B é excluído."""
    raw = raw_sintetico()
    m = ((raw["cod_ibge"] == B) & (raw["mes"] <= pd.Timestamp("2012-12-01"))).to_numpy()
    raw.loc[m, ["precip_mm", "precip_max_dia_mm", "dias_chuva"]] = np.nan
    raw.loc[m, "dias_validos"] = 0
    raw.loc[m, "pixels_validos_frac"] = np.float32(0.0)
    return montar_proj(tmp_path, raw)


def extrair(html: str) -> dict:
    m = re.search(r'<script id="dados" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


def test_ponta_a_ponta_grava_o_html_com_o_hash_do_manifest(proj):
    cfg, raiz = proj
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    destino = raiz / "site" / "piaui_resiliente.html"
    assert Path(r["arquivo"]) == destino and destino.is_file()
    assert r["municipios"] == 3 and r["meses"] == NT and r["excluidos"] == 0
    html = destino.read_text(encoding="utf-8")
    d = extrair(html)
    m = json.loads((raiz / "data" / "curated" / "manifest.json").read_text(encoding="utf-8"))
    assert d["verificacao"]["dataset_sha256"] == m["dataset_sha256"] == r["dataset_sha256"]
    assert d["verificacao"]["verificado_em"] == "2026-11-19T15:00:00Z"
    assert d["fonte"]["extraido_em"] == ["2026-11-18T23:10:00+00:00", "2026-11-19T00:40:00+00:00"]   # do mais antigo ao mais novo
    assert not list((raiz / "site").glob("*.tmp"))
    assert "MVP experimental" in html and "Indisponível" in html and "__DADOS__" not in html


def test_ponta_a_ponta_municipio_excluido_aparece_sem_spi_e_com_motivo(proj_com_excluido):
    cfg, raiz = proj_com_excluido
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["excluidos"] == 1
    d = extrair((raiz / "site" / "piaui_resiliente.html").read_text(encoding="utf-8"))
    por = {m["cod"]: m for m in d["municipios"]}
    assert por[B]["exc"] is True and por[B]["motivo"].startswith("cobertura no treino")
    assert por[A]["exc"] is False and por[A]["motivo"] == "" and d["n_excluidos"] == 1
    i = [m["cod"] for m in d["municipios"]].index(B)
    for k in ("3", "6"):
        assert (decodificar(d["spi"][k], "<i2").reshape(3, NT)[i] == bs.NA16).all()
        assert (decodificar(d["cls"][k], "uint8").reshape(3, NT)[i] == bs.SEM_DADO).all()
    assert (decodificar(d["precip"], "<i2").reshape(3, NT)[i][:384] == bs.NA16).all()   # sem satélite até 2012-12


def test_ponta_a_ponta_saida_escolhida(proj, tmp_path_factory):
    cfg, raiz = proj
    alvo = tmp_path_factory.mktemp("fora") / "a" / "b.html"
    r = bs.gerar_site(cfg, alvo, agora=VERIFICADO)
    assert Path(r["arquivo"]) == alvo and alvo.is_file() and not (raiz / "site").exists()


def test_ponta_a_ponta_gerar_duas_vezes_da_o_mesmo_arquivo(proj):
    cfg, raiz = proj
    bs.gerar_site(cfg, agora=VERIFICADO)
    a = (raiz / "site" / "piaui_resiliente.html").read_bytes()
    bs.gerar_site(cfg, agora=VERIFICADO)
    assert (raiz / "site" / "piaui_resiliente.html").read_bytes() == a


@pytest.mark.parametrize("alvo", ["data/reference/malha_pi_2022_gee.geojson", "demo_observado"])
def test_arquivo_alterado_depois_do_manifest_nao_gera_site(proj, alvo):
    cfg, raiz = proj
    p = raiz / alvo if alvo.endswith(".geojson") else bd.destino(cfg, alvo)
    p.write_bytes(p.read_bytes() + b" ")
    with pytest.raises(mf.ManifestIntegridadeError):
        bs.gerar_site(cfg, agora=VERIFICADO)
    assert not (raiz / "site").exists()


def test_lote_do_raw_alterado_nao_gera_site(proj):
    cfg, raiz = proj
    lote = sorted(chirps_raw.diretorio_raw(cfg).glob(f"{chirps_raw._PREFIXO}*.parquet"))[0]
    lote.write_bytes(lote.read_bytes() + b"\x00")
    with pytest.raises(mf.ManifestIntegridadeError):
        bs.gerar_site(cfg, agora=VERIFICADO)


def test_sem_manifest_nao_gera_site(proj):
    cfg, raiz = proj
    for p in raiz.rglob("manifest*.json"):
        p.unlink()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        bs.gerar_site(cfg, agora=VERIFICADO)
    assert not (raiz / "site").exists()


def test_params_alterado_depois_do_manifest_nao_gera_site(proj):
    cfg, raiz = proj
    p = raiz / "params.yaml"
    p.write_text(p.read_text(encoding="utf-8") + "\n# mexi\n", encoding="utf-8")
    with pytest.raises(mf.ManifestIntegridadeError):
        bs.gerar_site(load_config(p), agora=VERIFICADO)


def test_cli_ok_imprime_aviso_e_resumo(proj, capsys):
    cfg, raiz = proj
    assert bs.main(["--params", str(raiz / "params.yaml"), "--saida", str(raiz / "x.html")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("AVISO: MVP experimental") and "site gravado:" in out and "municípios=3" in out
    assert (raiz / "x.html").is_file()


def test_cli_com_erro_devolve_1_e_nao_grava(proj, capsys):
    cfg, raiz = proj
    p = bd.destino(cfg, "demo_observado")
    p.write_bytes(p.read_bytes() + b" ")
    assert bs.main(["--params", str(raiz / "params.yaml"), "--saida", str(raiz / "x.html")]) == 1
    assert "ERRO:" in capsys.readouterr().err and not (raiz / "x.html").exists()


def test_template_conta_so_meses_confiaveis_na_ficha_e_declara_a_regra_no_rodape():
    t = TEMPLATE.read_text(encoding="utf-8")
    for texto in ("Meses em seca (SPI ≤ −1,0, só meses confiáveis):", "Mês mais seco (só meses confiáveis):",
                  "deles com baixa confiança", "D.regra_baixa", "R.media_min_mm_por_mes", "R.q_max",
                  "${regraBc} Esses meses continuam"):
        assert texto in t, texto
    # o mês de baixa confiança sai do "mês mais seco" e das contagens de seca: o `continue` vem antes de elas serem tocadas
    i = t.index("total++; if (BX[k][i * NT + t]) { baixas++; continue; }")
    assert t.index("secos++", i) > i and t.index("pior = x", i) > i
