"""Testes de src/web/build_site.py (Item 11). Sem internet; dados sintéticos só em tests/ (dados_sinteticos.py)."""
import ast
import base64
import dataclasses
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dados_sinteticos import (A, AGORA, B, C, MESES, NT, PARAMS_REAL, RAIZ, VERIFICADO, escrever_raw, feature,
                              malha_tres, quadrado, raw_sintetico)
from src.data import build_demo as bd
from src.data import chirps_raw, contexto_enso as ce, manifest as mf
from src.data.config import load_config
from src.web import build_site as bs
from src.web import destaques, paginas


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
        assert re.fullmatch(r"#[0-9A-Fa-f]{6}", c[3]) and re.fullmatch(r"#[0-9A-Fa-f]{6}", c[4])


def test_rampas_partem_das_cores_da_marca():
    from src.web import marca
    assert bs.CLASSES[1][3] == marca.TERRACOTA and bs.CLASSES[5][3] == marca.PETROLEO     # falta = terracota; excesso = petróleo
    seca, umido = bs.CLASSES[:3], bs.CLASSES[4:]
    assert all(c[5] for c in seca) and not any(c[5] for c in umido)


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
                 "periodo", "n_excluidos", "verificacao", "regra_baixa", "n_municipios", "abertura", "pct_baixa_ultimo",
                 "maiores_secas", "clima", "enso", "planejamento", "previsao", "conferencia", "tem_previsao"}
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


# ================================================================== regra de baixa confiança
def test_regra_de_baixa_confianca_vai_para_o_pacote_como_esta_no_params(payload, cfg):
    assert payload["regra_baixa"] == {"q_max": 0.5, "media_min_mm_por_mes": 10.0}
    assert payload["regra_baixa"] == {"q_max": cfg.spi.q_baixa_confianca, "media_min_mm_por_mes": cfg.spi.media_min_mm_por_mes}


def test_regra_de_baixa_confianca_segue_o_params_alterado(insumos, cfg):
    outro = cfg.model_copy(update={"spi": cfg.spi.model_copy(update={"media_min_mm_por_mes": 0.0, "q_baixa_confianca": 0.7})})
    p = bs.montar_payload(insumos, outro, verificado_em=VERIFICADO)
    assert p["regra_baixa"] == {"q_max": 0.7, "media_min_mm_por_mes": 0.0}


# ================================================================== destaques, clima e El Niño no pacote
def test_destaques_do_pacote_vem_dos_mesmos_arrays_que_a_tela_decodifica(payload):
    assert payload["n_municipios"] == 3
    cls = decodificar(payload["cls"]["3"], "uint8").reshape(3, NT)
    bx = decodificar(payload["baixa"]["3"], "uint8").reshape(3, NT)
    assert payload["abertura"] == destaques.mes_abertura(cls, bx) and 0 <= payload["abertura"] < NT
    ultimo = cls[:, -1] != bs.SEM_DADO
    assert payload["pct_baixa_ultimo"] == round(100 * bx[:, -1][ultimo].sum() / ultimo.sum())
    assert payload["maiores_secas"] == [{**x, "mes": payload["meses"][x["t"]]} for x in destaques.maiores_secas(cls, bx)]
    for x in payload["maiores_secas"]:
        assert x["n_seca"] <= x["n_confiaveis"] <= x["n_validos"] <= 3 and x["mes"] <= payload["periodo"]["fim"]


def test_clima_do_pacote_tem_forma_e_nao_usa_chuva_depois_do_treino(payload, insumos, cfg):
    c = payload["clima"]
    assert c["treino_fim"] == payload["periodo"]["treino_fim"]
    assert len(c["climatologia"]) == 12 and all(v >= 0 for v in c["climatologia"])
    est = c["estacao"]
    assert est["largura"] == 4 and len(est["meses"]) == 4 and 0 < c["pct_estacao"] <= 100
    # mexer na chuva DEPOIS do treino muda os anos mostrados, mas não a média de referência (sem vazamento)
    raw = insumos.raw.copy()
    depois = raw["mes"] > pd.Timestamp("2015-12-01")
    raw.loc[depois, "precip_mm"] = raw.loc[depois, "precip_mm"] * np.float32(3.0)
    p2 = bs.montar_payload(dataclasses.replace(insumos, raw=raw), cfg, verificado_em=VERIFICADO)
    assert p2["clima"]["climatologia"] == c["climatologia"] and p2["clima"]["estacao"]["media_mm"] == est["media_mm"]
    assert p2["clima"]["estacao"]["anos"] != est["anos"] and p2["precip"] != payload["precip"]


def test_enso_ausente_vai_nulo_e_presente_vai_validado(payload, insumos, cfg):
    assert payload["enso"] is None
    ctx = ce.carregar(RAIZ / ce.ARQUIVO_PADRAO)
    p = bs.montar_payload(dataclasses.replace(insumos, enso=ctx), cfg, verificado_em=VERIFICADO)
    assert p["enso"] == ce.para_pacote(ctx) and p["enso"]["fontes"]
    json.dumps(p, allow_nan=False)


# ================================================================== pasta do site
def test_fontes_embutidas_sao_woff2_com_licenca_ao_lado():
    for fonte, licenca in ((bs.FONTE_CORPO, "AtkinsonHyperlegibleNext-OFL.txt"), (bs.FONTE_MARCA, "Outfit-OFL.txt")):
        assert fonte.read_bytes()[:4] == b"wOF2" and (fonte.parent / licenca).is_file()


def test_css_recebe_as_duas_fontes_e_nenhum_marcador_sobra():
    css = bs._css(b"\x00\x01\x02", b"\x03\x04\x05")
    assert "AAEC" in css and "AwQF" in css and "__FONTE_" not in css


@pytest.mark.parametrize("quebrar", ["__FONTE_CORPO__", "__FONTE_MARCA__"])
def test_css_sem_marcador_ou_com_marcador_repetido_e_erro(quebrar, tmp_path, monkeypatch):
    base = "a{b:url(__FONTE_CORPO__)} c{d:url(__FONTE_MARCA__)}"
    monkeypatch.setattr(bs, "ASSETS", tmp_path)
    (tmp_path / "site.css").write_text(base.replace(quebrar, ""), encoding="utf-8")
    with pytest.raises(bs.SiteError, match=quebrar):
        bs._css(b"x", b"y")
    (tmp_path / "site.css").write_text(base + " " + quebrar, encoding="utf-8")
    with pytest.raises(bs.SiteError, match=quebrar):
        bs._css(b"x", b"y")


def test_dados_js_e_json_puro_numa_atribuicao_e_recusa_nan(payload):
    p = dict(payload)
    p["titulo"] = "Piauí </script> \u2028 fim"
    js = bs._dados_js(p)
    assert js.startswith("window.PR_DADOS=") and js.endswith(";\n") and "\u2028" not in js and "\u2029" not in js
    assert json.loads(js[len("window.PR_DADOS="):-2])["titulo"] == p["titulo"]      # ida e volta sem perder nada
    p["viewBox"] = [1000.0, float("nan")]
    with pytest.raises(ValueError):
        bs._dados_js(p)


def test_montar_arquivos_tem_todas_as_paginas_e_so_os_assets_usados(payload):
    a = bs.montar_arquivos(payload)
    paginas_esperadas = {p.arquivo for p in paginas.PAGINAS}
    assert paginas_esperadas == {"index.html", "monitor.html", "municipios.html", "el-nino.html", "metodo.html", "sobre.html"}
    assets = {k for k in a if k.startswith("assets/")}
    usados = {f"assets/{s}.js" for p in paginas.PAGINAS for s in p.scripts} | {"assets/core.js"}
    assert assets == usados | {"assets/dados.js", "assets/site.css", "assets/marca/gota.svg"}
    assert set(a) - assets == paginas_esperadas
    assert all(isinstance(t, str) and t for t in a.values())


# ================================================================== higiene do módulo
@pytest.mark.parametrize("modulo", ["build_site.py", "paginas.py", "marca.py", "destaques.py"])
def test_modulo_nao_depende_de_web_sql_nem_de_ml_e_nao_inventa_dados(modulo):
    src = (RAIZ / "src" / "web" / modulo).read_text(encoding="utf-8")
    importados = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            importados |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            importados.add(n.module)
    proibidos = {"fastapi", "flask", "sqlite3", "sqlalchemy", "torch", "sklearn", "requests", "httpx", "urllib3",
                 "urllib.request", "http", "socket"}           # web, SQL, ML e rede; urllib.parse (só texto) pode
    assert not [m for m in importados if any(m == x or m.startswith(x + ".") for x in proibidos)], importados
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


def extrair(pasta: Path) -> dict:
    txt = (pasta / "assets" / "dados.js").read_text(encoding="utf-8")
    assert txt.startswith("window.PR_DADOS=") and txt.endswith(";\n")
    return json.loads(txt[len("window.PR_DADOS="):-2])


def lido(pasta: Path) -> dict[str, bytes]:
    return {str(f.relative_to(pasta)): f.read_bytes() for f in sorted(pasta.rglob("*")) if f.is_file()}


def test_ponta_a_ponta_grava_a_pasta_com_o_hash_do_manifest(proj):
    cfg, raiz = proj
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    destino = raiz / "site"
    assert Path(r["pasta"]) == destino and Path(r["inicio"]) == destino / "index.html" and (destino / "index.html").is_file()
    assert r["municipios"] == 3 and r["meses"] == NT and r["excluidos"] == 0 and r["arquivos"] == len(lido(destino))
    d = extrair(destino)
    m = json.loads((raiz / "data" / "curated" / "manifest.json").read_text(encoding="utf-8"))
    assert d["verificacao"]["dataset_sha256"] == m["dataset_sha256"] == r["dataset_sha256"]
    assert d["verificacao"]["verificado_em"] == "2026-11-19T15:00:00Z"
    assert d["fonte"]["extraido_em"] == ["2026-11-18T23:10:00+00:00", "2026-11-19T00:40:00+00:00"]   # do mais antigo ao mais novo
    assert not list(destino.rglob("*.tmp"))
    for pagina in paginas.PAGINAS:
        html = (destino / pagina.arquivo).read_text(encoding="utf-8")
        assert "MVP experimental" in html and m["dataset_sha256"] in html and "{{" not in html
    if os.name != "nt":
        assert {(f.stat().st_mode & 0o777) for f in destino.rglob("*") if f.is_file()} == {0o644}   # servível por quem não é o dono


def test_ponta_a_ponta_sem_arquivo_do_el_nino_a_camada_aparece_como_indisponivel(proj):
    cfg, raiz = proj
    assert not (raiz / ce.ARQUIVO_PADRAO).exists()
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["enso"] is False and extrair(raiz / "site")["enso"] is None
    html = (raiz / "site" / "el-nino.html").read_text(encoding="utf-8")
    assert "indisponível" in html.lower() and "ensodisc" not in html


def test_ponta_a_ponta_com_arquivo_do_el_nino_leva_as_fontes_e_a_data(proj):
    cfg, raiz = proj
    destino_ref = raiz / ce.ARQUIVO_PADRAO
    destino_ref.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(RAIZ / ce.ARQUIVO_PADRAO, destino_ref)
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["enso"] is True
    d = extrair(raiz / "site")
    assert d["enso"] == ce.para_pacote(ce.carregar(destino_ref))
    html = (raiz / "site" / "el-nino.html").read_text(encoding="utf-8")
    assert "cpc.ncep.noaa.gov" in html and "portal.inmet.gov.br" in html


def test_ponta_a_ponta_arquivo_do_el_nino_invalido_nao_gera_site(proj):
    cfg, raiz = proj
    ref = raiz / ce.ARQUIVO_PADRAO
    ref.parent.mkdir(parents=True, exist_ok=True)
    ref.write_text('{"versao": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        bs.gerar_site(cfg, agora=VERIFICADO)
    assert not (raiz / "site").exists()


def test_ponta_a_ponta_municipio_excluido_aparece_sem_spi_e_com_motivo(proj_com_excluido):
    cfg, raiz = proj_com_excluido
    r = bs.gerar_site(cfg, agora=VERIFICADO)
    assert r["excluidos"] == 1
    d = extrair(raiz / "site")
    por = {m["cod"]: m for m in d["municipios"]}
    assert por[B]["exc"] is True and por[B]["motivo"].startswith("cobertura no treino")
    assert por[A]["exc"] is False and por[A]["motivo"] == "" and d["n_excluidos"] == 1
    i = [m["cod"] for m in d["municipios"]].index(B)
    for k in ("3", "6"):
        assert (decodificar(d["spi"][k], "<i2").reshape(3, NT)[i] == bs.NA16).all()
        assert (decodificar(d["cls"][k], "uint8").reshape(3, NT)[i] == bs.SEM_DADO).all()
    assert (decodificar(d["precip"], "<i2").reshape(3, NT)[i][:384] == bs.NA16).all()   # sem satélite até 2012-12
    # o excluído não entra na média da chuva do clima, que seria puxada para baixo por meses sem satélite
    assert all(v >= 0 for v in d["clima"]["climatologia"])


def test_ponta_a_ponta_saida_escolhida_e_uma_pasta(proj, tmp_path_factory):
    cfg, raiz = proj
    alvo = tmp_path_factory.mktemp("fora") / "a" / "b"
    r = bs.gerar_site(cfg, alvo, agora=VERIFICADO)
    assert Path(r["pasta"]) == alvo and (alvo / "index.html").is_file() and not (raiz / "site").exists()


@pytest.mark.parametrize("nome", ["x.html", "x.HTM"])
def test_saida_com_cara_de_arquivo_html_e_erro_e_nao_grava(proj, tmp_path, nome):
    cfg, raiz = proj
    with pytest.raises(bs.SiteError, match="pasta"):
        bs.gerar_site(cfg, tmp_path / nome, agora=VERIFICADO)
    assert not (tmp_path / nome).exists() and not (raiz / "site").exists()


def test_ponta_a_ponta_gerar_duas_vezes_da_o_mesmo_conteudo(proj):
    cfg, raiz = proj
    bs.gerar_site(cfg, agora=VERIFICADO)
    a = lido(raiz / "site")
    bs.gerar_site(cfg, agora=VERIFICADO)
    assert lido(raiz / "site") == a and len(a) >= 14


def test_erro_em_uma_pagina_nao_deixa_site_pela_metade(proj, monkeypatch):
    cfg, raiz = proj
    original, chamadas = paginas.renderizar_pagina, []

    def quebra_na_terceira(pagina, *args, **kw):
        chamadas.append(pagina.id)
        if len(chamadas) == 3:
            raise paginas.PaginaError("quebrou de propósito")
        return original(pagina, *args, **kw)

    monkeypatch.setattr(paginas, "renderizar_pagina", quebra_na_terceira)
    with pytest.raises(paginas.PaginaError):
        bs.gerar_site(cfg, agora=VERIFICADO)
    assert len(chamadas) == 3 and not (raiz / "site").exists()


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
    assert not (raiz / "site").exists()


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
    assert bs.main(["--params", str(raiz / "params.yaml"), "--saida", str(raiz / "saida")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("AVISO: MVP experimental") and "site gravado em" in out and "municípios=3" in out
    assert f"abra: {raiz / 'saida' / 'index.html'}" in out and "sem data/reference/contexto_enso.json" in out
    assert (raiz / "saida" / "index.html").is_file()


def test_cli_com_erro_devolve_1_e_nao_grava(proj, capsys):
    cfg, raiz = proj
    p = bd.destino(cfg, "demo_observado")
    p.write_bytes(p.read_bytes() + b" ")
    assert bs.main(["--params", str(raiz / "params.yaml"), "--saida", str(raiz / "saida")]) == 1
    assert "ERRO:" in capsys.readouterr().err and not (raiz / "saida").exists()
