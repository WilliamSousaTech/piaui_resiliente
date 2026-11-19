"""Testes de src/data/diag_spi.py. Sem internet; dados sintéticos só aqui (nunca em data/ nem na saída real).

Janela real do params.yaml: 1981-01..2026-09 (549 meses), treino até 2015-12. Três municípios de mentira.
Os números do diagnóstico são conferidos por contas independentes (pandas/scipy), não pela mesma função.
"""
import ast
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.data import build_demo as bd
from src.data import chirps_raw, diag_spi as dg
from src.data.config import load_config

RAIZ = Path(__file__).resolve().parents[1]
PARAMS_REAL = RAIZ / "params.yaml"
MESES = pd.period_range("1981-01", "2026-09", freq="M")
NT = len(MESES)
A, B, C = "2200001", "2200002", "2200003"
ALVO = pd.Period("2026-09", "M")
AGORA_TESTE = pd.Timestamp("2026-11-19T12:30:45Z").to_pydatetime()


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


@pytest.fixture(scope="module")
def cfg():
    return load_config(PARAMS_REAL)


@pytest.fixture(scope="module")
def raw_base():
    return raw_sintetico()


@pytest.fixture(scope="module")
def demo_res(cfg, raw_base):
    return bd.montar_demo(raw_base, cfg)


def diag(raw, demo_res, cfg, *, mes=ALVO, k=3, params=None):
    return dg.calcular_diagnostico(raw, demo_res.demo_observado,
                                   demo_res.spi_gamma_params if params is None else params,
                                   treino_fim=cfg.split.treino_fim, mes=mes, k=k)


def diag_novo(raw, cfg, **kw):
    """Recalcula o demo a partir do RAW alterado (como o build real faria) e roda o diagnóstico."""
    r = bd.montar_demo(raw, cfg)
    return diag(raw, r, cfg, **kw), r


@pytest.fixture(scope="module")
def base(raw_base, demo_res, cfg):
    return diag(raw_base, demo_res, cfg)


def linha(r, cod):
    return r["detalhe"].set_index("cod_ibge").loc[cod]


# ================================================================== contas independentes
def test_soma_da_janela_e_a_soma_dos_tres_ultimos_meses(base, raw_base):
    for cod in (A, B, C):
        p = raw_base[raw_base["cod_ibge"] == cod].set_index("mes")["precip_mm"].astype("float64")
        esperado = p.loc["2026-07-01":"2026-09-01"].sum()
        assert linha(base, cod)["soma_mm"] == pytest.approx(esperado, rel=1e-9)


def test_media_historica_e_so_do_treino_mesmo_mes_calendario(base, raw_base):
    for cod in (A, B, C):
        p = raw_base[raw_base["cod_ibge"] == cod].set_index("mes")["precip_mm"].astype("float64")
        S = p.rolling(3).sum()
        alvo = S[(S.index.month == 9) & (S.index <= "2015-12-01")].dropna()
        assert linha(base, cod)["media_hist_mm"] == pytest.approx(alvo.mean(), rel=1e-9)
        assert linha(base, cod)["p90_mm"] == pytest.approx(np.percentile(alvo, 90), rel=1e-9)


def test_dado_posterior_ao_treino_nao_muda_a_climatologia_nem_o_limiar(raw_base, demo_res, cfg, base):
    """Anti-vazamento: inflar 2016+ não pode mexer em média histórica, p90, q nem nos mm de corte."""
    raw = raw_base.copy()
    raw.loc[raw["mes"] > pd.Timestamp("2015-12-01"), "precip_mm"] *= 5.0
    depois, r2 = diag_novo(raw, cfg)
    pd.testing.assert_frame_equal(r2.spi_gamma_params, demo_res.spi_gamma_params)
    for c in ("media_hist_mm", "p90_mm", "q", "mm_para_umido", "mm_para_seca", "insample_umido", "insample_seca"):
        np.testing.assert_allclose(depois["detalhe"][c], base["detalhe"][c], equal_nan=True)
    assert (depois["detalhe"]["soma_mm"] > base["detalhe"]["soma_mm"]).all()


def _spi_em(demo_res, cod, x, q):
    """SPI de uma janela de x mm, com a Gama do treino da célula (cod, SPI-3, setembro): conta independente."""
    p = demo_res.spi_gamma_params
    r = p[(p["cod_ibge"] == cod) & (p["escala_meses"] == 3) & (p["mes_calendario"] == 9)].iloc[0]
    return stats.norm.ppf(q + (1 - q) * stats.gamma.cdf(x, r["shape"], scale=r["scale"]))


def test_mm_para_umido_inverte_o_spi_em_mais_um(base, demo_res):
    for _, t in base["detalhe"].iterrows():
        assert _spi_em(demo_res, t["cod_ibge"], t["mm_para_umido"], t["q"]) == pytest.approx(1.0, abs=1e-6)


def test_mm_para_seca_inverte_o_spi_em_menos_um(base, demo_res):
    for _, t in base["detalhe"].iterrows():
        assert _spi_em(demo_res, t["cod_ibge"], t["mm_para_seca"], t["q"]) == pytest.approx(-1.0, abs=1e-6)


def test_q_acima_do_percentil_de_mais_um_basta_qualquer_chuva(raw_base, demo_res, cfg):
    p = demo_res.spi_gamma_params.copy()
    m = (p["cod_ibge"] == A) & (p["escala_meses"] == 3) & (p["mes_calendario"] == 9)
    p.loc[m, "q"] = 0.9                                   # H(0) = 0,9 > Phi(1) = 0,841
    r = diag(raw_base, demo_res, cfg, params=p)
    assert linha(r, A)["mm_para_umido"] == 0.0
    assert np.isnan(linha(r, A)["mm_para_seca"])          # com q > Phi(-1), nenhuma chuva leva a SPI <= -1
    assert linha(r, B)["mm_para_umido"] > 0                # os outros não mudam


def test_celula_invalida_nao_gera_limiar(raw_base, demo_res, cfg):
    p = demo_res.spi_gamma_params.copy()
    m = (p["cod_ibge"] == B) & (p["escala_meses"] == 3) & (p["mes_calendario"] == 9)
    p.loc[m, "valido"] = False
    r = diag(raw_base, demo_res, cfg, params=p)
    assert np.isnan(linha(r, B)["mm_para_umido"]) and np.isnan(linha(r, B)["mm_para_seca"])
    assert np.isfinite(linha(r, A)["mm_para_umido"])


def test_fracao_in_sample_bate_com_a_contagem_do_demo(base, demo_res):
    d = demo_res.demo_observado.assign(mes=lambda x: pd.to_datetime(x["mes"]))
    for cod in (A, B, C):
        s = d[(d["cod_ibge"] == cod) & (d["mes"].dt.month == 9) & (d["mes"] <= "2015-12-01")]["spi3"].dropna().round(2)
        assert linha(base, cod)["insample_umido"] == pytest.approx((s >= 1.0).mean(), abs=1e-12)
        assert linha(base, cod)["insample_seca"] == pytest.approx((s <= -1.0).mean(), abs=1e-12)


def test_grupos_vem_do_spi_arredondado_em_duas_casas_como_no_site(base, demo_res):
    d = demo_res.demo_observado.assign(mes=lambda x: pd.to_datetime(x["mes"]))
    s = d[d["mes"] == "2026-09-01"].set_index("cod_ibge")["spi3"].astype("float64").round(2)
    esperado = np.where(s >= 1.0, "umido", np.where(s <= -1.0, "seca", "normal"))
    assert list(base["detalhe"].set_index("cod_ibge").loc[s.index, "grupo"]) == list(esperado)
    assert sum(g["n"] for g in base["grupos"].values()) == base["n_validos"] == 3


def test_recorde_e_acima_do_p90_com_chuva_extrema_no_alvo(raw_base, demo_res, cfg):
    raw = raw_base.copy()
    m = (raw["cod_ibge"] == A) & (raw["mes"] >= pd.Timestamp("2026-07-01"))
    raw.loc[m, "precip_mm"] = np.float32(600.0)           # 1800 mm na janela: acima de qualquer setembro anterior
    r, _ = diag_novo(raw, cfg)
    assert bool(linha(r, A)["recorde"]) and bool(linha(r, A)["acima_p90"])
    assert not bool(linha(r, C)["recorde"]) and linha(r, A)["spi"] > 1.0
    assert r["grupos"]["umido"]["recorde"] >= 1


def test_sem_recorde_quando_a_janela_e_pequena(raw_base, demo_res, cfg):
    raw = raw_base.copy()
    m = (raw["cod_ibge"] == B) & (raw["mes"] >= pd.Timestamp("2026-07-01"))
    raw.loc[m, "precip_mm"] = np.float32(0.5)
    r, _ = diag_novo(raw, cfg)
    assert not bool(linha(r, B)["recorde"]) and not bool(linha(r, B)["acima_p90"])
    assert linha(r, B)["spi"] < -1.0 and r["grupos"]["seca"]["n"] >= 1
    n = {g: r["grupos"][g]["n"] for g in dg.GRUPOS}
    assert sum(n.values()) == r["n_validos"] == 3 and n["normal"] == 3 - n["umido"] - n["seca"]


def test_mes_do_spi_minimo_bate_com_idxmin_do_demo(base, demo_res):
    d = demo_res.demo_observado.assign(mes=lambda x: pd.to_datetime(x["mes"]))
    cont = np.zeros(12, dtype=int)
    for cod in (A, B, C):
        s = d[d["cod_ibge"] == cod].set_index("mes")["spi3"].dropna()
        cont[s.idxmin().month - 1] += 1
    assert base["cal_do_spi_minimo"] == list(cont) and sum(cont) == 3


def test_serie_do_estado_tem_12_meses_e_mediana_conferida(base, raw_base):
    s = base["serie_estado"]
    assert [x["mes"] for x in s] == [str(p) for p in pd.period_range("2025-10", "2026-09", freq="M")]
    p = raw_base.assign(mes=lambda x: pd.to_datetime(x["mes"]))
    set26 = p[p["mes"] == "2026-09-01"]["precip_mm"].astype("float64")
    assert s[-1]["estado_mm"] == pytest.approx(set26.median(), rel=1e-9)
    treino_set = p[(p["mes"].dt.month == 9) & (p["mes"] <= "2015-12-01")]["precip_mm"].astype("float64")
    assert s[-1]["clim_mm"] == pytest.approx(treino_set.median(), rel=1e-9)


def test_mes_no_treino_e_sinalizado_e_escala_6_funciona(raw_base, demo_res, cfg):
    r = diag(raw_base, demo_res, cfg, mes=pd.Period("2010-09", "M"), k=6)
    assert r["mes_no_treino"] is True and r["escala"] == 6
    assert "in-sample" in dg.formatar(r, "aviso")
    assert diag(raw_base, demo_res, cfg)["mes_no_treino"] is False


# ================================================================== erros
def test_erros_claros(raw_base, demo_res, cfg):
    with pytest.raises(dg.DiagError, match="fora da janela"):
        diag(raw_base, demo_res, cfg, mes=pd.Period("2026-10", "M"))
    with pytest.raises(dg.DiagError, match="sem a coluna"):
        diag(raw_base, demo_res, cfg, k=12)
    so3 = demo_res.spi_gamma_params[demo_res.spi_gamma_params["escala_meses"] == 3]
    with pytest.raises(dg.DiagError, match="sem a escala SPI-6"):
        diag(raw_base, demo_res, cfg, k=6, params=so3)


def test_demo_com_duplicata_e_recusado(raw_base, demo_res, cfg):
    demo = pd.concat([demo_res.demo_observado, demo_res.demo_observado.iloc[:1]], ignore_index=True)
    with pytest.raises(dg.DiagError, match="duplicatas"):
        dg.calcular_diagnostico(raw_base, demo, demo_res.spi_gamma_params, treino_fim=cfg.split.treino_fim, mes=ALVO, k=3)


# ================================================================== texto
def test_formatar_traz_aviso_cabecalho_e_secoes(base):
    t = dg.formatar(base, "MVP experimental de teste")
    linhas = t.splitlines()
    assert linhas[0] == "AVISO: MVP experimental de teste"
    assert "DIAGNÓSTICO SPI-3 em 2026-09" in linhas[1] and "municípios=3" in linhas[1]
    assert any(l.startswith("úmidos (SPI>=+1) | 2 |") or l.startswith("úmidos (SPI>=+1) | 1 |") or
               l.startswith("úmidos (SPI>=+1) | 3 |") or l.startswith("úmidos (SPI>=+1) | 0 |") for l in linhas)
    cal = next(l for l in linhas if l.startswith("mês-calendário do SPI mínimo"))
    assert len(cal.split(": ")[1].split()) == 12
    assert sum(1 for l in linhas if l.startswith(("2025-", "2026-"))) == 12
    assert t.count("nan") == 0 and "None" not in t


def test_formatar_sem_grupo_usa_travessao_e_nao_nan(raw_base, demo_res, cfg):
    raw = raw_base.copy()
    raw.loc[raw["mes"] >= pd.Timestamp("2026-07-01"), "precip_mm"] = np.float32(150.0)
    t = dg.formatar(diag_novo(raw, cfg)[0], "aviso")
    assert "nan" not in t.lower().replace("nan%", "") and "—" in t


# ================================================================== ponta a ponta (projeto de mentira, manifest real)
def montar_proj(tmp_path):
    shutil.copy(PARAMS_REAL, tmp_path / "params.yaml")
    ref = tmp_path / "data" / "reference"
    ref.mkdir(parents=True)
    (ref / "referencia.sha256.json").write_text(json.dumps(
        {"conteudo_sha256": "ab" * 32, "contagens": {"malha": 224, "escopo": 3},
         "entradas": {"PI_Municipios_2022.zip": "cd" * 32}, "parametros": {"incluir_transicao": True}}), encoding="utf-8")
    (ref / "municipios_semiarido.csv").write_bytes(
        "cod_ibge,nome,uf,situacao,area_km2\n2200001,Aaa,PI,res150,10.0\n".encode("utf-8"))
    pd.DataFrame({"CD_MUN": [A], "geometry": [b"wkb"]}).to_parquet(ref / "malha_pi_2022.parquet", index=False)
    (ref / "malha_pi_2022_gee.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8")
    cfg = load_config(tmp_path / "params.yaml")
    d = chirps_raw.diretorio_raw(cfg)
    raw = raw_sintetico()
    for ini, fim, quando in (("1981-01", "2000-12", "2026-11-19T00:40:00+00:00"), ("2001-01", "2026-09", "2026-11-18T23:10:00+00:00")):
        i, f = pd.Period(ini, "M"), pd.Period(fim, "M")
        m = (raw["mes"] >= i.to_timestamp()) & (raw["mes"] <= f.to_timestamp())
        chirps_raw.gravar_lote(raw[m].reset_index(drop=True), d, (A, B, C), i, f, meta={"extraido_em_utc": quando})
    bd.construir_demo(cfg, municipios=(A, B, C), agora=AGORA_TESTE)
    return tmp_path


def test_cli_ponta_a_ponta_imprime_diagnostico(tmp_path, capsys):
    raiz = montar_proj(tmp_path)
    assert dg.main(["--params", str(raiz / "params.yaml"), "--mes", "2026-09"]) == 0
    saida = capsys.readouterr().out
    assert saida.startswith("AVISO: MVP experimental") and "DIAGNÓSTICO SPI-3 em 2026-09" in saida
    assert "pouca chuva (<10.0 mm/mês)" in saida and "calibração do limite (SPI-3)" in saida
    assert dg.main(["--params", str(raiz / "params.yaml"), "--escala", "6"]) == 0      # padrão: janela.fim
    assert "DIAGNÓSTICO SPI-6 em 2026-09" in capsys.readouterr().out


def test_cli_recusa_escala_fora_da_config_e_mes_fora_da_janela(tmp_path, capsys):
    raiz = montar_proj(tmp_path)
    assert dg.main(["--params", str(raiz / "params.yaml"), "--escala", "12"]) == 1
    assert "spi.janelas_meses" in capsys.readouterr().err
    assert dg.main(["--params", str(raiz / "params.yaml"), "--mes", "2026-10"]) == 1
    assert "fora da janela" in capsys.readouterr().err


def test_cli_recusa_artefato_adulterado(tmp_path, capsys):
    raiz = montar_proj(tmp_path)
    alvo = raiz / "data" / "curated" / "demo_observado.parquet"
    alvo.write_bytes(alvo.read_bytes() + b"x")
    assert dg.main(["--params", str(raiz / "params.yaml")]) == 1
    assert "ERRO" in capsys.readouterr().err


def test_cli_nao_grava_nada(tmp_path):
    raiz = montar_proj(tmp_path)
    antes = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in raiz.rglob("*") if p.is_file()}
    assert dg.main(["--params", str(raiz / "params.yaml")]) == 0
    depois = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in raiz.rglob("*") if p.is_file()}
    assert antes == depois


# ================================================================== higiene do módulo
def test_modulo_e_so_leitura_sem_web_sql_ml_nem_dados_inventados():
    src = (RAIZ / "src" / "data" / "diag_spi.py").read_text(encoding="utf-8")
    raizes = set()
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Import):
            raizes |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            raizes.add(n.module.split(".")[0])
    assert not raizes & {"fastapi", "flask", "sqlite3", "sqlalchemy", "torch", "sklearn", "requests", "urllib"}
    assert "default_rng" not in src and "np.random" not in src and "random." not in src
    for escrita in (".to_parquet", ".write_text", ".write_bytes", "open(", ".to_csv", "shutil", "os.remove"):
        assert escrita not in src
    assert "caminho_verificado" in src and "verificar_manifest" in src


# ================================================================== bordas dos limiares e das contagens
def _demo_com(demo_res, cod, mes, spi3=None, baixa=None):
    d = demo_res.demo_observado.copy()
    m = (d["cod_ibge"] == cod) & (pd.to_datetime(d["mes"]) == pd.Timestamp(mes))
    assert m.sum() == 1
    if spi3 is not None:
        d.loc[m, "spi3"] = np.float32(spi3)
    if baixa is not None:
        d.loc[m, "spi3_baixa_conf"] = baixa
    return d


def test_grupo_usa_o_valor_de_duas_casas_na_borda(raw_base, demo_res, cfg):
    d = _demo_com(demo_res, A, "2026-09-01", spi3=0.996)         # 0,996 aparece como 1,00 -> úmido
    d = d.mask((d["cod_ibge"] == B) & (pd.to_datetime(d["mes"]) == "2026-09-01"), d.assign(spi3=-0.996))
    r = dg.calcular_diagnostico(raw_base, d, demo_res.spi_gamma_params, treino_fim=cfg.split.treino_fim, mes=ALVO, k=3)
    g = r["detalhe"].set_index("cod_ibge")["grupo"]
    assert g[A] == "umido" and g[B] == "seca"
    d2 = _demo_com(demo_res, A, "2026-09-01", spi3=0.994)        # 0,994 aparece como 0,99 -> normal
    r2 = dg.calcular_diagnostico(raw_base, d2, demo_res.spi_gamma_params, treino_fim=cfg.split.treino_fim, mes=ALVO, k=3)
    assert r2["detalhe"].set_index("cod_ibge")["grupo"][A] == "normal"


def test_limiar_in_sample_e_inclusivo_em_mais_e_menos_um(raw_base, demo_res, cfg):
    d = demo_res.demo_observado.copy()
    for ano, v in ((1990, 1.0), (1991, -1.0), (1992, 0.99), (1993, -0.99)):
        m = (d["cod_ibge"] == A) & (pd.to_datetime(d["mes"]) == pd.Timestamp(f"{ano}-09-01"))
        d.loc[m, "spi3"] = np.float32(v)
    r = dg.calcular_diagnostico(raw_base, d, demo_res.spi_gamma_params, treino_fim=cfg.split.treino_fim, mes=ALVO, k=3)
    d["mes"] = pd.to_datetime(d["mes"])
    s = d[(d["cod_ibge"] == A) & (d["mes"].dt.month == 9) & (d["mes"] <= "2015-12-01")]["spi3"].dropna().astype("float64").round(2)
    ref = r["detalhe"].set_index("cod_ibge").loc[A]
    assert ref["insample_umido"] == pytest.approx((s >= 1.0).sum() / len(s), abs=1e-12)
    assert ref["insample_seca"] == pytest.approx((s <= -1.0).sum() / len(s), abs=1e-12)
    assert (s == 1.0).sum() >= 1 and (s == -1.0).sum() >= 1


def test_contagens_de_baixa_confianca_e_de_q_acima_do_limiar(raw_base, demo_res, cfg):
    d = _demo_com(demo_res, A, "2026-09-01", baixa=True)
    p = demo_res.spi_gamma_params.copy()
    m = (p["cod_ibge"] == B) & (p["escala_meses"] == 3) & (p["mes_calendario"] == 9)
    p.loc[m, "q"] = 0.6
    r = dg.calcular_diagnostico(raw_base, d, p, treino_fim=cfg.split.treino_fim, mes=ALVO, k=3)
    det = r["detalhe"].set_index("cod_ibge")
    assert bool(det.loc[A, "baixa_conf"]) and not bool(det.loc[B, "baixa_conf"])
    assert r["q"]["n_acima_limiar"] == 1 and r["q"]["maximo"] == pytest.approx(0.6)
    assert sum(g["baixa_conf"] for g in r["grupos"].values()) == 1
    assert sum(g["q_acima_limiar"] for g in r["grupos"].values()) == 1
    r2 = dg.calcular_diagnostico(raw_base, d, p, treino_fim=cfg.split.treino_fim, mes=ALVO, k=3, q_baixa=0.7)
    assert r2["q"]["n_acima_limiar"] == 0


def test_razao_e_soma_sobre_media_e_nan_sem_media_positiva(base):
    d = base["detalhe"]
    np.testing.assert_allclose(d["razao"], d["soma_mm"] / d["media_hist_mm"], rtol=1e-12)


def test_razao_fica_nan_quando_a_media_historica_e_zero(raw_base, demo_res, cfg):
    raw = raw_base.copy()
    raw.loc[(raw["cod_ibge"] == C) & (raw["mes"] <= pd.Timestamp("2015-12-01")), "precip_mm"] = np.float32(0.0)
    r, _ = diag_novo(raw, cfg)
    assert r["detalhe"].set_index("cod_ibge").loc[C, "media_hist_mm"] == 0.0
    assert np.isnan(r["detalhe"].set_index("cod_ibge").loc[C, "razao"])


def _params_com_q(demo_res, cod, q):
    p = demo_res.spi_gamma_params.copy()
    m = (p["cod_ibge"] == cod) & (p["escala_meses"] == 3) & (p["mes_calendario"] == 9)
    p.loc[m, "q"] = q
    return p


@pytest.mark.parametrize("q", [0.05, 0.3, 0.8])
def test_mm_para_umido_inverte_o_spi_tambem_com_zeros_no_treino(raw_base, demo_res, cfg, q):
    r = diag(raw_base, demo_res, cfg, params=_params_com_q(demo_res, A, q))
    x = linha(r, A)["mm_para_umido"]
    assert x > 0 and _spi_em(demo_res, A, x, q) == pytest.approx(1.0, abs=1e-6)


def test_mm_para_seca_inverte_o_spi_com_zeros_no_treino(raw_base, demo_res, cfg):
    q = 0.1                                                # q < Phi(-1) = 0,1587: ainda é alcançável
    r = diag(raw_base, demo_res, cfg, params=_params_com_q(demo_res, A, q))
    x = linha(r, A)["mm_para_seca"]
    assert x > 0 and _spi_em(demo_res, A, x, q) == pytest.approx(-1.0, abs=1e-6)


def test_q_exatamente_no_percentil_vira_zero_mm_nos_dois_sentidos(raw_base, demo_res, cfg):
    r = diag(raw_base, demo_res, cfg, params=_params_com_q(demo_res, A, float(stats.norm.cdf(-1.0))))
    assert linha(r, A)["mm_para_seca"] == 0.0
    r = diag(raw_base, demo_res, cfg, params=_params_com_q(demo_res, A, float(stats.norm.cdf(1.0))))
    assert linha(r, A)["mm_para_umido"] == 0.0


def test_q_igual_ao_limiar_de_baixa_confianca_nao_conta(raw_base, demo_res, cfg):
    r = dg.calcular_diagnostico(raw_base, demo_res.demo_observado, _params_com_q(demo_res, A, 0.5),
                                treino_fim=cfg.split.treino_fim, mes=ALVO, k=3, q_baixa=0.5)
    assert r["q"]["n_acima_limiar"] == 0                   # q > limiar, estrito, como em spi.py
    assert sum(g["q_acima_limiar"] for g in r["grupos"].values()) == 0


def test_empate_com_o_maximo_anterior_nao_e_recorde(raw_base, demo_res, cfg):
    p = raw_base[raw_base["cod_ibge"] == C].set_index("mes")["precip_mm"].astype("float64")
    S = p.rolling(3).sum()
    ano = S[(S.index.month == 9) & (S.index < "2026-01-01")].idxmax()          # setembro de maior soma
    janela = p.loc[ano - pd.DateOffset(months=2): ano].to_numpy()
    raw = raw_base.copy()
    m = (raw["cod_ibge"] == C) & (raw["mes"] >= pd.Timestamp("2026-07-01"))
    raw.loc[m, "precip_mm"] = janela.astype("float32")
    r, _ = diag_novo(raw, cfg)
    det = r["detalhe"].set_index("cod_ibge").loc[C]
    assert det["soma_mm"] == pytest.approx(janela.sum(), rel=1e-5)
    assert not bool(det["recorde"])
    raw.loc[m, "precip_mm"] = (janela * 1.01).astype("float32")
    assert bool(diag_novo(raw, cfg)[0]["detalhe"].set_index("cod_ibge").loc[C, "recorde"])


# ================================================================== calibração do limite e pouca chuva esperada
def test_calibracao_bate_com_conta_independente_dos_parametros(base, demo_res):
    p = demo_res.spi_gamma_params
    p = p[(p["escala_meses"] == 3) & p["valido"]]
    cal = base["calibracao"]
    assert [c["mes"] for c in cal] == list(range(1, 13))
    for c in cal:
        x = p[p["mes_calendario"] == c["mes"]]
        media = ((1 - x["q"]) * x["shape"] * x["scale"] / 3).to_numpy()
        assert c["n"] == len(media) == 3
        assert c["mediana"] == pytest.approx(float(np.median(media)), rel=1e-12)
        for lim, pct in c["pct"].items():
            assert pct == pytest.approx((media < lim).mean() * 100, abs=1e-9)


def test_calibracao_so_conta_celulas_validas_e_usa_a_escala_pedida(raw_base, demo_res, cfg):
    p = demo_res.spi_gamma_params.copy()
    m = (p["cod_ibge"] == B) & (p["escala_meses"] == 3) & (p["mes_calendario"] == 5)
    p.loc[m, "valido"] = False
    r = diag(raw_base, demo_res, cfg, params=p)
    assert [c["n"] for c in r["calibracao"]][4] == 2 and r["calibracao"][0]["n"] == 3
    r6 = diag(raw_base, demo_res, cfg, k=6)
    x = demo_res.spi_gamma_params
    x = x[(x["escala_meses"] == 6) & (x["mes_calendario"] == 1) & x["valido"]]
    assert r6["calibracao"][0]["mediana"] == pytest.approx(float(np.median((1 - x["q"]) * x["shape"] * x["scale"] / 6)), rel=1e-12)


def test_pouca_chuva_e_estritamente_menor_que_o_limite(raw_base, demo_res, cfg):
    sem = diag(raw_base, demo_res, cfg)
    media = float(sem["detalhe"].set_index("cod_ibge").loc[A, "media_mm_mes"])
    assert media > 0 and not sem["detalhe"]["pouca_chuva"].any()          # limite padrão 0 desliga
    kw = dict(treino_fim=cfg.split.treino_fim, mes=ALVO, k=3)
    no_limite = dg.calcular_diagnostico(raw_base, demo_res.demo_observado, demo_res.spi_gamma_params, media_min_mm_mes=media, **kw)
    acima = dg.calcular_diagnostico(raw_base, demo_res.demo_observado, demo_res.spi_gamma_params, media_min_mm_mes=media * 1.0000001, **kw)
    assert not bool(no_limite["detalhe"].set_index("cod_ibge").loc[A, "pouca_chuva"])
    assert bool(acima["detalhe"].set_index("cod_ibge").loc[A, "pouca_chuva"])
    assert sum(g["pouca_chuva"] for g in acima["grupos"].values()) == int(acima["detalhe"]["pouca_chuva"].sum()) >= 1


def test_celula_invalida_nao_conta_como_pouca_chuva(raw_base, demo_res, cfg):
    p = demo_res.spi_gamma_params.copy()
    m = (p["cod_ibge"] == B) & (p["escala_meses"] == 3) & (p["mes_calendario"] == 9)
    p.loc[m, "valido"] = False
    r = dg.calcular_diagnostico(raw_base, demo_res.demo_observado, p, treino_fim=cfg.split.treino_fim, mes=ALVO, k=3,
                                media_min_mm_mes=1e6)
    det = r["detalhe"].set_index("cod_ibge")
    assert not bool(det.loc[B, "pouca_chuva"]) and np.isnan(det.loc[B, "media_mm_mes"]) and bool(det.loc[A, "pouca_chuva"])


def test_formatar_mostra_a_calibracao_e_o_limite_no_cabecalho(raw_base, demo_res, cfg):
    r = dg.calcular_diagnostico(raw_base, demo_res.demo_observado, demo_res.spi_gamma_params,
                                treino_fim=cfg.split.treino_fim, mes=ALVO, k=3, media_min_mm_mes=10.0)
    linhas = dg.formatar(r, "aviso").splitlines()
    assert any("pouca chuva (<10.0 mm/mês)" in l for l in linhas)
    i = next(n for n, l in enumerate(linhas) if l.startswith("calibração do limite (SPI-3)"))
    assert linhas[i + 1].startswith("mês | n | mediana mm/mês | <5 | <10 | <15 | <20 | <30")
    meses = [l.split(" | ")[0] for l in linhas[i + 2: i + 14]]
    assert meses == [f"{m:02d}" for m in range(1, 13)]


# ================================================================== params.yaml novo antes de regenerar (uso real)
def _troca_no_params(raiz, antigo, novo):
    p = raiz / "params.yaml"
    t = p.read_text(encoding="utf-8")
    assert t.count(antigo) == 1
    p.write_text(t.replace(antigo, novo), encoding="utf-8")


def test_cli_aceita_limiar_novo_no_params_e_avisa(tmp_path, capsys):
    raiz = montar_proj(tmp_path)
    assert dg.main(["--params", str(raiz / "params.yaml")]) == 0
    assert "NOTA:" not in capsys.readouterr().out                       # params intacto: sem nota
    _troca_no_params(raiz, "media_min_mm_por_mes: 10.0", "media_min_mm_por_mes: 20.0")
    assert dg.main(["--params", str(raiz / "params.yaml"), "--mes", "2026-09"]) == 0
    saida = capsys.readouterr().out
    assert "pouca chuva (<20.0 mm/mês)" in saida                       # usa o limiar do params ATUAL
    assert "NOTA: o params.yaml mudou desde o manifest" in saida and "--substituir-manifest" in saida
    assert saida.index("NOTA:") > saida.index("calibração do limite")  # a nota vem depois do relatório


@pytest.mark.parametrize("antigo,novo,campo", [
    ('treino_fim: "2015-12"', 'treino_fim: "2015-06"', "treino_fim"),
    ("janelas_meses: [3, 6]", "janelas_meses: [3]", "janelas_spi"),
    ('fim: "2026-09"', 'fim: "2026-08"', "janela_fim"),
    ('inicio: "1981-01"', 'inicio: "1981-02"', "janela_inicio"),
])
def test_cli_recusa_params_novo_que_muda_o_ajuste_da_gama(tmp_path, capsys, antigo, novo, campo):
    raiz = montar_proj(tmp_path)
    _troca_no_params(raiz, antigo, novo)
    assert dg.main(["--params", str(raiz / "params.yaml")]) == 1
    err = capsys.readouterr().err
    assert "ERRO" in err and campo in err and "build_demo" in err


def test_cli_params_ausente_ainda_diagnostica_com_nota(tmp_path, capsys):
    raiz = montar_proj(tmp_path)
    copia = raiz / "params_copia.yaml"
    shutil.copy(raiz / "params.yaml", copia)
    (raiz / "params.yaml").unlink()
    assert dg.main(["--params", str(copia)]) == 0
    assert "NOTA: o params.yaml mudou" in capsys.readouterr().out
