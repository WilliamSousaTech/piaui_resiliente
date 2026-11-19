"""Ficha do município para a IA (dados sintéticos só aqui, em tests/). Cada número é recalculado de forma independente."""
import ast
import base64
import copy
from pathlib import Path

import numpy as np
import pytest

from dados_sinteticos import A, B, C
from src.ia import fatos as ft
from src.web import build_site as bs


def _cent(x):          # o mesmo arredondamento do pacote (np.round), mas a partir do DataFrame
    return int(np.round(float(x) * 100))


def _linha(insumos, cod):
    d = insumos.demo[insumos.demo["cod_ibge"] == cod].sort_values("mes").reset_index(drop=True)
    r = insumos.raw[insumos.raw["cod_ibge"] == cod].sort_values("mes").reset_index(drop=True)
    return d, r


# ------------------------------------------------------------------ contrato com o pacote do site
def test_constantes_iguais_as_do_site():
    assert ft.NA16 == bs.NA16 and ft.SEM_DADO == bs.SEM_DADO


def test_classes_do_pacote_a_seca_sao_as_tres_primeiras(payload):
    assert [c["seca"] for c in payload["classes"]] == [True, True, True, False, False, False, False]


# ------------------------------------------------------------------ valores
@pytest.mark.parametrize("cod", [A, B, C])
def test_fatos_batem_com_conta_independente(payload, insumos, cod):
    f = ft.fatos_municipio(payload, cod)
    d, r = _linha(insumos, cod)
    t = payload["meses"].index(f.mes)
    assert f.mes == payload["meses"][payload["abertura"]]
    assert f.escala == payload["escala_alvo"] == 3

    ok = ~d["spi3_invalid"].to_numpy(bool) & ~np.isnan(d["spi3"].to_numpy("float64"))
    assert f.spi_cent == (_cent(d["spi3"][t]) if ok[t] else None)

    conf = ok & ~d["spi3_baixa_conf"].to_numpy(bool)
    cents = np.array([_cent(x) if o else 0 for x, o in zip(d["spi3"], ok)])
    seca = ok & (cents <= -100)
    j = slice(t - 11, t + 1)
    assert f.leituras_confiaveis_12m == int(conf[j].sum())
    assert f.meses_em_seca_12m == int((seca & conf)[j].sum())
    assert f.em_seca == bool(seca[t])
    assert f.baixa_confianca == bool(d["spi3_baixa_conf"][t] and ok[t])

    tenths = np.round(r["precip_mm"].to_numpy("float64") * 10).astype("int64")
    assert f.chuva_mes_dec == int(tenths[t])
    assert f.chuva_12m_dec == int(tenths[j].sum())


def test_nome_e_codigo_vem_da_malha(payload):
    f = ft.fatos_municipio(payload, A)
    assert (f.cod_ibge, f.nome) == (A, "Alfa do Norte")


def test_escala_6_e_mes_explicito(payload):
    f = ft.fatos_municipio(payload, B, mes="2020-03", escala=6)
    assert (f.mes, f.escala) == ("2020-03", 6)
    assert "SPI-6" in ft.ficha_texto(f)


def test_primeiro_mes_com_historico_de_12_meses(payload):
    ft.fatos_municipio(payload, A, mes="1981-12")
    with pytest.raises(ft.FatosError, match="12 meses"):
        ft.fatos_municipio(payload, A, mes="1981-11")


@pytest.mark.parametrize("kw, trecho", [({"cod_ibge": "9999999"}, "não está no pacote"),
                                       ({"cod_ibge": A, "mes": "2030-01"}, "fora da janela"),
                                       ({"cod_ibge": A, "escala": 9}, "SPI-9")])
def test_entradas_invalidas(payload, kw, trecho):
    with pytest.raises(ft.FatosError, match=trecho):
        ft.fatos_municipio(payload, **kw)


# ------------------------------------------------------------------ causalidade: o futuro não muda a ficha
def _trocar(payload, chave, sub, dtype, fn):
    a = np.frombuffer(base64.b64decode(payload[chave][sub] if sub else payload[chave]), dtype=dtype).copy()
    fn(a)
    novo = base64.b64encode(a.astype(np.dtype(dtype)).tobytes()).decode()
    if sub:
        payload[chave][sub] = novo
    else:
        payload[chave] = novo


def test_alterar_meses_depois_do_mes_nao_muda_a_ficha(payload):
    mes = "2010-06"
    antes = ft.fatos_municipio(payload, A, mes=mes)
    p = copy.deepcopy(payload)
    nm, nt = len(p["municipios"]), len(p["meses"])
    t = p["meses"].index(mes)

    def futuro(a):
        a.reshape(nm, nt)[:, t + 1:] = 7
    _trocar(p, "spi", "3", "<i2", futuro)
    _trocar(p, "cls", "3", "u1", futuro)
    _trocar(p, "baixa", "3", "u1", futuro)
    _trocar(p, "precip", None, "<i2", futuro)
    assert ft.fatos_municipio(p, A, mes=mes) == antes


# ------------------------------------------------------------------ baixa confiança e mês sem SPI dentro da janela
def test_baixa_confianca_e_sem_dado_ficam_fora_da_contagem_de_secas(payload):
    mes = "2010-06"
    p = copy.deepcopy(payload)
    nm, nt = len(p["municipios"]), len(p["meses"])
    t = p["meses"].index(mes)
    i = [m["cod"] for m in p["municipios"]].index(A)
    classes = [0, 0, 0, 3, 3, 3, 255, 3, 3, 3, 3, 3]            # 3 meses de seca, 1 mês sem SPI
    baixas = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]               # o primeiro mês de seca é de baixa confiança

    def pôr(vals):
        return lambda a: a.reshape(nm, nt).__setitem__((i, slice(t - 11, t + 1)), vals)
    _trocar(p, "cls", "3", "u1", pôr(classes))
    _trocar(p, "baixa", "3", "u1", pôr(baixas))
    f = ft.fatos_municipio(p, A, mes=mes)
    assert f.leituras_confiaveis_12m == 10                      # 12 − 1 de baixa confiança − 1 sem SPI
    assert f.meses_em_seca_12m == 2                             # a seca de baixa confiança não conta


# ------------------------------------------------------------------ município sem dado
def _sem_dado(payload, cod):
    p = copy.deepcopy(payload)
    i = [m["cod"] for m in p["municipios"]].index(cod)
    nm, nt = len(p["municipios"]), len(p["meses"])
    p["municipios"][i]["exc"] = True
    for chave, sub, dtype, valor in (("spi", "3", "<i2", ft.NA16), ("cls", "3", "u1", ft.SEM_DADO),
                                     ("baixa", "3", "u1", 0)):
        _trocar(p, chave, sub, dtype, lambda a, v=valor: a.reshape(nm, nt).__setitem__(i, v))
    return p


def test_municipio_excluido_nao_e_utilizavel_e_nao_gera_ficha(payload):
    f = ft.fatos_municipio(_sem_dado(payload, B), B)
    assert f.excluido and f.spi_cent is None and f.classe is None and not f.utilizavel
    with pytest.raises(ft.FatosError, match="nada a escrever"):
        ft.ficha_texto(f)


# ------------------------------------------------------------------ formatação
@pytest.mark.parametrize("v, esc, esperado", [(-123, 100, "−1,23"), (5, 100, "0,05"), (-5, 100, "−0,05"),
                                              (0, 100, "0,00"), (12345, 10, "1.234,5"), (-8, 10, "−0,8"),
                                              (100, 100, "1,00"), (999999, 10, "99.999,9")])
def test_decimal_br_sem_float(v, esc, esperado):
    assert ft._decimal_br(v, esc) == esperado


def test_ficha_tem_so_o_que_o_site_mostra(payload):
    f = ft.fatos_municipio(payload, A)
    txt = ft.ficha_texto(f)
    assert txt.splitlines()[0] == f"Município: Alfa do Norte ({A})"
    assert ft.mes_por_extenso(f.mes) in txt and f"SPI-{f.escala} observado: " in txt
    assert f.classe in txt
    assert len(txt.splitlines()) == 7


def test_mes_por_extenso():
    assert ft.mes_por_extenso("2026-07") == "julho de 2026"
    assert ft.mes_por_extenso("1999-03") == "março de 1999"


# ------------------------------------------------------------------ números permitidos
def test_numeros_normalizam_milhar_e_decimal():
    assert ft.numeros("1.234,5 mm e 1234,5 mm") == ["1234,5", "1234,5"]
    assert ft.numeros("SPI-3 de −1,23 em 2026") == ["3", "1,23", "2026"]
    assert ft.numeros("1.5") == ["1,5"]
    assert ft.numeros("sem número") == []


def test_numeros_fora_lista_sem_repetir_na_ordem(payload):
    ok = ft.numeros_permitidos(ft.ficha_texto(ft.fatos_municipio(payload, A)))
    assert ft.numeros_fora("choveu 999 mm, depois 999 mm e 4242", ok) == ["999", "4242"]
    assert ft.numeros_fora("sem números", ok) == []


def test_numeros_permitidos_junta_varios_textos():
    assert ft.numeros_permitidos("a 12", "b 3,5") == frozenset({"12", "3,5"})


def test_o_proprio_numero_da_ficha_e_aceito(payload):
    f = ft.fatos_municipio(payload, C)
    ficha = ft.ficha_texto(f)
    assert ft.numeros_fora(ficha, ft.numeros_permitidos(ficha)) == []


# ------------------------------------------------------------------ higiene
def test_fatos_nao_importa_web_nem_rede_nem_framework():
    arvore = ast.parse(Path(ft.__file__).read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(arvore):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    proibidos = ("src.web", "requests", "httpx", "urllib", "socket", "http", "fastapi", "flask", "sqlite3", "sqlalchemy",
                 "torch")
    assert not [m for m in mods if m == "src" or any(m == p or m.startswith(p + ".") for p in proibidos)]
