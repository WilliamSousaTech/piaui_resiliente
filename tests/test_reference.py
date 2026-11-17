"""Testes de src/data/reference.py (Item 2b). Sem internet, sem GEE; dados sintéticos.

Malha sintética: 224 quadrados no PI (EPSG:4674), com vértices colineares extras para
que a simplificação tenha o que remover sem alterar a área.
  - 214 + polo (2207801) = 215 municípios 'res150'
  - 1 'transicao' (2200214)
  - 8 'fora' (2200215..2200222)
Anexo sintético: 1.427 linhas (215 do PI + 1.212 do MA).
"""
import json
import shutil
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
import shapely
from shapely.geometry import Point, Polygon, box

from src.data import reference as ref
from src.data.config import PendenciaError, load_config

PARAMS_REAL = Path(__file__).resolve().parents[1] / "params.yaml"
POLO = "2207801"
CODS = [f"22{i:05d}" for i in range(223)] + [POLO]
NOMES = {c: f"Cidade {i}" for i, c in enumerate(CODS)}
NOMES[POLO] = "Paulistana"
RES150_PI = CODS[:214] + [POLO]
TRANS = CODS[214]
FORA = CODS[215:223]
NOME_CSV = "municipios_semiarido.csv"
ARQUIVOS_SAIDA = sorted([NOME_CSV, ref.PARQUET, ref.GEOJSON, ref.SELO])


# ------------------------------------------------------------------ fábricas
def quadrado(i: int) -> Polygon:
    x, y = -45.0 + (i % 20) * 0.1, -10.0 + (i // 20) * 0.1
    return shapely.segmentize(box(x, y, x + 0.09, y + 0.09), 0.01)


def malha_gdf(crs: str = "EPSG:4674") -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"CD_MUN": CODS, "NM_MUN": [NOMES[c] for c in CODS], "SIGLA_UF": "PI", "AREA_KM2": 100.0},
        geometry=[quadrado(i) for i in range(len(CODS))], crs=crs)


def escreve_zip(gdf: gpd.GeoDataFrame, destino: Path) -> Path:
    pasta = destino.parent / ("_shp_" + destino.stem)
    pasta.mkdir(parents=True, exist_ok=True)
    gdf.to_file(pasta / "malha.shp")
    with zipfile.ZipFile(destino, "w") as z:
        for f in sorted(pasta.iterdir()):
            z.write(f, f.name)
    shutil.rmtree(pasta)
    return destino


def anexo_df() -> pd.DataFrame:
    pi = [(c, "PI", NOMES[c].upper()) for c in RES150_PI]
    outras = [(f"21{i:05d}", "MA", f"OUTRA {i}") for i in range(1427 - 215)]
    return pd.DataFrame(pi + outras, columns=["cod_ibge", "uf", "nome"])


def trans_df() -> pd.DataFrame:
    return pd.DataFrame([(TRANS, "PI", NOMES[TRANS].upper(), "Res. 176/2024 (teste)")],
                        columns=["cod_ibge", "uf", "nome", "fonte"])


def grava_csv(df: pd.DataFrame, p: Path) -> None:
    df.to_csv(p, index=False, encoding="utf-8", lineterminator="\n")


def monta_projeto(raiz: Path, zip_base: Path, ano: str = "2022", zip_malha: Path | None = None,
                  trocas: dict[str, str] | None = None):
    txt = PARAMS_REAL.read_text(encoding="utf-8-sig")
    assert "malha_municipal_ano: 2022" in txt
    for antigo, novo in {"malha_municipal_ano: 2022": f"malha_municipal_ano: {ano}", **(trocas or {})}.items():
        assert txt.count(antigo) == 1, antigo
        txt = txt.replace(antigo, novo)
    raiz.mkdir(parents=True, exist_ok=True)
    (raiz / "params.yaml").write_text(txt, encoding="utf-8")
    fonte = raiz / "data" / "reference" / "fonte"
    fonte.mkdir(parents=True)
    shutil.copy(zip_malha or zip_base, fonte / ref.ZIP_MALHA)
    grava_csv(anexo_df(), fonte / ref.CSV_ANEXO)
    grava_csv(trans_df(), fonte / ref.CSV_TRANSICAO)
    return load_config(raiz / "params.yaml")


def load_config_de(cfg, trocas: dict[str, str]):
    p = cfg.raiz / "params.yaml"
    txt = p.read_text(encoding="utf-8")
    for antigo, novo in trocas.items():
        assert txt.count(antigo) == 1, antigo
        txt = txt.replace(antigo, novo)
    p.write_text(txt, encoding="utf-8")
    return load_config(p)


@pytest.fixture(scope="session")
def zip_base(tmp_path_factory):
    return escreve_zip(malha_gdf(), tmp_path_factory.mktemp("malha") / "PI_Municipios_2022.zip")


@pytest.fixture(scope="session")
def malha(zip_base):
    return ref.ler_malha_ibge(zip_base, "EPSG:4674")


@pytest.fixture
def projeto(tmp_path, zip_base):
    return monta_projeto(tmp_path, zip_base)


def classifica(malha, anexo=None, trans=None, incluir=True, polo=POLO):
    return ref.classificar(malha, anexo_df() if anexo is None else anexo,
                           trans_df() if trans is None else trans, incluir, polo)


def dir_ref(cfg) -> Path:
    return cfg.caminho("reference")


def estado(d: Path) -> dict:
    return {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in d.iterdir() if p.is_file()}


# ----------------------------------------------------------- normalizar_nome
@pytest.mark.parametrize("a,b", [
    ("ÁGUA BRANCA", "Água Branca"),
    ("SÃO JOÃO DO PIAUÍ", "São João do Piauí"),
    ("BARRA D'ALCÂNTARA", "Barra D’Alcântara"),
    ("PAU D-ARCO", "Pau d'Arco"),
])
def test_normalizar_nome_ignora_acento_caixa_e_pontuacao(a, b):
    assert ref.normalizar_nome(a) == ref.normalizar_nome(b)


def test_normalizar_nome_distingue_nomes_diferentes():
    assert ref.normalizar_nome("Luís Correia") != ref.normalizar_nome("Luis Correa")


# ---------------------------------------------------------------- ler_lista
CAB = "cod_ibge,uf,nome\n"


def texto(tmp_path, conteudo, enc="utf-8", nome="x.csv"):
    p = tmp_path / nome
    p.write_bytes(conteudo.encode(enc))
    return p


def test_ler_lista_le_codigo_como_texto_de_7_digitos(tmp_path):
    df = ref.ler_lista(texto(tmp_path, CAB + "2200053,PI,ACAUÃ\n2100055,MA,X\n"))
    assert df["cod_ibge"].tolist() == ["2200053", "2100055"]
    assert df["cod_ibge"].str.len().eq(7).all()
    assert df["nome"].iloc[0] == "ACAUÃ"


def test_ler_lista_tolera_bom_e_lê_a_primeira_coluna_certa(tmp_path):
    df = ref.ler_lista(texto(tmp_path, "﻿" + CAB + "2200053,PI,A\n"))
    assert list(df.columns) == ["cod_ibge", "uf", "nome"]
    assert df["cod_ibge"].iloc[0] == "2200053"


def test_ler_lista_aceita_so_cabecalho_com_extras(tmp_path):
    df = ref.ler_lista(texto(tmp_path, CAB.strip() + ",fonte\n"), ("fonte",))
    assert len(df) == 0 and list(df.columns) == ["cod_ibge", "uf", "nome", "fonte"]


@pytest.mark.parametrize("conteudo", [
    "codigo,uf,nome\n2200053,PI,A\n",             # coluna errada
    CAB + "220005,PI,A\n",                         # 6 dígitos
    CAB + "22000530,PI,A\n",                       # 8 dígitos
    CAB + "22000AB,PI,A\n",                        # não numérico
    CAB + "2200053,PI,A\n2200053,PI,B\n",          # duplicado
    CAB + "2200053,PI,\n",                         # nome vazio
    CAB + "2200053,PI,   \n",                      # nome só espaços
    CAB + "2200053,pi,A\n",                        # uf minúscula
    CAB + "2200053,MA,A\n",                        # prefixo 22 com uf != PI
    CAB + "2100053,PI,A\n",                        # uf PI com prefixo != 22
    "",                                            # arquivo vazio
])
def test_ler_lista_rejeita_formato_invalido(tmp_path, conteudo):
    with pytest.raises(ref.RefSchemaError):
        ref.ler_lista(texto(tmp_path, conteudo))


def test_ler_lista_rejeita_arquivo_fora_de_utf8(tmp_path):
    with pytest.raises(ref.RefSchemaError, match="UTF-8"):
        ref.ler_lista(texto(tmp_path, CAB + "2200053,PI,ÁGUA\n", enc="latin-1"))


def test_ler_lista_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        ref.ler_lista(tmp_path / "nao_existe.csv")


# ------------------------------------------------------------ ler_malha_ibge
def test_ler_malha_devolve_224_com_colunas_tipos_e_crs(malha):
    assert len(malha) == 224
    assert list(malha.columns) == ["cod_ibge", "nome", "uf", "area_km2", "geometry"]
    assert malha.crs.to_string() == "EPSG:4674"
    assert malha["cod_ibge"].str.fullmatch(r"22\d{5}").all()
    assert malha["area_km2"].dtype == "float64" and (malha["area_km2"] > 0).all()
    assert POLO in set(malha["cod_ibge"])


def test_ler_malha_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        ref.ler_malha_ibge(tmp_path / "nada.zip", "EPSG:4674")


def malha_quebrada(tmp_path, mexer, crs="EPSG:4674"):
    g = malha_gdf(crs)
    g = mexer(g)
    return escreve_zip(g, tmp_path / "PI_Municipios_2022.zip")


def _dup(g):
    g.loc[1, "CD_MUN"] = g.loc[0, "CD_MUN"]
    return g


def _bowtie(g):
    g.loc[0, "geometry"] = Polygon([(-45, -10), (-44.9, -9.9), (-44.9, -10), (-45, -9.9)])
    return g


def _uf(g):
    g.loc[0, "SIGLA_UF"] = "MA"
    return g


def _cod_nao_pi(g):
    g.loc[0, "CD_MUN"] = "2100000"
    return g


def _area_zero(g):
    g.loc[0, "AREA_KM2"] = 0.0
    return g


@pytest.mark.parametrize("mexer,crs", [
    (lambda g: g.drop(columns="AREA_KM2"), "EPSG:4674"),   # coluna ausente
    (lambda g: g.iloc[:-1], "EPSG:4674"),                  # 223 municípios
    (_dup, "EPSG:4674"),                                   # código duplicado
    (_bowtie, "EPSG:4674"),                                # polígono inválido
    (_uf, "EPSG:4674"),                                    # UF != PI
    (_cod_nao_pi, "EPSG:4674"),                            # código fora do PI
    (_area_zero, "EPSG:4674"),                             # área <= 0
    (lambda g: g, "EPSG:4326"),                            # CRS diferente do esperado
])
def test_ler_malha_rejeita_malha_invalida(tmp_path, mexer, crs):
    z = malha_quebrada(tmp_path, mexer, crs)
    with pytest.raises(ref.RefSchemaError):
        ref.ler_malha_ibge(z, "EPSG:4674")


# --------------------------------------------------------------- classificar
def test_classificar_conta_res150_transicao_e_fora(malha):
    cls = classifica(malha)
    cont = cls["situacao"].value_counts().to_dict()
    assert cont == {"res150": 215, "transicao": 1, "fora": 8}
    assert int(cls["no_escopo"].sum()) == 216 and len(cls) == 224
    assert cls.loc[cls["cod_ibge"] == TRANS, "situacao"].item() == "transicao"
    assert set(cls.loc[cls["situacao"] == "fora", "cod_ibge"]) == set(FORA)


def test_classificar_sem_transicao_exclui_do_escopo_mas_mantem_a_situacao(malha):
    cls = classifica(malha, incluir=False)
    assert int(cls["no_escopo"].sum()) == 215
    assert not cls.loc[cls["cod_ibge"] == TRANS, "no_escopo"].item()
    assert (cls["situacao"] == "transicao").sum() == 1


def test_classificar_com_transicao_vazia(malha):
    vazia = trans_df().iloc[0:0]
    cls = classifica(malha, trans=vazia)
    assert int(cls["no_escopo"].sum()) == 215 and (cls["situacao"] == "fora").sum() == 9


def test_classificar_nao_altera_a_malha_de_entrada(malha):
    antes = malha.copy()
    classifica(malha)
    assert list(malha.columns) == list(antes.columns)
    assert malha["cod_ibge"].tolist() == antes["cod_ibge"].tolist()


def test_classificar_aceita_nome_com_acento_e_caixa_diferentes(malha):
    anexo = anexo_df()
    anexo.loc[anexo["cod_ibge"] == POLO, "nome"] = "PAULISTÂNA".replace("Â", "A")
    assert int(classifica(malha, anexo=anexo)["no_escopo"].sum()) == 216


def test_classificar_anexo_com_total_errado(malha):
    with pytest.raises(ref.RefIntegridadeError, match="1426"):
        classifica(malha, anexo=anexo_df().iloc[:-1])


def test_classificar_anexo_com_quantidade_de_pi_errada(malha):
    anexo = anexo_df()
    anexo.loc[0, "uf"] = "MA"
    with pytest.raises(ref.RefIntegridadeError, match="214"):
        classifica(malha, anexo=anexo)


def test_classificar_transicao_repetida_no_anexo(malha):
    trans = trans_df()
    trans.loc[0, ["cod_ibge", "nome"]] = [RES150_PI[0], NOMES[RES150_PI[0]].upper()]
    with pytest.raises(ref.RefIntegridadeError, match="já está no anexo"):
        classifica(malha, trans=trans)


def test_classificar_codigo_ausente_na_malha(malha):
    trans = trans_df()
    trans.loc[0, "cod_ibge"] = "2299999"
    with pytest.raises(ref.RefIntegridadeError, match="2299999"):
        classifica(malha, trans=trans)


def test_classificar_nome_divergente_entre_lista_e_malha(malha):
    trans = trans_df()
    trans.loc[0, "nome"] = "OUTRO NOME"
    with pytest.raises(ref.RefIntegridadeError, match="divergem"):
        classifica(malha, trans=trans)


def test_classificar_polo_fora_do_escopo(malha):
    with pytest.raises(ref.RefIntegridadeError, match="polo"):
        classifica(malha, polo=FORA[0])


def test_classificar_polo_em_transicao_exige_incluir_transicao(malha):
    assert int(classifica(malha, polo=TRANS)["no_escopo"].sum()) == 216
    with pytest.raises(ref.RefIntegridadeError, match="polo"):
        classifica(malha, polo=TRANS, incluir=False)


# ----------------------------------------------------------- simplificar_gee
def escopo_de(malha):
    cls = classifica(malha)
    return cls[cls["no_escopo"]][["cod_ibge", "nome", "situacao", "geometry"]]


def test_simplificar_reduz_vertices_mantem_validade_ordem_e_crs(malha):
    esc = escopo_de(malha)
    g, m = ref.simplificar_gee(esc, "EPSG:4326", 0.001)
    assert g.crs.to_string() == "EPSG:4326" and len(g) == 216
    assert g["cod_ibge"].tolist() == esc["cod_ibge"].tolist()
    assert g.geometry.is_valid.all() and not g.geometry.is_empty.any()
    assert m["vertices_depois"] < m["vertices_antes"]
    assert m["var_area_max"] <= 0.01


def test_simplificar_preserva_ordem_lon_lat(malha):
    g, _ = ref.simplificar_gee(escopo_de(malha), "EPSG:4326", 0.001)
    x, y = g.geometry.iloc[0].exterior.coords[0]
    assert -46 < x < -44 and -11 < y < -9


@pytest.mark.parametrize("tol", [0, -0.001, 0.02])
def test_simplificar_rejeita_tolerancia_fora_do_intervalo(malha, tol):
    with pytest.raises(ref.RefSchemaError):
        ref.simplificar_gee(escopo_de(malha), "EPSG:4326", tol)


def test_simplificar_sem_vertices_para_remover_levanta_erro(malha):
    esc = escopo_de(malha).copy()
    esc["geometry"] = [box(*g.bounds) for g in esc.geometry]  # 5 vértices cada
    with pytest.raises(ref.RefIntegridadeError, match="não reduziu"):
        ref.simplificar_gee(esc, "EPSG:4326", 0.001)


def test_simplificar_que_distorce_a_area_levanta_erro():
    circulos = gpd.GeoDataFrame(
        {"cod_ibge": ["2200000"], "nome": ["A"], "situacao": ["res150"]},
        geometry=[Point(-45, -10).buffer(0.04, quad_segs=16)], crs="EPSG:4674")
    with pytest.raises(ref.RefIntegridadeError, match="área"):
        ref.simplificar_gee(circulos, "EPSG:4326", 0.009)


# ------------------------------------------------------------- hash_conteudo
def test_hash_nao_depende_da_ordem_das_linhas(malha):
    cls = classifica(malha)
    assert ref.hash_conteudo(cls) == ref.hash_conteudo(cls.sample(frac=1, random_state=3))


def test_hash_muda_com_a_situacao_de_um_municipio(malha):
    cls = classifica(malha)
    outro = cls.copy()
    outro.loc[outro["cod_ibge"] == TRANS, "situacao"] = "res150"
    assert ref.hash_conteudo(cls) != ref.hash_conteudo(outro)


def test_hash_muda_com_deslocamento_minimo_da_geometria(malha):
    cls = classifica(malha)
    outro = cls.copy()
    outro["geometry"] = cls.geometry.translate(1e-9, 0)
    assert ref.hash_conteudo(cls) != ref.hash_conteudo(outro)


# --------------------------------------------------------- gerar_referencia
def test_gerar_cria_os_quatro_arquivos_sem_sobras(projeto):
    res = ref.gerar_referencia(projeto)
    assert res.status == "gerado"
    assert res.contagens == {"malha": 224, "escopo": 216, "res150": 215, "transicao": 1, "fora": 8}
    assert sorted(p.name for p in dir_ref(projeto).iterdir() if p.is_file()) == ARQUIVOS_SAIDA
    assert not list(dir_ref(projeto).glob("*.tmp"))


def test_csv_do_escopo_formato_e_conteudo(projeto):
    ref.gerar_referencia(projeto)
    p = dir_ref(projeto) / NOME_CSV
    bruto = p.read_bytes()
    assert not bruto.startswith(b"\xef\xbb\xbf") and b"\r" not in bruto
    assert bruto.decode("utf-8").splitlines()[0] == "cod_ibge,nome,uf,situacao,area_km2"
    df = pd.read_csv(p, dtype={"cod_ibge": str})
    assert len(df) == 216 and df["cod_ibge"].str.len().eq(7).all()
    assert df["cod_ibge"].is_unique and df["cod_ibge"].tolist() == sorted(df["cod_ibge"])
    assert POLO in set(df["cod_ibge"]) and not set(FORA) & set(df["cod_ibge"])
    assert df["situacao"].value_counts().to_dict() == {"res150": 215, "transicao": 1}
    assert (df["uf"] == "PI").all() and (df["area_km2"] == 100.0).all()


def test_csv_sai_ordenado_mesmo_com_malha_em_ordem_invertida(tmp_path, zip_base):
    invertida = escreve_zip(malha_gdf().iloc[::-1].reset_index(drop=True), tmp_path / "inv" / "PI_Municipios_2022.zip")
    cfg = monta_projeto(tmp_path / "p", zip_base, zip_malha=invertida)
    ref.gerar_referencia(cfg)
    cods = ref.ler_municipios_escopo(cfg)["cod_ibge"].tolist()
    assert cods == sorted(cods) and len(cods) == 216


def test_geojson_para_gee_declara_crs_4326(projeto):
    ref.gerar_referencia(projeto)
    crs = json.dumps(ref.ler_geojson_gee(projeto).get("crs"))
    assert "4326" in crs or "CRS84" in crs


def test_geojson_para_gee_tem_216_features_codigo_texto_e_lon_lat(projeto):
    ref.gerar_referencia(projeto)
    fc = ref.ler_geojson_gee(projeto)
    assert fc["type"] == "FeatureCollection" and len(fc["features"]) == 216
    cods = [f["properties"]["cod_ibge"] for f in fc["features"]]
    assert all(isinstance(c, str) and len(c) == 7 for c in cods) and len(set(cods)) == 216
    x, y = fc["features"][0]["geometry"]["coordinates"][0][0][:2]
    assert -46 < x < -44 and -11 < y < -9


def test_parquet_guarda_os_224_em_4674_com_situacao_e_escopo(projeto):
    ref.gerar_referencia(projeto)
    g = gpd.read_parquet(dir_ref(projeto) / ref.PARQUET)
    assert len(g) == 224 and g.crs.to_string() == "EPSG:4674"
    assert int(g["no_escopo"].sum()) == 216 and set(g["situacao"]) == set(ref.SITUACOES)
    assert g["cod_ibge"].str.len().eq(7).all()


def test_selo_registra_hashes_entradas_parametros_e_confere(projeto):
    res = ref.gerar_referencia(projeto)
    selo = json.loads((dir_ref(projeto) / ref.SELO).read_text(encoding="utf-8"))
    assert sorted(selo["entradas"]) == sorted([ref.ZIP_MALHA, ref.CSV_ANEXO, ref.CSV_TRANSICAO])
    assert sorted(selo["arquivos"]) == sorted([NOME_CSV, ref.PARQUET, ref.GEOJSON])
    assert selo["parametros"]["incluir_transicao"] is True and selo["parametros"]["polo"] == POLO
    assert selo["conteudo_sha256"] == res.conteudo_sha256
    assert {"geopandas", "shapely", "pyogrio", "pyarrow", "pandas"} <= set(selo["bibliotecas"])
    assert ref.verificar_integridade(projeto)["conteudo_sha256"] == res.conteudo_sha256
    g = gpd.read_parquet(dir_ref(projeto) / ref.PARQUET)
    assert ref.hash_conteudo(g) == res.conteudo_sha256


def test_gerar_de_novo_nao_toca_nos_arquivos(projeto):
    primeiro = ref.gerar_referencia(projeto)
    antes = estado(dir_ref(projeto))
    segundo = ref.gerar_referencia(projeto)
    assert segundo.status == "ja_existente" and segundo.conteudo_sha256 == primeiro.conteudo_sha256
    assert estado(dir_ref(projeto)) == antes


def test_parametro_diferente_levanta_imutavel_e_nao_altera(projeto):
    ref.gerar_referencia(projeto)
    antes = estado(dir_ref(projeto))
    with pytest.raises(ref.RefImutavelError):
        ref.gerar_referencia(projeto, incluir_transicao=False)
    with pytest.raises(ref.RefImutavelError):
        ref.gerar_referencia(projeto, tolerancia_graus=0.002)
    assert estado(dir_ref(projeto)) == antes


def test_fonte_alterada_levanta_imutavel(projeto):
    ref.gerar_referencia(projeto)
    t = projeto.caminho("reference") / "fonte" / ref.CSV_TRANSICAO
    grava_csv(trans_df().assign(fonte="outra fonte"), t)
    with pytest.raises(ref.RefImutavelError):
        ref.gerar_referencia(projeto)


@pytest.mark.parametrize("alvo", [NOME_CSV, ref.PARQUET, ref.GEOJSON])
def test_arquivo_gerado_adulterado_levanta_integridade(projeto, alvo):
    ref.gerar_referencia(projeto)
    with open(dir_ref(projeto) / alvo, "ab") as f:
        f.write(b"x")
    with pytest.raises(ref.RefIntegridadeError, match=alvo):
        ref.gerar_referencia(projeto)
    with pytest.raises(ref.RefIntegridadeError):
        ref.verificar_integridade(projeto)


@pytest.mark.parametrize("alvo", [NOME_CSV, ref.PARQUET, ref.GEOJSON, ref.SELO])
def test_estado_parcial_levanta_integridade(projeto, alvo):
    ref.gerar_referencia(projeto)
    (dir_ref(projeto) / alvo).unlink()
    with pytest.raises(ref.RefIntegridadeError, match="parcial"):
        ref.gerar_referencia(projeto)


def test_leitores_exigem_referencia_gerada(projeto):
    with pytest.raises(FileNotFoundError):
        ref.ler_municipios_escopo(projeto)
    with pytest.raises(FileNotFoundError):
        ref.ler_geojson_gee(projeto)


def test_ler_municipios_escopo_devolve_216_com_codigo_texto(projeto):
    ref.gerar_referencia(projeto)
    df = ref.ler_municipios_escopo(projeto)
    assert len(df) == 216 and list(df.columns) == list(ref.COLS_CSV_ESCOPO)
    assert df["cod_ibge"].str.len().eq(7).all() and POLO in set(df["cod_ibge"])


def test_ler_municipios_escopo_recusa_csv_adulterado(projeto):
    ref.gerar_referencia(projeto)
    p = dir_ref(projeto) / NOME_CSV
    p.write_bytes(p.read_bytes().replace(b"Paulistana", b"Paulistano"))
    with pytest.raises(ref.RefIntegridadeError):
        ref.ler_municipios_escopo(projeto)


def selo_de(cfg) -> dict:
    return json.loads((dir_ref(cfg) / ref.SELO).read_text(encoding="utf-8"))


def test_sem_argumentos_usa_os_valores_do_params_real(projeto):
    res = ref.gerar_referencia(projeto)
    assert res.contagens["escopo"] == 216
    assert selo_de(projeto)["parametros"]["incluir_transicao"] is True
    assert selo_de(projeto)["parametros"]["tolerancia_graus"] == 0.001


def test_params_com_transicao_falsa_gera_215_sem_argumentos(tmp_path, zip_base):
    cfg = monta_projeto(tmp_path, zip_base, trocas={"incluir_transicao: true": "incluir_transicao: false"})
    assert ref.gerar_referencia(cfg).contagens["escopo"] == 215
    assert selo_de(cfg)["parametros"]["incluir_transicao"] is False


def test_params_com_outra_tolerancia_vai_para_o_selo(tmp_path, zip_base):
    cfg = monta_projeto(tmp_path, zip_base, trocas={"simplificacao_graus: 0.001": "simplificacao_graus: 0.002"})
    ref.gerar_referencia(cfg)
    assert selo_de(cfg)["parametros"]["tolerancia_graus"] == 0.002


def test_argumento_explicito_prevalece_sobre_o_params(tmp_path, zip_base):
    cfg = monta_projeto(tmp_path, zip_base, trocas={"incluir_transicao: true": "incluir_transicao: false",
                                                    "simplificacao_graus: 0.001": "simplificacao_graus: 0.002"})
    res = ref.gerar_referencia(cfg, incluir_transicao=True, tolerancia_graus=0.003)
    assert res.contagens["escopo"] == 216
    assert selo_de(cfg)["parametros"]["tolerancia_graus"] == 0.003


def test_mudar_o_params_depois_de_gerar_levanta_imutavel(tmp_path, zip_base):
    cfg = monta_projeto(tmp_path, zip_base)
    ref.gerar_referencia(cfg)
    outro = load_config_de(cfg, {"incluir_transicao: true": "incluir_transicao: false"})
    with pytest.raises(ref.RefImutavelError):
        ref.gerar_referencia(outro)


def test_sem_transicao_gera_215_no_escopo(projeto):
    res = ref.gerar_referencia(projeto, incluir_transicao=False)
    assert res.contagens["escopo"] == 215
    assert len(ref.ler_municipios_escopo(projeto)) == 215
    assert len(ref.ler_geojson_gee(projeto)["features"]) == 215


def test_falha_de_validacao_nao_deixa_arquivos(projeto):
    fonte = projeto.caminho("reference") / "fonte"
    grava_csv(anexo_df().iloc[:-1], fonte / ref.CSV_ANEXO)
    with pytest.raises(ref.RefIntegridadeError):
        ref.gerar_referencia(projeto)
    assert [p.name for p in dir_ref(projeto).iterdir()] == ["fonte"]


def test_tolerancia_invalida_nao_deixa_arquivos(projeto):
    with pytest.raises(ref.RefSchemaError):
        ref.gerar_referencia(projeto, tolerancia_graus=0)
    assert [p.name for p in dir_ref(projeto).iterdir()] == ["fonte"]


def test_fonte_ausente_lista_o_que_falta(projeto):
    (projeto.caminho("reference") / "fonte" / ref.CSV_TRANSICAO).unlink()
    with pytest.raises(FileNotFoundError, match=ref.CSV_TRANSICAO):
        ref.gerar_referencia(projeto)


def test_ano_da_malha_pendente_ou_diferente_de_2022(tmp_path, zip_base):
    with pytest.raises(PendenciaError):
        ref.gerar_referencia(monta_projeto(tmp_path / "a", zip_base, ano="VERIFICAR"))
    with pytest.raises(ref.RefSchemaError, match="2022"):
        ref.gerar_referencia(monta_projeto(tmp_path / "b", zip_base, ano="2023"))


# ---------------------------------------------------------------- _atomico
def test_atomico_falha_na_escrita_preserva_o_destino_e_remove_o_tmp(tmp_path):
    destino = tmp_path / "saida.txt"
    destino.write_bytes(b"antigo")

    def escreve_e_quebra(tmp):
        tmp.write_bytes(b"pela metade")
        raise OSError("disco cheio")

    with pytest.raises(OSError):
        ref._atomico(destino, escreve_e_quebra)
    assert destino.read_bytes() == b"antigo" and sorted(p.name for p in tmp_path.iterdir()) == ["saida.txt"]


# ------------------------------------------------------ dados reais (opcional)
def test_dados_reais_224_215_216_e_polo(tmp_path):
    cfg = load_config(PARAMS_REAL)
    fonte = cfg.caminho("reference") / "fonte"
    if not all((fonte / n).is_file() for n in (ref.ZIP_MALHA, ref.CSV_ANEXO, ref.CSV_TRANSICAO)):
        pytest.skip("fontes reais não estão em data/reference/fonte")
    malha = ref.ler_malha_ibge(fonte / ref.ZIP_MALHA, cfg.escopo.crs_origem)
    cls = ref.classificar(malha, ref.ler_lista(fonte / ref.CSV_ANEXO),
                          ref.ler_lista(fonte / ref.CSV_TRANSICAO, ("fonte",)), True,
                          cfg.escopo.piloto.polo_cod_ibge)
    assert len(malha) == 224 and int(cls["no_escopo"].sum()) == 216
    assert cls["situacao"].value_counts().to_dict() == {"res150": 215, "fora": 8, "transicao": 1}
    assert cls.loc[cls["cod_ibge"] == "2205706", "situacao"].item() == "transicao"
    assert cls.loc[cls["cod_ibge"] == "2207801", "nome"].item() == "Paulistana"
