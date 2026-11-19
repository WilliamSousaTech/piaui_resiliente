"""Testes de src/data/manifest.py. Sem internet; arquivos de exemplo só aqui (nunca em data/)."""
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import manifest as mf
from src.data import reference, splits
from src.data.config import load_config

PARAMS_REAL = Path(__file__).resolve().parents[1] / "params.yaml"
GITATTRIBUTES = Path(__file__).resolve().parents[1] / ".gitattributes"
AGORA = datetime(2026, 11, 17, 12, 30, 45, 123456, tzinfo=timezone.utc)
HASH_REF = "ab" * 32


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


@pytest.fixture
def proj(tmp_path):
    """Projeto de mentira numa pasta temporária: params.yaml real, selo da referência e 3 arquivos."""
    shutil.copy(PARAMS_REAL, tmp_path / "params.yaml")
    ref = tmp_path / "data" / "reference"
    ref.mkdir(parents=True)
    (ref / "referencia.sha256.json").write_text(json.dumps(
        {"conteudo_sha256": HASH_REF, "contagens": {"malha": 224, "escopo": 216},
         "entradas": {"PI_Municipios_2022.zip": "cd" * 32}, "parametros": {"incluir_transicao": True}}),
        encoding="utf-8")
    (ref / "municipios_semiarido.csv").write_bytes("cod_ibge,nome\n2200053,Acauã\n2200251,Alagoinha do Piauí\n".encode("utf-8"))
    cur = tmp_path / "data" / "curated"
    cur.mkdir(parents=True)
    pd.DataFrame({"cod_ibge": ["2200053", "2200251"], "mes": pd.to_datetime(["2000-01-01", "2000-02-01"]),
                  "precip_mm": np.array([10.5, 20.0], dtype="float32"), "ok": [True, False]}
                 ).to_parquet(cur / "curated_mensal.parquet", index=False)
    art = tmp_path / "artifacts"
    art.mkdir()
    (art / "nota.txt").write_bytes(b"linha 1\nlinha 2\n")
    cfg = load_config(tmp_path / "params.yaml")
    arquivos = {"curated": "data/curated/curated_mensal.parquet",
                "municipios": "data/reference/municipios_semiarido.csv",
                "nota": "artifacts/nota.txt"}
    return cfg, arquivos, tmp_path


def construir(proj, **kw):
    cfg, arquivos, _ = proj
    return mf.construir_manifest(cfg, arquivos, agora=AGORA, **kw)


# ================================================================== hash
def test_hash_de_texto_ignora_crlf(tmp_path):
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    a.write_bytes(b"x,y\n1,2\n3,4\n")
    b.write_bytes(b"x,y\r\n1,2\r\n3,4\r\n")
    assert mf.sha256_arquivo(a) == mf.sha256_arquivo(b) == sha(b"x,y\n1,2\n3,4\n")


def test_hash_de_binario_usa_bytes_brutos(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.parquet"
    dados = b"ab\r\ncd\r\n"
    a.write_bytes(dados)
    b.write_bytes(dados)
    assert mf.sha256_arquivo(a) == mf.sha256_arquivo(b) == sha(dados)


@pytest.mark.parametrize("bloco", [1, 2, 3, 4, 5, 7, 64])
@pytest.mark.parametrize("conteudo", [b"ab\r\ncd\r\nef", b"\r\n", b"a\r\r\nb", b"x\r", b"\r\r\r\n\n", b"", b"sem fim de linha"])
def test_hash_de_texto_nao_depende_do_tamanho_do_bloco(tmp_path, monkeypatch, bloco, conteudo):
    """\\r\\n partido entre dois blocos de leitura tem que dar o mesmo hash do arquivo inteiro."""
    p = tmp_path / "t.yaml"
    p.write_bytes(conteudo)
    monkeypatch.setattr(mf, "_BLOCO", bloco)
    assert mf.sha256_arquivo(p) == sha(conteudo.replace(b"\r\n", b"\n"))


def test_extensoes_de_texto_coincidem_com_o_gitattributes():
    texto = set()
    for linha in GITATTRIBUTES.read_text(encoding="utf-8").splitlines():
        partes = linha.split()
        if len(partes) >= 2 and partes[0].startswith("*.") and "eol=lf" in partes:
            texto.add(partes[0][1:])
    assert texto == set(mf.EXTENSOES_TEXTO)


def test_nome_do_selo_da_referencia_acompanha_reference():
    assert mf.SELO_REFERENCIA == reference.SELO


# ================================================================== construção
def test_estrutura_e_campos_basicos(proj):
    cfg, arquivos, raiz = proj
    m = construir(proj)
    assert m["versao_manifest"] == 1 and m["algoritmo_hash"] == "sha256"
    assert "experimental" in m["maturity"].lower() and m["maturity"] == cfg.projeto.maturidade
    assert m["params_sha256"] == mf.sha256_arquivo(raiz / "params.yaml")
    assert m["projeto"] == {"nome": cfg.projeto.nome, "equipe": cfg.projeto.equipe,
                            "versao_params": cfg.projeto.versao_params, "seed": cfg.projeto.seed}
    assert m["janela"] == {"inicio": "1981-01", "fim": "2026-09"}
    assert m["alvo"] == {"escala_spi_meses": 3, "horizonte_h": 1, "janelas_spi": [3, 6]}
    assert m["escopo"]["incluir_transicao"] is True and m["escopo"]["modo"] == cfg.escopo.modo
    assert m["fontes"]["chirps"]["gee_id"] == "UCSB-CHG/CHIRPS/DAILY"
    assert m["fontes"]["mapbiomas"] == {"colecao": 10, "defasagem_anos": 1}
    assert m["fontes"]["s2id"]["cobrade"] == ["1.4.1.1.0", "1.4.1.2.0"]
    assert set(m["ambiente"]) == {"python", "bibliotecas"} and "pandas" in m["ambiente"]["bibliotecas"]
    json.dumps(m, allow_nan=False)                      # JSON puro


def test_cortes_e_embargo_vem_do_splits(proj):
    cfg, _, _ = proj
    m = construir(proj)
    assert m["split"] == {**splits.cortes(cfg).para_manifest(), "teste_liberado": False}
    assert m["split"]["embargo_meses"] == 5 and m["split"]["treino_fim"] == "2015-12"


def test_referencia_vem_do_selo(proj):
    m = construir(proj)
    assert m["referencia"] == {"conteudo_sha256": HASH_REF, "contagens": {"malha": 224, "escopo": 216},
                               "entradas": {"PI_Municipios_2022.zip": "cd" * 32},
                               "parametros": {"incluir_transicao": True}}


def test_data_em_utc_sem_microssegundos_e_com_fuso(proj):
    assert construir(proj)["gerado_em_utc"] == "2026-11-17T12:30:45Z"
    cfg, arquivos, _ = proj
    fortaleza = datetime(2026, 11, 17, 9, 30, 45, tzinfo=timezone(timedelta(hours=-3)))
    assert mf.construir_manifest(cfg, arquivos, agora=fortaleza)["gerado_em_utc"] == "2026-11-17T12:30:45Z"


def test_data_sem_fuso_e_recusada(proj):
    cfg, arquivos, _ = proj
    with pytest.raises(mf.ManifestError, match="fuso"):
        mf.construir_manifest(cfg, arquivos, agora=datetime(2026, 11, 17, 12, 0))


def test_agora_padrao_e_utc_atual(proj):
    cfg, arquivos, _ = proj
    antes = datetime.now(timezone.utc).replace(microsecond=0)
    m = mf.construir_manifest(cfg, arquivos)
    gerado = datetime.strptime(m["gerado_em_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    assert antes <= gerado <= datetime.now(timezone.utc) + timedelta(seconds=1)


def test_arquivos_relativos_ordenados_e_descritos(proj):
    _, _, raiz = proj
    m = construir(proj)
    assert list(m["arquivos"]) == ["curated", "municipios", "nota"]
    assert m["arquivos"]["curated"]["caminho"] == "data/curated/curated_mensal.parquet"
    assert m["arquivos"]["nota"] == {"caminho": "artifacts/nota.txt", "sha256": sha(b"linha 1\nlinha 2\n")}
    pq = m["arquivos"]["curated"]
    assert pq["linhas"] == 2 and pq["colunas"] == ["cod_ibge", "mes", "precip_mm", "ok"]
    assert pq["dtypes"]["precip_mm"] == "float" and pq["dtypes"]["ok"] == "bool"
    assert pq["dtypes"]["mes"].startswith("timestamp")
    assert pq["sha256"] == sha((raiz / "data/curated/curated_mensal.parquet").read_bytes())


def test_arquivos_saem_ordenados_pelo_nome_mesmo_se_a_entrada_nao_estiver(proj):
    cfg, arquivos, _ = proj
    desordenado = {"nota": arquivos["nota"], "municipios": arquivos["municipios"], "curated": arquivos["curated"]}
    m = mf.construir_manifest(cfg, desordenado, agora=AGORA)
    assert list(m["arquivos"]) == ["curated", "municipios", "nota"]
    assert m == construir(proj)


def test_deterministico_com_a_mesma_data(proj):
    assert construir(proj) == construir(proj)


def test_caminho_absoluto_dentro_da_raiz_vira_relativo(proj):
    cfg, _, raiz = proj
    m = mf.construir_manifest(cfg, {"nota": raiz / "artifacts" / "nota.txt"}, agora=AGORA)
    assert m["arquivos"]["nota"]["caminho"] == "artifacts/nota.txt"


def test_dataset_sha256_muda_com_arquivo_e_com_params_mas_nao_com_a_data(proj):
    cfg, arquivos, raiz = proj
    base = construir(proj)["dataset_sha256"]
    outro_dia = mf.construir_manifest(cfg, arquivos, agora=AGORA + timedelta(days=3))
    assert outro_dia["dataset_sha256"] == base
    assert construir(proj, linhas_por_bloco={"train": 1}, extras={"a": 1})["dataset_sha256"] == base
    (raiz / "artifacts" / "nota.txt").write_bytes(b"outra\n")
    assert construir(proj)["dataset_sha256"] != base
    (raiz / "artifacts" / "nota.txt").write_bytes(b"linha 1\nlinha 2\n")
    assert construir(proj)["dataset_sha256"] == base
    (raiz / "params.yaml").write_bytes((raiz / "params.yaml").read_bytes() + b"# mudou\n")
    assert construir(proj)["dataset_sha256"] != base


def test_linhas_por_bloco_e_extras_opcionais(proj):
    m = construir(proj, linhas_por_bloco={"train": 10, "val": 5, "test": 0, "embargo": 3},
                  extras={"preparo": {"edicao": "MUNIC 2020", "variavel": "MGRD05"}})
    assert m["linhas_por_bloco"] == {"train": 10, "val": 5, "test": 0, "embargo": 3}
    assert m["extras"]["preparo"]["edicao"] == "MUNIC 2020"
    sem = construir(proj)
    assert "linhas_por_bloco" not in sem and "extras" not in sem


@pytest.mark.parametrize("blocos", [{"treino": 1}, {"train": -1}, {"train": True}, {"train": 1.5}, {"train": "7"}])
def test_linhas_por_bloco_invalido(proj, blocos):
    with pytest.raises(mf.ManifestError, match="linhas_por_bloco"):
        construir(proj, linhas_por_bloco=blocos)


@pytest.mark.parametrize("extras", [{"x": float("nan")}, {"x": {1, 2}}, {"x": object()}])
def test_extras_precisam_ser_json_puro(proj, extras):
    with pytest.raises(mf.ManifestError, match="JSON puro"):
        construir(proj, extras=extras)


# ================================================================== entradas inválidas
def test_sem_arquivos(proj):
    cfg, _, _ = proj
    with pytest.raises(mf.ManifestError, match="ao menos um"):
        mf.construir_manifest(cfg, {})


@pytest.mark.parametrize("nome", ["", "   ", 3, None])
def test_nome_logico_invalido(proj, nome):
    cfg, _, _ = proj
    with pytest.raises(mf.ManifestError, match="nome lógico"):
        mf.construir_manifest(cfg, {nome: "artifacts/nota.txt"})


def test_arquivo_inexistente_e_pasta(proj):
    cfg, _, _ = proj
    with pytest.raises(mf.ManifestError, match="inexistente"):
        mf.construir_manifest(cfg, {"x": "artifacts/nao_existe.txt"})
    with pytest.raises(mf.ManifestError, match="inexistente"):
        mf.construir_manifest(cfg, {"x": "artifacts"})


def test_caminho_fora_da_raiz(proj, tmp_path_factory):
    cfg, _, _ = proj
    fora = tmp_path_factory.mktemp("fora") / "intruso.txt"
    fora.write_text("x", encoding="utf-8")
    with pytest.raises(mf.ManifestError, match="fora da raiz"):
        mf.construir_manifest(cfg, {"x": fora})
    with pytest.raises(mf.ManifestError, match="fora da raiz"):
        mf.construir_manifest(cfg, {"x": "../intruso.txt"})
    with pytest.raises(mf.ManifestError, match="fora da raiz"):
        mf.construir_manifest(cfg, {"x": "artifacts/../../intruso.txt"})


def test_link_simbolico_que_escapa_da_raiz(proj, tmp_path_factory):
    cfg, _, raiz = proj
    fora = tmp_path_factory.mktemp("fora2") / "alvo.txt"
    fora.write_text("x", encoding="utf-8")
    try:
        os.symlink(fora, raiz / "artifacts" / "link.txt")
    except (OSError, NotImplementedError):
        pytest.skip("sem permissão para criar link simbólico")
    with pytest.raises(mf.ManifestError, match="fora da raiz"):
        mf.construir_manifest(cfg, {"x": "artifacts/link.txt"})


def test_mesmo_arquivo_em_dois_nomes(proj):
    cfg, _, _ = proj
    with pytest.raises(mf.ManifestError, match="dois nomes"):
        mf.construir_manifest(cfg, {"a": "artifacts/nota.txt", "b": "./artifacts/../artifacts/nota.txt"})


def test_manifest_nao_lista_a_si_mesmo(proj):
    cfg, arquivos, raiz = proj
    destino = raiz / cfg.manifest.arquivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("{}", encoding="utf-8")
    with pytest.raises(mf.ManifestError, match="a si mesmo"):
        mf.construir_manifest(cfg, {**arquivos, "eu": cfg.manifest.arquivo})


def test_params_ausente(proj):
    cfg, arquivos, raiz = proj
    (raiz / "params.yaml").unlink()
    with pytest.raises(mf.ManifestError, match="params.yaml"):
        mf.construir_manifest(cfg, arquivos)


def test_selo_da_referencia_ausente_ou_ruim(proj):
    cfg, arquivos, raiz = proj
    selo = raiz / "data" / "reference" / "referencia.sha256.json"
    original = selo.read_text(encoding="utf-8")
    selo.write_text("{ não é json", encoding="utf-8")
    with pytest.raises(mf.ManifestError, match="ilegível"):
        mf.construir_manifest(cfg, arquivos)
    selo.write_text(json.dumps({"conteudo_sha256": "curto", "contagens": {}}), encoding="utf-8")
    with pytest.raises(mf.ManifestError, match="64 hex"):
        mf.construir_manifest(cfg, arquivos)
    selo.write_text(json.dumps({"conteudo_sha256": HASH_REF}), encoding="utf-8")
    with pytest.raises(mf.ManifestError, match="contagens"):
        mf.construir_manifest(cfg, arquivos)
    selo.unlink()
    with pytest.raises(mf.ManifestError, match="ausente"):
        mf.construir_manifest(cfg, arquivos)
    selo.write_text(original, encoding="utf-8")
    assert mf.construir_manifest(cfg, arquivos)["referencia"]["conteudo_sha256"] == HASH_REF


def test_parquet_ilegivel(proj):
    cfg, _, raiz = proj
    (raiz / "artifacts" / "quebrado.parquet").write_bytes(b"isto nao e parquet")
    with pytest.raises(mf.ManifestError, match="Parquet ilegível"):
        mf.construir_manifest(cfg, {"x": "artifacts/quebrado.parquet"})


# ================================================================== gravação
def gravar(proj, **kw):
    cfg, _, _ = proj
    return mf.gravar_manifest(cfg, construir(proj), **kw)


def test_gravar_em_lf_ordenado_e_sem_resto_temporario(proj):
    cfg, _, raiz = proj
    assert gravar(proj) == "gerado"
    destino = raiz / cfg.manifest.arquivo
    bruto = destino.read_bytes()
    assert b"\r" not in bruto and bruto.endswith(b"\n")
    assert json.loads(bruto) == construir(proj)
    chaves = list(json.loads(bruto))
    assert chaves == sorted(chaves)
    assert not list(destino.parent.glob("*.tmp"))


def test_gravar_de_novo_com_o_mesmo_conjunto_nao_reescreve(proj):
    cfg, arquivos, raiz = proj
    gravar(proj)
    destino = raiz / cfg.manifest.arquivo
    antes = destino.read_bytes()
    outro_dia = mf.construir_manifest(cfg, arquivos, agora=AGORA + timedelta(days=5))
    assert mf.gravar_manifest(cfg, outro_dia) == "ja_existente"
    assert destino.read_bytes() == antes                 # a data antiga permanece


def test_conjunto_diferente_exige_substituir(proj):
    cfg, _, raiz = proj
    gravar(proj)
    (raiz / "artifacts" / "nota.txt").write_bytes(b"mudou\n")
    novo = construir(proj)
    with pytest.raises(mf.ManifestImutavelError):
        mf.gravar_manifest(cfg, novo)
    assert mf.gravar_manifest(cfg, novo, substituir=True) == "substituido"
    assert json.loads((raiz / cfg.manifest.arquivo).read_text(encoding="utf-8"))["dataset_sha256"] == novo["dataset_sha256"]


def test_gravar_recusa_manifest_adulterado_ou_incompleto(proj):
    cfg, _, _ = proj
    base = construir(proj)
    adulterado = json.loads(json.dumps(base))
    adulterado["arquivos"]["nota"]["sha256"] = "0" * 64
    with pytest.raises(mf.ManifestError, match="dataset_sha256"):
        mf.gravar_manifest(cfg, adulterado)
    sem_aviso = {k: v for k, v in base.items() if k != "maturity"}
    with pytest.raises(mf.ManifestError, match="maturity"):
        mf.gravar_manifest(cfg, sem_aviso)
    with pytest.raises(mf.ManifestError, match="JSON puro"):
        mf.gravar_manifest(cfg, {**base, "x": object()})
    with pytest.raises(mf.ManifestError, match="JSON puro"):
        mf.gravar_manifest(cfg, {**base, "x": float("nan")})
    with pytest.raises(mf.ManifestError, match="versao_manifest"):
        mf.gravar_manifest(cfg, {**base, "versao_manifest": 2})


def test_manifest_existente_ilegivel(proj):
    cfg, _, raiz = proj
    destino = raiz / cfg.manifest.arquivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("lixo {", encoding="utf-8")
    with pytest.raises(mf.ManifestIntegridadeError, match="ilegível"):
        gravar(proj)


# ================================================================== verificação
def test_verificar_manifest_integro(proj):
    cfg, _, _ = proj
    gravar(proj)
    m = mf.verificar_manifest(cfg)
    assert m["dataset_sha256"] == construir(proj)["dataset_sha256"]


def test_arquivo_texto_regravado_com_crlf_continua_integro(proj):
    """Simula o checkout no Windows: o texto vira CRLF e o manifest tem que continuar valendo."""
    cfg, _, raiz = proj
    gravar(proj)
    csv = raiz / "data" / "reference" / "municipios_semiarido.csv"
    csv.write_bytes(csv.read_bytes().replace(b"\n", b"\r\n"))
    mf.verificar_manifest(cfg)
    assert mf.caminho_verificado(cfg, "municipios") == csv.resolve()


def test_arquivo_alterado_e_detectado_pelo_nome(proj):
    cfg, _, raiz = proj
    gravar(proj)
    (raiz / "artifacts" / "nota.txt").write_bytes(b"adulterado\n")
    with pytest.raises(mf.ManifestIntegridadeError, match=r"nota: SHA-256 difere"):
        mf.verificar_manifest(cfg)


def test_binario_alterado_e_detectado(proj):
    cfg, _, raiz = proj
    gravar(proj)
    pq = raiz / "data" / "curated" / "curated_mensal.parquet"
    pq.write_bytes(pq.read_bytes() + b"\x00")
    with pytest.raises(mf.ManifestIntegridadeError, match=r"curated: SHA-256 difere"):
        mf.verificar_manifest(cfg)


def test_arquivo_ausente_e_varios_problemas_de_uma_vez(proj):
    cfg, _, raiz = proj
    gravar(proj)
    (raiz / "artifacts" / "nota.txt").unlink()
    (raiz / "data" / "reference" / "municipios_semiarido.csv").write_bytes(b"outro\n")
    with pytest.raises(mf.ManifestIntegridadeError) as e:
        mf.verificar_manifest(cfg)
    assert "nota: arquivo ausente" in str(e.value) and "municipios: SHA-256 difere" in str(e.value)


def test_params_alterado_so_falha_quando_conferido(proj):
    cfg, _, raiz = proj
    gravar(proj)
    (raiz / "params.yaml").write_bytes((raiz / "params.yaml").read_bytes() + b"# mudou\n")
    with pytest.raises(mf.ManifestIntegridadeError, match="params.yaml: mudou"):
        mf.verificar_manifest(cfg)
    assert mf.verificar_manifest(cfg, conferir_params=False)["maturity"]


def test_manifest_ausente_ou_ilegivel(proj):
    cfg, _, raiz = proj
    with pytest.raises(mf.ManifestIntegridadeError, match="ausente"):
        mf.verificar_manifest(cfg)
    destino = raiz / cfg.manifest.arquivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text("{ quebrado", encoding="utf-8")
    with pytest.raises(mf.ManifestIntegridadeError, match="ilegível"):
        mf.verificar_manifest(cfg)


def editar_manifest(proj, editar):
    cfg, _, raiz = proj
    destino = raiz / cfg.manifest.arquivo
    m = json.loads(destino.read_text(encoding="utf-8"))
    editar(m)
    destino.write_text(json.dumps(m), encoding="utf-8")


def test_hash_editado_no_json_nao_confere_com_o_dataset(proj):
    cfg, _, _ = proj
    gravar(proj)
    editar_manifest(proj, lambda m: m["arquivos"]["nota"].update(sha256="1" * 64))
    with pytest.raises(mf.ManifestIntegridadeError, match="dataset_sha256"):
        mf.verificar_manifest(cfg)


@pytest.mark.parametrize("campo", ["maturity", "arquivos", "params_sha256", "dataset_sha256"])
def test_manifest_sem_campo_obrigatorio(proj, campo):
    cfg, _, _ = proj
    gravar(proj)
    editar_manifest(proj, lambda m: m.pop(campo))
    with pytest.raises(mf.ManifestIntegridadeError, match="manifest inválido"):
        mf.verificar_manifest(cfg)


@pytest.mark.parametrize("inseguro", ["../intruso.txt", "/etc/passwd", "C:/Windows/x.txt", "artifacts\\nota.txt",
                                      "artifacts/../../x", "artifacts/../artifacts/nota.txt", "", 7])
def test_caminho_inseguro_no_json_e_recusado(proj, inseguro):
    cfg, _, _ = proj
    gravar(proj)
    editar_manifest(proj, lambda m: m["arquivos"]["nota"].update(caminho=inseguro))
    if isinstance(inseguro, int):
        with pytest.raises(mf.ManifestIntegridadeError, match="sem 'caminho'"):
            mf.verificar_manifest(cfg)
    else:
        with pytest.raises(mf.ManifestIntegridadeError, match="inseguro"):
            mf.verificar_manifest(cfg)
        with pytest.raises(mf.ManifestIntegridadeError, match="inseguro"):
            mf.caminho_verificado(cfg, "nota")


# ================================================================== caminho_verificado
def test_caminho_verificado_devolve_o_arquivo_integro(proj):
    cfg, _, raiz = proj
    gravar(proj)
    p = mf.caminho_verificado(cfg, "curated")
    assert p == (raiz / "data" / "curated" / "curated_mensal.parquet").resolve() and p.is_file()


def test_caminho_verificado_nome_desconhecido(proj):
    cfg, _, _ = proj
    gravar(proj)
    with pytest.raises(mf.ManifestError, match="não consta"):
        mf.caminho_verificado(cfg, "inexistente")


def test_caminho_verificado_confere_so_o_arquivo_pedido(proj):
    cfg, _, raiz = proj
    gravar(proj)
    (raiz / "artifacts" / "nota.txt").write_bytes(b"adulterado\n")
    assert mf.caminho_verificado(cfg, "curated").is_file()          # o outro arquivo está íntegro
    with pytest.raises(mf.ManifestIntegridadeError, match="nota"):
        mf.caminho_verificado(cfg, "nota")


def test_caminho_verificado_sem_manifest(proj):
    cfg, _, _ = proj
    with pytest.raises(mf.ManifestIntegridadeError, match="ausente"):
        mf.caminho_verificado(cfg, "curated")
