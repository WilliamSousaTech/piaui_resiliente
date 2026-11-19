"""Testes do .gitattributes (fim de linha fixo em LF).

Protege o hash: o SHA-256 do params.yaml (manifest) e o dos CSVs de referência (selo) não podem
depender de o Git ter convertido LF em CRLF. Não lê o working tree (no Windows ele pode estar em CRLF):
confere a política do arquivo e, se houver `git`, o que o próprio Git resolve para cada caminho.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
ARQUIVO = RAIZ / ".gitattributes"

TEXTO = ["*.py", "*.md", "*.txt", "*.toml", "*.yaml", "*.yml", "*.json", "*.geojson", "*.csv", "*.html"]
BINARIOS = ["*.parquet", "*.zip", "*.pdf", "*.png", "*.xlsx", "*.npy", "*.pkl", "*.pt", "*.pth", "*.woff2"]


def regras() -> dict[str, set[str]]:
    """padrão -> conjunto de atributos (comentários e linhas em branco ignorados)."""
    out: dict[str, set[str]] = {}
    for bruta in ARQUIVO.read_text(encoding="utf-8").splitlines():
        linha = bruta.strip()
        if not linha or linha.startswith("#"):
            continue
        padrao, *attrs = linha.split()
        out.setdefault(padrao, set()).update(attrs)
    return out


def test_arquivo_existe_na_raiz_e_nao_esta_vazio():
    assert ARQUIVO.is_file(), "falta o .gitattributes na raiz do projeto"
    assert regras()


def test_regra_geral_text_auto_com_lf():
    assert {"text=auto", "eol=lf"} <= regras()["*"]


@pytest.mark.parametrize("padrao", TEXTO)
def test_extensoes_de_texto_fixam_lf(padrao):
    attrs = regras().get(padrao)
    assert attrs is not None, f"{padrao} sem regra"
    assert "text" in attrs and "eol=lf" in attrs


@pytest.mark.parametrize("padrao", BINARIOS)
def test_binarios_nao_sofrem_conversao(padrao):
    attrs = regras().get(padrao)
    assert attrs is not None, f"{padrao} sem regra"
    assert "binary" in attrs


def test_nenhuma_regra_usa_crlf():
    assert not any(a.startswith("eol=crlf") for attrs in regras().values() for a in attrs)


# ---- o que o Git realmente resolve (precisa de `git` instalado) -------------------------------
CASOS_GIT = [
    ("params.yaml", {"eol": "lf", "text": "set"}),
    ("data/reference/municipios_semiarido.csv", {"eol": "lf", "text": "set"}),
    ("data/reference/malha_pi_2022_gee.geojson", {"eol": "lf", "text": "set"}),
    ("data/reference/referencia.sha256.json", {"eol": "lf", "text": "set"}),
    ("docs/contrato_de_dados.md", {"eol": "lf", "text": "set"}),
    ("src/data/spi.py", {"eol": "lf", "text": "set"}),
    ("data/raw/chirps/chirps_mensal_1981-01_1982-06.parquet", {"text": "unset"}),
    ("data/reference/fonte/PI_Municipios_2022.zip", {"text": "unset"}),
    ("data/reference/fonte/resolucao150.pdf", {"text": "unset"}),
    ("artifacts/modelo.pt", {"text": "unset"}),
    ("src/web/assets/AtkinsonHyperlegibleNext-latin-wght.woff2", {"text": "unset"}),
    ("src/web/template.html", {"eol": "lf", "text": "set"}),
]


@pytest.fixture(scope="module")
def repo_temporario(tmp_path_factory):
    if shutil.which("git") is None:
        pytest.skip("git não está instalado")
    d = tmp_path_factory.mktemp("gitattrs")
    subprocess.run(["git", "init", "-q"], cwd=d, check=True)
    (d / ".gitattributes").write_bytes(ARQUIVO.read_bytes())
    return d


@pytest.mark.parametrize("caminho,esperado", CASOS_GIT)
def test_git_resolve_os_atributos_esperados(repo_temporario, caminho, esperado):
    r = subprocess.run(["git", "check-attr", "-a", "--", caminho], cwd=repo_temporario,
                       capture_output=True, text=True, check=True)
    obtido = {}
    for linha in r.stdout.splitlines():
        _, nome, valor = (p.strip() for p in linha.split(":", 2))
        obtido[nome] = valor
    for nome, valor in esperado.items():
        assert obtido.get(nome) == valor, f"{caminho}: {nome}={obtido.get(nome)!r}, esperado {valor!r} ({r.stdout!r})"
