"""Testes de src/data/config.py (Item 1). Rodam sem internet e sem GEE."""
import re
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from src.data.config import (Config, ConfigError, PendenciaError, exigir_resolvido,
                             load_config)

PARAMS_REAL = Path(__file__).resolve().parents[1] / "params.yaml"
CHAVES_PENDENTES = [("escopo", "piloto", "composicao_status"), ("fontes", "inmet", "estacoes")]


@pytest.fixture
def base():
    return yaml.safe_load(PARAMS_REAL.read_text(encoding="utf-8"))


def escreve(tmp_path, dados, nome="params.yaml"):
    p = tmp_path / nome
    p.write_text(yaml.safe_dump(dados, allow_unicode=True), encoding="utf-8")
    return p


def define(d, caminho, valor):
    for k in caminho[:-1]:
        d = d[k]
    d[caminho[-1]] = valor


def resolve_pendencias(d):
    for c in CHAVES_PENDENTES:
        define(d, c, "resolvido")


# 1. smoke ------------------------------------------------------------------
def test_params_real_carrega():
    cfg = load_config(PARAMS_REAL)
    assert isinstance(cfg.split.treino_fim, pd.Period)
    assert cfg.split.treino_fim < cfg.split.val_fim < cfg.split.teste_fim
    assert cfg.split.teste_liberado is False  # teste congelado na Fase 1
    assert cfg.fontes.s2id.entra_como_feature is False
    assert cfg.fontes.mapbiomas.defasagem_anos >= 1


def test_params_real_versao_0_2_1():
    cfg = load_config(PARAMS_REAL)
    assert cfg.projeto.versao_params == "0.2.1"
    assert cfg.spi.janelas_meses == (3, 6) and cfg.spi.convencao_zero == "q"
    assert cfg.split.embargo_meses == 5 and cfg.alvo.escala_spi_meses in cfg.spi.janelas_meses
    assert cfg.fontes.chirps.limiar_dia_chuva_mm == 1.0
    assert cfg.escopo.incluir_transicao is True
    assert cfg.escopo.simplificacao_graus == 0.001
    assert "escopo.fonte_semiarido" not in cfg.pendencias()  # fonte descrita sem travar módulos
    assert set(cfg.pendencias()) == {"escopo.piloto.composicao_status", "fontes.inmet.estacoes"}


# 2-14. violações do contrato -------------------------------------------------
VIOLACOES = [
    ("chave extra", ("spi", "typo"), 1, "spi.typo"),
    ("mes invalido", ("split", "treino_fim"), "2015-13", "AAAA-MM"),
    ("mes sem aspas/formato", ("split", "val_fim"), "2020/12", "AAAA-MM"),
    ("datas fora de ordem", ("split", "val_fim"), "2014-12", "datas fora de ordem"),
    ("janela antes do CHIRPS", ("janela", "inicio"), "1980-12", "1981-01"),
    ("embargo curto", ("split", "embargo_meses"), 1, "embargo_meses menor"),
    ("embargo consome bloco", ("split", "embargo_meses"), 60, "embargo consome"),
    ("horizonte 0", ("alvo", "horizonte_h"), 0, "alvo.horizonte_h"),
    ("horizonte 13", ("alvo", "horizonte_h"), 13, "alvo.horizonte_h"),
    ("escala 0", ("alvo", "escala_spi_meses"), 0, "alvo.escala_spi_meses"),
    ("escala 25", ("alvo", "escala_spi_meses"), 25, "alvo.escala_spi_meses"),
    ("cod_ibge inteiro", ("escopo", "piloto", "polo_cod_ibge"), 2207801, "polo_cod_ibge"),
    ("cod_ibge 3 digitos", ("escopo", "piloto", "polo_cod_ibge"), "220", "polo_cod_ibge"),
    ("cod_ibge outro estado", ("escopo", "piloto", "polo_cod_ibge"), "3550308", "polo_cod_ibge"),
    ("split por mes_origem", ("split", "por"), "mes_origem", "split.por"),
    ("scaler/gama fora do treino", ("spi", "ajustar_somente_em"), "tudo", "ajustar_somente_em"),
    ("defasagem 0", ("fontes", "mapbiomas", "defasagem_anos"), 0, "defasagem_anos"),
    ("S2iD como feature", ("fontes", "s2id", "entra_como_feature"), True, "só auditoria"),
    ("caminho absoluto", ("caminhos", "raw"), "/data/raw", "caminhos.raw"),
    ("caminho windows absoluto", ("caminhos", "raw"), "C:\\dados\\raw", "caminhos.raw"),
    ("caminho com ..", ("caminhos", "raw"), "../raw", "caminhos.raw"),
    ("camadas iguais", ("caminhos", "processed"), "data/raw", "distintas"),
    ("projeto GEE no yaml", ("gee", "projeto_env"), "meu-projeto-123", "gee.projeto_env"),
    ("lote acima de 5000", ("gee", "max_linhas_por_lote"), 5001, "max_linhas_por_lote"),
    ("versao antiga", ("projeto", "versao_params"), "0.1.0", "versão antiga"),
    ("versao 0.1.1 antiga", ("projeto", "versao_params"), "0.1.1", "versão antiga"),
    ("versao 0.1.2 antiga", ("projeto", "versao_params"), "0.1.2", "mínimo 0.2.1"),
    ("versao 0.2.0 antiga", ("projeto", "versao_params"), "0.2.0", "mínimo 0.2.1"),
    ("versao malformada", ("projeto", "versao_params"), "abc", "versao_params"),
    ("incluir_transicao texto", ("escopo", "incluir_transicao"), "sim", "escopo.incluir_transicao"),
    ("incluir_transicao inteiro", ("escopo", "incluir_transicao"), 1, "escopo.incluir_transicao"),
    ("simplificacao zero", ("escopo", "simplificacao_graus"), 0.0, "escopo.simplificacao_graus"),
    ("simplificacao negativa", ("escopo", "simplificacao_graus"), -0.001, "escopo.simplificacao_graus"),
    ("simplificacao acima do maximo", ("escopo", "simplificacao_graus"), 0.011, "escopo.simplificacao_graus"),
    ("simplificacao inteira", ("escopo", "simplificacao_graus"), 1, "escopo.simplificacao_graus"),
    ("simplificacao texto", ("escopo", "simplificacao_graus"), "0.001", "escopo.simplificacao_graus"),
    ("limiar de dia de chuva zero", ("fontes", "chirps", "limiar_dia_chuva_mm"), 0.0, "limiar_dia_chuva_mm"),
    ("limiar de dia de chuva negativo", ("fontes", "chirps", "limiar_dia_chuva_mm"), -1.0, "limiar_dia_chuva_mm"),
    ("limiar de dia de chuva acima do maximo", ("fontes", "chirps", "limiar_dia_chuva_mm"), 50.1, "limiar_dia_chuva_mm"),
    ("limiar de dia de chuva texto", ("fontes", "chirps", "limiar_dia_chuva_mm"), "1.0", "limiar_dia_chuva_mm"),
    ("limiar de dia de chuva booleano", ("fontes", "chirps", "limiar_dia_chuva_mm"), True, "limiar_dia_chuva_mm"),
    ("janelas vazia", ("spi", "janelas_meses"), [], "spi.janelas_meses"),
    ("janelas repetidas", ("spi", "janelas_meses"), [3, 3], "estritamente crescentes"),
    ("janelas decrescentes", ("spi", "janelas_meses"), [6, 3], "estritamente crescentes"),
    ("janela zero", ("spi", "janelas_meses"), [0, 3], "entre 1 e 24"),
    ("janela 25", ("spi", "janelas_meses"), [3, 25], "entre 1 e 24"),
    ("janela texto", ("spi", "janelas_meses"), [3, "6"], "spi.janelas_meses"),
    ("janela decimal", ("spi", "janelas_meses"), [3, 6.0], "spi.janelas_meses"),
    ("janela booleana", ("spi", "janelas_meses"), [3, True], "spi.janelas_meses"),
    ("janelas escalar", ("spi", "janelas_meses"), 3, "spi.janelas_meses"),
    ("janelas sem a escala do alvo", ("spi", "janelas_meses"), [6], "alvo.escala_spi_meses precisa constar"),
    ("embargo curto para SPI-6", ("split", "embargo_meses"), 4, "embargo_meses menor"),
    ("janela 7 exige embargo 6", ("spi", "janelas_meses"), [3, 7], "embargo_meses menor"),
    ("sem aviso de maturidade", ("projeto", "maturidade"), "produção", "experimental"),
    ("seed booleano", ("projeto", "seed"), True, "projeto.seed"),
    ("colecao texto sem VERIFICAR", ("fontes", "mapbiomas", "colecao"), "dez", "colecao"),
    ("colecao zero", ("fontes", "mapbiomas", "colecao"), 0, "colecao"),
    ("cobertura acima de 1", ("qualidade", "min_cobertura_pixels"), 1.5, "min_cobertura_pixels"),
]


@pytest.mark.parametrize("rotulo,caminho,valor,trecho", VIOLACOES, ids=[v[0] for v in VIOLACOES])
def test_violacao_levanta_config_error(tmp_path, base, rotulo, caminho, valor, trecho):
    define(base, caminho, valor)
    with pytest.raises(ConfigError, match=re.escape(trecho)):
        load_config(escreve(tmp_path, base))


def test_chave_ausente(tmp_path, base):
    del base["spi"]["q_baixa_confianca"]
    with pytest.raises(ConfigError, match=r"spi\.q_baixa_confianca.*ausente"):
        load_config(escreve(tmp_path, base))


@pytest.mark.parametrize("chave", ["incluir_transicao", "simplificacao_graus"])
def test_chave_nova_do_escopo_ausente(tmp_path, base, chave):
    del base["escopo"][chave]
    with pytest.raises(ConfigError, match=rf"escopo\.{chave}.*ausente"):
        load_config(escreve(tmp_path, base))


def test_limiar_dia_de_chuva_ausente(tmp_path, base):
    del base["fontes"]["chirps"]["limiar_dia_chuva_mm"]
    with pytest.raises(ConfigError, match=r"fontes\.chirps\.limiar_dia_chuva_mm.*ausente"):
        load_config(escreve(tmp_path, base))


def test_treino_curto_abaixo_do_minimo(tmp_path, base):
    base["split"].update(treino_fim="2005-12", val_fim="2010-12")
    with pytest.raises(ConfigError, match="abaixo de spi.min_anos_treino"):
        load_config(escreve(tmp_path, base))


# valores de borda aceitos -------------------------------------------------------
ACEITOS = [(("alvo", "horizonte_h"), 1), (("alvo", "horizonte_h"), 12),
           (("janela", "inicio"), "1981-01"), (("fontes", "mapbiomas", "colecao"), 10),
           (("fontes", "mapbiomas", "colecao"), "VERIFICAR"),
           (("projeto", "versao_params"), "0.2.1"), (("projeto", "versao_params"), "0.2.2"),
           (("spi", "janelas_meses"), [3]), (("spi", "janelas_meses"), [3, 6]),
           (("spi", "janelas_meses"), [1, 3, 6]), (("split", "embargo_meses"), 5),
           (("fontes", "chirps", "limiar_dia_chuva_mm"), 0.1),
           (("fontes", "chirps", "limiar_dia_chuva_mm"), 50.0),
           (("fontes", "chirps", "limiar_dia_chuva_mm"), 1),  # inteiro é aceito (pydantic converte)
           (("gee", "max_linhas_por_lote"), 5000),
           (("escopo", "incluir_transicao"), False), (("escopo", "simplificacao_graus"), 0.01),
           (("escopo", "simplificacao_graus"), 0.0001)]


@pytest.mark.parametrize("caminho,valor", ACEITOS)
def test_valores_de_borda_validos(tmp_path, base, caminho, valor):
    define(base, caminho, valor)
    load_config(escreve(tmp_path, base))


# pendências e gates ------------------------------------------------------------
def test_pendencias_detecta_verificar_em_valor_e_em_texto(tmp_path, base):
    resolve_pendencias(base)
    define(base, ("spi", "convencao_zero"), "VERIFICAR")
    define(base, ("escopo", "fonte_semiarido"), "Res. 150/2021; VERIFICAR lista vigente")
    cfg = load_config(escreve(tmp_path, base))
    assert set(cfg.pendencias()) == {"spi.convencao_zero", "escopo.fonte_semiarido"}


def test_pendencias_vazio_quando_tudo_resolvido(tmp_path, base):
    resolve_pendencias(base)
    assert load_config(escreve(tmp_path, base)).pendencias() == []


def test_exigir_resolvido_por_modulo(tmp_path, base):
    resolve_pendencias(base)
    define(base, ("fontes", "inmet", "estacoes"), "VERIFICAR")
    cfg = load_config(escreve(tmp_path, base))
    exigir_resolvido(cfg, "fontes.chirps", "fontes.mapbiomas", "split")  # módulo CHIRPS passa
    for chave in ("fontes.inmet.estacoes", "fontes.inmet"):  # campo e submodelo
        with pytest.raises(PendenciaError, match="fontes.inmet"):
            exigir_resolvido(cfg, chave)


def test_exigir_resolvido_chave_inexistente_nao_e_pendencia(tmp_path, base):
    cfg = load_config(escreve(tmp_path, base))
    with pytest.raises(ConfigError, match="chave inexistente") as exc:
        exigir_resolvido(cfg, "fontes.xyz")
    assert not isinstance(exc.value, PendenciaError)


# arquivo --------------------------------------------------------------------------
def test_arquivo_inexistente(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nao_existe.yaml")


@pytest.mark.parametrize("conteudo,trecho", [
    (b"", "vazio"), (b"a: [1, 2", "YAML inválido"),
    ("nome: Piauí".encode("latin-1"), "UTF-8")], ids=["vazio", "sintaxe", "latin1"])
def test_arquivo_ruim(tmp_path, conteudo, trecho):
    p = tmp_path / "params.yaml"
    p.write_bytes(conteudo)
    with pytest.raises(ConfigError, match=trecho):
        load_config(p)


def test_utf8_com_bom_carrega(tmp_path):
    p = tmp_path / "params.yaml"
    p.write_bytes(b"\xef\xbb\xbf" + PARAMS_REAL.read_bytes())
    assert load_config(p).projeto.versao_params


# raiz, caminhos e imutabilidade ---------------------------------------------------
def test_raiz_e_caminho_vem_do_local_do_params(tmp_path, base):
    cfg = load_config(escreve(tmp_path, base))
    assert cfg.raiz == tmp_path.resolve()
    assert cfg.caminho("raw") == tmp_path.resolve() / cfg.caminhos.raw
    with pytest.raises(ConfigError, match="desconhecido"):
        cfg.caminho("xyz")


def test_config_sem_load_nao_tem_raiz(base):
    with pytest.raises(ConfigError, match="raiz desconhecida"):
        Config.model_validate(base).raiz


def test_config_e_imutavel(tmp_path, base):
    cfg = load_config(escreve(tmp_path, base))
    with pytest.raises(ValidationError):
        cfg.split.embargo_meses = 0
