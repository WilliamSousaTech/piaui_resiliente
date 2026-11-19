"""CLI dos rascunhos de IA: cliente falso, projeto sintético em tmp_path, nenhuma chamada de rede."""
import json

import numpy as np
import pandas as pd
import pytest

from dados_sinteticos import A, B, C, raw_sintetico
from src.ia import arquivo as aq
from src.ia import checklist as ck
from src.ia import cli
from src.ia import cliente as cl
from src.ia import rascunhos as rs
from test_build_site import montar_proj
from test_ia_site import CHECKLIST

BOM = json.dumps({"resumo": "Texto simples sobre o município.", "itens": [
    {"id": "mobilizar_defesa_civil", "porque": "Prepara a equipe."},
    {"id": "revisar_abastecimento", "porque": "Confere a água."}]})
RUIM = json.dumps({"resumo": "Choveu 999 mm.", "itens": [{"id": "mobilizar_defesa_civil", "porque": "x"}] * 2})


class Falso:
    modelo = "modelo-falso"

    def __init__(self, *respostas):
        self.respostas, self.chamadas = list(respostas), 0

    def gerar(self, sistema, usuario, *, max_tokens=1200):
        self.chamadas += 1
        return self.respostas.pop(0)


@pytest.fixture
def proj(tmp_path):
    cfg, raiz = montar_proj(tmp_path, raw_sintetico())
    (raiz / ck.ARQUIVO_PADRAO).parent.mkdir(parents=True, exist_ok=True)
    (raiz / ck.ARQUIVO_PADRAO).write_text(json.dumps(CHECKLIST), encoding="utf-8")
    return cfg, raiz


def rodar(raiz, *args, cliente=None):
    saida = []
    fabrica = (lambda modelo: cliente) if cliente is not None else (lambda modelo: pytest.fail("não devia criar cliente"))
    codigo = cli.main(["--params", str(raiz / "params.yaml"), *args], fabrica_cliente=fabrica, saida=saida.append)
    return codigo, "\n".join(saida)


def arquivo(raiz):
    return aq.carregar(raiz / aq.ARQUIVO_PADRAO)


# ------------------------------------------------------------------ códigos
def test_municipios_remove_repetidos_e_valida():
    assert cli._municipios("2200001, 2200002,2200001", 5) == ["2200001", "2200002"]
    with pytest.raises(cli.CliError, match="7 dígitos: 123, abc"):
        cli._municipios("2200001,123,abc", 5)
    with pytest.raises(cli.CliError, match="ao menos um"):
        cli._municipios(" , ", 5)
    with pytest.raises(cli.CliError, match="limite de 2"):
        cli._municipios("2200001,2200002,2200003", 2)


# ------------------------------------------------------------------ gerar
def test_gerar_grava_rascunho_pendente_e_orienta_o_proximo_passo(proj):
    _, raiz = proj
    c = Falso(BOM, BOM)
    codigo, saida = rodar(raiz, "gerar", "--municipios", f"{A},{B}", cliente=c)
    assert codigo == 0 and c.chamadas == 2
    arq = arquivo(raiz)
    assert {k: r.status for k, r in arq.rascunhos.items()} == {A: "pendente_revisao", B: "pendente_revisao"}
    assert arq.checklist_sha256 == ck.sha256(raiz / ck.ARQUIVO_PADRAO)
    assert saida.splitlines()[0].startswith("AVISO: MVP experimental") and "Próximo passo" in saida
    assert saida.splitlines()[-1] == rs.AVISO_MVP
    assert f"{A} Alfa do Norte: pendente de revisão (1 tentativa(s))" in saida


def test_gerar_nunca_aprova_e_o_site_continua_sem_a_pagina(proj):
    cfg, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    from src.web import build_site as bs
    r = bs.gerar_site(cfg)
    assert r["planejamento"] == 0 and not (raiz / "site" / "planejamento.html").exists()


def test_gerar_sem_checklist_falha_antes_de_criar_o_cliente(proj):
    _, raiz = proj
    (raiz / ck.ARQUIVO_PADRAO).unlink()
    codigo, _ = rodar(raiz, "gerar", "--municipios", A)           # a fábrica falha o teste se for chamada
    assert codigo == 1 and arquivo(raiz) is None


def test_gerar_sem_chave_diz_como_definir_e_nao_grava(proj, capsys):
    _, raiz = proj

    def sem_chave(modelo):
        raise cl.ChaveAusente("defina ANTHROPIC_API_KEY")
    codigo = cli.main(["--params", str(raiz / "params.yaml"), "gerar", "--municipios", A], fabrica_cliente=sem_chave,
                      saida=lambda *_: None)
    assert codigo == 1 and "ANTHROPIC_API_KEY" in capsys.readouterr().err and arquivo(raiz) is None


def test_gerar_recusado_nao_grava_e_mostra_os_motivos(proj):
    _, raiz = proj
    c = Falso(RUIM, RUIM, RUIM)
    codigo, saida = rodar(raiz, "gerar", "--municipios", A, cliente=c)
    assert codigo == 0 and c.chamadas == 3 and arquivo(raiz) is None
    assert "recusado em todas as tentativas" in saida and "número fora da ficha (999)" in saida
    assert "Resumo: recusados: 1" in saida
    assert "Próximo passo" not in saida


def test_gerar_segunda_tentativa_vale_e_fica_registrada(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(RUIM, BOM))
    assert arquivo(raiz).rascunhos[A].tentativas == 2


def test_gerar_grava_a_cada_sucesso_mesmo_se_o_modelo_falhar_no_meio(proj):
    _, raiz = proj

    class QuebraNoSegundo(Falso):
        def gerar(self, *a, **k):
            if self.chamadas == 1:
                raise cl.ErroLLM("HTTP 529 do provedor")
            return super().gerar(*a, **k)
    codigo, _ = rodar(raiz, "gerar", "--municipios", f"{A},{B}", cliente=QuebraNoSegundo(BOM))
    assert codigo == 1 and list(arquivo(raiz).rascunhos) == [A]       # o primeiro, já pago, não se perde


def test_gerar_respeita_o_limite_antes_de_chamar_o_modelo(proj):
    _, raiz = proj
    codigo, _ = rodar(raiz, "gerar", "--municipios", f"{A},{B},{C}", "--max", "2")
    assert codigo == 1 and arquivo(raiz) is None


def test_gerar_pula_revisado_sem_chamar_o_modelo_e_sobrescreve_se_pedido(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    assert rodar(raiz, "aprovar", A, "--por", "Ana Souza")[0] == 0
    c = Falso()                                                         # fila vazia: chamar o modelo seria erro
    codigo, saida = rodar(raiz, "gerar", "--municipios", A, cliente=c)
    assert codigo == 0 and c.chamadas == 0 and "já revisado" in saida and arquivo(raiz).rascunhos[A].status == "revisado"
    c2 = Falso(BOM)
    rodar(raiz, "gerar", "--municipios", A, "--sobrescrever", cliente=c2)
    assert c2.chamadas == 1 and arquivo(raiz).rascunhos[A].status == "pendente_revisao"


def test_gerar_municipio_desconhecido_e_erro(proj):
    _, raiz = proj
    codigo, _ = rodar(raiz, "gerar", "--municipios", "2299999", cliente=Falso())
    assert codigo == 1


def test_gerar_municipio_sem_dado_nao_chama_o_modelo(tmp_path):
    raw = raw_sintetico()
    m = ((raw["cod_ibge"] == B) & (raw["mes"] <= pd.Timestamp("2012-12-01"))).to_numpy()
    raw.loc[m, ["precip_mm", "precip_max_dia_mm", "dias_chuva"]] = np.nan
    raw.loc[m, "dias_validos"] = 0
    raw.loc[m, "pixels_validos_frac"] = np.float32(0.0)
    _, raiz = montar_proj(tmp_path, raw)
    (raiz / ck.ARQUIVO_PADRAO).parent.mkdir(parents=True, exist_ok=True)
    (raiz / ck.ARQUIVO_PADRAO).write_text(json.dumps(CHECKLIST), encoding="utf-8")
    c = Falso()
    codigo, saida = rodar(raiz, "gerar", "--municipios", B, cliente=c)
    assert codigo == 0 and c.chamadas == 0 and "sem SPI no mês de referência" in saida and arquivo(raiz) is None


def test_gerar_de_outro_dataset_ou_checklist_exige_recomecar(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    p = raiz / ck.ARQUIVO_PADRAO
    p.write_text(json.dumps({**CHECKLIST, "versao": "v8"}), encoding="utf-8")
    codigo, _ = rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    assert codigo == 1
    codigo, _ = rodar(raiz, "gerar", "--municipios", B, "--recomecar", cliente=Falso(BOM))
    assert codigo == 0 and list(arquivo(raiz).rascunhos) == [B]


# ------------------------------------------------------------------ revisão
def test_aprovar_grava_quem_e_quando_e_manda_regerar_o_site(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    codigo, saida = rodar(raiz, "aprovar", A, "--por", "Ana Souza")
    r = arquivo(raiz).rascunhos[A]
    assert codigo == 0 and (r.status, r.revisado_por) == ("revisado", "Ana Souza") and r.revisado_em_utc.endswith("Z")
    assert "python -m src.web.build_site" in saida


def test_aprovar_recusa_texto_editado_com_numero_de_fora(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    p = raiz / aq.ARQUIVO_PADRAO
    d = json.loads(p.read_text(encoding="utf-8"))
    d["rascunhos"][A]["resumo"] = "Choveu 777 mm."
    p.write_text(json.dumps(d), encoding="utf-8")
    codigo, _ = rodar(raiz, "aprovar", A, "--por", "Ana Souza")
    assert codigo == 1 and arquivo(raiz).rascunhos[A].status == "pendente_revisao"


def test_aprovar_com_checklist_alterado_ou_dataset_diferente_e_recusado(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    p = raiz / ck.ARQUIVO_PADRAO
    original = p.read_text(encoding="utf-8")
    p.write_text(json.dumps({**CHECKLIST, "versao": "v8"}), encoding="utf-8")
    assert rodar(raiz, "aprovar", A, "--por", "Ana Souza")[0] == 1
    p.write_text(original, encoding="utf-8")
    arq = arquivo(raiz)
    aq.gravar(aq.ArquivoRascunhos(dataset_sha256="ef" * 32, checklist_sha256=arq.checklist_sha256, rascunhos=arq.rascunhos),
              raiz / aq.ARQUIVO_PADRAO)
    assert rodar(raiz, "aprovar", A, "--por", "Ana Souza")[0] == 1
    assert arquivo(raiz).rascunhos[A].status == "pendente_revisao"


def test_aprovar_sem_rascunhos_e_codigo_inexistente(proj):
    _, raiz = proj
    assert rodar(raiz, "aprovar", A, "--por", "Ana Souza")[0] == 1                 # nem arquivo existe
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    assert rodar(raiz, "aprovar", B, "--por", "Ana Souza")[0] == 1


def test_rejeitar(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    codigo, saida = rodar(raiz, "rejeitar", A, "--por", "Ana Souza")
    assert codigo == 0 and arquivo(raiz).rascunhos[A].status == "rejeitado" and "rejeitado por Ana Souza" in saida


def test_por_e_obrigatorio(proj, capsys):
    _, raiz = proj
    with pytest.raises(SystemExit):
        cli.main(["--params", str(raiz / "params.yaml"), "aprovar", A], saida=lambda *_: None)
    assert "--por" in capsys.readouterr().err


# ------------------------------------------------------------------ ver e listar
def test_ver_mostra_ficha_resumo_itens_e_rotulo(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Falso(BOM))
    codigo, saida = rodar(raiz, "ver", A)
    assert codigo == 0
    for trecho in ("Alfa do Norte", "FICHA (os únicos números permitidos)", "Município: Alfa do Norte", "RESUMO",
                   "Texto simples sobre o município.", "Mobilizar a Defesa Civil [mobilizar_defesa_civil]",
                   "por quê: Prepara a equipe.", "situação: pendente_revisao", rs.ROTULO_NAO_REVISADO):
        assert trecho in saida, trecho
    assert rs.ROTULO not in saida                                       # pendente não pode dizer que foi revisado
    assert rodar(raiz, "ver", B)[0] == 1
    rodar(raiz, "aprovar", A, "--por", "Ana Souza")
    _, depois = rodar(raiz, "ver", A)
    assert rs.ROTULO in depois and rs.ROTULO_NAO_REVISADO not in depois


def test_listar_mostra_status_revisor_e_se_vale_para_o_dataset_atual(proj):
    _, raiz = proj
    assert rodar(raiz, "listar")[0] == 1                                             # ainda sem arquivo
    rodar(raiz, "gerar", "--municipios", f"{A},{B}", cliente=Falso(BOM, BOM))
    rodar(raiz, "aprovar", A, "--por", "Ana Souza")
    codigo, saida = rodar(raiz, "listar")
    assert codigo == 0 and "o atual" in saida and "2 rascunho(s)" in saida
    assert f"{A} Alfa do Norte | " in saida and "| revisado | Ana Souza" in saida and "| pendente_revisao" in saida


def test_comando_obrigatorio(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2 and "usage: python -m src.ia.cli" in capsys.readouterr().err


# ------------------------------------------------------------------ --provedor (Gemini, Groq e reserva)
class ModeloFalso:
    """Cliente falso com nome próprio; levanta `erro` se houver."""

    def __init__(self, modelo, *respostas, erro=None):
        self.modelo, self.respostas, self.erro = modelo, list(respostas), erro

    def gerar(self, sistema, usuario, *, max_tokens=1200):
        if self.erro is not None:
            raise self.erro
        return self.respostas.pop(0)


def rodar_prov(raiz, *args, fabrica_provedores, fabrica_cliente=None):
    saida = []
    fc = fabrica_cliente or (lambda modelo: pytest.fail("o provedor padrão não devia ser usado aqui"))
    codigo = cli.main(["--params", str(raiz / "params.yaml"), *args], fabrica_cliente=fc,
                      fabrica_provedores=fabrica_provedores, saida=saida.append)
    return codigo, "\n".join(saida)


def test_provedor_gemini_groq_monta_a_lista_com_o_texto_e_o_modelo_pedidos(proj):
    _, raiz = proj
    visto = {}

    def fabrica(texto, modelo, *, aviso):
        visto.update(texto=texto, modelo=modelo, aviso=aviso)
        return ModeloFalso("m-gemini", BOM)
    codigo, saida = rodar_prov(raiz, "gerar", "--provedor", "gemini,groq", "--municipios", A, fabrica_provedores=fabrica)
    assert codigo == 0 and visto["texto"] == "gemini,groq" and visto["modelo"] is None
    assert "modelo: m-gemini" in saida and arquivo(raiz).rascunhos[A].modelo == "m-gemini"


def test_provedor_um_so_repassa_o_modelo(proj):
    _, raiz = proj
    visto = {}

    def fabrica(texto, modelo, *, aviso):
        visto.update(texto=texto, modelo=modelo)
        return ModeloFalso("gemini-teste", BOM)
    codigo, _ = rodar_prov(raiz, "gerar", "--provedor", "gemini", "--modelo", "gemini-teste", "--municipios", A,
                           fabrica_provedores=fabrica)
    assert codigo == 0 and visto == {"texto": "gemini", "modelo": "gemini-teste"}


@pytest.mark.parametrize("args", [(), ("--provedor", "anthropic"), ("--provedor", " Anthropic ")])
def test_sem_provedor_ou_anthropic_usa_a_fabrica_de_sempre(proj, args):
    _, raiz = proj
    codigo, _ = rodar_prov(raiz, "gerar", *args, "--municipios", A, fabrica_cliente=lambda modelo: ModeloFalso("claude-f", BOM),
                           fabrica_provedores=lambda *a, **k: pytest.fail("não devia usar a lista de provedores"))
    assert codigo == 0 and arquivo(raiz).rascunhos[A].modelo == "claude-f"


def test_reserva_ponta_a_ponta_o_rascunho_grava_o_modelo_de_quem_respondeu_e_a_troca_e_avisada(proj):
    from src.ia.cliente_reserva import ClienteComReserva
    _, raiz = proj

    def fabrica(texto, modelo, *, aviso):
        return ClienteComReserva([ModeloFalso("gemini-x", erro=cl.ErroLLM("HTTP 429 do provedor: cota")),
                                  ModeloFalso("groq-y", BOM, BOM)], aviso=aviso)
    codigo, saida = rodar_prov(raiz, "gerar", "--provedor", "gemini,groq", "--municipios", f"{A},{B}", fabrica_provedores=fabrica)
    assert codigo == 0
    assert {r.modelo for r in arquivo(raiz).rascunhos.values()} == {"groq-y"}
    assert saida.count("AVISO: gemini-x: HTTP 429 do provedor: cota") == 2
    assert "respostas por modelo: groq-y = 2" in saida


def test_cliente_unico_nao_imprime_respostas_por_modelo(proj):
    _, raiz = proj
    _, saida = rodar_prov(raiz, "gerar", "--provedor", "gemini", "--municipios", A,
                          fabrica_provedores=lambda t, m, *, aviso: ModeloFalso("gemini-x", BOM))
    assert "respostas por modelo" not in saida


def test_provedor_desconhecido_e_erro_antes_de_gravar(proj, capsys):
    _, raiz = proj
    codigo = cli.main(["--params", str(raiz / "params.yaml"), "gerar", "--provedor", "grok", "--municipios", A],
                      saida=lambda *_: None)
    assert codigo == 1 and "provedor desconhecido: grok" in capsys.readouterr().err and arquivo(raiz) is None


def test_lista_sem_nenhuma_chave_diz_quais_variaveis_definir_e_nao_grava(proj, capsys, monkeypatch):
    _, raiz = proj
    for v in ("GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(v, raising=False)
    saida = []
    codigo = cli.main(["--params", str(raiz / "params.yaml"), "gerar", "--provedor", "gemini,groq", "--municipios", A],
                      saida=saida.append)
    err = capsys.readouterr().err
    assert codigo == 1 and "GEMINI_API_KEY ou GROQ_API_KEY" in err and arquivo(raiz) is None
    assert "AVISO: gemini: ignorado" in "\n".join(saida) and "AVISO: groq: ignorado" in "\n".join(saida)


def test_modelo_com_varios_provedores_e_erro(proj, capsys):
    _, raiz = proj
    codigo = cli.main(["--params", str(raiz / "params.yaml"), "gerar", "--provedor", "gemini,groq", "--modelo", "x",
                       "--municipios", A], saida=lambda *_: None)
    assert codigo == 1 and "--modelo vale para um provedor só" in capsys.readouterr().err and arquivo(raiz) is None


def test_ajuda_do_gerar_cita_provedor_e_os_tres_modelos(capsys):
    with pytest.raises(SystemExit):
        cli.main(["gerar", "--help"])
    ajuda = " ".join(capsys.readouterr().out.split())
    assert "--provedor" in ajuda and "gemini,groq" in ajuda and "PR_GEMINI_MODELO" in ajuda and "PR_GROQ_MODELO" in ajuda


# ------------------------------------------------------------------ --todos, --so-faltantes, --ordem, andamento e tokens
class Registra:
    """Responde BOM, anota a ordem dos municípios (pelo código na ficha) e simula falha ou texto ruim em alguns."""
    modelo = "modelo-registra"

    def __init__(self, uso=None, erro_em=(), ruim_em=()):
        self.visto, self._uso, self.ultimo_uso = [], uso, None
        self.erro_em, self.ruim_em = set(erro_em), set(ruim_em)

    def gerar(self, sistema, usuario, *, max_tokens=1200):
        cod = next(c for c in (A, B, C) if f"({c})" in usuario)
        self.visto.append(cod)
        if cod in self.erro_em:
            raise cl.ErroLLM("HTTP 529 do provedor")
        self.ultimo_uso = self._uso
        return RUIM if cod in self.ruim_em else BOM


def linhas_de_andamento(saida):
    return [ln for ln in saida.splitlines() if ln.startswith("[")]


def test_todos_gera_para_todos_os_municipios_com_dado_e_mostra_o_andamento(proj):
    _, raiz = proj
    c = Registra()
    codigo, saida = rodar(raiz, "gerar", "--todos", cliente=c)
    assert codigo == 0 and c.visto == [A, B, C] and set(arquivo(raiz).rascunhos) == {A, B, C}
    andamento = linhas_de_andamento(saida)
    assert [ln.split()[0] for ln in andamento] == ["[1/3]", "[2/3]", "[3/3]"]
    assert andamento[0] == f"[1/3] {A} Alfa do Norte: pendente de revisão (1 tentativa(s))"
    assert "Resumo: novos: 3" in saida and "Próximo passo" in saida


@pytest.mark.parametrize("args", [("--todos", "--municipios", A), ()])
def test_todos_e_municipios_sao_excludentes_e_um_dos_dois_e_obrigatorio(proj, capsys, args):
    _, raiz = proj
    with pytest.raises(SystemExit) as e:
        cli.main(["--params", str(raiz / "params.yaml"), "gerar", *args], saida=lambda *_: None)
    assert e.value.code == 2 and "usage: python -m src.ia.cli" in capsys.readouterr().err


def test_so_faltantes_pula_quem_ja_tem_rascunho_de_qualquer_status_sem_chamar_o_modelo(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", f"{A},{B}", cliente=Registra())
    assert rodar(raiz, "rejeitar", B, "--por", "Ana Souza")[0] == 0
    c = Registra()
    codigo, saida = rodar(raiz, "gerar", "--todos", "--so-faltantes", cliente=c)
    assert codigo == 0 and c.visto == [C]
    assert "já tinham rascunho: 2" in saida and "novos: 1" in saida
    assert arquivo(raiz).rascunhos[B].status == "rejeitado"             # a decisão humana não é refeita
    assert [ln.split()[1] for ln in linhas_de_andamento(saida)] == [C]  # com --todos, quem foi pulado não faz barulho


def test_so_faltantes_com_lista_explicita_diz_por_que_pulou(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Registra())
    c = Registra()
    _, saida = rodar(raiz, "gerar", "--municipios", f"{A},{B}", "--so-faltantes", cliente=c)
    assert c.visto == [B] and f"{A} Alfa do Norte: já tem rascunho (pendente_revisao); nada a fazer" in saida


def test_sem_so_faltantes_o_pendente_e_refeito_como_sempre(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Registra())
    c = Registra()
    rodar(raiz, "gerar", "--municipios", A, cliente=c)
    assert c.visto == [A]


def test_so_faltantes_e_sobrescrever_se_contradizem(proj, capsys):
    _, raiz = proj
    codigo, _ = rodar(raiz, "gerar", "--todos", "--so-faltantes", "--sobrescrever", cliente=Registra())
    assert codigo == 1 and "se contradizem" in capsys.readouterr().err and arquivo(raiz) is None


def test_todos_pula_revisado_como_a_lista_explicita(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Registra())
    rodar(raiz, "aprovar", A, "--por", "Ana Souza")
    c = Registra()
    _, saida = rodar(raiz, "gerar", "--todos", cliente=c)
    assert c.visto == [B, C] and "já revisados: 1" in saida and arquivo(raiz).rascunhos[A].status == "revisado"


def test_max_com_todos_limita_as_chamadas_e_diz_quantos_faltam(proj):
    _, raiz = proj
    c = Registra()
    codigo, saida = rodar(raiz, "gerar", "--todos", "--max", "2", cliente=c)
    assert codigo == 0 and c.visto == [A, B] and set(arquivo(raiz).rascunhos) == {A, B}
    assert "Faltam 1 município(s)" in saida and "--so-faltantes" in saida
    c2 = Registra()
    _, saida2 = rodar(raiz, "gerar", "--todos", "--so-faltantes", "--max", "2", cliente=c2)
    assert c2.visto == [C] and "Faltam" not in saida2


def test_max_nao_conta_quem_foi_pulado(proj):
    _, raiz = proj
    rodar(raiz, "gerar", "--municipios", A, cliente=Registra())
    c = Registra()
    _, saida = rodar(raiz, "gerar", "--todos", "--so-faltantes", "--max", "1", cliente=c)
    assert c.visto == [B] and "Faltam 1 município(s)" in saida


def test_max_zero_com_todos_e_erro(proj, capsys):
    _, raiz = proj
    codigo, _ = rodar(raiz, "gerar", "--todos", "--max", "0", cliente=Registra())
    assert codigo == 1 and "pelo menos 1" in capsys.readouterr().err


def test_ordem_seca_comeca_pelo_spi_mais_baixo(proj):
    _, raiz = proj
    c = Registra()
    rodar(raiz, "gerar", "--todos", "--ordem", "seca", cliente=c)       # SPI sintético: C = 0,49 < A = 1,72 < B = 1,98
    assert c.visto == [C, A, B]
    c2 = Registra()
    rodar(raiz, "gerar", "--todos", "--ordem", "seca", "--max", "1", "--recomecar", cliente=c2)
    assert c2.visto == [C]                                              # com limite, leva o mais seco


def test_ordem_invalida_e_erro_de_uso(proj, capsys):
    _, raiz = proj
    with pytest.raises(SystemExit) as e:
        cli.main(["--params", str(raiz / "params.yaml"), "gerar", "--todos", "--ordem", "alfabetica"], saida=lambda *_: None)
    assert e.value.code == 2


@pytest.mark.parametrize("uso, esperado", [
    ({"total_tokens": 5}, 5), ({"totalTokenCount": 7}, 7), ({"prompt_tokens": 3, "completion_tokens": 4}, 7),
    ({"input_tokens": 3, "output_tokens": 4}, 7), ({"total_tokens": 5, "prompt_tokens": 1, "completion_tokens": 1}, 5),
    ({}, None), (None, None), ("x", None), ({"total_tokens": True}, None), ({"total_tokens": -1}, None),
    ({"total_tokens": "5"}, None), ({"total_tokens": 5.0}, None), ({"prompt_tokens": 3}, None),
    ({"input_tokens": 3, "output_tokens": True}, None)])
def test_tokens_do_uso(uso, esperado):
    assert cli.tokens_do_uso(uso) == esperado


def test_tokens_somam_todas_as_chamadas_inclusive_as_recusadas_e_aparecem_por_municipio_e_no_total(proj):
    _, raiz = proj
    respostas = iter([RUIM, BOM, BOM])                                  # A: recusado e depois bom (2 chamadas); B: bom

    class Fila(Registra):
        def gerar(self, sistema, usuario, *, max_tokens=1200):
            super().gerar(sistema, usuario, max_tokens=max_tokens)
            return next(respostas)
    _, saida = rodar(raiz, "gerar", "--municipios", f"{A},{B}", cliente=Fila(uso={"total_tokens": 100}))
    assert f"{A} Alfa do Norte: pendente de revisão (2 tentativa(s)) | 200 tokens" in saida
    assert f"{B} Beta do Sul: pendente de revisão (1 tentativa(s)) | 100 tokens" in saida
    assert "tokens no total: 300" in saida


def test_sem_informacao_de_uso_nao_inventa_tokens(proj):
    _, raiz = proj
    _, saida = rodar(raiz, "gerar", "--municipios", A, cliente=Registra())
    assert "tokens" not in saida


def test_falha_do_provedor_num_municipio_nao_trava_os_seguintes_e_o_codigo_de_saida_avisa(proj):
    _, raiz = proj
    c = Registra(erro_em=[A])
    codigo, saida = rodar(raiz, "gerar", "--todos", cliente=c)
    assert codigo == 1 and c.visto == [A, B, C] and set(arquivo(raiz).rascunhos) == {B, C}
    assert f"[1/3] {A} Alfa do Norte: falhou: HTTP 529 do provedor" in saida
    assert "falharam: 1" in saida and "rode de novo com --so-faltantes" in saida


def test_falhas_seguidas_interrompem_e_dizem_quantos_faltam(proj):
    cfg, _ = proj
    c = Registra(erro_em=[A, B, C])
    r = cli.executar_gerar(cfg, [], c, todos=True, max_falhas_seguidas=2)
    assert c.visto == [A, B] and r["restantes"] == 1 and "2 falhas seguidas" in r["interrompido"]
    assert "HTTP 529 do provedor" in r["interrompido"]


def test_uma_resposta_boa_zera_a_contagem_de_falhas_seguidas(proj):
    cfg, _ = proj
    c = Registra(erro_em=[A, C])
    r = cli.executar_gerar(cfg, [], c, todos=True, max_falhas_seguidas=2)
    assert c.visto == [A, B, C] and r["interrompido"] is None and r["restantes"] == 0


def test_texto_recusado_pela_validacao_tambem_zera_a_contagem_porque_o_provedor_respondeu(proj):
    cfg, _ = proj
    c = Registra(erro_em=[A, C], ruim_em=[B])
    r = cli.executar_gerar(cfg, [], c, todos=True, max_falhas_seguidas=2, tentativas=1)
    assert c.visto == [A, B, C] and r["interrompido"] is None
    assert [r["municipios"][k]["situacao"] for k in (A, B, C)] == ["falhou", "recusado", "falhou"]


def test_tres_falhas_seguidas_e_o_padrao_e_o_comando_avisa_que_parou(proj):
    _, raiz = proj
    codigo, saida = rodar(raiz, "gerar", "--todos", cliente=Registra(erro_em=[A, B, C]))
    assert codigo == 1 and "Parei: 3 falhas seguidas" in saida and arquivo(raiz) is None


def test_max_falhas_seguidas_invalido_e_erro(proj):
    cfg, _ = proj
    with pytest.raises(cli.CliError, match="pelo menos 1"):
        cli.executar_gerar(cfg, [], Registra(), todos=True, max_falhas_seguidas=0)


def proj_com_b_sem_dado(tmp_path):
    raw = raw_sintetico()
    m = ((raw["cod_ibge"] == B) & (raw["mes"] <= pd.Timestamp("2012-12-01"))).to_numpy()
    raw.loc[m, ["precip_mm", "precip_max_dia_mm", "dias_chuva"]] = np.nan
    raw.loc[m, "dias_validos"] = 0
    raw.loc[m, "pixels_validos_frac"] = np.float32(0.0)
    cfg_e_raiz = montar_proj(tmp_path, raw)
    raiz = cfg_e_raiz[1]
    (raiz / ck.ARQUIVO_PADRAO).parent.mkdir(parents=True, exist_ok=True)
    (raiz / ck.ARQUIVO_PADRAO).write_text(json.dumps(CHECKLIST), encoding="utf-8")
    return cfg_e_raiz


def test_todos_ignora_municipios_sem_dado_e_conta_no_resumo(tmp_path):
    _, raiz = proj_com_b_sem_dado(tmp_path)
    c = Registra()
    codigo, saida = rodar(raiz, "gerar", "--todos", cliente=c)
    assert codigo == 0 and c.visto == [A, C] and "sem dado: 1" in saida
    assert [ln.split()[1] for ln in linhas_de_andamento(saida)] == [A, C]


def test_ordem_seca_deixa_o_municipio_sem_dado_por_ultimo(tmp_path):
    cfg, _ = proj_com_b_sem_dado(tmp_path)
    r = cli.executar_gerar(cfg, [], Registra(), todos=True, ordem="seca")
    assert list(r["municipios"]) == [C, A, B] and r["municipios"][B]["situacao"] == "sem_dado"


def test_ordem_desconhecida_na_funcao_e_erro(proj):
    cfg, _ = proj
    with pytest.raises(cli.CliError, match="ordem desconhecida"):
        cli.executar_gerar(cfg, [], Registra(), todos=True, ordem="alfabetica")
