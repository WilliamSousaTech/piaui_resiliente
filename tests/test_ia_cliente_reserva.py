"""Cliente com reserva: clientes falsos, nenhuma rede."""
import pytest

from src.ia import cliente as cl
from src.ia.cliente_reserva import ClienteComReserva


class Falso:
    """Cliente falso: devolve `texto` ou levanta `erro`; registra as chamadas."""

    def __init__(self, modelo, texto="ok", erro=None, uso=None):
        self._modelo, self.texto, self.erro, self.chamadas, self.ultimo_uso = modelo, texto, erro, [], uso

    @property
    def modelo(self):
        return self._modelo

    def gerar(self, sistema, usuario, *, max_tokens=1200):
        self.chamadas.append((sistema, usuario, max_tokens))
        if self.erro is not None:
            raise self.erro
        return self.texto


def erro(msg="falhou"):
    return cl.ErroLLM(msg)


# ------------------------------------------------------------------ ordem e troca
def test_o_primeiro_que_responde_vence_e_os_outros_nem_sao_chamados():
    a, b = Falso("a", "texto-a"), Falso("b", "texto-b")
    c = ClienteComReserva([a, b])
    assert c.gerar("S", "U", max_tokens=77) == "texto-a"
    assert a.chamadas == [("S", "U", 77)] and b.chamadas == []
    assert c.modelo == "a" and c.falhas == [] and c.usados == {"a": 1}


def test_quando_o_primeiro_falha_passa_ao_segundo_e_o_modelo_e_o_de_quem_respondeu():
    a, b = Falso("a", erro=erro("limite")), Falso("b", "texto-b")
    c = ClienteComReserva([a, b])
    assert c.modelo == "a"                                   # antes da primeira resposta: o primeiro da lista
    assert c.gerar("S", "U", max_tokens=5) == "texto-b"
    assert c.modelo == "b" and b.chamadas == [("S", "U", 5)]
    assert c.falhas == [("a", "limite")] and c.usados == {"b": 1}


def test_percorre_a_lista_inteira_ate_achar_um_que_responda():
    c = ClienteComReserva([Falso("a", erro=erro("1")), Falso("b", erro=erro("2")), Falso("c", "texto-c")])
    assert c.gerar("S", "U") == "texto-c" and c.modelo == "c"
    assert c.falhas == [("a", "1"), ("b", "2")]


def test_cada_chamada_comeca_de_novo_pelo_primeiro_da_lista():
    a = Falso("a", erro=erro("limite"))
    b = Falso("b", "texto-b")
    c = ClienteComReserva([a, b])
    c.gerar("S", "U")
    a.erro = None
    a.texto = "texto-a"
    assert c.gerar("S", "U") == "texto-a" and c.modelo == "a"
    assert c.falhas == [] and c.usados == {"b": 1, "a": 1}


def test_o_modelo_nao_muda_quando_todos_falham():
    c = ClienteComReserva([Falso("a"), Falso("b", "x")])
    c.gerar("S", "U")
    c._clientes[0].erro = erro()
    c._clientes[1].erro = erro()
    with pytest.raises(cl.ErroLLM):
        c.gerar("S", "U")
    assert c.modelo == "a"


# ------------------------------------------------------------------ todos falham
def test_todos_falham_sobe_erro_que_lista_o_motivo_de_cada_um():
    c = ClienteComReserva([Falso("a", erro=erro("limite do a")), Falso("b", erro=erro("rede do b"))])
    with pytest.raises(cl.ErroLLM) as e:
        c.gerar("S", "U")
    assert str(e.value) == "nenhum provedor respondeu: a: limite do a | b: rede do b"
    assert c.falhas == [("a", "limite do a"), ("b", "rede do b")] and c.usados == {}


@pytest.mark.parametrize("falha", [cl.RespostaTruncada("cortado"), cl.ChaveAusente("sem chave"), cl.ErroLLM("x")])
def test_toda_subclasse_de_errollm_passa_a_vez(falha):
    c = ClienteComReserva([Falso("a", erro=falha), Falso("b", "texto-b")])
    assert c.gerar("S", "U") == "texto-b"


@pytest.mark.parametrize("falha", [ValueError("max_tokens"), TypeError("x"), KeyError("k"), RuntimeError("bug")])
def test_erro_que_nao_e_do_provedor_sobe_na_hora_sem_chamar_o_proximo(falha):
    a, b = Falso("a", erro=falha), Falso("b", "texto-b")
    with pytest.raises(type(falha)):
        ClienteComReserva([a, b]).gerar("S", "U")
    assert b.chamadas == []


# ------------------------------------------------------------------ aviso, uso e contagem
def test_aviso_e_chamado_a_cada_falha_com_modelo_e_motivo_antes_de_a_chamada_terminar():
    avisos = []
    c = ClienteComReserva([Falso("a", erro=erro("1")), Falso("b", erro=erro("2")), Falso("c", "ok")],
                          aviso=lambda m, motivo: avisos.append((m, motivo)))
    c.gerar("S", "U")
    assert avisos == [("a", "1"), ("b", "2")]


def test_aviso_tambem_e_chamado_quando_todos_falham():
    avisos = []
    c = ClienteComReserva([Falso("a", erro=erro("1"))], aviso=lambda m, motivo: avisos.append((m, motivo)))
    with pytest.raises(cl.ErroLLM):
        c.gerar("S", "U")
    assert avisos == [("a", "1")]


def test_ultimo_uso_vem_de_quem_respondeu_e_vira_none_se_nao_for_dict():
    c = ClienteComReserva([Falso("a", erro=erro()), Falso("b", uso={"tokens": 3})])
    c.gerar("S", "U")
    assert c.ultimo_uso == {"tokens": 3}
    c2 = ClienteComReserva([Falso("a", uso=["x"])])
    c2.gerar("S", "U")
    assert c2.ultimo_uso is None


def test_usados_acumula_entre_chamadas_e_falhas_so_guarda_a_ultima():
    a, b = Falso("a", erro=erro("1")), Falso("b", "ok")
    c = ClienteComReserva([a, b])
    c.gerar("S", "U")
    c.gerar("S", "U")
    assert c.usados == {"b": 2} and c.falhas == [("a", "1")]


# ------------------------------------------------------------------ construção e protocolo
def test_lista_vazia_e_erro():
    with pytest.raises(ValueError):
        ClienteComReserva([])


@pytest.mark.parametrize("ruim", ["texto", None, object(), 5])
def test_item_que_nao_cumpre_o_protocolo_e_erro(ruim):
    with pytest.raises(TypeError):
        ClienteComReserva([Falso("a"), ruim])


def test_cumpre_o_protocolo_e_mostra_os_modelos_na_ordem():
    c = ClienteComReserva((Falso("z"), Falso("a")))                 # fora de ordem alfabética de propósito
    assert isinstance(c, cl.ClienteLLM) and c.modelos == ("z", "a")
    assert repr(c) == "ClienteComReserva(modelos=['z', 'a'])"


def test_funciona_com_o_gerador_de_rascunho_e_grava_o_modelo_de_quem_respondeu():
    from datetime import datetime, timezone

    from src.ia import fatos as ft
    from src.ia import rascunhos as rs
    from src.ia.checklist import carregar

    lista = carregar()
    f = ft.Fatos(cod_ibge="2207801", nome="Paulistana", mes="2026-07", escala=3, excluido=False, spi_cent=-127,
                 classe="Seca moderada", em_seca=True, baixa_confianca=False, chuva_mes_dec=68, chuva_12m_dec=9223,
                 meses_em_seca_12m=3, leituras_confiaveis_12m=10)
    ficha = ft.ficha_texto(f)
    ids = [i.id for i in lista.itens][:2]
    resposta = ('{"resumo": "Em julho de 2026, Paulistana está em seca moderada, com SPI-3 de −1,27.", "itens": ['
                + ", ".join('{"id": "%s", "porque": "Leitura de julho de 2026 pede atenção."}' % i for i in ids) + "]}")
    assert rs.validar("Em julho de 2026, Paulistana está em seca moderada, com SPI-3 de −1,27.",
                      [(i, "Leitura de julho de 2026 pede atenção.") for i in ids], ficha, lista) == []
    c = ClienteComReserva([Falso("gemini-x", erro=erro("limite")), Falso("groq-y", resposta)])
    r = rs.gerar_rascunho(f, lista, c, agora=datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc))
    assert r.modelo == "groq-y" and r.tentativas == 1


# ------------------------------------------------------------------ descanso do provedor que falhou
class Relogio:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_sem_descanso_o_comportamento_antigo_continua_o_provedor_que_falhou_e_tentado_de_novo():
    a, b = Falso("a", erro=erro("limite")), Falso("b", "texto-b")
    c = ClienteComReserva([a, b])
    c.gerar("S", "U")
    c.gerar("S", "U")
    assert len(a.chamadas) == 2 and c.em_descanso == ()


def test_provedor_que_falhou_e_pulado_ate_o_descanso_acabar():
    rel = Relogio()
    a, b = Falso("a", erro=erro("limite")), Falso("b", "texto-b")
    c = ClienteComReserva([a, b], descanso_s=300, relogio=rel)
    c.gerar("S", "U")                                   # a falha e entra em descanso; b responde
    assert len(a.chamadas) == 1 and c.em_descanso == ("a",)
    rel.t += 299
    c.gerar("S", "U")                                   # ainda em descanso: a nem é chamado
    assert len(a.chamadas) == 1 and len(b.chamadas) == 2 and c.em_descanso == ("a",)
    rel.t += 1                                          # 300 s exatos: o descanso acabou
    c.gerar("S", "U")
    assert len(a.chamadas) == 2 and c.em_descanso == ("a",)   # falhou de novo, volta ao descanso


def test_pular_nao_gera_aviso_nem_entra_em_falhas():
    rel, avisos = Relogio(), []
    c = ClienteComReserva([Falso("a", erro=erro("limite")), Falso("b", "ok")], descanso_s=60, relogio=rel,
                          aviso=lambda m, motivo: avisos.append(m))
    c.gerar("S", "U")
    c.gerar("S", "U")
    assert avisos == ["a"] and c.falhas == [] and c.usados == {"b": 2}


def test_se_todos_estao_em_descanso_tenta_todos_na_ordem_em_vez_de_travar():
    rel = Relogio()
    a, b = Falso("a", erro=erro("1")), Falso("b", erro=erro("2"))
    c = ClienteComReserva([a, b], descanso_s=300, relogio=rel)
    with pytest.raises(cl.ErroLLM):
        c.gerar("S", "U")
    assert c.em_descanso == ("a", "b")
    a.erro, a.texto = None, "voltou"
    assert c.gerar("S", "U") == "voltou"                # ambos em descanso: tenta de novo a partir do primeiro
    assert c.em_descanso == ("b",)                      # a se recuperou e saiu do descanso


def test_provedor_que_volta_a_responder_sai_do_descanso():
    rel = Relogio()
    a, b = Falso("a", erro=erro("x")), Falso("b", erro=erro("y"))
    c = ClienteComReserva([a, b], descanso_s=300, relogio=rel)
    with pytest.raises(cl.ErroLLM):
        c.gerar("S", "U")
    a.erro, a.texto = None, "ok-a"
    c.gerar("S", "U")
    assert "a" not in c.em_descanso
    c.gerar("S", "U")                                   # a não está mais em descanso: é o primeiro de novo
    assert len(a.chamadas) == 3


def test_erro_final_menciona_quem_foi_pulado_e_por_que():
    rel = Relogio()
    a, b = Falso("a", erro=erro("limite do a")), Falso("b", "ok")
    c = ClienteComReserva([a, b], descanso_s=300, relogio=rel)
    c.gerar("S", "U")                                   # a entra em descanso
    b.erro = erro("rede do b")
    # a (em descanso) é pulado e b falha: nem todos estão em descanso, então só b é tentado
    with pytest.raises(cl.ErroLLM) as e:
        c.gerar("S", "U")
    assert str(e.value) == "nenhum provedor respondeu: b: rede do b | a: em descanso (falhou antes: limite do a)"
    assert c.falhas == [("b", "rede do b")]                 # `falhas` só lista quem foi de fato chamado


def test_passado_o_descanso_volta_a_ordem_original_da_lista():
    rel = Relogio()
    a, b, d = Falso("a", erro=erro("1")), Falso("b", erro=erro("2")), Falso("d", "texto-d")
    c = ClienteComReserva([a, b, d], descanso_s=100, relogio=rel)
    c.gerar("S", "U")
    rel.t += 101                                        # só passou o descanso dos dois; ambos falham de novo
    a.erro = None
    a.texto = "texto-a"
    assert c.gerar("S", "U") == "texto-a"
    assert len(b.chamadas) == 1                         # a respondeu antes de b ser tentado


@pytest.mark.parametrize("ruim", [-1, -0.001])
def test_descanso_negativo_e_erro(ruim):
    with pytest.raises(ValueError):
        ClienteComReserva([Falso("a")], descanso_s=ruim)


def test_descanso_conta_a_partir_do_momento_da_falha_e_nao_do_comeco_da_chamada():
    rel = Relogio()

    class Lento(Falso):
        def gerar(self, sistema, usuario, *, max_tokens=1200):
            rel.t += 100                                    # a espera do timeout acontece antes do erro
            return super().gerar(sistema, usuario, max_tokens=max_tokens)
    a, b = Lento("a", erro=erro("timeout")), Falso("b", "texto-b")
    c = ClienteComReserva([a, b], descanso_s=300, relogio=rel)
    c.gerar("S", "U")                                       # começou em 1000, falhou em 1100: descansa até 1400
    rel.t = 1399
    assert c.em_descanso == ("a",)
    c.gerar("S", "U")
    assert len(a.chamadas) == 1                             # contando do começo da chamada (1300) já teria voltado


def test_em_descanso_esvazia_quando_o_tempo_passa_mesmo_sem_nova_chamada():
    rel = Relogio()
    c = ClienteComReserva([Falso("a", erro=erro("x")), Falso("b", "ok")], descanso_s=300, relogio=rel)
    c.gerar("S", "U")
    rel.t += 299
    assert c.em_descanso == ("a",)
    rel.t += 1                                              # 300 s exatos
    assert c.em_descanso == ()
    rel.t += 500
    assert c.em_descanso == ()
