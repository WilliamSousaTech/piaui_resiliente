"""Destaques do site: mês de abertura e maiores secas (dados sintéticos só aqui, em tests/)."""
import numpy as np
import pytest

from src.web import destaques as dq

ND = 255


def grade(linhas):
    return np.array(linhas, dtype="uint8")


def test_abertura_e_o_ultimo_mes_com_maioria_confiavel():
    # 4 municípios, 5 meses; o último mês tem só 1 de 4 confiável
    cls = grade([[3, 3, 3, 3, 3]] * 4)
    baixa = grade([[0, 0, 0, 0, 1], [0, 0, 0, 1, 1], [0, 0, 0, 0, 1], [0, 0, 0, 1, 0]])
    assert dq.mes_abertura(cls, baixa) == 3          # mês 3: 2 de 4 confiáveis (metade conta); mês 4: 1 de 4


def test_abertura_metade_exata_conta_e_menos_que_metade_nao():
    cls = grade([[3, 3]] * 4)
    baixa = grade([[0, 1], [0, 1], [0, 0], [0, 0]])      # mês 1: 2 de 4
    assert dq.mes_abertura(cls, baixa) == 1
    baixa = grade([[0, 1], [0, 1], [0, 1], [0, 0]])      # mês 1: 1 de 4
    assert dq.mes_abertura(cls, baixa) == 0


def test_abertura_ignora_meses_sem_spi_e_conta_so_quem_tem_spi():
    cls = grade([[ND, ND, 3], [ND, ND, 3], [ND, ND, ND]])
    baixa = grade([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    assert dq.mes_abertura(cls, baixa) == 2           # 1 de 2 com SPI é confiável; o município sem dado não entra


def test_abertura_sem_mes_bom_cai_no_ultimo_com_spi():
    cls = grade([[3, 3, ND]] * 3)
    baixa = grade([[1, 1, 0]] * 3)
    assert dq.mes_abertura(cls, baixa) == 1


def test_abertura_sem_nenhum_spi_e_erro():
    with pytest.raises(ValueError, match="nenhum mês tem SPI"):
        dq.mes_abertura(grade([[ND, ND]]), grade([[0, 0]]))


@pytest.mark.parametrize("a,b", [((2, 3), (3, 2)), ((2, 3), (6,))])
def test_formas_diferentes_sao_erro(a, b):
    with pytest.raises(ValueError, match="mesma forma"):
        dq.mes_abertura(np.zeros(a, "uint8"), np.zeros(b, "uint8"))


def test_maiores_secas_ordena_por_fatia_entre_confiaveis():
    # 4 municípios, 3 meses bem separados (separacao=1): fatias 1/4, 4/4, 2/4
    cls = grade([[0, 0, 3], [3, 1, 2], [3, 2, 3], [3, 0, 3]])
    baixa = np.zeros_like(cls)
    r = dq.maiores_secas(cls, baixa, n=3, separacao=1)
    assert [x["t"] for x in r] == [1, 2, 0]              # t=0 e t=2 empatam em 1/4: o mais recente primeiro
    assert r[0] == {"t": 1, "n_seca": 4, "n_confiaveis": 4, "n_validos": 4}


def test_maiores_secas_nao_conta_baixa_confianca_como_seca():
    cls = grade([[0, 0], [0, 3], [0, 3], [0, 3]])
    baixa = grade([[1, 0], [1, 0], [0, 0], [0, 0]])
    r = dq.maiores_secas(cls, baixa, n=2, separacao=1)
    por_mes = {x["t"]: x for x in r}
    assert por_mes[0]["n_seca"] == 2 and por_mes[0]["n_confiaveis"] == 2     # só contam os 2 confiáveis (metade exata)
    assert por_mes[1]["n_seca"] == 1


def test_maiores_secas_mes_sem_maioria_confiavel_nao_vira_destaque():
    cls = grade([[0, 3]] * 4)
    baixa = grade([[1, 0]] * 3 + [[0, 0]])               # mês 0: 1 de 4 confiável, e todos em seca
    r = dq.maiores_secas(cls, baixa, n=5, separacao=1)
    assert [x["t"] for x in r] == [1]


def test_maiores_secas_um_por_episodio():
    nt = 30
    cls = np.full((4, nt), 3, dtype="uint8")
    cls[:, 5] = 0
    cls[:, 6] = 0                      # mesmo episódio de t=5
    cls[:2, 20] = 0
    r = dq.maiores_secas(cls, np.zeros_like(cls), n=5, separacao=12)
    assert [x["t"] for x in r] == [6, 20]              # empate 5 e 6: o mais recente primeiro; 5 sai por vizinhança


def test_maiores_secas_respeita_n_e_sem_seca_devolve_meses_com_zero():
    cls = np.full((3, 40), 3, dtype="uint8")
    r = dq.maiores_secas(cls, np.zeros_like(cls), n=2, separacao=12)
    assert len(r) == 2 and all(x["n_seca"] == 0 for x in r)


@pytest.mark.parametrize("n,sep", [(0, 12), (3, 0)])
def test_maiores_secas_argumentos_invalidos(n, sep):
    with pytest.raises(ValueError):
        dq.maiores_secas(grade([[3]]), grade([[0]]), n=n, separacao=sep)


def test_maiores_secas_sem_nenhum_mes_elegivel_devolve_lista_vazia():
    assert dq.maiores_secas(grade([[ND, ND]]), grade([[0, 0]])) == []


# ------------------------------------------------------------------ clima
import pandas as pd


def chuva_padrao(anos=(2000, 2001, 2002, 2003), por_mes=None, nm=3):
    """Chuva sintética: cada mês do calendário recebe o valor de `por_mes`; um município por linha."""
    por_mes = por_mes or [10, 20, 100, 150, 100, 20, 0, 0, 0, 5, 20, 40]
    meses = pd.period_range(f"{anos[0]}-01", f"{anos[-1]}-12", freq="M")
    linha = np.array([por_mes[p.month - 1] for p in meses], dtype="float64")
    return np.tile(linha, (nm, 1)), meses


def test_climatologia_devolve_doze_valores_da_serie_construida():
    chuva, meses = chuva_padrao()
    assert dq.climatologia_mensal(chuva, meses, "2003-12") == [10, 20, 100, 150, 100, 20, 0, 0, 0, 5, 20, 40]


def test_climatologia_so_usa_anos_ate_treino_fim():
    chuva, meses = chuva_padrao()
    chuva = chuva.copy()
    chuva[:, meses >= pd.Period("2002-01")] += 1000           # depois do treino: não pode aparecer
    assert dq.climatologia_mensal(chuva, meses, "2001-12")[2] == 100


def test_climatologia_ignora_municipios_nao_incluidos_e_nan():
    chuva, meses = chuva_padrao()
    chuva = chuva.copy()
    chuva[2, :] = 9999                                        # município excluído
    chuva[0, 5] = np.nan                                      # um NaN solto não derruba a média
    r = dq.climatologia_mensal(chuva, meses, "2003-12", incluir=[True, True, False])
    assert r[0] == 10 and r[1] == 20 and r[5] == 20


def test_climatologia_mes_sem_dado_e_erro():
    chuva, meses = chuva_padrao()
    chuva = chuva.copy()
    chuva[:, meses.month == 7] = np.nan
    with pytest.raises(ValueError, match="mês 7"):
        dq.climatologia_mensal(chuva, meses, "2003-12")


@pytest.mark.parametrize("forma", [(3, 47), (36,)])
def test_chuva_com_forma_errada_e_erro(forma):
    chuva, meses = chuva_padrao()
    with pytest.raises(ValueError, match="município x mês"):
        dq.climatologia_mensal(np.zeros(forma), meses, "2003-12")


def test_incluir_vazio_ou_com_tamanho_errado_e_erro():
    chuva, meses = chuva_padrao()
    with pytest.raises(ValueError, match="incluir"):
        dq.climatologia_mensal(chuva, meses, "2003-12", incluir=[False, False, False])
    with pytest.raises(ValueError, match="incluir"):
        dq.climatologia_mensal(chuva, meses, "2003-12", incluir=[True])


def test_estacao_chuvosa_acha_os_quatro_meses_mais_chuvosos():
    chuva, meses = chuva_padrao()
    r = dq.estacao_chuvosa(chuva, meses, "2003-12")
    assert r["meses"] == [2, 3, 4, 5] and r["media_mm"] == 370        # 20+100+150+100 é a maior soma de quatro seguidos
    assert [x["ano"] for x in r["anos"]] == [2000, 2001, 2002, 2003] and all(x["desvio_pct"] == 0 for x in r["anos"])


def test_estacao_que_passa_do_ano_conta_no_ano_de_inicio():
    por_mes = [150, 100, 20, 0, 0, 0, 0, 0, 0, 5, 40, 100]            # nov-dez-jan-fev: 40+100+150+100 = 390
    chuva, meses = chuva_padrao(por_mes=por_mes)
    r = dq.estacao_chuvosa(chuva, meses, "2003-12")
    assert r["meses"] == [11, 12, 1, 2] and r["media_mm"] == 390
    assert [x["ano"] for x in r["anos"]] == [2000, 2001, 2002]          # a de 2003 pediria jan-fev de 2004, que não existe


def test_desvio_e_relativo_a_media_do_treino_e_estacoes_incompletas_ficam_fora():
    chuva, meses = chuva_padrao(anos=(2000, 2001, 2002, 2003))
    chuva = chuva.copy()
    chuva[:, (meses.year == 2003) & (meses.month >= 2) & (meses.month <= 5)] *= 2   # estação de 2003 dobra
    r = dq.estacao_chuvosa(chuva, meses, "2002-12")
    por_ano = {x["ano"]: x for x in r["anos"]}
    assert por_ano[2001]["desvio_pct"] == 0 and por_ano[2003]["desvio_pct"] == 100
    assert r["media_mm"] == 370                                  # a média ignora 2003 (depois do treino)
    parcial = chuva[:, :-2]                                       # corta nov e dez de 2003: nada muda para fev-mai
    assert 2003 in [x["ano"] for x in dq.estacao_chuvosa(parcial, meses[:-2], "2002-12")["anos"]]
    cortada = chuva[:, :-10]                                      # termina em fev/2003: estação de 2003 incompleta
    assert 2003 not in [x["ano"] for x in dq.estacao_chuvosa(cortada, meses[:-10], "2002-12")["anos"]]


def test_estacao_com_nan_na_janela_nao_entra():
    chuva, meses = chuva_padrao()
    chuva = chuva.copy()
    chuva[1, (meses.year == 2001) & (meses.month == 3)] = np.nan
    assert 2001 not in [x["ano"] for x in dq.estacao_chuvosa(chuva, meses, "2003-12")["anos"]]


def test_estacao_largura_invalida_e_sem_treino_e_erro():
    chuva, meses = chuva_padrao()
    with pytest.raises(ValueError, match="largura"):
        dq.estacao_chuvosa(chuva, meses, "2003-12", largura=0)
    with pytest.raises(ValueError, match="sem chuva observada"):
        dq.estacao_chuvosa(chuva, meses, "1990-12")
