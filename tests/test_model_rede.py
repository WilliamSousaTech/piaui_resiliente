"""Testes de src/model/rede.py. Tensores minúsculos feitos à mão; nada de arquivo do projeto nem rede."""
import json

import numpy as np
import pytest
import torch

from src.model import rede as rd

ARQ = rd.Arquitetura(n_entradas=6, n_cabecas=2, ocultas=(8, 4), dropout=0.0)


def lote(n=32, d=6, c=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.randn(n, c, generator=g)
    return x, y, torch.ones(n, c, dtype=torch.bool)


# ------------------------------------------------------------------ arquitetura e forward
def test_forward_shape_dtype_e_cabecas_independentes():
    r = rd.criar(ARQ, 1)
    x, _, _ = lote()
    out = r(x)
    assert out.shape == (32, 2) and out.dtype == torch.float32
    assert isinstance(r.cabecas, torch.nn.ModuleList) and len(r.cabecas) == 2
    assert not torch.equal(out[:, 0], out[:, 1])                           # cabeças com pesos próprios


def test_forward_recusa_entrada_errada():
    r = rd.criar(ARQ, 1)
    with pytest.raises(rd.RedeError, match="shape"):
        r(torch.zeros(4, 5))
    with pytest.raises(rd.RedeError, match="shape"):
        r(torch.zeros(6))
    with pytest.raises(rd.RedeError, match="float32"):
        r(torch.zeros(4, 6, dtype=torch.float64))


@pytest.mark.parametrize("kw", [dict(n_entradas=0, n_cabecas=1), dict(n_entradas=3, n_cabecas=0),
                                dict(n_entradas=3, n_cabecas=1, ocultas=()),
                                dict(n_entradas=3, n_cabecas=1, ocultas=(4, 0)),
                                dict(n_entradas=3, n_cabecas=1, ocultas=(4.5,)),
                                dict(n_entradas=3, n_cabecas=1, dropout=1.0),
                                dict(n_entradas=3, n_cabecas=1, dropout=-0.1)])
def test_arquitetura_invalida_e_recusada(kw):
    with pytest.raises(rd.RedeError):
        rd.Arquitetura(**kw)


def test_arquitetura_ida_e_volta_e_dict_invalido():
    assert rd.Arquitetura.de_dict(json.loads(json.dumps(ARQ.para_dict()))) == ARQ
    with pytest.raises(rd.RedeError, match="inválida"):
        rd.Arquitetura.de_dict({"n_entradas": 3})
    with pytest.raises(rd.RedeError, match="inválida"):
        rd.Arquitetura.de_dict({**ARQ.para_dict(), "ocultas": "x"})


# ------------------------------------------------------------------ zero pesos pré-treinados
def test_pesos_iniciais_sao_aleatorios_e_determinados_so_pela_semente():
    a, b, c = rd.criar(ARQ, 42), rd.criar(ARQ, 42), rd.criar(ARQ, 43)
    for (k, va), vb, vc in zip(a.state_dict().items(), b.state_dict().values(), c.state_dict().values()):
        assert torch.equal(va, vb), k                                       # mesma semente, mesmos pesos
    assert any(not torch.equal(va, vc) for va, vc in zip(a.state_dict().values(), c.state_dict().values()))
    w = a.tronco[0].weight
    assert w.std() > 0 and not torch.equal(w, torch.zeros_like(w))
    with pytest.raises(rd.RedeError, match="seed"):
        rd.criar(ARQ, 1.5)
    with pytest.raises(rd.RedeError, match="seed"):
        rd.criar(ARQ, True)


# ------------------------------------------------------------------ perda mascarada
def test_perda_bate_com_a_conta_a_mao_e_faz_media_entre_cabecas():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = torch.tensor([[0.0, 2.0], [1.0, 0.0], [5.0, 0.0]])
    m = torch.tensor([[True, True], [True, False], [False, False]])
    # cabeça 0: linhas 0 e 1 -> (1 + 4)/2 = 2,5; cabeça 1: só a linha 0 -> 0. Média simples das cabeças: 1,25
    assert rd.perda_mascarada(pred, y, m).item() == pytest.approx(1.25)


def test_posicao_sem_rotulo_nao_gera_nan_nem_gradiente():
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    y = torch.tensor([[0.0, float("nan")], [float("inf"), 0.0]])
    m = torch.tensor([[True, False], [False, True]])
    perda = rd.perda_mascarada(pred, y, m)
    perda.backward()
    assert torch.isfinite(perda) and torch.isfinite(pred.grad).all()
    assert pred.grad[0, 1] == 0 and pred.grad[1, 0] == 0                   # fora da máscara: gradiente exatamente 0
    assert pred.grad[0, 0] != 0 and pred.grad[1, 1] != 0


def test_lote_sem_rotulo_algum_da_perda_zero_com_gradiente_finito():
    pred = torch.randn(4, 2, requires_grad=True)
    y = torch.full((4, 2), float("nan"))
    perda = rd.perda_mascarada(pred, y, torch.zeros(4, 2, dtype=torch.bool))
    perda.backward()
    assert perda.item() == 0.0 and (pred.grad == 0).all()


def test_cabeca_sem_rotulo_no_lote_e_ignorada_nao_diluida():
    pred = torch.tensor([[2.0, 9.0], [2.0, 9.0]])
    y = torch.zeros(2, 2)
    m = torch.tensor([[True, False], [True, False]])                       # só a cabeça 0 tem rótulo: erro 4
    assert rd.perda_mascarada(pred, y, m).item() == pytest.approx(4.0)


def test_perda_recusa_entradas_incoerentes():
    p = torch.zeros(3, 2)
    with pytest.raises(rd.RedeError, match="mesmo shape"):
        rd.perda_mascarada(p, torch.zeros(3, 1), torch.ones(3, 2, dtype=torch.bool))
    with pytest.raises(rd.RedeError, match="mesmo shape"):
        rd.perda_mascarada(torch.zeros(3), torch.zeros(3), torch.ones(3, dtype=torch.bool))
    with pytest.raises(rd.RedeError, match="bool"):
        rd.perda_mascarada(p, torch.zeros(3, 2), torch.ones(3, 2))
    y = torch.zeros(3, 2)
    y[1, 1] = float("nan")
    with pytest.raises(rd.RedeError, match="não finito"):
        rd.perda_mascarada(p, y, torch.ones(3, 2, dtype=torch.bool))       # NaN onde a máscara diz que há rótulo


# ------------------------------------------------------------------ treino de verdade: forward -> loss -> backward -> step
def test_um_passo_de_otimizador_muda_todos_os_pesos_e_a_perda_cai():
    r = rd.criar(ARQ, 3)
    x, y, m = lote(64)
    antes = {k: v.clone() for k, v in r.state_dict().items()}
    opt = torch.optim.Adam(r.parameters(), lr=1e-2)
    perdas = []
    for _ in range(30):
        opt.zero_grad()
        perda = rd.perda_mascarada(r(x), y, m)
        perda.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in r.parameters())
        opt.step()
        perdas.append(perda.item())
    assert perdas[-1] < perdas[0]
    assert all(not torch.equal(antes[k], v) for k, v in r.state_dict().items())


def test_dropout_so_age_no_modo_treino():
    r = rd.criar(rd.Arquitetura(6, 2, (16,), dropout=0.5), 5)
    x, _, _ = lote()
    r.eval()
    assert torch.equal(r(x), r(x))
    r.train()
    assert not torch.equal(r(x), r(x))


# ------------------------------------------------------------------ salvar e carregar
def test_ida_e_volta_preserva_as_previsoes_e_o_meta(tmp_path):
    r = rd.criar(ARQ, 9)
    r.eval()
    x, _, _ = lote()
    p = tmp_path / "m" / "rede.pt"
    meta = {"dataset_sha256": "ab" * 32, "seed": 9, "historico": [0.5, 0.4]}
    sha = rd.salvar(r, p, meta)
    assert len(sha) == 64 and (tmp_path / "m" / "rede.json").is_file() and not list(p.parent.glob("*.tmp"))
    r2, meta2 = rd.carregar(p)
    assert meta2 == meta and not r2.training
    assert torch.equal(r(x), r2(x))
    assert json.loads((tmp_path / "m" / "rede.json").read_text(encoding="utf-8"))["pesos_sha256"] == sha


def test_pesos_adulterados_ou_arquivos_faltando_sao_recusados(tmp_path):
    p = tmp_path / "rede.pt"
    rd.salvar(rd.criar(ARQ, 1), p, {})
    p.write_bytes(p.read_bytes() + b"\0")
    with pytest.raises(rd.RedeError, match="SHA-256"):
        rd.carregar(p)
    with pytest.raises(rd.RedeError, match="faltam"):
        rd.carregar(tmp_path / "outra.pt")
    q = tmp_path / "so_pesos.pt"
    rd.salvar(rd.criar(ARQ, 1), q, {})
    q.with_suffix(".json").unlink()
    with pytest.raises(rd.RedeError, match="faltam"):
        rd.carregar(q)


def test_json_invalido_ou_de_outra_versao_e_recusado(tmp_path):
    p = tmp_path / "rede.pt"
    rd.salvar(rd.criar(ARQ, 1), p, {})
    j = p.with_suffix(".json")
    doc = json.loads(j.read_text(encoding="utf-8"))
    j.write_text(json.dumps({**doc, "versao_formato": 99}), encoding="utf-8")
    with pytest.raises(rd.RedeError, match="versão"):
        rd.carregar(p)
    j.write_text("{não é json", encoding="utf-8")
    with pytest.raises(rd.RedeError, match="inválido"):
        rd.carregar(p)
    j.write_text(json.dumps({k: v for k, v in doc.items() if k != "meta"}), encoding="utf-8")
    with pytest.raises(rd.RedeError, match="inválido"):
        rd.carregar(p)


class _Perigoso:
    def __reduce__(self):                                                  # um pickle malicioso executaria isto
        return (print, ("executou",))


def test_arquivo_de_pesos_com_pickle_de_objeto_e_recusado(tmp_path, capsys):
    p = tmp_path / "rede.pt"
    rd.salvar(rd.criar(ARQ, 1), p, {})
    torch.save({"x": _Perigoso()}, p)                                       # troca o conteúdo e acerta o SHA-256 de propósito
    j = p.with_suffix(".json")
    doc = json.loads(j.read_text(encoding="utf-8"))
    doc["pesos_sha256"] = rd._sha256(p)
    j.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(rd.RedeError, match="não foi possível carregar"):
        rd.carregar(p)
    assert "executou" not in capsys.readouterr().out


def test_arquitetura_diferente_da_dos_pesos_e_recusada(tmp_path):
    p = tmp_path / "rede.pt"
    rd.salvar(rd.criar(ARQ, 1), p, {})
    j = p.with_suffix(".json")
    doc = json.loads(j.read_text(encoding="utf-8"))
    doc["arquitetura"]["ocultas"] = [8, 5]
    j.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(rd.RedeError, match="não foi possível carregar"):
        rd.carregar(p)


def test_pesos_com_cabeca_faltando_ou_sobrando_sao_recusados_em_vez_de_meio_aleatorios(tmp_path):
    p = tmp_path / "rede.pt"
    rd.salvar(rd.criar(rd.Arquitetura(6, 1, (8, 4), 0.0), 1), p, {})          # pesos de uma rede de 1 cabeça
    j = p.with_suffix(".json")
    doc = json.loads(j.read_text(encoding="utf-8"))
    doc["arquitetura"]["n_cabecas"] = 2                                       # o json diz 2: a segunda ficaria aleatória
    j.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(rd.RedeError, match="não foi possível carregar"):
        rd.carregar(p)
    rd.salvar(rd.criar(ARQ, 1), p, {})                                        # agora o contrário: pesos de 2, json de 1
    doc = json.loads(j.read_text(encoding="utf-8"))
    doc["arquitetura"]["n_cabecas"] = 1
    j.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(rd.RedeError, match="não foi possível carregar"):
        rd.carregar(p)


def test_salvar_recusa_nome_e_meta_invalidos(tmp_path):
    r = rd.criar(ARQ, 1)
    with pytest.raises(rd.RedeError, match=".pt"):
        rd.salvar(r, tmp_path / "rede.bin", {})
    with pytest.raises(rd.RedeError, match="JSON puro"):
        rd.salvar(r, tmp_path / "rede.pt", {"x": float("nan")})
    with pytest.raises(rd.RedeError, match="JSON puro"):
        rd.salvar(r, tmp_path / "rede.pt", {"x": np.float32(1.0)})
    assert not list(tmp_path.iterdir())                                     # nada gravado quando recusa
