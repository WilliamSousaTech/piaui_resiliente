"""Baixador do SAR/ANA: guarda os bytes exatos, sela com SHA-256, recusa reescrever e nunca usa a rede nos testes."""
import hashlib
import json
import ssl
import urllib.error
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data import baixar_sar as bs

RAIZ = Path(__file__).resolve().parents[1]
INI, FIM = date(2015, 1, 1), date(2026, 11, 19)
AGORA = lambda: datetime(2026, 11, 19, 12, 0, tzinfo=timezone.utc)      # noqa: E731
CORPO = "<html><table><tr><td>12/11/2026</td><td>açude é</td></tr></table></html>".encode("utf-8")


def _nunca_dorme(_s):
    pass


def abrir_ok(url, timeout):
    return 200, CORPO + url.encode()                                    # corpo diferente por reservatório


def _rodar(tmp_path, codigos, abrir=abrir_ok, **kw):
    return bs.baixar_todos(codigos, tmp_path / "sar", INI, FIM, abrir=abrir, dormir=_nunca_dorme, agora=AGORA, **kw)


def test_url_exata_com_datas_codificadas_e_parametros_na_ordem_da_ana():
    assert bs.montar_url(12012, date(2015, 2, 3), FIM) == (        # dia 03 e mês 02: a troca dd/mm por mm/dd apareceria
        "https://www.ana.gov.br/sar0/Medicao?dropDownListEstados=&dropDownListReservatorios=12012"
        "&dataInicial=03%2F02%2F2015&dataFinal=19%2F11%2F2026&button=Buscar")


def test_lista_real_tem_25_reservatorios_do_piaui_todos_no_escopo_de_216():
    cods = bs.ler_codigos(RAIZ / bs.LISTA_PADRAO)
    assert len(cods) == 25 and len({c for c, _ in cods}) == 25
    df = pd.read_csv(RAIZ / bs.LISTA_PADRAO, dtype=str)
    esc = pd.read_csv(RAIZ / "data" / "reference" / "municipios_semiarido.csv", dtype=str)
    assert set(df["cod_ibge"]) <= set(esc["cod_ibge"])
    assert df["cod_ibge"].str.fullmatch(r"22[0-9]{5}").all()                  # código IBGE do Piauí, 7 dígitos, como texto


def test_lista_recusa_codigo_repetido_nao_numerico_e_coluna_faltando(tmp_path):
    def csv(linhas, cab="codigo,reservatorio,municipio_sar,cod_ibge,nome_ibge"):
        p = tmp_path / "l.csv"
        p.write_text(cab + "\n" + "\n".join(linhas) + "\n", encoding="utf-8")
        return p
    with pytest.raises(bs.BaixarSarError, match="repetido"):
        bs.ler_codigos(csv(["1,A,X,2200053,Y", "1,B,X,2200053,Y"]))
    for ruim in ("../x", "12a", "1 2", "123456789", "-1", ""):
        with pytest.raises(bs.BaixarSarError, match="inteiro"):
            bs.ler_codigos(csv([f"{ruim},A,X,2200053,Y"]))
    with pytest.raises(bs.BaixarSarError, match="colunas"):
        bs.ler_codigos(csv(["1,A"], cab="codigo,reservatorio"))
    with pytest.raises(bs.BaixarSarError, match="vazia"):
        bs.ler_codigos(csv([]))


def test_grava_bytes_exatos_selo_com_sha256_e_zip_para_anexar(tmp_path):
    selo = _rodar(tmp_path, [(12012, "ALDEIAS"), (12017, "ALGODÕES I *")])
    saida = tmp_path / "sar"
    for cod in (12012, 12017):
        esperado = CORPO + bs.montar_url(cod, INI, FIM).encode()
        assert (saida / f"{cod}.html").read_bytes() == esperado                 # nem um byte alterado
        info = selo["arquivos"][f"{cod}.html"]
        assert info["sha256"] == hashlib.sha256(esperado).hexdigest() and info["bytes"] == len(esperado)
        assert info["url"] == bs.montar_url(cod, INI, FIM) and info["baixado_em_utc"] == "2026-11-19T12:00:00+00:00"
    assert json.loads((saida / bs.SELO).read_text(encoding="utf-8")) == selo and selo["falhas"] == {}
    with zipfile.ZipFile(saida / bs.ZIP) as z:
        assert sorted(z.namelist()) == ["12012.html", "12017.html", bs.SELO]
        assert z.read("12012.html") == (saida / "12012.html").read_bytes()
    assert not list(saida.glob("*.tmp"))


def test_recusa_reescrever_sem_substituir_e_nao_toca_no_que_existe(tmp_path):
    _rodar(tmp_path, [(1, "A")])
    antes = (tmp_path / "sar" / "1.html").read_bytes()
    with pytest.raises(bs.BaixarSarError, match="substituir"):
        _rodar(tmp_path, [(1, "A")], abrir=lambda u, t: (200, b"<table>OUTRO</table>"))
    assert (tmp_path / "sar" / "1.html").read_bytes() == antes
    _rodar(tmp_path, [(1, "A")], abrir=lambda u, t: (200, b"<table>OUTRO</table>"), substituir=True)
    assert (tmp_path / "sar" / "1.html").read_bytes() == b"<table>OUTRO</table>"


def test_falha_de_um_nao_perde_os_outros_e_fica_no_selo(tmp_path):
    def abrir(url, timeout):
        if "=2&" in url:
            return 404, b"nao"
        if "=3&" in url:
            return 200, b"<html>manutencao</html>"                              # 200 sem tabela
        return abrir_ok(url, timeout)
    selo = _rodar(tmp_path, [(1, "A"), (2, "B"), (3, "C")], abrir=abrir)
    assert sorted(selo["arquivos"]) == ["1.html"]
    assert selo["falhas"]["2"].startswith("B: HTTP 404") and "sem tabela" in selo["falhas"]["3"]
    assert not (tmp_path / "sar" / "2.html").exists() and not (tmp_path / "sar" / "3.html").exists()


def test_tenta_de_novo_em_erro_5xx_e_de_rede_mas_nao_em_4xx(tmp_path):
    chamadas = {"n": 0}

    def abrir(url, timeout):
        chamadas["n"] += 1
        if chamadas["n"] == 1:
            return 503, b""
        if chamadas["n"] == 2:
            raise bs.BaixarSarError("sem conexão")
        return abrir_ok(url, timeout)
    selo = _rodar(tmp_path, [(1, "A")], abrir=abrir)
    assert chamadas["n"] == 3 and "1.html" in selo["arquivos"] and selo["falhas"] == {}
    chamadas["n"] = 0
    _rodar(tmp_path / "x", [(9, "Z")], abrir=lambda u, t: (chamadas.__setitem__("n", chamadas["n"] + 1) or 403, b""))
    assert chamadas["n"] == 1                                                   # 403 não repete


def test_pausa_entre_reservatorios_e_espera_crescente_entre_tentativas(tmp_path):
    esperas = []
    falha = lambda u, t: (500, b"")                                              # noqa: E731
    bs.baixar_todos([(1, "A"), (2, "B")], tmp_path / "s", INI, FIM, abrir=falha, pausa=2.0, tentativas=3,
                    dormir=esperas.append, agora=AGORA)
    assert esperas == [2.0, 4.0, 2.0, 2.0, 4.0]       # A: 2s,4s | pausa entre A e B: 2s | B: 2s,4s


def test_intervalo_invalido_e_data_no_futuro(tmp_path):
    for a, b in ((FIM, INI), (INI, INI)):                                       # invertido e igual
        with pytest.raises(bs.BaixarSarError, match="anterior"):
            bs.baixar_todos([(1, "A")], tmp_path / "s", a, b, abrir=abrir_ok, agora=AGORA)
    with pytest.raises(bs.BaixarSarError, match="futuro"):
        bs.baixar_todos([(1, "A")], tmp_path / "s", INI, date(2026, 11, 20), abrir=abrir_ok, agora=AGORA)
    assert not (tmp_path / "s" / "1.html").exists()


def test_repetir_so_os_que_falharam_junta_ao_selo_existente(tmp_path):
    def abrir(url, timeout):
        return (500, b"") if "=2&" in url else abrir_ok(url, timeout)
    _rodar(tmp_path, [(1, "A"), (2, "B")], abrir=abrir, tentativas=1)
    selo = _rodar(tmp_path, [(2, "B")])
    assert sorted(selo["arquivos"]) == ["1.html", "2.html"] and selo["falhas"] == {}
    with zipfile.ZipFile(tmp_path / "sar" / bs.ZIP) as z:
        assert sorted(z.namelist()) == ["1.html", "2.html", bs.SELO]


def test_cli_codigo_fora_da_lista_data_ruim_e_saida_de_sucesso(tmp_path):
    msgs = []
    base = ["--lista", str(RAIZ / bs.LISTA_PADRAO), "--saida", str(tmp_path / "sar"), "--pausa", "0"]
    assert bs.main(base + ["--codigos", "99999"], saida=msgs.append, abrir=abrir_ok, dormir=_nunca_dorme) == 2
    assert "fora da lista" in msgs[-1]
    assert bs.main(base + ["--fim", "2099-01-01"], saida=msgs.append, abrir=abrir_ok, dormir=_nunca_dorme) == 2
    assert "futuro" in msgs[-1]
    assert bs.main(base + ["--codigos", "12012,12017"], saida=msgs.append, abrir=abrir_ok, dormir=_nunca_dorme) == 0
    assert "Baixados 2 de 2" in msgs[-2] and msgs[-1].endswith(bs.ZIP)
    assert sorted(p.name for p in (tmp_path / "sar").glob("*.html")) == ["12012.html", "12017.html"]


def test_cli_com_falha_devolve_1_e_diz_quais_repetir(tmp_path):
    msgs = []
    base = ["--lista", str(RAIZ / bs.LISTA_PADRAO), "--saida", str(tmp_path / "sar"), "--pausa", "0", "--codigos", "12012,12017"]
    ruim = lambda u, t: (404, b"") if "=12017&" in u else abrir_ok(u, t)        # noqa: E731
    assert bs.main(base, saida=msgs.append, abrir=ruim, dormir=_nunca_dorme) == 1
    assert any("falhou 12017" in m for m in msgs) and msgs[-1].endswith("--codigos 12017")


def test_erro_de_certificado_nao_manda_desligar_a_verificacao(monkeypatch):
    def urlopen(req, timeout=None):
        raise urllib.error.URLError(ssl.SSLCertVerificationError("falha"))
    monkeypatch.setattr(bs.urllib.request, "urlopen", urlopen)
    with pytest.raises(bs.BaixarSarError, match="certificado") as e:
        bs._abrir_urllib("https://www.ana.gov.br/x", 5)
    assert "Não desligue" in str(e.value)


def test_urllib_devolve_status_http_de_erro_e_so_fala_https(monkeypatch):
    def urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 502, "ruim", {}, __import__("io").BytesIO(b"x"))
    monkeypatch.setattr(bs.urllib.request, "urlopen", urlopen)
    assert bs._abrir_urllib("https://www.ana.gov.br/x", 5) == (502, b"x")
    assert bs.URL_BASE.startswith("https://") and bs.montar_url(1, INI, FIM).startswith("https://")
