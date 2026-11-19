"""Baixa, SEM interpretar, a série diária de cada reservatório do Piauí no SAR (ANA).

Por que existe: o ambiente de desenvolvimento não alcança o site da ANA. Este comando roda na máquina da equipe e
guarda os bytes exatos que a ANA devolveu (um .html por reservatório), com URL, data do download, status e SHA-256
(`selo.json`), mais um .zip para anexar ao chat. Nada é convertido aqui: o leitor (`reservatorios.py`) só será escrito
depois de olhar estes arquivos reais (contrato 5.4: não definir colunas antes de ver o arquivo).

A rota é a página de histórico do SAR (`/sar0/Medicao`), a mesma que o pacote `reservatoriosBR` usa. Não é uma API
documentada pela ANA: se o formato mudar, o leitor recusa. A verificação do certificado HTTPS fica sempre ligada.

Sem ML e sem framework web. Uso (na raiz do projeto):
    python -m src.data.baixar_sar
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import pandas as pd

URL_BASE = "https://www.ana.gov.br/sar0/Medicao"
LISTA_PADRAO = Path("data") / "reference" / "reservatorios_pi_sar.csv"
SAIDA_PADRAO = Path("data") / "raw" / "sar"
SELO = "selo.json"
ZIP = "sar_pi_bruto.zip"
USER_AGENT = "piaui-resiliente/0.3 (projeto de pesquisa, equipe V.I.G.I.A.)"
INICIO_PADRAO = date(2015, 1, 1)
COLUNAS_LISTA = ("codigo", "reservatorio", "municipio_sar", "cod_ibge", "nome_ibge")

Abrir = Callable[[str, float], "tuple[int, bytes]"]


class BaixarSarError(ValueError):
    """Lista inválida, intervalo inválido, arquivo que já existe ou falha de rede."""


def montar_url(codigo: int, ini: date, fim: date) -> str:
    pares = [("dropDownListEstados", ""), ("dropDownListReservatorios", codigo),
             ("dataInicial", ini.strftime("%d/%m/%Y")), ("dataFinal", fim.strftime("%d/%m/%Y")), ("button", "Buscar")]
    return f"{URL_BASE}?{urlencode(pares)}"


def ler_codigos(caminho: Path = LISTA_PADRAO) -> list[tuple[int, str]]:
    """(código SAR, nome) de cada reservatório da lista de referência. Recusa coluna faltando, código repetido ou não numérico."""
    df = pd.read_csv(caminho, dtype=str, keep_default_na=False)
    if not set(COLUNAS_LISTA) <= set(df.columns):
        raise BaixarSarError(f"a lista precisa das colunas {COLUNAS_LISTA}")
    if df.empty:
        raise BaixarSarError("a lista de reservatórios está vazia")
    if not df["codigo"].map(lambda v: re.fullmatch(r"[0-9]{1,8}", v) is not None).all():
        raise BaixarSarError("código de reservatório que não é número inteiro")
    if df["codigo"].duplicated().any():
        raise BaixarSarError("código de reservatório repetido na lista")
    return [(int(c), n) for c, n in zip(df["codigo"], df["reservatorio"])]


def _abrir_urllib(url: str, timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:     # verificação de certificado: padrão, ligada
            return int(r.status), r.read()
    except urllib.error.HTTPError as e:
        return int(e.code), e.read()
    except urllib.error.URLError as e:
        if isinstance(e.reason, ssl.SSLCertVerificationError):
            raise BaixarSarError("o certificado HTTPS do site da ANA não foi aceito. Não desligue a verificação: "
                                 "copie esta mensagem e envie ao chat") from e
        raise BaixarSarError(f"sem conexão com a ANA: {e.reason}") from e
    except TimeoutError as e:
        raise BaixarSarError("a ANA não respondeu a tempo") from e


def _gravar_atomico(destino: Path, dados: bytes) -> None:
    tmp = destino.with_name(destino.name + ".tmp")
    tmp.write_bytes(dados)
    os.replace(tmp, destino)


def _ler_selo(saida: Path) -> dict:
    p = saida / SELO
    if not p.exists():
        return {"fonte": URL_BASE, "aviso": "bytes exatos da resposta da ANA, sem interpretação", "arquivos": {}, "falhas": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def _baixar_um(url: str, abrir: Abrir, timeout: float, tentativas: int, pausa: float, dormir: Callable[[float], None]) -> bytes:
    motivo = ""
    for n in range(1, tentativas + 1):
        try:
            status, corpo = abrir(url, timeout)
        except BaixarSarError as e:
            motivo = str(e)
        else:
            if status == 200 and b"<table" in corpo.lower():
                return corpo
            motivo = f"HTTP {status}" if status != 200 else "resposta sem tabela de dados"
            if 400 <= status < 500:
                break
        if n < tentativas:
            dormir(pausa * n)
    raise BaixarSarError(motivo)


def baixar_todos(codigos: list[tuple[int, str]], saida: Path, ini: date, fim: date, *, abrir: Abrir = _abrir_urllib,
                 pausa: float = 1.0, timeout: float = 60.0, tentativas: int = 3, substituir: bool = False,
                 dormir: Callable[[float], None] = time.sleep,
                 agora: Callable[[], datetime] = lambda: datetime.now(timezone.utc)) -> dict:
    """Grava `<código>.html`, `selo.json` e `sar_pi_bruto.zip` em `saida`. Devolve o selo. Recusa reescrever sem `substituir`."""
    if ini >= fim:
        raise BaixarSarError("a data inicial precisa ser anterior à final")
    if fim > agora().date():
        raise BaixarSarError("a data final não pode estar no futuro")
    saida = Path(saida)
    saida.mkdir(parents=True, exist_ok=True)
    ja = [c for c, _ in codigos if (saida / f"{c}.html").exists()]
    if ja and not substituir:
        raise BaixarSarError(f"já existem arquivos para {sorted(ja)}; use --substituir para baixar de novo")
    selo = _ler_selo(saida)
    for i, (cod, nome) in enumerate(codigos):
        if i:
            dormir(pausa)
        url = montar_url(cod, ini, fim)
        try:
            corpo = _baixar_um(url, abrir, timeout, tentativas, pausa, dormir)
        except BaixarSarError as e:
            selo["falhas"][str(cod)] = f"{nome}: {e}"
            continue
        _gravar_atomico(saida / f"{cod}.html", corpo)
        selo["falhas"].pop(str(cod), None)
        selo["arquivos"][f"{cod}.html"] = {"codigo": cod, "reservatorio": nome, "url": url, "status": 200,
                                           "baixado_em_utc": agora().isoformat(timespec="seconds"), "bytes": len(corpo),
                                           "sha256": hashlib.sha256(corpo).hexdigest()}
    _gravar_atomico(saida / SELO, (json.dumps(selo, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    tmp = saida / (ZIP + ".tmp")
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as z:
        for nome_arq in sorted(selo["arquivos"]):
            if (saida / nome_arq).exists():
                z.write(saida / nome_arq, nome_arq)
        z.write(saida / SELO, SELO)
    os.replace(tmp, saida / ZIP)
    return selo


def main(argv: list[str] | None = None, *, saida: Callable[[str], None] = print, abrir: Abrir = _abrir_urllib,
         dormir: Callable[[float], None] = time.sleep) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.data.baixar_sar", description=__doc__.split("\n\n")[0])
    ap.add_argument("--lista", type=Path, default=LISTA_PADRAO)
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    ap.add_argument("--inicio", type=date.fromisoformat, default=INICIO_PADRAO, help="AAAA-MM-DD")
    ap.add_argument("--fim", type=date.fromisoformat, default=None, help="AAAA-MM-DD (padrão: hoje, UTC)")
    ap.add_argument("--codigos", default="", help="só estes códigos, separados por vírgula (para repetir os que falharam)")
    ap.add_argument("--pausa", type=float, default=1.0, help="segundos entre pedidos")
    ap.add_argument("--substituir", action="store_true")
    a = ap.parse_args(argv)
    try:
        todos = ler_codigos(a.lista)
        if a.codigos:
            pedidos = [int(x) for x in a.codigos.split(",") if x.strip()]
            falta = sorted(set(pedidos) - {c for c, _ in todos})
            if falta:
                raise BaixarSarError(f"códigos fora da lista: {falta}")
            todos = [(c, n) for c, n in todos if c in pedidos]
        fim = a.fim or datetime.now(timezone.utc).date()
        selo = baixar_todos(todos, a.saida, a.inicio, fim, abrir=abrir, pausa=a.pausa, substituir=a.substituir, dormir=dormir)
    except (BaixarSarError, ValueError, OSError) as e:
        saida(f"ERRO: {e}")
        return 2
    n_ok = sum(1 for c, _ in todos if f"{c}.html" in selo["arquivos"] and str(c) not in selo["falhas"])
    saida(f"Baixados {n_ok} de {len(todos)} reservatórios em {a.saida}.")
    for cod, motivo in sorted(selo["falhas"].items()):
        saida(f"  falhou {cod}: {motivo}")
    if selo["falhas"]:
        saida("Repita só os que falharam: python -m src.data.baixar_sar --codigos " + ",".join(sorted(selo["falhas"])))
        return 1
    saida(f"Anexe ao chat o arquivo: {a.saida / ZIP}")
    return 0


if __name__ == "__main__":      # pragma: no cover
    raise SystemExit(main())
