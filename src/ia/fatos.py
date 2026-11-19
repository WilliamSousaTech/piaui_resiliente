"""Ficha do município: os únicos números que a IA pode usar num rascunho.

Lê o pacote de dados do site (`payload`, montado por `src.web.build_site.montar_payload` a partir de arquivos já
conferidos pelo manifest) e devolve só inteiros e textos que o site já mostra. A ficha em texto (`ficha_texto`) é o
que o modelo de linguagem recebe; qualquer número que apareça no rascunho e não esteja nela é recusado
(`numeros_fora`). Assim a IA não calcula, não estima e não inventa valor.

Sem rede, sem ML, sem framework web e sem importar `src.web`: o contrato com o site é só o formato do pacote
(chaves `municipios`, `meses`, `abertura`, `escala_alvo`, `spi`, `cls`, `baixa`, `precip`, `classes`, `escala_spi`,
`escala_chuva`), conferido em `tests/test_ia_fatos.py`.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass

import numpy as np

NA16 = -32768                 # "sem dado" nos inteiros de 16 bits do pacote
SEM_DADO = 255                # "sem dado" nas classes do pacote
JANELA_MESES = 12
MESES = ("janeiro", "fevereiro", "março", "abril", "maio", "junho", "julho", "agosto", "setembro", "outubro",
         "novembro", "dezembro")
_NUMERO = re.compile(r"\d+(?:[.,]\d+)*")
_MILHAR = re.compile(r"\.(?=\d{3}(?!\d))")


class FatosError(ValueError):
    """Município ou mês que não permite montar a ficha."""


@dataclass(frozen=True)
class Fatos:
    cod_ibge: str
    nome: str
    mes: str                          # AAAA-MM, o mês de referência
    escala: int                       # SPI-k
    excluido: bool                    # município sem dado (D11)
    spi_cent: int | None              # SPI x 100
    classe: str | None
    em_seca: bool | None
    baixa_confianca: bool
    chuva_mes_dec: int | None         # mm x 10
    chuva_12m_dec: int | None         # mm x 10; None se faltar algum dos 12 meses
    meses_em_seca_12m: int            # entre as leituras confiáveis dos 12 meses até `mes`
    leituras_confiaveis_12m: int

    @property
    def utilizavel(self) -> bool:
        """Só há o que escrever quando o município tem SPI no mês de referência."""
        return not self.excluido and self.spi_cent is not None


def _ints(b64: str, dtype: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=dtype)


def _matriz(b64: str, dtype: str, nm: int, nt: int, onde: str) -> np.ndarray:
    a = _ints(b64, dtype)
    if a.size != nm * nt:
        raise FatosError(f"{onde}: {a.size} valores, esperado {nm} x {nt}")
    return a.reshape(nm, nt)


def fatos_municipio(payload: dict, cod_ibge: str, mes: str | None = None, escala: int | None = None) -> Fatos:
    """Ficha de `cod_ibge` em `mes` (padrão: o mês de abertura do site) e escala `escala` (padrão: a do alvo)."""
    cods = [m["cod"] for m in payload["municipios"]]
    if cod_ibge not in cods:
        raise FatosError(f"município {cod_ibge!r} não está no pacote")
    i = cods.index(cod_ibge)
    meses = list(payload["meses"])
    mes = mes or meses[int(payload["abertura"])]
    if mes not in meses:
        raise FatosError(f"mês {mes!r} fora da janela {meses[0]}..{meses[-1]}")
    t = meses.index(mes)
    if t < JANELA_MESES - 1:
        raise FatosError(f"mês {mes}: menos de {JANELA_MESES} meses de histórico na janela")
    k = int(escala if escala is not None else payload["escala_alvo"])
    if str(k) not in payload["spi"]:
        raise FatosError(f"escala SPI-{k} não está no pacote")

    nm, nt = len(cods), len(meses)
    spi = _matriz(payload["spi"][str(k)], "<i2", nm, nt, f"spi{k}")[i]
    cls = _matriz(payload["cls"][str(k)], "u1", nm, nt, f"cls{k}")[i]
    baixa = _matriz(payload["baixa"][str(k)], "u1", nm, nt, f"baixa{k}")[i]
    chuva = _matriz(payload["precip"], "<i2", nm, nt, "precip")[i]
    classes = payload["classes"]
    seca_id = {idx for idx, c in enumerate(classes) if c["seca"]}      # a posição na lista é o número da classe

    janela = slice(t - JANELA_MESES + 1, t + 1)
    confiavel = (cls[janela] != SEM_DADO) & (baixa[janela] == 0)
    em_seca = np.isin(cls[janela], list(seca_id)) & confiavel
    chuva_j = chuva[janela]
    tem_spi = cls[t] != SEM_DADO
    return Fatos(
        cod_ibge=cod_ibge, nome=str(payload["municipios"][i]["nome"]), mes=mes, escala=k,
        excluido=bool(payload["municipios"][i]["exc"]),
        spi_cent=int(spi[t]) if tem_spi and spi[t] != NA16 else None,
        classe=str(classes[int(cls[t])]["rotulo"]) if tem_spi else None,
        em_seca=bool(int(cls[t]) in seca_id) if tem_spi else None,
        baixa_confianca=bool(baixa[t]) if tem_spi else False,
        chuva_mes_dec=int(chuva[t]) if chuva[t] != NA16 else None,
        chuva_12m_dec=None if (chuva_j == NA16).any() else int(chuva_j.astype("int64").sum()),
        meses_em_seca_12m=int(em_seca.sum()), leituras_confiaveis_12m=int(confiavel.sum()))


# ------------------------------------------------------------------ texto da ficha
def _decimal_br(valor: int, escala: int) -> str:
    """Inteiro com `escala` (10 ou 100) em texto brasileiro, sem float: -123 (escala 100) -> '−1,23'."""
    casas = {10: 1, 100: 2}[escala]
    sinal = "−" if valor < 0 else ""
    inteiro, resto = divmod(abs(valor), escala)
    milhar = f"{inteiro:,}".replace(",", ".")
    return f"{sinal}{milhar},{resto:0{casas}d}"


def mes_por_extenso(iso: str) -> str:
    ano, mm = iso.split("-")
    return f"{MESES[int(mm) - 1]} de {ano}"


def ficha_texto(f: Fatos) -> str:
    """O texto que o modelo recebe. Só aparece aqui o que a IA pode citar."""
    if not f.utilizavel:
        raise FatosError(f"{f.nome} ({f.cod_ibge}) não tem SPI em {f.mes}: nada a escrever")
    linhas = [
        f"Município: {f.nome} ({f.cod_ibge})",
        f"Mês de referência: {mes_por_extenso(f.mes)}",
        f"SPI-{f.escala} observado: {_decimal_br(f.spi_cent, 100)} (classe: {f.classe})",
        "Leitura de baixa confiança: " + ("sim (pouca chuva esperada na janela; o SPI pode exagerar)"
                                          if f.baixa_confianca else "não"),
        "Chuva no mês: " + (f"{_decimal_br(f.chuva_mes_dec, 10)} mm" if f.chuva_mes_dec is not None else "sem dado"),
        f"Chuva nos {JANELA_MESES} meses até o mês de referência: "
        + (f"{_decimal_br(f.chuva_12m_dec, 10)} mm" if f.chuva_12m_dec is not None else "sem dado completo"),
        f"Meses em seca (SPI-{f.escala} de −1,0 ou menos) entre as leituras confiáveis dos últimos {JANELA_MESES} meses: "
        f"{f.meses_em_seca_12m} de {f.leituras_confiaveis_12m}",
    ]
    return "\n".join(linhas)


# ------------------------------------------------------------------ números permitidos
def normalizar_numero(token: str) -> str:
    """'1.234,5' e '1234,5' viram '1234,5'; '1.5' vira '1,5'. Só para comparar."""
    return _MILHAR.sub("", token).replace(".", ",")


def numeros(texto: str) -> list[str]:
    return [normalizar_numero(t) for t in _NUMERO.findall(texto)]


def numeros_permitidos(*textos: str) -> frozenset[str]:
    return frozenset(n for t in textos for n in numeros(t))


def numeros_fora(texto: str, permitidos: frozenset[str]) -> list[str]:
    """Números do `texto` que não estão entre os permitidos (sem repetição, na ordem em que aparecem)."""
    vistos: list[str] = []
    for n in numeros(texto):
        if n not in permitidos and n not in vistos:
            vistos.append(n)
    return vistos
