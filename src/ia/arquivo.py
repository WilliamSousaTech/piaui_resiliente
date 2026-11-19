"""Arquivo dos rascunhos de IA e a revisão humana (`data/reference/rascunhos_ia.json`).

O arquivo pertence a um `dataset_sha256` e a uma versão da base do checklist (`checklist_sha256`): se qualquer um dos
dois mudar, os rascunhos antigos deixam de valer e não vão ao site. Fica fora do manifest, como o contexto do El Niño,
porque é texto de apoio com revisor e data próprios, e não dado de modelo.

Fluxo: `gerar` cria rascunhos `pendente_revisao`; uma pessoa lê, edita o texto se quiser (no próprio JSON) e roda
`aprovar`, que valida o texto de novo (um número ou citação colocados à mão também são recusados) e grava quem revisou
e quando; ou roda `rejeitar`. No site, `revisado` leva o rótulo de `rascunhos.ROTULO`. `pendente_revisao` só aparece se o
site for montado com `incluir_nao_revisados` (decisão D14), sempre com `rascunhos.ROTULO_NAO_REVISADO` e depois de passar
de novo pela validação; `rejeitado` nunca aparece.

Sem rede, sem ML, sem framework web.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.ia import rascunhos as rs
from src.ia.checklist import Checklist

ARQUIVO_PADRAO = Path("data") / "reference" / "rascunhos_ia.json"
_HEX64 = r"^[0-9a-f]{64}$"
_REVISOR = re.compile(r"^[^\x00-\x1f\x7f<>]{2,80}$")


class ArquivoError(ValueError):
    """Arquivo ilegível, de outro dataset ou checklist, ou operação de revisão que não pode ser feita."""


class ArquivoRascunhos(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    versao_arquivo: Literal[1] = 1
    dataset_sha256: str = Field(pattern=_HEX64)
    checklist_sha256: str = Field(pattern=_HEX64)
    rascunhos: dict[str, rs.Rascunho] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _chaves(self):
        erradas = [k for k, r in self.rascunhos.items() if k != r.cod_ibge]
        if erradas:
            raise ValueError(f"chave diferente do cod_ibge do rascunho: {', '.join(erradas)}")
        return self


def _agora_iso(agora: datetime) -> str:
    if agora.tzinfo is None:
        raise ValueError("`agora` precisa ter fuso (UTC)")
    return agora.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ leitura e gravação
def carregar(caminho: Path | str = ARQUIVO_PADRAO) -> ArquivoRascunhos | None:
    """O arquivo, ou None se ainda não existe. Existir e estar inválido é erro (nunca se ignora em silêncio)."""
    p = Path(caminho)
    if not p.is_file():
        return None
    try:
        return ArquivoRascunhos.model_validate(json.loads(p.read_text(encoding="utf-8")))
    except (OSError, ValueError) as e:                 # pydantic.ValidationError e JSONDecodeError herdam de ValueError
        raise ArquivoError(f"{p} inválido: {e}") from e


def gravar(arq: ArquivoRascunhos, caminho: Path | str = ARQUIVO_PADRAO) -> Path:
    """Escrita atômica, UTF-8, LF, rascunhos em ordem de código."""
    p = Path(caminho)
    p.parent.mkdir(parents=True, exist_ok=True)
    corpo = arq.model_dump(mode="json")
    corpo["rascunhos"] = {k: corpo["rascunhos"][k] for k in sorted(corpo["rascunhos"])}
    texto = json.dumps(corpo, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=p.parent, prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(texto)
        os.chmod(tmp, 0o644)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return p


def abrir_ou_criar(caminho: Path | str, dataset_sha256: str, checklist_sha256: str, *,
                   recomecar: bool = False) -> ArquivoRascunhos:
    """O arquivo existente (se for do mesmo dataset e checklist) ou um vazio. Outro dataset/checklist exige `recomecar`."""
    novo = ArquivoRascunhos(dataset_sha256=dataset_sha256, checklist_sha256=checklist_sha256)
    atual = carregar(caminho)
    if atual is None or recomecar:
        return novo
    if (atual.dataset_sha256, atual.checklist_sha256) != (dataset_sha256, checklist_sha256):
        motivo = "dataset" if atual.dataset_sha256 != dataset_sha256 else "checklist"
        raise ArquivoError(f"o arquivo de rascunhos é de outro {motivo}; os rascunhos antigos não valem mais. "
                           f"Use --recomecar para descartá-los")
    return atual


# ------------------------------------------------------------------ operações (devolvem um arquivo novo)
def _trocar(arq: ArquivoRascunhos, r: rs.Rascunho) -> ArquivoRascunhos:
    return ArquivoRascunhos(dataset_sha256=arq.dataset_sha256, checklist_sha256=arq.checklist_sha256,
                            rascunhos={**arq.rascunhos, r.cod_ibge: r})


def registrar(arq: ArquivoRascunhos, r: rs.Rascunho, *, sobrescrever_revisado: bool = False) -> ArquivoRascunhos:
    antigo = arq.rascunhos.get(r.cod_ibge)
    if antigo is not None and antigo.status == "revisado" and not sobrescrever_revisado:
        raise ArquivoError(f"{r.nome} ({r.cod_ibge}) já tem rascunho revisado; use --sobrescrever para trocar")
    return _trocar(arq, r)


def _existente(arq: ArquivoRascunhos, cod: str) -> rs.Rascunho:
    r = arq.rascunhos.get(cod)
    if r is None:
        raise ArquivoError(f"não há rascunho para {cod}")
    return r


def _revisor(por: str) -> str:
    por = (por or "").strip()
    if not _REVISOR.match(por):
        raise ArquivoError("informe quem revisou (2 a 80 caracteres, sem < > nem quebra de linha)")
    return por


def aprovar(arq: ArquivoRascunhos, cod: str, por: str, *, agora: datetime, checklist: Checklist) -> ArquivoRascunhos:
    """Marca como revisado. Valida o texto de novo, porque a pessoa pode tê-lo editado à mão no JSON."""
    r = _existente(arq, cod)
    if r.status != "pendente_revisao":
        raise ArquivoError(f"{r.nome} ({cod}) está {r.status}; só se aprova rascunho pendente")
    motivos = rs.validar(r.resumo, [(i.id, i.porque) for i in r.itens], r.ficha, checklist)
    if motivos:
        raise ArquivoError("o texto não passa na validação: " + "; ".join(motivos))
    return _trocar(arq, rs.Rascunho.model_validate({**r.model_dump(), "status": "revisado", "revisado_por": _revisor(por),
                                                    "revisado_em_utc": _agora_iso(agora)}))


def rejeitar(arq: ArquivoRascunhos, cod: str, por: str, *, agora: datetime) -> ArquivoRascunhos:
    r = _existente(arq, cod)
    if r.status == "rejeitado":
        raise ArquivoError(f"{r.nome} ({cod}) já está rejeitado")
    return _trocar(arq, rs.Rascunho.model_validate({**r.model_dump(), "status": "rejeitado", "revisado_por": _revisor(por),
                                                    "revisado_em_utc": _agora_iso(agora)}))


# ------------------------------------------------------------------ o que vai ao site
def _item_do_pacote(r: rs.Rascunho, itens: dict) -> dict:
    return {"cod": r.cod_ibge, "nome": r.nome, "mes": r.mes_referencia, "escala": r.escala, "resumo": r.resumo,
            "revisado": r.status == "revisado", "revisado_por": r.revisado_por, "revisado_em": r.revisado_em_utc,
            "modelo": r.modelo,
            "itens": [{"titulo": itens[i.id].titulo, "descricao": itens[i.id].descricao,
                       "fonte": itens[i.id].fonte, "porque": i.porque} for i in r.itens]}


def para_pacote(arq: ArquivoRascunhos | None, checklist: Checklist | None, *, dataset_sha256: str,
                checklist_sha256: str | None, incluir_nao_revisados: bool = False) -> tuple[dict | None, list[str]]:
    """(pacote, avisos). Só leva rascunhos do dataset e do checklist atuais; None se não há nenhum para mostrar.

    Por padrão só os `revisado`. Com `incluir_nao_revisados`, entram também os `pendente_revisao` que passam de novo
    pela validação (o JSON pode ter sido editado à mão); cada item diz em `revisado` se uma pessoa o aprovou."""
    if arq is None:
        return None, []
    avisos: list[str] = []
    if arq.dataset_sha256 != dataset_sha256:
        avisos.append("os rascunhos de IA são de outro dataset e foram deixados de fora do site")
    if checklist is None or checklist_sha256 is None:
        avisos.append("há rascunhos de IA, mas a base do checklist não foi encontrada ou é inválida; eles ficaram fora do site")
    elif arq.checklist_sha256 != checklist_sha256:
        avisos.append("a base do checklist mudou depois dos rascunhos de IA; eles ficaram fora do site")
    if avisos:
        return None, avisos
    itens = checklist.por_id()
    incluidos = [r for r in arq.rascunhos.values() if r.status == "revisado"]
    pendentes = [r for r in arq.rascunhos.values() if r.status == "pendente_revisao"]
    if pendentes and not incluir_nao_revisados:
        avisos.append(f"{len(pendentes)} rascunho(s) de IA aguardam revisão e não aparecem no site")
    elif pendentes:
        reprovados = [r for r in pendentes
                      if rs.validar(r.resumo, [(i.id, i.porque) for i in r.itens], r.ficha, checklist)]
        bons = [r for r in pendentes if r not in reprovados]
        incluidos += bons
        if reprovados:
            avisos.append(f"{len(reprovados)} rascunho(s) pendente(s) não passam na validação e ficaram fora do site: "
                          + ", ".join(sorted(r.cod_ibge for r in reprovados)))
        if bons:
            avisos.append(f"{len(bons)} rascunho(s) de IA ainda não revisado(s) aparecem no site com o rótulo de não revisado")
    if not incluidos:
        return None, avisos
    lista = [_item_do_pacote(r, itens) for r in sorted(incluidos, key=lambda x: (x.nome, x.cod_ibge))]
    return {"rotulo": rs.ROTULO, "rotulo_nao_revisado": rs.ROTULO_NAO_REVISADO,
            "checklist": {"versao": checklist.versao, "elaborado_por": checklist.elaborado_por,
                          "atualizado_em": checklist.atualizado_em.isoformat()},
            "rascunhos": lista}, avisos
