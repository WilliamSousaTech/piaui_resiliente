"""Extração CHIRPS via Google Earth Engine — Item 2c.

`sonda`: LÊ o GEE e imprime fatos; não grava nada em data/raw.
`rodar`: extrai por município e mês e grava o RAW em lotes imutáveis (chirps_raw.gravar_lote).
Lote já gravado é só verificado (SHA-256), sem chamar o GEE: reexecutar retoma de onde parou.
`import ee` só neste arquivo (CA-09).

Indicadores (contrato 5.1, agregação por pixel e depois média municipal, grade nativa):
precip_mm (soma mensal), precip_max_dia_mm (máx. diário), dias_chuva (dias >= limiar),
dias_validos (D4: mínimo entre os pixels com dado), pixels_validos_frac (contagem).

Uso (PowerShell, com $env:GEE_PROJECT definido):
    python -m src.data.extract_chirps sonda
    python -m src.data.extract_chirps rodar --lote-teste   # só o 1º lote: mede tempo e cota
    python -m src.data.extract_chirps rodar                # janela completa (retoma se interrompida)
    python -m src.data.extract_chirps rodar --lotes 1:10   # só os lotes 1 a 10 (3 terminais = ~3x mais rápido)
Depois de todas as faixas, `rodar` sem argumentos só confere os lotes (SHA-256) e a janela inteira.
"""
from __future__ import annotations

import argparse
import os
import threading
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pandas as pd

from src.data import chirps_raw as raw
from src.data.config import Config, exigir_resolvido, load_config
from src.data.reference import ler_geojson_gee

MESES_SONDA = ("2020-03", "2020-09")  # hipótese: um mês úmido e um seco; a sonda mostra
LUIS_CORREIA = "2205706"              # litoral: testa máscara e pixels parciais
TENTATIVAS = 3                        # chamadas ao GEE: 1ª + 2 repetições, espera dobrando
ESPERA_S = 5.0
LIMITE_CHAMADA_S = 120.0              # por tentativa: uma consulta típica leva ~8 s; sem limite uma chamada presa trava tudo


class ExtracaoError(RuntimeError):
    """Falha de extração (GEE indisponível após repetições, escopo não suportado)."""


def _pontos(coords):
    if coords and isinstance(coords[0], (int, float)):
        yield coords
    else:
        for c in coords:
            yield from _pontos(c)


def area_bbox(geom: dict) -> float:
    """Área (graus²) da caixa envolvente: proxy barato de 'poucos pixels'."""
    p = list(_pontos(geom["coordinates"]))
    xs, ys = [a[0] for a in p], [a[1] for a in p]
    return (max(xs) - min(xs)) * (max(ys) - min(ys))


def escolher_amostra(geojson: dict, polo: str) -> dict[str, str]:
    """cod_ibge -> motivo: polo, menor caixa envolvente e (se no escopo) Luís Correia."""
    feats = geojson["features"]
    cods = {f["properties"]["cod_ibge"] for f in feats}
    menor = min(feats, key=lambda f: area_bbox(f["geometry"]))["properties"]["cod_ibge"]
    amostra: dict[str, str] = {}
    for cod, motivo in ((polo, "polo"), (menor, "menor bbox"), (LUIS_CORREIA, "litoral")):
        if cod in cods:
            amostra.setdefault(cod, motivo)
    return amostra


def intervalo_mes(mes: str) -> tuple[str, str]:
    """'2020-03' -> ('2020-03-01', '2020-04-01'); o fim é exclusivo no filterDate."""
    p = pd.Period(mes, freq="M")
    return p.start_time.strftime("%Y-%m-%d"), (p + 1).start_time.strftime("%Y-%m-%d")


def _carregar_ee(cfg: Config):
    var = cfg.gee.projeto_env
    proj = os.environ.get(var)
    if not proj:
        raise RuntimeError(f'Variável {var} não definida (PowerShell: $env:{var} = "<id do projeto>").')
    import ee  # único import do ee no projeto

    ee.Initialize(project=proj)
    return ee


def _colecao(ee, geojson: dict, cods) -> object:
    feats = [f for f in geojson["features"] if f["properties"]["cod_ibge"] in cods]
    return ee.FeatureCollection([
        ee.Feature(ee.Geometry(f["geometry"], None, False), {"cod_ibge": f["properties"]["cod_ibge"]})
        for f in feats])


def _imagem_mensal(ee, col, ini: str, fim: str, limiar: float):
    m = col.filterDate(ini, fim)
    return (m.sum().rename("soma")
            .addBands(m.max().rename("max_dia"))
            .addBands(m.min().rename("min_dia"))
            .addBands(m.map(lambda im: im.gte(limiar)).sum().rename("dias_chuva"))
            .addBands(m.count().selfMask().rename("dias_validos")))   # 0 vira máscara: min só entre pixels com dado


def _reduzir(img, fc, reducer, **kw) -> dict[str, dict]:
    r = img.reduceRegions(collection=fc, reducer=reducer, **kw).getInfo()
    return {f["properties"]["cod_ibge"]: f["properties"] for f in r["features"]}


def _p(*a) -> None:
    print("SONDA", *a, sep=" | ")


def sonda(cfg: Config, ee=None) -> None:
    ee = ee or _carregar_ee(cfg)
    ch = cfg.fontes.chirps
    geo = ler_geojson_gee(cfg)
    amostra = escolher_amostra(geo, cfg.escopo.piloto.polo_cod_ibge)
    fc = _colecao(ee, geo, set(amostra))
    ic = ee.ImageCollection(ch.gee_id)
    im0 = ic.first()
    proj = im0.select(ch.banda).projection()
    data = lambda ms: ee.Date(ms).format("YYYY-MM-dd").getInfo()  # noqa: E731

    _p("colecao", ch.gee_id, "imagens", ic.size().getInfo())
    _p("datas", data(ic.aggregate_min("system:time_start")),
       data(ic.aggregate_max("system:time_start")), "esperado até", cfg.janela.fim)
    _p("bandas", im0.bandNames().getInfo(), "esperada", ch.banda)
    _p("propriedades", im0.propertyNames().getInfo())
    _p("projecao", proj.crs().getInfo(), "nominalScale", proj.nominalScale().getInfo(),
       "params.escala_m", ch.escala_m)
    _p("amostra", amostra)

    col = ic.select(ch.banda)
    um = ee.Image.constant(1).rename("um").setDefaultProjection(proj)
    for mes in MESES_SONDA:
        ini, fim = intervalo_mes(mes)
        img = _imagem_mensal(ee, col, ini, fim, ch.limiar_dia_chuva_mm)
        nativa = img.setDefaultProjection(proj)
        t0 = time.perf_counter()
        a = _reduzir(nativa, fc, ee.Reducer.mean())                    # grade nativa
        b = _reduzir(img, fc, ee.Reducer.mean(), scale=ch.escala_m)    # escala nominal
        c = _reduzir(nativa.select("soma").addBands(um), fc, ee.Reducer.count())
        _p("mes", mes, "dias_no_mes", pd.Period(mes, freq="M").days_in_month,
           "segundos", round(time.perf_counter() - t0, 1))
        for cod, motivo in amostra.items():
            ma, mb, cc = a.get(cod, {}), b.get(cod, {}), c.get(cod, {})
            _p(cod, motivo, "nativa", {k: ma.get(k) for k in ("soma", "max_dia", "min_dia", "dias_chuva", "dias_validos")},
               "soma@escala_nominal", mb.get("soma"), "pixels_total", cc.get("um"), "pixels_validos", cc.get("soma"))


def _num(x, padrao: float = 0.0) -> float:
    return padrao if x is None else float(x)


def montar_linhas(mes: pd.Period, cods, media: dict, minimo: dict, contagem: dict) -> list[dict]:
    """Une as 3 reduções do GEE em linhas do RAW. Lógica pura (testável sem GEE).

    Sem pixel válido (ou propriedade nula): as 3 colunas de chuva viram NaN (nunca 0) e a
    cobertura vira 0; município ausente no retorno do GEE entra como sem dado (o validar
    do RAW decide se o lote é aceitável).
    """
    linhas = []
    for cod in cods:
        m, mi, c = media.get(cod, {}), minimo.get(cod, {}), contagem.get(cod, {})
        n_ok, n_tot = _num(c.get("soma")), _num(c.get("um"))
        frac = min(1.0, n_ok / n_tot) if n_tot > 0 else 0.0
        chaves = ("soma", "max_dia", "dias_chuva")
        sem = n_ok <= 0 or any(m.get(k) is None for k in chaves)
        nan = float("nan")
        linhas.append({
            "cod_ibge": cod, "mes": mes.to_timestamp(),
            "precip_mm": nan if sem else float(m["soma"]),
            "precip_max_dia_mm": nan if sem else float(m["max_dia"]),
            "dias_chuva": nan if sem else float(m["dias_chuva"]),
            "dias_validos": 0 if n_ok <= 0 else int(round(_num(mi.get("dias_validos")))),
            "dias_no_mes": mes.days_in_month, "pixels_validos_frac": frac})
    return linhas


def _chamar_com_limite(fn, limite_s):
    """Roda `fn` numa thread daemon e levanta TimeoutError (é um OSError, logo é repetido) se passar do limite.

    A chamada presa não pode ser cancelada; ela é abandonada e uma nova tentativa começa. A thread é daemon
    para não impedir o fim do programa."""
    if not limite_s:
        return fn()
    caixa: dict = {}

    def alvo():
        try:
            caixa["ok"] = fn()
        except BaseException as e:  # noqa: BLE001 - repassado intacto ao chamador
            caixa["erro"] = e

    t = threading.Thread(target=alvo, daemon=True)
    t.start()
    t.join(limite_s)
    if t.is_alive():
        raise TimeoutError(f"sem resposta do GEE em {limite_s:.0f}s")
    if "erro" in caixa:
        raise caixa["erro"]
    return caixa["ok"]


def _com_retry(fn, ee, dormir, limite_s=None):
    erros = (getattr(ee, "EEException", Exception), OSError)
    for i in range(TENTATIVAS):
        try:
            return _chamar_com_limite(fn, limite_s)
        except erros as e:
            if i == TENTATIVAS - 1:
                raise ExtracaoError(f"GEE falhou após {TENTATIVAS} tentativas: {e}") from e
            dormir(ESPERA_S * 2 ** i)


def _extrair_mes(ee, ctx, mes: pd.Period, fc, dormir) -> tuple[list[dict], int]:
    ini, fim = intervalo_mes(str(mes))
    img = _imagem_mensal(ee, ctx.col, ini, fim, ctx.limiar).setDefaultProjection(ctx.proj)
    tenta = lambda f: _com_retry(f, ee, dormir, getattr(ctx, "limite_s", None))  # noqa: E731
    media = tenta(lambda: _reduzir(img.select(["soma", "max_dia", "dias_chuva"]), ctx.fc, ee.Reducer.mean()))
    minimo = tenta(lambda: _reduzir(img.select(["dias_validos"]).addBands(ctx.um), ctx.fc, ee.Reducer.min()))
    contagem = tenta(lambda: _reduzir(img.select(["soma"]).addBands(ctx.um), ctx.fc, ee.Reducer.count()))
    n_img = tenta(lambda: ctx.col.filterDate(ini, fim).size().getInfo())
    return montar_linhas(mes, ctx.cods, media, minimo, contagem), int(n_img)


def rodar(cfg: Config, ee=None, *, so_primeiro_lote: bool = False, lotes: tuple[int, int] | None = None,
          limite_s: float | None = LIMITE_CHAMADA_S, dormir=time.sleep,
          relogio=time.perf_counter, saida=print) -> None:
    """`lotes` = (a, b), 1-based e inclusivo, sobre a lista completa de lotes da janela (fatia para vários terminais)."""
    if so_primeiro_lote and lotes is not None:
        raise ExtracaoError("use só um entre --lote-teste e --lotes")
    exigir_resolvido(cfg, "fontes.chirps", "janela", "gee")
    if cfg.escopo.modo != "semiarido":
        raise ExtracaoError("escopo.modo='piloto' exige data/reference/municipios_piloto.csv "
                            "(VERIFICAR); use modo 'semiarido'")
    geo = ler_geojson_gee(cfg)
    cods = sorted(f["properties"]["cod_ibge"] for f in geo["features"])
    meses = raw.meses_janela(cfg)
    todos = raw.dividir_em_lotes(meses, len(cods), cfg.gee.max_linhas_por_lote)
    if lotes is not None:
        a, b = lotes
        if not (1 <= a <= b <= len(todos)):
            raise ExtracaoError(f"--lotes {a}:{b} fora de 1:{len(todos)}")
        escolhidos = todos[a - 1:b]
    else:
        escolhidos = todos[:1] if so_primeiro_lote else todos
    parcial = so_primeiro_lote or lotes is not None
    dir_raw = raw.diretorio_raw(cfg)
    pendentes = []
    for ini, fim in escolhidos:
        if raw.lote_gravado(dir_raw, ini, fim):
            raw.verificar_integridade(raw.caminho_lote(dir_raw, ini, fim))
        else:
            pendentes.append((ini, fim))
    sel = [f"selecao={lotes[0]}:{lotes[1]}/{len(todos)}"] if lotes is not None else []
    saida("RODAR", f"municipios={len(cods)}", f"lotes={len(escolhidos)}", *sel,
          f"ja_gravados={len(escolhidos) - len(pendentes)}", f"a_extrair={len(pendentes)}", sep=" | ")
    if pendentes:
        ee = ee or _carregar_ee(cfg)
        ch = cfg.fontes.chirps
        col = ee.ImageCollection(ch.gee_id).select(ch.banda)
        proj = col.first().projection()
        ctx = SimpleNamespace(
            col=col, proj=proj, fc=_colecao(ee, geo, set(cods)), cods=cods,
            um=ee.Image.constant(1).rename("um").setDefaultProjection(proj),
            limiar=ch.limiar_dia_chuva_mm, limite_s=limite_s)
        for k, (ini, fim) in enumerate(pendentes, 1):
            t0 = relogio()
            partes, n_imgs = [], {}
            n_lote = todos.index((ini, fim)) + 1
            for mes in pd.period_range(ini, fim, freq="M"):
                t_mes = relogio()
                linhas, n_imgs[str(mes)] = _extrair_mes(ee, ctx, mes, ctx.fc, dormir)
                partes += linhas
                saida("RODAR", f"lote {n_lote}/{len(todos)}", f"mes {mes}", f"{relogio() - t_mes:.0f}s", sep=" | ")
            meta = {"gee_id": ch.gee_id, "banda": ch.banda, "limiar_dia_chuva_mm": ch.limiar_dia_chuva_mm,
                    "ee_version": getattr(ee, "__version__", "desconhecida"),
                    "extraido_em_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "n_imagens_por_mes": n_imgs}
            df = pd.DataFrame(partes, columns=raw.COLUNAS)
            res = raw.gravar_lote(df, dir_raw, cods, ini, fim, meta=meta)
            seg = relogio() - t0
            avisos = [m for m, n in n_imgs.items() if n != pd.Period(m).days_in_month]
            saida("RODAR", f"lote {n_lote}/{len(todos)} ({k}/{len(pendentes)} desta execução)", f"{ini}..{fim}",
                  res.status, f"linhas={res.n_linhas}", f"precip_NaN={res.n_precip_nan}", f"frac_min={df['pixels_validos_frac'].min():.3f}",
                  f"dias_validos_min={int(df['dias_validos'].min())}", f"meses_com_lacuna={avisos}",
                  f"seg={seg:.0f}", f"restam≈{seg * (len(pendentes) - k) / 60:.0f}min", sep=" | ")
    if not parcial:
        todo = raw.ler_raw(dir_raw, cods, meses)
        saida("RODAR", "RAW completo e íntegro", f"linhas={len(todo)}",
              f"precip_NaN={int(todo['precip_mm'].isna().sum())}", sep=" | ")


def _faixa_lotes(texto: str, ap) -> tuple[int, int]:
    try:
        a, b = (int(x) for x in texto.split(":"))
    except ValueError:
        ap.error(f"--lotes precisa ser A:B com inteiros (ex.: 1:10), recebi {texto!r}")
    if not 1 <= a <= b:
        ap.error(f"--lotes {texto}: exige 1 <= A <= B")
    return a, b


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m src.data.extract_chirps")
    ap.add_argument("comando", choices=["sonda", "rodar"])
    ap.add_argument("--lotes", metavar="A:B", help="só os lotes A a B (1-based, inclusivo); permite rodar em vários terminais")
    ap.add_argument("--lote-teste", action="store_true",
                    help="rodar apenas o primeiro lote (mede tempo e cota antes da janela completa)")
    a = ap.parse_args(argv)
    if a.lote_teste and a.comando != "rodar":
        ap.error("--lote-teste só vale para o comando rodar")
    if a.lotes and a.comando != "rodar":
        ap.error("--lotes só vale para o comando rodar")
    if a.lotes and a.lote_teste:
        ap.error("use só um entre --lote-teste e --lotes")
    faixa = _faixa_lotes(a.lotes, ap) if a.lotes else None
    cfg = load_config()
    sonda(cfg) if a.comando == "sonda" else rodar(cfg, so_primeiro_lote=a.lote_teste, lotes=faixa)


if __name__ == "__main__":
    main()
