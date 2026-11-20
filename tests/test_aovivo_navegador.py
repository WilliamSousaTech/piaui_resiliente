"""Painel ao vivo do Monitor em navegador de verdade (Chromium via Playwright), com a Open-Meteo SIMULADA.

Nenhuma chamada sai de verdade: toda requisição para api.open-meteo.com é interceptada e respondida pelo teste. Cobre o
caminho feliz, as falhas (sem rede, limite de uso, resposta ilegível), o guardado no navegador, a troca rápida de município,
os dois temas e o celular. Pula sozinho se o Playwright ou um Chrome/Chromium não estiverem instalados.
Defina PR_TELAS=<pasta> para salvar capturas de tela desta bateria."""
import dataclasses
import json
import os
import re
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from src.web import build_site as bs

playwright_sync = pytest.importorskip("playwright.sync_api")
FUSO = ZoneInfo("America/Fortaleza")
TELAS = os.environ.get("PR_TELAS")


# ------------------------------------------------------------------ Open-Meteo de mentira (datas relativas a HOJE, sem depender do dia da execução)
def resposta(mexer=None):
    hoje = datetime.now(FUSO).date()
    dias = [hoje + timedelta(days=k) for k in range(-30, 16)]
    chuva = [(0.0 if k % 4 == 0 else round(1.5 + (k % 7), 1)) if k <= 0 else (0.0 if k % 3 == 0 else float(k % 5 + 1)) for k in range(-30, 16)]
    ht, solo = [], []
    for d in dias:
        for hh in range(24):
            ht.append(f"{d.isoformat()}T{hh:02d}:00")
            solo.append(round(0.31 - len(solo) * 0.00006, 5))
    j = {"latitude": -5.1, "longitude": -42.8, "utc_offset_seconds": -10800, "timezone": "America/Fortaleza",
         "daily": {"time": [d.isoformat() for d in dias], "precipitation_sum": chuva},
         "hourly": {"time": ht, "soil_moisture_0_to_1cm": solo}}
    if mexer:
        mexer(j)
    return j


class Api:
    """Interceptador: registra as chamadas e responde conforme `modo` (ok | rede | limite | lixo | lento)."""

    def __init__(self, modo="ok"):
        self.modo, self.chamadas = modo, []

    def __call__(self, route):
        self.chamadas.append(route.request.url)
        cors = {"access-control-allow-origin": "*"}
        if self.modo == "rede":
            route.abort("connectionfailed")
        elif self.modo == "limite":
            route.fulfill(status=429, headers=cors, content_type="application/json", body='{"error":true,"reason":"Too many requests"}')
        elif self.modo == "lixo":
            route.fulfill(status=200, headers=cors, content_type="application/json", body='{"foo": 1}')
        else:
            route.fulfill(status=200, headers=cors, content_type="application/json", body=json.dumps(resposta()))

    @property
    def parametros(self):
        return [dict(kv.split("=", 1) for kv in u.split("?", 1)[1].split("&")) for u in self.chamadas]


# ------------------------------------------------------------------ site e navegador
@pytest.fixture(scope="module")
def pasta(payload, tmp_path_factory):
    raiz = tmp_path_factory.mktemp("site_aovivo")
    for caminho, conteudo in bs.montar_arquivos(payload).items():
        f = raiz / caminho
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(conteudo if isinstance(conteudo, bytes) else conteudo.encode("utf-8"))
    return raiz


@pytest.fixture(scope="module")
def navegador():
    with playwright_sync.sync_playwright() as p:
        candidatos = [os.environ.get("CHROME_PATH"), shutil.which("google-chrome"), shutil.which("chromium"), shutil.which("chrome"), None]
        ultimo = None
        for exe in candidatos:
            try:
                b = p.chromium.launch(executable_path=exe, args=["--no-sandbox"]) if exe else p.chromium.launch(args=["--no-sandbox"])
                break
            except Exception as e:                                     # noqa: BLE001 - tenta o próximo navegador
                ultimo = e
        else:
            pytest.skip(f"nenhum Chrome/Chromium disponível: {str(ultimo)[:120]}")
        yield b
        b.close()


class Sessao:
    def __init__(self, navegador, pasta, api, *, tema="light", largura=1280, altura=900, antes=None):
        self.ctx = navegador.new_context(viewport={"width": largura, "height": altura})
        self.ctx.add_init_script(f"try{{localStorage.getItem('pr-tema')||localStorage.setItem('pr-tema','{tema}')}}catch(e){{}}")
        self.api, self.pagina_url = api, (pasta / "monitor.html").as_uri()
        self.ctx.route("https://api.open-meteo.com/**", api)
        self.pedidos, self.erros = [], []
        self.ctx.on("request", lambda r: self.pedidos.append(r.url))
        self.pg = self.ctx.new_page()
        self.pg.on("pageerror", lambda e: self.erros.append(str(e)))
        self.pg.on("console", lambda m: self.erros.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text else None)
        if antes:
            antes(self.ctx)

    def abrir(self, hash_=""):
        self.pg.goto(self.pagina_url + hash_)
        return self

    def esperar_conteudo(self):
        self.pg.wait_for_selector("#aoConteudo:not([hidden])", timeout=8000)

    def foto(self, nome):
        if TELAS:
            Path(TELAS).mkdir(parents=True, exist_ok=True)
            self.pg.locator("#aoVivo").scroll_into_view_if_needed()
            self.pg.locator("#colMun").screenshot(path=str(Path(TELAS) / f"{nome}.png"))

    def fechar(self):
        self.ctx.close()


@pytest.fixture
def abrir(navegador, pasta):
    abertas = []

    def f(modo="ok", **kw):
        s = Sessao(navegador, pasta, Api(modo), **kw)
        abertas.append(s)
        return s
    yield f
    for s in abertas:
        s.fechar()


def texto(s, seletor):
    return s.pg.locator(seletor).inner_text()


# ================================================================== caminho feliz
def test_painel_aparece_com_numeros_grafico_tabela_e_atribuicao(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    assert "Consultado hoje às" in texto(s, "#aoStatus")
    st = texto(s, "#aoStats")
    assert "Últimos 30 dias" in st and "Próximos 16 dias" in st and "Umidade do solo (0 a 1 cm)" in st and "m³/m³" in st
    marcas = s.pg.evaluate("document.querySelectorAll('#aoGraf .b-chuva, #aoGraf .t-seco, #aoGraf .b-falta').length")
    assert marcas == 46                                                         # um sinal por dia: barra de chuva, traço de dia seco ou "sem dado"
    esperado_prev = sum(1 for mm in resposta()["daily"]["precipitation_sum"][30:] if mm > 0)          # hoje e os 15 dias seguintes, só os com chuva
    assert s.pg.locator("#aoGraf .b-prev").count() == esperado_prev
    assert s.pg.locator("#aoGraf .t-seco").count() == sum(1 for mm in resposta()["daily"]["precipitation_sum"] if mm == 0)
    assert s.pg.locator("#aoGraf .l-hoje").count() == 1 and "hoje" in s.pg.locator("#aoGraf svg").text_content()
    assert s.pg.locator("#aoCorpo tr").count() == 46
    assert s.pg.locator("#aoResumoTab").inner_text() == "Ver os 46 dias em tabela"
    assert s.pg.locator("#aoVivo .proveniencia a").get_attribute("href") == "https://open-meteo.com/"
    assert "estimativa" in texto(s, "#aoVivo .ao-nota").lower() and "não altera o SPI" in texto(s, "#aoVivo .ao-nota")
    assert s.erros == []


def test_consulta_usa_a_coordenada_do_municipio_aberto_e_so_fala_com_a_open_meteo(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    cod = s.pg.evaluate("window.PR_DADOS.municipios[window.__painel.S.sel].cod")
    lat, lon = s.pg.evaluate("(() => { const m = window.PR_DADOS.municipios[window.__painel.S.sel]; return [m.lat, m.lon]; })()")
    assert len(s.api.chamadas) == 1
    p = s.api.parametros[0]
    assert float(p["latitude"]) == pytest.approx(lat, abs=5e-5) and float(p["longitude"]) == pytest.approx(lon, abs=5e-5)
    assert p["past_days"] == "30" and p["forecast_days"] == "16" and p["daily"] == "precipitation_sum" and p["hourly"] == "soil_moisture_0_to_1cm"
    externos = [u for u in s.pedidos if not u.startswith(("file:", "data:", "blob:", "about:"))]
    assert externos and all(u.startswith("https://api.open-meteo.com/v1/forecast?") for u in externos), externos
    assert s.pg.evaluate(f"localStorage.getItem('pr-aovivo:{cod}') !== null") is True


def test_dica_ao_passar_o_mouse_mostra_dia_e_milimetros(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    caixa = s.pg.locator("#aoGraf svg").bounding_box()
    s.pg.mouse.move(caixa["x"] + caixa["width"] * 0.5, caixa["y"] + caixa["height"] * 0.5)
    s.pg.wait_for_selector("#tip.on", timeout=3000)
    t = texto(s, "#tip")
    assert re.search(r"\d{2}/\d{2}", t) and ("mm" in t or "Sem dado" in t)
    s.pg.mouse.move(2, 2)
    assert s.pg.locator("#tip.on").count() == 0


def test_o_resto_do_monitor_segue_igual_com_o_painel(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    assert s.pg.locator("#munFicha").is_visible() and texto(s, "#munSpi") != ""
    assert s.pg.locator("#grafSpi svg").count() >= 1 and s.pg.locator("#munStats").is_visible()
    assert s.pg.evaluate("window.PR_DADOS.verificacao.dataset_sha256") == s.pg.evaluate("window.__painel.D.verificacao.dataset_sha256")
    assert s.pg.locator("#aoVivo").evaluate("e => e.closest('#munFicha') !== null")          # o painel mora dentro da ficha


# ================================================================== falhas
def test_sem_rede_e_sem_guardado_mostra_aviso_e_botao_e_o_resto_funciona(abrir):
    s = abrir("rede").abrir()
    s.pg.wait_for_selector("#aoRetry:not([hidden])", timeout=8000)
    assert "Sem conexão com a Open-Meteo agora." in texto(s, "#aoStatus") and "série verificada" in texto(s, "#aoStatus")
    assert s.pg.locator("#aoConteudo").is_hidden() and s.pg.locator("#aoGraf svg").count() == 0
    assert s.pg.locator("#munFicha").is_visible() and texto(s, "#munSpi") != ""
    assert s.erros == []                                                          # falha de rede é esperada: não pode virar erro de script


def test_tentar_de_novo_funciona_quando_a_rede_volta(abrir):
    s = abrir("rede").abrir()
    s.pg.wait_for_selector("#aoRetry:not([hidden])", timeout=8000)
    s.api.modo = "ok"
    s.pg.click("#aoRetry")
    s.esperar_conteudo()
    assert s.pg.locator("#aoRetry").is_hidden() and "Consultado hoje às" in texto(s, "#aoStatus")
    assert len(s.api.chamadas) == 2


def test_limite_de_uso_da_open_meteo_tem_mensagem_propria(abrir):
    s = abrir("limite").abrir()
    s.pg.wait_for_selector("#aoRetry:not([hidden])", timeout=8000)
    assert "pediu para esperar" in texto(s, "#aoStatus")


def test_resposta_ilegivel_nao_desenha_nada(abrir):
    s = abrir("lixo").abrir()
    s.pg.wait_for_selector("#aoRetry:not([hidden])", timeout=8000)
    assert "formato que não consegui ler" in texto(s, "#aoStatus")
    assert s.pg.locator("#aoGraf svg").count() == 0 and s.pg.locator("#aoStats").inner_text() == ""
    assert s.erros == []


# ================================================================== guardado no navegador
def test_segunda_visita_em_30_minutos_nao_chama_a_rede(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    assert len(s.api.chamadas) == 1
    s.pg.reload()
    s.esperar_conteudo()
    assert len(s.api.chamadas) == 1                                               # veio do guardado
    assert "guardada neste navegador" in texto(s, "#aoStatus")


def test_sem_rede_com_guardado_antigo_mostra_o_guardado_marcado(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    cod = s.pg.evaluate("window.PR_DADOS.municipios[window.__painel.S.sel].cod")
    s.pg.evaluate(f"""(() => {{ const k = 'pr-aovivo:{cod}', o = JSON.parse(localStorage.getItem(k)); o.t -= 3 * 3600 * 1000; localStorage.setItem(k, JSON.stringify(o)); }})()""")
    s.api.modo = "rede"
    s.pg.reload()
    s.esperar_conteudo()
    st = texto(s, "#aoStatus")
    assert "Sem conexão com a Open-Meteo agora." in st and "Mostrando a última consulta" in st and "desatualizada" in st
    assert "velho" in s.pg.locator("#aoStatus").get_attribute("class") and s.pg.locator("#aoRetry").is_hidden()
    assert s.pg.locator("#aoGraf svg").count() == 1 and s.pg.locator("#aoCorpo tr").count() == 46
    assert len(s.api.chamadas) == 2                                               # tentou de novo (guardado velho) e caiu no guardado


def test_guardado_corrompido_ou_adulterado_e_ignorado_sem_quebrar(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    cod = s.pg.evaluate("window.PR_DADOS.municipios[window.__painel.S.sel].cod")
    s.pg.evaluate(f"localStorage.setItem('pr-aovivo:{cod}', '<img src=x onerror=alert(1)>{{')")
    s.api.modo = "rede"
    s.pg.reload()
    s.pg.wait_for_selector("#aoRetry:not([hidden])", timeout=8000)
    assert s.pg.evaluate(f"localStorage.getItem('pr-aovivo:{cod}')") is None      # descartado
    assert s.erros == []


def test_armazenamento_bloqueado_nao_derruba_o_painel(abrir):
    def bloquear(ctx):
        ctx.add_init_script("Object.defineProperty(window, 'localStorage', { get() { throw new DOMException('bloqueado', 'SecurityError'); } });")
    s = abrir("ok", antes=bloquear).abrir()
    s.esperar_conteudo()
    assert "Consultado hoje às" in texto(s, "#aoStatus")


# ================================================================== troca de município
def test_cliques_seguidos_viram_uma_consulta_so_e_vale_o_ultimo(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    s.api.chamadas.clear()
    nomes = s.pg.evaluate("""(() => { const P = window.__painel; const n = P.D.municipios.length;
        const a = (P.S.sel + 1) % n, b = (P.S.sel + 2) % n; P.selecionar(a); P.selecionar(b); return [P.D.municipios[b].nome, P.D.municipios[b].lat]; })()""")
    s.pg.wait_for_function("document.querySelector('#aoGraf svg') && document.querySelector('#aoGraf svg').getAttribute('aria-label').includes(%s)" % json.dumps(nomes[0]),
                           timeout=8000)
    assert len(s.api.chamadas) == 1
    assert float(s.api.parametros[0]["latitude"]) == pytest.approx(nomes[1], abs=5e-5)


def test_resposta_atrasada_de_um_municipio_nao_pinta_o_outro(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    s.pg.evaluate("localStorage.clear()")
    s.api.chamadas.clear()
    atrasadas = []

    def lenta(route):                                                             # segura a primeira resposta até o teste liberar
        atrasadas.append(route)

    s.ctx.unroute("https://api.open-meteo.com/**")
    s.ctx.route("https://api.open-meteo.com/**", lenta)
    nome_a = s.pg.evaluate("""(() => { const P = window.__painel, a = (P.S.sel + 1) % P.D.municipios.length; P.selecionar(a); return P.D.municipios[a].nome; })()""")
    s.pg.wait_for_timeout(450)                                                    # passa do atraso de 250 ms: a consulta de A sai
    assert len(atrasadas) == 1
    nome_b = s.pg.evaluate("""(() => { const P = window.__painel, b = (P.S.sel + 1) % P.D.municipios.length; P.selecionar(b); return P.D.municipios[b].nome; })()""")
    assert nome_a != nome_b
    s.pg.wait_for_timeout(450)
    for r in atrasadas:                                                           # só agora as respostas chegam, na ordem inversa
        try:
            r.fulfill(status=200, headers={"access-control-allow-origin": "*"}, content_type="application/json", body=json.dumps(resposta()))
        except Exception:                                                         # noqa: BLE001 - a consulta de A foi cancelada pelo navegador
            pass
    s.pg.wait_for_timeout(500)
    rotulo = s.pg.evaluate("(document.querySelector('#aoGraf svg') || {getAttribute: () => ''}).getAttribute('aria-label')")
    assert nome_a not in (rotulo or "")


def test_municipio_sem_coordenada_esconde_o_painel(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    s.pg.evaluate("window.PR_DADOS.municipios[window.__painel.S.sel === 0 ? 1 : 0].lat = null; "
                  "window.__painel.selecionar(window.__painel.S.sel === 0 ? 1 : 0)")
    s.pg.wait_for_selector("#aoVivo", state="hidden", timeout=3000)


# ================================================================== aparência
@pytest.mark.parametrize("tema, seca, chuva", [("light", "rgb(180, 71, 42)", "rgb(15, 110, 122)"), ("dark", "rgb(224, 115, 79)", "rgb(79, 179, 191)")])
def test_cores_seguem_o_tema_terracota_para_seco_e_petroleo_para_chuva(abrir, tema, seca, chuva):
    s = abrir("ok", tema=tema).abrir()
    s.esperar_conteudo()
    assert s.pg.evaluate("document.documentElement.dataset.theme") == tema or tema == "light"
    cor = lambda sel, prop: s.pg.evaluate(f"getComputedStyle(document.querySelector('{sel}')).{prop}")       # noqa: E731
    assert cor("#aoGraf .t-seco", "fill") == seca and cor("#aoGraf .b-chuva", "fill") == chuva
    assert cor("#aoGraf .b-prev", "fillOpacity") == "0.45"
    assert cor(".ao-leg .sw-seco", "backgroundColor") == seca
    s.foto(f"painel_{tema}")


def test_alternar_o_tema_nao_exige_nova_consulta(abrir):
    s = abrir("ok", tema="light").abrir()
    s.esperar_conteudo()
    antes = len(s.api.chamadas)
    s.pg.click("#btnTema")
    s.pg.wait_for_function("document.documentElement.dataset.theme === 'dark'")
    assert len(s.api.chamadas) == antes
    assert s.pg.evaluate("getComputedStyle(document.querySelector('#aoGraf .t-seco')).fill") == "rgb(224, 115, 79)"


def test_celular_sem_rolagem_lateral_causada_pelo_painel(abrir):
    s = abrir("ok", largura=390, altura=800).abrir()
    s.esperar_conteudo()
    medidas = s.pg.evaluate("""(() => { const v = document.querySelector('#aoVivo').getBoundingClientRect(), g = document.querySelector('#aoGraf svg').getBoundingClientRect();
        return { vivoDir: v.right, svgDir: g.right, janela: window.innerWidth }; })()""")
    assert medidas["vivoDir"] <= medidas["janela"] + 1 and medidas["svgDir"] <= medidas["janela"] + 1, medidas
    s.foto("painel_celular")


def test_tabela_abre_e_fica_acessivel_por_teclado(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    s.pg.click("#aoResumoTab")
    regiao = s.pg.locator(".ao-tab-corpo")
    assert regiao.is_visible() and regiao.get_attribute("tabindex") == "0" and regiao.get_attribute("role") == "region"
    assert "sem dado" not in texto(s, "#aoCorpo")                                  # mock completo: nenhum buraco
    primeira = s.pg.locator("#aoCorpo tr").first.inner_text()
    assert re.search(r"[a-zç]{3} \d{2}/\d{2}\s+[\d,]+\s+estimado pelo modelo", primeira)


def test_dia_sem_dado_aparece_como_buraco_nunca_como_zero(abrir):
    s = abrir("ok").abrir()
    s.esperar_conteudo()
    s.pg.evaluate("localStorage.clear()")

    def com_buraco(route):
        route.fulfill(status=200, headers={"access-control-allow-origin": "*"}, content_type="application/json",
                      body=json.dumps(resposta(lambda j: j["daily"]["precipitation_sum"].__setitem__(slice(5, 8), [None, None, None]))))
    s.ctx.unroute("https://api.open-meteo.com/**")
    s.ctx.route("https://api.open-meteo.com/**", com_buraco)
    s.pg.evaluate("window.__painel.selecionar((window.__painel.S.sel + 1) % window.PR_DADOS.municipios.length)")
    s.pg.wait_for_function("document.querySelectorAll('#aoGraf .b-falta').length === 3", timeout=8000)
    assert s.pg.locator("#aoCorpo tr:has-text('sem dado')").count() == 3
    assert "faltam 3 de 30 dias" in texto(s, "#aoStats")
    s.foto("painel_com_buraco")
