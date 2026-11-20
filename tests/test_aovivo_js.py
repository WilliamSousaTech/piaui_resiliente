"""Painel ao vivo do Monitor (assets/aovivo.js): lógica pura rodada no Node, com a Open-Meteo e o localStorage simulados.

A tela (DOM, gráfico, tema) é testada em navegador de verdade em tests/test_aovivo_navegador.py."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.web import build_site as bs

JS = Path(bs.ASSETS) / "aovivo.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node não instalado")

PREAMBULO = r"""
const vm = require('vm'), fs = require('fs');
const sandbox = { window: {}, document: { getElementById: () => null, addEventListener() {} },
                  AbortController, setTimeout, clearTimeout, Date, JSON, Math, Number, Promise };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
const A = sandbox.window.PR_AOVIVO;
const DIA = 86400000;
const AGORA = Date.UTC(2026, 10, 20, 16, 52);                       // 20/11/2026, 13:52 em Fortaleza (UTC-3)
const iso = (ms) => new Date(ms).toISOString().slice(0, 10);

/* 46 dias: 30 passados (24 com 1 mm e 6 secos), hoje com 4,5 mm e 15 de previsão (12 com 2 mm e 3 secos) */
function resp(mexer) {
  const base = Date.UTC(2026, 10, 20), time = [], chuva = [], ht = [], solo = [];
  for (let k = -30; k <= 15; k++) {
    time.push(iso(base + k * DIA));
    chuva.push(k < 0 ? (k % 5 === 0 ? 0 : 1) : k === 0 ? 4.5 : (k % 5 === 0 ? 0 : 2));
    for (let hh = 0; hh < 24; hh++) {
      ht.push(iso(base + k * DIA) + 'T' + String(hh).padStart(2, '0') + ':00');
      solo.push(Math.round((0.30 - solo.length * 0.00005) * 1e5) / 1e5);
    }
  }
  const j = { utc_offset_seconds: -10800, daily: { time, precipitation_sum: chuva }, hourly: { time: ht, soil_moisture_0_to_1cm: solo } };
  if (mexer) mexer(j);
  return j;
}
function erro(f) { try { f(); return null; } catch (e) { return [e.motivo || null, String(e.message)]; } }
async function erroA(f) { try { await f(); return null; } catch (e) { return [e.motivo || null, String(e.message)]; } }
function armazem(limite = Infinity) {
  const m = new Map();
  return { m, get length() { return m.size; }, key: (i) => [...m.keys()][i] ?? null, getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem(k, v) { if (m.size >= limite && !m.has(k)) { const e = new Error('cota'); e.name = 'QuotaExceededError'; throw e; } m.set(k, String(v)); },
    removeItem: (k) => { m.delete(k); } };
}
const falso = (corpo, status = 200) => async () => ({ ok: status >= 200 && status < 300, status, json: async () => (typeof corpo === 'function' ? corpo() : corpo) });
"""


def rodar(cenario: str):
    r = subprocess.run(["node", "-e", PREAMBULO + "(async () => {" + cenario + "})().catch((e) => { console.error(e); process.exit(1); });", str(JS)],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_sintaxe_do_script():
    r = subprocess.run(["node", "--check", str(JS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ================================================================== URL
def test_url_tem_os_parametros_certos_e_4_casas():
    u = rodar("console.log(JSON.stringify(A.montarUrl(-5.123456, -42.8)));")
    assert u.startswith("https://api.open-meteo.com/v1/forecast?")
    q = dict(kv.split("=") for kv in u.split("?")[1].split("&"))
    assert q == {"latitude": "-5.1235", "longitude": "-42.8000", "daily": "precipitation_sum", "hourly": "soil_moisture_0_to_1cm",
                 "past_days": "30", "forecast_days": "16", "timezone": "America%2FFortaleza"}


@pytest.mark.parametrize("lat, lon", [("NaN", "-42"), ("-5", "undefined"), ("'-5'", "-42"), ("-42", "-5"), ("10", "-42"), ("-5", "-30"), ("null", "null")])
def test_url_recusa_coordenada_ruim_ou_trocada(lat, lon):
    s = rodar(f"console.log(JSON.stringify(erro(() => A.montarUrl({lat}, {lon}))));")
    assert s == ["formato", "coordenada fora do Piauí"]


# ================================================================== leitura da resposta
def test_resposta_normal_30_passados_hoje_e_15_de_previsao():
    s = rodar("""const r = A.lerResposta(resp(), AGORA), t = r.dias.map((d) => d.tipo), s = A.resumo(r);
                 console.log(JSON.stringify({ hoje: r.hoje, n: r.dias.length, ini: r.dias[0].data, fim: r.dias[45].data,
                   passado: t.filter((x) => x === 'passado').length, hoje_n: t.filter((x) => x === 'hoje').length,
                   prev: t.filter((x) => x === 'previsao').length, s }));""")
    assert (s["hoje"], s["n"], s["ini"], s["fim"]) == ("2026-11-20", 46, "2026-10-21", "2026-12-05")
    assert (s["passado"], s["hoje_n"], s["prev"]) == (30, 1, 15)
    assert s["s"]["passado"] == {"mm": 24, "validos": 30, "total": 30, "faltam": 0, "chuvosos": 24}
    assert s["s"]["futuro"] == {"mm": 28.5, "validos": 16, "total": 16, "faltam": 0, "chuvosos": 13}


def test_hoje_vira_pelo_relogio_local_e_nao_pelo_utc():
    """02:30 UTC de 21/11 ainda é 23:30 de 20/11 em Fortaleza: o dia 20 continua sendo 'hoje'."""
    s = rodar("const r = A.lerResposta(resp(), Date.UTC(2026, 10, 21, 2, 30)); console.log(JSON.stringify([r.hoje, r.dias.filter((d) => d.tipo === 'passado').length]));")
    assert s == ["2026-11-20", 30]
    s = rodar("const r = A.lerResposta(resp(), Date.UTC(2026, 10, 21, 3, 30)); console.log(JSON.stringify([r.hoje, r.dias.filter((d) => d.tipo === 'passado').length]));")
    assert s == ["2026-11-21", 31]


def test_dia_sem_dado_ou_absurdo_vira_null_e_nunca_zero():
    s = rodar("""const j = resp((x) => { const p = x.daily.precipitation_sum;
                   p[3] = null; p[4] = -1; p[5] = 501; p[6] = 'abc'; p[7] = NaN; p[8] = undefined; p[9] = Infinity; });
                 const r = A.lerResposta(j, AGORA);
                 console.log(JSON.stringify({ m: r.dias.slice(3, 10).map((d) => d.mm), borda: [r.dias[11].mm, A.lerResposta(resp((x) => { x.daily.precipitation_sum[0] = 500; }), AGORA).dias[0].mm],
                   s: A.resumo(r).passado }));""")
    assert s["m"] == [None] * 7
    assert s["borda"] == [1, 500]
    assert s["s"]["faltam"] == 7 and s["s"]["validos"] == 23 and s["s"]["total"] == 30      # não soma zero no lugar do buraco


@pytest.mark.parametrize("mexer, motivo_msg", [
    ("x.daily.time[5] = '2026-10-27'", "buraco"),                                 # repete/pula dia
    ("x.daily.time.reverse()", "buraco"),
    ("x.daily.time[2] = '2026-13-45'", "data inválida"),
    ("x.daily.time[2] = '20/11/2026'", "data inválida"),
    ("x.daily.precipitation_sum.pop()", "não batem"),
    ("x.daily.time.length = 5; x.daily.precipitation_sum.length = 5", "quantidade"),
    ("x.daily.precipitation_sum = x.daily.precipitation_sum.map(() => null)", "insuficientes"),
    ("x.daily.precipitation_sum = x.daily.precipitation_sum.map((v, i) => (i < 24 ? null : v))", "insuficientes"),
    ("delete x.daily", "sem dados diários"),
    ("x.daily = 'oi'", "sem dados diários"),
    ("x.error = true", "fora do formato"),
])
def test_resposta_fora_do_formato_e_recusada(mexer, motivo_msg):
    s = rodar(f"console.log(JSON.stringify(erro(() => A.lerResposta(resp((x) => {{ {mexer}; }}), AGORA))));")
    assert s is not None and s[0] == "formato" and motivo_msg in s[1]


@pytest.mark.parametrize("bruto", ["null", "[]", "'texto'", "42", "undefined"])
def test_resposta_que_nem_e_objeto_e_recusada(bruto):
    assert rodar(f"console.log(JSON.stringify(erro(() => A.lerResposta({bruto}, AGORA))));")[0] == "formato"


def test_umidade_do_solo_usa_a_ultima_hora_ate_agora_e_ignora_futuro_e_buracos():
    s = rodar("""const r = A.lerResposta(resp(), AGORA);
                 const k = resp((x) => { const i = x.hourly.time.indexOf('2026-11-20T13:00'); x.hourly.soil_moisture_0_to_1cm[i] = null;
                   x.hourly.soil_moisture_0_to_1cm[i - 1] = 7; x.hourly.soil_moisture_0_to_1cm[i - 2] = -0.1; });
                 console.log(JSON.stringify([r.solo, A.lerResposta(k, AGORA).solo]));""")
    assert s[0]["quando"] == "2026-11-20T13:00" and s[0]["valor"] == pytest.approx(0.26335)
    assert s[0]["max"] == pytest.approx(0.30) and s[0]["min"] == pytest.approx(0.26335) and s[0]["n"] == 30 * 24 + 14
    assert s[1]["quando"] == "2026-11-20T10:00"                # 13h vazia, 12h absurda (7) e 11h negativa: vale a última hora boa


def test_solo_ausente_ou_sem_valor_valido_vira_null_sem_derrubar_a_chuva():
    s = rodar("""const a = A.lerResposta(resp((x) => { delete x.hourly; }), AGORA);
                 const b = A.lerResposta(resp((x) => { x.hourly.soil_moisture_0_to_1cm = x.hourly.soil_moisture_0_to_1cm.map(() => null); }), AGORA);
                 const c = A.lerResposta(resp((x) => { x.hourly.time.pop(); }), AGORA);
                 console.log(JSON.stringify([a.solo, b.solo, c.solo, a.dias.length, b.dias.length]));""")
    assert s == [None, None, None, 46, 46]


def test_offset_da_resposta_e_usado_quando_vem():
    """Se a API devolver outro fuso, 'hoje' segue o fuso devolvido (aqui UTC+0: 16:52 UTC é dia 20)."""
    s = rodar("const r = A.lerResposta(resp((x) => { x.utc_offset_seconds = 0; }), Date.UTC(2026, 10, 21, 2, 30)); console.log(JSON.stringify(r.hoje));")
    assert s == "2026-11-21"


# ================================================================== escala do gráfico
@pytest.mark.parametrize("maximo, topo", [(0, 10), (3.2, 10), (10, 10), (10.1, 20), (45, 50), (51, 80), (120, 150), (500, 500), (9999, 500)])
def test_escala_do_grafico(maximo, topo):
    assert rodar(f"console.log(A.escalaTopo({maximo}));") == topo


# ================================================================== guardado no navegador
def test_guardado_ida_e_volta_mantem_dias_solo_e_datas():
    s = rodar("""const arm = armazem(), r = A.lerResposta(resp(), AGORA);
                 const ok = A.gravarCache(arm, '2200053', r, AGORA), l = A.lerCache(arm, '2200053', AGORA + 60000);
                 console.log(JSON.stringify({ ok, t: l.t, igual: JSON.stringify(l.res) === JSON.stringify(r), chaves: [...arm.m.keys()], tam: arm.m.get('pr-aovivo:2200053').length }));""")
    assert s["ok"] is True and s["t"] == 1795193520000 and s["igual"] is True
    assert s["chaves"] == ["pr-aovivo:2200053"] and s["tam"] < 3000                    # ~1,5 KB por município


@pytest.mark.parametrize("quando, esperado", [
    ("AGORA + 6 * DIA", True), ("AGORA + 7 * DIA + 1", False), ("AGORA - 600000", False)])      # vence em 7 dias; relógio atrás = "do futuro"
def test_guardado_vence_em_7_dias_e_nao_vale_se_vier_do_futuro(quando, esperado):
    s = rodar(f"""const arm = armazem(); A.gravarCache(arm, '2200053', A.lerResposta(resp(), AGORA), AGORA);
                  console.log(JSON.stringify(A.lerCache(arm, '2200053', {quando}) !== null));""")
    assert s is esperado


@pytest.mark.parametrize("estrago", [
    "x.v = 2", "x.t = 'ontem'", "x.hoje = '20/11'", "x.d = 'tudo'", "x.d[3] = ['2026-10-24']", "x.d[3][0] = 'lixo'",
    "x.d.length = 4", "x.off = 'a'", "x.s = { valor: 'x', min: 0, max: 1, n: 1, quando: '2026-11-20T13:00' }",
    "x.s = { valor: 0.2, min: 0, max: 1, n: 1, quando: 'agora' }", "x = null",
])
def test_guardado_adulterado_ou_quebrado_e_ignorado_e_removido(estrago):
    s = rodar(f"""const arm = armazem(); A.gravarCache(arm, '2200053', A.lerResposta(resp(), AGORA), AGORA);
                  let x = JSON.parse(arm.m.get('pr-aovivo:2200053')); {estrago}; arm.m.set('pr-aovivo:2200053', JSON.stringify(x));
                  console.log(JSON.stringify([A.lerCache(arm, '2200053', AGORA + 1000), arm.m.has('pr-aovivo:2200053')]));""")
    assert s == [None, False]


def test_guardado_com_json_invalido_e_ignorado():
    s = rodar("""const arm = armazem(); arm.m.set('pr-aovivo:2200053', '{nao e json');
                 console.log(JSON.stringify([A.lerCache(arm, '2200053', AGORA), arm.m.has('pr-aovivo:2200053')]));""")
    assert s == [None, False]


def test_codigo_estranho_nao_toca_no_armazenamento():
    s = rodar("""const arm = armazem(); const r = A.lerResposta(resp(), AGORA);
                 console.log(JSON.stringify([A.gravarCache(arm, '../x', r, AGORA), A.gravarCache(arm, '', r, AGORA), A.gravarCache(arm, '22000530000000', r, AGORA),
                   A.lerCache(arm, 'abc', AGORA), arm.m.size]));""")
    assert s == [False, False, False, None, 0]


def test_armazenamento_ausente_ou_bloqueado_nao_quebra():
    s = rodar("""const r = A.lerResposta(resp(), AGORA);
                 const bloqueado = { get length() { throw new Error('bloqueado'); }, key() { throw new Error('x'); },
                   getItem() { throw new Error('bloqueado'); }, setItem() { throw new Error('bloqueado'); }, removeItem() { throw new Error('bloqueado'); } };
                 console.log(JSON.stringify([A.gravarCache(null, '2200053', r, AGORA), A.lerCache(null, '2200053', AGORA),
                   A.gravarCache(bloqueado, '2200053', r, AGORA), A.lerCache(bloqueado, '2200053', AGORA)]));""")
    assert s == [False, None, False, None]


def test_guardado_nunca_passa_de_40_entradas_e_descarta_as_mais_velhas():
    s = rodar("""const arm = armazem(); const r = A.lerResposta(resp(), AGORA);
                 for (let i = 0; i < 55; i++) A.gravarCache(arm, String(2200000 + i), r, AGORA + i * 1000);
                 const ch = [...arm.m.keys()];
                 console.log(JSON.stringify({ n: ch.length, tem0: arm.m.has('pr-aovivo:2200000'), tem14: arm.m.has('pr-aovivo:2200014'),
                   tem15: arm.m.has('pr-aovivo:2200015'), tem54: arm.m.has('pr-aovivo:2200054') }));""")
    assert s == {"n": 40, "tem0": False, "tem14": False, "tem15": True, "tem54": True}


def test_limpeza_mexe_so_nas_chaves_do_painel():
    s = rodar("""const arm = armazem(); arm.m.set('pr-tema', 'dark'); arm.m.set('outra', 'x'); arm.m.set('pr-aovivo:2200001', 'lixo');
                 const r = A.lerResposta(resp(), AGORA);
                 for (let i = 0; i < 45; i++) A.gravarCache(arm, String(2200100 + i), r, AGORA + i);
                 console.log(JSON.stringify([arm.m.get('pr-tema'), arm.m.get('outra'), arm.m.has('pr-aovivo:2200001'), [...arm.m.keys()].filter((k) => k.startsWith('pr-aovivo:')).length]));""")
    assert s == ["dark", "x", False, 40]


def test_cota_cheia_limpa_pela_metade_e_tenta_de_novo():
    s = rodar("""const arm = armazem(30); const r = A.lerResposta(resp(), AGORA);
                 for (let i = 0; i < 30; i++) arm.m.set('pr-aovivo:' + (2200000 + i), JSON.stringify(A.serializar(r, AGORA - 100000 + i)));
                 const ok = A.gravarCache(arm, '2200999', r, AGORA);
                 console.log(JSON.stringify([ok, arm.m.has('pr-aovivo:2200999'), arm.m.size, arm.m.has('pr-aovivo:2200000'), arm.m.has('pr-aovivo:2200029')]));""")
    assert s == [True, True, 21, False, True]


def test_cota_impossivel_devolve_false_sem_estourar():
    s = rodar("""const arm = armazem(0); console.log(JSON.stringify(A.gravarCache(arm, '2200053', A.lerResposta(resp(), AGORA), AGORA)));""")
    assert s is False


# ================================================================== consulta
def test_consulta_ok_chama_a_url_certa_sem_cookies_nem_referrer():
    s = rodar("""let chamada = null;
                 const f = async (u, o) => { chamada = [u, o.credentials, o.referrerPolicy, !!o.signal]; return { ok: true, status: 200, json: async () => resp() }; };
                 const r = await A.consultar(-5.1, -42.8, { fetchFn: f, agora: () => AGORA });
                 console.log(JSON.stringify([chamada, r.dias.length, r.hoje]));""")
    assert s[0][0].startswith("https://api.open-meteo.com/v1/forecast?latitude=-5.1000&longitude=-42.8000")
    assert s[0][1:] == ["omit", "no-referrer", True] and s[1:] == [46, "2026-11-20"]


@pytest.mark.parametrize("fetch_fn, motivo", [
    ("falso({}, 429)", "limite"), ("falso({}, 500)", "http"), ("falso({}, 400)", "http"), ("falso({ error: true, reason: 'x' })", "formato"),
    ("falso(() => { throw new SyntaxError('json'); })", "formato"), ("async () => { throw new TypeError('Failed to fetch'); }", "rede"),
    ("falso({ daily: { time: [], precipitation_sum: [] } })", "formato"),
])
def test_consulta_traduz_cada_falha_em_um_motivo(fetch_fn, motivo):
    s = rodar(f"console.log(JSON.stringify(await erroA(() => A.consultar(-5.1, -42.8, {{ fetchFn: {fetch_fn}, agora: () => AGORA }}))));")
    assert s[0] == motivo


def test_consulta_sem_resposta_estoura_o_tempo():
    s = rodar("""const f = (u, o) => new Promise((_, rej) => o.signal.addEventListener('abort', () => rej(new Error('abortado'))));
                 const t0 = Date.now(); const e = await erroA(() => A.consultar(-5.1, -42.8, { fetchFn: f, tempoMs: 30, agora: () => AGORA }));
                 console.log(JSON.stringify([e, Date.now() - t0 < 2000]));""")
    assert s[0][0] == "tempo" and s[1] is True


def test_consulta_cancelada_por_fora_nao_vira_erro_de_rede():
    s = rodar("""const ctl = new AbortController();
                 const f = (u, o) => new Promise((_, rej) => o.signal.addEventListener('abort', () => rej(new Error('abortado'))));
                 const p = erroA(() => A.consultar(-5.1, -42.8, { fetchFn: f, sinal: ctl.signal, tempoMs: 5000 }));
                 setTimeout(() => ctl.abort(), 10); const e = await p;
                 const ja = new AbortController(); ja.abort();
                 const e2 = await erroA(() => A.consultar(-5.1, -42.8, { fetchFn: f, sinal: ja.signal, tempoMs: 5000 }));
                 console.log(JSON.stringify([e, e2]));""")
    assert s[0][0] == "cancelado" and s[1][0] == "cancelado"


def test_consulta_com_coordenada_ruim_nem_chama_a_rede():
    s = rodar("""let n = 0; const f = async () => { n++; return { ok: true, status: 200, json: async () => resp() }; };
                 const e = await erroA(() => A.consultar(40, -42.8, { fetchFn: f, agora: () => AGORA }));
                 console.log(JSON.stringify([e[0], n]));""")
    assert s == ["formato", 0]


def test_fonte_nao_usa_o_que_o_site_nao_precisa():
    """Só fala com a Open-Meteo, não escreve HTML a partir de texto de fora e não guarda nada além do prefixo do painel."""
    codigo = JS.read_text(encoding="utf-8")
    assert not re.search(r"innerHTML|outerHTML|insertAdjacentHTML|document\.write|eval\(|new Function|XMLHttpRequest|WebSocket|sendBeacon|importScripts|\bimport\(", codigo)
    urls = set(re.findall(r"https?://[^\s'\"`)]+", codigo))
    assert urls == {"https://api.open-meteo.com/v1/forecast"}
    assert re.findall(r"['\"]pr-[a-z]+", codigo) == ["'pr-aovivo"]          # não mexe em 'pr-tema' nem em outra chave do site
    assert re.findall(r"\bfetch\b", codigo) == ["fetch"]                      # uma única chamada: window.fetch
