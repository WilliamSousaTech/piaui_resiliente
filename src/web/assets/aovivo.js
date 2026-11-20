/* Painel "ao vivo" do Monitor: chuva dos últimos 30 dias e dos próximos 16 (modelo da Open-Meteo) e umidade do solo.
   É uma camada à parte da série verificada: não toca no SPI, no pacote de dados nem no hash de integridade.
   Se a rede falhar, mostra a última consulta guardada neste navegador (marcada como guardada) ou um aviso com
   "Tentar de novo". Dia sem dado nunca vira zero. Único arquivo do site que fala com a rede. */
(() => {
'use strict';
const URL_API = 'https://api.open-meteo.com/v1/forecast';
const FUSO = 'America/Fortaleza';
const PASSADO = 30, FUTURO = 16;
const TTL_FRESCO_MS = 30 * 60 * 1000;               // consulta mais nova que isso é reaproveitada, sem nova chamada
const VALIDADE_CACHE_MS = 7 * 24 * 3600 * 1000;     // guardado há mais que isso não é mostrado
const LIMITE_ENTRADAS = 40;
const TEMPO_MAX_MS = 10000;
const MAX_DIA_MM = 500;
const PREFIXO = 'pr-aovivo:';
const BOX = { lat: [-11.0, -2.5], lon: [-46.2, -40.2] };    // caixa do Piauí (a mesma do build)
const RE_DATA = /^\d{4}-\d{2}-\d{2}$/;
const RE_HORA = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/;
const RE_COD = /^\d{6,8}$/;
const UM_DIA = 86400000;

class ErroAoVivo extends Error {
  constructor(msg, motivo) { super(msg); this.motivo = motivo; }       // rede | tempo | limite | http | formato | cancelado
}

const ehNum = (x) => typeof x === 'number' && Number.isFinite(x);
const diaMs = (s) => Date.parse(s + 'T00:00:00Z');

/* ---------------------------------------------------------------- lógica pura (testada no Node) */
function montarUrl(lat, lon) {
  if (!ehNum(lat) || !ehNum(lon) || lat < BOX.lat[0] || lat > BOX.lat[1] || lon < BOX.lon[0] || lon > BOX.lon[1]) {
    throw new ErroAoVivo('coordenada fora do Piauí', 'formato');
  }
  const q = [['latitude', lat.toFixed(4)], ['longitude', lon.toFixed(4)], ['daily', 'precipitation_sum'],
    ['hourly', 'soil_moisture_0_to_1cm'], ['past_days', PASSADO], ['forecast_days', FUTURO], ['timezone', FUSO]];
  return URL_API + '?' + q.map(([a, b]) => a + '=' + encodeURIComponent(b)).join('&');
}

/* 'AAAA-MM-DDTHH:MM' na hora local do ponto (a Open-Meteo devolve a diferença para UTC em utc_offset_seconds) */
function agoraLocal(agoraMs, offsetS) {
  if (!ehNum(agoraMs)) throw new ErroAoVivo('relógio inválido', 'formato');
  return new Date(agoraMs + offsetS * 1000).toISOString().slice(0, 16);
}

function construirDias(datas, mms, hoje) {
  if (!Array.isArray(datas) || !Array.isArray(mms) || datas.length !== mms.length) throw new ErroAoVivo('dias e chuva não batem', 'formato');
  const n = datas.length;
  if (n < 10 || n > 120) throw new ErroAoVivo('quantidade de dias fora do esperado', 'formato');
  const dias = [];
  let anterior = null;
  for (let i = 0; i < n; i++) {
    const data = datas[i];
    if (typeof data !== 'string' || !RE_DATA.test(data) || !ehNum(diaMs(data))) throw new ErroAoVivo('data inválida', 'formato');
    if (anterior !== null && diaMs(data) - diaMs(anterior) !== UM_DIA) throw new ErroAoVivo('dias fora de ordem ou com buraco', 'formato');
    anterior = data;
    const v = mms[i];
    const mm = ehNum(v) && v >= 0 && v <= MAX_DIA_MM ? v : null;                    // faltando ou absurdo: sem dado, nunca zero
    dias.push({ data, mm, tipo: data < hoje ? 'passado' : data === hoje ? 'hoje' : 'previsao' });
  }
  if (dias.filter((d) => d.mm !== null).length < n / 2) throw new ErroAoVivo('dados insuficientes', 'formato');
  return dias;
}

function extrairSolo(hourly, agoraH) {
  if (!hourly || !Array.isArray(hourly.time) || !Array.isArray(hourly.soil_moisture_0_to_1cm)
      || hourly.time.length !== hourly.soil_moisture_0_to_1cm.length) return null;
  let ult = null, mn = Infinity, mx = -Infinity, n = 0;
  for (let i = 0; i < hourly.time.length; i++) {
    const t = hourly.time[i], v = hourly.soil_moisture_0_to_1cm[i];
    if (typeof t !== 'string' || !RE_HORA.test(t) || t.slice(0, 13) > agoraH) continue;     // hora futura não é "agora"
    if (!ehNum(v) || v < 0 || v > 1) continue;
    if (ult === null || t >= ult.quando) ult = { valor: v, quando: t };
    mn = Math.min(mn, v); mx = Math.max(mx, v); n++;
  }
  return ult ? { valor: ult.valor, quando: ult.quando, min: mn, max: mx, n } : null;
}

function lerResposta(j, agoraMs) {
  if (!j || typeof j !== 'object' || Array.isArray(j) || j.error) throw new ErroAoVivo('resposta fora do formato', 'formato');
  const off = ehNum(j.utc_offset_seconds) ? j.utc_offset_seconds : -10800;
  const d = j.daily;
  if (!d || typeof d !== 'object') throw new ErroAoVivo('resposta sem dados diários', 'formato');
  const agora = agoraLocal(agoraMs, off), hoje = agora.slice(0, 10);
  const dias = construirDias(d.time, d.precipitation_sum, hoje);
  return { hoje, offset: off, dias, solo: extrairSolo(j.hourly, agora.slice(0, 13)) };
}

function somar(dias) {
  let mm = 0, validos = 0, chuvosos = 0;
  for (const d of dias) if (d.mm !== null) { mm += d.mm; validos++; if (d.mm >= 1) chuvosos++; }
  return { mm, validos, total: dias.length, faltam: dias.length - validos, chuvosos };
}
function resumo(res) {
  return { passado: somar(res.dias.filter((d) => d.tipo === 'passado')), futuro: somar(res.dias.filter((d) => d.tipo !== 'passado')) };
}

function escalaTopo(max) {
  for (const t of [10, 20, 30, 50, 80, 100, 150, 200, 300, 500]) if (max <= t) return t;
  return 500;
}

/* ---- guardado neste navegador (armazenamento local), sempre validado ao ler: o conteúdo pode estar velho ou adulterado */
const serializar = (res, t) => ({ v: 1, t, hoje: res.hoje, off: res.offset, d: res.dias.map((x) => [x.data, x.mm]), s: res.solo });

function desserializar(o, agoraMs) {
  if (!o || o.v !== 1 || !ehNum(o.t) || o.t > agoraMs + 300000 || agoraMs - o.t > VALIDADE_CACHE_MS) return null;
  if (typeof o.hoje !== 'string' || !RE_DATA.test(o.hoje) || !ehNum(o.off) || !Array.isArray(o.d)) return null;
  const pares = o.d;
  if (!pares.every((p) => Array.isArray(p) && p.length === 2)) return null;
  const dias = construirDias(pares.map((p) => p[0]), pares.map((p) => p[1]), o.hoje);
  let solo = null;
  const s = o.s;
  if (s) {
    if (!ehNum(s.valor) || !ehNum(s.min) || !ehNum(s.max) || !ehNum(s.n) || typeof s.quando !== 'string' || !RE_HORA.test(s.quando)) return null;
    solo = { valor: s.valor, quando: s.quando, min: s.min, max: s.max, n: s.n };
  }
  return { t: o.t, res: { hoje: o.hoje, offset: o.off, dias, solo } };
}

function lerCache(storage, cod, agoraMs) {
  if (!storage || !RE_COD.test(String(cod))) return null;
  try {
    const bruto = storage.getItem(PREFIXO + cod);
    if (!bruto) return null;
    const lido = desserializar(JSON.parse(bruto), agoraMs);
    if (!lido) storage.removeItem(PREFIXO + cod);                   // vencido, do futuro ou fora do esquema: não serve mais
    return lido;
  } catch (e) {
    try { storage.removeItem(PREFIXO + cod); } catch (e2) { /* armazenamento bloqueado: segue sem ele */ }
    return null;
  }
}

function limparCache(storage, agoraMs, manter) {
  const itens = [];
  for (let i = storage.length - 1; i >= 0; i--) {
    const k = storage.key(i);
    if (!k || !k.startsWith(PREFIXO)) continue;
    let t = NaN;
    try { t = JSON.parse(storage.getItem(k)).t; } catch (e) { /* ilegível: sai abaixo */ }
    if (!ehNum(t) || agoraMs - t > VALIDADE_CACHE_MS) { storage.removeItem(k); continue; }
    itens.push({ k, t });
  }
  itens.sort((a, b) => a.t - b.t);
  while (itens.length > manter) storage.removeItem(itens.shift().k);
}

function gravarCache(storage, cod, res, agoraMs) {
  if (!storage || !RE_COD.test(String(cod))) return false;
  let valor;
  try { valor = JSON.stringify(serializar(res, agoraMs)); } catch (e) { return false; }
  try { storage.setItem(PREFIXO + cod, valor); } catch (e) {
    try { limparCache(storage, agoraMs, Math.floor(LIMITE_ENTRADAS / 2)); storage.setItem(PREFIXO + cod, valor); } catch (e2) { return false; }
  }
  try { limparCache(storage, agoraMs, LIMITE_ENTRADAS); } catch (e) { /* limpar é bônus */ }
  return true;
}

/* ---- consulta à Open-Meteo: tempo máximo, cancelamento e erros com motivo */
async function consultar(lat, lon, { fetchFn, agora = Date.now, tempoMs = TEMPO_MAX_MS, sinal } = {}) {
  const url = montarUrl(lat, lon);
  if (sinal && sinal.aborted) throw new ErroAoVivo('cancelado', 'cancelado');
  const ctl = new AbortController();
  const aoCancelar = () => ctl.abort();
  if (sinal) sinal.addEventListener('abort', aoCancelar, { once: true });
  let estourou = false;
  const relogio = setTimeout(() => { estourou = true; ctl.abort(); }, tempoMs);
  try {
    const r = await fetchFn(url, { signal: ctl.signal, credentials: 'omit', referrerPolicy: 'no-referrer' });
    if (r.status === 429) throw new ErroAoVivo('limite de consultas', 'limite');
    if (!r.ok) throw new ErroAoVivo('HTTP ' + r.status, 'http');
    let j;
    try { j = await r.json(); } catch (e) {
      if (ctl.signal.aborted) throw e;
      throw new ErroAoVivo('resposta não é JSON', 'formato');
    }
    return lerResposta(j, agora());
  } catch (e) {
    if (e instanceof ErroAoVivo) throw e;
    if (estourou) throw new ErroAoVivo('tempo esgotado', 'tempo');
    if (sinal && sinal.aborted) throw new ErroAoVivo('cancelado', 'cancelado');
    throw new ErroAoVivo('sem conexão', 'rede');
  } finally {
    clearTimeout(relogio);
    if (sinal) sinal.removeEventListener('abort', aoCancelar);
  }
}

window.PR_AOVIVO = { montarUrl, lerResposta, resumo, escalaTopo, serializar, desserializar, lerCache, gravarCache, limparCache, consultar,
  ErroAoVivo, constantes: { URL_API, PASSADO, FUTURO, TTL_FRESCO_MS, VALIDADE_CACHE_MS, LIMITE_ENTRADAS, TEMPO_MAX_MS, PREFIXO } };

/* ---------------------------------------------------------------- tela */
const P = window.PR, raiz = document.getElementById('aoVivo');
if (!P || !raiz) return;
const { D, $, h, mk, fmt } = P;
let store = null;
try { store = window.localStorage; store.getItem(PREFIXO); } catch (e) { store = null; }       // bloqueado (modo privado, política): sem guardado
const E = { seq: 0, i: -1, ctl: null, timer: null };

const mmTxt = (v) => (v < 10 ? fmt(v, 1) : fmt(v, 0));
const soloTxt = (v) => fmt(v, 2);
const diaCurto = (iso) => iso.slice(8, 10) + '/' + iso.slice(5, 7);
const diaSemana = (iso) => new Date(diaMs(iso)).toLocaleDateString('pt-BR', { weekday: 'short', timeZone: 'UTC' }).replace('.', '');
const horaTxt = (t) => new Date(t).toLocaleTimeString('pt-BR', { hour: '2-digit', minute: '2-digit', timeZone: FUSO });
const dataLocal = (t) => new Date(t).toLocaleDateString('en-CA', { timeZone: FUSO });
const quandoTxt = (t) => (dataLocal(t) === dataLocal(Date.now()) ? 'hoje, ' : new Date(t).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit', timeZone: FUSO }) + ', ') + horaTxt(t);
const TIPO = { passado: 'estimado pelo modelo', hoje: 'hoje', previsao: 'previsão' };
const MOTIVO = {
  rede: 'Sem conexão com a Open-Meteo agora.', tempo: 'A Open-Meteo demorou demais para responder.',
  limite: 'A Open-Meteo pediu para esperar um pouco (limite de consultas).', http: 'A Open-Meteo recusou a consulta.',
  formato: 'A resposta da Open-Meteo veio num formato que não consegui ler.',
};

function stat(rotulo, valor, extra) {
  const s = h('span'); s.append(rotulo + ': ', h('b', null, valor));
  if (extra) s.append(' ' + extra);
  return s;
}

function textoStatus(info) {
  if (info.origem === 'rede') return `Consultado hoje às ${horaTxt(info.t)}.`;
  if (info.origem === 'guardado') return `Consulta de ${quandoTxt(info.t)}, guardada neste navegador por 30 minutos.`;
  return `${MOTIVO[info.motivo] || MOTIVO.rede} Mostrando a última consulta, de ${quandoTxt(info.t)}: a previsão pode estar desatualizada.`;
}

function grafico(alvo, res, nome) {
  const W = 460, H = 160, X0 = 34, X1 = W - 6, Y0 = 22, Y1 = 130;
  const n = res.dias.length, passo = (X1 - X0) / n, larg = Math.max(2, passo - 1.6);
  const topo = escalaTopo(Math.max(0, ...res.dias.map((d) => (d.mm === null ? 0 : d.mm))));
  const y = (v) => Y1 - (v / topo) * (Y1 - Y0);
  const r = resumo(res);
  const svg = mk('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': `Chuva diária estimada em ${nome}: ${mmTxt(r.passado.mm)} milímetros nos últimos ${r.passado.total} dias e ${mmTxt(r.futuro.mm)} milímetros previstos para os próximos ${r.futuro.total}. Os mesmos números estão na tabela abaixo.` });
  for (const v of [topo, topo / 2]) {
    mk('line', { x1: X0, x2: X1, y1: y(v), y2: y(v), class: 'linha-g' }, svg);
    mk('text', { x: X0 - 5, y: y(v) + 4, 'text-anchor': 'end', class: 'eixo-t' }, svg).textContent = fmt(v, 0) + (v === topo ? ' mm' : '');
  }
  mk('line', { x1: X0, x2: X1, y1: Y1, y2: Y1, class: 'linha-0' }, svg);
  mk('text', { x: X0 - 5, y: Y1 + 4, 'text-anchor': 'end', class: 'eixo-t' }, svg).textContent = '0';
  const marca = mk('rect', { y: Y0 - 8, width: passo, height: Y1 - Y0 + 8, class: 'ao-marca', visibility: 'hidden' }, svg);
  res.dias.forEach((d, i) => {
    const x = X0 + i * passo + (passo - larg) / 2, prev = d.tipo !== 'passado';
    if (d.mm === null) { mk('circle', { cx: x + larg / 2, cy: Y1 - 4, r: 1.7, class: 'b-falta' }, svg); return; }
    if (d.mm === 0) { mk('rect', { x, y: Y1 - 2, width: larg, height: 2, class: 't-seco' }, svg); return; }
    const alt = Math.max(1.5, Y1 - y(d.mm));
    mk('rect', { x, y: Y1 - alt, width: larg, height: alt, class: 'b-chuva' + (prev ? ' b-prev' : '') }, svg);
  });
  const ih = res.dias.findIndex((d) => d.tipo === 'hoje');
  if (ih >= 0) {
    const xh = X0 + ih * passo + passo / 2;
    mk('line', { x1: xh, x2: xh, y1: Y0 - 8, y2: Y1, class: 'l-hoje' }, svg);
    mk('text', { x: xh, y: Y0 - 12, 'text-anchor': 'middle', class: 'eixo-t' }, svg).textContent = 'hoje';
  }
  mk('text', { x: X0, y: H - 6, class: 'eixo-t' }, svg).textContent = diaCurto(res.dias[0].data);
  mk('text', { x: X1, y: H - 6, 'text-anchor': 'end', class: 'eixo-t' }, svg).textContent = diaCurto(res.dias[n - 1].data);
  const sobre = mk('rect', { x: X0, y: 0, width: X1 - X0, height: Y1 + 6, fill: 'transparent' }, svg);
  const indice = (ev) => { const b = svg.getBoundingClientRect(); return Math.min(n - 1, Math.max(0, Math.floor(((ev.clientX - b.left) / b.width * W - X0) / passo))); };
  const mostrar = (ev) => {
    const i = indice(ev), d = res.dias[i];
    marca.setAttribute('x', X0 + i * passo); marca.setAttribute('visibility', 'visible');
    P.mostrarTip(ev, h('div', 't1', `${diaSemana(d.data)} ${diaCurto(d.data)}`),
      h('div', 't3', d.mm === null ? 'Sem dado neste dia' : `${fmt(d.mm, 1)} mm`), h('div', 't3', TIPO[d.tipo]));
  };
  sobre.addEventListener('pointermove', mostrar);
  sobre.addEventListener('pointerdown', mostrar);
  sobre.addEventListener('pointerleave', () => { marca.setAttribute('visibility', 'hidden'); P.esconderTip(); });
  alvo.replaceChildren(svg);
}

function tabela(res) {
  const corpo = $('aoCorpo'); corpo.replaceChildren();
  for (const d of res.dias) {
    const tr = h('tr'), mm = h('td', 'n', d.mm === null ? 'sem dado' : fmt(d.mm, 1));
    tr.append(h('td', null, `${diaSemana(d.data)} ${diaCurto(d.data)}`), mm, h('td', null, TIPO[d.tipo]));
    corpo.appendChild(tr);
  }
  $('aoResumoTab').textContent = `Ver os ${res.dias.length} dias em tabela`;
}

function desenhar(res, info, nome) {
  raiz.hidden = false; $('aoRetry').hidden = true; $('aoConteudo').hidden = false;
  const st = $('aoStatus'); st.textContent = textoStatus(info); st.classList.toggle('velho', info.origem === 'sem-rede');
  const r = resumo(res), falta = (x) => (x.faltam ? `(faltam ${x.faltam} de ${x.total} dias)` : '');
  const nChuva = (x) => (x.chuvosos === 1 ? 'em 1 dia' : `em ${x.chuvosos} dias`) + ' com 1 mm ou mais';
  const stats = [
    stat(`Últimos ${r.passado.total} dias`, `${mmTxt(r.passado.mm)} mm`, `${nChuva(r.passado)} ${falta(r.passado)}`.trim()),
    stat(`Próximos ${r.futuro.total} dias`, `${mmTxt(r.futuro.mm)} mm`, `${nChuva(r.futuro)} ${falta(r.futuro)}`.trim()),
  ];
  stats.push(res.solo
    ? stat('Umidade do solo (0 a 1 cm)', `${soloTxt(res.solo.valor)} m³/m³`, `(de ${soloTxt(res.solo.min)} a ${soloTxt(res.solo.max)} nos últimos ${r.passado.total} dias)`)
    : stat('Umidade do solo (0 a 1 cm)', 'indisponível agora'));
  $('aoStats').replaceChildren(...stats);
  grafico($('aoGraf'), res, nome);
  tabela(res);
}

function estadoConsultando() {
  raiz.hidden = false; $('aoRetry').hidden = true; $('aoConteudo').hidden = true;
  const st = $('aoStatus'); st.textContent = 'Consultando a Open-Meteo…'; st.classList.remove('velho');
}
function estadoErro(motivo) {
  raiz.hidden = false; $('aoConteudo').hidden = true; $('aoRetry').hidden = false;
  const st = $('aoStatus'); st.classList.add('velho');
  st.textContent = `${MOTIVO[motivo] || MOTIVO.rede} O resto do Monitor segue valendo: a série verificada está acima.`;
}

async function buscar(i, meu, antigo) {
  const m = D.municipios[i];
  E.ctl = new AbortController();
  try {
    const res = await consultar(m.lat, m.lon, { fetchFn: (u, o) => window.fetch(u, o), sinal: E.ctl.signal });
    if (meu !== E.seq) return;
    const t = Date.now();
    gravarCache(store, m.cod, res, t);
    desenhar(res, { origem: 'rede', t }, m.nome);
  } catch (e) {
    if (meu !== E.seq || (e && e.motivo === 'cancelado')) return;
    const velho = antigo || lerCache(store, m.cod, Date.now());
    if (velho) desenhar(velho.res, { origem: 'sem-rede', t: velho.t, motivo: e && e.motivo }, m.nome);
    else estadoErro(e && e.motivo);
  }
}

function selecionar(i) {
  E.seq++; E.i = i;
  const meu = E.seq;
  clearTimeout(E.timer);
  if (E.ctl) { E.ctl.abort(); E.ctl = null; }
  const m = i >= 0 ? D.municipios[i] : null;
  if (!m || !Number.isFinite(m.lat) || !Number.isFinite(m.lon)) { raiz.hidden = true; return; }
  const antigo = lerCache(store, m.cod, Date.now());
  if (antigo && Date.now() - antigo.t < TTL_FRESCO_MS) { desenhar(antigo.res, { origem: 'guardado', t: antigo.t }, m.nome); return; }
  estadoConsultando();
  E.timer = setTimeout(() => buscar(i, meu, antigo), 250);          // cliques seguidos viram uma consulta só
}

$('aoRetry').addEventListener('click', () => selecionar(E.i));
document.addEventListener('pr:municipio', (ev) => selecionar(ev.detail && Number.isInteger(ev.detail.i) ? ev.detail.i : -1));
})();
