/* Núcleo compartilhado das páginas: decodificação do pacote, formatação, tema, dica flutuante, legenda e mapa.
   Sem rede: tudo vem de window.PR_DADOS (assets/dados.js). */
(() => {
'use strict';
const D = window.PR_DADOS;
if (!D) { document.body.insertAdjacentHTML('afterbegin', '<p style="padding:20px">Não consegui abrir assets/dados.js. Abra o site a partir da pasta gerada, sem mover os arquivos.</p>'); return; }
const NM = D.municipios.length, NT = D.meses.length, NA = -32768, ND = 255;
const ABR = ['jan','fev','mar','abr','mai','jun','jul','ago','set','out','nov','dez'];
const MESES_LONGOS = ['janeiro','fevereiro','março','abril','maio','junho','julho','agosto','setembro','outubro','novembro','dezembro'];
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const SPI = {}, CLS = {}, BX = {};
for (const k of D.escalas) { SPI[k] = dec(D.spi[k], Int16Array); CLS[k] = dec(D.cls[k], Uint8Array); BX[k] = dec(D.baixa[k], Uint8Array); }
const CH = dec(D.precip, Int16Array);
const ES = D.escala_spi, EC = D.escala_chuva;
const $ = (id) => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
const mk = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const a in attrs) e.setAttribute(a, attrs[a]); if (parent) parent.appendChild(e); return e; };
const h = (tag, cls, txt) => { const e = document.createElement(tag); if (cls) e.className = cls; if (txt != null) e.textContent = txt; return e; };
const fmt = (v, d = 2) => v.toLocaleString('pt-BR', { minimumFractionDigits: d, maximumFractionDigits: d }).replace('-', '−');
const fmtSpi = (v) => (v > 0 ? '+' : '') + fmt(v);
const nf = new Intl.NumberFormat('pt-BR');
const arred = (x) => Math.sign(x) * Math.round(Math.abs(x));   // meio para longe do zero, como paginas.num_br
const mesAbr = (t) => { const [a, m] = D.meses[t].split('-'); return ABR[+m - 1] + '/' + a; };
const mesLongo = (t) => { const [a, m] = D.meses[t].split('-'); return MESES_LONGOS[+m - 1] + ' de ' + a; };
const spiAt = (k, m, t) => { const v = SPI[k][m * NT + t]; return v === NA ? NaN : v / ES; };
const clsAt = (k, m, t) => CLS[k][m * NT + t];
const bxAt = (k, m, t) => BX[k][m * NT + t] === 1;
const chAt = (m, t) => { const v = CH[m * NT + t]; return v === NA ? NaN : v / EC; };
const nomeClasse = (c) => c === ND ? 'Sem dado' : D.classes[c].rotulo;
const ehSeca = (c) => c !== ND && D.classes[c].seca;
const dataBr = (iso) => { const d = new Date(iso); return isNaN(d) ? iso : d.toLocaleString('pt-BR', { dateStyle: 'short', timeStyle: 'short', timeZone: 'America/Fortaleza' }); };
const diaBr = (iso) => { const [a, m, d] = String(iso).slice(0, 10).split('-'); return `${d}/${m}/${a}`; };
const semAcento = (s) => s.normalize('NFD').replace(/[̀-ͯ]/g, '').toLowerCase();
const idxMun = (cod) => D.municipios.findIndex((m) => m.cod === String(cod));
const idxMes = (iso) => D.meses.indexOf(iso);
const hrefMonitor = (m, t, k) => `monitor.html#m=${D.municipios[m].cod}&t=${D.meses[t]}&k=${k}`;

/* hash da página: #m=<código IBGE>&t=AAAA-MM&k=3 */
const lerHash = () => { const p = {}; location.hash.replace(/^#/, '').split('&').forEach((kv) => { const [a, b] = kv.split('='); if (a) p[a] = decodeURIComponent(b || ''); }); return p; };
const escreverHash = (obj) => {
  const s = Object.entries(obj).filter(([, v]) => v != null && v !== '' && v !== false).map(([a, b]) => a + '=' + encodeURIComponent(b)).join('&');
  try { history.replaceState(null, '', s ? '#' + s : location.pathname + location.search); } catch (e) { /* sem histórico: segue sem atualizar o endereço */ }
};

/* tema claro/escuro */
const temaAtual = () => document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
const paleta = () => { const cs = getComputedStyle(document.documentElement); return Array.from({ length: 7 }, (_, i) => cs.getPropertyValue('--c' + i).trim()); };
const btnTema = $('btnTema');
if (btnTema) btnTema.addEventListener('click', () => {
  const novo = temaAtual() === 'dark' ? 'light' : 'dark';
  document.documentElement.dataset.theme = novo;
  try { localStorage.setItem('pr-tema', novo); } catch (e) { /* armazenamento indisponível: vale só nesta visita */ }
  document.dispatchEvent(new CustomEvent('pr:tema'));
});

/* dica flutuante */
const tip = $('tip');
const posTip = (ev) => {
  const r = tip.getBoundingClientRect(); let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
};
const esconderTip = () => { if (tip) tip.classList.remove('on'); };
const mostrarTip = (ev, ...nos) => { if (!tip) return; tip.replaceChildren(...nos); tip.classList.add('on'); posTip(ev); };
const dicaMun = (i, k, t, extra) => {
  const v = spiAt(k, i, t), c = clsAt(k, i, t), nos = [h('div', 't1', D.municipios[i].nome)];
  const l2 = h('div', 't2'), sw = h('i'); sw.style.background = c === ND ? 'transparent' : `var(--c${c})`;
  l2.append(sw, document.createTextNode(isNaN(v) ? 'Sem dado' : `SPI-${k} ${fmtSpi(v)} · ${nomeClasse(c)}`)); nos.push(l2);
  const ch = chAt(i, t); nos.push(h('div', 't3', mesAbr(t) + (isNaN(ch) ? '' : ` · chuva ${fmt(ch, 1)} mm`)));
  if (c !== ND && bxAt(k, i, t)) nos.push(h('div', 't3', 'Baixa confiança: pouca chuva esperada nesta época do ano'));
  if (extra) nos.push(h('div', 't3', extra));
  return nos;
};

/* legenda das sete classes */
const legenda = (el) => {
  el.classList.add('legenda'); el.replaceChildren();
  const faixa = h('div', 'faixa'); faixa.setAttribute('role', 'img'); faixa.setAttribute('aria-label', 'Escala do SPI, da seca extrema ao extremamente úmido');
  D.classes.forEach((c, i) => { const s = h('i'); s.style.background = `var(--c${i})`; s.title = `${c.rotulo} (${c.faixa})`; faixa.appendChild(s); });
  const rot = h('div', 'rotulos'); rot.append(h('span', null, 'Seca (SPI −1,0 ou menos)'), h('span', null, 'Próximo do normal'), h('span', null, 'Úmido (SPI 1,0 ou mais)'));
  const ex = h('div', 'extras');
  const a = h('span'); a.append(h('i', 'sw bc'), document.createTextNode('Pontilhado: baixa confiança')); const b = h('span'); b.append(h('i', 'sw nd'), document.createTextNode('Hachura: sem dado'));
  ex.append(a, b); el.append(faixa, rot, ex);
};

/* mapa: municípios coloridos pela classe do SPI, baixa confiança em pontilhado, anel de foco e de seleção */
const mapa = (svg, opc = {}) => {
  svg.classList.add('mapa-svg'); svg.setAttribute('viewBox', `0 0 ${D.viewBox[0]} ${D.viewBox[1]}`); svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  const defs = mk('defs', {}, svg);
  const pt = mk('pattern', { id: 'pontilhado', width: 9, height: 9, patternUnits: 'userSpaceOnUse' }, defs); mk('circle', { class: 'pt', cx: 2.5, cy: 2.5, r: 1.9 }, pt);
  const ha = mk('pattern', { id: 'hachura', width: 7, height: 7, patternUnits: 'userSpaceOnUse', patternTransform: 'rotate(45)' }, defs);
  mk('rect', { width: 7, height: 7, fill: 'var(--plane)' }, ha); mk('line', { x1: 0, y1: 0, x2: 0, y2: 7, stroke: 'var(--axis)', 'stroke-width': 2 }, ha);
  const gMun = mk('g', {}, svg), gBc = mk('g', {}, svg);
  const anelHov = mk('path', { id: 'anelHov', class: 'anel', d: '' }, svg), selBorda = mk('path', { id: 'anelSelBorda', class: 'anel', d: '' }, svg), anelSel = mk('path', { id: 'anelSel', class: 'anel', d: '' }, svg);
  const pinH = mk('circle', { class: 'pinH', cx: 0, cy: 0, r: 1 }, svg), pin = mk('circle', { class: 'pin', cx: 0, cy: 0, r: 1 }, svg); pinH.style.display = pin.style.display = 'none';
  const paths = D.municipios.map((m, i) => mk('path', { d: m.d, class: 'mun nd', 'data-i': i }, gMun));
  const bc = D.municipios.map((m) => { const p = mk('path', { d: m.d, class: 'bc' }, gBc); p.style.display = 'none'; return p; });
  const mover = (el, i) => el.setAttribute('d', i < 0 ? '' : D.municipios[i].d);
  const est = { k: opc.k || D.escala_alvo, t: opc.t == null ? NT - 1 : opc.t, hov: -1 };
  const pintar = (k, t) => {
    est.k = k; est.t = t; let nbc = 0, nsb = 0;
    for (let i = 0; i < NM; i++) {
      const c = clsAt(k, i, t), cn = 'mun ' + (c === ND ? 'nd' : 'c' + c);
      if (paths[i].getAttribute('class') !== cn) paths[i].setAttribute('class', cn);
      const baixa = c !== ND && bxAt(k, i, t); bc[i].style.display = baixa ? '' : 'none'; if (baixa) { nbc++; if (ehSeca(c)) nsb++; }
    }
    return { nbc, nsb };
  };
  const selecionar = (i) => {
    mover(anelSel, i); mover(selBorda, i); let mostrar = false;
    if (i >= 0) { const b = paths[i].getBBox(), m = Math.max(b.width, b.height);
      if (m > 0 && m < 90) { mostrar = true; for (const e of [pinH, pin]) { e.setAttribute('cx', b.x + b.width / 2); e.setAttribute('cy', b.y + b.height / 2); e.setAttribute('r', m / 2 + 13); } } }
    pinH.style.display = pin.style.display = mostrar ? '' : 'none';
  };
  svg.addEventListener('pointermove', (ev) => {
    const p = ev.target.closest && ev.target.closest('.mun');
    if (!p) { est.hov = -1; mover(anelHov, -1); esconderTip(); return; }
    const i = +p.dataset.i; if (i !== est.hov) { est.hov = i; mover(anelHov, i); }
    mostrarTip(ev, ...dicaMun(i, est.k, est.t, D.municipios[i].exc ? 'Município sem dado confiável' : ''));
    if (opc.aoPassar) opc.aoPassar(i);
  });
  svg.addEventListener('pointerleave', () => { est.hov = -1; mover(anelHov, -1); esconderTip(); });
  svg.addEventListener('click', (ev) => { const p = ev.target.closest && ev.target.closest('.mun'); if (p && opc.aoClicar) opc.aoClicar(+p.dataset.i); });
  return { pintar, selecionar, paths };
};

/* rodapé: o hash do conjunto de dados abre e fecha */
document.querySelectorAll('button.hash').forEach((b) => { const cheio = b.dataset.cheio; b.addEventListener('click', () => { b.textContent = b.textContent.endsWith('…') ? cheio : cheio.slice(0, 16) + '…'; }); });

window.PR = { D, NM, NT, NA, ND, SPI, CLS, BX, CH, ES, EC, $, mk, h, fmt, fmtSpi, nf, arred, mesAbr, mesLongo, spiAt, clsAt, bxAt, chAt, nomeClasse, ehSeca,
  dataBr, diaBr, semAcento, idxMun, idxMes, hrefMonitor, lerHash, escreverHash, temaAtual, paleta, tip, posTip, mostrarTip, esconderTip, dicaMun, legenda, mapa };
})();
