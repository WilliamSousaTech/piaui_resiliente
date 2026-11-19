/* El Niño (página): chuva média por mês e desvio das estações chuvosas. Só chuva já observada (CHIRPS). */
(() => {
'use strict';
const P = window.PR; if (!P) return;
const { D, $, mk, h, fmt, nf, arred, mostrarTip, esconderTip } = P;
const C = D.clima, EST = C.estacao;
const MESES = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'];
const ABR = ['jan', 'fev', 'mar', 'abr', 'mai', 'jun', 'jul', 'ago', 'set', 'out', 'nov', 'dez'];
const pct = (x) => { const r = arred(x); return (r > 0 ? '+' : '') + (r === 0 ? '0' : r.toLocaleString('pt-BR').replace('-', '−')) + '%'; };

/* estação em cada mês do calendário (1..12), para destacar as barras */
const naEstacao = new Set(EST.meses);

const largura = (svg) => Math.max(300, Math.floor(svg.parentElement.clientWidth || 520));

function desenharClima() {
  const svg = $('grClim'); if (!svg) return; svg.replaceChildren();
  const W = largura(svg), H = Math.round(W * 0.56), L = 34, R = 6, T = 22, B = 26, pw = W - L - R, ph = H - T - B;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const v = C.climatologia, teto = Math.max(50, Math.ceil(Math.max(...v) / 50) * 50), bw = pw / 12;
  [0, .5, 1].forEach((f) => { const y = T + ph * (1 - f); mk('line', { class: f === 0 ? 'linha-0' : 'linha-g', x1: L, x2: W - R, y1: y, y2: y }, svg);
    const t = mk('text', { class: 'eixo-t', x: L - 6, y: y + 4, 'text-anchor': 'end' }, svg); t.textContent = Math.round(teto * f); });
  v.forEach((x, i) => {
    const hh = x / teto * ph, bx = L + i * bw + bw * .14, ww = bw * .72, m = i + 1;
    mk('rect', { class: naEstacao.has(m) ? 'b-chuvosa' : 'b-seco', x: bx, y: T + ph - hh, width: ww, height: Math.max(hh, 0), rx: 2 }, svg);
    const val = mk('text', { class: 'val-t', x: bx + ww / 2, y: T + ph - hh - 5, 'text-anchor': 'middle' }, svg); val.textContent = Math.round(x);
    const lab = mk('text', { class: 'eixo-t', x: bx + ww / 2, y: H - 8, 'text-anchor': 'middle' }, svg); lab.textContent = ABR[i];
    const hit = mk('rect', { x: L + i * bw, y: T, width: bw, height: ph, fill: 'transparent' }, svg);
    hit.addEventListener('pointermove', (ev) => mostrarTip(ev, h('div', 't1', MESES[i]), h('div', 't2', `${fmt(x, 1)} mm em média`), h('div', 't3', naEstacao.has(m) ? 'Faz parte da estação chuvosa' : '')));
    hit.addEventListener('pointerleave', esconderTip);
  });
  const u = mk('text', { class: 'eixo-t', x: L, y: 12 }, svg); u.textContent = 'mm por mês';
}

function desenharEstacoes() {
  const svg = $('grEst'); if (!svg) return; svg.replaceChildren();
  const anos = EST.anos, n = anos.length;
  const W = largura(svg), H = Math.round(W * 0.56), L = 40, R = 6, T = 22, B = 26, pw = W - L - R, ph = H - T - B;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  const maxAbs = Math.max(25, Math.ceil(Math.max(...anos.map((a) => Math.abs(a.desvio_pct))) / 25) * 25);
  const y = (p) => T + ph / 2 - p / maxAbs * (ph / 2), bw = pw / n;
  [-maxAbs, 0, maxAbs].forEach((g) => { mk('line', { class: g === 0 ? 'linha-0' : 'linha-g', x1: L, x2: W - R, y1: y(g), y2: y(g) }, svg);
    const t = mk('text', { class: 'eixo-t', x: L - 6, y: y(g) + 4, 'text-anchor': 'end' }, svg); t.textContent = g === 0 ? '0' : pct(g); });
  const secas = [...anos].sort((a, b) => a.desvio_pct - b.desvio_pct).slice(0, 3).map((a) => a.ano), ultimo = anos[n - 1].ano;
  const y0 = y(0), ys = anos.map((a) => y(a.desvio_pct)), colocados = [];
  /* rótulo do ano: dentro do gráfico, fora das barras vizinhas e sem encostar em outro rótulo */
  const rotular = (a, i) => {
    const neg = a.desvio_pct < 0, texto = `${a.ano} ${pct(a.desvio_pct)}`;
    const t = mk('text', { class: 'val-t', x: 0, y: 0, 'text-anchor': 'start' }, svg); t.textContent = texto;
    const w = (t.getComputedTextLength && t.getComputedTextLength()) || texto.length * 6.6;
    const l = Math.min(Math.max(L, L + i * bw + bw / 2 - w / 2), W - R - w);
    const j0 = Math.max(0, Math.floor((l - L) / bw)), j1 = Math.min(n - 1, Math.floor((l + w - L) / bw)), vizinhas = ys.slice(j0, j1 + 1);
    let base = neg ? Math.max(y0, ...vizinhas.filter((v) => v > y0)) + 13 : Math.min(y0, ...vizinhas.filter((v) => v < y0)) - 5;
    for (let k = 0; k < 3 && colocados.some((c) => l < c.r + 4 && l + w > c.l - 4 && Math.abs(base - c.y) < 13); k++) base += neg ? 13 : -13;
    base = Math.min(Math.max(base, 10), H - B - 2);
    colocados.push({ l, r: l + w, y: base });
    t.setAttribute('x', l); t.setAttribute('y', base);
  };
  anos.forEach((a, i) => {
    const bx = L + i * bw + bw * .12, ww = Math.max(1.5, bw * .76), neg = a.desvio_pct < 0, y1 = ys[i];
    mk('rect', { class: neg ? 'b-seca' : 'b-umida', x: bx, y: Math.min(y0, y1), width: ww, height: Math.max(1, Math.abs(y1 - y0)), rx: 1 }, svg);
    const hit = mk('rect', { x: L + i * bw, y: T, width: bw, height: ph, fill: 'transparent' }, svg);
    hit.addEventListener('pointermove', (ev) => mostrarTip(ev, h('div', 't1', `Estação de ${a.ano}`), h('div', 't2', `${nf.format(Math.round(a.total_mm))} mm (${pct(a.desvio_pct)} da média)`), h('div', 't3', `Média: ${nf.format(Math.round(EST.media_mm))} mm`)));
    hit.addEventListener('pointerleave', esconderTip);
  });
  anos.forEach((a, i) => { if (secas.includes(a.ano) || a.ano === ultimo) rotular(a, i); });
  const a0 = anos[0].ano, a1 = anos[n - 1].ano;
  for (let a = Math.ceil(a0 / 10) * 10; a <= a1; a += 10) { const i = anos.findIndex((x) => x.ano === a); if (i < 0) continue;
    const t = mk('text', { class: 'eixo-t', x: L + i * bw + bw / 2, y: H - 8, 'text-anchor': 'middle' }, svg); t.textContent = a; }
  const u = mk('text', { class: 'eixo-t', x: L, y: 12 }, svg); u.textContent = '% em relação à média';
}

const tudo = () => { desenharClima(); desenharEstacoes(); };
tudo();
let espera = 0; addEventListener('resize', () => { clearTimeout(espera); espera = setTimeout(tudo, 120); });

/* aviso de atualização do CPC: compara a data de hoje com a próxima discussão marcada na fonte */
{
  const cpc = D.enso && D.enso.fontes.find((f) => f.id === D.enso.situacao.fonte);
  const el = $('ensoAtualizacao');
  if (el && cpc && cpc.proxima_atualizacao) el.hidden = !(new Date() > new Date(cpc.proxima_atualizacao + 'T23:59:59-03:00'));
}
})();
