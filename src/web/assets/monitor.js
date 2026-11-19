/* Monitor: mapa, linha do tempo (matriz município x mês) e ficha do município. Só mostra chuva já observada. */
(() => {
'use strict';
const P = window.PR; if (!P) return;
const { D, NM, NT, ND, CLS, BX, $, mk, h, fmt, fmtSpi, nf, mesAbr, mesLongo, spiAt, clsAt, bxAt, chAt, nomeClasse, ehSeca, semAcento, mostrarTip, esconderTip } = P;
const S = { k: D.escala_alvo, t: D.abertura, sel: -1, tocando: false, timer: null, nbc: 0, nsb: 0 };
const rgb = (hex) => [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)];
let PAL = P.paleta();

/* estatísticas por escala e mês */
const SEC = {};
for (const k of D.escalas) {
  const nd = new Int16Array(NT), nv = new Int16Array(NT);
  for (let m = 0; m < NM; m++) for (let t = 0; t < NT; t++) { const c = CLS[k][m * NT + t]; if (c !== ND) { nv[t]++; if (D.classes[c].seca) nd[t]++; } }
  SEC[k] = { nd, nv };
}
const piorMes = (k) => { let b = 0; for (let t = 1; t < NT; t++) if (SEC[k].nd[t] > SEC[k].nd[b]) b = t; return b; };

/* controles fixos */
document.querySelectorAll('.fimSerie').forEach((e) => { e.textContent = mesAbr(NT - 1); });
{
  const seg = $('segEscala');
  for (const k of D.escalas) {
    const b = h('button', null, 'SPI-' + k); b.type = 'button'; b.dataset.k = k;
    b.title = `SPI-${k}: chuva dos últimos ${k} meses comparada ao histórico de ${D.periodo.inicio.slice(0, 4)} a ${D.periodo.treino_fim.slice(0, 4)}`;
    b.addEventListener('click', () => { S.k = k; atualizar(true); });
    seg.appendChild(b);
  }
  const dl = $('listaMun'); D.municipios.forEach((m) => { const o = h('option'); o.value = m.nome; dl.appendChild(o); });
}

/* mapa */
const mapa = P.mapa($('mapa'), { k: S.k, t: S.t, aoClicar: (i) => selecionar(i) });
P.legenda($('legendaMapa'));

/* matriz município x mês */
const heat = $('heat'), off = document.createElement('canvas');
off.width = NT; off.height = NM;
const heatOv = $('heatOv'); heatOv.setAttribute('viewBox', `0 0 ${NT} ${NM}`);
const ovPh = mk('line', { class: 'ph', x1: 0, x2: 0, y1: 0, y2: NM }, heatOv), ovPh2 = mk('line', { class: 'ph2', x1: 0, x2: 0, y1: 0, y2: NM }, heatOv);
const ovR1 = mk('line', { class: 'rsh', x1: 0, x2: NT, y1: 0, y2: 0 }, heatOv), ovR2 = mk('line', { class: 'rs', x1: 0, x2: NT, y1: 0, y2: 0 }, heatOv);
const ovMx = mk('line', { class: 'mira', x1: 0, x2: 0, y1: 0, y2: NM }, heatOv), ovMy = mk('line', { class: 'mira', x1: 0, x2: NT, y1: 0, y2: 0 }, heatOv);
const linhaDe = new Int16Array(NM); D.ordem.forEach((m, r) => { linhaDe[m] = r; });
const desenharMatriz = () => {
  const k = S.k, ctx = off.getContext('2d'), img = ctx.createImageData(NT, NM), px = img.data;
  const cores = PAL.map(rgb), vazio = rgb(getComputedStyle(document.documentElement).getPropertyValue('--axis').trim());
  for (let r = 0; r < NM; r++) { const m = D.ordem[r];
    for (let t = 0; t < NT; t++) { const c = CLS[k][m * NT + t], o = (r * NT + t) * 4, col = c === ND ? vazio : cores[c];
      px[o] = col[0]; px[o + 1] = col[1]; px[o + 2] = col[2]; px[o + 3] = c === ND ? 120 : (BX[k][m * NT + t] ? 105 : 255); } }
  ctx.putImageData(img, 0, 0);
  heat.width = NT * 3; heat.height = NM;
  const g = heat.getContext('2d'); g.imageSmoothingEnabled = false; g.clearRect(0, 0, heat.width, heat.height); g.drawImage(off, 0, 0, NT * 3, NM);
};
const tPorX = (ev, el) => { const r = el.getBoundingClientRect(); return Math.min(NT - 1, Math.max(0, Math.floor((ev.clientX - r.left) / r.width * NT))); };
let arrastando = false;
heatOv.addEventListener('pointerdown', (ev) => {
  arrastando = true; heatOv.setPointerCapture(ev.pointerId);
  const r = heatOv.getBoundingClientRect(), row = Math.min(NM - 1, Math.max(0, Math.floor((ev.clientY - r.top) / r.height * NM)));
  irPara(tPorX(ev, heatOv)); selecionar(D.ordem[row]);
});
heatOv.addEventListener('pointermove', (ev) => {
  const r = heatOv.getBoundingClientRect(), t = tPorX(ev, heatOv), row = Math.min(NM - 1, Math.max(0, Math.floor((ev.clientY - r.top) / r.height * NM)));
  ovMx.setAttribute('x1', t + .5); ovMx.setAttribute('x2', t + .5); ovMy.setAttribute('y1', row + .5); ovMy.setAttribute('y2', row + .5);
  ovMx.style.display = ovMy.style.display = '';
  if (arrastando) irPara(t);
  mostrarTip(ev, ...P.dicaMun(D.ordem[row], S.k, t));
});
heatOv.addEventListener('pointerup', () => { arrastando = false; });
heatOv.addEventListener('pointerleave', () => { ovMx.style.display = ovMy.style.display = 'none'; esconderTip(); });
ovMx.style.display = ovMy.style.display = 'none';

/* gráfico "% em seca" */
const shareSvg = $('share'), shOv = $('shOv');
const phS = mk('line', { class: 'ph2', y1: 0, y2: 58 }, shOv), miraS = mk('line', { class: 'mira', y1: 0, y2: 58 }, shOv);
miraS.style.display = 'none';
const desenharShare = () => {
  const { nd, nv } = SEC[S.k]; shareSvg.replaceChildren();
  mk('line', { class: 'g', x1: 0, x2: 1000, y1: 0.5, y2: 0.5 }, shareSvg); mk('line', { class: 'g', x1: 0, x2: 1000, y1: 29, y2: 29 }, shareSvg); mk('line', { class: 'g', x1: 0, x2: 1000, y1: 57.5, y2: 57.5 }, shareSvg);
  let d = 'M0 58', l = '';
  for (let t = 0; t < NT; t++) { const x = (t + .5) / NT * 1000, y = 57 - (nv[t] ? nd[t] / nv[t] : 0) * 55; d += `L${x.toFixed(1)} ${y.toFixed(1)}`; l += (t ? 'L' : 'M') + `${x.toFixed(1)} ${y.toFixed(1)}`; }
  mk('path', { class: 'a', d: d + 'L1000 58Z' }, shareSvg); mk('path', { class: 'l', d: l }, shareSvg);
};
shOv.addEventListener('pointerdown', (ev) => { arrastando = true; shOv.setPointerCapture(ev.pointerId); irPara(tPorX(ev, shOv)); });
shOv.addEventListener('pointermove', (ev) => {
  const t = tPorX(ev, shOv), x = (t + .5) / NT * 1000; miraS.setAttribute('x1', x); miraS.setAttribute('x2', x); miraS.style.display = '';
  if (arrastando) irPara(t);
  const { nd, nv } = SEC[S.k]; mostrarTip(ev, h('div', 't1', mesAbr(t)), h('div', 't2', `${nf.format(nd[t])} de ${nf.format(nv[t])} municípios em seca (${nv[t] ? Math.round(nd[t] / nv[t] * 100) : 0}%)`));
});
shOv.addEventListener('pointerup', () => { arrastando = false; });
shOv.addEventListener('pointerleave', () => { miraS.style.display = 'none'; esconderTip(); });

/* eixos */
{
  const anos = $('anos'), a0 = +D.meses[0].slice(0, 4), a1 = +D.meses[NT - 1].slice(0, 4);
  for (let a = Math.ceil(a0 / 5) * 5; a <= a1; a += 5) { const t = D.meses.indexOf(a + '-01'); if (t < 0) continue; const s = h('span', null, String(a)); s.style.left = ((t + .5) / NT * 100) + '%'; anos.appendChild(s); }
  const gs = $('gutShare'); gs.style.height = '58px';
  [['100%', 'translateY(0)', '0%'], ['0%', 'translateY(-100%)', '100%']].forEach(([tx, tr, top]) => { const s = h('span', null, tx); s.style.top = top; s.style.transform = tr; gs.appendChild(s); });
  const gh = $('gutHeat'); gh.style.height = '176px';
  [['norte', 'translateY(0)', '4px'], ['sul', 'translateY(-100%)', '172px']].forEach(([tx, tr, top]) => { const s = h('span', null, tx); s.style.top = top; s.style.transform = tr; gh.appendChild(s); });
}

/* ficha do município */
const W = 1000, H1 = 132, H2 = 64;
const desenharFicha = () => {
  const i = S.sel; $('munVazio').hidden = i >= 0; $('munFicha').hidden = i < 0;
  if (i < 0) return;
  const m = D.municipios[i], k = S.k;
  $('munNome').textContent = m.nome;
  $('munSub').textContent = `Código IBGE ${m.cod} · ${m.exc ? 'sem dado confiável' : 'semiárido piauiense'}`;
  const v = spiAt(k, i, S.t), c = clsAt(k, i, S.t);
  $('munSpi').textContent = isNaN(v) ? '—' : fmtSpi(v);
  $('munSw').style.background = c === ND ? 'transparent' : `var(--c${c})`; $('munSw').style.border = c === ND ? '1px solid var(--axis)' : '0';
  $('munClasse').textContent = `${nomeClasse(c)} · SPI-${k} em ${mesAbr(S.t)}`;
  const av = $('munAviso');
  if (m.exc) { av.hidden = false; av.textContent = 'Este município ficou sem SPI porque a cobertura de satélite no período de calibração ficou abaixo do mínimo' + (m.motivo ? ` (${m.motivo})` : '') + '. “Sem dado” não significa ausência de seca.'; }
  else if (bxAt(k, i, S.t)) { av.hidden = false; av.textContent = 'Baixa confiança: neste período do ano costuma chover muito pouco, então poucos milímetros mudam muito o SPI. Ele é pouco informativo sobre seca.'; }
  else if (isNaN(v)) { av.hidden = false; av.textContent = 'Sem SPI neste mês: o histórico ainda não completa a janela do índice ou a célula de calibração é inválida.'; }
  else av.hidden = true;
  /* SPI */
  const g1 = $('grafSpi'); g1.replaceChildren(); g1.appendChild(h('div', 'tit', `SPI-${k} mês a mês`));
  const s1 = mk('svg', { viewBox: `0 0 ${W} ${H1}`, preserveAspectRatio: 'none', height: H1 }, g1); s1.style.height = H1 + 'px';
  const y0 = H1 / 2, ymap = (x) => y0 - Math.max(-3.5, Math.min(3.5, x)) / 3.5 * (y0 - 3);
  const cl = mk('clipPath', { id: 'cpPos' }, mk('defs', {}, s1)); mk('rect', { x: 0, y: 0, width: W, height: y0 }, cl);
  const cn = mk('clipPath', { id: 'cpNeg' }, s1.querySelector('defs')); mk('rect', { x: 0, y: y0, width: W, height: y0 }, cn);
  [-2, 0, 2].forEach((g) => mk('line', { class: g === 0 ? 'zero' : 'gl', x1: 0, x2: W, y1: ymap(g), y2: ymap(g) }, s1));
  mk('line', { class: 'gl', x1: 0, x2: W, y1: ymap(-1), y2: ymap(-1), 'stroke-opacity': .6 }, s1);
  let area = '', linha = '', aberto = false, ini = 0;
  const fechar = (fim) => { if (aberto) { area += `L${((fim + .5) / NT * W).toFixed(1)} ${y0}L${((ini + .5) / NT * W).toFixed(1)} ${y0}Z`; aberto = false; } };
  for (let t = 0; t < NT; t++) { const x = spiAt(k, i, t); if (isNaN(x)) { fechar(t - 1); continue; }
    const px = ((t + .5) / NT * W).toFixed(1), py = ymap(x).toFixed(1);
    if (!aberto) { aberto = true; ini = t; area += `M${px} ${py}`; linha += `M${px} ${py}`; } else { area += `L${px} ${py}`; linha += `L${px} ${py}`; } }
  fechar(NT - 1);
  mk('path', { class: 'area-umida', d: area, 'clip-path': 'url(#cpPos)' }, s1); mk('path', { class: 'area-seca', d: area, 'clip-path': 'url(#cpNeg)' }, s1); mk('path', { class: 'linha', d: linha }, s1);
  [['+2', 2], ['0', 0], ['−2', -2]].forEach(([tx, g]) => { const s = h('span', 'yl', tx); s.style.top = (17 + ymap(g) - 5.5) + 'px'; g1.appendChild(s); });
  /* chuva */
  const g2 = $('grafChuva'); g2.replaceChildren(); g2.appendChild(h('div', 'tit', 'Chuva do mês, em milímetros'));
  const s2 = mk('svg', { viewBox: `0 0 ${W} ${H2}`, preserveAspectRatio: 'none' }, g2); s2.style.height = H2 + 'px';
  let mx = 0; for (let t = 0; t < NT; t++) { const x = chAt(i, t); if (!isNaN(x) && x > mx) mx = x; }
  const teto = Math.max(50, Math.ceil(mx / 50) * 50), bw = W / NT;
  mk('line', { class: 'gl', x1: 0, x2: W, y1: H2 - .5, y2: H2 - .5 }, s2);
  for (let t = 0; t < NT; t++) { const x = chAt(i, t); if (isNaN(x) || x <= 0) continue; const hh = x / teto * (H2 - 4); mk('rect', { class: 'barra', x: (t * bw + bw * .1).toFixed(2), y: (H2 - hh).toFixed(2), width: (bw * .8).toFixed(2), height: hh.toFixed(2) }, s2); }
  const yl = h('span', 'yl', '0'); yl.style.top = (17 + H2 - 9) + 'px'; g2.appendChild(yl); const y50 = h('span', 'yl', teto + ''); y50.style.top = '17px'; g2.appendChild(y50);
  /* marcador do mês atual e mira nos dois */
  [[s1, H1], [s2, H2]].forEach(([s, hh]) => {
    const ag = mk('line', { class: 'agora', y1: 0, y2: hh, x1: 0, x2: 0 }, s); ag.dataset.role = 'agora';
    const mira = mk('line', { class: 'mira', y1: 0, y2: hh, x1: 0, x2: 0 }, s); mira.style.display = 'none'; mira.dataset.role = 'mira';
    const hit = mk('rect', { x: 0, y: 0, width: W, height: hh, fill: 'transparent' }, s);
    hit.addEventListener('pointermove', (ev) => { const t = tPorX(ev, hit);
      [s1, s2].forEach((z) => { const q = z.querySelector('[data-role=mira]'); q.style.display = ''; q.setAttribute('x1', (t + .5) / NT * W); q.setAttribute('x2', (t + .5) / NT * W); });
      const x = spiAt(k, i, t), cc = clsAt(k, i, t), ch = chAt(i, t);
      mostrarTip(ev, h('div', 't1', mesAbr(t)), h('div', 't2', isNaN(x) ? 'SPI-' + k + ': sem dado' : `SPI-${k} ${fmtSpi(x)} · ${nomeClasse(cc)}`), h('div', 't3', isNaN(ch) ? 'Chuva: sem dado' : `Chuva ${fmt(ch, 1)} mm`)); });
    hit.addEventListener('pointerleave', () => { [s1, s2].forEach((z) => { z.querySelector('[data-role=mira]').style.display = 'none'; }); esconderTip(); });
    hit.addEventListener('click', (ev) => irPara(tPorX(ev, hit)));
  });
  /* eixo */
  const ex = $('eixoMun'); ex.replaceChildren(); const a0 = +D.meses[0].slice(0, 4), a1 = +D.meses[NT - 1].slice(0, 4);
  for (let a = Math.ceil(a0 / 10) * 10; a <= a1; a += 10) { const t = D.meses.indexOf(a + '-01'); if (t < 0) continue; const s = h('span', null, String(a)); s.style.left = ((t + .5) / NT * 100) + '%'; ex.appendChild(s); }
  /* números: só meses confiáveis entram em "meses em seca" e "mês mais seco" */
  let secos = 0, validos = 0, pior = NaN, piorT = -1, baixas = 0, total = 0;
  for (let t = 0; t < NT; t++) { const x = spiAt(k, i, t); if (isNaN(x)) continue; total++; if (bxAt(k, i, t)) { baixas++; continue; }
    validos++; if (D.classes[clsAt(k, i, t)].seca) secos++; if (isNaN(pior) || x < pior) { pior = x; piorT = t; } }
  const st = $('munStats'); st.replaceChildren();
  const par = (a, b) => { const d = h('div'); d.append(document.createTextNode(a + ' '), h('b', null, b)); st.appendChild(d); };
  par('Meses em seca (SPI ≤ −1,0, só meses confiáveis):', validos ? `${nf.format(secos)} de ${nf.format(validos)} (${Math.round(secos / validos * 100)}%)` : '—');
  par('Mês mais seco (só meses confiáveis):', piorT < 0 ? '—' : `${mesAbr(piorT)} (${fmtSpi(pior)})`);
  par('Meses de baixa confiança:', total ? `${nf.format(baixas)} de ${nf.format(total)}` : '—');
  posicionarAgora();
};
const posicionarAgora = () => {
  document.querySelectorAll('#munFicha line[data-role=agora]').forEach((l) => { const x = (S.t + .5) / NT * W; l.setAttribute('x1', x); l.setAttribute('x2', x); });
};

/* painel do mês */
const desenharMes = () => {
  const k = S.k, t = S.t, { nd, nv } = SEC[k];
  $('mesLongo').textContent = mesLongo(t);
  $('heroN').textContent = nf.format(nd[t]);
  $('heroTxt').textContent = nd[t] === 1 ? 'município em seca' : 'municípios em seca';
  $('heroSub').textContent = `de ${nf.format(nv[t])} com dado (${nv[t] ? Math.round(nd[t] / nv[t] * 100) : 0}%)`;
  $('heroBc').hidden = !S.nsb; if (S.nsb) $('heroBc').textContent = `${nf.format(S.nsb)} ${S.nsb === 1 ? 'deles tem' : 'deles têm'} baixa confiança`;
  $('notaEscala').textContent = `SPI-${k}: chuva dos últimos ${k} meses comparada ao histórico. Em seca = SPI de −1,0 ou menos. Mede falta de chuva, não prova que um açude vai secar.`;
  const cont = new Array(8).fill(0); for (let m = 0; m < NM; m++) { const c = CLS[k][m * NT + t]; cont[c === ND ? 7 : c]++; }
  const max = Math.max(1, ...cont), dist = $('dist'); dist.replaceChildren();
  const linhaDist = (c, rot, faixa) => { const li = h('li'); const sw = h('span', 'sw' + (c === 7 ? ' nd' : '')); if (c < 7) sw.style.background = `var(--c${c})`;
    const tx = h('span', 'tx'); tx.append(h('b', null, rot)); if (faixa) tx.append(h('small', null, faixa));
    li.append(sw, tx, h('span', 'n', nf.format(cont[c]))); const b = h('div', 'barra'); const i = h('i'); i.style.width = (cont[c] / max * 100) + '%'; b.appendChild(i); li.appendChild(b); dist.appendChild(li); };
  D.classes.forEach((c, i) => linhaDist(i, c.rotulo, c.faixa)); linhaDist(7, 'Sem dado', null);
  $('notaBc').hidden = !S.nbc;
  if (S.nbc) $('bcN').textContent = `${nf.format(S.nbc)} ${S.nbc === 1 ? 'município com baixa confiança' : 'municípios com baixa confiança'}`;
  const lst = []; for (let m = 0; m < NM; m++) { const v = spiAt(k, m, t); if (!isNaN(v)) lst.push([v, m]); } lst.sort((a, b) => a[0] - b[0]);
  const ul = $('top5'); ul.replaceChildren();
  lst.slice(0, 5).forEach(([v, m]) => { const li = h('li'), b = h('button'); b.type = 'button'; const sw = h('span', 'sw'); sw.style.background = `var(--c${clsAt(k, m, t)})`;
    b.append(sw, h('span', null, D.municipios[m].nome), h('span', 'v', fmtSpi(v))); b.addEventListener('click', () => selecionar(m)); li.appendChild(b); ul.appendChild(li); });
  if (!lst.length) ul.appendChild(h('li', 'vazio', 'Sem SPI neste mês.'));
};

/* o que ainda não existe */
{
  const ic = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="8" cy="8" r="5.6"/><path d="m4 12 8-8"/></svg>';
  const itens = [['Previsão do próximo mês', 'Indisponível: modelo em calibração. Só aparece depois de uma avaliação auditada, com margem de erro.'],
                 ['Antecedência em relação aos decretos (S2iD)', 'Indisponível: sem dado oficial validado.'],
                 ['Plano de Contingência (MUNIC/IBGE)', 'Sem informação oficial carregada neste painel.']];
  const ul = $('indisp'); itens.forEach(([a, b]) => { const li = h('li'); li.innerHTML = ic; const d = h('div'); d.append(h('b', null, a), document.createElement('br'), document.createTextNode(b)); li.appendChild(d); ul.appendChild(li); });
}

/* navegação no tempo */
const irPara = (t) => { S.t = Math.max(0, Math.min(NT - 1, t)); atualizar(false); };
const parar = () => { if (S.timer) clearInterval(S.timer); S.timer = null; S.tocando = false; rotularPlay(); };
const tocar = () => { if (S.t >= NT - 1) S.t = 0; S.tocando = true; rotularPlay(); S.timer = setInterval(() => { if (S.t >= NT - 1) { parar(); return; } S.t++; atualizar(false); }, 70); atualizar(false); };
const rotularPlay = () => { const b = $('btnPlay'); b.setAttribute('aria-pressed', S.tocando); b.innerHTML = S.tocando ? '<svg viewBox="0 0 16 16" fill="currentColor"><rect x="3.5" y="2.5" width="3" height="11" rx="1"/><rect x="9.5" y="2.5" width="3" height="11" rx="1"/></svg>Pausar' : '<svg viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg>Reproduzir'; };
$('btnPlay').addEventListener('click', () => S.tocando ? parar() : tocar());
$('btnAnt').addEventListener('click', () => { parar(); irPara(S.t - 1); });
$('btnProx').addEventListener('click', () => { parar(); irPara(S.t + 1); });
$('btnRecente').addEventListener('click', () => { parar(); irPara(NT - 1); });
$('btnAbertura').addEventListener('click', () => { parar(); irPara(D.abertura); });
$('btnPior').addEventListener('click', () => { parar(); irPara(piorMes(S.k)); });
$('tl').addEventListener('keydown', (ev) => {
  const passo = { ArrowLeft: -1, ArrowRight: 1, PageDown: -12, PageUp: 12 }[ev.key];
  if (passo) { ev.preventDefault(); parar(); irPara(S.t + passo); }
  else if (ev.key === 'Home') { ev.preventDefault(); parar(); irPara(0); } else if (ev.key === 'End') { ev.preventDefault(); parar(); irPara(NT - 1); }
  else if (ev.key === ' ') { ev.preventDefault(); S.tocando ? parar() : tocar(); }
});

/* seleção, busca, tabela */
function selecionar(i) { S.sel = i; mapa.selecionar(i); desenharFicha(); atualizarMatrizSel(); $('busca').value = i >= 0 ? D.municipios[i].nome : ''; gravarHash(); }
$('busca').addEventListener('change', (ev) => { const q = semAcento(ev.target.value.trim()); const i = D.municipios.findIndex((m) => semAcento(m.nome) === q); if (i >= 0) selecionar(i); });
const atualizarMatrizSel = () => { const r = S.sel >= 0 ? linhaDe[S.sel] : -1; [ovR1, ovR2].forEach((l) => { l.style.display = r < 0 ? 'none' : ''; l.setAttribute('y1', r + .5); l.setAttribute('y2', r + .5); }); };
document.addEventListener('pr:tema', () => { PAL = P.paleta(); desenharMatriz(); });
const dlg = $('dlg');
const preencherTabela = () => {
  const k = S.k, t = S.t; $('dlgT').textContent = `Todos os municípios em ${mesLongo(t)} · SPI-${k}`;
  const corpo = $('dlgCorpo'); corpo.replaceChildren();
  const ord = D.municipios.map((m, i) => i).sort((a, b) => { const x = spiAt(k, a, t), y = spiAt(k, b, t); return (isNaN(x) ? 99 : x) - (isNaN(y) ? 99 : y); });
  ord.forEach((i) => { const m = D.municipios[i], v = spiAt(k, i, t), c = clsAt(k, i, t), ch = chAt(i, t), tr = h('tr');
    const td0 = h('td', null, m.nome), td1 = h('td', 'n', isNaN(v) ? '—' : fmtSpi(v)), td2 = h('td'), sw = h('span', 'sw'); sw.style.background = c === ND ? 'transparent' : `var(--c${c})`; sw.style.border = c === ND ? '1px solid var(--axis)' : '0'; td2.append(sw, document.createTextNode(nomeClasse(c)));
    const td3 = h('td', 'n', isNaN(ch) ? '—' : fmt(ch, 1)), td4 = h('td', null, m.exc ? 'sem dado confiável' : (bxAt(k, i, t) ? 'baixa confiança' : ''));
    tr.append(td0, td1, td2, td3, td4); tr.addEventListener('click', () => { dlg.close(); selecionar(i); }); corpo.appendChild(tr); });
};
$('btnTabela').addEventListener('click', () => { preencherTabela(); dlg.showModal(); });
$('dlgFechar').addEventListener('click', () => dlg.close());
dlg.addEventListener('click', (ev) => { if (ev.target === dlg) dlg.close(); });

/* endereço: #m=<código IBGE>&t=AAAA-MM&k=3 */
function gravarHash() { P.escreverHash({ m: S.sel >= 0 ? D.municipios[S.sel].cod : '', t: D.meses[S.t], k: S.k }); }

/* atualização geral */
function atualizar(trocouEscala) {
  const k = S.k, t = S.t;
  document.querySelectorAll('#segEscala button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.k === k)));
  const r = mapa.pintar(k, t); S.nbc = r.nbc; S.nsb = r.nsb;
  desenharMes();
  $('mesAtual').textContent = mesLongo(t);
  $('tl').setAttribute('aria-valuemax', NT - 1); $('tl').setAttribute('aria-valuenow', t); $('tl').setAttribute('aria-valuetext', mesLongo(t));
  const x = t + .5; ovPh.setAttribute('x1', x); ovPh.setAttribute('x2', x); ovPh2.setAttribute('x1', x); ovPh2.setAttribute('x2', x);
  const xs = x / NT * 1000; phS.setAttribute('x1', xs); phS.setAttribute('x2', xs);
  $('legTempo').textContent = 'Faixa de cima: % de municípios em seca. Matriz: um município por linha, do norte ao sul; esmaecido = baixa confiança.';
  if (trocouEscala) { desenharShare(); desenharMatriz(); }
  if (S.sel >= 0) desenharFicha();
  if (dlg.open) preencherTabela();
  if (!S.tocando) gravarHash();
}

/* ponto de partida: o endereço, ou o município com o menor SPI no mês aberto */
{
  const p = P.lerHash();
  if (p.k && D.escalas.includes(+p.k)) S.k = +p.k;
  const ti = p.t ? P.idxMes(p.t) : -1; if (ti >= 0) S.t = ti;
  rotularPlay(); atualizar(true);
  let m = p.m ? P.idxMun(p.m) : -1;
  if (m < 0) { let mv = Infinity; for (let i = 0; i < NM; i++) { const v = spiAt(S.k, i, S.t); if (!isNaN(v) && !bxAt(S.k, i, S.t) && v < mv) { mv = v; m = i; } } }
  selecionar(m);
}
window.__painel = { S, D, irPara, selecionar, atualizar, trocarEscala: (k) => { S.k = k; atualizar(true); } };
})();
