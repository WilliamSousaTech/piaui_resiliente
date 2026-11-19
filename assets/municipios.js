/* Municípios: tabela dos 216 municípios em um mês, com busca, filtros e ordenação. */
(() => {
'use strict';
const P = window.PR; if (!P) return;
const { D, NM, NT, ND, $, h, fmt, fmtSpi, nf, mesAbr, mesLongo, spiAt, clsAt, bxAt, chAt, nomeClasse, ehSeca, semAcento, hrefMonitor } = P;
const MESES = ['janeiro', 'fevereiro', 'março', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro'];
const S = { k: D.escala_alvo, t: D.abertura, texto: '', soSeca: false, soConf: false, col: 'spi', dir: 1 };

/* resumo por município e escala (só leituras confiáveis): meses em seca e mês mais seco */
const AGG = {};
for (const k of D.escalas) {
  AGG[k] = D.municipios.map((_, i) => {
    let validos = 0, secos = 0, pior = NaN, piorT = -1;
    for (let t = 0; t < NT; t++) { const v = spiAt(k, i, t); if (isNaN(v) || bxAt(k, i, t)) continue; validos++; if (ehSeca(clsAt(k, i, t))) secos++; if (isNaN(pior) || v < pior) { pior = v; piorT = t; } }
    return { validos, secos, pct: validos ? secos / validos : NaN, pior, piorT };
  });
}

/* controles */
{
  const seg = $('segEscala');
  for (const k of D.escalas) { const b = h('button', null, 'SPI-' + k); b.type = 'button'; b.dataset.k = k; b.addEventListener('click', () => { S.k = k; desenhar(); }); seg.appendChild(b); }
  const a0 = +D.meses[0].slice(0, 4), a1 = +D.meses[NT - 1].slice(0, 4), selA = $('selAno'), selM = $('selMes');
  for (let a = a1; a >= a0; a--) selA.appendChild(new Option(String(a), String(a)));
  MESES.forEach((m, i) => selM.appendChild(new Option(m, String(i + 1))));
  const ir = () => {
    const iso = `${selA.value}-${String(selM.value).padStart(2, '0')}`; let t = P.idxMes(iso);
    if (t < 0) t = selA.value < D.meses[0].slice(0, 4) || iso < D.meses[0] ? 0 : NT - 1;
    S.t = t; desenhar();
  };
  selA.addEventListener('change', ir); selM.addEventListener('change', ir);
  $('busca').addEventListener('input', (ev) => { S.texto = semAcento(ev.target.value.trim()); desenhar(); });
  $('soSeca').addEventListener('change', (ev) => { S.soSeca = ev.target.checked; desenhar(); });
  $('soConf').addEventListener('change', (ev) => { S.soConf = ev.target.checked; desenhar(); });
  document.querySelectorAll('th button[data-col]').forEach((b) => b.addEventListener('click', () => {
    if (S.col === b.dataset.col) S.dir = -S.dir; else { S.col = b.dataset.col; S.dir = 1; } desenhar();
  }));
}

const leitura = (i) => { const c = clsAt(S.k, i, S.t); return c === ND ? 'Sem dado' : (bxAt(S.k, i, S.t) ? 'Baixa confiança' : 'Confiável'); };
const chave = {
  nome: (i) => semAcento(D.municipios[i].nome),
  spi: (i) => spiAt(S.k, i, S.t),
  chuva: (i) => chAt(i, S.t),
  pct: (i) => AGG[S.k][i].pct,
  pior: (i) => AGG[S.k][i].pior,
};

function desenhar() {
  const k = S.k, t = S.t;
  const selA = $('selAno'), selM = $('selMes'); selA.value = D.meses[t].slice(0, 4); selM.value = String(+D.meses[t].slice(5, 7));
  document.querySelectorAll('#segEscala button').forEach((b) => b.setAttribute('aria-pressed', String(+b.dataset.k === k)));
  document.querySelectorAll('th button[data-col]').forEach((b) => { b.dataset.ordem = b.dataset.col === S.col ? String(S.dir) : ''; });
  let ids = D.municipios.map((_, i) => i).filter((i) => {
    if (S.texto && !semAcento(D.municipios[i].nome).includes(S.texto)) return false;
    const c = clsAt(k, i, t);
    if (S.soSeca && !ehSeca(c)) return false;
    if (S.soConf && (c === ND || bxAt(k, i, t))) return false;
    return true;
  });
  const f = chave[S.col];
  ids.sort((a, b) => { const x = f(a), y = f(b), nx = typeof x === 'number' && isNaN(x), ny = typeof y === 'number' && isNaN(y);
    if (nx || ny) return nx === ny ? 0 : (nx ? 1 : -1);
    return (x < y ? -1 : x > y ? 1 : 0) * S.dir || D.municipios[a].cod.localeCompare(D.municipios[b].cod); });
  const corpo = $('corpo'); corpo.replaceChildren();
  ids.forEach((i) => {
    const m = D.municipios[i], v = spiAt(k, i, t), c = clsAt(k, i, t), ch = chAt(i, t), ag = AGG[k][i], tr = h('tr');
    const td0 = h('td'), a = h('a', null, m.nome); a.href = hrefMonitor(i, t, k); td0.appendChild(a);
    const td1 = h('td', 'n', isNaN(v) ? '—' : fmtSpi(v));
    const td2 = h('td'), sw = h('span', 'sw'); sw.style.background = c === ND ? 'transparent' : `var(--c${c})`; if (c === ND) sw.style.border = '1px solid var(--axis)'; td2.append(sw, document.createTextNode(nomeClasse(c)));
    const td3 = h('td', 'n', isNaN(ch) ? '—' : fmt(ch, 1));
    const td4 = h('td', 'n'); if (ag.validos) { td4.append(document.createTextNode(`${Math.round(ag.pct * 100)}%`)); const bar = h('span', 'barrinha'), bi = h('i'); bi.style.width = (ag.pct * 100).toFixed(1) + '%'; bar.appendChild(bi); td4.appendChild(bar); } else td4.textContent = '—';
    const td5 = h('td', 'n', ag.piorT < 0 ? '—' : `${mesAbr(ag.piorT)} (${fmtSpi(ag.pior)})`);
    const td6 = h('td', null, m.exc ? 'Sem dado confiável' : leitura(i));
    tr.append(td0, td1, td2, td3, td4, td5, td6); corpo.appendChild(tr);
  });
  $('contagem').textContent = `${nf.format(ids.length)} de ${nf.format(NM)} municípios em ${mesLongo(t)} · SPI-${k}`;
  $('vazioTab').hidden = ids.length > 0;
}
desenhar();
})();
