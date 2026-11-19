/* Início: mapa do mês de abertura, números do mês, maiores secas e o resumo do El Niño. */
(() => {
'use strict';
const P = window.PR; if (!P) return;
const { D, NM, $, h, nf, fmtSpi, mesLongo, mesAbr, spiAt, clsAt, bxAt, ehSeca, nomeClasse, hrefMonitor } = P;
const k = D.escala_alvo, t = D.abertura;

P.legenda($('legenda'));
const ficha = $('fichaRapida');
const mapa = P.mapa($('mapa'), { k, t, aoClicar: (i) => mostrar(i), aoPassar: null });
const { nbc } = mapa.pintar(k, t);
$('mapaMes').textContent = `${mesLongo(t)} · SPI-${k}`;

function mostrar(i) {
  mapa.selecionar(i); ficha.replaceChildren();
  const v = spiAt(k, i, t), c = clsAt(k, i, t), m = D.municipios[i];
  const l1 = h('div', 'linha1'); l1.append(h('strong', 'nome-mun', m.nome), h('span', 'valor', isNaN(v) ? '—' : fmtSpi(v)));
  const cl = h('span', 'classe'), sw = h('i'); sw.style.background = c === 255 ? 'transparent' : `var(--c${c})`; if (c === 255) sw.style.border = '1px solid var(--axis)';
  cl.append(sw, document.createTextNode(nomeClasse(c))); l1.appendChild(cl); ficha.appendChild(l1);
  const baixa = c !== 255 && bxAt(k, i, t);
  ficha.appendChild(h('p', null, m.exc ? 'Sem dado confiável de satélite neste município. “Sem dado” não significa ausência de seca.'
    : baixa ? 'Baixa confiança: neste período do ano chove pouco e poucos milímetros mudam muito o índice.' : `Leitura confiável em ${mesLongo(t)}.`));
  const a = h('a', 'btn', 'Abrir a série completa'); a.href = hrefMonitor(i, t, k); a.style.marginTop = '10px'; ficha.appendChild(a);
}

/* números do mês: só leituras confiáveis entram na contagem de seca */
{
  let validos = 0, conf = 0, seca = 0;
  for (let i = 0; i < NM; i++) { const c = clsAt(k, i, t); if (c === 255) continue; validos++; if (!bxAt(k, i, t)) { conf++; if (ehSeca(c)) seca++; } }
  $('nSeca').textContent = nf.format(seca); $('nConf').textContent = nf.format(conf); $('nBaixa').textContent = nf.format(nbc);
  $('rotSeca').textContent = `em seca, entre as leituras confiáveis de ${mesLongo(t)}`;
  $('rotConf').textContent = `municípios com leitura confiável, de ${nf.format(validos)} com SPI`;
  $('rotBaixa').textContent = 'com baixa confiança: ficam fora desta contagem';
  const ul = $('maisSecos'), lst = [];
  for (let i = 0; i < NM; i++) { const v = spiAt(k, i, t); if (!isNaN(v) && !bxAt(k, i, t)) lst.push([v, i]); } lst.sort((a, b) => a[0] - b[0]);
  lst.slice(0, 5).forEach(([v, i]) => { const li = h('li'), a = h('a'); a.href = hrefMonitor(i, t, k); const sw = h('span', 'sw'); sw.style.background = `var(--c${clsAt(k, i, t)})`;
    a.append(sw, h('span', null, D.municipios[i].nome), h('span', 'v', fmtSpi(v))); li.appendChild(a); ul.appendChild(li); });
  if (!lst.length) ul.appendChild(h('li', 'vazio', 'Sem leitura confiável neste mês.'));
  $('rotMaisSecos').textContent = `Menores SPI-${k} entre as leituras confiáveis`;
}

/* maiores secas desde o início da série (um destaque por episódio) */
{
  const ul = $('ranking');
  D.maiores_secas.forEach((x) => {
    const li = h('li'), a = h('a'); a.href = `monitor.html#t=${D.meses[x.t]}&k=${k}`;
    const barra = h('span', 'barra'), i = h('i'); i.style.width = (x.n_seca / x.n_confiaveis * 100).toFixed(1) + '%'; barra.appendChild(i);
    a.append(h('span', 'mes', mesAbr(x.t)), barra, h('span', 'qtd', `${nf.format(x.n_seca)} de ${nf.format(x.n_confiaveis)} municípios`));
    a.title = `Abrir ${mesLongo(x.t)} no monitor`; li.appendChild(a); ul.appendChild(li);
  });
}

/* El Niño: o selo de atualização compara a data de hoje com a próxima discussão do CPC */
{
  const cpc = D.enso.fontes.find((f) => f.id === D.enso.situacao.fonte);
  const velho = cpc && cpc.proxima_atualizacao && new Date() > new Date(cpc.proxima_atualizacao + 'T23:59:59-03:00');
  $('ensoAtualizacao').hidden = !velho;
}
})();
