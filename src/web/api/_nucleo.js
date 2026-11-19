'use strict';
/* Núcleo do plano ao vivo: tudo o que não depende de rede nem do ambiente. Espelha `src/ia/rascunhos.py` (validar,
   interpretar, limpar_texto, montar_pedido) e `src/ia/fatos.py` (números permitidos). Quem manda é o Python: o
   `tests/test_api_nucleo.py` roda os dois lados nos mesmos textos e exige o mesmo resultado.

   Diferença deliberada, só aqui: o texto ao vivo não passa por uma pessoa antes de aparecer, então um número decimal
   escrito com o sinal trocado em relação à ficha (SPI -0,30 virando 0,30) também é recusado. O Python não faz isso
   (decisão aberta, contrato 5.8). Recusar a mais é seguro; aceitar a mais não seria.

   `\b` e `\w` do JavaScript só entendem ASCII; aqui eles viram as versões Unicode (`LIM_I`, `LIM_F`, `PAL`) para
   "prevê" e "previsão" serem pegas como no Python. */

const PAL = '[\\p{L}\\p{N}_]';
const LIM_I = '(?<![\\p{L}\\p{N}_])';
const LIM_F = '(?![\\p{L}\\p{N}_])';

const CITACAO = new RegExp(
  LIM_I + '(?:leis?|decretos?|portarias?|resolu[cç][aã]o|resolu[cç][oõ]es|arts?|artigos?|incisos?|' +
  'par[aá]grafos?|nbr|cobrade|s2id|constitui[cç][aã]o)' + LIM_F + '|§', 'iu');
const CERTEZA = new RegExp(
  LIM_I + '(?:certamente|com certeza|sem d[uú]vida|garant' + PAL + '+|definitivamente|inevit[aá]ve' + PAL + '+|' +
  'prev[eê]|prever|prevemos|previs[aã]o|previs[oõ]es|prognóstico|colaps' + PAL + '+|' +
  'vai (?:secar|faltar|piorar|chover|acabar)|ir[aá] (?:secar|faltar|piorar|chover|acabar))' + LIM_F, 'iu');
const FORMATO = /[<>`*#[\]]|https?:|www\.|\n|\r/i;
const NUMERO = /\p{Nd}+(?:[.,]\p{Nd}+)*/gu;
const MILHAR = /\.(?=\p{Nd}{3}(?!\p{Nd}))/gu;
const MENOS = new Set(['\u2212', '-', '\u2013']);       // menos, hífen-menos e meia-risca

class RascunhoError extends Error {}

const comprimento = (t) => [...t].length;               // pontos de código, como o len() do Python
const repr = (t) => (t.includes("'") && !t.includes('"') ? `"${t}"` : `'${t.replace(/\\/g, '\\\\').replace(/'/g, "\\'")}'`);

function limparTexto(t) {
  if (typeof t !== 'string') return t;
  return t.replace(/[\u2010\u2011]/gu, '-').replace(/(?! )\p{Zs}/gu, ' ');
}

function estranhos(t) {
  const achados = new Set();
  for (const c of t) {
    if (c === '\n' || c === '\r') continue;
    if (/^\p{C}$/u.test(c) || (/^\p{Z}$/u.test(c) && c !== ' ')) achados.add(c.codePointAt(0));
  }
  return [...achados].sort((a, b) => a - b);
}

function normalizarNumero(token) {
  return token.replace(MILHAR, '').replace(/\./g, ',');
}

function numeros(texto) {
  return (texto.match(NUMERO) || []).map(normalizarNumero);
}

function numerosPermitidos(...textos) {
  return new Set(textos.flatMap(numeros));
}

function numerosFora(texto, permitidos) {
  const vistos = [];
  for (const n of numeros(texto)) if (!permitidos.has(n) && !vistos.includes(n)) vistos.push(n);
  return vistos;
}

/* Decimais com sinal: "−1,23" (menos de verdade) ou "-1,23" (hífen) logo antes do número. Um hífen colado em letra ou
   dígito ("SPI-3", "2026-07") é hífen, não sinal. Devolve [valor normalizado, negativo?]. */
function decimaisComSinal(texto) {
  const saida = [];
  for (const m of texto.matchAll(NUMERO)) {
    const normal = normalizarNumero(m[0]);
    if (!normal.includes(',')) continue;
    const ant = m.index > 0 ? [...texto.slice(0, m.index)].pop() : '';
    let negativo = false;
    if (MENOS.has(ant)) {
      const antes = [...texto.slice(0, m.index - ant.length)].pop() || '';
      negativo = ant === '\u2212' || !/^[\p{L}\p{N}]$/u.test(antes);
    }
    saida.push([normal, negativo]);
  }
  return saida;
}

function sinaisDivergentes(texto, ficha) {
  const naFicha = new Set(decimaisComSinal(ficha).map(([n, neg]) => `${neg ? '-' : '+'}${n}`));
  const ruins = [];
  for (const [n, neg] of decimaisComSinal(texto)) {
    const chave = `${neg ? '-' : '+'}${n}`;
    const outra = `${neg ? '+' : '-'}${n}`;
    if (!naFicha.has(chave) && naFicha.has(outra) && !ruins.includes(n)) ruins.push(n);
  }
  return ruins;
}

function textoOk(rotulo, texto, limite, permitidos, ficha) {
  if (typeof texto !== 'string' || !texto.trim()) return [`${rotulo}: vazio ou não é texto`];
  const motivos = [];
  if (comprimento(texto) > limite) motivos.push(`${rotulo}: ${comprimento(texto)} caracteres, o limite é ${limite}`);
  if (texto !== texto.trim()) motivos.push(`${rotulo}: espaço nas pontas`);
  if (FORMATO.test(texto)) motivos.push(`${rotulo}: tem HTML, markdown, link ou quebra de linha`);
  const est = estranhos(texto);
  if (est.length) {
    const cods = est.slice(0, 5).map((n) => 'U+' + n.toString(16).toUpperCase().padStart(4, '0')).join(', ');
    motivos.push(`${rotulo}: tem caractere invisível ou especial (${cods}); use só letras, números, ` +
                 'pontuação e espaço comuns, com um espaço entre as palavras');
  }
  let c = CITACAO.exec(texto);
  if (c) motivos.push(`${rotulo}: cita fonte ou norma (${repr(c[0])}); isso é da equipe`);
  c = CERTEZA.exec(texto);
  if (c) motivos.push(`${rotulo}: linguagem de certeza ou de previsão (${repr(c[0])})`);
  const fora = numerosFora(texto, permitidos);
  if (fora.length) motivos.push(`${rotulo}: número fora da ficha (${fora.join(', ')})`);
  const sinais = sinaisDivergentes(texto, ficha);
  if (sinais.length) motivos.push(`${rotulo}: sinal diferente do da ficha (${sinais.join(', ')})`);
  return motivos;
}

/* `pacote` é o `_pacote.json`: sistema, checklist, limites. `itens` é uma lista de [id, porque]. */
function validar(resumo, itens, ficha, pacote) {
  const lim = pacote.limites;
  const permitidos = numerosPermitidos(ficha, pacote.checklist_texto);
  let motivos = textoOk('resumo', resumo, lim.max_resumo, permitidos, ficha);
  if (!Array.isArray(itens) || !itens.length) return [...motivos, 'itens: lista vazia'];
  if (itens.length > lim.max_itens) motivos.push(`itens: ${itens.length}, o máximo é ${lim.max_itens}`);
  if (itens.length < Math.min(2, pacote.checklist.length)) motivos.push('itens: escolha pelo menos 2');
  const conhecidos = new Set(pacote.checklist.map((i) => i.id));
  const vistos = [];
  itens.forEach(([id, porque], k) => {
    const n = k + 1;
    if (!conhecidos.has(id)) motivos.push(`item ${n}: id ${repr(String(id))} não existe na base do checklist`);
    else if (vistos.includes(id)) motivos.push(`item ${n}: id ${repr(String(id))} repetido`);
    vistos.push(id);
    motivos = motivos.concat(textoOk(`item ${n} (${id})`, porque, lim.max_porque, permitidos, ficha));
  });
  return motivos;
}

function interpretar(resposta) {
  let t = String(resposta).trim();
  if (t.startsWith('```')) t = t.replace(/^```[a-zA-Z]*\s*|\s*```$/g, '').trim();
  let d;
  try { d = JSON.parse(t); } catch (e) { throw new RascunhoError('a resposta não é um JSON válido'); }
  const chaves = (o) => Object.keys(o).sort().join(',');
  if (d === null || typeof d !== 'object' || Array.isArray(d) || chaves(d) !== 'itens,resumo') {
    throw new RascunhoError('o JSON precisa ter exatamente as chaves "resumo" e "itens"');
  }
  if (!Array.isArray(d.itens)) throw new RascunhoError('"itens" precisa ser uma lista');
  const itens = d.itens.map((x, k) => {
    if (x === null || typeof x !== 'object' || Array.isArray(x) || chaves(x) !== 'id,porque' ||
        typeof x.id !== 'string' || typeof x.porque !== 'string') {
      throw new RascunhoError(`item ${k + 1}: precisa ser {"id": texto, "porque": texto}`);
    }
    return [x.id, limparTexto(x.porque)];
  });
  return [limparTexto(d.resumo), itens];
}

function montarPedido(ficha, pacote, recusas) {
  const partes = [`FICHA DO MUNICÍPIO\n${ficha}`, `CHECKLIST (escolha só destes ids)\n${pacote.checklist_texto}`];
  if (recusas && recusas.length) {
    partes.push('A resposta anterior foi recusada pelos motivos abaixo. Corrija e responda de novo, só com o JSON.\n' +
                recusas.map((m) => `- ${m}`).join('\n'));
  }
  partes.push('Escreva o rascunho agora.');
  return partes.join('\n\n');
}

/* O plano que o navegador recebe: texto da IA (só depois de validado) e, de cada item, título, descrição e fonte vindos
   da base da equipe. Sempre "pendente_revisao": nada que nasce ao vivo é revisado. */
function montarPlano(pacote, cod, resumo, itens, { modelo, tentativas, agora }) {
  const m = pacote.municipios[cod];
  const base = new Map(pacote.checklist.map((i) => [i.id, i]));
  return {
    cod_ibge: cod, nome: m.nome, mes_referencia: pacote.mes_referencia, escala: pacote.escala, ficha: m.ficha,
    resumo,
    itens: itens.map(([id, porque]) => ({ id, titulo: base.get(id).titulo, descricao: base.get(id).descricao,
                                           fonte: base.get(id).fonte, porque })),
    modelo, gerado_em_utc: agora.toISOString().replace(/\.\d{3}Z$/, 'Z'), tentativas,
    status: 'pendente_revisao', rotulo: pacote.rotulo_nao_revisado, aviso: pacote.aviso_mvp,
  };
}

module.exports = { RascunhoError, limparTexto, estranhos, numeros, numerosPermitidos, numerosFora, sinaisDivergentes,
                   textoOk, validar, interpretar, montarPedido, montarPlano };
