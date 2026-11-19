/* Previsão: busca por município e filtro "com sinal de seca". Só mostra ou esconde linhas que a página já traz;
   sem JavaScript, todos os municípios continuam visíveis. */
(() => {
'use strict';
const semAcento = (s) => String(s).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
const linhas = Array.from(document.querySelectorAll('#corpoPrev tr'));
const caixa = document.getElementById('filtrosPrev');
const busca = document.getElementById('buscaPrev');
const contagem = document.getElementById('contagemPrev');
const vazio = document.getElementById('vazioPrev');
if (!linhas.length || !caixa || !busca || !contagem) return;

const S = { texto: '', f: 'todos' };
const nomes = linhas.map((tr) => { const a = tr.querySelector('th'); return semAcento(a ? a.textContent : ''); });

function aplicar() {
  let n = 0;
  linhas.forEach((tr, i) => {
    const okTexto = !S.texto || nomes[i].includes(S.texto);
    const okSinal = S.f === 'todos' || tr.dataset.sinal === '1';
    tr.hidden = !(okTexto && okSinal);
    if (!tr.hidden) n++;
  });
  contagem.textContent = `${n} de ${linhas.length} ${linhas.length === 1 ? 'município' : 'municípios'}`;
  if (vazio) vazio.hidden = n > 0;
  caixa.querySelectorAll('#segSinal button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.f === S.f)));
}

busca.addEventListener('input', (ev) => { S.texto = semAcento(ev.target.value.trim()); aplicar(); });
caixa.querySelectorAll('#segSinal button').forEach((b) => b.addEventListener('click', () => { S.f = b.dataset.f; aplicar(); }));

caixa.hidden = false;
aplicar();
})();
