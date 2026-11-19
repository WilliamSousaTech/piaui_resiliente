/* Planejamento: busca por município e filtro por situação da revisão. Só mostra ou esconde blocos que a página já traz;
   sem JavaScript, todos os planos continuam visíveis. */
(() => {
'use strict';
const semAcento = (s) => String(s).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
const blocos = Array.from(document.querySelectorAll('details.rascunho'));
const caixa = document.getElementById('filtrosPlano');
const busca = document.getElementById('buscaPlano');
const contagem = document.getElementById('contagemPlano');
const vazio = document.getElementById('vazioPlano');
if (!blocos.length || !caixa || !busca || !contagem) return;

const S = { texto: '', f: 'todos' };
const nomes = blocos.map((b) => { const h = b.querySelector('h3'); return semAcento(h ? h.textContent : ''); });

function aplicar() {
  let n = 0;
  blocos.forEach((b, i) => {
    const okTexto = !S.texto || nomes[i].includes(S.texto);
    const okSituacao = S.f === 'todos' || (S.f === 'revisados') === (b.dataset.rev === '1');
    b.hidden = !(okTexto && okSituacao);
    if (!b.hidden) n++;
  });
  contagem.textContent = `${n} de ${blocos.length} ${blocos.length === 1 ? 'plano' : 'planos'}`;
  if (vazio) vazio.hidden = n > 0;
  caixa.querySelectorAll('#segRevisao button').forEach((b) => b.setAttribute('aria-pressed', String(b.dataset.f === S.f)));
}

busca.addEventListener('input', (ev) => { S.texto = semAcento(ev.target.value.trim()); aplicar(); });
caixa.querySelectorAll('#segRevisao button').forEach((b) => b.addEventListener('click', () => { S.f = b.dataset.f; aplicar(); }));

/* link direto para um município (planejamento.html#m2207801): mostra e abre o bloco */
function abrirAncora() {
  const id = (location.hash || '').slice(1);
  const el = id ? document.getElementById(id) : null;
  if (el && el.matches('details.rascunho')) { el.hidden = false; el.open = true; el.scrollIntoView(); }
}
window.addEventListener('hashchange', abrirAncora);

caixa.hidden = false;
aplicar();
abrirAncora();
})();
