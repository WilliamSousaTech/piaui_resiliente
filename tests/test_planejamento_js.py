"""Busca e filtro da página Planejamento (assets/planejamento.js), rodados no Node com um DOM de mentira."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from src.web import build_site as bs

JS = Path(bs.ASSETS) / "planejamento.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node não instalado")

# Um DOM mínimo: só o que o script usa. `cenario` (JS) monta a página, roda o script e imprime um JSON.
PREAMBULO = r"""
const vm = require('vm'), fs = require('fs');
function el(extra = {}) {
  const e = { hidden: false, open: false, dataset: {}, attrs: {}, ouvintes: {}, textContent: '', scrolled: false, filhos: {},
    addEventListener(t, f) { (this.ouvintes[t] = this.ouvintes[t] || []).push(f); },
    setAttribute(k, v) { this.attrs[k] = v; },
    scrollIntoView() { this.scrolled = true; },
    matches(sel) { return sel === 'details.rascunho' && this.eBloco === true; },
    querySelector(sel) { return this.filhos[sel] || null; },
    querySelectorAll(sel) { return this.filhos[sel] || []; },
    disparar(tipo, ev) { (this.ouvintes[tipo] || []).forEach((f) => f(ev)); },
    ...extra };
  return e;
}
function bloco(cod, nome, rev) {
  return el({ id: 'm' + cod, eBloco: true, hidden: false, dataset: { rev: rev }, filhos: { h3: el({ textContent: nome }) } });
}
function montar(blocos, opcoes = {}) {
  const botoes = ['todos', 'revisados', 'nao'].map((f) => el({ dataset: { f } }));
  const caixa = el({ hidden: true, filhos: { '#segRevisao button': botoes } });
  const busca = el(), contagem = el(), vazio = el({ hidden: true });
  const porId = { filtrosPlano: caixa, buscaPlano: busca, contagemPlano: contagem, vazioPlano: vazio };
  blocos.forEach((b) => { porId[b.id] = b; });
  const janela = el();
  const sandbox = { document: { querySelectorAll: (s) => (s === 'details.rascunho' ? blocos : []), getElementById: (i) => porId[i] || null },
                    window: janela, location: { hash: opcoes.hash || '' } };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
  return { blocos, botoes, caixa, busca, contagem, vazio, janela, porId, sandbox };
}
const visiveis = (p) => p.blocos.filter((b) => !b.hidden).map((b) => b.id);
const digitar = (p, texto) => p.busca.disparar('input', { target: { value: texto } });
const clicar = (p, f) => p.botoes.find((b) => b.dataset.f === f).disparar('click');
const BLOCOS = () => [bloco('2200053', 'Acauã', '1'), bloco('2207801', 'Paulistana', '1'), bloco('2201101', 'Barras', '0'), bloco('2202000', 'São João do Piauí', '0')];
"""


def rodar(cenario: str):
    r = subprocess.run(["node", "-e", PREAMBULO + cenario, str(JS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_sintaxe_do_script():
    r = subprocess.run(["node", "--check", str(JS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_ao_carregar_mostra_os_controles_e_a_contagem_com_tudo_visivel():
    s = rodar("const p = montar(BLOCOS()); console.log(JSON.stringify([p.caixa.hidden, visiveis(p), p.contagem.textContent, p.vazio.hidden]));")
    assert s == [False, ["m2200053", "m2207801", "m2201101", "m2202000"], "4 de 4 planos", True]


def test_busca_ignora_acento_caixa_e_espacos():
    s = rodar("""const p = montar(BLOCOS()); digitar(p, '  ACAUA ');
                 const a = visiveis(p); digitar(p, 'sao joao'); const b = visiveis(p); digitar(p, 'pauli'); const c = visiveis(p);
                 console.log(JSON.stringify([a, b, c, p.contagem.textContent]));""")
    assert s == [["m2200053"], ["m2202000"], ["m2207801"], "1 de 4 planos"]


def test_busca_que_nao_acha_nada_mostra_a_mensagem_e_apagar_traz_tudo_de_volta():
    s = rodar("""const p = montar(BLOCOS()); digitar(p, 'xyz');
                 const a = [visiveis(p), p.vazio.hidden, p.contagem.textContent]; digitar(p, '');
                 console.log(JSON.stringify([a, visiveis(p).length, p.vazio.hidden]));""")
    assert s == [[[], False, "0 de 4 planos"], 4, True]


def test_filtro_de_situacao_separa_revisados_dos_nao_revisados_e_marca_o_botao():
    s = rodar("""const p = montar(BLOCOS()); clicar(p, 'revisados');
                 const a = [visiveis(p), p.botoes.map((b) => b.attrs['aria-pressed'])]; clicar(p, 'nao');
                 const b = [visiveis(p), p.botoes.map((x) => x.attrs['aria-pressed'])]; clicar(p, 'todos');
                 console.log(JSON.stringify([a, b, visiveis(p).length]));""")
    assert s == [[["m2200053", "m2207801"], ["false", "true", "false"]], [["m2201101", "m2202000"], ["false", "false", "true"]], 4]


def test_busca_e_filtro_valem_juntos():
    s = rodar("""const p = montar(BLOCOS()); clicar(p, 'nao'); digitar(p, 'barras'); const a = visiveis(p);
                 digitar(p, 'acaua'); const b = visiveis(p); clicar(p, 'todos'); const c = visiveis(p);
                 console.log(JSON.stringify([a, b, c, p.vazio.hidden]));""")
    assert s == [["m2201101"], [], ["m2200053"], True]


def test_contagem_no_singular():
    s = rodar("const p = montar([bloco('2200053', 'Acauã', '1')]); console.log(JSON.stringify(p.contagem.textContent));")
    assert s == "1 de 1 plano"


def test_link_direto_abre_e_mostra_o_bloco_mesmo_se_o_filtro_o_escondia():
    s = rodar("""const p = montar(BLOCOS(), { hash: '#m2201101' });
                 const b = p.porId.m2201101; const a = [b.open, b.hidden, b.scrolled];
                 clicar(p, 'revisados'); const escondido = b.hidden;
                 console.log(JSON.stringify([a, escondido]));""")
    assert s == [[True, False, True], True]


def test_hashchange_abre_e_mostra_outro_bloco():
    s = rodar("""const p = montar(BLOCOS()); clicar(p, 'revisados'); const alvo = p.porId.m2202000;
                 const antes = [alvo.hidden, alvo.open, alvo.scrolled];
                 p.sandbox.location.hash = '#m2202000'; p.janela.disparar('hashchange');
                 console.log(JSON.stringify([antes, [alvo.hidden, alvo.open, alvo.scrolled]]));""")
    assert s == [[True, False, False], [False, True, True]]


def test_busca_acha_no_meio_do_nome():
    s = rodar("const p = montar(BLOCOS()); digitar(p, 'do piaui'); console.log(JSON.stringify(visiveis(p)));")
    assert s == ["m2202000"]


def test_ancora_que_nao_e_de_um_plano_e_ignorada():
    s = rodar("""const p = montar(BLOCOS(), { hash: '#filtrosPlano' });
                 console.log(JSON.stringify([p.caixa.open, p.caixa.scrolled, visiveis(p).length]));""")
    assert s == [False, False, 4]


def test_sem_blocos_o_script_nao_faz_nada_e_nao_quebra():
    s = rodar("const p = montar([]); console.log(JSON.stringify([p.caixa.hidden, p.contagem.textContent, Object.keys(p.janela.ouvintes)]));")
    assert s == [True, "", []]


def test_ids_usados_pelo_script_existem_na_pagina_de_planejamento():
    from test_ia_site import AGORA  # noqa: F401  (garante que o módulo de apoio carrega)
    codigo = JS.read_text(encoding="utf-8")
    ids = set(re.findall(r"getElementById\('([^']+)'\)", codigo))
    html = (Path(bs.paginas.DIR_MODELOS) / "planejamento.html").read_text(encoding="utf-8")
    assert ids == {"filtrosPlano", "buscaPlano", "contagemPlano", "vazioPlano"}
    for i in ids:
        assert f'id="{i}"' in html, i
    assert 'id="segRevisao"' in html and 'data-f="todos"' in html and 'data-f="revisados"' in html and 'data-f="nao"' in html


def test_script_nao_chama_nada_de_fora():
    codigo = JS.read_text(encoding="utf-8")
    assert not re.search(r"fetch\(|XMLHttpRequest|import\s|eval\(|innerHTML|localStorage|https?://", codigo)
