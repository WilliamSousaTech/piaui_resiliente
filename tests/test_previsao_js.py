"""Busca e filtro da página Previsão (assets/previsao.js), rodados no Node com um DOM de mentira."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from src.web import build_site as bs

JS = Path(bs.ASSETS) / "previsao.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node não instalado")

PREAMBULO = r"""
const vm = require('vm'), fs = require('fs');
function el(extra = {}) {
  return { hidden: false, dataset: {}, attrs: {}, ouvintes: {}, textContent: '', filhos: {},
    addEventListener(t, f) { (this.ouvintes[t] = this.ouvintes[t] || []).push(f); },
    setAttribute(k, v) { this.attrs[k] = v; },
    querySelector(sel) { return this.filhos[sel] || null; },
    querySelectorAll(sel) { return this.filhos[sel] || []; },
    disparar(tipo, ev) { (this.ouvintes[tipo] || []).forEach((f) => f(ev)); },
    ...extra };
}
const linha = (nome, sinal) => el({ dataset: { sinal }, filhos: { th: el({ textContent: nome }) } });
function montar(linhas, tudo = true) {
  const botoes = ['todos', 'sinal'].map((f) => el({ dataset: { f } }));
  const caixa = el({ hidden: true, filhos: { '#segSinal button': botoes } });
  const busca = el(), contagem = el(), vazio = el({ hidden: true });
  const porId = { filtrosPrev: caixa, buscaPrev: busca, contagemPrev: contagem, vazioPrev: vazio };
  const sandbox = { document: { querySelectorAll: (s) => (s === '#corpoPrev tr' ? linhas : []), getElementById: (i) => porId[i] || null } };
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox);
  return { linhas, botoes, caixa, busca, contagem, vazio };
}
const visiveis = (p) => p.linhas.map((l, i) => [i, l.hidden]).filter(([, h]) => !h).map(([i]) => i);
const digitar = (p, t) => p.busca.disparar('input', { target: { value: t } });
const clicar = (p, f) => p.botoes.find((b) => b.dataset.f === f).disparar('click');
const LINHAS = () => [linha('São Félix do Piauí', '1'), linha('Altos', '1'), linha('Teresina', '0'), linha('Várzea Grande', '0')];
"""


def rodar(cenario: str):
    r = subprocess.run(["node", "-e", PREAMBULO + cenario, str(JS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_sintaxe_do_script():
    r = subprocess.run(["node", "--check", str(JS)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_ao_carregar_mostra_os_controles_e_tudo():
    s = rodar("const p = montar(LINHAS()); console.log(JSON.stringify([p.caixa.hidden, visiveis(p), p.contagem.textContent, p.vazio.hidden]));")
    assert s == [False, [0, 1, 2, 3], "4 de 4 municípios", True]


def test_busca_ignora_acento_caixa_e_espacos():
    s = rodar("""const p = montar(LINHAS()); digitar(p, '  SAO felix ');
                 const a = visiveis(p); digitar(p, 'varzea'); const b = visiveis(p);
                 console.log(JSON.stringify([a, b, p.contagem.textContent]));""")
    assert s == [[0], [3], "1 de 4 municípios"]


def test_filtro_com_sinal_e_busca_juntos_e_botao_marcado():
    s = rodar("""const p = montar(LINHAS()); clicar(p, 'sinal');
                 const a = [visiveis(p), p.botoes.map((b) => b.attrs['aria-pressed']), p.contagem.textContent];
                 digitar(p, 'alt'); const b = visiveis(p); digitar(p, 'teres'); const c = [visiveis(p), p.vazio.hidden];
                 clicar(p, 'todos'); const d = visiveis(p);
                 console.log(JSON.stringify([a, b, c, d]));""")
    assert s == [[[0, 1], ["false", "true"], "2 de 4 municípios"], [1], [[], False], [2]]


def test_sem_resultado_mostra_a_mensagem_e_apagar_traz_tudo():
    s = rodar("""const p = montar(LINHAS()); digitar(p, 'xyz'); const a = [visiveis(p), p.vazio.hidden, p.contagem.textContent];
                 digitar(p, ''); console.log(JSON.stringify([a, visiveis(p).length, p.vazio.hidden]));""")
    assert s == [[[], False, "0 de 4 municípios"], 4, True]


def test_um_so_municipio_usa_singular():
    s = rodar("const p = montar([linha('Altos', '1')]); console.log(JSON.stringify(p.contagem.textContent));")
    assert s == "1 de 1 município"


def test_sem_a_pagina_o_script_nao_faz_nada():
    s = rodar("""const sandbox = { document: { querySelectorAll: () => [], getElementById: () => null } };
                 vm.createContext(sandbox); vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), sandbox); console.log(JSON.stringify('ok'));""")
    assert s == "ok"
