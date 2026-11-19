# Prompt profissional e uso da API de IA no site (proposta)

Versão 0.1 · 2026-11-19 · Status: **proposta, aguarda OK da equipe**. Este arquivo é só documentação: não muda código, não muda o contrato e não publica nada. Quando a equipe disser OK, o texto da seção 3 entra em `src/ia/rascunhos.py` (`SISTEMA`) com testes, um arquivo por vez.

O site continua em MVP experimental, em calibração com S2iD/TCE-PI.

## 1. Resumo

- A IA **só escreve rascunho** a partir da ficha do município (dado observado, nunca previsão). Quem garante as regras é o **código** (`rascunhos.validar`), não o prompt. O prompt só reduz o número de recusas.
- O jeito mais seguro para sábado: gerar os rascunhos **no PC da equipe** com a API, revisar (`aprovar`) e publicar como página estática. A chave **não vai** para a Vercel.
- O jeito "ao vivo" (botão no site que chama a API) é possível, mas exige uma função de servidor na Vercel, uma regra nova no contrato (D14) e proteções contra abuso. Está descrito na seção 6, **não está construído**.

## 2. O que a IA faz e o que não faz

| Faz | Não faz |
|---|---|
| Lê a ficha do município e a lista de ações da equipe (checklist) | Não prevê chuva, seca ou volume de açude |
| Escolhe de 2 a 6 ações pelo id e explica por que merecem atenção | Não cria ações nem muda o sentido delas |
| Escreve um resumo de até 600 caracteres e um "porque" de até 300 por ação | Não calcula, não estima, não compara números |
| | Não cita lei, decreto, portaria, artigo ou S2iD (a fonte de cada ação é da equipe) |
| | Não usa linguagem de certeza, de previsão ou de colapso |

Travas no código (já implementadas e testadas, contrato 5.8): ids existentes, números só da ficha ou do checklist, sem citação legal, sem certeza/previsão, sem HTML/markdown/link, rascunho nasce `pendente_revisao`, só `revisado` por uma pessoa vai ao site, e só se bater com o `dataset_sha256` e o `checklist_sha256` atuais.

## 3. Prompt de sistema proposto

> **Aplicado em 2026-11-19** em `src/ia/rascunhos.py` (`SISTEMA`), com uma seção a mais, CARACTERES (só espaço e hífen comuns). A regra de proporção continua sendo proposta: a equipe precisa confirmar. Os limites vêm das constantes do código.

Compatível com o validador atual (mesmas regras, mesmo formato de saída). Os limites (600, 300, 6) vêm das constantes de `rascunhos.py`.

```text
PAPEL
Você é analista técnico de apoio à Defesa Civil municipal no semiárido do Piauí. Escreve rascunhos de planejamento que um técnico da prefeitura vai ler, conferir e só então usar. Você não decide, não prevê e não substitui o plano do município: organiza o que os dados já mostram e aponta quais ações do checklist merecem atenção primeiro.

PÚBLICO E TOM
O leitor é coordenador ou técnico de Defesa Civil de um município pequeno, com pouco tempo e sem formação em estatística. Escreva em português do Brasil, com frases curtas, verbos de ação e sem jargão. Na primeira vez que usar "SPI", diga em poucas palavras que é um índice que compara a chuva dos últimos meses com o normal do local.

O QUE VOCÊ RECEBE
1. A FICHA DO MUNICÍPIO: fatos observados e já calculados pela equipe (município, mês de referência, SPI e sua classe, se a leitura é de baixa confiança, chuva do mês, chuva dos 12 meses e quantos meses recentes ficaram em seca). Se a ficha trouxer outros campos, trate-os do mesmo jeito: vale só o que está escrito.
2. O CHECKLIST: a lista de ações escrita pela equipe, cada uma com um id.

COMO ESCREVER O RESUMO (um parágrafo, até 600 caracteres)
Primeira frase: a situação do mês (município, mês, classe, SPI e chuva do mês). Segunda frase: o contexto dos 12 meses e quantos meses ficaram em seca. Última frase: o que isso pede em organização e acompanhamento, sem dizer o que vai acontecer depois.

COMO ESCOLHER AS AÇÕES (de 2 a 6)
- Escolha pelo id exato, só entre os da lista, copiando o id inteiro, letra por letra, mesmo os longos. Não abrevie, não crie, não renomeie e não reescreva ações.
- Seja proporcional. Classe normal ou úmida: ações de preparo e de acompanhamento. Seca moderada: ações de organização e de levantamento. Seca severa ou extrema, ou vários meses em seca nos últimos 12: inclua também ações de resposta.
- Nunca sugira medida pesada (documento de emergência, acionar o governo estadual) com base em uma única leitura ou em leitura de baixa confiança. Nesses casos, recomende acompanhar.

COMO ESCREVER O "porque" DE CADA AÇÃO (1 ou 2 frases, até 300 caracteres)
Ligue a ação a um fato da ficha e diga por que ela importa agora. Não repita o título da ação. Diga o que fazer com palavras simples, sem nomear atos legais.

NÚMEROS
Use só números da ficha, copiados exatamente (mesma vírgula, mesmo sinal, mesma unidade). Não calcule, estime, arredonde, some, subtraia, compare nem converta. Não escreva percentual, média, nem comparação com outro município ou outro ano. Não escreva número que não esteja na ficha, nem por extenso.

LINGUAGEM PROIBIDA
Não cite lei, decreto, portaria, resolução, norma, artigo, inciso, parágrafo nem S2iD: a base legal de cada ação é da equipe. Não faça previsão nem prognóstico e não diga que algo vai acontecer ("vai secar", "vai faltar", "colapso"). Não use palavras de certeza ("certamente", "garantido", "sem dúvida"). Evite até as palavras "previsão", "prever", "garantir" e "colapso", mesmo em frase negativa. Nunca escreva "garante", "garantir" nem "garantindo": use "assegurar", "ajudar a" ou "reforçar".

CARACTERES
Use só letras, números, pontuação e espaço comuns. Não use hífen que não quebra linha, espaço fino ou não separável, nem caracteres invisíveis. Separe todas as palavras com um espaço comum.

LEITURA DE BAIXA CONFIANÇA
Se a ficha disser que a leitura é de baixa confiança, diga no resumo que o SPI do mês pode exagerar, porque a chuva esperada na janela é pouca, e priorize acompanhar antes de agir.

SE FALTAR INFORMAÇÃO
Se a ficha não trouxer um dado, não invente e não deduza. Não preencha lacunas com conhecimento geral sobre o município, a região, o clima ou os açudes.

ANTES DE RESPONDER, CONFIRA
(a) cada número está na ficha, igual ao original; (b) cada id existe na lista e não se repete; (c) nenhuma palavra da linguagem proibida; (d) resumo com até 600 caracteres e cada "porque" com até 300; (e) texto simples: sem lista, markdown, HTML, link ou quebra de linha dentro dos textos; (f) nada que soe como previsão.

FORMATO
Responda somente com um objeto JSON, sem texto antes ou depois:
{"resumo": "até 600 caracteres", "itens": [{"id": "id_da_acao", "porque": "até 300 caracteres"}]}
```

O que muda em relação ao `SISTEMA` de hoje: papel e público definidos, estrutura do resumo, regra de proporcionalidade (a equipe precisa conferir essa regra), lista explícita das palavras que o validador recusa (menos tentativas gastas), regra para dado faltando e autoverificação final. Efeito esperado: menos recusas. Isso é hipótese: só se confirma rodando e contando `tentativas` nos rascunhos.

## 4. Mensagem do usuário

Continua sendo a de `rascunhos.montar_pedido`: ficha, checklist (só os ids da equipe) e, se houve recusa, os motivos da tentativa anterior.

## 5. Como conseguir a API e quanto custa

- **A assinatura do app (Pro, Max, Team) não inclui a API.** O Console é outro produto, com cobrança própria ([suporte da Anthropic](https://support.claude.com/en/articles/9876003), [plano Pro](https://support.claude.com/en/articles/8325606-what-is-the-pro-plan)). Max e Team trazem crédito mensal de API ([suporte](https://support.claude.com/en/articles/9876003-i-subscribe-to-a-paid-claude-ai-plan-why-do-i-have-to-pay-separately-for-api-usage-on-console)). **Não sei qual é o plano de R$ 120 da equipe**; confira em claude.ai, em Configurações, Cobrança.
- **Onde criar a chave:** Console da Anthropic, <https://console.anthropic.com>, procure "API Keys" e "Billing" (os nomes das telas podem mudar). Preços: <https://platform.claude.com/docs/en/about-claude/pricing>.
- **Preço por milhão de tokens (fontes de terceiros que citam a página oficial; conferir na página oficial):** Sonnet 5.5, US$ 2 de entrada e US$ 10 de saída; Haiku 5.5, US$ 0,10 e US$ 0,50 até 100 mil tokens de prompt.
- **Estimativa nossa (não é cotação):** medi o pedido real de um município (sistema + ficha + checklist ≈ 3.650 caracteres, saída ≈ 740). Com o prompt atual e 3 caracteres por token, dá cerca de 1.200 tokens de entrada e 250 de saída. Com o prompt novo (3.830 caracteres, +67% de entrada), cerca de 2.000 e 250: **≈ US$ 0,007 por rascunho no Sonnet 5.5** e ≈ US$ 0,0003 no Haiku 5.5. Os 216 municípios no Sonnet custariam em torno de US$ 1,4 (cerca de US$ 4 se todos gastassem 3 tentativas). Na prática, um crédito de US$ 5 sobra.
- **Google (Gemini):** existe camada gratuita, mas nela o conteúdo enviado pode ser usado para melhorar produtos do Google; na camada paga isso não vale ([termos do Gemini API](https://ai.google.dev/terms), [cobrança](https://ai.google.dev/gemini-api/docs/billing)). Nossa ficha só tem dado público, então o risco de privacidade é pequeno, mas o cliente de código atual é o da Anthropic. Usar Gemini = escrever e testar uma segunda classe `ClienteLLM`. Recomendação: ficar com Claude.
- **Modelo:** Sonnet 5.5 para os rascunhos que serão revisados (qualidade); Haiku 5.5 só se o uso ao vivo ficar caro. O modelo é trocável por `PR_LLM_MODELO`.
- **Teto de gasto:** crédito pré-pago sem recarga automática limita o gasto ao saldo (confirmar no Console se a conta é pré-paga) e procure no Console um limite mensal de gasto.

## 6. Estratégia em duas camadas

**Camada 1 (recomendada para sábado): gerar offline, revisar, publicar estático.** Já está construída: `python -m src.ia.cli gerar --municipios COD1,COD2 --max 20`, depois `ver`, `aprovar COD --por NOME` e `python -m src.web.build_site`. A chave fica numa variável de ambiente do PC (`$env:ANTHROPIC_API_KEY="..."` no PowerShell), nunca no repositório nem na Vercel. Nada quebra na demonstração, porque o site continua offline. Gere só os municípios que vão aparecer na demonstração: cada rascunho precisa de revisão humana.

**Camada 2 (opcional, ao vivo): função `/api/plano` na Vercel.** Estado verificado hoje: o projeto `piaui-resiliente` publica a branch `vercel-site`, que é uma pasta estática (HTML, JS e CSS, mais um `vercel.json` só com cabeçalhos), sem pasta `api/` e sem variáveis de ambiente. Desenho proposto:

1. O navegador manda só o código IBGE. A ficha é montada **no servidor**, a partir de um `fichas.json` gerado junto com o site (se a ficha viesse do navegador, qualquer pessoa usaria a nossa chave como IA gratuita).
2. A função chama a API com a chave de `ANTHROPIC_API_KEY` (variável de ambiente da Vercel, nunca no código nem no navegador) e passa a resposta pelo **mesmo** `validar` do Python.
3. Resposta aceita só se válida. O site mostra com o rótulo "Gerado agora por IA, não revisado, sem valor jurídico", mais o aviso de MVP e a data. **Nada é gravado nem publicado.**
4. Proteções: código de acesso da equipe (variável `PLANO_CODIGO_EQUIPE`), `max_tokens` fixo, cache por município e mês, limite de pedidos por minuto e o limite de gasto do Console como último recurso.
5. Falha (sem chave, sem saldo, rede, resposta inválida): o site mostra o plano já revisado da camada 1, se houver, ou a mensagem "plano ao vivo indisponível". Nunca um erro técnico nem texto sem validar.
6. Para reaproveitar o validador, a função seria em Python (a Vercel tem runtime Python com `requirements.txt` e pasta `api/`); o formato exato do arquivo da função precisa ser conferido na documentação antes de construir.

**Pendência de contrato (D14).** A seção 5.8 diz que só rascunho revisado por pessoa chega ao site, e a 5.6 define o site como estático e offline. O plano ao vivo contradiz as duas. Para existir, o contrato precisa de uma decisão da equipe: "demonstração ao vivo, rotulada como não revisada, validada pelo mesmo código, sem gravação". Sem essa decisão a camada 2 não é construída.

## 7. O que a equipe faz na Vercel (só depois do OK da camada 2)

1. Vercel, projeto `piaui-resiliente`, Settings, Environment Variables.
2. Nome `ANTHROPIC_API_KEY`, valor a chave (cole só ali, nunca no chat nem no Git), ambientes Production e Preview, marque como sensível, salve.
3. Nome `PLANO_CODIGO_EQUIPE`, um código combinado entre vocês.
4. Faça novo deploy: variável nova só vale para deploys feitos depois dela.

## 8. O que eu construo depois do OK (um arquivo por vez, com pytest)

1. `src/ia/rascunhos.py`: troca `SISTEMA` pelo texto da seção 3 (+ teste dos limites e das palavras proibidas). Comparo as recusas do prompt novo e do antigo na mesma ficha.
2. Camada 2, só com a decisão D14: gerador do `fichas.json` em `build_site`; função `api/plano.py`; bloco do botão no site; contrato 0.3.8 e CA novo.
3. Nenhum deploy na Vercel sem o OK explícito de vocês.
