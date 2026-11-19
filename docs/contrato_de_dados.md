# Contrato de Dados — Piauí Resiliente · Fase 1

Versão 0.3.3 · 2026-11-19 · Status: v0.2.0 aprovada pela equipe em 2026-11-15 (desenho "Do alerta à ação, com prova local"); v0.2.1 incorporou a diretriz de 2026-11-17 "100% dados oficiais e atualizados (zero dados fictícios)"; v0.2.2 fechou D3 (H(0) = q) e o embargo (5 meses); v0.2.3 fechou D9 e D10 (`quality.py`, seção 6); v0.2.4 registrou `manifest.py` (5.3) e o `.gitattributes`; v0.2.5 registrou `build_demo.py` (4.1, 5.3, 8), fechou D11 (município excluído) e fixou o local dos artefatos; **v0.3.0 reabre e fecha D7 com a janela até 2026-09 (dados atualizados, pedido de 2026-11-17), registra a extração em lotes com `--lotes` e abre D12 (migração para o CHIRPS v3) e a pendência dos meses de 2026 na Fase 2; v0.3.1 registra o gerador do site (`src/web/build_site.py`, seção 5.6, Item 11); v0.3.2 acrescenta ao SPI a baixa confiança por pouca chuva esperada (D13, `params.yaml` 0.3.1) e o diagnóstico `src/data/diag_spi.py`; aguarda o OK de leitura; v0.3.3 reorganiza o site em seis páginas offline, com a marca do projeto e o contexto oficial do El Niño (seção 5.6, itens 8 a 11; fontes na seção 2; CA-16)** · Depende de `params.yaml` ≥ 0.3.1 (vigente).

> **Aviso de maturidade:** MVP experimental — em calibração com S2iD/TCE-PI. Nenhum indicador derivado deste dataset tem eficácia operacional comprovada para decisão governamental.

## 1. Objetivo, diferencial e limites de uso

### 1.1 Objetivo e escopo
- **Objetivo da Fase 1:** produzir um dataset mensal por município, com alvos sem vazamento, splits cronológicos e `manifest.json` com SHA-256. Nenhum modelo é treinado aqui; o treino em PyTorch (Fase 2) e a interface (Fase 3) **consomem** este contrato.
- **Extração e treino:** semiárido piauiense, lista fixa de **216 municípios** (215 da Res. CONDEL/SUDENE 150/2021 + Luís Correia, mantido pela regra de transição da Res. 176/2024). Com `escopo.incluir_transicao: false` cai para 215. Versão e data de consulta ficam no manifest.
- **Piloto:** Território Chapada Vale do Rio Itaim (polo: Paulistana, `2207801`), para validação qualitativa, auditoria S2iD e demonstração. Composição declarada pela equipe, **VERIFICAR** na SEPLAN-PI. O piloto participa do treino (split temporal, não espacial). `data/reference/municipios_piloto.csv` só será gerado com a lista confirmada.
- **Fora do escopo deste contrato:** API, MLflow, geração de planos por IA generativa.

### 1.2 Diferencial: "Do alerta à ação, com prova local" (hipóteses a demonstrar, não fatos)
Prever seca por município com um mês de antecedência **já existe** em ferramentas oficiais (Monitor de Secas da ANA, que descreve a condição atual; Índice Integrado de Seca do Cemaden, por município, com projeção para o mês seguinte). Por isso o SPI à frente, sozinho, **não** é o diferencial. Os pilares são:
1. **Antecedência contra o decreto (a prova).** Por município, meses entre o primeiro alerta e o decreto de estiagem/seca do S2iD (seção 4.1). Ineditismo para o Piauí **não verificado**; não afirmar sem checar.
2. **Cartão do município (a ação).** Nível de alerta, tendência com incerteza, reservatório monitorado (se houver) e checklist/rascunho técnico para a Defesa Civil. Na primeira versão o checklist vem de **modelo fixo** escrito pela equipe e ancorado na Lei 12.608/2012, na Lei 12.340/2010 (elementos do plano de contingência) e nas diretrizes do TCE-PI/MIDR. As diretrizes do TCE-PI/MIDR e os artigos exatos a citar são **VERIFICAR** (texto-fonte ainda não obtido): não citar artigo antes de conferir o texto legal. Os campos do município são preenchidos só com dado oficial. O documento é "rascunho técnico", nunca substitui o plano nem tem valor jurídico. IA generativa só entra depois, com revisão humana.
3. **Camada de preparo.** Onde há ou não Plano de Contingência para a seca e/ou COMPDEC (MUNIC/IBGE; ICM/MIDR), como **contexto**, nunca como entrada do modelo. Município sem registro oficial aparece como **"sem informação oficial"**, nunca como "sem plano". Edição e ano de referência sempre visíveis. Disponibilidade por município no PI: **VERIFICAR**.

### 1.3 Limites de uso (obrigatórios na interface e nos materiais)
1. **SPI ≠ colapso de manancial.** O SPI-3 mede falta de chuva em 3 meses (seca meteorológica). Não prova que um açude vai secar. O estoque hídrico é um alvo separado (seção 3).
2. **Horizonte de 1 mês.** O sistema trata variabilidade climática, não projeções de mudança climática de longo prazo. Proibido prometer "segurança frente a mudanças climáticas severas".
3. **Persistência.** O SPI-3 de *t*+1 usa 3 meses de chuva e 2 deles (*t*−1 e *t*) já são conhecidos. Parte do acerto vem só de repetir o passado. Toda afirmação de ganho exige comparação com persistência e climatologia, e com o Cemaden quando houver acesso à série (**VERIFICAR**).
4. **Quem usa.** Município sem Comdec não age sozinho; o usuário natural inclui a Defesa Civil estadual.
5. **Incerteza sempre à vista** (MAE/RMSE ou probabilidades) e aviso de MVP em toda tela, API e relatório.

### 1.4 Regra de transparência: zero dados fictícios (diretriz de 2026-11-17; substitui a regra "dados ilustrativos" da v0.2.0)
1. **Rastreabilidade.** Todo valor exibido (interface, API, relatório, slide, demonstração) vem de uma fonte oficial da seção 2, passou na validação de schema e consta no `manifest.json` com SHA-256.
2. **Nada simulado.** Não existe dado ilustrativo, de exemplo, aleatório ou de preenchimento em nenhuma camada (dados, artefatos, interface, slides). A marca "DADOS ILUSTRATIVOS" da v0.2.0 foi **revogada**: o que não é validado não aparece com valor.
3. **Camada sem dado oficial validado** é omitida ou exibida como **"Indisponível — sem dado oficial validado"**, com o motivo (ex.: "sem reservatório monitorado pela ANA neste município").
4. **Previsão do modelo** só é exibida depois que a avaliação auditada (CA-13) estiver registrada, sempre com MAE/RMSE ou probabilidade e aviso de MVP. Antes disso o painel diz "Indisponível — modelo em calibração".
5. **Fixtures sintéticas** existem **somente em `tests/`**, porque testar código exige entradas controladas. Nunca em `data/`, `artifacts/`, interface ou apresentação.
6. **Limitação não é motivo de omissão.** Dado oficial com limitação conhecida (SPI de treino in-sample, CHIRPS como produto de satélite e estações) é exibido com a limitação declarada.
7. **"Atualizado" tem data.** Toda camada mostra fonte, período coberto e data de extração (do manifest). A janela do dataset vai de 1981-01 a 2026-09 (`janela.fim`, D7), o último mês com CHIRPS final no GEE. O teste congelado continua terminando em 2025-12.

### 1.5 Níveis da demonstração (Campus Party, novembro de 2026; pitch com plataforma navegável; **data exata a registrar, D8**)
| Nível | Conteúdo | Depende de | Dados |
|---|---|---|---|
| 1 | Mapa interativo dos 216 municípios, máquina do tempo 1981–2025 e mapa de calor município × mês, com SPI **observado** | Itens 2c a 5 | reais |
| 2 | Histograma de antecedência contra os decretos S2iD (e camada de preparo MUNIC, se validada) | Nível 1 + Item 8 (+ MUNIC) | reais (validação); teste só uma vez (seção 4.1) |
| 3 | Previsão com incerteza e cartão completo | Fase 2 (PyTorch) | reais **só se a avaliação auditada existir**; senão, "Indisponível" |

Regra de corte: o que não estiver validado na data da apresentação aparece como "Indisponível", nunca simulado. Ordem de prioridade: Nível 1, Nível 2, camada MUNIC, cartão (parcial), Nível 3.

### 1.6 Requisitos da interface (minimalista, tema claro e escuro)
1. **Mapa coroplético** dos 216 municípios (risco/SPI) com filtro rápido **"em alerta de seca + sem Plano de Contingência (MUNIC/IBGE)"**. "Sem plano" só vale para município com registro oficial negativo; municípios "sem informação oficial" ficam em categoria própria.
2. **Máquina do tempo 1981–2025** (`demo_observado`) com a prova de antecedência: quantos meses antes do decreto oficial do S2iD o alerta vermelho disparou (N de municípios e decretos visível; valores negativos reportados).
3. **Cartão do município:** série histórica, previsão com faixa de incerteza (Nível 3), nível do reservatório quando monitorado, checklist e rascunho técnico para a Defesa Civil.
4. Aviso de MVP em toda tela; cada painel mostra fonte, período e data de extração.
5. A interface lê **apenas** artefatos listados no manifest e confere o SHA-256 na carga.
6. **Páginas e marca (v0.3.3):** seis páginas (Início, Monitor, Municípios, El Niño, Método e O projeto), com a paleta e a gota da folha "Construção" (5.6, item 11). A página inicial abre com o mapa dos municípios no último mês em que a maioria das leituras é confiável e traz o contexto oficial do El Niño, sempre com instituição, data e link, e nunca como previsão do nosso modelo (5.6, itens 9 e 10).

## 2. Fontes e decisões de referência

| Item | Decisão | Status | Observação |
|---|---|---|---|
| Malha municipal | IBGE 2022, `PI_Municipios_2022.zip` (224 municípios, SIRGAS 2000, EPSG:4674) | CONFIRMADO | Lida de `data/reference/fonte/`, sem rede. Reprojetada para EPSG:4326 só para o GEE. `geobr` descartado. SHA-256 no selo (5.0). |
| Semiárido | Anexo da Res. CONDEL/SUDENE 150/2021 (1.427; 215 no PI) + transição Res. 176/2024 (Luís Correia, `2205706`) | **RESOLVIDO PARA O PI (D2)**, com 2 resíduos | Luís Correia vem de fonte secundária (MeioNews): confirmar no PDF da Res. 176/2024. Vigência dos 50 municípios após 03/01/2025 não confirmada. Não bloqueiam. |
| Precipitação | CHIRPS diário via GEE: `UCSB-CHG/CHIRPS/DAILY`, banda `precipitation` | **CONFIRMADO (sonda 2026-11-15)** | 16.709 imagens de 1981-01-01 a 2026-09-30, calendário completo; EPSG:4326; `nominalScale` 5565,97 m (`escala_m: 5566`); mínimo diário 0 (sem negativo ou sentinela na amostra); Luís Correia 32/32 pixels válidos; polo 66 pixels; menor município da amostra (`2209450`) 2 pixels; grade nativa e escala nominal diferem ≤ ~0,08%. A unidade (mm/dia) vem da documentação do catálogo (as propriedades da imagem não trazem unidade); valores plausíveis (polo: 223 mm em 2020-03 e ~2 mm em 2020-09). Amostra pequena: ponderação de borda **não provada** (`reducer_espacial: media`). |
| Extração do CHIRPS | `src/data/extract_chirps.py` (`sonda`, `rodar`) e RAW imutável em `src/data/chirps_raw.py` | IMPLEMENTADO e testado offline; **execução real pendente** | `rodar --lote-teste` não retornou em ~23 min (diagnóstico em andamento; ver seção 11). Nenhum RAW real gravado ainda. |
| Acesso ao GEE | Projeto Google Cloud registrado como **não comercial**, nível Comunidade (150 h de EECU), categoria Adaptação | CONFIRMADO | ID só em `GEE_PROJECT`. Reavaliar o enquadramento antes de uso operacional com órgãos públicos. |
| Cobertura do solo | MapBiomas **Coleção 10** | DECIDIDO (D1) | Uma única coleção em todos os anos; defasagem de 1 ano. |
| Desastres | S2iD/MIDR, COBRADE `1.4.1.1.0` (Estiagem) e `1.4.1.2.0` (Seca) | CONFIRMADO | Formato dos códigos e colunas: **VERIFICAR** (Item 8); **bloqueia o Nível 2**. Uso: auditoria e prova de antecedência; nunca feature. |
| Estações | INMET (BDMEP) | VERIFICAR | Validação cruzada do CHIRPS (Item 6). Cobertura no semiárido desconhecida. |
| Estoque de reservatórios | ANA/SAR, módulo Nordeste e Semiárido | **VERIFICAR (sonda)** | Cobertura fraca no PI (só parte dos reservatórios cadastrada; ANA monitora reservatórios de domínio da União). Conjunto monitorado parece variar entre boletins. O boletim traz o texto "seco" no lugar do volume. Endpoint e início da série: **VERIFICAR**. Município sem reservatório monitorado: a interface informa que não há dado (item 1.4.3). |
| Água por satélite | JRC `JRC/GSW1_4/MonthlyHistory` (1984-03 a 2021-12) | VERIFICAR | Termina antes do fim do teste (2025). Fase posterior. |
| Preparo municipal | MUNIC/IBGE (variável **MGRD05**, "O município possui Plano de Contingência e/ou Preservação para a seca?", edição 2020) e ICM/MIDR (que pondera essa variável com peso 2,0) | **VERIFICAR** | Contexto, não feature. A confirmar: edição e ano mais recentes disponíveis, variável para existência de COMPDEC, disponibilidade por `cod_ibge` no PI. Edição/ano de referência exibidos na interface. |
| Diretrizes e normas do checklist | Lei 12.608/2012, Lei 12.340/2010, diretrizes TCE-PI/MIDR | Leis CONFIRMADAS; diretrizes **VERIFICAR** | Documento do TCE-PI ainda não obtido. Não citar artigos antes de conferir o texto. |
| Contexto do El Niño (site, v0.3.3) | CPC/NOAA, *ENSO Diagnostic Discussion* (emitida em 2026-11-12; próxima em 2026-12-10) e INMET, *Painel El Niño* (boletim mensal nº 5, 2026-10-29), transcritos em `data/reference/contexto_enso.json` | **CONFIRMADO (páginas oficiais abertas em 2026-11-18)** | Textos em paráfrase e números copiados da fonte, com instituição, data e link em cada item. Fora do manifest, porque não é entrada de modelo; validado por `src/data/contexto_enso.py`. O site avisa quando a data da próxima discussão passa. Contexto, nunca previsão do nosso modelo. |
| Série histórica do ONI/RONI | CPC/NOAA | **VERIFICAR** | O ambiente de desenvolvimento não alcança o CPC. Sem a série, os anos de El Niño não são marcados nos gráficos, e a página El Niño diz isso. Baixar o arquivo oficial e registrar URL, data e SHA-256 antes de usar. |
| Previsores de chuva (ENSO, Atlântico) | Possíveis entradas para ganhar além da persistência | VERIFICAR | Fonte e forma a definir antes da Fase 2. O contexto do El Niño do site (linha acima) é só informação ao leitor, não entrada de modelo. |
| Benchmarks | Monitor de Secas (ANA) e Índice Integrado de Seca (Cemaden) | VERIFICAR | Acesso a séries históricas por município a confirmar. |

## 3. Unidade de amostra e alvos
- **Linha:** (`cod_ibge`, `mes_origem` = *t*). Toda feature usa apenas dados ≤ *t*.
- **`mes_alvo`** = *t* + *h* (*h* = 1). **O split é definido por `mes_alvo`.**

### 3.1 Alvos (modelo multitarefa na Fase 2: tronco compartilhado e uma cabeça por alvo)
| Cabeça | Alvo | Dados | Cobertura | Status |
|---|---|---|---|---|
| A | `alvo_spi3_h1` = SPI-3 em *t*+1 (seca meteorológica) | CHIRPS | 216 municípios | planejada |
| A2 | `alvo_spi6_h1` = SPI-6 em *t*+1 (seca mais longa) | CHIRPS | 216 | SPI-6 calculado em `spi.py`; cabeça na Fase 2; sem dado novo |
| B | Estoque de reservatórios (% de volume ou classe "crítico") | ANA/SAR | só municípios com reservatório monitorado e cobertura validada | **VERIFICAR** |
| B' | Área de espelho d'água (substituto por satélite) | JRC + extensão | todos | **VERIFICAR**; fase posterior |
| C | Decreto de estiagem/seca (S2iD) | S2iD | semiárido | fase posterior |

Regras do multitarefa: perda **com máscara** (sem rótulo, não entra na perda; nunca preencher rótulo com 0); mesmos splits e embargos para todas as cabeças; baselines e métricas **por cabeça**; teste congelado vale para todas; ablação decide se cada cabeça fica. A cabeça B só vale onde há cobertura oficial validada; fora dela a interface mostra o SPI e informa que não há estimativa de estoque. Limiar de "colapso" (ex.: volume < 10%) é decisão da equipe (**D6**).

**Baselines (somente comparativos; o núcleo é a rede PyTorch treinada do zero):** persistência, climatologia, Ridge e XGBoost/Random Forest, por cabeça.

## 4. Splits

| Bloco | `mes_alvo` | Meses |
|---|---|---|
| treino | ≈1982-04 a 2015-12 (início depende de `max_lag_meses` e, com SPI-6, desloca cerca de 3 meses; `splits.py` rotula como treino todo `mes_alvo` da janela até 2015-12 e o `features.py` descarta as linhas sem histórico) | ~405 |
| embargo | 2016-01 a 2016-05 | 5 |
| validação | 2016-06 a 2020-12 | 55 |
| embargo | 2021-01 a 2021-05 | 5 |
| teste congelado | 2021-06 a 2025-12 | 55 |
| pós-teste (só observado) | 2026-01 a 2026-09 | 9 |

Regras:
1. Mesmos cortes para todos os municípios. Nunca há shuffle.
2. Embargo de **5 meses para todas as cabeças** (mínimo sem sobreposição das janelas do alvo = maior janela do SPI − 1: 2 para o SPI-3 e 5 para o SPI-6). `config.py` e `splits.py` recusam embargo menor.
3. Os 9 meses de 2026 (`janela.fim` > `split.teste_fim`) entram no RAW, na qualidade, no SPI e no `demo_observado`, mas **não** pertencem a treino, validação nem teste. `splits.rotular` recusa `mes_alvo` depois de `teste_fim`; a Fase 2 deve filtrar esses meses ou rotulá-los como pós-teste (pendência na seção 11). Os parâmetros do SPI continuam ajustados só até 2015-12.
4. O teste fica em arquivo próprio, com hash no manifest, e só é lido com `split.teste_liberado: true` (avaliação final da Fase 2).
5. A EDA registra, por bloco, linhas e % de `spi3 ≤ -1`. Se o teste tiver poucos eventos de seca, as métricas dele são declaradas não conclusivas.
6. Scalers, parâmetros Gama e qualquer estatística são ajustados **somente** no treino (≤ 2015-12).
7. `src/data/splits.py` rotula cada `mes_alvo` (`train`, `val`, `test`, `embargo`), recusa meses fora da janela e recusa a leitura do bloco `test` com `teste_liberado: false`.

### 4.1 Demonstração e antecedência sem violar o teste congelado
1. **Arquivo de demonstração próprio** (`demo_observado`): SPI observado de todo o período 1981–2025, sem alvos, sem split e sem métricas, gerado com os mesmos parâmetros Gama do treino. A demonstração **não lê** o arquivo do teste congelado. O SPI de treino é in-sample (limitação registrada e exibida). Gerado por `src/data/build_demo.py` (`python -m src.data.build_demo`) a partir do RAW do CHIRPS, na ordem: qualidade (seção 6) → SPI sobre a série válida (seção 7, D9) → regra D11 → gravação imutável → manifest. Não importa `splits.rotular`, `splits.bloco` nem `exigir_teste_liberado` (CA-12, testado).
   - **D11 (município excluído por D10):** permanece nos 216, com `spi3`/`spi6` = NaN, `spiK_invalid = True` e `spiK_baixa_conf = False` em todos os meses. A interface mostra "sem dado". A regra é aplicada marcando `valido = False` nas células de `spi_gamma_params` do município, de modo que `calcular_spi(serie, cfg, parametros=spi_gamma_params)` reproduz o `demo_observado` exatamente. Cobertura, qualidade e climatologia do município continuam publicadas (auditoria).
2. **Antecedência** = meses entre o primeiro mês em que o SPI cruza o limiar de alerta e o mês do decreto, por município com ≥ 1 decreto COBRADE `1.4.1.1.0`/`1.4.1.2.0`. Valores negativos (alerta depois do decreto) são **reportados**, nunca descartados.
3. O limiar de alerta é fixado e documentado na **validação** (2016-06 a 2020-12). A antecedência no período de teste é calculada **uma única vez**, na avaliação final, com `teste_liberado: true`.
4. O histograma exibe N (municípios e decretos). Poucos decretos por município: análise estatística só no semiárido inteiro; qualitativa no piloto.

## 5. Schemas

### 5.0 Referência territorial (`data/reference/`) — Item 2b
Gerada por `src/data/reference.py` a partir de `data/reference/fonte/`. Escrita atômica, imutável, com selo SHA-256.

**Entradas (`data/reference/fonte/`):**
| Arquivo | Conteúdo |
|---|---|
| `PI_Municipios_2022.zip` | Malha IBGE 2022 do PI: `CD_MUN`, `NM_MUN`, `SIGLA_UF`, `AREA_KM2`, EPSG:4674, 224 polígonos válidos |
| `semiarido_res150_2021.csv` | Anexo da Res. 150/2021: `cod_ibge,uf,nome`, 1.427 linhas (215 do PI), UTF-8, LF |
| `semiarido_transicao_res176_2024.csv` | Transição: `cod_ibge,uf,nome,fonte` (1 linha: `2205706`, PI, Luís Correia) |

**Saídas:**
| Arquivo | Linhas | Schema |
|---|---|---|
| `municipios_semiarido.csv` (`escopo.arquivo_semiarido`) | 216 (215 se `incluir_transicao: false`) | `cod_ibge:str(7)`, `nome:str`, `uf:str` ("PI"), `situacao:str` (`res150`\|`transicao`), `area_km2:float` (3 casas). Só municípios no escopo, ordenados por `cod_ibge`, UTF-8 sem BOM, LF. **Ler `cod_ibge` como texto.** |
| `malha_pi_2022.parquet` | 224 | GeoParquet EPSG:4674: `cod_ibge:str(7)`, `nome`, `uf`, `area_km2:float64`, `geometry`, `situacao` (`res150`\|`transicao`\|`fora`), `no_escopo:bool` |
| `malha_pi_2022_gee.geojson` | 216 | FeatureCollection EPSG:4326; propriedades `cod_ibge:str`, `nome`, `situacao`. Simplificada com `escopo.simplificacao_graus` (0,001° ≈ 110 m), topologia preservada. Medido: variação de área máx. 0,38%; 140.971 → 17.997 vértices; ~712 KB. |
| `referencia.sha256.json` | — | Selo do conjunto: SHA-256 de saídas e entradas, parâmetros, contagens, métricas da simplificação, `conteudo_sha256` canônico e versões das bibliotecas. |

**Classificação dos 224 municípios do PI:** 215 `res150`, 1 `transicao`, 8 `fora` (Antônio Almeida, Baixa Grande do Ribeiro, Cajueiro da Praia, Guadalupe, Marcos Parente, Porto Alegre do Piauí, Ribeiro Gonçalves, Santa Filomena).

**Garantias verificadas por código:** códigos de 7 dígitos únicos com prefixo 22; todo código das listas existe na malha; nomes conferem; transição disjunta do anexo; polo `2207801` no escopo; estado parcial é erro; entradas ou parâmetros diferentes levantam `RefImutavelError`; arquivo alterado levanta `RefIntegridadeError`.

### 5.1 `raw_chirps_mensal` (RAW, imutável) — **implementado (`src/data/chirps_raw.py`, v0.2.1)**
`cod_ibge:str(7)`, `mes:datetime64[ns]` (dia 1), `precip_mm:float32`, `precip_max_dia_mm:float32`, `dias_chuva:float32`, `dias_validos:int16`, `dias_no_mes:int16`, `pixels_validos_frac:float32`.

Definições (agregação **por pixel e depois média municipal**, na grade nativa do CHIRPS; decisões **D4** e **D5** fechadas):
- `precip_mm`: soma mensal por pixel, média municipal.
- `precip_max_dia_mm` (**D5**): média municipal do **máximo diário de cada pixel** no mês. Não é o máximo diário da média municipal (mais suavizado). Sinal de chuva extrema.
- `dias_chuva` (**D5**): média municipal do número de dias com precipitação ≥ `fontes.chirps.limiar_dia_chuva_mm` (**1,0 mm**) por pixel. **float32**, porque a média entre pixels é fracionária. Sinal de veranico quando baixo.
- `dias_validos` (**D4**): **mínimo** entre os pixels do município do nº de dias com dado (a máscara `selfMask` faz o mínimo valer só entre pixels com dado).
- `pixels_validos_frac`: pixels com dado / pixels do município (0 a 1).

Invariantes validadas em `validar` (violação = `RawSchemaError`, lote não é gravado; todas as violações são listadas de uma vez):
- `precip_mm`, `precip_max_dia_mm` e `dias_chuva` são **NaN juntos ou nenhum**. `precip_mm` preenchido exige `dias_validos` > 0 e `pixels_validos_frac` > 0, e fica em [0, 2000] mm (teto de sanidade; acima disso é erro de unidade ou de soma).
- `precip_max_dia_mm` ≥ 0 e ≤ `precip_mm` (tolerância de float32 `1e-3`); `dias_chuva` em [0, `dias_no_mes`]; `precip_mm = 0` implica `precip_max_dia_mm = 0` e `dias_chuva = 0`.
- `dias_validos` em [0, `dias_no_mes`]; `pixels_validos_frac` em [0, 1]; `dias_no_mes` igual ao calendário; `mes` é o dia 1 às 00:00.
- Códigos de 7 dígitos; grade **exatamente** município × mês do lote (sem linha faltando, sobrando ou duplicada).

Convenção do extrator (`montar_linhas`, não é regra da `validar`): município sem pixel válido, com propriedade nula do GEE ou ausente do retorno vira trio NaN, `pixels_validos_frac = 0` e `dias_validos = 0`. Nunca 0 mm por imputação.

Imutabilidade e execução:
- Lotes em Parquet (escrita atômica) com selo `.sha256.json`. Lote idêntico é "pulado"; conteúdo diferente levanta erro; estado parcial é erro. O selo guarda também `meta` (proveniência: coleção, banda, limiar, versão do `ee`, data UTC e nº de imagens do GEE por mês); `meta` **não** entra no hash nem na comparação.
- 216 municípios com no máximo 4.000 linhas por lote (`gee.max_linhas_por_lote`) ⇒ 18 meses por lote ⇒ **31 lotes** para 1981-01 a 2026-09 (549 meses): 30 de 18 meses (1981-01 a 2025-12) e o último com 9 meses (2026-01 a 2026-09). Os 30 primeiros não dependem da janela.
- `rodar` confere por SHA-256 os lotes já gravados sem chamar o GEE (retomável) e repete cada chamada ao GEE até 3 vezes (esperas de 5 s e 10 s), com limite de 120 s por tentativa (uma chamada presa vira `TimeoutError` e é repetida). Imprime uma linha por mês e uma por lote. `--lotes A:B` extrai só a faixa (1 a 31, inclusiva), para rodar em vários terminais; com faixa ou `--lote-teste` a conferência da janela inteira não é feita, e o `rodar` sem argumentos a faz no fim. Conferência final: `ler_raw` sobre os 549 meses.
- Os 3 `reduceRegions` por mês (média, mínimo, contagem) e o `size()` das imagens foram medidos no GEE real: cerca de 7–8 s por `reduceRegions` com 216 polígonos, 24–27 s por mês e cerca de 449 s por lote de 18 meses (lote 1, 2026-11-17).

### 5.2 `curated_mensal`
| Coluna | dtype | Descrição |
|---|---|---|
| `cod_ibge` | str(7) | código IBGE |
| `mes_origem`, `mes_alvo` | datetime64[ns] | dia 1 |
| `precip_mm`, `precip_max_dia_mm`, `dias_chuva` | float32 | após imputação causal (seção 6); NaN onde a lacuna passa do limite |
| `cobertura_dias`, `cobertura_pixels` | float32 | 0–1 |
| `precip_mm_missing` | bool | imputado ou ausente |
| `spi3`, `spi6` | float32 | SPI em *t*, calculado sobre a série **válida**, sem imputação (D9) |
| `spi3_invalid`, `spi3_baixa_conf`, `spi6_invalid`, `spi6_baixa_conf` | bool | ver seção 7 |
| `alvo_spi3_h1`, `alvo_spi6_h1` | float32 | SPI em *t*+1 |
| `split` | category | `train`, `val`, `test`, `embargo` |
| `mb_*` | float32 | MapBiomas do ano Y-1 (nomes no Item 7) |
| features de lag | float32 | Item 9 |

### 5.3 Tabelas auxiliares
- **`cobertura_mensal`** (relatório município × mês, gerado antes de imputar): `cod_ibge:str`, `mes:datetime64[ns]`, `cobertura_dias:float32` (`dias_validos / dias_no_mes`), `cobertura_pixels:float32`, `valido:bool`, `motivo:str` (critérios reprovados entre `sem_dado`, `cobertura_dias`, `cobertura_pixels`, separados por `+`; vazio se válido).
- **`qualidade_municipio`:** `cod_ibge:str`, `frac_meses_validos:float32` (meses válidos de `janela.inicio` a `split.treino_fim` ÷ meses desse intervalo, D10), `excluded:bool`, `motivo:str`.
- **`audit_s2id`** (nunca entra como feature): `cod_ibge:str`, `data_decreto:datetime64[ns]`, `cobrade:str`, `tipo:str`. Nomes de coluna **VERIFICAR** contra o arquivo real.
- **`demo_observado`:** `cod_ibge:str`, `mes:datetime64[ns]`, `spi3:float32`, `spi3_invalid:bool`, `spi3_baixa_conf:bool` (e equivalentes do SPI-6). Sem alvo, sem split. Grade completa município × mês da janela, ordenada por (`cod_ibge`, `mes`).
- **Local dos artefatos gerados por `build_demo`** (pastas de `caminhos` do `params.yaml`; Parquet, escrita atômica, imutáveis: existente e idêntico = `ja_existente`, diferente = erro, sem gravar nenhum dos que faltam):

  | Artefato | Arquivo |
  |---|---|
  | `demo_observado` | `data/curated/demo_observado.parquet` |
  | `cobertura_mensal` | `data/processed/cobertura_mensal.parquet` |
  | `qualidade_municipio` | `data/processed/qualidade_municipio.parquet` |
  | `spi_gamma_params` | `artifacts/spi_gamma_params.parquet` |
  | `climatologia_precip` | `artifacts/climatologia_precip.parquet` |

  O manifest do demo lista esses cinco, os lotes do RAW (`raw_chirps/<lote>.parquet`) e a referência territorial (`referencia/<arquivo>`: CSV do escopo, malha em Parquet, GeoJSON do GEE e selo), e traz em `extras.demo`: `n_municipios`, `n_meses`, `n_excluidos`, `escalas_spi`, `spi_calibrado_ate` e `spi_treino_in_sample = true`. O resumo impresso pela CLI repete o aviso de maturidade.
- **`manifest.json`** (`src/data/manifest.py`; destino em `manifest.arquivo`). Campos: `versao_manifest` (1), `algoritmo_hash`, `gerado_em_utc`, `maturity` (aviso de MVP experimental, obrigatório), `projeto` (nome, equipe, `versao_params`, `seed`), `params_sha256`, `dataset_sha256`, `ambiente` (Python e versões das bibliotecas), `janela`, `split` (cortes, `embargo_meses` e `teste_liberado`, de `splits.cortes`), `alvo` (escala, horizonte e `janelas_spi`), `escopo` (modo, `incluir_transicao`, fonte do semiárido, ano da malha), `referencia` (`conteudo_sha256`, contagens, entradas e parâmetros, lidos do selo `referencia.sha256.json`), `fontes` (CHIRPS, MapBiomas, S2iD), `arquivos`, e opcionalmente `linhas_por_bloco` e `extras` (ex.: edição e ano da camada de preparo). Em `arquivos`, cada nome lógico traz `caminho` (relativo à raiz, em POSIX) e `sha256`; Parquet traz também `linhas`, `colunas` e `dtypes` (tipos Arrow).
  - **Hash:** texto (`.py .md .txt .toml .yaml .yml .json .geojson .csv .html .css .js`, a mesma lista do `.gitattributes`) é calculado sobre o conteúdo normalizado em LF; binários, sobre os bytes brutos. `dataset_sha256` = hash dos pares (nome, SHA-256) mais `params_sha256`.
  - **Segurança dos caminhos:** caminho absoluto, `..`, barra invertida, unidade do Windows e qualquer saída da raiz (inclusive por link simbólico) são recusados, na geração e na leitura. O manifest não lista a si mesmo.
  - **Imutabilidade:** mesmo `dataset_sha256` = "ja_existente" (não reescreve); outro conjunto exige `substituir=True`.
  - **Verificação:** `verificar_manifest` recalcula todos os hashes (e o `params.yaml`, se pedido) e lista todos os problemas; `caminho_verificado(nome)` confere só o arquivo pedido e devolve o caminho. A interface abre dados somente por `caminho_verificado` (seção 1.6, item 5).

### 5.4 Estoque de reservatórios (cabeça B) — schema **a definir após a sonda do SAR**
Não definir colunas antes de ver o arquivo/API real. Pontos já conhecidos: valor textual "seco" deve virar evento "estoque zero", não dado faltante; o conjunto de reservatórios muda no tempo (painel desbalanceado); a máscara de perda cobre municípios sem reservatório monitorado.

### 5.5 Camada de preparo municipal (MUNIC/ICM) — schema **a definir após ver o arquivo real**
Regras já fixadas: chave `cod_ibge` como texto; três estados por município (`sim`, `nao`, `sem_informacao`); carrega edição e ano de referência; nunca é feature; nunca é preenchida por inferência (ausência de registro não vira "não").

### 5.6 Site (Item 11) — **implementado (`src/web/build_site.py`, `paginas.py`, `marca.py`, `destaques.py`, modelos em `src/web/paginas/` e `src/web/assets/`; v0.3.3)**
Uma pasta de páginas estáticas que funciona offline, inclusive abrindo `index.html` direto do disco (sem CDN, sem `fetch`, sem módulos de script; links para fora só como `<a>`), gerada por `python -m src.web.build_site` (padrão: pasta `site/`; `--saida` escolhe outra **pasta**, e um caminho terminado em `.html` é recusado). Tudo é montado na memória e só então gravado; os arquivos saem com permissão 644. Até a v0.3.2 o site era um único HTML (`site/piaui_resiliente.html`).
1. **Entradas:** só o que o manifest lista, depois de `verificar_manifest` (SHA-256 de todos os arquivos e do `params.yaml`): `demo_observado`, `qualidade_municipio`, lotes `raw_chirps/*` (chuva mensal e data de extração do selo) e `referencia/malha_pi_2022_gee.geojson`. Qualquer divergência interrompe a geração e nada é gravado.
2. **Grade exata:** município × mês da janela inteira, sem linha faltando, sobrando ou repetida; `qualidade_municipio` e a malha precisam ter os mesmos municípios.
3. **Pacote de dados (`assets/dados.js`, uma atribuição `window.PR_DADOS=...;` de JSON puro, sem `NaN`):** SPI x 100 em `int16`, chuva x 10 em `int16` (sentinela `-32768` = sem dado; estouro é erro), classe em `uint8` (0 a 6; `255` = sem dado), marca de baixa confiança em `uint8`; tudo em base64 little-endian. Municípios ordenados por `cod_ibge`; a matriz do tempo usa a ordem norte → sul. Além das matrizes, o pacote leva `n_municipios`, `abertura`, `pct_baixa_ultimo`, `maiores_secas`, `clima` e `enso` (itens 9 e 10).
4. **Classes (McKee et al. 1993, as mesmas da OMM):** seca extrema ≤ −2,0; severa −1,99 a −1,5; moderada −1,49 a −1,0; próximo do normal −0,99 a 0,99; úmido moderado 1,0 a 1,49; muito úmido 1,5 a 1,99; extremamente úmido ≥ 2,0. **"Em seca" = SPI ≤ −1,0.** A classe é calculada sobre o número que a tela mostra (duas casas), para nunca aparecer "−1,00" como "próximo do normal".
5. **Conteúdo do Monitor (`monitor.html`):** mapa dos 216 municípios no mês escolhido (escalas SPI-3 e SPI-6), contagem por classe, municípios mais secos do mês, ficha do município (série do SPI, chuva mensal, meses em seca, mês mais seco), matriz município × mês com % de municípios em seca, reprodução do tempo, tabela de todos os municípios e tema claro e escuro. **Baixa confiança** (`spiK_baixa_conf`, seção 7, item 5) é sinalizada em toda tela: pontilhado no mapa (com a contagem do mês), célula esmaecida na matriz, aviso no balão e na ficha (com o total de meses). Os motivos são os da seção 7, item 5, e o rodapé declara a regra exata (lida do pacote: `regra_baixa`). Esses municípios continuam nas contagens do mapa e nas classes; na ficha do município, "meses em seca" e "mês mais seco" contam só meses confiáveis, e o resumo do mês informa quantos dos municípios em seca têm baixa confiança.
6. **Transparência (1.4 e 1.6):** aviso de MVP no topo; rodapé com fonte, período, data de extração, `dataset_sha256`, data da verificação e limitações (SPI de treino in-sample; meses depois de 2025-12 só observados; município sem cobertura aparece como "sem dado"; SPI mede falta de chuva e não prova colapso de manancial). **Previsão do próximo mês, antecedência em relação ao S2iD e Plano de Contingência (MUNIC) aparecem como "Indisponível"** até haver dado oficial validado e avaliação auditada.
7. **Verificado em testes (`tests/test_build_site.py`, `test_paginas.py`, `test_marca.py`, `test_destaques.py`, `test_contexto_enso.py`):** limites das classes, codificação e estouro, projeção da malha (Polygon, MultiPolygon com furo, geometria inválida), grade, pacote decodificado contra o `demo_observado` e o RAW, município excluído, páginas sem recurso externo (nem `fetch`, nem módulos), links e ids do script, contraste das cores do CSS nos dois temas, cores das classes iguais às do pacote, arredondamento igual em Python e JavaScript, geometria da gota, sintaxe do JavaScript (`node --check`, quando o Node existe), ponta a ponta com manifest e adulteração de cada arquivo (nada é gravado).
8. **Páginas:** `index.html` (objetivo, mapa no mês de abertura, números do mês, cinco municípios mais secos, bloco do El Niño e maiores secas), `monitor.html` (ferramenta completa do item 5), `municipios.html` (tabela de todos os municípios em um mês, com filtros e ordenação; a região rolável é acessível por teclado), `el-nino.html` (contexto oficial e as estações chuvosas observadas), `metodo.html` (método, baixa confiança, o que ainda não existe e limites) e `sobre.html` (projeto, estado de cada entrega e marca). Os links profundos usam `#m=<cod_ibge>&t=AAAA-MM&k=<escala>`. O tema (claro ou escuro) fica no armazenamento do navegador, dentro de `try`, e é só conveniência.
9. **Mês de abertura e destaques (`destaques.py`):** o site abre no último mês em que pelo menos metade dos municípios com SPI tem leitura confiável, e a regra está escrita na página (com julho de 2026 como abertura, setembro tem cerca de 86% de leituras de baixa confiança). "Maiores secas" são os meses com maior fração de municípios em seca entre as leituras confiáveis, um por episódio (separação de 12 meses), só entre meses em que a maioria das leituras é confiável. A climatologia mensal e a estação chuvosa (quatro meses seguidos; o ano da estação é o do início; só estações completas) usam, na média de referência, apenas a chuva até `split.treino_fim`; município excluído não entra. Há teste de vazamento: mudar a chuva depois do treino não muda a média de referência. Texto da página e rótulo do gráfico arredondam igual (meio para longe do zero, em Python e em JavaScript, com teste cruzado em Node).
10. **Contexto do El Niño (`data/reference/contexto_enso.json`, lido por `src/data/contexto_enso.py`):** fontes (instituição, título, data de emissão, link e próxima atualização), situação (alerta, Niño-3.4 e probabilidades copiadas do CPC), leituras regionais do INMET e limites. O esquema é validado (campos extras proibidos, links https, referências cruzadas, datas e percentuais de 0 a 100) e o texto do arquivo é escapado ao virar HTML. Sem o arquivo, o site mostra "Indisponível". Fica fora do manifest e de qualquer modelo. Quando passa a data da próxima discussão do CPC, a página avisa para conferir a fonte. Todo bloco traz "Contexto, não previsão do nosso modelo".
11. **Marca (`marca.py`, folha "Construção"):** petróleo `#0F6E7A` (o que há), terracota `#B4472A` (o que falta), tinta `#12262B` e creme `#F6F1E8`; letras Outfit (wordmark: "Piauí" 700 e "Resiliente" 300) e Atkinson Hyperlegible Next (texto), ambas SIL OFL e embutidas no CSS. A gota é desenhada pela geometria da folha (círculo de raio r, tangentes a 60°, traço r÷4, vazio de um traço, nível na linha do centro) e as proporções são testadas. As classes do SPI usam duas rampas, terracota para falta de chuva e petróleo para excesso, validadas por contraste nos dois temas; terracota só aparece para falta de chuva.

### 5.7 Diagnóstico do SPI (`src/data/diag_spi.py`) — **implementado (v0.3.2)**
1. **Uso:** `python -m src.data.diag_spi [--mes AAAA-MM] [--escala 3|6]` (padrão: `janela.fim` e `alvo.escala_spi_meses`). Somente leitura: confere o hash de todos os arquivos do manifest, lê só o que ele lista e não grava nada. Exceção deliberada: o hash do `params.yaml` não é exigido, para permitir testar um limiar novo (`spi.media_min_mm_por_mes`) antes de regenerar o demo; nesse caso o relatório termina com uma `NOTA`, e o diagnóstico recusa se o `params.yaml` mudou `split.treino_fim`, `janela.inicio`, `janela.fim` ou `spi.janelas_meses` (campos que mudam o ajuste da Gama). O site e o `build_demo` continuam exigindo o hash. Não é previsão e não treina nada.
2. **Pergunta:** os municípios "úmidos" (ou "em seca") de um mês refletem chuva, base de chuva quase nula, ajuste ruim da Gama ou dado recente suspeito? H1 base baixa (mm da janela contra média histórica; mm necessários para SPI ≥ +1), H2 ajuste (fração dos anos de treino com SPI ≥ +1; esperado ≈ 15,9%), H3 dado recente (chuva mediana do estado nos 12 meses até o alvo contra a mediana do treino).
3. **Calibração do limite (D13):** imprime, por mês do ano, a mediana da média da janela por mês e a % de municípios abaixo de 5, 10, 15, 20 e 30 mm por mês, só com os parâmetros do treino (`spi_gamma_params`); serve para escolher `spi.media_min_mm_por_mes` antes de regerar o dataset.
4. **Pureza:** `calcular_diagnostico` não faz I/O; `carregar` e `main` leem via `manifest.caminho_verificado`. Testes em `tests/test_diag_spi.py` (contas independentes e verificação por mutação).

## 6. Política de NaN e qualidade — implementada em `src/data/quality.py`
1. 0 mm real é diferente de máscara ou NaN. Nunca converter NaN em 0.
2. **Mês válido** = `precip_mm` existe **e** `dias_validos / dias_no_mes` ≥ `min_cobertura_dias_mes` **e** `pixels_validos_frac` ≥ `min_cobertura_pixels` (limiares inclusivos; a comparação dos pixels é em float32). Mês inválido tem as 3 colunas de chuva em NaN (nunca 0) na série válida.
3. O relatório de cobertura (município × mês, com `motivo`) é gerado **antes** de qualquer imputação (`cobertura_mensal`, seção 5.3).
4. **D9 — SPI sobre a série válida.** O `spi.py` recebe a série válida, sem imputação: mês sem dado fica SPI NaN ("sem dado" na interface, item 7) e o alvo correspondente fica mascarado na perda (nunca preenchido). A imputação existe só para alimentar features.
5. **Imputação causal.** Mediana por (`cod_ibge`, mês-calendário) dos meses **válidos** com `mes` ≤ `split.treino_fim`, incluindo zeros (artefato `climatologia_precip`, seção 8). O *k*-ésimo mês **seguido** sem dado é preenchido se *k* ≤ `max_meses_imputados_consecutivos`; do (*k*+1)-ésimo em diante fica NaN até voltar um mês válido, quando o contador reinicia. A decisão sobre um mês usa só o passado, de modo que alterar o que vem depois de *t* não muda nenhuma linha em ≤ *t* (CA-06). As 3 colunas de chuva recebem **a própria** mediana, e `precip_mm_missing = True` em todo mês sem dado, preenchido ou não. Proibido `bfill` e `interpolate`.
6. **D10 — exclusão de município só pelo treino.** `frac_meses_validos` é medida de `janela.inicio` a `split.treino_fim`; estritamente abaixo de `min_fracao_meses_validos_municipio` (0,95) vira `excluded`, com motivo. Validação e teste não entram na decisão (seria selecionar com informação do futuro). Município excluído continua nos 216 e aparece como "sem dado"; nunca é removido em silêncio.
7. Município com polígono sem pixel válido retorna `null` no GEE: registrado e tratado como ausência (mês inválido).
8. Na interface, mês/município com SPI inválido ou sem dado válido aparece como **"sem dado"**, nunca suavizado ou preenchido para exibição (seção 1.4).

## 7. SPI (método) — implementado em `src/data/spi.py`
1. `P(m)` = precipitação mensal do município. `S_k(t) = soma de P(t-k+1) … P(t)`, com *k* em `spi.janelas_meses` (**[3, 6]**). Qualquer mês ausente na janela torna `S_k(t)` NaN; os primeiros *k*−1 meses também (SPI indisponível, `spiK_invalid = True`). A entrada é a série **válida** (seção 6, D9), nunca a imputada.
2. Para cada (`cod_ibge`, mês-calendário do mês final, escala), usar só os `S_k` com mês final ≤ `split.treino_fim`. `q` = fração de zeros desse conjunto. Gama por máxima verossimilhança (`loc=0`, `scipy.stats.gamma.fit(floc=0)`) nos positivos.
3. `H(x) = q + (1 − q)·G(x)` e `SPI = Φ⁻¹(H(x))`. Parâmetros do treino aplicados a todo o período. H é limitado a [1e-6, 1 − 1e-6]: |SPI| ≤ 4,75 e nunca há ±inf.
4. **Convenção para x = 0 (D3, decidida): H(0) = q.** Como G(0) = 0, é a própria fórmula do item 3, com q = zeros/n (forma de distribuição mista descrita na documentação do EDO/JRC e do ESMValTool; o guia da OMM, WMO-No. 1090, é a referência geral do índice e seu trecho sobre zeros **não foi lido**). Variantes da literatura (q = zeros/(n+1); ponto médio do intervalo dos zeros) **não** são usadas; o efeito delas sobre alertas em meses secos pode ser medido depois como análise de sensibilidade.
5. Uma célula (cod_ibge × mês-calendário × escala) é **inválida** (`valido = False`, SPI NaN, `spiK_invalid = True`) quando: amostras < `spi.min_anos_treino` (30); positivos < `spi.min_positivos_ajuste` (10); positivos todos iguais; ou o MLE não converge. `spiK_baixa_conf = True` (com SPI válido) quando há **pouca chuva esperada** na janela, por um de dois motivos: (a) `q > spi.q_baixa_confianca` (0,5, estritamente maior); ou (b) **D13:** média do treino da janela por mês da janela `< spi.media_min_mm_por_mes` (10 mm, proposta a calibrar), com média = (1 − q) · forma · escala / k (no MLE da Gama com loc = 0, forma · escala é a média dos positivos). O SPI não muda, só a marcação; ela depende só dos parâmetros do treino, então não vaza.
6. A calibração tem, no SPI-3, 34 pontos (jan/fev) ou 35 (mar–dez) por município × mês-calendário; no SPI-6, 34 (jan–mai) ou 35 (jun–dez).
7. O SPI das linhas de treino é in-sample. Registrado como limitação.
8. Saída de `calcular_spi`: SPI em formato longo (`cod_ibge`, `mes`, `spi3`, `spi3_invalid`, `spi3_baixa_conf`, `spi6`, …; mesmas colunas do `demo_observado`) e a tabela `spi_gamma_params` (seção 8). Com `parametros` fornecidos, apenas aplica (sem reajuste): é o caminho da inferência e da avaliação.
9. **Verificado em séries sintéticas (testes):** alterar o futuro não muda parâmetros nem SPI passado; SPI de treino com média ~0 e desvio ~1; fórmula igual a uma conta independente. **Pendente com dado real:** qualidade do ajuste Gama (KS por célula) e fração de células inválidas, após o RAW real.

## 8. Artefatos de treino (`artifacts/`)
- `spi_gamma_params`: `cod_ibge`, `mes_calendario`, `escala_meses`, `q`, `shape`, `scale`, `n_amostras`, `n_positivos`, `valido`. Município excluído (D11) tem `valido = False` em todas as suas células.
- `climatologia_precip`: `cod_ibge`, `mes_calendario`, `mediana_mm`, `mediana_max_dia_mm`, `mediana_dias_chuva`, `n_amostras` (meses válidos do treino na célula; 0 deixa a mediana NaN).
- Scalers de features: Item 9 e Fase 2, ajustados só no treino.

## 9. Regras para features
- Lags com shift positivo; `rolling(..., center=False)`. Alterar valores em *t*+1…*t*+k não pode mudar features em ≤ *t* (CA-06).
- MapBiomas: para o mês de origem no ano Y, usar o mapa de Y-1, com uma única coleção.
- S2iD: só auditoria/antecedência. Não é feature. MUNIC/ICM: só contexto, não feature.

## 10. Limitações conhecidas
- **SPI em meses secos:** pouco informativo no semiárido (muitos zeros). Métricas da Fase 2 por estação.
- **Zeros e D3:** com H(0) = q, um mês de chuva zero recebe SPI = Φ⁻¹(q), pouco negativo quando q é alto; daí a flag `baixa_conf`. O ajuste Gama pode ser fraco em células com muitos zeros, e |SPI| fica limitado a 4,75.
- **Estação seca e SPI (D13):** onde a chuva esperada na janela é de poucos milímetros, uma diferença pequena vira SPI alto ou baixo (limitação conhecida do índice; ver o aviso do NDMC/UNL sobre SPI em regiões de chuva normalmente baixa). Diagnóstico de 2026-11-18 com os dados reais (SPI-3 de set/2026): os 72 municípios "úmidos" tiveram mediana de 7,1 mm na janela contra média histórica de 2,6 mm, e bastavam 5,2 mm para SPI ≥ +1; a fração in-sample de SPI ≥ +1 foi 17,1% (esperado 15,9%) e a chuva de out/2025 a abr/2026 ficou entre 0,52 e 1,30 vez a mediana do treino. Conclusão: conta correta e leitura enganosa, não erro de ajuste nem de dado. Tratamento: marcar (opção A), não esconder.
- **Imputação:** valor imputado não é observação. A mediana usa todos os anos do treino, inclusive os posteriores ao mês imputado (estatística de treino, como um scaler). Lacuna acima de `max_meses_imputados_consecutivos` fica NaN, e o modelo precisa tratar NaN com máscara.
- **Município excluído (D11):** aparece como "sem dado" e sem alerta; isso não significa ausência de seca. O tratamento do município excluído nas features e no treino da Fase 2 ainda não foi definido (pendência na seção 11).
- **SPI ≠ estoque hídrico** e **persistência**: ver seção 1.3.
- **Ferramentas existentes:** o diferencial é hipótese até a comparação com os benchmarks (seção 2) ser feita.
- **MapBiomas retrospectivo:** vazamento fraco e residual (**VERIFICAR** no documento metodológico).
- **Distribuição por bloco:** a validação pode conter a cauda da seca de 2012–2018 e o teste ter menos eventos (hipótese a confirmar na EDA).
- **CHIRPS:** pixel de 0,05° (~5,6 km, ~31 km²). Nenhum município do escopo é menor que 1 pixel, mas 6 têm menos de 4 (o menor da amostra tem 2). Produto de satélite combinado com estações, não medição pontual. Ponderação de borda não provada. Sem estações perto, a validação com INMET fica limitada.
- **Indicadores extras do CHIRPS** (máximo diário e dias de chuva) são médias de métricas por pixel (D5), não métricas da média municipal.
- **Estoque (B):** poucos reservatórios monitorados no PI, conjunto instável, poucos eventos no teste.
- **Auditoria S2iD:** poucos decretos por município; qualitativa no piloto, estatística só no semiárido inteiro.
- **Camada de preparo:** a MUNIC é uma fotografia de uma edição (a mais recente a confirmar); o município pode ter criado o plano depois. Por isso o ano de referência é sempre exibido.
- **Janela de dados:** vai até 2026-09 (D7). A interface não afirma a situação "atual" do município além dessa data, e declara que os meses de 2026 são observados, fora de treino, validação e teste.
- **Lista do semiárido:** vigência dos 50 municípios pós 03/01/2025 não confirmada; Luís Correia com fonte secundária.
- **Enquadramento do GEE:** uso não comercial (pesquisa). Reavaliar antes de uso operacional ou contratado.
- **Números sobre planejamento municipal** citados em materiais são nacionais (não achados para o PI): Plano de Contingência em 27,4% dos municípios em 2023 (preliminar, anais de congresso); menos de 10% com três instrumentos alinhados; cerca de 60% sem estrutura de gestão de desastres. Não usar como dado do Piauí.

## 11. Pendências (`VERIFICAR`)
| ID | Pendência | Resolve em |
|---|---|---|
| D1 | ~~Coleção do MapBiomas~~ — **Coleção 10** | encerrada |
| D2 | ~~Lista do semiárido para o PI~~ — **215 + 1 = 216**. Resíduos: Luís Correia no PDF da Res. 176/2024 e vigência dos 50 após 03/01/2025 | resíduo não bloqueante |
| D3 | ~~Convenção do SPI para x = 0~~ — **H(0) = q** | encerrada (v0.2.2) |
| D4 | ~~`dias_validos`~~ — **mínimo entre pixels com dado** | encerrada (v0.2.1) |
| D5 | ~~Indicadores extras e `limiar_dia_chuva_mm`~~ — **média municipal de métricas por pixel; limiar 1,0 mm** | encerrada (v0.2.1) |
| D6 | Limiar operacional de "colapso" do manancial (ex.: < 10%) | após sonda do SAR |
| D7 | ~~Estender o `demo_observado` até 2026-09 ou manter `janela.fim = 2025-12`~~ — **estendida até 2026-09** (pedido de dados atualizados de 2026-11-17; sonda: 16.709 imagens de 1981-01-01 a 2026-09-30). O teste congelado segue terminando em 2025-12 | encerrada (v0.3.0) |
| D8 | Data exata da apresentação na Campus Party e corte de escopo | agora |
| D9 | ~~SPI sobre série válida ou imputada~~ — **série válida; imputação só para features** | encerrada (v0.2.3) |
| D10 | ~~Janela da exclusão de município por cobertura~~ — **só o treino** | encerrada (v0.2.3) |
| D11 | ~~SPI do município excluído~~ — **NaN, `invalid = True`, `baixa_conf = False`; município fica nos 216 como "sem dado"** | encerrada (v0.2.5) |
| D13 | Baixa confiança por pouca chuva esperada: critério (b) da seção 7, item 5, com `spi.media_min_mm_por_mes` = 10 mm por mês da janela (**proposta, sem fonte própria**; a literatura confirma o problema, não dá um limite único). Calibrar com `python -m src.data.diag_spi` (tabela por mês do ano) antes de congelar | aberta: calibrar; vale para a demonstração |
| D12 | Migrar para o CHIRPS v3 (o CHC encerra a produção do v2 depois de dezembro de 2026 e recomenda o v3). Coleções do GEE: `UCSB-CHC/CHIRPS/V3/DAILY_SAT` (base IMERG), `UCSB-CHC/CHIRPS/V3/DAILY_RNL` (base ERA5) e `UCSB-CHC/CHIRPS/V3/PENTAD`. Exige nova extração, nova sonda, novos parâmetros do SPI e novo manifest; `precip_max_dia_mm` e `dias_chuva` dependem do método diário escolhido. Não misturar v2 e v3 na mesma série | após a demonstração |
| — | **Meses de 2026 na Fase 2:** `splits.rotular` recusa `mes_alvo` > `teste_fim`. Decidir entre filtrar os 9 meses antes de rotular ou criar o rótulo "pós-teste"; os alvos de 2026 não entram em treino, validação nem teste | antes de `features.py` |
| — | **Extração real do CHIRPS:** causa do travamento antigo identificada (chamada HTTP intermitente sem limite). Corrigido com limite de 120 s e nova tentativa. Lote 1 gravado (449 s); lotes 2 a 30 em extração em 3 terminais (2026-11-18); lote 31 (2026-01 a 2026-09) e `build_demo` sobre o RAW real a seguir | agora (bloqueia Nível 1) |
| — | SAR: endpoint, início da série, reservatórios do PI no escopo, tratamento de "seco" | sonda do SAR |
| — | Composição do piloto (SEPLAN-PI) e `municipios_piloto.csv` | quando a lista chegar |
| — | Estações INMET; formato do arquivo | Item 6 |
| — | Formato do código COBRADE e colunas no S2iD (**bloqueia o Nível 2**) | Item 8 |
| — | MUNIC/ICM: edição, variável de plano de contingência e de COMPDEC, disponibilidade por `cod_ibge` no PI | ingestão MUNIC |
| — | Diretrizes do TCE-PI e do MIDR (texto-fonte); previsores oceânicos; acesso a séries do Cemaden/Monitor de Secas | antes da Fase 2 |
| — | ~~Embargo adequado para a cabeça A2 (SPI-6)~~ — **5 meses para todas as cabeças** | encerrada (v0.2.2) |
| — | Qualidade do ajuste Gama nos dados reais (KS por célula, % de células inválidas) | após o RAW real |
| — | ~~`scipy` no `requirements.txt`~~ — **`scipy==1.18.1`** | encerrada (v0.2.3) |
| — | Calibrar os limiares de `qualidade.*` (`min_cobertura_*`, 0,95, máx. de meses imputados) com o RAW real: o CHIRPS quase não tem lacunas, então o esperado é poucos meses inválidos | após o RAW real |
| — | Tratamento do município excluído (D11) nas features e no treino da Fase 2 (omitir da amostra ou manter com máscara) | Item de features |
| — | Comentário do `params.yaml` e docstring do `spi.py` ainda dizem "padrão OMM" para H(0) = q; alinhar ao §7 (trecho da OMM não lido) | próxima alteração nesses arquivos |
| — | Série ONI/RONI do CPC para marcar os anos de El Niño nos gráficos (o ambiente de desenvolvimento não alcança o CPC) | antes de marcar os anos |
| — | Rodapé e página Método dizem "CHIRPS v2.0 final": rever os dois textos se a D12 (CHIRPS v3) for aprovada | D12 |
| — | ~~Quebra de linha~~ — **`.gitattributes` com `eol=lf` (teste incluído) e hash de texto normalizado em LF no `manifest.py`** | encerrada (v0.2.4) |

## 12. Critérios de aceite
Os **CA-01 a CA-10** do System Prompt da Fase 1 valem para este contrato: CA-01 → seção 4; CA-02 e CA-03 → seção 7; CA-04 → seção 8; CA-05 → seção 6; CA-06 → seção 9; CA-07 e CA-08 → manifest e RAW imutável.

Critérios adicionais:
- **CA-11 Zero dados fictícios (reescrito na v0.2.1).** Dado qualquer artefato exibido (interface, API, relatório, slide), então cada valor é rastreável a um arquivo com SHA-256 no manifest e a uma fonte da seção 2; camada sem dado oficial validado aparece omitida ou "Indisponível", nunca com valor ilustrativo. *Erro:* valor sem rastreio, ou fixture sintética fora de `tests/` → falha de revisão.
- **CA-12 Demo sem teste.** Dado o gerador do `demo_observado`, quando executa, então não lê o arquivo do teste congelado e não escreve alvo, split ou métrica (`tests/test_build_demo.py`: nenhuma coluna de alvo, bloco ou métrica; `rotular`, `bloco` e `exigir_teste_liberado` trocados por funções que falham e o gerador roda igual).
- **CA-13 Baselines.** Dada qualquer métrica de modelo, então vem acompanhada de persistência e climatologia (e Ridge/XGBoost por cabeça), com incerteza. Sem comparação, o resultado não é declarado ganho.
- **CA-14 Antecedência única no teste.** O cálculo de antecedência no período de teste ocorre uma vez, após fixar o limiar na validação.
- **CA-16 Site multipágina, offline e honesto (v0.3.3).** Dado o site gerado, então: (a) cada página abre direto do disco, sem erro de console e sem pedido de rede; (b) toda página mostra o aviso de MVP e o `dataset_sha256`; (c) o texto da página e o rótulo do gráfico mostram o mesmo número; (d) o contexto do El Niño traz instituição, data e link, e nunca aparece como previsão do nosso modelo; (e) o que não tem dado oficial validado aparece como "Indisponível". Verificação: `tests/test_paginas.py`, `tests/test_build_site.py` e conferência visual em navegador (claro, escuro e celular). *Erro:* mostrar número de exemplo ou "previsão" para preencher espaço.
- **CA-15 Preparo sem inferência (v0.2.1).** Dado um município sem registro MUNIC/ICM, então a interface mostra "sem informação oficial", e o filtro "sem Plano de Contingência" não o inclui. *Erro:* tratar ausência de registro como ausência de plano.

## 13. Fontes consultadas
- SPI (método): OMM, Standardized Precipitation Index User Guide, WMO-No. 1090 (2012): https://public.wmo.int/en/resources/library/standardized-precipitation-index-user-guide · EDO/JRC, factsheet SPI: https://edo.jrc.ec.europa.eu/documents/factsheets/factsheet_spi_ado.pdf · ESMValTool, SPI: https://esmvaltool.readthedocs.io/en/v1.1.0.dev1/namelists/namelist_spi.html
- Res. CONDEL/SUDENE nº 150/2021: https://www.gov.br/sudene/pt-br/centrais-de-conteudo/resolucao1502021.pdf (PDF local em `data/reference/fonte/`)
- Res. CONDEL/SUDENE nº 176/2024 (transição): fonte secundária (MeioNews) para Luís Correia; **confirmar no PDF oficial**
- Quadro Geográfico IBGE 2025, Semiárido: https://www.ibge.gov.br/apps/quadrogeografico/pdf/qg_2025_080_semiarido.pdf
- IBGE, Semiárido Brasileiro: https://www.ibge.gov.br/geociencias/cartas-e-mapas/redes-geograficas/15974-semiarido-brasileiro.html
- IBGE, Malha Municipal 2022 (`PI_Municipios_2022.zip`, baixado pela equipe)
- Sonda do CHIRPS no GEE (2026-11-15): saída registrada na seção 2 (coleção `UCSB-CHG/CHIRPS/DAILY`)
- MapBiomas Coleção 10 (factsheet): https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2025/09/Factsheet-Amazonia_C10_15.09.pdf
- COBRADE: https://www.cetran.pr.gov.br/sites/cetran/arquivos_restritos/files/migrados/File/Leis/Codificacao_e_Classificacao_Brasileira_de_Desastres_COBRADE2.pdf
- ANA, Manual do SAR: https://www.ana.gov.br/sar/arquivos/manual-sar_v1-0.pdf/@@download/file/Manual%20SAR_V1.1.pdf
- ANA, boletim mensal de açudes do semiárido (ex.: 03/2026): https://www.gov.br/ana/pt-br/sala-de-situacao/acudes-do-semiarido/boletins/mensal/03_2026.pdf
- Monitor de Secas (ANA/FUNCEME): https://www.jornaldosudoeste.com/?p=156222
- Cemaden, boletim de secas (05/2026): https://www.gov.br/cemaden/pt-br/assuntos/monitoramento/impactos-seca/monitoramento-de-seca-para-o-brasil/monitoramento-de-secas-e-impactos-no-brasil-2013-maio-2026/Boletim_secas_05_2026.pdf
- CPC/NOAA, ENSO Diagnostic Discussion (aberta em 2026-11-18): https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml
- INMET, Painel El Niño, boletim mensal nº 5 (2026-10-29): https://portal.inmet.gov.br/uploads/notastecnicas/Painel-El-Ni%C3%B1o-5-edi%C3%A7%C3%A3o1.pdf
- Folha "Construção" da marca Piauí Resiliente (arquivo da equipe, 2026-11-18)
- Lei 12.608/2012: https://legis.senado.leg.br/norma/589531/publicacao/15760049 · Lei 12.340/2010: https://www.legisweb.com.br/legislacao/?id=87625
- MIDR, nota técnica do ICM: https://www.gov.br/mdr/pt-br/assuntos/protecao-e-defesa-civil/NotasTcnicasMetodologiaICM.pdf
- IFPI, monitoramento de reservatórios do PI por satélite: https://seer.ufu.br/index.php/caminhosdegeografia/article/download/66989/36954/322263

## 14. Histórico de versões
- **0.3.3 (2026-11-19):** site reorganizado em seis páginas estáticas e offline (seção 5.6, itens 8 a 11; `--saida` passa a ser uma pasta); marca do projeto (paleta, tipos e gota por geometria) e duas rampas do SPI validadas por contraste; mês de abertura por regra declarada e maiores secas por episódio (`destaques.py`); contexto oficial do El Niño (CPC/NOAA e INMET) em `data/reference/contexto_enso.json`, fora do manifest, com fonte e data em cada item; arredondamento único em Python e JavaScript; fontes ONI/RONI e CHIRPS v3 como pendências; CA-16. `params.yaml` e o `dataset_sha256` não mudaram.
- **0.3.2 (2026-11-18):** diagnóstico `src/data/diag_spi.py` (somente leitura, via manifest; H1 base baixa, H2 ajuste, H3 dado recente) mostrou que o "úmido" de set/2026 era base de chuva quase nula; `spi.media_min_mm_por_mes` (D13, `params.yaml` 0.3.1) acrescenta a baixa confiança por chuva esperada; o site declara a regra, conta só meses confiáveis na ficha e informa quantos municípios em seca têm baixa confiança.
- **0.3.1 (2026-11-18):** gerador do site registrado (seção 5.6, Item 11): HTML único e offline, lido só via manifest conferido; classes do SPI e definição de "em seca"; camadas sem dado oficial como "Indisponível". Primeira execução com o RAW real (118.584 linhas) e `build_demo` real: 216 municípios, 0 excluídos, SPI-3 inválido 0,4% (só o aquecimento de 2 meses). Baixa confiança visível no mapa, na matriz e na ficha.
- **0.3.0 (2026-11-18):** D7 reaberta e fechada com a janela até 2026-09 (549 meses, 31 lotes; `params.yaml` 0.3.0); bloco pós-teste (2026-01 a 2026-09) e regra 3 da seção 4; extração com limite de 120 s por chamada, linha por mês e `--lotes A:B` (seção 5.1); tempos medidos no GEE real; D12 (CHIRPS v3) e pendência dos meses de 2026 na Fase 2 (seção 11).
- **0.2.5 (2026-11-17):** `build_demo.py` registrado (4.1: fluxo qualidade → SPI → D11 → gravação → manifest; 5.3: local dos cinco artefatos, manifest do demo e `extras.demo`); D11 fechada (município excluído com SPI NaN, `valido = False` nos parâmetros; seção 8); D7 fechada em 2025-12 (reabrível até a extração); limitação do município excluído (seção 10) e nova pendência para a Fase 2 (seção 11); CA-12 com o teste que o comprova.
- **0.2.4 (2026-11-17):** `manifest.py` registrado (5.3: campos, hash de texto normalizado em LF, `dataset_sha256`, segurança de caminhos, imutabilidade e verificação); `.gitattributes` com `eol=lf`; pendência de quebra de linha encerrada.
- **0.2.3 (2026-11-17):** `quality.py` registrado (seção 6 reescrita: mês válido, D9, imputação causal com contador de meses seguidos, D10); `cobertura_mensal` (5.3); `climatologia_precip` com 3 medianas e `n_amostras` (seção 8); SPI sobre série válida (5.2 e seção 7); limitações da imputação (seção 10); D9, D10 e `scipy` encerradas; novas pendências de calibração, comentário "OMM" e quebra de linha.
- **0.2.2 (2026-11-17):** D3 fechada (H(0) = q) e embargo de 5 meses para todas as cabeças (validação 2016-06 a 2020-12 e teste 2021-06 a 2025-12, 55 meses cada); `split.embargo_meses`, `spi.janelas_meses` e `spi.convencao_zero` no `params.yaml` 0.2.1 e na config; `splits.py` (seção 4) e `spi.py` (seção 7) registrados; regras de célula inválida, limite de H e baixa confiança; pendências novas (ajuste Gama em dado real, `scipy`).
- **0.2.1 (2026-11-17):** diretriz "zero dados fictícios" (1.4 reescrita; "DADOS ILUSTRATIVOS" revogada; CA-11 reescrito; novo CA-15); requisitos da interface (1.6) e regra de corte por nível (1.5); camada de preparo MUNIC/ICM detalhada (1.2, seção 2, 5.5) com estado "sem informação oficial"; resultados da sonda do CHIRPS e status da extração (seção 2); RAW do CHIRPS passa de proposta a **implementado**, com D4 e D5 fechadas, invariantes, lotes (30) e `meta` do selo (5.1); baselines Ridge/XGBoost (3.1); pendências D7 e D8 e diagnóstico da extração (seção 11); `params.yaml` 0.2.0 existe (seção 15).
- **0.2.0 (2026-11-15):** objetivo, diferencial e limites de uso (seção 1); regra "dados ilustrativos"; níveis da demonstração; alvos multitarefa A, A2, B, B', C (seção 3.1); demonstração sem violar o teste congelado e antecedência (4.1); RAW do CHIRPS com `precip_max_dia_mm` e `dias_chuva` (5.1, proposta); SPI-6; `demo_observado`; CA-11 a CA-14; pendências D4 a D6.
- **0.1.2 (2026-11-15):** referência territorial; escopo fixado em 216; malha IBGE local; GEE registrado como não comercial; `escopo.incluir_transicao` e `escopo.simplificacao_graus`.
- **0.1.1 (2026-11-14):** D1 resolvido (MapBiomas Coleção 10); Item 0 encerrado.

## 15. Parâmetros do `params.yaml` 0.3.1
| Chave | Valor | Estado |
|---|---|---|
| `projeto.versao_params` | `0.3.1` | alterada de 0.3.0 (versão mínima da config: 0.2.1) |
| `janela.fim` | `2026-09` | alterada de 2025-12 (D7); `split.teste_fim` continua `2025-12` |
| `fontes.chirps.reducer_espacial` | `media` | criada (sonda 2026-11-15) |
| `fontes.chirps.escala_m` | `5566` | criada (`nominalScale` = 5565,97 m) |
| `fontes.chirps.limiar_dia_chuva_mm` | `1.0` | criada (validada em `config.py`: > 0 e ≤ 50) |
| `split.embargo_meses` | `5` | alterada de 3 (maior janela do SPI − 1) |
| `spi.janelas_meses` | `[3, 6]` | criada (crescente; deve conter `alvo.escala_spi_meses`) |
| `spi.convencao_zero` | `q` | resolvida (D3) |
| `spi.media_min_mm_por_mes` | `10.0` (proposta; 0 desliga; validada em `config.py`: ≥ 0 e ≤ 100) | criada (D13); calibrar |
| `alerta.limiar_spi` | `-1.0` (proposta; fixar na validação) | prevista; Item 8 / antecedência |
| `fontes.sar.*`, `fontes.munic.*` | a definir após as sondas e a leitura dos arquivos reais | prevista |
| `escopo.piloto.composicao_status`, `fontes.inmet.estacoes` | `VERIFICAR` | pendências reais da config |
