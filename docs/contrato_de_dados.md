# Contrato de Dados — Piauí Resiliente · Fase 1

Versão 0.1.1 · 2026-11-14 · Status: **aprovado pela equipe (Item 0 encerrado)** · Depende de `params.yaml` ≥ 0.1.1

> **Aviso de maturidade:** MVP experimental — em calibração com S2iD/TCE-PI. Nenhum indicador derivado deste dataset tem eficácia operacional comprovada para decisão governamental.

## 1. Escopo e uso
- **Objetivo da Fase 1:** produzir um dataset mensal por município, com alvo SPI-3 sem vazamento, splits cronológicos e `manifest.json` com SHA-256. Nenhum modelo é treinado aqui.
- **Extração e treino:** todo o semiárido piauiense (lista fixa; versão e data de consulta registradas no manifest).
- **Piloto:** Território Chapada Vale do Rio Itaim (polo: Paulistana, `2207801`), usado para validação qualitativa, auditoria S2iD e demonstração. A composição é declarada pela equipe e está **VERIFICAR** na SEPLAN-PI. O piloto é subconjunto do semiárido e participa do treino, pois o split é temporal, não espacial.
- **Fora do escopo:** modelos PyTorch, API, MLflow, geração de planos por IA generativa.

## 2. Fontes e decisões de referência

| Item | Decisão | Status | Observação |
|---|---|---|---|
| Malha municipal | IBGE 2022 via `geobr.read_municipality(year=2022)` | CONFIRMADO | O geobr entrega SIRGAS2000 (EPSG:4674). Reprojetar para EPSG:4326 só para o GEE. Fixar versão do geobr e salvar cópia em `data/reference/` com SHA-256. |
| Semiárido | Resolução CONDEL/SUDENE nº 150/2021 | PARCIAL | A resolução credenciou 1.427 municípios e retirou 50, mas esses 50 foram mantidos em regra de transição (prorrogada pela Res. 176/2024). O Quadro Geográfico IBGE 2025 lista 1.477. Definir qual lista vale para o PI e fixar uma versão (**D2**). |
| Precipitação | CHIRPS diário via GEE: `UCSB-CHG/CHIRPS/DAILY`, banda `precipitation` | A CONFERIR | Conferir ID, banda e escala no primeiro `getInfo()`. |
| Cobertura do solo | MapBiomas **Coleção 10** | **DECIDIDO (D1, 2026-11-14)** | A Coleção 9 cobre 1985–2023 e não atende alvos de 2025 com defasagem de 1 ano. A Coleção 10 cobre 1985–2024. Uma única coleção em todos os anos. |
| Desastres | S2iD, COBRADE `1.4.1.1.0` (Estiagem) e `1.4.1.2.0` (Seca) | CONFIRMADO | Códigos conferidos na classificação oficial. O formato dos códigos no arquivo do S2iD é **VERIFICAR**. Uso: só auditoria. |
| Estações | INMET (BDMEP) | VERIFICAR | Listar estações dentro ou perto do semiárido e conferir o formato real do arquivo (sentinela, decimal, encoding, fuso). Uso: validação cruzada do CHIRPS. |

## 3. Unidade de amostra e alvo
- **Linha:** (`cod_ibge`, `mes_origem` = *t*). Toda feature usa apenas dados ≤ *t*.
- **Alvo:** `alvo_spi3_h1` = SPI-3 no mês *t*+1. O SPI-3 de um mês *m* usa a precipitação de *m*−2, *m*−1 e *m*.
- **`mes_alvo`** = *t* + *h* (com *h* = 1). **O split é definido por `mes_alvo`.**

## 4. Splits

| Bloco | `mes_alvo` | Meses |
|---|---|---|
| treino | ≈1982-04 a 2015-12 (início depende de `max_lag_meses`) | ~405 |
| embargo | 2016-01 a 2016-03 | 3 |
| validação | 2016-04 a 2020-12 | 57 |
| embargo | 2021-01 a 2021-03 | 3 |
| teste congelado | 2021-04 a 2025-12 | 57 |

Regras:
1. Mesmos cortes para todos os municípios. Nunca há shuffle.
2. Embargo de 3 meses. O mínimo teórico é 2, porque a janela do SPI-3 se sobrepõe em 2 meses; o 3 fica como margem.
3. O teste fica em arquivo próprio, com hash no manifest, e só é lido quando `split.teste_liberado: true` (avaliação final da Fase 2).
4. A EDA registra, por bloco, o número de linhas e o % de `spi3 ≤ -1`. Se o teste tiver poucos eventos de seca, as métricas dele são declaradas não conclusivas.

## 5. Schemas

### 5.1 `raw_chirps_mensal` (RAW, imutável)
`cod_ibge:str(7)`, `mes:datetime64[ns]` (dia 1), `precip_mm:float32` (soma mensal da média municipal), `dias_validos:int16`, `dias_no_mes:int16`, `pixels_validos_frac:float32`.

### 5.2 `curated_mensal`
| Coluna | dtype | Descrição |
|---|---|---|
| `cod_ibge` | str(7) | código IBGE |
| `mes_origem`, `mes_alvo` | datetime64[ns] | dia 1 |
| `precip_mm` | float32 | após imputação causal |
| `cobertura_dias`, `cobertura_pixels` | float32 | 0–1 |
| `precip_mm_missing` | bool | imputado ou ausente |
| `spi3` | float32 | SPI-3 em *t* |
| `spi3_invalid`, `spi3_baixa_conf` | bool | ver seção 7 |
| `alvo_spi3_h1` | float32 | SPI-3 em *t*+1 |
| `split` | category | `train`, `val`, `test`, `embargo` |
| `mb_*` | float32 | MapBiomas do ano Y-1 (nomes definidos no Item 7) |
| features de lag | float32 | definidas no Item 9 |

### 5.3 Tabelas auxiliares
- **`qualidade_municipio`:** `cod_ibge:str`, `frac_meses_validos:float32`, `excluded:bool`, `motivo:str`.
- **`audit_s2id`** (nunca entra como feature): `cod_ibge:str`, `data_decreto:datetime64[ns]`, `cobrade:str`, `tipo:str`. Nomes de coluna **VERIFICAR** contra o arquivo real.
- **`manifest.json`:** versão do dataset, data de criação (UTC), versões do Python e das bibliotecas, seed, SHA-256 do `params.yaml`, e por arquivo: caminho, SHA-256, nº de linhas, colunas e dtypes. Também: datas de corte e embargo, nº de linhas por bloco, fonte, versão e data da lista do semiárido, ano e versão da malha, coleção do MapBiomas e `"maturity"`.

## 6. Política de NaN e qualidade
1. 0 mm real é diferente de máscara ou NaN. Nunca converter NaN em 0.
2. O mês só é válido com `cobertura_dias` ≥ `min_cobertura_dias_mes` e `cobertura_pixels` ≥ `min_cobertura_pixels`.
3. O relatório de cobertura (município × mês) é gerado **antes** de qualquer imputação.
4. Imputação de `precip_mm`: mediana mensal calculada **só no treino**, até `max_meses_imputados_consecutivos`. Sempre com `precip_mm_missing = True`. Proibido `bfill` e `interpolate` que olhe o futuro.
5. Município com `frac_meses_validos` abaixo do limiar vira `excluded` com motivo registrado, nunca removido em silêncio.
6. Município com polígono sem pixel válido retorna `null` no GEE. Isso é registrado e tratado como ausência.

## 7. SPI-3 (método)
1. `P(m)` = precipitação mensal do município. `S3(t) = P(t-2) + P(t-1) + P(t)`. Qualquer mês ausente na janela torna `S3(t)` NaN.
2. Para cada (`cod_ibge`, mês-calendário do mês final), usar só os `S3` com mês final no **treino**. `q` = fração de zeros. A Gama é ajustada por máxima verossimilhança (`loc=0`) nos valores positivos.
3. `H(x) = q + (1 − q)·G(x)` e `SPI = Φ⁻¹(H(x))`. Os parâmetros do treino são aplicados a validação e teste.
4. **Convenção para x = 0** (`H(0) = q` ou outra) varia entre implementações. Decisão em `spi.convencao_zero` (**D3**), conferida contra a literatura no Item 5.
5. Casos de borda:
   - série toda zero, menos de `min_positivos_ajuste` positivos ou Gama sem convergir → `spi3_invalid = True` (nunca ±inf);
   - `q > q_baixa_confianca` (típico de jun–out no semiárido) → `spi3_baixa_conf = True`.
6. A calibração tem 34 pontos (jan/fev) ou 35 (mar–dez) por município × mês-calendário. É suficiente para ≥30, mas deixa o ajuste ruidoso.
7. O SPI das linhas de treino é in-sample, porque a Gama foi ajustada com os mesmos dados. Registrado como limitação.

## 8. Artefatos de treino (`artifacts/`)
- `spi_gamma_params`: `cod_ibge`, `mes_calendario`, `q`, `shape`, `scale`, `n_amostras`, `n_positivos`, `valido`.
- `climatologia_precip`: `cod_ibge`, `mes_calendario`, `mediana_mm` (usada na imputação).
- Scalers de features ficam para o Item 9 e a Fase 2, ajustados só no treino.

## 9. Regras para features
- Lags com shift positivo; `rolling(..., center=False)`. Alterar valores em *t*+1…*t*+k não pode mudar features em ≤ *t* (CA-06).
- MapBiomas: para o mês de origem no ano Y, usar o mapa de Y-1, mantendo **uma única coleção** em todos os anos.
- S2iD: só auditoria. Não é feature.

## 10. Limitações conhecidas
- **SPI-3 em meses secos:** pouco informativo no semiárido (muitos zeros). As métricas da Fase 2 saem separadas por estação.
- **MapBiomas retrospectivo:** o mapa de Y-1 pode ter sido classificado com filtros temporais que usam anos vizinhos. É um vazamento fraco e residual (**VERIFICAR** no documento metodológico do MapBiomas).
- **Distribuição por bloco:** a validação pode conter a cauda da seca de 2012–2018 do Nordeste e o teste pode ter menos eventos. É uma hipótese a confirmar na EDA.
- **CHIRPS:** ~5 km. Sem estações perto, a validação cruzada com o INMET fica limitada.
- **Auditoria S2iD:** decretos são poucos por município. A auditoria é qualitativa no piloto e estatística só no semiárido inteiro.

## 11. Pendências (`VERIFICAR`)
| ID | Pendência | Resolve em |
|---|---|---|
| D1 | ~~Coleção do MapBiomas~~ — **resolvida: Coleção 10** | encerrada |
| D2 | Lista do semiárido vigente para o PI (1.427 vs 1.477) e versão fixada | Item 1 |
| D3 | Convenção do SPI para x = 0 | Item 5 |
| — | Composição do território piloto (SEPLAN-PI) | Item 1 |
| — | Estações INMET; formato do arquivo | Item 6 |
| — | Formato do código COBRADE e colunas no S2iD | Item 8 |

## 12. Critérios de aceite
Os critérios **CA-01 a CA-10** do System Prompt da Fase 1 valem para este contrato. Em especial:
- CA-01 → seção 4
- CA-02 e CA-03 → seção 7
- CA-04 → seção 8
- CA-05 → seção 6
- CA-06 → seção 9
- CA-07 e CA-08 → manifest e RAW imutável

## 13. Fontes consultadas
- Resolução CONDEL/SUDENE nº 150/2021: https://www.gov.br/sudene/pt-br/centrais-de-conteudo/resolucao1502021.pdf
- Quadro Geográfico IBGE 2025, Semiárido: https://www.ibge.gov.br/apps/quadrogeografico/pdf/qg_2025_080_semiarido.pdf
- IBGE, Semiárido Brasileiro (Res. 176/2024): https://www.ibge.gov.br/geociencias/cartas-e-mapas/redes-geograficas/15974-semiarido-brasileiro.html
- geobr: https://ipeagit.github.io/geobr/
- MapBiomas Coleção 9: https://brasil.mapbiomas.org/en/?p=873
- MapBiomas Coleção 10 (factsheet): https://brasil.mapbiomas.org/wp-content/uploads/sites/4/2025/09/Factsheet-Amazonia_C10_15.09.pdf
- COBRADE: https://www.cetran.pr.gov.br/sites/cetran/arquivos_restritos/files/migrados/File/Leis/Codificacao_e_Classificacao_Brasileira_de_Desastres_COBRADE2.pdf
