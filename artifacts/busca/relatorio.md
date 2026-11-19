# Busca de configurações da rede

> MVP experimental, em calibração com S2iD/TCE-PI. Métricas de validação; o teste congelado não foi lido. A metade B da validação só é lida uma vez, ao final da busca; antes disso, nenhum número aqui vem dela.

- Protocolo 1, rodada 1: 40 de 40 tentativas (orçamento fixo), sorteio com semente 42; escolha só na metade A da validação (5832 linhas, meses-alvo antes de 2018-09).
- dataset_sha256 `7043e6c2f9e012d2…`. Limites do treino: {'epocas_max': 80, 'paciencia': 10, 'lote': 512}.
- Piso na metade A (MAE médio das cabeças): persistência 0,459, baseline parcial + climatologia 0,240.
- Rodada 2 (adaptativa, só metade A): 20 tentativas por vizinhança da melhor configuração; para quando 20 seguidas não ganham 0.0005 de MAE médio (limite 200). Parou: 20 tentativas seguidas sem ganho de pelo menos 0.0005 no MAE médio da metade A.

## Vencedora na metade A: tentativa 31

MAE médio 0,2296 (SPI-3 0,2764, SPI-6 0,1829). Correção sobre o baseline: sim; contexto regional: não; perda l1; camadas (64, 32); dropout 0.1; weight decay 0.01; lr 0.001.

A tentativa vencedora foi escolhida entre várias nos mesmos dados: o MAE dela na metade A é otimista. O número que vale é o da metade B.

## Efeito médio de cada fator (MAE médio na metade A)

| Fator | Valor | Tentativas | MAE médio | Melhor |
|---|---|---|---|---|
| Correção sobre o baseline | não | 12 | 0,296 | 0,277 |
| Correção sobre o baseline | sim | 48 | 0,245 | 0,230 |
| Contexto regional | não | 35 | 0,253 | 0,230 |
| Contexto regional | sim | 25 | 0,259 | 0,238 |
| Perda | huber | 14 | 0,267 | 0,240 |
| Perda | l1 | 24 | 0,245 | 0,230 |
| Perda | mse | 22 | 0,259 | 0,243 |
| Camadas ocultas | (128, 64) | 11 | 0,267 | 0,241 |
| Camadas ocultas | (16,) | 7 | 0,262 | 0,236 |
| Camadas ocultas | (32, 16) | 6 | 0,259 | 0,244 |
| Camadas ocultas | (32,) | 6 | 0,269 | 0,242 |
| Camadas ocultas | (48, 24) | 1 | 0,232 | 0,232 |
| Camadas ocultas | (64, 32) | 27 | 0,248 | 0,230 |
| Camadas ocultas | (64, 64) | 1 | 0,235 | 0,235 |
| Camadas ocultas | (96, 48) | 1 | 0,234 | 0,234 |
| Dropout | 0.0 | 17 | 0,262 | 0,236 |
| Dropout | 0.05 | 2 | 0,237 | 0,235 |
| Dropout | 0.1 | 29 | 0,250 | 0,230 |
| Dropout | 0.2 | 1 | 0,243 | 0,243 |
| Dropout | 0.3 | 11 | 0,262 | 0,236 |
| Weight decay | 0.0001 | 12 | 0,262 | 0,243 |
| Weight decay | 0.001 | 16 | 0,262 | 0,235 |
| Weight decay | 0.003 | 1 | 0,234 | 0,234 |
| Weight decay | 0.01 | 26 | 0,253 | 0,230 |
| Weight decay | 0.03 | 2 | 0,234 | 0,233 |
| Weight decay | 0.1 | 3 | 0,236 | 0,234 |
| Taxa de aprendizado | 0.0005 | 3 | 0,236 | 0,233 |
| Taxa de aprendizado | 0.001 | 37 | 0,251 | 0,230 |
| Taxa de aprendizado | 0.002 | 1 | 0,250 | 0,250 |
| Taxa de aprendizado | 0.003 | 19 | 0,267 | 0,242 |

## Todas as tentativas (da melhor para a pior na metade A)

| # | Resíduo | Regional | Perda | Camadas | Dropout | WD | lr | Época | MAE SPI-3 | MAE SPI-6 | MAE médio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | sim | não | l1 | (64, 32) | 0.1 | 0.01 | 0.001 | 3 | 0,276 | 0,183 | 0,2296 |
| 41 | sim | não | l1 | (48, 24) | 0.1 | 0.01 | 0.001 | 7 | 0,279 | 0,185 | 0,2320 |
| 49 | sim | não | l1 | (64, 32) | 0.1 | 0.03 | 0.0005 | 3 | 0,280 | 0,185 | 0,2326 |
| 54 | sim | não | l1 | (64, 32) | 0.1 | 0.01 | 0.0005 | 7 | 0,281 | 0,186 | 0,2336 |
| 55 | sim | não | l1 | (64, 32) | 0.1 | 0.003 | 0.001 | 4 | 0,281 | 0,187 | 0,2340 |
| 44 | sim | não | l1 | (96, 48) | 0.1 | 0.1 | 0.001 | 2 | 0,281 | 0,187 | 0,2341 |
| 50 | sim | não | l1 | (64, 64) | 0.1 | 0.01 | 0.001 | 3 | 0,282 | 0,188 | 0,2350 |
| 52 | sim | não | l1 | (64, 32) | 0.1 | 0.1 | 0.001 | 1 | 0,281 | 0,189 | 0,2351 |
| 59 | sim | não | l1 | (64, 32) | 0.05 | 0.01 | 0.001 | 5 | 0,283 | 0,187 | 0,2352 |
| 13 | sim | não | l1 | (64, 32) | 0.1 | 0.001 | 0.001 | 4 | 0,282 | 0,189 | 0,2353 |
| 51 | sim | não | l1 | (64, 32) | 0.1 | 0.03 | 0.001 | 3 | 0,284 | 0,188 | 0,2360 |
| 21 | sim | não | l1 | (64, 32) | 0.0 | 0.001 | 0.001 | 2 | 0,283 | 0,189 | 0,2360 |
| 33 | sim | não | l1 | (16,) | 0.3 | 0.001 | 0.001 | 5 | 0,284 | 0,189 | 0,2362 |
| 60 | sim | sim | l1 | (64, 32) | 0.1 | 0.1 | 0.001 | 3 | 0,286 | 0,190 | 0,2377 |
| 57 | sim | sim | l1 | (64, 32) | 0.1 | 0.01 | 0.001 | 1 | 0,288 | 0,190 | 0,2385 |
| 58 | sim | sim | l1 | (64, 32) | 0.05 | 0.01 | 0.001 | 1 | 0,286 | 0,192 | 0,2390 |
| 43 | sim | não | huber | (64, 32) | 0.1 | 0.01 | 0.001 | 1 | 0,288 | 0,191 | 0,2400 |
| 53 | sim | não | l1 | (128, 64) | 0.0 | 0.01 | 0.001 | 1 | 0,288 | 0,193 | 0,2405 |
| 34 | sim | não | l1 | (32,) | 0.0 | 0.001 | 0.003 | 3 | 0,292 | 0,191 | 0,2417 |
| 56 | sim | sim | l1 | (64, 32) | 0.1 | 0.01 | 0.0005 | 1 | 0,289 | 0,196 | 0,2422 |
| 47 | sim | não | huber | (64, 32) | 0.2 | 0.01 | 0.001 | 4 | 0,290 | 0,196 | 0,2428 |
| 25 | sim | não | mse | (64, 32) | 0.3 | 0.0001 | 0.003 | 2 | 0,294 | 0,193 | 0,2434 |
| 5 | sim | sim | mse | (32, 16) | 0.3 | 0.001 | 0.001 | 2 | 0,294 | 0,193 | 0,2437 |
| 28 | sim | sim | l1 | (128, 64) | 0.0 | 0.0001 | 0.001 | 1 | 0,291 | 0,196 | 0,2439 |
| 8 | sim | sim | huber | (128, 64) | 0.3 | 0.01 | 0.001 | 1 | 0,294 | 0,196 | 0,2451 |
| 38 | sim | não | mse | (16,) | 0.3 | 0.0001 | 0.003 | 1 | 0,292 | 0,200 | 0,2462 |
| 40 | sim | não | huber | (32,) | 0.0 | 0.0001 | 0.001 | 2 | 0,291 | 0,204 | 0,2476 |
| 42 | sim | não | mse | (64, 32) | 0.1 | 0.01 | 0.001 | 1 | 0,297 | 0,199 | 0,2481 |
| 6 | sim | sim | huber | (16,) | 0.1 | 0.01 | 0.001 | 1 | 0,297 | 0,200 | 0,2485 |
| 2 | sim | não | mse | (64, 32) | 0.1 | 0.0001 | 0.001 | 1 | 0,294 | 0,203 | 0,2489 |
| 18 | sim | não | mse | (64, 32) | 0.3 | 0.01 | 0.003 | 1 | 0,297 | 0,202 | 0,2494 |
| 48 | sim | não | mse | (64, 32) | 0.1 | 0.01 | 0.002 | 1 | 0,298 | 0,201 | 0,2497 |
| 26 | sim | sim | mse | (64, 32) | 0.3 | 0.0001 | 0.003 | 1 | 0,299 | 0,202 | 0,2502 |
| 19 | sim | não | mse | (16,) | 0.1 | 0.0001 | 0.001 | 2 | 0,297 | 0,204 | 0,2504 |
| 35 | sim | sim | mse | (64, 32) | 0.1 | 0.001 | 0.001 | 1 | 0,297 | 0,204 | 0,2506 |
| 46 | sim | não | mse | (64, 32) | 0.1 | 0.001 | 0.001 | 1 | 0,296 | 0,206 | 0,2511 |
| 12 | sim | sim | mse | (128, 64) | 0.3 | 0.01 | 0.003 | 1 | 0,302 | 0,200 | 0,2512 |
| 37 | sim | sim | huber | (32, 16) | 0.0 | 0.0001 | 0.003 | 1 | 0,297 | 0,206 | 0,2514 |
| 45 | sim | sim | mse | (64, 32) | 0.1 | 0.01 | 0.001 | 1 | 0,301 | 0,203 | 0,2519 |
| 27 | sim | sim | mse | (128, 64) | 0.0 | 0.001 | 0.001 | 1 | 0,298 | 0,206 | 0,2520 |
| 16 | sim | sim | mse | (32, 16) | 0.0 | 0.0001 | 0.001 | 1 | 0,300 | 0,208 | 0,2543 |
| 22 | sim | sim | mse | (32, 16) | 0.0 | 0.001 | 0.001 | 1 | 0,301 | 0,208 | 0,2549 |
| 39 | sim | sim | huber | (32, 16) | 0.0 | 0.01 | 0.003 | 1 | 0,306 | 0,207 | 0,2565 |
| 32 | sim | sim | huber | (32,) | 0.1 | 0.001 | 0.003 | 1 | 0,305 | 0,209 | 0,2571 |
| 29 | sim | sim | huber | (16,) | 0.0 | 0.001 | 0.003 | 2 | 0,307 | 0,211 | 0,2588 |
| 9 | sim | sim | mse | (32,) | 0.1 | 0.01 | 0.003 | 1 | 0,309 | 0,211 | 0,2600 |
| 14 | sim | não | mse | (128, 64) | 0.1 | 0.01 | 0.003 | 1 | 0,311 | 0,217 | 0,2639 |
| 20 | sim | sim | mse | (128, 64) | 0.0 | 0.01 | 0.003 | 1 | 0,327 | 0,224 | 0,2754 |
| 10 | não | não | l1 | (16,) | 0.0 | 0.01 | 0.003 | 17 | 0,340 | 0,215 | 0,2774 |
| 15 | não | não | l1 | (64, 32) | 0.0 | 0.001 | 0.003 | 2 | 0,353 | 0,217 | 0,2849 |
| 7 | não | não | huber | (128, 64) | 0.0 | 0.001 | 0.001 | 3 | 0,357 | 0,218 | 0,2873 |
| 24 | não | não | huber | (128, 64) | 0.1 | 0.01 | 0.001 | 4 | 0,353 | 0,225 | 0,2889 |
| 23 | não | não | l1 | (128, 64) | 0.1 | 0.0001 | 0.003 | 2 | 0,355 | 0,227 | 0,2908 |
| 3 | não | sim | l1 | (32, 16) | 0.0 | 0.01 | 0.001 | 7 | 0,368 | 0,220 | 0,2943 |
| 4 | não | não | mse | (128, 64) | 0.3 | 0.001 | 0.001 | 4 | 0,362 | 0,240 | 0,3008 |
| 1 | não | não | mse | (64, 32) | 0.1 | 0.0001 | 0.001 | 4 | 0,363 | 0,240 | 0,3014 |
| 36 | não | sim | huber | (32,) | 0.3 | 0.001 | 0.001 | 19 | 0,358 | 0,248 | 0,3028 |
| 11 | não | sim | huber | (32,) | 0.0 | 0.01 | 0.003 | 3 | 0,373 | 0,233 | 0,3030 |
| 30 | não | não | mse | (64, 32) | 0.1 | 0.001 | 0.003 | 2 | 0,363 | 0,248 | 0,3057 |
| 17 | não | sim | huber | (16,) | 0.3 | 0.0001 | 0.003 | 16 | 0,360 | 0,266 | 0,3131 |

## Confirmação na metade B (lida uma única vez)

6048 linhas, meses-alvo a partir de 2018-09. Conjunto de 5 redes da configuração vencedora. Baselines ajustados no treino (Ridge com alfa escolhido na metade A).

### alvo_spi3_h1

| Modelo | MAE | RMSE | Viés | Seca: precisão | Seca: revocação | Seca: F1 | Acerto de classe |
|---|---|---|---|---|---|---|---|
| Persistência | 0,638 | 0,831 | -0,015 | 0,284 | 0,299 | 0,291 | 0,680 |
| Climatologia | 0,752 | 0,922 | -0,458 | — | 0,000 | — | 0,701 |
| Parcial + climatologia | 0,414 | 0,641 | -0,235 | 0,545 | 0,263 | 0,355 | 0,794 |
| Ridge | 0,512 | 0,695 | -0,133 | 0,286 | 0,072 | 0,115 | 0,744 |
| Floresta aleatória | 0,486 | 0,692 | -0,148 | 0,234 | 0,044 | 0,074 | 0,761 |
| Rede final (conjunto) | 0,395 | 0,626 | -0,178 | 0,561 | 0,255 | 0,351 | 0,809 |

Prever sempre "normal" acertaria 0,701 das classes: o acerto de classe só vale acima disso.

- Rede final − Persistência (persistência): diferença de MAE -0,2432 [-0,3490; -0,1493] (95%, bootstrap de 28 meses): a rede erra menos (intervalo inteiro abaixo de zero).
- Rede final − Parcial + climatologia (melhor baseline): diferença de MAE -0,0186 [-0,0380; 0,0003] (95%, bootstrap de 28 meses): diferença indistinguível de ruído (o intervalo contém zero).

Alerta de seca com o limiar ajustado na metade A (a previsão dispara o alerta quando o SPI previsto fica abaixo do limiar; seca real continua sendo SPI <= -1):

| Modelo | Limiar | Precisão | Revocação | F1 | Secas reais pegas | Alarmes falsos |
|---|---|---|---|---|---|---|
| Persistência | -0,90 | 0,248 | 0,319 | 0,279 | 80 de 251 | 242 |
| Climatologia | -1,50 | — | 0,000 | — | 0 de 251 | 0 |
| Parcial + climatologia | -1,00 | 0,545 | 0,263 | 0,355 | 66 de 251 | 55 |
| Ridge | -0,75 | 0,314 | 0,199 | 0,244 | 50 de 251 | 109 |
| Floresta aleatória | -0,70 | 0,200 | 0,163 | 0,180 | 41 de 251 | 164 |
| Rede final (conjunto) | -1,00 | 0,561 | 0,255 | 0,351 | 64 de 251 | 50 |

Margem de erro (80% dos erros na metade A), cobertura obtida na metade B:

| Margem | Valor | Cobertura B | Estação chuvosa | Estação seca |
|---|---|---|---|---|
| Global | ±0,459 | 0,691 | 0,537 | 0,844 |
| Por estação | chuvosa ±0,612, seca ±0,261 | 0,698 | 0,661 | 0,736 |
| Proporcional à dispersão do conjunto | ±0,474 em média | 0,718 | 0,599 | 0,837 |

### alvo_spi6_h1

| Modelo | MAE | RMSE | Viés | Seca: precisão | Seca: revocação | Seca: F1 | Acerto de classe |
|---|---|---|---|---|---|---|---|
| Persistência | 0,473 | 0,665 | -0,008 | 0,297 | 0,349 | 0,321 | 0,755 |
| Climatologia | 0,790 | 0,944 | -0,592 | — | 0,000 | — | 0,669 |
| Parcial + climatologia | 0,326 | 0,561 | -0,198 | 0,296 | 0,270 | 0,282 | 0,836 |
| Ridge | 0,369 | 0,571 | -0,123 | 0,275 | 0,238 | 0,255 | 0,821 |
| Floresta aleatória | 0,337 | 0,558 | -0,122 | 0,333 | 0,230 | 0,272 | 0,825 |
| Rede final (conjunto) | 0,309 | 0,545 | -0,156 | 0,297 | 0,262 | 0,278 | 0,843 |

Prever sempre "normal" acertaria 0,669 das classes: o acerto de classe só vale acima disso.

- Rede final − Persistência (persistência): diferença de MAE -0,1641 [-0,2286; -0,1000] (95%, bootstrap de 28 meses): a rede erra menos (intervalo inteiro abaixo de zero).
- Rede final − Parcial + climatologia (melhor baseline): diferença de MAE -0,0169 [-0,0329; -0,0019] (95%, bootstrap de 28 meses): a rede erra menos (intervalo inteiro abaixo de zero).

Alerta de seca com o limiar ajustado na metade A (a previsão dispara o alerta quando o SPI previsto fica abaixo do limiar; seca real continua sendo SPI <= -1):

| Modelo | Limiar | Precisão | Revocação | F1 | Secas reais pegas | Alarmes falsos |
|---|---|---|---|---|---|---|
| Persistência | -0,90 | 0,234 | 0,357 | 0,283 | 45 de 126 | 147 |
| Climatologia | -1,50 | — | 0,000 | — | 0 de 126 | 0 |
| Parcial + climatologia | -0,95 | 0,279 | 0,310 | 0,293 | 39 de 126 | 101 |
| Ridge | -0,80 | 0,218 | 0,341 | 0,266 | 43 de 126 | 154 |
| Floresta aleatória | -0,95 | 0,317 | 0,254 | 0,282 | 32 de 126 | 69 |
| Rede final (conjunto) | -0,95 | 0,285 | 0,294 | 0,289 | 37 de 126 | 93 |

Margem de erro (80% dos erros na metade A), cobertura obtida na metade B:

| Margem | Valor | Cobertura B | Estação chuvosa | Estação seca |
|---|---|---|---|---|
| Global | ±0,331 | 0,689 | 0,455 | 0,923 |
| Por estação | chuvosa ±0,562, seca ±0,076 | 0,722 | 0,656 | 0,787 |
| Proporcional à dispersão do conjunto | ±0,312 em média | 0,710 | 0,490 | 0,931 |

## Leitura

- Rede final supera a persistência em todas as cabeças: sim.
- Rede final supera o melhor baseline em todas as cabeças: não.
- A metade B não participou de escolha alguma (nem de baseline). Ela é uma medida honesta, mas de 28 meses de uma só região: o teste congelado continua sendo a medida final.

MVP experimental, em calibração com S2iD/TCE-PI. Métricas de validação; o teste congelado não foi lido. A metade B da validação só é lida uma vez, ao final da busca; antes disso, nenhum número aqui vem dela.
