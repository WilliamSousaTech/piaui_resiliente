# Busca de configurações da rede

> MVP experimental, em calibração com S2iD/TCE-PI. Métricas de validação; o teste congelado não foi lido. A metade B da validação foi lida uma única vez, ao final.

- Protocolo 1: 40 de 40 tentativas (orçamento fixo), sorteio com semente 42; escolha só na metade A da validação (5832 linhas, meses-alvo antes de 2018-09).
- dataset_sha256 `7043e6c2f9e012d2…`. Limites do treino: {'epocas_max': 80, 'paciencia': 10, 'lote': 512}.
- Piso na metade A (MAE médio das cabeças): persistência 0,459, baseline parcial + climatologia 0,240.

## Vencedora na metade A: tentativa 31

MAE médio 0,2296 (SPI-3 0,2764, SPI-6 0,1829). Correção sobre o baseline: sim; contexto regional: não; perda l1; camadas (64, 32); dropout 0.1; weight decay 0.01; lr 0.001.

A tentativa vencedora foi escolhida entre várias nos mesmos dados: o MAE dela na metade A é otimista. O número que vale é o da metade B.

## Efeito médio de cada fator (MAE médio na metade A)

| Fator | Valor | Tentativas | MAE médio | Melhor |
|---|---|---|---|---|
| Correção sobre o baseline | não | 12 | 0,296 | 0,277 |
| Correção sobre o baseline | sim | 28 | 0,249 | 0,230 |
| Contexto regional | não | 20 | 0,263 | 0,230 |
| Contexto regional | sim | 20 | 0,263 | 0,244 |
| Perda | huber | 12 | 0,272 | 0,245 |
| Perda | l1 | 10 | 0,257 | 0,230 |
| Perda | mse | 18 | 0,261 | 0,243 |
| Camadas ocultas | (128, 64) | 10 | 0,270 | 0,244 |
| Camadas ocultas | (16,) | 7 | 0,262 | 0,236 |
| Camadas ocultas | (32, 16) | 6 | 0,259 | 0,244 |
| Camadas ocultas | (32,) | 6 | 0,269 | 0,242 |
| Camadas ocultas | (64, 32) | 11 | 0,258 | 0,230 |
| Dropout | 0.0 | 16 | 0,264 | 0,236 |
| Dropout | 0.1 | 13 | 0,264 | 0,230 |
| Dropout | 0.3 | 11 | 0,262 | 0,236 |
| Weight decay | 0.0001 | 12 | 0,262 | 0,243 |
| Weight decay | 0.001 | 15 | 0,263 | 0,235 |
| Weight decay | 0.01 | 13 | 0,265 | 0,230 |
| Taxa de aprendizado | 0.001 | 21 | 0,260 | 0,230 |
| Taxa de aprendizado | 0.003 | 19 | 0,267 | 0,242 |

## Todas as tentativas (da melhor para a pior na metade A)

| # | Resíduo | Regional | Perda | Camadas | Dropout | WD | lr | Época | MAE SPI-3 | MAE SPI-6 | MAE médio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 31 | sim | não | l1 | (64, 32) | 0.1 | 0.01 | 0.001 | 3 | 0,276 | 0,183 | 0,2296 |
| 13 | sim | não | l1 | (64, 32) | 0.1 | 0.001 | 0.001 | 4 | 0,282 | 0,189 | 0,2353 |
| 21 | sim | não | l1 | (64, 32) | 0.0 | 0.001 | 0.001 | 2 | 0,283 | 0,189 | 0,2360 |
| 33 | sim | não | l1 | (16,) | 0.3 | 0.001 | 0.001 | 5 | 0,284 | 0,189 | 0,2362 |
| 34 | sim | não | l1 | (32,) | 0.0 | 0.001 | 0.003 | 3 | 0,292 | 0,191 | 0,2417 |
| 25 | sim | não | mse | (64, 32) | 0.3 | 0.0001 | 0.003 | 2 | 0,294 | 0,193 | 0,2434 |
| 5 | sim | sim | mse | (32, 16) | 0.3 | 0.001 | 0.001 | 2 | 0,294 | 0,193 | 0,2437 |
| 28 | sim | sim | l1 | (128, 64) | 0.0 | 0.0001 | 0.001 | 1 | 0,291 | 0,196 | 0,2439 |
| 8 | sim | sim | huber | (128, 64) | 0.3 | 0.01 | 0.001 | 1 | 0,294 | 0,196 | 0,2451 |
| 38 | sim | não | mse | (16,) | 0.3 | 0.0001 | 0.003 | 1 | 0,292 | 0,200 | 0,2462 |
| 40 | sim | não | huber | (32,) | 0.0 | 0.0001 | 0.001 | 2 | 0,291 | 0,204 | 0,2476 |
| 6 | sim | sim | huber | (16,) | 0.1 | 0.01 | 0.001 | 1 | 0,297 | 0,200 | 0,2485 |
| 2 | sim | não | mse | (64, 32) | 0.1 | 0.0001 | 0.001 | 1 | 0,294 | 0,203 | 0,2489 |
| 18 | sim | não | mse | (64, 32) | 0.3 | 0.01 | 0.003 | 1 | 0,297 | 0,202 | 0,2494 |
| 26 | sim | sim | mse | (64, 32) | 0.3 | 0.0001 | 0.003 | 1 | 0,299 | 0,202 | 0,2502 |
| 19 | sim | não | mse | (16,) | 0.1 | 0.0001 | 0.001 | 2 | 0,297 | 0,204 | 0,2504 |
| 35 | sim | sim | mse | (64, 32) | 0.1 | 0.001 | 0.001 | 1 | 0,297 | 0,204 | 0,2506 |
| 12 | sim | sim | mse | (128, 64) | 0.3 | 0.01 | 0.003 | 1 | 0,302 | 0,200 | 0,2512 |
| 37 | sim | sim | huber | (32, 16) | 0.0 | 0.0001 | 0.003 | 1 | 0,297 | 0,206 | 0,2514 |
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

MVP experimental, em calibração com S2iD/TCE-PI. Métricas de validação; o teste congelado não foi lido. A metade B da validação foi lida uma única vez, ao final.
