"""Modelo preditivo do Piauí Resiliente (Fase 2).

O núcleo é uma rede PyTorch treinada do zero (pesos aleatórios, seed do params.yaml). Random Forest,
Ridge, persistência e climatologia existem só como baselines comparativos. Lógica de domínio e de ML:
nada aqui depende de framework web nem de SQL.
"""
