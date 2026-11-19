"""Contraste de cor (WCAG 2.x) para os testes de acessibilidade da marca e do CSS."""
from __future__ import annotations

import re

_HEX = re.compile(r"#([0-9A-Fa-f]{6})$")


def luminancia(hex_: str) -> float:
    m = _HEX.match(hex_)
    if not m:
        raise ValueError(f"cor hexadecimal de 6 dígitos esperada, veio {hex_!r}")
    h = m.group(1)
    canais = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in canais)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contraste(a: str, b: str) -> float:
    claro, escuro = sorted((luminancia(a), luminancia(b)), reverse=True)
    return (claro + 0.05) / (escuro + 0.05)
