"""Funções de texto usadas para validar as informações extraídas."""

import re
import unicodedata

# Marcadores usados no catálogo (o "•" aparece como "բ" na extração do PDF).
_MARCADORES = "•բ●▪◦·"


def normalizar(texto):
    """Minúsculas, sem acentos, sem marcadores e com espaços simples."""
    if not texto:
        return ""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    texto = texto.lower()
    texto = texto.replace("’", "'").replace("‘", "'")
    for marcador in _MARCADORES:
        texto = texto.replace(marcador, " ")
    texto = re.sub(r"[^\w$',.%/-]+", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokens(texto):
    texto = normalizar(texto)
    texto = re.sub(r"(\d)\s+(ml|g|kg|l)\b", r"\1\2", texto)  # "100 ml" -> "100ml"
    return [t for t in re.split(r"[\s,.;:()/+-]+", texto) if t]


def texto_presente(trecho, texto_pagina):
    """True se o trecho aparece na página (ignorando acentos e quebras)."""
    trecho_norm = normalizar(trecho)
    if not trecho_norm:
        return False
    return trecho_norm in normalizar(texto_pagina)


def tokens_presentes(trecho, texto_pagina, cobertura_minima=1.0):
    """True se (quase) todas as palavras do trecho existem na página."""
    palavras = tokens(trecho)
    if not palavras:
        return False
    da_pagina = set(tokens(texto_pagina))
    encontradas = sum(1 for p in palavras if p in da_pagina)
    return encontradas / len(palavras) >= cobertura_minima


def converter_preco(texto):
    """Converte 'R$ 1.234,56' em 1234.56. Retorna None se não houver preço."""
    if not texto:
        return None
    achado = re.search(r"(\d{1,3}(?:\.\d{3})*|\d+),(\d{2})", texto)
    if not achado:
        return None
    inteiro = achado.group(1).replace(".", "")
    return float(f"{inteiro}.{achado.group(2)}")


def formatar_preco(valor):
    """1234.5 -> 'R$ 1.234,50'."""
    if valor is None:
        return None
    texto = f"{valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {texto}"


def precos_da_pagina(texto_pagina):
    """Conjunto de valores de preço que aparecem no texto da página."""
    valores = set()
    for achado in re.finditer(r"R\$\s*(\d{1,3}(?:\.\d{3})*|\d+),(\d{2})", texto_pagina):
        inteiro = achado.group(1).replace(".", "")
        valores.add(float(f"{inteiro}.{achado.group(2)}"))
    return valores


def escapar_markdown(texto):
    """Evita que 'R$ 10 ... R$ 20' seja interpretado como fórmula LaTeX."""
    return str(texto).replace("$", "\\$")
