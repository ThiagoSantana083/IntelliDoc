"""Análises sobre os produtos já armazenados.

Todas as funções trabalham apenas com os dados estruturados do SQLite:
nada é extraído novamente do PDF e nenhuma chamada à API é feita aqui.

Análises disponíveis:
    contagem      Quantos produtos atendem ao filtro (com os nomes).
    imagens       Quantos têm foto associada e quantos não têm (com o motivo).
    duplicados    Duplicados e possíveis duplicados (nome, volume, tipo, preço).
    precos        Menor, maior e média de preço.
"""

import re
from collections import defaultdict
from pathlib import Path

from catalogo import config
from catalogo.texto import formatar_preco, normalizar

LIMITE_NOMES = 40  # nomes listados por grupo; o restante é resumido

_VOLUME = re.compile(r"(\d+(?:[.,]\d+)?)\s*(ml|kg|g|l|un\.?|unidades?)\b", re.IGNORECASE)

_MOTIVO_SEM_IMAGEM = {
    "nao_confirmada": "foto recortada, mas não confirmada como sendo do produto",
    "nao_localizada": "foto não localizada na página",
    "ambigua": "a mesma foto parecia servir a mais de um produto",
}


# --------------------------------------
# Utilidades
# --------------------------------------
def caminho_imagem(produto):
    """Caminho da foto do produto, se o arquivo existir."""
    if not produto.get("imagem"):
        return None
    caminho = Path(produto["imagem"])
    if not caminho.is_absolute():
        caminho = config.BASE_DIR / caminho
    return caminho if caminho.is_file() else None


def separar_volume(nome):
    """'Mahogany For Men 100ml' -> ('mahogany for men', '100ml')."""
    volumes = [
        f"{numero.replace(',', '.')}{unidade.lower().rstrip('.')}"
        for numero, unidade in _VOLUME.findall(nome or "")
    ]
    base = _VOLUME.sub(" ", nome or "").replace("+", " ")
    return normalizar(base), " + ".join(volumes)


def rotulo(produto):
    """Nome legível com tipo, preço e página."""
    partes = [f"**{produto['nome']}**"]
    if produto["tipo"]:
        partes.append(produto["tipo"])
    if produto["preco_texto"]:
        partes.append(produto["preco_texto"])
    pagina = f"p. {produto['pagina']}"
    if produto.get("codigo"):
        pagina += f", cód. {produto['codigo']}"
    partes.append(pagina)
    return " — ".join(partes)


def lista_nomes(produtos, limite=LIMITE_NOMES):
    linhas = [f"- {rotulo(p)}" for p in produtos[:limite]]
    if len(produtos) > limite:
        linhas.append(f"- … e mais {len(produtos) - limite} produto(s) (veja os cartões abaixo).")
    return "\n".join(linhas)


def _plural(n, singular, plural):
    return singular if n == 1 else plural


# --------------------------------------
# Contagem
# --------------------------------------
def texto_contagem(produtos, descricao):
    n = len(produtos)
    return (
        f"Há **{n} {_plural(n, 'produto', 'produtos')}** para: {descricao}.\n\n"
        + lista_nomes(produtos)
    )


# --------------------------------------
# Imagens
# --------------------------------------
def separar_por_imagem(produtos):
    com, sem = [], []
    for p in produtos:
        (com if caminho_imagem(p) else sem).append(p)
    return com, sem


def texto_imagens(produtos, descricao):
    com, sem = separar_por_imagem(produtos)
    total = len(produtos)
    percentual = round(100 * len(com) / total) if total else 0

    linhas = [
        f"Dos **{total} produtos** ({descricao}):",
        f"- :material/image: **{len(com)}** {_plural(len(com), 'possui', 'possuem')} imagem associada ({percentual}%)",
        f"- :material/hide_image: **{len(sem)}** {_plural(len(sem), 'não possui', 'não possuem')} imagem",
    ]

    if sem:
        linhas.append("\n**Produtos sem imagem:**")
        for p in sem[:LIMITE_NOMES]:
            motivo = _MOTIVO_SEM_IMAGEM.get(p.get("imagem_status"), "arquivo de imagem não encontrado")
            linhas.append(f"- {rotulo(p)} · _{motivo}_")
        if len(sem) > LIMITE_NOMES:
            linhas.append(f"- … e mais {len(sem) - LIMITE_NOMES} produto(s).")
    if com:
        linhas.append("\n**Produtos com imagem:**")
        linhas.append(lista_nomes(com))

    if sem:
        linhas.append(
            "\nAs imagens só são associadas quando o recorte do catálogo é confirmado como sendo "
            "do produto; por isso alguns ficam sem foto em vez de receber uma imagem incerta."
        )
    return "\n".join(linhas), sem + com  # cartões: primeiro os sem imagem


# --------------------------------------
# Duplicados
# --------------------------------------
def encontrar_duplicados(produtos):
    """Agrupa produtos repetidos ou semelhantes.

    Retorna um dicionário com:
      mesmo_codigo   O mesmo código aparece mais de uma vez (duplicado certo).
      provaveis      Mesmo nome, volume e tipo, com códigos diferentes
                     (possível duplicado; preço igual ou diferente).
      variacoes_tipo Mesmo nome e volume, mas tipos diferentes
                     (ex.: sabonete e hidratante) — não são duplicados.
      variacoes_volume Mesmo nome em volumes diferentes — não são duplicados.
      varias_paginas Produto que aparece em mais de uma página do catálogo
                     (já unificado em um único cartão).
    """
    por_codigo = defaultdict(list)
    por_nome_volume_tipo = defaultdict(list)
    por_nome_volume = defaultdict(list)
    por_nome = defaultdict(list)

    for p in produtos:
        base, volume = separar_volume(p["nome"])
        if p["codigo"]:
            por_codigo[(p["codigo"])].append(p)
        por_nome_volume_tipo[(base, volume, normalizar(p["tipo"]))].append(p)
        por_nome_volume[(base, volume)].append(p)
        por_nome[base].append(p)

    mesmo_codigo = [g for g in por_codigo.values() if len(g) > 1]
    ids_codigo = {p["id"] for g in mesmo_codigo for p in g}

    provaveis = []
    for grupo in por_nome_volume_tipo.values():
        codigos = {p["codigo"] for p in grupo}
        if len(grupo) > 1 and len(codigos) > 1 and not {p["id"] for p in grupo} <= ids_codigo:
            provaveis.append(grupo)

    variacoes_tipo = [
        g for g in por_nome_volume.values()
        if len(g) > 1 and len({normalizar(p["tipo"]) for p in g}) > 1
    ]
    variacoes_volume = [
        g for g in por_nome.values()
        if len(g) > 1 and len({separar_volume(p["nome"])[1] for p in g}) > 1
    ]
    varias_paginas = [p for p in produtos if p.get("outras_paginas")]

    return {
        "mesmo_codigo": mesmo_codigo,
        "provaveis": provaveis,
        "variacoes_tipo": variacoes_tipo,
        "variacoes_volume": variacoes_volume,
        "varias_paginas": varias_paginas,
    }


def _grupo_em_texto(grupo):
    return "\n".join(f"  - {rotulo(p)} · _{p['nome_arquivo']}_" for p in grupo)


def texto_duplicados(produtos, descricao):
    achados = encontrar_duplicados(produtos)
    mesmo_codigo = achados["mesmo_codigo"]
    provaveis = achados["provaveis"]
    total_dup = len(mesmo_codigo) + len(provaveis)

    linhas = [
        f"Analisei **{len(produtos)} produtos** ({descricao}) comparando código, nome, "
        "volume, tipo e preço."
    ]

    if total_dup == 0:
        linhas.append("\n:material/check_circle: **Nenhum produto duplicado foi encontrado na listagem.**")
    else:
        linhas.append(
            f"\n:material/warning: Encontrei **{total_dup} {_plural(total_dup, 'grupo', 'grupos')}** "
            "de possíveis duplicados:"
        )

    if mesmo_codigo:
        linhas.append(f"\n**Mesmo código repetido ({len(mesmo_codigo)}):**")
        for grupo in mesmo_codigo:
            linhas.append(f"- Código {grupo[0]['codigo']} aparece {len(grupo)} vezes:")
            linhas.append(_grupo_em_texto(grupo))

    if provaveis:
        linhas.append(f"\n**Mesmo nome, volume e tipo com códigos diferentes ({len(provaveis)}):**")
        for grupo in provaveis:
            precos = {p["preco_texto"] for p in grupo}
            situacao = "mesmo preço" if len(precos) == 1 else "preços diferentes"
            linhas.append(f"- {grupo[0]['nome']} ({situacao}) — pode ser o mesmo item ou itens de linhas diferentes:")
            linhas.append(_grupo_em_texto(grupo))

    # Semelhanças que NÃO são duplicados, para deixar a análise transparente
    variacoes_tipo = achados["variacoes_tipo"]
    if variacoes_tipo:
        linhas.append(
            f"\n**Não são duplicados:** {len(variacoes_tipo)} "
            f"{_plural(len(variacoes_tipo), 'nome aparece', 'nomes aparecem')} com o mesmo volume, "
            "mas em tipos de produto diferentes:"
        )
        for grupo in variacoes_tipo[:15]:
            tipos = ", ".join(sorted({p["tipo"] or "tipo não informado" for p in grupo}))
            linhas.append(f"- {grupo[0]['nome']}: {tipos}")

    variacoes_volume = achados["variacoes_volume"]
    if variacoes_volume:
        nomes = ", ".join(sorted({separar_volume(g[0]["nome"])[0].title() for g in variacoes_volume})[:15])
        linhas.append(
            f"\n{len(variacoes_volume)} {_plural(len(variacoes_volume), 'linha tem', 'linhas têm')} "
            f"o mesmo nome em volumes diferentes (variações, não duplicados): {nomes}."
        )

    if achados["varias_paginas"]:
        linhas.append(
            f"\n{len(achados['varias_paginas'])} produto(s) aparecem em mais de uma página do catálogo "
            "e já foram unificados em um único cartão."
        )

    envolvidos = [p for g in mesmo_codigo + provaveis for p in g]
    vistos, cartoes = set(), []
    for p in envolvidos:
        if p["id"] not in vistos:
            vistos.add(p["id"])
            cartoes.append(p)
    return "\n".join(linhas), cartoes


# --------------------------------------
# Preços
# --------------------------------------
def texto_precos(produtos, descricao):
    com_preco = [p for p in produtos if p["preco_valor"] is not None]
    if not com_preco:
        return f"Nenhum dos {len(produtos)} produtos ({descricao}) tem preço identificado.", produtos

    valores = [p["preco_valor"] for p in com_preco]
    minimo, maximo = min(valores), max(valores)
    media = sum(valores) / len(valores)
    mais_baratos = [p for p in com_preco if p["preco_valor"] == minimo]
    mais_caros = [p for p in com_preco if p["preco_valor"] == maximo]

    linhas = [
        f"Preços dos **{len(com_preco)} produtos** com preço identificado ({descricao}):",
        f"- Menor preço: **{formatar_preco(minimo)}** — " + ", ".join(p["nome"] for p in mais_baratos),
        f"- Maior preço: **{formatar_preco(maximo)}** — " + ", ".join(p["nome"] for p in mais_caros),
        f"- Preço médio: **{formatar_preco(media)}**",
    ]
    sem_preco = len(produtos) - len(com_preco)
    if sem_preco:
        linhas.append(f"- {sem_preco} produto(s) sem preço identificado não entraram no cálculo.")

    ordenados = sorted(com_preco, key=lambda p: p["preco_valor"])
    return "\n".join(linhas), ordenados
