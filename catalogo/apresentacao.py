"""Apresentação dos produtos em cartões (Streamlit)."""

import math
from functools import lru_cache

import streamlit as st
from PIL import ImageDraw, ImageFont

from catalogo import config
from catalogo.analise import caminho_imagem
from catalogo.fotos import base_cartao
from catalogo.texto import escapar_markdown

PRODUTOS_POR_PAGINA = 12

_ROTULOS_PUBLICO = {
    "masculino": "Masculino",
    "feminino": "Feminino",
    "unissex": "Unissex",
    "infantil": "Infantil",
}


@lru_cache(maxsize=1)
def _imagem_indisponivel():
    """Moldura neutra (mesma proporção dos cartões) para produtos sem foto."""
    largura, altura = config.LARGURA_CARTAO, config.ALTURA_CARTAO
    imagem, _mascara = base_cartao()  # mesmo fundo claro e cantos das fotos
    desenho = ImageDraw.Draw(imagem)
    desenho.rounded_rectangle([28, 28, largura - 28, altura - 28], 24, outline=(210, 212, 220), width=3)
    texto = "Imagem não localizada"
    fonte = _fonte(34)
    caixa = desenho.textbbox((0, 0), texto, font=fonte)
    desenho.text(
        ((largura - (caixa[2] - caixa[0])) / 2, (altura - (caixa[3] - caixa[1])) / 2),
        texto, fill=(120, 124, 140), font=fonte,
    )
    return imagem


def _fonte(tamanho):
    """Fonte do sistema com acentos; a fonte padrão do Pillow não desenha 'ã'."""
    for nome in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    return ImageFont.load_default(size=tamanho)


def exibir_cartao(produto):
    nome = produto["nome"]
    caminho = caminho_imagem(produto)

    with st.container(border=True, height="stretch"):
        if caminho:
            st.image(str(caminho), width="stretch", alt=f"Foto do produto {nome}")
        else:
            st.image(_imagem_indisponivel(), width="stretch", alt="Imagem do produto não localizada")

        st.markdown(f"**{escapar_markdown(nome)}**")
        if produto["tipo"]:
            st.caption(escapar_markdown(produto["tipo"]))

        if produto["preco_texto"]:
            st.markdown(f"### :primary[{escapar_markdown(produto['preco_texto'])}]")
        else:
            st.markdown(":gray[Preço não localizado no catálogo]")

        with st.container(horizontal=True, gap="small"):
            if produto["familia_olfativa"]:
                st.badge(produto["familia_olfativa"], icon=":material/local_florist:", color="violet")
            if produto["publico"]:
                st.badge(_ROTULOS_PUBLICO.get(produto["publico"], produto["publico"]),
                         icon=":material/person:", color="blue")
            if produto["linha"]:
                st.badge(produto["linha"], icon=":material/category:", color="gray")

        if produto["descricao"]:
            descricao = produto["descricao"]
            if len(descricao) > 180:
                descricao = descricao[:180].rsplit(" ", 1)[0] + "…"
            st.caption(escapar_markdown(descricao))

        st.space("stretch")
        origem = f":material/description: {escapar_markdown(produto['nome_arquivo'])} · página {produto['pagina']}"
        if produto.get("outras_paginas"):
            origem += f" (também na {produto['outras_paginas']})"
        if produto["codigo"]:
            origem += f" · cód. {produto['codigo']}"
        st.caption(origem)


def _mudar_pagina(chave, delta, total):
    atual = st.session_state.get(chave, 0)
    st.session_state[chave] = min(max(atual + delta, 0), total - 1)


def exibir_grade(produtos, chave):
    """Grade responsiva (até 3 colunas) com paginação."""
    if not produtos:
        return

    total_paginas = math.ceil(len(produtos) / PRODUTOS_POR_PAGINA)
    chave_pagina = f"pagina_{chave}"
    pagina = min(st.session_state.get(chave_pagina, 0), total_paginas - 1)

    inicio = pagina * PRODUTOS_POR_PAGINA
    visiveis = produtos[inicio:inicio + PRODUTOS_POR_PAGINA]

    with st.grid(3, min_column_width=230, gap="medium"):
        for produto in visiveis:
            exibir_cartao(produto)

    if total_paginas > 1:
        with st.container(horizontal=True, vertical_alignment="center", horizontal_alignment="center"):
            st.button(
                "Anterior", icon=":material/chevron_left:", key=f"ant_{chave}",
                disabled=pagina == 0, on_click=_mudar_pagina, args=(chave_pagina, -1, total_paginas),
            )
            st.caption(
                f"Página {pagina + 1} de {total_paginas} · "
                f"produtos {inicio + 1}–{inicio + len(visiveis)} de {len(produtos)}"
            )
            st.button(
                "Próxima", icon=":material/chevron_right:", key=f"prox_{chave}",
                disabled=pagina >= total_paginas - 1, on_click=_mudar_pagina,
                args=(chave_pagina, 1, total_paginas),
            )
