"""Identificação e estruturação dos produtos de cada página do PDF.

Para cada página com preços:
  1. O texto e a posição de cada bloco de texto são lidos com o PyMuPDF.
  2. O modelo de visão recebe a página renderizada e esses blocos e
     devolve os produtos estruturados e a caixa aproximada de cada foto.
  3. Cada campo é conferido com o texto real da página. O que não puder
     ser comprovado é descartado (nada de preços, nomes ou famílias
     inventados).
  4. A foto é refinada, verificada e salva (ver imagens.py).
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

import pymupdf
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from catalogo import config, imagens
from catalogo.texto import (
    converter_preco,
    normalizar,
    precos_da_pagina,
    texto_presente,
    tokens,
    tokens_presentes,
)


# --------------------------------------
# Estrutura esperada do modelo
# --------------------------------------
class Caixa(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int


class ProdutoExtraido(BaseModel):
    codigo: Optional[str] = Field(description="Código numérico do produto, se houver.")
    nome: str = Field(description="Nome comercial com volume, sem o tipo. Ex.: 'Frangipani Flower 100ml'.")
    tipo: Optional[str] = Field(description="Tipo do produto como no texto. Ex.: 'Fragrância Desodorante Corporal'.")
    preco: Optional[str] = Field(description="Preço exatamente como no texto. Ex.: 'R$ 159,00'.")
    familia_olfativa: Optional[str] = Field(description="Família olfativa, somente se indicada na página.")
    publico: Optional[Literal["masculino", "feminino", "unissex", "infantil"]]
    evidencia_publico: Optional[str] = Field(description="Trecho exato do texto que indica o público.")
    linha: Optional[str] = Field(description="Linha ou coleção, somente se indicada explicitamente.")
    descricao: Optional[str] = Field(description="Texto descritivo do produto copiado literalmente.")
    caixa_foto: Optional[Caixa] = Field(description="Caixa aproximada da foto da embalagem, em pixels.")


class PaginaExtraida(BaseModel):
    produtos: list[ProdutoExtraido]


@lru_cache(maxsize=1)
def _modelo_pagina():
    return ChatOpenAI(model=config.MODELO_VISAO, temperature=0, max_retries=4).with_structured_output(
        PaginaExtraida, method="json_schema", strict=True
    )


INSTRUCOES = """Você analisa UMA página de um catálogo de produtos em PDF.
A imagem tem {largura}x{altura} pixels e uma grade vermelha a cada 100 px.
Abaixo estão os blocos de texto reais da página com suas posições [x0,y0,x1,y1] em pixels da imagem.

Extraia SOMENTE os produtos à venda que aparecem no TEXTO (normalmente com código e preço).
Regras:
- Copie os valores exatamente como aparecem no texto. Nunca invente ou complete informações.
- nome: nome comercial com volume/tamanho, sem o tipo (ex.: 'Frangipani Flower 100ml').
- tipo: descrição do tipo (ex.: 'Fragrância Desodorante Corporal', 'Kit Especial'), ou null.
- preco: o preço do produto exatamente como no texto (ex.: 'R$ 159,00'), ou null.
- familia_olfativa: somente quando a página indicar a família olfativa do produto
  (geralmente a linha com marcador logo após o preço, ex.: 'Oriental Baunilha').
  Use null para características que não são família olfativa (ex.: 'Sem Álcool Etílico', itens de kit).
- publico: masculino/feminino/unissex/infantil SOMENTE se houver indicação explícita na página
  (etiqueta como 'MASCULINO', título de seção ou nome como 'For Men'). Caso contrário null.
  evidencia_publico: o trecho exato do texto que comprova o público (ou null).
- linha: linha/coleção somente se escrita na página (ex.: 'Coleção Jardim Secreto' -> 'Jardim Secreto').
- descricao: parágrafo descritivo do produto copiado literalmente, ou null.
- caixa_foto: caixa aproximada (pixels da imagem) da foto da EMBALAGEM deste produto.
  Use a posição do texto para associar cada foto ao produto correto: a foto costuma ficar
  ao lado ou acima do texto do produto. Se não houver uma foto clara deste produto
  específico, use null. Nunca use a mesma foto para dois produtos diferentes.
Se a página não tiver produtos à venda, retorne uma lista vazia.

Blocos de texto:
{blocos}"""


# --------------------------------------
# Validação
# --------------------------------------
def pagina_tem_produtos(texto):
    return bool(re.search(r"R\$\s*\d", texto))


def _validar_produto(bruto, texto_pagina, precos_validos):
    """Mantém apenas o que pode ser comprovado no texto da página."""
    nome = (bruto.nome or "").strip()
    if not nome:
        return None

    exigencia = 1.0 if len(tokens(nome)) <= 4 else 0.85
    if not tokens_presentes(nome, texto_pagina, exigencia):
        return None

    codigo = (bruto.codigo or "").strip() or None
    if codigo and not re.search(rf"(?<!\d){re.escape(codigo)}(?!\d)", texto_pagina):
        codigo = None

    preco_valor = converter_preco(bruto.preco)
    preco_texto = bruto.preco.strip() if bruto.preco else None
    if preco_valor is None or preco_valor not in precos_validos:
        preco_valor, preco_texto = None, None

    # Um item sem código e sem preço não é tratado como produto à venda.
    if codigo is None and preco_valor is None:
        return None

    def campo(valor, verificador):
        valor = (valor or "").strip()
        return valor if valor and verificador(valor) else None

    familia = campo(bruto.familia_olfativa, lambda v: texto_presente(v, texto_pagina))
    tipo = campo(bruto.tipo, lambda v: tokens_presentes(v, texto_pagina))
    linha = campo(bruto.linha, lambda v: tokens_presentes(v, texto_pagina))
    descricao = campo(bruto.descricao, lambda v: tokens_presentes(v, texto_pagina, 0.9))

    publico = bruto.publico
    if publico and not (bruto.evidencia_publico and texto_presente(bruto.evidencia_publico, texto_pagina)):
        publico = None

    return {
        "codigo": codigo,
        "nome": nome,
        "tipo": tipo,
        "preco_texto": preco_texto,
        "preco_valor": preco_valor,
        "familia_olfativa": familia,
        "publico": publico,
        "linha": linha,
        "descricao": descricao,
    }


def _blocos_de_texto(pagina, zoom):
    linhas = []
    for x0, y0, x1, y1, texto, _n, tipo in pagina.get_text("blocks"):
        if tipo != 0 or not texto.strip():
            continue
        caixa = [int(x0 * zoom), int(y0 * zoom), int(x1 * zoom), int(y1 * zoom)]
        linhas.append(f"{caixa} {' '.join(texto.split())}")
    return "\n".join(linhas)


def _ancora(pagina, produto):
    """Posição do código (ou do nome) do produto na página."""
    for termo in (produto["codigo"], produto["nome"]):
        if termo:
            achados = pagina.search_for(termo)
            if achados:
                return achados[0]
    return None


def _nome_arquivo_imagem(produto, numero_pagina, indice):
    base = produto["codigo"] or re.sub(r"[^a-z0-9]+", "-", normalizar(produto["nome"]))[:40]
    return f"p{numero_pagina:03d}_{indice:02d}_{base}.jpg"


# --------------------------------------
# Processamento de uma página
# --------------------------------------
def processar_pagina(caminho_pdf, indice_pagina, pasta_imagens):
    """Extrai os produtos de uma página. Retorna uma lista de dicionários.

    Abre o próprio documento para poder ser executada em paralelo.
    """
    numero_pagina = indice_pagina + 1
    documento = pymupdf.open(caminho_pdf)
    try:
        pagina = documento[indice_pagina]
        texto_pagina = pagina.get_text()
        if not pagina_tem_produtos(texto_pagina):
            return []

        zoom = config.DPI_ANALISE / 72
        imagem_pagina = imagens.renderizar(pagina, zoom)
        largura, altura = imagem_pagina.size

        instrucao = INSTRUCOES.format(
            largura=largura, altura=altura, blocos=_blocos_de_texto(pagina, zoom)
        )
        resultado = _modelo_pagina().invoke(
            [imagens.mensagem_com_imagem(instrucao, imagens.com_grade(imagem_pagina))]
        )

        precos_validos = precos_da_pagina(texto_pagina)
        produtos = []
        caixas = []
        vistos = set()

        for bruto in resultado.produtos:
            produto = _validar_produto(bruto, texto_pagina, precos_validos)
            if produto is None:
                continue

            chave = produto["codigo"] or normalizar(produto["nome"])
            if chave in vistos:
                continue
            vistos.add(chave)

            produto["pagina"] = numero_pagina
            produto["imagem"] = None
            produto["imagem_status"] = "nao_localizada"

            caixa = None
            if bruto.caixa_foto:
                c = bruto.caixa_foto
                caixa = pymupdf.Rect(
                    min(c.x0, c.x1) / zoom, min(c.y0, c.y1) / zoom,
                    max(c.x0, c.x1) / zoom, max(c.y0, c.y1) / zoom,
                ) & pagina.rect
                if caixa.is_empty or caixa.width < 10 or caixa.height < 10:
                    caixa = None

            produtos.append(produto)
            caixas.append(caixa)

        # Refinamento da posição de cada foto (duas passagens com zoom)
        for posicao, produto in enumerate(produtos):
            caixa = caixas[posicao]
            if caixa is None:
                continue
            descricao = " ".join(p for p in (produto["tipo"], produto["nome"]) if p)
            caixas[posicao] = imagens.localizar_foto(pagina, caixa, descricao, _ancora(pagina, produto))

        # Fotos praticamente iguais para produtos diferentes são ambíguas
        for i in range(len(produtos)):
            for j in range(i + 1, len(produtos)):
                if caixas[i] is not None and caixas[j] is not None and imagens.iou(caixas[i], caixas[j]) > 0.6:
                    produtos[i]["imagem_status"] = produtos[j]["imagem_status"] = "ambigua"

        # Verificação e salvamento
        for posicao, produto in enumerate(produtos):
            caixa = caixas[posicao]
            if caixa is None or produto["imagem_status"] == "ambigua":
                continue
            descricao = " ".join(p for p in (produto["tipo"], produto["nome"]) if p)
            foto = imagens.recortar_foto(pagina, caixa)
            if not imagens.verificar_recorte(foto, descricao):
                produto["imagem_status"] = "nao_confirmada"
                continue
            destino = Path(pasta_imagens) / _nome_arquivo_imagem(produto, numero_pagina, posicao)
            imagens.salvar_foto(foto, destino)
            try:
                produto["imagem"] = destino.relative_to(config.BASE_DIR).as_posix()
            except ValueError:
                produto["imagem"] = destino.as_posix()
            produto["imagem_status"] = "ok"

        return produtos
    finally:
        documento.close()
