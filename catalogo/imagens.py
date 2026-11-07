"""Recorte das fotografias dos produtos a partir das páginas do PDF.

Por que recortar a página em vez de extrair as imagens incorporadas?
Nos catálogos analisados, cada página é uma imagem fatiada em blocos
(uma grade de 7 imagens que cobre a página inteira). As imagens
incorporadas não correspondem a produtos isolados. Por isso a foto de cada
produto é recortada da própria página renderizada pelo PyMuPDF: continua
sendo a imagem original do catálogo, sem nenhuma imagem externa.

Localização da foto (sempre a partir do texto e do layout da página):
  1. O modelo de visão recebe a página e as posições dos textos e indica
     uma caixa aproximada da foto de cada produto (feito em extracao.py).
  2. A região é ampliada e o modelo refina a caixa em um recorte com zoom,
     onde a localização é muito mais precisa (duas passagens).
  3. Um segundo modelo confere se o recorte final mostra realmente o
     produto. Se não confirmar, o cartão é exibido sem fotografia.
"""

import base64
import io
from functools import lru_cache

import pymupdf
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from PIL import Image, ImageDraw, ImageStat
from pydantic import BaseModel, Field

from catalogo import config

TAMANHO_ZOOM = 800  # lado maior (px) do recorte enviado para refinamento


# --------------------------------------
# Modelos de saída estruturada
# --------------------------------------
class CaixaRefinada(BaseModel):
    encontrado: bool = Field(description="True se a embalagem do produto está visível no recorte.")
    x0: int
    y0: int
    x1: int
    y1: int


class Verificacao(BaseModel):
    corresponde: bool = Field(
        description="True somente se o produto principal do recorte é o produto indicado."
    )
    motivo: str


@lru_cache(maxsize=1)
def _modelo_refinamento():
    return ChatOpenAI(model=config.MODELO_VISAO, temperature=0, max_retries=4).with_structured_output(
        CaixaRefinada, method="json_schema", strict=True
    )


@lru_cache(maxsize=1)
def _modelo_verificacao():
    return ChatOpenAI(model=config.MODELO_VERIFICACAO, temperature=0, max_retries=4).with_structured_output(
        Verificacao, method="json_schema", strict=True
    )


# --------------------------------------
# Utilidades de imagem
# --------------------------------------
def renderizar(pagina, zoom, recorte=None):
    """Renderiza a página (ou parte dela) como imagem PIL RGB."""
    pix = pagina.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=recorte, alpha=False)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def com_grade(imagem, passo=100):
    """Desenha uma grade de coordenadas, que ajuda o modelo a medir posições."""
    copia = imagem.copy()
    desenho = ImageDraw.Draw(copia)
    largura, altura = copia.size
    for x in range(0, largura, passo):
        desenho.line([(x, 0), (x, altura)], fill=(255, 0, 0), width=1)
        desenho.text((x + 2, 2), str(x), fill=(255, 0, 0))
    for y in range(0, altura, passo):
        desenho.line([(0, y), (largura, y)], fill=(255, 0, 0), width=1)
        desenho.text((2, y + 2), str(y), fill=(255, 0, 0))
    return copia


def para_data_url(imagem, qualidade=90):
    buffer = io.BytesIO()
    imagem.save(buffer, format="JPEG", quality=qualidade)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def mensagem_com_imagem(texto, imagem, detalhe="high"):
    return HumanMessage(
        content=[
            {"type": "text", "text": texto},
            {"type": "image_url", "image_url": {"url": para_data_url(imagem), "detail": detalhe}},
        ]
    )


def expandir(caixa, fator, limite, minimo=0.0):
    """Aumenta a caixa em `fator` da largura/altura (ou `minimo` pontos) para cada lado."""
    dx, dy = max(caixa.width * fator, minimo), max(caixa.height * fator, minimo)
    return pymupdf.Rect(caixa.x0 - dx, caixa.y0 - dy, caixa.x1 + dx, caixa.y1 + dy) & limite


def iou(a, b):
    """Interseção sobre união entre duas caixas."""
    inter = pymupdf.Rect(a) & pymupdf.Rect(b)
    if inter.is_empty:
        return 0.0
    area_inter = inter.width * inter.height
    uniao = a.width * a.height + b.width * b.height - area_inter
    return area_inter / uniao if uniao else 0.0


# --------------------------------------
# Refinamento e verificação
# --------------------------------------
def refinar_caixa(pagina, caixa, descricao_produto, ancora=None, margem=0.8, minimo=0.0):
    """Refina a caixa da foto do produto em uma região ampliada da página.

    caixa e ancora estão em coordenadas do PDF (pontos).
    Retorna a nova caixa (pontos) ou None se o produto não for encontrado.
    """
    regiao = expandir(caixa, margem, pagina.rect, minimo)
    if regiao.is_empty or regiao.width < 5 or regiao.height < 5:
        return None

    zoom = TAMANHO_ZOOM / max(regiao.width, regiao.height)
    imagem = renderizar(pagina, zoom, regiao)
    largura, altura = imagem.size

    dica = ""
    if ancora is not None and regiao.intersects(ancora):
        ax = int((ancora.x0 - regiao.x0) * zoom)
        ay = int((ancora.y0 - regiao.y0) * zoom)
        dica = f" O texto com o código do produto começa perto de x={ax}, y={ay} deste recorte."

    instrucao = (
        f"Recorte ampliado de uma página de catálogo ({largura}x{altura} px, grade vermelha "
        f"a cada 100 px). Retorne a caixa justa, em pixels deste recorte, que envolve a "
        f"embalagem INTEIRA do produto '{descricao_produto}', da tampa à base, sem incluir "
        f"textos, pessoas ou outros produtos.{dica} Se essa embalagem não estiver visível, "
        f"retorne encontrado=false."
    )
    resposta = _modelo_refinamento().invoke([mensagem_com_imagem(instrucao, com_grade(imagem))])

    if not resposta.encontrado:
        return None

    x0, x1 = sorted((resposta.x0, resposta.x1))
    y0, y1 = sorted((resposta.y0, resposta.y1))
    nova = pymupdf.Rect(
        regiao.x0 + x0 / zoom,
        regiao.y0 + y0 / zoom,
        regiao.x0 + x1 / zoom,
        regiao.y0 + y1 / zoom,
    ) & regiao

    if nova.is_empty or nova.width < 8 or nova.height < 8:
        return None
    return nova


def verificar_recorte(imagem, descricao_produto):
    """Confirma com um segundo modelo se o recorte mostra o produto."""
    instrucao = (
        f"Esta imagem foi recortada de um catálogo. Ela deveria mostrar a embalagem do produto "
        f"'{descricao_produto}'. Responda corresponde=true somente se a embalagem principal "
        f"visível for desse produto (rótulo, formato e cores coerentes com o nome). Responda "
        f"false se mostrar outro produto, apenas texto, pessoas ou somente o fundo."
    )
    resposta = _modelo_verificacao().invoke([mensagem_com_imagem(instrucao, imagem, detalhe="high")])
    return resposta.corresponde


# --------------------------------------
# Salvamento
# --------------------------------------
def montar_cartao(imagem):
    """Centraliza a foto em uma moldura 4:5 sem distorcer a proporção.

    A cor de fundo é a média da borda da própria foto, para que a moldura
    fique discreta tanto em fundos claros quanto escuros.
    """
    largura, altura = config.LARGURA_CARTAO, config.ALTURA_CARTAO
    borda = Image.new("RGB", (imagem.width * 2 + imagem.height * 2, 1))
    borda.paste(imagem.crop((0, 0, imagem.width, 1)), (0, 0))
    borda.paste(imagem.crop((0, imagem.height - 1, imagem.width, imagem.height)), (imagem.width, 0))
    esquerda = imagem.crop((0, 0, 1, imagem.height)).rotate(90, expand=True)
    direita = imagem.crop((imagem.width - 1, 0, imagem.width, imagem.height)).rotate(90, expand=True)
    borda.paste(esquerda, (imagem.width * 2, 0))
    borda.paste(direita, (imagem.width * 2 + imagem.height, 0))
    cor_fundo = tuple(int(v) for v in ImageStat.Stat(borda).mean)

    escala = min(largura * 0.94 / imagem.width, altura * 0.94 / imagem.height)
    nova = imagem.resize(
        (max(1, int(imagem.width * escala)), max(1, int(imagem.height * escala))),
        Image.Resampling.LANCZOS,
    )
    cartao = Image.new("RGB", (largura, altura), cor_fundo)
    cartao.paste(nova, ((largura - nova.width) // 2, (altura - nova.height) // 2))
    return cartao


def localizar_foto(pagina, caixa_aproximada, descricao_produto, ancora=None):
    """Refina a caixa aproximada em até três chamadas.

    1ª passagem: região ampla ao redor da caixa aproximada (que pode estar
       deslocada); se o produto não aparecer, tenta uma região ainda maior.
    2ª passagem: região justa ao redor do resultado, para ajustar as bordas.
       Se ela falhar, mantém o resultado da 1ª passagem.
    """
    caixa = refinar_caixa(pagina, caixa_aproximada, descricao_produto, ancora, 0.8, minimo=80)
    if caixa is None:
        caixa = refinar_caixa(pagina, caixa_aproximada, descricao_produto, ancora, 2.0, minimo=160)
    if caixa is None:
        return None
    ajustada = refinar_caixa(pagina, caixa, descricao_produto, ancora, 0.35, minimo=15)
    return ajustada or caixa


def recortar_foto(pagina, caixa, folga=0.10):
    """Renderiza a região final da foto em boa resolução."""
    regiao = expandir(caixa, folga, pagina.rect)
    zoom = min(4.0, max(2.0, 900 / max(regiao.width, regiao.height)))
    return renderizar(pagina, zoom, regiao)


def salvar_foto(imagem, destino):
    """Salva a versão padronizada para o cartão (JPEG)."""
    destino.parent.mkdir(parents=True, exist_ok=True)
    montar_cartao(imagem).save(destino, format="JPEG", quality=90, optimize=True)
    return destino
