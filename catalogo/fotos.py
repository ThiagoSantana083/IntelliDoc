"""Fotos isoladas dos produtos, no estilo de loja virtual.

Em vez de exibir um recorte retangular da página (com fundo da foto,
pedaços de texto e, às vezes, o produto cortado), o produto é separado
do fundo e centralizado sobre um fundo claro com uma sombra suave.

Como funciona (tudo localmente, sem custo de API, exceto a conferência):
  1. O fundo é removido com o modelo de segmentação "isnet-general-use"
     (biblioteca rembg, executada no próprio computador) em uma área bem
     maior que a posição gravada do produto.
  2. A posição gravada serve só para identificar QUAL objeto é o produto;
     o contorno completo dele é mantido, nunca limitado à caixa. Se o
     contorno encosta na borda da área analisada, a área é ampliada.
  3. Textos e linhas do PDF são retirados; tampas e bombas claras sobre
     fundo branco são recuperadas.
  4. O gpt-4.1-mini confere se a foto isolada mostra o produto inteiro e
     sem outros objetos (mãos, frutas, decorações). Se não aprovar, usa o
     recorte da página clareado a partir do contorno completo — o produto
     nunca é cortado.

A foto continua sendo a imagem original do catálogo: nada é gerado ou
buscado na internet.

Para atualizar as fotos de produtos já processados (sem nova extração):
    python -m catalogo.fotos
    python -m catalogo.fotos --forcar   # refaz também as já isoladas
"""

import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pymupdf
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from catalogo import config, imagens, repositorio

COR_FUNDO = (243, 244, 246)        # fundo claro do cartão
COR_SOMBRA = (196, 198, 204)
RAIO_CANTOS = 36

_trava_modelo = threading.Lock()   # o modelo de segmentação roda um por vez (memória)


# --------------------------------------
# Geometria
# --------------------------------------
def crescer(caixa, fator, limite):
    caixa = pymupdf.Rect(caixa)
    dx, dy = caixa.width * fator, caixa.height * fator
    return pymupdf.Rect(caixa.x0 - dx, caixa.y0 - dy, caixa.x1 + dx, caixa.y1 + dy) & limite


def caixa_para_texto(caixa):
    return ",".join(f"{v:.2f}" for v in (caixa.x0, caixa.y0, caixa.x1, caixa.y1))


def texto_para_caixa(texto):
    try:
        valores = [float(v) for v in texto.split(",")]
        caixa = pymupdf.Rect(*valores)
        return None if caixa.is_empty else caixa
    except (AttributeError, ValueError, TypeError):
        return None


# --------------------------------------
# Remoção do fundo
# --------------------------------------
@lru_cache(maxsize=1)
def _sessao():
    from rembg import new_session  # importação tardia: o modelo é carregado só quando necessário

    return new_session(config.MODELO_RECORTE)


def _completar_em_fundo_branco(rgb, alfa, entorno_px):
    """Em fundo branco, recupera partes claras que o modelo perde.

    Válvulas, bombas e tampas brancas ou transparentes sobre fundo branco
    costumam sumir na segmentação. Quando o entorno imediato do produto é
    branco, tudo o que difere do branco (com os buracos internos
    preenchidos) também é considerado parte do produto.
    """
    x0, y0, x1, y1 = entorno_px
    x1, y1 = min(x1, rgb.shape[1]), min(y1, rgb.shape[0])
    faixa = 4
    borda = np.concatenate([
        rgb[y0:y0 + faixa, x0:x1].reshape(-1, 3), rgb[max(y1 - faixa, 0):y1, x0:x1].reshape(-1, 3),
        rgb[y0:y1, x0:x0 + faixa].reshape(-1, 3), rgb[y0:y1, max(x1 - faixa, 0):x1].reshape(-1, 3),
    ])
    if len(borda) == 0 or (borda.min(axis=1) > 235).mean() < 0.6:
        return alfa  # fundo fotográfico: fica só a segmentação do modelo

    diferente = ((255 - rgb.min(axis=2)) > 22).astype(np.uint8) * 255
    diferente = cv2.morphologyEx(diferente, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    # Preenche o interior (ex.: corpo branco contornado por sombras/reflexos)
    com_margem = cv2.copyMakeBorder(diferente, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inundado = com_margem.copy()
    cv2.floodFill(inundado, np.zeros((inundado.shape[0] + 2, inundado.shape[1] + 2), np.uint8), (0, 0), 255)
    solido = (com_margem | cv2.bitwise_not(inundado))[1:-1, 1:-1]

    # Só completa perto do produto, para não puxar fotos ou objetos vizinhos
    perto = np.zeros_like(solido)
    perto[y0:y1, x0:x1] = solido[y0:y1, x0:x1]
    return np.maximum(alfa, perto)


def _partes_do_produto(alfa, caixa_px, kit=False):
    """Componentes da máscara que formam o produto (sem limitar à caixa).

    A caixa serve apenas para identificar qual objeto é o produto; o
    contorno completo do objeto é mantido, mesmo que ultrapasse a caixa.
      1. Principais: componentes com a maior parte dentro da caixa, ou que
         cobrem boa parte da caixa (quando a caixa está deslocada).
      2. Complementos: peças alinhadas logo acima ou abaixo dos principais
         (bombas, válvulas, tampas).
      3. Em kits, todos os itens próximos da caixa (vários objetos separados).
    Retorna a máscara booleana ou None se nada for encontrado.
    """
    bx0, by0, bx1, by1 = caixa_px
    area_caixa = max(1, (bx1 - bx0) * (by1 - by0))
    total, rotulos, estat, _ = cv2.connectedComponentsWithStats((alfa > 100).astype(np.uint8), 8)
    if total <= 1:
        return None
    maior = estat[1:, cv2.CC_STAT_AREA].max()

    # Caixa ampliada usada para os itens de kits
    lx = int((bx1 - bx0) * 0.5)
    ly = int((by1 - by0) * 0.5)
    kx0, ky0, kx1, ky1 = max(bx0 - lx, 0), max(by0 - ly, 0), bx1 + lx, by1 + ly

    manter = np.zeros(alfa.shape, bool)
    principais, restantes = [], []
    for i in range(1, total):
        area = estat[i, cv2.CC_STAT_AREA]
        if area < 0.02 * maior:
            continue
        componente = rotulos == i
        dentro = componente[by0:by1, bx0:bx1].sum()
        perto = componente[ky0:ky1, kx0:kx1].sum()
        if dentro / area >= 0.5 or dentro >= 0.25 * area_caixa or (kit and perto / area >= 0.3):
            manter |= componente
            principais.append(i)
        else:
            restantes.append(i)
    if not principais:
        return None

    mx0 = min(estat[i, cv2.CC_STAT_LEFT] for i in principais)
    my0 = min(estat[i, cv2.CC_STAT_TOP] for i in principais)
    mx1 = max(estat[i, cv2.CC_STAT_LEFT] + estat[i, cv2.CC_STAT_WIDTH] for i in principais)
    my1 = max(estat[i, cv2.CC_STAT_TOP] + estat[i, cv2.CC_STAT_HEIGHT] for i in principais)

    for i in restantes:
        x, y = estat[i, cv2.CC_STAT_LEFT], estat[i, cv2.CC_STAT_TOP]
        w, h = estat[i, cv2.CC_STAT_WIDTH], estat[i, cv2.CC_STAT_HEIGHT]
        sobreposicao = max(0, min(x + w, mx1) - max(x, mx0)) / max(w, 1)
        distancia = max(my0 - (y + h), y - my1, 0)
        if sobreposicao >= 0.6 and distancia <= 0.12 * (my1 - my0):
            manter |= rotulos == i
    return manter


def _segmentar(pagina, caixa, regiao, kit):
    """Segmenta o produto em uma região. Retorna (rgba, alfa, zoom) ou None."""
    from rembg import remove

    zoom = min(4.0, 1600 / max(regiao.width, regiao.height))
    pix = pagina.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=regiao, alpha=False)
    imagem = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

    with _trava_modelo:
        rgba = np.array(remove(imagem, session=_sessao()))
    # O modelo zera a cor do que considera fundo; as cores vêm sempre do original
    rgba[:, :, :3] = np.array(imagem)

    def em_pixels(r):
        r = pymupdf.Rect(r) & regiao
        return (
            int((r.x0 - regiao.x0) * zoom), int((r.y0 - regiao.y0) * zoom),
            int((r.x1 - regiao.x0) * zoom) + 1, int((r.y1 - regiao.y0) * zoom) + 1,
        )

    alfa = _completar_em_fundo_branco(
        np.array(imagem), rgba[:, :, 3].copy(), em_pixels(crescer(caixa, 0.06, regiao))
    )

    # Textos e linhas finas do layout não fazem parte do produto
    for bloco in pagina.get_text("blocks"):
        r = pymupdf.Rect(bloco[:4])
        if r.intersects(regiao):
            x0, y0, x1, y1 = em_pixels(r)
            alfa[y0:y1, x0:x1] = 0
    for desenho in pagina.get_drawings():
        r = desenho["rect"]
        if (r.width < 3 or r.height < 3) and r.intersects(regiao):
            x0, y0, x1, y1 = em_pixels(pymupdf.Rect(r.x0 - 1, r.y0 - 1, r.x1 + 1, r.y1 + 1))
            alfa[y0:y1, x0:x1] = 0

    caixa_px = em_pixels(caixa)
    manter = _partes_do_produto(alfa, caixa_px, kit)
    if manter is None:
        return None
    alfa = np.where(manter, alfa, 0)
    # Interior sólido (objetos escuros ficavam translúcidos e desbotados no
    # fundo claro); só as bordas mantêm a transição suave
    alfa = np.where(alfa > 128, 255, alfa).astype(np.uint8)
    # Preenche buracos internos (reflexos brilhantes viravam manchas claras)
    solido = cv2.copyMakeBorder((alfa > 128).astype(np.uint8) * 255, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    inundado = solido.copy()
    cv2.floodFill(inundado, np.zeros((inundado.shape[0] + 2, inundado.shape[1] + 2), np.uint8), (0, 0), 255)
    buracos = cv2.bitwise_not(inundado)[1:-1, 1:-1] > 0
    alfa[buracos] = 255

    # Segmentação muito pequena em relação à caixa: provavelmente falhou
    area_caixa = max(1, (caixa_px[2] - caixa_px[0]) * (caixa_px[3] - caixa_px[1]))
    if (alfa > 100).sum() < 0.12 * area_caixa:
        return None
    rgba[:, :, 3] = alfa
    return rgba, alfa, zoom


def isolar_produto(pagina, caixa, kit=False):
    """Separa o produto do fundo sem cortá-lo.

    Retorna (imagem RGBA, contorno, encosta_na_borda, contorno_inicial).
    Quando o contorno encosta na borda da área analisada (o produto
    continua além dela), a área é ampliada e a análise é refeita. Se ainda
    encostar, encosta_na_borda=True e a foto isolada não deve ser usada.
    contorno_inicial é o da primeira tentativa (área menor, com 35% de
    folga), usado quando não há outro contorno válido.
    """
    caixa = pymupdf.Rect(caixa)
    limite = pagina.rect
    area_caixa = max(1.0, caixa.width * caixa.height)
    resultado = (None, None, False)
    inicial = None
    for fator in (0.35, 0.8, 1.6):
        regiao = crescer(caixa, fator, limite)
        segmentado = _segmentar(pagina, caixa, regiao, kit)
        if segmentado is None:
            break
        rgba, alfa, zoom = segmentado
        ys, xs = np.where(alfa > 10)
        contorno = pymupdf.Rect(
            regiao.x0 + xs.min() / zoom, regiao.y0 + ys.min() / zoom,
            regiao.x0 + (xs.max() + 1) / zoom, regiao.y0 + (ys.max() + 1) / zoom,
        )
        # Encosta na borda da região em um lado que não é a borda da página?
        margem = 3 / zoom
        encosta = (
            (contorno.x0 - regiao.x0 < margem and regiao.x0 > limite.x0 + 0.5)
            or (regiao.x1 - contorno.x1 < margem and regiao.x1 < limite.x1 - 0.5)
            or (contorno.y0 - regiao.y0 < margem and regiao.y0 > limite.y0 + 0.5)
            or (regiao.y1 - contorno.y1 < margem and regiao.y1 < limite.y1 - 0.5)
        )
        # Contorno muito maior que o produto: a segmentação pegou a cena
        # (pessoa, mesa, paisagem). Fica com a tentativa anterior, marcada
        # como "encosta" para usar o recorte em vez da foto isolada.
        if contorno.width * contorno.height > 3.5 * area_caixa:
            if resultado[0] is not None:
                return resultado[0], resultado[1], True, inicial
            return None, None, True, crescer(caixa, 0.3, limite)

        imagem = Image.fromarray(rgba).crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
        resultado = (imagem, contorno, encosta)
        if inicial is None:
            inicial = contorno
        if not encosta:
            break
    return (*resultado, inicial)


# --------------------------------------
# Composição do cartão
# --------------------------------------
def clarear(imagem):
    """Deixa a foto um pouco mais clara e viva, preservando a transparência."""
    alfa = imagem.getchannel("A") if imagem.mode == "RGBA" else None
    rgb = imagem.convert("RGB")
    rgb = ImageEnhance.Brightness(rgb).enhance(1.06)
    rgb = ImageEnhance.Contrast(rgb).enhance(1.04)
    rgb = ImageEnhance.Sharpness(rgb).enhance(1.15)
    if alfa is None:
        return rgb
    rgb.putalpha(alfa)
    return rgb


def base_cartao():
    """Fundo claro com cantos arredondados (transparente fora dos cantos)."""
    largura, altura = config.LARGURA_CARTAO, config.ALTURA_CARTAO
    base = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    mascara = Image.new("L", (largura, altura), 0)
    ImageDraw.Draw(mascara).rounded_rectangle([0, 0, largura - 1, altura - 1], RAIO_CANTOS, fill=255)
    base.paste(Image.new("RGBA", (largura, altura), (*COR_FUNDO, 255)), (0, 0), mascara)
    return base, mascara


def compor_cartao(produto_rgba):
    """Produto isolado, centralizado, com sombra suave (como em uma loja virtual)."""
    largura, altura = config.LARGURA_CARTAO, config.ALTURA_CARTAO
    cartao, mascara = base_cartao()

    escala = min(largura * 0.74 / produto_rgba.width, altura * 0.78 / produto_rgba.height)
    produto = produto_rgba.resize(
        (max(1, int(produto_rgba.width * escala)), max(1, int(produto_rgba.height * escala))),
        Image.Resampling.LANCZOS,
    )
    x = (largura - produto.width) // 2
    y = (altura - produto.height) // 2 - 10

    sombra = Image.new("L", (largura, altura), 0)
    ImageDraw.Draw(sombra).ellipse(
        [x + produto.width * 0.1, y + produto.height - 12, x + produto.width * 0.9, y + produto.height + 14],
        fill=80,
    )
    sombra = sombra.filter(ImageFilter.GaussianBlur(10))
    cartao.paste(Image.new("RGBA", (largura, altura), (*COR_SOMBRA, 255)), (0, 0), sombra)
    cartao.alpha_composite(produto, (x, y))

    cartao.putalpha(mascara)
    return cartao


def compor_cartao_recorte(foto_rgb):
    """Alternativa quando a foto isolada não é aprovada: recorte da página
    clareado e inteiro, com o fundo do cartão na cor da borda da própria foto
    (em fundo branco, a emenda desaparece)."""
    largura, altura = config.LARGURA_CARTAO, config.ALTURA_CARTAO
    rgb = np.array(foto_rgb.convert("RGB"))
    borda = np.concatenate([rgb[:2].reshape(-1, 3), rgb[-2:].reshape(-1, 3),
                            rgb[:, :2].reshape(-1, 3), rgb[:, -2:].reshape(-1, 3)])
    cor = tuple(int(v) for v in np.median(borda, axis=0))

    cartao = Image.new("RGBA", (largura, altura), (*cor, 255))
    escala = min(largura * 0.92 / foto_rgb.width, altura * 0.92 / foto_rgb.height)
    foto = foto_rgb.convert("RGB").resize(
        (max(1, int(foto_rgb.width * escala)), max(1, int(foto_rgb.height * escala))),
        Image.Resampling.LANCZOS,
    )
    cartao.paste(foto, ((largura - foto.width) // 2, (altura - foto.height) // 2))
    _, mascara = base_cartao()
    cartao.putalpha(mascara)
    return cartao


# --------------------------------------
# Geração da foto de um produto
# --------------------------------------
def regiao_sem_texto(pagina, contorno, folga=0.06):
    """Área ao redor do produto, sem os blocos de texto vizinhos.

    Nunca corta para dentro do contorno do produto: a área sempre contém o
    contorno inteiro; só a folga ao redor é reduzida para evitar textos.
    """
    contorno = pymupdf.Rect(contorno)
    regiao = crescer(contorno, folga, pagina.rect)
    centro_x, centro_y = (contorno.x0 + contorno.x1) / 2, (contorno.y0 + contorno.y1) / 2
    for bloco in pagina.get_text("blocks"):
        texto = pymupdf.Rect(bloco[:4])
        if texto.is_empty or not texto.intersects(regiao):
            continue
        tx, ty = (texto.x0 + texto.x1) / 2, (texto.y0 + texto.y1) / 2
        dx = (tx - centro_x) / max(contorno.width, 1)
        dy = (ty - centro_y) / max(contorno.height, 1)
        if abs(dy) >= abs(dx):
            if dy > 0:
                regiao.y1 = max(min(regiao.y1, texto.y0), contorno.y1)
            else:
                regiao.y0 = min(max(regiao.y0, texto.y1), contorno.y0)
        elif dx > 0:
            regiao.x1 = max(min(regiao.x1, texto.x0), contorno.x1)
        else:
            regiao.x0 = min(max(regiao.x0, texto.x1), contorno.x0)
    return regiao


def _apagar_textos(pagina, regiao, caixa, foto):
    """Remove textos impressos sobre a foto (preço, nome) preenchendo com o fundo.

    Usa a posição exata de cada bloco de texto no PDF. Os rótulos das
    embalagens fazem parte da foto (não são texto do PDF), então só o texto
    impresso pelo catálogo é removido.
    """
    escala_x = foto.width / regiao.width
    escala_y = foto.height / regiao.height
    mascara = np.zeros((foto.height, foto.width), np.uint8)
    for bloco in pagina.get_text("blocks"):
        texto = pymupdf.Rect(bloco[:4])
        if texto.is_empty or not texto.intersects(regiao):
            continue
        r = pymupdf.Rect(texto.x0 - 1.5, texto.y0 - 1.5, texto.x1 + 1.5, texto.y1 + 1.5) & regiao
        mascara[int((r.y0 - regiao.y0) * escala_y):int((r.y1 - regiao.y0) * escala_y) + 1,
                int((r.x0 - regiao.x0) * escala_x):int((r.x1 - regiao.x0) * escala_x) + 1] = 255
    if not mascara.any():
        return foto
    restaurada = cv2.inpaint(np.array(foto.convert("RGB")), mascara, 7, cv2.INPAINT_TELEA)
    return Image.fromarray(restaurada)


def _renderizar(pagina, regiao):
    zoom = min(4.0, max(2.0, 900 / max(regiao.width, regiao.height)))
    return imagens.renderizar(pagina, zoom, regiao)


def gerar_foto(pagina, caixa, descricao_produto, destino):
    """Cria a foto do cartão. Retorna o tipo gerado: 'isolada' ou 'recorte'.

    1. Foto isolada do contorno completo do produto, usada somente se não
       encostar na borda da área analisada e se for aprovada na conferência.
    2. Caso contrário, recorte da página clareado a partir do contorno
       completo (o produto nunca é cortado), sem os textos vizinhos.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    kit = "kit" in descricao_produto.lower()

    isolado, contorno, encosta, inicial = None, None, False, None
    try:
        isolado, contorno, encosta, inicial = isolar_produto(pagina, caixa, kit)
    except Exception as erro:  # noqa: BLE001
        print(f"[IntelliDoc] Falha ao remover o fundo de {descricao_produto}: {erro!r}")

    if isolado is not None and not encosta:
        referencia = _renderizar(pagina, crescer(contorno, 0.15, pagina.rect))
        cartao = compor_cartao(clarear(isolado))
        previa = Image.new("RGB", cartao.size, (255, 255, 255))
        previa.paste(cartao, (0, 0), cartao)
        if imagens.verificar_foto_isolada(previa, referencia, descricao_produto):
            cartao.save(destino, format="WEBP", quality=90, method=6)
            return "isolada"

    # Maior contorno que ainda não pegou a cena inteira: garante o produto
    # completo (prioridade: nunca cortar). Sem segmentação, a caixa ampliada.
    if contorno is not None:
        base = contorno
    elif inicial is not None:
        base = inicial
    else:
        base = crescer(caixa, 0.3, pagina.rect)
    regiao = regiao_sem_texto(pagina, base)
    foto = _apagar_textos(pagina, regiao, caixa, _renderizar(pagina, regiao))
    compor_cartao_recorte(clarear(foto)).save(destino, format="WEBP", quality=90, method=6)
    return "recorte"


# --------------------------------------
# Atualização das fotos já armazenadas
# --------------------------------------
_sift = None
_trava_sift = threading.Lock()


def relocalizar(pagina, caminho_imagem):
    """Encontra na página a posição de uma foto antiga (JPEG do cartão).

    Usa correspondência de pontos (SIFT) entre a foto e a página
    renderizada. Não usa a API. Retorna a caixa do produto ou None.
    """
    global _sift
    cartao = np.array(Image.open(caminho_imagem).convert("RGB")).astype(int)
    fundo = cartao[2, 2]
    conteudo = np.abs(cartao - fundo).sum(2) > 30
    ys, xs = np.where(conteudo)
    if len(xs) == 0:
        return None
    interna = cartao[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)

    zoom = 3
    pix = pagina.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    imagem_pagina = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3)

    with _trava_sift:
        if _sift is None:
            _sift = cv2.SIFT_create()
        k1, d1 = _sift.detectAndCompute(cv2.cvtColor(interna, cv2.COLOR_RGB2GRAY), None)
        k2, d2 = _sift.detectAndCompute(cv2.cvtColor(imagem_pagina, cv2.COLOR_RGB2GRAY), None)
    if d1 is None or d2 is None or len(k1) < 8:
        return None

    pares = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    bons = [a for a, b in (p for p in pares if len(p) == 2) if a.distance < 0.75 * b.distance]
    if len(bons) < 12:
        return None
    matriz, inliers = cv2.estimateAffinePartial2D(
        np.float32([k1[m.queryIdx].pt for m in bons]),
        np.float32([k2[m.trainIdx].pt for m in bons]),
        method=cv2.RANSAC, ransacReprojThreshold=4,
    )
    if matriz is None or inliers is None or int(inliers.sum()) < 15:
        return None

    altura, largura = interna.shape[:2]
    cantos = np.float32([[0, 0], [largura, 0], [0, altura], [largura, altura]])
    mapeados = (matriz[:, :2] @ cantos.T).T + matriz[:, 2]
    regiao = pymupdf.Rect(
        mapeados[:, 0].min() / zoom, mapeados[:, 1].min() / zoom,
        mapeados[:, 0].max() / zoom, mapeados[:, 1].max() / zoom,
    )
    # A foto antiga tinha 10% de folga em cada lado: volta para a caixa do produto
    dx, dy = regiao.width * 0.10 / 1.2, regiao.height * 0.10 / 1.2
    caixa = pymupdf.Rect(regiao.x0 + dx, regiao.y0 + dy, regiao.x1 - dx, regiao.y1 - dy) & pagina.rect
    return None if caixa.is_empty else caixa


def _atualizar_produto(produto):
    caminho_atual = Path(produto["imagem"])
    if not caminho_atual.is_absolute():
        caminho_atual = config.BASE_DIR / caminho_atual

    with pymupdf.open(produto["documento"]) as documento:
        pagina = documento[produto["pagina"] - 1]
        caixa = texto_para_caixa(produto.get("caixa"))
        if caixa is None:
            if not caminho_atual.is_file():
                return produto, None, "sem arquivo"
            caixa = relocalizar(pagina, caminho_atual)
        if caixa is None:
            return produto, None, "posição não encontrada"

        descricao = " ".join(p for p in (produto["tipo"], produto["nome"]) if p)
        destino = caminho_atual.with_suffix(".webp")
        # Gera em um arquivo temporário e só então substitui a foto atual
        temporario = destino.with_name(destino.stem + ".novo.webp")
        tipo = gerar_foto(pagina, caixa, descricao, temporario)

    os.replace(temporario, destino)
    if caminho_atual != destino and caminho_atual.is_file():
        caminho_atual.unlink()
    return produto, (destino, tipo, caixa), "ok"


def atualizar_fotos(forcar=False, progresso=None):
    """Recria as fotos dos produtos já armazenados no novo estilo.

    Não extrai o PDF novamente: usa os dados do banco, a posição gravada
    da foto (ou a relocaliza a partir da foto antiga) e o PDF original.
    """
    produtos = [
        p for p in repositorio.listar_produtos()
        if p["imagem"] and (forcar or p.get("imagem_tipo") != "isolada")
        and Path(p["documento"]).is_file()
    ]
    resumo = {"total": len(produtos), "isoladas": 0, "recortes": 0, "falhas": 0}
    if not produtos:
        return resumo

    feitos = 0
    with ThreadPoolExecutor(max_workers=4) as executor:
        tarefas = [executor.submit(_atualizar_produto, p) for p in produtos]
        for tarefa in as_completed(tarefas):
            try:
                produto, novo, situacao = tarefa.result()
            except Exception as erro:  # noqa: BLE001
                print(f"[IntelliDoc] Erro ao atualizar foto: {erro!r}")
                resumo["falhas"] += 1
            else:
                if novo is None:
                    print(f"[IntelliDoc] {produto['nome']} (p. {produto['pagina']}): {situacao}")
                    resumo["falhas"] += 1
                else:
                    destino, tipo, caixa = novo
                    repositorio.atualizar_foto(
                        produto["id"], destino.relative_to(config.BASE_DIR).as_posix(),
                        tipo, caixa_para_texto(caixa),
                    )
                    resumo["isoladas" if tipo == "isolada" else "recortes"] += 1
            feitos += 1
            if progresso:
                progresso(feitos, len(produtos))
    return resumo


if __name__ == "__main__":
    def _mostrar(feitos, total):
        print(f"\r  fotos: {feitos}/{total}", end="", flush=True)

    print("Atualizando as fotos dos produtos...")
    resultado = atualizar_fotos(forcar="--forcar" in sys.argv[1:], progresso=_mostrar)
    print("\n  resultado:", resultado)
