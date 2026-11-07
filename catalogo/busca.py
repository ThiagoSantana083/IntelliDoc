"""Recuperação de produtos para perguntas sobre o catálogo.

Complementa (não substitui) a busca semântica do ChromaDB:

  * Perguntas gerais sobre o conteúdo -> RAG atual (buscar_chunks + gerar_resposta).
  * Perguntas sobre produtos -> consulta ao catálogo estruturado inteiro:
      1. O modelo apenas interpreta a pergunta e devolve os filtros
         (nome/linha, público, família olfativa, categoria, preço, ordenação).
      2. Nome/linha, público, família olfativa e preço são filtrados de forma
         exata, em código, sobre TODOS os produtos armazenados.
      3. Somente a categoria (ex.: "perfumes", "kits") é avaliada pelo
         modelo, em lotes, usando apenas os dados extraídos do catálogo.

Assim, "quais são os produtos masculinos?" não depende dos 10 chunks mais
parecidos: todos os produtos armazenados são verificados.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from catalogo import analise, config, repositorio
from catalogo.texto import escapar_markdown, formatar_preco, normalizar, tokens

TAMANHO_LOTE = 40

# Palavras que não ajudam a identificar um produto pelo nome.
_PALAVRAS_VAZIAS = {
    "a", "o", "as", "os", "da", "de", "do", "das", "dos", "e", "em", "na", "no",
    "linha", "colecao", "produto", "produtos", "catalogo",
}

# Indicações explícitas de público no nome do produto.
_PUBLICO_NO_NOME = {
    "masculino": r"\b(for men|men|man|homme|masculino)\b",
    "feminino": r"\b(for women|women|woman|femme|feminino)\b",
    "infantil": r"\b(kids|infantil)\b",
    "unissex": r"\b(unissex|unisex)\b",
}


class Interpretacao(BaseModel):
    tipo: Literal["produtos", "geral"] = Field(
        description="'produtos' para encontrar, listar, contar, analisar ou consultar preço/dados de produtos; "
        "'geral' para dúvidas sobre o conteúdo textual do documento."
    )
    analise: Literal["listar", "contar", "imagens", "duplicados", "precos"] = Field(
        description="O que o usuário quer saber sobre os produtos filtrados: 'listar' (mostrar/quais são), "
        "'contar' (quantos existem), 'imagens' (quantos têm ou não têm imagem/foto), "
        "'duplicados' (repetidos/duplicados), 'precos' (preço médio, faixa de preços, mais caro e mais barato juntos)."
    )
    imagem: Optional[Literal["com", "sem"]] = Field(
        description="Filtro por foto quando a pergunta pede para LISTAR os produtos com foto ('com') ou "
        "sem foto ('sem'). null quando a pergunta só quer contar imagens."
    )
    termo: Optional[str] = Field(
        description="Nome, linha ou coleção específica citada (ex.: 'Frangipani Flower 100ml', "
        "'Mahogany For Men'), sem palavras como 'linha' ou 'produtos'. null se não houver."
    )
    tipo_produto: Optional[str] = Field(
        description="Tipo específico citado junto ao nome (ex.: 'Fragrância Desodorante Corporal', "
        "'Hidratante'), ou null."
    )
    publico: Optional[Literal["masculino", "feminino", "unissex", "infantil"]]
    familia_olfativa: Optional[str] = Field(description="Família olfativa pedida (ex.: 'Oriental Baunilha'), ou null.")
    categoria: Optional[str] = Field(
        description="Categoria ampla pedida (ex.: 'perfumes', 'kits', 'sabonetes', 'cuidados com o corpo'), "
        "ou null. Não repita aqui o termo, o público ou a família."
    )
    preco_min: Optional[float]
    preco_min_inclusivo: bool
    preco_max: Optional[float]
    preco_max_inclusivo: bool
    ordenacao: Literal["catalogo", "menor_preco", "maior_preco"]
    limite: Optional[int] = Field(description="Quantidade máxima pedida (ex.: 'os 3 mais baratos'), ou null.")


class Selecao(BaseModel):
    ids: list[int]


@lru_cache(maxsize=1)
def _modelo_interpretacao():
    return ChatOpenAI(model=config.MODELO_TEXTO, temperature=0, max_retries=3).with_structured_output(
        Interpretacao, method="json_schema", strict=True
    )


@lru_cache(maxsize=1)
def _modelo_selecao():
    return ChatOpenAI(model=config.MODELO_TEXTO, temperature=0, max_retries=3).with_structured_output(
        Selecao, method="json_schema", strict=True
    )


@dataclass
class ResultadoBusca:
    interpretacao: Interpretacao
    produtos: list = field(default_factory=list)   # todos os produtos que atendem aos filtros
    exibir: list = field(default_factory=list)     # produtos mostrados em cartões
    total_verificado: int = 0
    documentos_pendentes: list = field(default_factory=list)
    texto: str = ""


# --------------------------------------
# Interpretação
# --------------------------------------
def interpretar(pergunta):
    instrucao = (
        "Você interpreta perguntas feitas a um assistente de catálogos de produtos em PDF.\n"
        "Classifique a pergunta e extraia os filtros. Preencha apenas o que a pergunta pede.\n"
        "Preços em reais como número (ex.: 100.0). 'inferior a 100' -> preco_max=100 e "
        "preco_max_inclusivo=false; 'até 100' -> preco_max_inclusivo=true.\n"
        "Exemplos de 'produtos': preço de um produto, listar produtos masculinos, perfumes de uma "
        "família olfativa, produtos de uma linha, produtos abaixo de um preço, o mais barato.\n"
        "Exemplos de 'geral': perguntas sobre textos, condições, instruções ou temas do documento.\n"
        "Exemplos:\n"
        "- 'Qual o preço do Desodorante Corporal Frangipani Flower 100ml?' -> termo='Frangipani Flower 100ml', "
        "tipo_produto='Desodorante Corporal'\n"
        "- 'Mostre os produtos da linha Mahogany For Men' -> termo='Mahogany For Men'\n"
        "- 'Quais perfumes têm fragrância Oriental Baunilha?' -> familia_olfativa='Oriental Baunilha', "
        "categoria='perfumes'\n"
        "- 'Quais são os produtos masculinos?' -> publico='masculino', analise='listar'\n"
        "- 'Quantos produtos masculinos existem?' -> publico='masculino', analise='contar'\n"
        "- 'Quantos produtos masculinos possuem imagem associada?' -> publico='masculino', analise='imagens'\n"
        "- 'Quais produtos não têm foto?' -> analise='listar', imagem='sem'\n"
        "- 'Existem produtos masculinos duplicados na listagem?' -> publico='masculino', analise='duplicados'\n"
        "- 'Qual a faixa de preço dos perfumes femininos?' -> publico='feminino', categoria='perfumes', analise='precos'\n"
        "- 'Qual o perfume mais barato?' -> categoria='perfumes', analise='listar', ordenacao='menor_preco', limite=1\n"
        "Use ordenacao='catalogo' quando não houver pedido de ordenação.\n\n"
        f"Pergunta: {pergunta}"
    )
    return _modelo_interpretacao().invoke(instrucao)


# --------------------------------------
# Filtros exatos (sem IA)
# --------------------------------------
def _palavras_chave(texto):
    return [t for t in tokens(texto) if t not in _PALAVRAS_VAZIAS]


def _filtrar_termo(produtos, termo, tipo_produto):
    """Todas as palavras do termo devem estar no nome, linha ou tipo do produto."""
    palavras = _palavras_chave(termo)
    if not palavras:
        return produtos
    encontrados = []
    for p in produtos:
        conteudo = set(tokens(" ".join(x for x in (p["nome"], p["linha"], p["tipo"]) if x)))
        if all(palavra in conteudo for palavra in palavras):
            encontrados.append(p)

    # O tipo (ex.: 'Desodorante Corporal') desempata quando há variações.
    palavras_tipo = _palavras_chave(tipo_produto)
    if palavras_tipo and len(encontrados) > 1:
        com_tipo = [p for p in encontrados if all(t in set(tokens(p["tipo"] or "")) for t in palavras_tipo)]
        if com_tipo:
            encontrados = com_tipo
    return encontrados


def _filtrar_publico(produtos, publico):
    """Público indicado na página ou, explicitamente, no nome do produto."""
    padrao = _PUBLICO_NO_NOME.get(publico)
    resultado = []
    for p in produtos:
        if p["publico"] == publico:
            resultado.append(p)
        elif p["publico"] is None and padrao and re.search(padrao, normalizar(p["nome"])):
            resultado.append(p)
    return resultado


def _filtrar_familia(produtos, familia):
    palavras = _palavras_chave(familia)
    if not palavras:
        return produtos
    return [
        p for p in produtos
        if p["familia_olfativa"] and all(w in set(tokens(p["familia_olfativa"])) for w in palavras)
    ]


def _filtrar_preco(produtos, interp):
    if interp.preco_min is None and interp.preco_max is None:
        return produtos
    filtrados = []
    for p in produtos:
        valor = p["preco_valor"]
        if valor is None:
            continue
        if interp.preco_min is not None:
            if valor < interp.preco_min or (not interp.preco_min_inclusivo and valor == interp.preco_min):
                continue
        if interp.preco_max is not None:
            if valor > interp.preco_max or (not interp.preco_max_inclusivo and valor == interp.preco_max):
                continue
        filtrados.append(p)
    return filtrados


# --------------------------------------
# Seleção por categoria (IA, sobre todos os candidatos)
# --------------------------------------
def _linha_produto(produto):
    partes = [
        f"id={produto['id']}",
        f"nome={produto['nome']}",
        f"tipo={produto['tipo'] or '-'}",
        f"linha={produto['linha'] or '-'}",
        f"familia={produto['familia_olfativa'] or '-'}",
    ]
    return " | ".join(partes)


def _selecionar_lote(criterio, lote):
    instrucao = (
        "Selecione, na lista abaixo, os produtos que pertencem ao critério pedido.\n"
        "Use SOMENTE o nome, o tipo, a linha e a família de cada produto; não use conhecimento externo.\n"
        "'Perfume' ou 'fragrância' corresponde a tipos como Fragrância Desodorante Corporal, "
        "Eau de Parfum, Eau de Toilette, Colônia, Deo Parfum, Body Splash ou Body Spray perfumado.\n"
        "Na dúvida, não selecione. Retorne apenas os ids.\n\n"
        f"Critério: {criterio}\n\nProdutos:\n"
        + "\n".join(_linha_produto(p) for p in lote)
    )
    resposta = _modelo_selecao().invoke(instrucao)
    validos = {p["id"] for p in lote}
    return [i for i in resposta.ids if i in validos]


# Categorias comuns reconhecidas em código pelo tipo/nome do produto
# (palavra da pergunta -> termos que precisam aparecer no tipo ou no nome).
_CATEGORIAS = {
    "perfume": ("fragrancia", "eau de", "colonia", "parfum", "body splash"),
    "fragrancia": ("fragrancia", "eau de", "colonia", "parfum", "body splash"),
    "colonia": ("colonia",),
    "kit": ("kit",),
    "presente": ("kit", "presente"),
    "sabonete": ("sabonete",),
    "hidratante": ("hidratante",),
    "desodorante": ("desodorante",),
    "shampoo": ("shampoo",),
    "condicionador": ("condicionador",),
    "oleo": ("oleo",),
    "body splash": ("body splash",),
    "refil": ("refil",),
    "ambiente": ("ambiente", "difusor", "aromatizador"),
    "difusor": ("difusor",),
}


def _categoria_conhecida(categoria):
    """Termos de busca para uma categoria comum, ou None se não for reconhecida."""
    texto = normalizar(categoria)
    palavras = {p.rstrip("s") for p in texto.split()} | {texto.rstrip("s")}
    termos = []
    for chave, valores in _CATEGORIAS.items():
        if chave in palavras or chave in texto:
            termos.extend(valores)
    return tuple(dict.fromkeys(termos)) or None


def _filtrar_categoria(produtos, categoria):
    termos = _categoria_conhecida(categoria)
    if termos:
        return [
            p for p in produtos
            if any(t in normalizar(f"{p['tipo'] or ''} {p['nome']}") for t in termos)
        ]

    # Categoria não reconhecida: o modelo avalia todos os candidatos em lotes.
    lotes = [produtos[i:i + TAMANHO_LOTE] for i in range(0, len(produtos), TAMANHO_LOTE)]
    selecionados = set()
    with ThreadPoolExecutor(max_workers=min(6, len(lotes))) as executor:
        for ids in executor.map(lambda lote: _selecionar_lote(categoria, lote), lotes):
            selecionados.update(ids)
    return [p for p in produtos if p["id"] in selecionados]


# --------------------------------------
# Resultado
# --------------------------------------
def _sem_repeticao(produtos):
    """Um mesmo produto (código ou nome + preço) aparece uma única vez."""
    vistos, unicos = set(), []
    for p in produtos:
        chave = p["codigo"] or (normalizar(p["nome"]), p["preco_valor"])
        if chave not in vistos:
            vistos.add(chave)
            unicos.append(p)
    return unicos


def _descrever_filtros(interp):
    partes = []
    if interp.termo:
        partes.append(f"“{interp.termo}”")
    if interp.categoria:
        partes.append(interp.categoria)
    if interp.publico:
        partes.append(f"público {interp.publico}")
    if interp.familia_olfativa:
        partes.append(f"família olfativa {interp.familia_olfativa}")
    if interp.preco_min is not None:
        partes.append(f"preço {'a partir de' if interp.preco_min_inclusivo else 'acima de'} {formatar_preco(interp.preco_min)}")
    if interp.preco_max is not None:
        partes.append(f"preço {'até' if interp.preco_max_inclusivo else 'abaixo de'} {formatar_preco(interp.preco_max)}")
    if interp.imagem == "com" and interp.analise != "imagens":
        partes.append("com imagem")
    elif interp.imagem == "sem" and interp.analise != "imagens":
        partes.append("sem imagem")
    return ", ".join(partes) or "todos os produtos"


def _texto_listagem(produtos, interp):
    # Frase de "preço do produto" só quando a pergunta cita um produto específico.
    if len(produtos) == 1 and interp.termo:
        p = produtos[0]
        nome = " ".join(x for x in (p["tipo"], p["nome"]) if x)
        preco = f"custa **{p['preco_texto']}**" if p["preco_texto"] else "não tem preço indicado no catálogo"
        return f"**{nome}** {preco}. Ele aparece na página {p['pagina']} de *{p['nome_arquivo']}*."
    n = len(produtos)
    return f"Encontrei **{n} {'produto' if n == 1 else 'produtos'}** para: {_descrever_filtros(interp)}."


def _observacoes(resultado, catalogos):
    """Notas de transparência comuns a todas as respostas sobre produtos."""
    interp = resultado.interpretacao
    linhas = [
        f"Verifiquei todos os {resultado.total_verificado} produtos identificados em "
        f"{catalogos} catálogo(s) processado(s), usando apenas os dados já armazenados."
    ]
    if interp.preco_min is not None or interp.preco_max is not None:
        linhas.append("Produtos sem preço identificado não entram no filtro de preço.")
    if interp.publico:
        linhas.append(
            "O público considera a indicação da página (etiqueta ou texto) ou o nome do produto "
            "(ex.: “For Men”). Produtos sem essa indicação não foram incluídos."
        )
    if interp.familia_olfativa:
        linhas.append("Só entram produtos cuja página informa a família olfativa.")
    if interp.analise == "listar" and interp.limite:
        linhas.append(f"Exibindo no máximo {interp.limite} produto(s), conforme pedido.")
    if resultado.documentos_pendentes:
        nomes = ", ".join(resultado.documentos_pendentes)
        linhas.append(f"Atenção: os produtos de **{nomes}** ainda não foram extraídos e não entraram nesta busca.")
    return linhas


def _filtrar(catalogo, interp):
    """Aplica todos os filtros da pergunta (sem remover repetições)."""
    candidatos = catalogo
    if interp.termo:
        por_termo = _filtrar_termo(candidatos, interp.termo, interp.tipo_produto)
        # Sem correspondência exata (ex.: grafia diferente): o modelo avalia o termo.
        candidatos = por_termo if por_termo else _filtrar_categoria(candidatos, interp.termo)
    if interp.publico:
        candidatos = _filtrar_publico(candidatos, interp.publico)
    if interp.familia_olfativa:
        candidatos = _filtrar_familia(candidatos, interp.familia_olfativa)
    candidatos = _filtrar_preco(candidatos, interp)
    if interp.categoria and candidatos:
        candidatos = _filtrar_categoria(candidatos, interp.categoria)
    if interp.imagem == "com":
        candidatos = [p for p in candidatos if analise.caminho_imagem(p)]
    elif interp.imagem == "sem":
        candidatos = [p for p in candidatos if not analise.caminho_imagem(p)]
    return candidatos


def _ordenar(produtos, interp):
    if interp.ordenacao == "menor_preco":
        produtos.sort(key=lambda p: (p["preco_valor"] is None, p["preco_valor"] or 0))
    elif interp.ordenacao == "maior_preco":
        produtos.sort(key=lambda p: (p["preco_valor"] is None, -(p["preco_valor"] or 0)))
    return produtos


# --------------------------------------
# Função principal
# --------------------------------------
def buscar_produtos(pergunta, documentos_indexados=()):
    """Retorna ResultadoBusca, ou None quando a pergunta não é sobre produtos.

    A resposta depende do que foi perguntado (interp.analise):
      listar      cartões dos produtos encontrados
      contar      quantidade + nomes
      imagens     quantos têm e quantos não têm imagem (com os nomes)
      duplicados  duplicados e possíveis duplicados (antes de remover repetições)
      precos      menor, maior e média de preço

    Só a interpretação da pergunta usa a API; as análises usam os dados
    estruturados já armazenados.

    documentos_indexados: caminhos ("source") presentes no ChromaDB, usados
    para avisar quais documentos ainda não tiveram os produtos extraídos.
    """
    catalogo = repositorio.listar_produtos()
    if not catalogo:
        return None

    interp = interpretar(pergunta)
    # Filtros e análises próprios de produtos confirmam a intenção, mesmo que
    # a classificação do modelo oscile entre 'produtos' e 'geral'.
    tem_filtro_de_produto = any(
        valor is not None
        for valor in (interp.publico, interp.familia_olfativa, interp.categoria,
                      interp.preco_min, interp.preco_max, interp.imagem)
    ) or interp.analise in ("imagens", "duplicados")
    if interp.tipo != "produtos" and not tem_filtro_de_produto:
        return None

    resumo = repositorio.resumo_documentos()
    pendentes = [
        Path(doc).name for doc in documentos_indexados
        if doc not in resumo or resumo[doc]["status"] != "completo"
    ]

    # Ao analisar imagens, os dois grupos (com e sem foto) precisam ser contados.
    if interp.analise == "imagens":
        interp.imagem = None

    filtrados = _filtrar(catalogo, interp)
    unicos = _ordenar(_sem_repeticao(filtrados), interp)

    resultado = ResultadoBusca(
        interpretacao=interp,
        produtos=unicos,
        total_verificado=len(catalogo),
        documentos_pendentes=pendentes,
    )
    if not filtrados:
        return resultado  # sem texto: a interface usa o RAG de texto

    descricao = _descrever_filtros(interp)
    if interp.analise == "duplicados":
        # Usa a lista ANTES de remover repetições, senão os duplicados sumiriam.
        principal, resultado.exibir = analise.texto_duplicados(filtrados, descricao)
    elif interp.analise == "imagens":
        principal, resultado.exibir = analise.texto_imagens(unicos, descricao)
    elif interp.analise == "contar":
        principal, resultado.exibir = analise.texto_contagem(unicos, descricao), unicos
    elif interp.analise == "precos" and len(unicos) > 1:
        principal, resultado.exibir = analise.texto_precos(unicos, descricao)
    else:
        if interp.limite and interp.limite > 0:
            unicos = unicos[: interp.limite]
        resultado.produtos = unicos
        principal, resultado.exibir = _texto_listagem(unicos, interp), unicos

    texto = principal + "\n\n" + "\n\n".join(_observacoes(resultado, len(resumo)))
    resultado.texto = escapar_markdown(texto)  # "R$" não pode virar fórmula LaTeX
    return resultado
