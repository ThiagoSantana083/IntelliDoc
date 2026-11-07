# ======================================
#             IntelliDoc
#    Interface Web (Streamlit)
# ======================================
#
# Executar a partir da pasta do projeto:
#     streamlit run interface.py
#
# Este arquivo cuida apenas da interface. A lógica fica em:
#   criar_db.py  -> RAG de texto (chunks, ChromaDB, respostas)  [inalterado]
#   catalogo/    -> produtos, fotos, busca no catálogo e cartões

import hashlib
import html
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# --------------------------------------
# Caminhos do projeto
# --------------------------------------
# criar_db.py usa o caminho relativo "./data/db". Fixar o diretório de
# trabalho na pasta do projeto garante que o banco persistente correto
# seja usado, mesmo que o Streamlit seja iniciado de outra pasta.
BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

PASTA_DOCUMENTOS = BASE_DIR / "data" / "documents"
PASTA_DOCUMENTOS.mkdir(parents=True, exist_ok=True)

load_dotenv(BASE_DIR / ".env")

import openai  # noqa: E402  (importado após carregar o .env)

from catalogo import busca, repositorio  # noqa: E402
from catalogo.apresentacao import exibir_grade  # noqa: E402
from catalogo.processamento import processar_catalogo  # noqa: E402
from catalogo.texto import escapar_markdown  # noqa: E402
from criar_db import (  # noqa: E402
    validar_documento,
    extrair_paginas_pdf,
    dividir_texto_chunks,
    armazenar_chunks,
    criar_banco_vetorial,
    buscar_chunks,
    gerar_resposta,
)


# --------------------------------------
# Configuração da página
# --------------------------------------
st.set_page_config(
    page_title="IntelliDoc",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",  # aberta no computador, recolhida no celular
)

# Identidade visual do IntelliDoc (cabeçalho e etiquetas de fonte)
st.html(
    """
    <style>
    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        margin-bottom: .5rem;
    }
    .hero h1 { color: #FFFFFF; margin: 0; font-size: 2.2rem; padding: 0; }
    .hero p  { color: #E0E7FF; margin: .35rem 0 0; font-size: 1.05rem; }

    .fontes { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .2rem; }
    .fonte {
        display: inline-flex; align-items: center; gap: .35rem;
        padding: .3rem .7rem; border-radius: 999px; font-size: .85rem;
        background: rgba(79, 70, 229, .10);
        border: 1px solid rgba(79, 70, 229, .35);
        word-break: break-word;
    }

    @media (max-width: 640px) {
        .hero { padding: 1.1rem 1.2rem; }
        .hero h1 { font-size: 1.6rem; }
    }
    </style>
    """
)


# --------------------------------------
# Funções auxiliares da interface
# --------------------------------------
def chave_api_configurada():
    """Verifica se a chave existe, sem nunca exibir o seu valor."""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


@st.cache_resource(show_spinner=False)
def obter_banco():
    """Conexão reutilizada com o banco vetorial persistente."""
    return criar_banco_vetorial()


def hash_bytes(conteudo):
    return hashlib.sha256(conteudo).hexdigest()


def hash_arquivo(caminho):
    return hash_bytes(Path(caminho).read_bytes())


def listar_documentos_indexados():
    """Retorna {caminho_do_documento: quantidade_de_chunks} a partir do ChromaDB."""
    dados = obter_banco().get(include=["metadatas"])

    contagem = {}
    for metadado in dados.get("metadatas") or []:
        if not metadado or "source" not in metadado:
            continue
        contagem[metadado["source"]] = contagem.get(metadado["source"], 0) + 1

    return dict(sorted(contagem.items(), key=lambda item: Path(item[0]).name.lower()))


def documento_ja_indexado(caminho):
    resultado = obter_banco().get(
        where={"source": str(Path(caminho).resolve())},
        limit=1,
    )
    return bool(resultado["ids"])


def processar_upload(arquivo_enviado, forcar=False):
    """Salva o PDF em data/documents e o armazena no ChromaDB.

    Evita duplicidade de duas formas:
      1. Se um arquivo idêntico (mesmo nome e mesmo conteúdo) já estiver
         indexado, o processamento é ignorado (sem custo de embeddings).
      2. Caso contrário, armazenar_chunks() remove a versão anterior do
         mesmo arquivo antes de inserir os novos chunks.
    """
    nome = Path(arquivo_enviado.name).name  # remove qualquer caminho do nome
    destino = PASTA_DOCUMENTOS / nome
    conteudo = arquivo_enviado.getvalue()

    if (
        not forcar
        and destino.exists()
        and hash_arquivo(destino) == hash_bytes(conteudo)
        and documento_ja_indexado(destino)
    ):
        return {"status": "ignorado", "nome": nome, "caminho": destino}

    destino.write_bytes(conteudo)

    if not validar_documento(destino):
        return {"status": "erro", "nome": nome, "mensagem": "Arquivo PDF inválido."}

    try:
        paginas = extrair_paginas_pdf(destino)
    except Exception as erro:  # noqa: BLE001
        print(f"[IntelliDoc] Erro ao ler {nome}: {erro!r}")
        return {"status": "erro", "nome": nome, "mensagem": "Não foi possível ler o PDF (arquivo corrompido ou protegido)."}

    if not any(texto.strip() for texto in paginas):
        return {
            "status": "erro",
            "nome": nome,
            "mensagem": "Nenhum texto encontrado. O PDF pode ser digitalizado (imagem).",
        }

    chunks = dividir_texto_chunks(paginas, destino)
    armazenar_chunks(chunks)

    return {
        "status": "processado",
        "nome": nome,
        "caminho": destino,
        "paginas": len(paginas),
        "chunks": len(chunks),
    }


def extrair_produtos(caminho, forcar=False):
    """Extrai produtos e fotos exibindo uma barra de progresso."""
    nome = Path(caminho).name
    barra = st.progress(0.0, text=f"Identificando produtos e fotos em {nome}…")

    def atualizar(feitas, total):
        barra.progress(feitas / total, text=f"Analisando páginas de {nome}: {feitas}/{total}")

    try:
        resumo = processar_catalogo(caminho, forcar=forcar, progresso=atualizar)
    finally:
        barra.empty()
    return resumo


def extrair_fontes(resultados):
    """Lista única de (arquivo, página) na ordem de relevância."""
    fontes = []
    vistos = set()

    for documento, _pontuacao in resultados:
        nome = Path(documento.metadata.get("source", "Desconhecido")).name
        pagina = documento.metadata.get("page", "não informada")
        chave = (nome, pagina)

        if chave not in vistos:
            vistos.add(chave)
            fontes.append(
                {
                    "arquivo": nome,
                    "pagina": pagina,
                    "trecho": documento.page_content,
                }
            )

    return fontes


def mensagem_erro_openai(erro):
    """Mensagens amigáveis, sem expor detalhes da chave."""
    if isinstance(erro, openai.AuthenticationError):
        return "Falha de autenticação na OpenAI. Verifique a chave configurada no arquivo .env."
    if isinstance(erro, openai.RateLimitError):
        return "Limite de uso ou créditos da OpenAI atingido. Tente novamente mais tarde."
    if isinstance(erro, openai.APIConnectionError):
        return "Não foi possível conectar à OpenAI. Verifique sua conexão com a internet."
    if isinstance(erro, openai.NotFoundError):
        return "O modelo configurado não está disponível para esta chave da OpenAI."
    if isinstance(erro, openai.APIError):
        return "A OpenAI retornou um erro ao processar a solicitação. Tente novamente."
    return None


def exibir_fontes(fontes):
    if not fontes:
        return

    chips = "".join(
        f'<span class="fonte">📄 {html.escape(str(f["arquivo"]))} · p. {html.escape(str(f["pagina"]))}</span>'
        for f in fontes
    )
    st.markdown("**Fontes consultadas**")
    st.html(f'<div class="fontes">{chips}</div>')

    with st.expander("Ver trechos utilizados"):
        for fonte in fontes:
            st.markdown(f"**{escapar_markdown(fonte['arquivo'])} — página {fonte['pagina']}**")
            st.caption(escapar_markdown(fonte["trecho"][:600] + ("…" if len(fonte["trecho"]) > 600 else "")))


def exibir_mensagem(indice, mensagem):
    """Exibe uma mensagem do histórico (texto, fontes e cartões)."""
    with st.chat_message(mensagem["role"], avatar="🧑" if mensagem["role"] == "user" else "📄"):
        st.markdown(mensagem["content"] if mensagem.get("formatado") else escapar_markdown(mensagem["content"]))
        if mensagem.get("produto_ids"):
            produtos = repositorio.obter_produtos(mensagem["produto_ids"])
            if produtos:
                exibir_grade(produtos, chave=f"msg{indice}")
            else:
                st.warning("Os produtos desta resposta não estão mais armazenados.")
        if mensagem.get("fontes"):
            exibir_fontes(mensagem["fontes"])


# --------------------------------------
# Estado da sessão
# --------------------------------------
if "mensagens" not in st.session_state:
    st.session_state.mensagens = []

if "versao_uploader" not in st.session_state:
    st.session_state.versao_uploader = 0


# --------------------------------------
# Cabeçalho
# --------------------------------------
st.html(
    """
    <div class="hero">
        <h1>📄 IntelliDoc</h1>
        <p>Catálogo inteligente — envie PDFs, pergunte sobre os produtos e veja os resultados em cartões.</p>
    </div>
    """
)

if not chave_api_configurada():
    st.error(
        "Crie o arquivo `.env` na pasta do projeto com a variável `OPENAI_API_KEY` "
        "e reinicie a aplicação.",
        title="A chave da OpenAI não foi encontrada",
    )
    st.stop()


# --------------------------------------
# Barra lateral: upload e documentos
# --------------------------------------
with st.sidebar:
    st.header(":material/upload_file: Documentos")

    arquivos = st.file_uploader(
        "Envie arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.versao_uploader}",
        help="Os arquivos são salvos em data/documents e indexados no ChromaDB.",
    )

    extrair_catalogo = st.toggle(
        "Identificar produtos e fotos",
        value=True,
        help="Extrai nome, preço, família olfativa e a foto de cada produto (usa o modelo de visão da OpenAI).",
    )

    forcar = st.checkbox(
        "Reprocessar mesmo se já indexado",
        value=False,
        help="Substitui a versão armazenada do documento (não gera duplicatas).",
    )

    processar = st.button(
        "Processar e armazenar",
        type="primary",
        icon=":material/play_arrow:",
        width="stretch",
        disabled=not arquivos,
    )

    if processar and arquivos:
        resumo = []

        for indice, arquivo in enumerate(arquivos, start=1):
            try:
                with st.spinner(f"Gerando embeddings de {arquivo.name} ({indice}/{len(arquivos)})…"):
                    item = processar_upload(arquivo, forcar=forcar)
            except Exception as erro:  # noqa: BLE001
                print(f"[IntelliDoc] Erro ao processar {arquivo.name}: {erro!r}")
                item = {
                    "status": "erro",
                    "nome": arquivo.name,
                    "mensagem": mensagem_erro_openai(erro) or "Erro inesperado ao processar o arquivo.",
                }

            if extrair_catalogo and item["status"] in ("processado", "ignorado"):
                try:
                    item["catalogo"] = extrair_produtos(item["caminho"], forcar=forcar)
                except Exception as erro:  # noqa: BLE001
                    print(f"[IntelliDoc] Erro ao extrair produtos de {arquivo.name}: {erro!r}")
                    item["catalogo"] = {
                        "status": "erro",
                        "mensagem": mensagem_erro_openai(erro) or "Erro inesperado ao extrair os produtos.",
                    }
            resumo.append(item)

        for item in resumo:
            if item["status"] == "processado":
                st.success(f"**{item['nome']}**: {item['paginas']} páginas, {item['chunks']} chunks.")
            elif item["status"] == "ignorado":
                st.info(f"**{item['nome']}**: texto já indexado e não alterado.")
            else:
                st.error(f"**{item['nome']}**: {item['mensagem']}")

            catalogo = item.get("catalogo")
            if not catalogo:
                continue
            if catalogo["status"] in ("processado", "parcial"):
                texto = (
                    f"{catalogo['produtos']} produtos identificados, "
                    f"{catalogo['com_imagem']} com foto."
                )
                if catalogo["status"] == "parcial":
                    st.warning(texto + f" {len(catalogo['erros'])} página(s) falharam; reprocesse depois.")
                else:
                    st.success(texto)
            elif catalogo["status"] == "ignorado":
                st.info("Produtos já extraídos desta versão do arquivo.")
            else:
                st.error(catalogo.get("mensagem", "Falha ao extrair os produtos."))

        # Limpa o seletor de arquivos para o próximo envio
        st.session_state.versao_uploader += 1

    st.divider()
    st.subheader(":material/library_books: Base de conhecimento")

    try:
        documentos = listar_documentos_indexados()
    except Exception as erro:  # noqa: BLE001
        print(f"[IntelliDoc] Erro ao ler o banco vetorial: {erro!r}")
        documentos = None
        st.error("Não foi possível ler o banco vetorial.")

    try:
        resumo_catalogo = repositorio.resumo_documentos()
    except Exception as erro:  # noqa: BLE001
        print(f"[IntelliDoc] Erro ao ler o banco de produtos: {erro!r}")
        resumo_catalogo = {}
        st.error("Não foi possível ler o banco de produtos.")

    if documentos:
        total_produtos = sum(r["produtos"] for d, r in resumo_catalogo.items() if d in documentos)

        with st.grid(3, min_column_width=60, gap="small"):
            st.metric("Documentos", len(documentos))
            st.metric("Produtos", total_produtos)
            st.metric("Chunks", sum(documentos.values()))

        pendentes = []
        for caminho, total_chunks in documentos.items():
            info = resumo_catalogo.get(caminho)
            with st.container(border=True):
                st.markdown(f":material/picture_as_pdf: **{escapar_markdown(Path(caminho).name)}**")
                if info and info["status"] == "completo":
                    detalhes = f"{total_chunks} chunks · {info['produtos']} produtos · {info['com_imagem'] or 0} com foto"
                elif info:
                    detalhes = f"{total_chunks} chunks · {info['produtos']} produtos (extração parcial)"
                    pendentes.append(caminho)
                else:
                    detalhes = f"{total_chunks} chunks · produtos ainda não extraídos"
                    pendentes.append(caminho)
                st.caption(detalhes)

        if pendentes and st.button(
            f"Extrair produtos ({len(pendentes)} pendente{'s' if len(pendentes) > 1 else ''})",
            icon=":material/auto_awesome:",
            width="stretch",
        ):
            for caminho in pendentes:
                try:
                    resultado = extrair_produtos(caminho)
                    if resultado["status"] in ("processado", "parcial"):
                        st.success(f"{resultado['nome']}: {resultado['produtos']} produtos, {resultado['com_imagem']} com foto.")
                    elif resultado["status"] == "erro":
                        st.error(f"{resultado['nome']}: {resultado['mensagem']}")
                except Exception as erro:  # noqa: BLE001
                    print(f"[IntelliDoc] Erro ao extrair produtos de {caminho}: {erro!r}")
                    st.error(mensagem_erro_openai(erro) or "Erro inesperado ao extrair os produtos.")
            st.rerun()
    elif documentos is not None:
        st.caption("Nenhum documento indexado ainda.")

    st.divider()
    if st.button("Limpar conversa", icon=":material/delete_sweep:", width="stretch"):
        st.session_state.mensagens = []
        st.rerun()


# --------------------------------------
# Área principal: perguntas e respostas
# --------------------------------------
if not st.session_state.mensagens:
    st.info(
        "Envie seus PDFs pela barra lateral (ou use os já indexados) e faça uma pergunta abaixo. "
        "Exemplos: *Qual é o preço do Desodorante Corporal Frangipani Flower 100ml?* · "
        "*Quais são os produtos masculinos?* · *Quais produtos custam menos de R\\$ 100,00?*",
        icon=":material/lightbulb:",
    )

for indice, mensagem in enumerate(st.session_state.mensagens):
    exibir_mensagem(indice, mensagem)

pergunta = st.chat_input("Pergunte sobre os produtos ou o conteúdo dos seus documentos…")

if pergunta is not None:
    pergunta = pergunta.strip()

    if not pergunta:
        st.warning("Digite uma pergunta válida.")
    else:
        st.session_state.mensagens.append({"role": "user", "content": pergunta})
        exibir_mensagem(len(st.session_state.mensagens) - 1, st.session_state.mensagens[-1])

        resposta = None
        try:
            # 1) Perguntas sobre produtos: consulta ao catálogo estruturado inteiro
            with st.spinner("Consultando o catálogo de produtos…"):
                indexados = list(documentos) if documentos else []
                resultado = busca.buscar_produtos(pergunta, indexados)

            if resultado and resultado.texto:
                # Listagem, contagem, imagens, duplicados ou preços (dados já armazenados)
                resposta = {
                    "role": "assistant",
                    "content": resultado.texto,
                    "formatado": True,
                    "produto_ids": [p["id"] for p in resultado.exibir],
                }
            else:
                # 2) Perguntas gerais (ou nenhum produto encontrado): RAG de texto
                with st.spinner("Buscando nos documentos…"):
                    resultados = buscar_chunks(pergunta)

                if not resultados:
                    texto = (
                        "Não encontrei informações relevantes nos documentos indexados. "
                        "Verifique se algum PDF já foi processado."
                    )
                    fontes = []
                else:
                    with st.spinner("Gerando resposta…"):
                        texto = gerar_resposta(pergunta, resultados)
                    fontes = extrair_fontes(resultados)

                if resultado is not None:
                    texto = (
                        "Nenhum produto do catálogo estruturado atendeu a esse critério. "
                        "Veja abaixo o que encontrei nos trechos do documento:\n\n" + texto
                    )
                resposta = {"role": "assistant", "content": texto, "fontes": fontes}

        except Exception as erro:  # noqa: BLE001
            print(f"[IntelliDoc] Erro ao responder: {erro!r}")
            with st.chat_message("assistant", avatar="📄"):
                st.error(mensagem_erro_openai(erro) or "Ocorreu um erro inesperado ao gerar a resposta.")

        if resposta:
            st.session_state.mensagens.append(resposta)
            exibir_mensagem(len(st.session_state.mensagens) - 1, resposta)
