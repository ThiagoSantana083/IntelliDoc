# ======================================
#             IntelliDoc
#    Interface Web (Streamlit)
# ======================================
#
# Executar a partir da pasta do projeto:
#     streamlit run interface.py
#
# A interface apenas reutiliza as funções de criar_db.py.
# O mecanismo RAG (chunks, embeddings, busca e geração) não é alterado.

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
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; max-width: 1100px; }

    .hero {
        padding: 1.6rem 1.8rem;
        border-radius: 16px;
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        color: #FFFFFF;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: #FFFFFF; margin: 0; font-size: 2.2rem; }
    .hero p  { color: #E0E7FF; margin: .35rem 0 0; font-size: 1.05rem; }

    .fontes { display: flex; flex-wrap: wrap; gap: .5rem; margin-top: .6rem; }
    .fonte {
        display: inline-flex; align-items: center; gap: .35rem;
        padding: .3rem .7rem; border-radius: 999px; font-size: .85rem;
        background: rgba(79, 70, 229, .10);
        border: 1px solid rgba(79, 70, 229, .35);
        word-break: break-word;
    }

    @media (max-width: 640px) {
        .block-container { padding-left: 1rem; padding-right: 1rem; }
        .hero { padding: 1.1rem 1.2rem; }
        .hero h1 { font-size: 1.6rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
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
    """Retorna {nome_do_arquivo: quantidade_de_chunks} a partir do ChromaDB."""
    dados = obter_banco().get(include=["metadatas"])

    contagem = {}
    for metadado in dados.get("metadatas") or []:
        if not metadado or "source" not in metadado:
            continue
        nome = Path(metadado["source"]).name
        contagem[nome] = contagem.get(nome, 0) + 1

    return dict(sorted(contagem.items()))


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
        return {"status": "ignorado", "nome": nome}

    destino.write_bytes(conteudo)

    if not validar_documento(destino):
        return {"status": "erro", "nome": nome, "mensagem": "Arquivo PDF inválido."}

    paginas = extrair_paginas_pdf(destino)

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
        "paginas": len(paginas),
        "chunks": len(chunks),
    }


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
    st.markdown(f'<div class="fontes">{chips}</div>', unsafe_allow_html=True)

    with st.expander("Ver trechos utilizados"):
        for fonte in fontes:
            st.markdown(f"**{fonte['arquivo']} — página {fonte['pagina']}**")
            st.caption(fonte["trecho"][:600] + ("…" if len(fonte["trecho"]) > 600 else ""))


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
st.markdown(
    """
    <div class="hero">
        <h1>📄 IntelliDoc</h1>
        <p>Assistente de Documentação Inteligente — envie PDFs e faça perguntas sobre eles.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not chave_api_configurada():
    st.error(
        "A chave da OpenAI não foi encontrada. Crie o arquivo `.env` na pasta do "
        "projeto com a variável `OPENAI_API_KEY` e reinicie a aplicação."
    )
    st.stop()


# --------------------------------------
# Barra lateral: upload e documentos
# --------------------------------------
with st.sidebar:
    st.header("📥 Documentos")

    arquivos = st.file_uploader(
        "Envie arquivos PDF",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.versao_uploader}",
        help="Os arquivos são salvos em data/documents e indexados no ChromaDB.",
    )

    forcar = st.checkbox(
        "Reprocessar mesmo se já indexado",
        value=False,
        help="Substitui a versão armazenada do documento (não gera duplicatas).",
    )

    processar = st.button(
        "Processar e armazenar",
        type="primary",
        use_container_width=True,
        disabled=not arquivos,
    )

    if processar and arquivos:
        barra = st.progress(0.0, text="Iniciando…")
        resumo = []

        for indice, arquivo in enumerate(arquivos, start=1):
            barra.progress(
                (indice - 1) / len(arquivos),
                text=f"Processando {arquivo.name} ({indice}/{len(arquivos)})…",
            )
            try:
                with st.spinner(f"Gerando embeddings de {arquivo.name}…"):
                    resumo.append(processar_upload(arquivo, forcar=forcar))
            except Exception as erro:  # noqa: BLE001
                print(f"[IntelliDoc] Erro ao processar {arquivo.name}: {erro!r}")
                resumo.append(
                    {
                        "status": "erro",
                        "nome": arquivo.name,
                        "mensagem": mensagem_erro_openai(erro)
                        or "Erro inesperado ao processar o arquivo.",
                    }
                )

        barra.progress(1.0, text="Concluído.")

        for item in resumo:
            if item["status"] == "processado":
                st.success(
                    f"**{item['nome']}** armazenado: {item['paginas']} páginas, "
                    f"{item['chunks']} chunks."
                )
            elif item["status"] == "ignorado":
                st.info(f"**{item['nome']}** já está indexado e não foi alterado.")
            else:
                st.error(f"**{item['nome']}**: {item['mensagem']}")

        # Limpa o seletor de arquivos para o próximo envio
        st.session_state.versao_uploader += 1

    st.divider()
    st.subheader("📚 Base de conhecimento")

    try:
        documentos = listar_documentos_indexados()
    except Exception as erro:  # noqa: BLE001
        print(f"[IntelliDoc] Erro ao ler o banco vetorial: {erro!r}")
        documentos = None
        st.error("Não foi possível ler o banco vetorial.")

    if documentos:
        col1, col2 = st.columns(2)
        col1.metric("Documentos", len(documentos))
        col2.metric("Chunks", sum(documentos.values()))

        for nome, total in documentos.items():
            st.markdown(f"- 📄 {nome}  \n  <small>{total} chunks</small>", unsafe_allow_html=True)
    elif documentos is not None:
        st.caption("Nenhum documento indexado ainda.")

    st.divider()
    if st.button("🧹 Limpar conversa", use_container_width=True):
        st.session_state.mensagens = []
        st.rerun()


# --------------------------------------
# Área principal: perguntas e respostas
# --------------------------------------
if not st.session_state.mensagens:
    st.info(
        "Envie seus PDFs pela barra lateral (ou use os já indexados) e faça uma "
        "pergunta abaixo. Exemplo: *Qual é o preço do Desodorante Corporal "
        "Frangipani Flower 100ml?*"
    )

for mensagem in st.session_state.mensagens:
    with st.chat_message(mensagem["role"], avatar="🧑" if mensagem["role"] == "user" else "📄"):
        st.markdown(mensagem["content"])
        if mensagem.get("fontes"):
            exibir_fontes(mensagem["fontes"])

pergunta = st.chat_input("Pergunte algo sobre os seus documentos…")

if pergunta is not None:
    pergunta = pergunta.strip()

    if not pergunta:
        st.warning("Digite uma pergunta válida.")
    else:
        st.session_state.mensagens.append({"role": "user", "content": pergunta})

        with st.chat_message("user", avatar="🧑"):
            st.markdown(pergunta)

        with st.chat_message("assistant", avatar="📄"):
            resposta = None
            fontes = []

            try:
                with st.spinner("Buscando nos documentos…"):
                    resultados = buscar_chunks(pergunta)

                if not resultados:
                    resposta = (
                        "Não encontrei informações relevantes nos documentos "
                        "indexados. Verifique se algum PDF já foi processado."
                    )
                else:
                    with st.spinner("Gerando resposta…"):
                        resposta = gerar_resposta(pergunta, resultados)
                    fontes = extrair_fontes(resultados)

            except Exception as erro:  # noqa: BLE001
                print(f"[IntelliDoc] Erro ao responder: {erro!r}")
                st.error(
                    mensagem_erro_openai(erro)
                    or "Ocorreu um erro inesperado ao gerar a resposta."
                )

            if resposta:
                st.markdown(resposta)
                exibir_fontes(fontes)
                st.session_state.mensagens.append(
                    {"role": "assistant", "content": resposta, "fontes": fontes}
                )
