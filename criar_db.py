from pathlib import Path
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

def validar_documento(caminho):
    arquivo = Path(caminho)

    if arquivo.exists() and arquivo.is_file() and arquivo.suffix.lower() == ".pdf":
        return True
    else:
        return False

def extrair_paginas_pdf(caminho):
    """Extrai o texto de cada página de um PDF a partir do caminho informado.

    Versão sem interação com o terminal (sem input/print), usada pela
    interface web e reaproveitada por extrair_texto_pdf().
    """

    leitor_do_arquivo = PdfReader(caminho)

    textos_por_pagina = []

    for pagina in leitor_do_arquivo.pages:

        pagina_texto = pagina.extract_text()

        if pagina_texto:
            textos_por_pagina.append(pagina_texto)
        else:
            textos_por_pagina.append("")

    return textos_por_pagina

def extrair_texto_pdf():

    print("\n--- Extração de Texto de Documento ---")

    caminho = input("Digite o Caminho do arquivo: ")

    if validar_documento(caminho):

        textos_por_pagina = extrair_paginas_pdf(caminho)

        print("Total de páginas processadas:", len(textos_por_pagina))
        print("Texto extraído do arquivo PDF.")

        return textos_por_pagina, caminho

    else:
        print("\nArquivo inválido ou não encontrado.")
        return

def dividir_texto_chunks(documento, caminho):

    separador_do_texto_em_chunks = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True
    )

    textos = []
    metadados = []

    for numero_pagina, texto in enumerate(documento, start=1):

        if texto.strip():

            textos.append(texto)

            metadados.append({
                "source": str(Path(caminho).resolve()),
                "page": numero_pagina
            })

    chunks = separador_do_texto_em_chunks.create_documents(
        textos,
        metadatas=metadados
    )

    return chunks

def criar_banco_vetorial():
    modelo_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db = Chroma(
        collection_name="intellidoc",
        embedding_function=modelo_embeddings,
        persist_directory="./data/db"
    )

    return db


def armazenar_chunks(chunks):

    if not chunks:
        print("\nNenhum conteúdo encontrado para armazenar.")
        return

    db = criar_banco_vetorial()

    caminho = chunks[0].metadata["source"]

    documento_existente = db.get(
        where={"source": caminho}
    )

    if documento_existente["ids"]:

        db.delete(
            where={"source": caminho}
        )

        print("\nVersão anterior do documento removida.")

    db.add_documents(documents=chunks)

    print("\nDocumento armazenado com sucesso!")
    print("Total de chunks armazenados:", len(chunks))

    return db



def buscar_chunks(pergunta):

    db = criar_banco_vetorial()

    resultados = db.similarity_search_with_relevance_scores(
        pergunta,
        k=10
    )

    return resultados



def gerar_resposta(pergunta, resultados):

    if not resultados:
        return "Não encontrei informações relevantes no catálogo."

    modelo = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    contexto = ""

    for documento, pontuacao in resultados:

        nome_arquivo = Path(
            documento.metadata["source"]
        ).name

        pagina = documento.metadata.get(
            "page", "não informada"
        )

        contexto += f"""
        Arquivo: {nome_arquivo}
        Página: {pagina}

        Conteúdo:
        {documento.page_content}
        """

    prompt = f"""
    Você é o IntelliDoc, um assistente de consulta de catálogos.

    Responda utilizando exclusivamente as informações
    presentes no contexto.

    Instruções:
    1. Responda de maneira clara, objetiva e organizada.
    2. Não invente informações.
    3. Informe o nome do produto quando identificado.
    4. Informe o preço somente quando estiver disponível no contexto.
    5. Informe o nome do arquivo e a página onde o produto aparece.
    6. Se o preço não estiver disponível, informe que não foi localizado.
    7. Se não houver evidências suficientes para responder,
       informe isso claramente.
    8. Não apresente informações sobre pontuação de relevância,
       chunks, embeddings ou funcionamento interno do sistema.
    9. Não afirme que uma informação não existe no catálogo inteiro
       quando ela não estiver nos trechos recuperados.

    Organize a resposta neste formato, quando as informações estiverem
    disponíveis:

    Nome do produto:
    Resposta:
    Preço:
    Localização no catálogo:

    Contexto:
    {contexto}

    Pergunta:
    {pergunta}

    Resposta:
    """

    resposta = modelo.invoke(prompt)

    return resposta.content.strip()
