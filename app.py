
# ======================================
#             IntelliDoc
#    Assistente de Documentação Inteligente
# ======================================

# Importanto dependências para o projeto

 
from dotenv import load_dotenv  # Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

from criar_db import (
    validar_documento,
    extrair_texto_pdf,
    dividir_texto_chunks,
    armazenar_chunks,
    buscar_chunks,
    gerar_resposta
)

# Lista que armazenará os documentos cadastrados
documentos = []


def cadastrar_documento():
    print("\n--- Cadastro de Documento ---")

    nome = input("Nome do documento: ")
    caminho = input("Caminho do arquivo: ")

    # Verifica se o documento é válido
    if validar_documento(caminho):

        documento = {
            "nome": nome,
            "caminho": caminho,
            "processado": False
        }

        documentos.append(documento)

        print("\nDocumento cadastrado com sucesso!")

    else:
        print("\nNão foi possível cadastrar o documento.")
        print("Arquivo inexistente ou inválido.")



def listar_documentos():
    print("\n--- Documentos Cadastrados ---")

    if len(documentos) == 0:
        print("Nenhum documento cadastrado.")
        return

    for indice, documento in enumerate(documentos, start=1):

        if documento["processado"]:
            status = "Processado"
        else:
            status = "Pendente"

        print(f"{indice}. {documento['nome']} - {status}")

def consultar_documento():
    print("\n--- Consulta de Documento ---")
    if len(documentos) == 0:
        print("Nenhum documento cadastrado.")
        return
    else:
        indice_do_documento = int(input("Digite o número do documento que deseja consultar: "))
        for indice in range(len(documentos)):
            if indice == indice_do_documento - 1:
                documento = documentos[indice]
                print(f"\nNome: {documento['nome']}")
                print(f"Caminho: {documento['caminho']}")
                print(f"Status: {'Processado' if documento['processado'] else 'Pendente'}")
                return


def excluir_documento():
    print("\n--- Exclusão de Documento ---")

    if len(documentos) == 0:
        print("Nenhum documento cadastrado.")
        return

    indice_do_documento = int(input("Digite o número do documento que deseja excluir: "))

    for indice in range(len(documentos)):
        if indice == indice_do_documento - 1:

            documento = documentos[indice]

            print(f"\nDocumento selecionado: {documento['nome']}")
            print("\nDeseja realmente excluir este documento?")
            print("1 - Sim")
            print("2 - Não")

            resposta = input("\nEscolha uma opção: ")

            if resposta == "1":
                documentos.pop(indice)
                print(f"\nDocumento '{documento['nome']}' excluído com sucesso!")

            elif resposta == "2":
                print("\nExclusão cancelada.")

            else:
                print("\nOpção inválida. Exclusão cancelada.")

            return

    print("\nDocumento não encontrado.")

def consultar_intellidoc():

    print("\n--- Consultar IntelliDoc ---")

    pergunta = input("\nDigite sua pergunta: ")

    if not pergunta.strip():
        print("\nDigite uma pergunta válida.")
        return

    resultados = buscar_chunks(pergunta)

    resposta = gerar_resposta(pergunta, resultados)

    print("\n--- Resposta do IntelliDoc ---")
    print(resposta)

def main():

    while True:
        print("\n========== IntelliDoc ==========")
        print("1 - Cadastrar documento")
        print("2 - Listar documentos")
        print("3 - Consultar documento")
        print("4 - Excluir documento")
        print("5 - Extrair texto de documento")
        print("6 - Consultar IntelliDoc")
        print("0 - Sair")
        print("================================")

        opcao = input("Escolha uma opção: ")

        if opcao == "1":
            cadastrar_documento()

        elif opcao == "2":
            listar_documentos()

        elif opcao == "3":
            consultar_documento()

        elif opcao == "4":
            excluir_documento()

        elif opcao == "5":
            resultado = extrair_texto_pdf()

            if resultado:
                documento, caminho = resultado

                chunks = dividir_texto_chunks(documento, caminho)

                print("\nTotal de chunks:", len(chunks))

                armazenar_chunks(chunks)

        elif opcao == "6":
            consultar_intellidoc()

        elif opcao == "0":
            print("\nEncerrando o IntelliDoc...")
            break

        else:
            print("\nOpção inválida. Tente novamente.")


if __name__ == "__main__":
    main()
