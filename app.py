
# ======================================
#             IntelliDoc
#    Assistente de Documentação Inteligente
# ======================================

# Importanto dependências para o projeto
from pathlib import Path

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

def validar_documento(caminho):
    arquivo = Path(caminho)

    if arquivo.exists() and arquivo.is_file() and arquivo.suffix.lower() == ".pdf":
        return True
    else:
        return False

def main():

    while True:
        print("\n========== IntelliDoc ==========")
        print("1 - Cadastrar documento")
        print("2 - Listar documentos")
        print("3 - Consultar documento")
        print("4 - Excluir documento")
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

        elif opcao == "0":
            print("\nEncerrando o IntelliDoc...")
            break

        else:
            print("\nOpção inválida. Tente novamente.")


if __name__ == "__main__":
    main()
