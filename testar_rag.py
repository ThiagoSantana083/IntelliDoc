
from dotenv import load_dotenv
load_dotenv()

from criar_db import buscar_chunks, gerar_resposta

perguntas = [
    "Qual é o preço do Desodorante Corporal Frangipani Flower 100ml?",
    "Qual é o preço do computador vendido pela Mahogany?"
]

for pergunta in perguntas:

    print("\n" + "=" * 60)
    print("PERGUNTA:", pergunta)

    resultados = buscar_chunks(pergunta)

    if resultados:
        print("\nTrechos encontrados:", len(resultados))

        for posicao, (documento, relevancia) in enumerate(resultados, start=1):

            print(f"\nResultado {posicao}")
            print("Relevância:", round(relevancia, 4))
            print("Página:", documento.metadata.get("page"))
            print("Conteúdo:")
            print(documento.page_content[:300])
            print("-" * 60)

        resposta = gerar_resposta(pergunta, resultados)

        print("\n--- RESPOSTA DO INTELLIDOC ---")
        print(resposta)

    else:
        print("Nenhum resultado encontrado.")