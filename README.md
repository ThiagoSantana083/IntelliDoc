# IntelliDoc — Assistente de Documentação Inteligente

Assistente RAG (Retrieval-Augmented Generation) para catálogos e documentos em PDF.
Responde perguntas sobre o conteúdo e apresenta os produtos encontrados em cartões
visuais, com a foto original recortada do catálogo, nome, preço, família olfativa,
página e arquivo de origem.

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

Crie o arquivo `.env` na pasta do projeto:

```
OPENAI_API_KEY=sua-chave-aqui
```

Interface web:

```bash
streamlit run interface.py
```

Aplicação de terminal (original): `python app.py`

## Como funciona

| Etapa | Onde | O que faz |
|-------|------|-----------|
| Texto | `criar_db.py` | Extrai o texto do PDF, divide em chunks, gera embeddings (`text-embedding-3-small`) e grava no ChromaDB (`data/db`). Responde perguntas gerais com `gpt-4o-mini`. |
| Produtos | `catalogo/extracao.py` | Para cada página com preços, um modelo de visão lê a página e os blocos de texto e estrutura os produtos. Cada campo é conferido com o texto real da página; o que não puder ser comprovado é descartado. |
| Fotos | `catalogo/imagens.py` | A foto de cada produto é recortada da própria página (PyMuPDF), refinada com zoom e confirmada por um segundo modelo. Sem confirmação, o cartão mostra "Imagem não localizada". |
| Armazenamento | `catalogo/repositorio.py` | Produtos em SQLite (`data/produtos.sqlite3`); fotos em `data/imagens/`. |
| Busca | `catalogo/busca.py` | O modelo interpreta a pergunta; nome/linha, público, família olfativa e preço são filtrados de forma exata sobre todos os produtos. Perguntas gerais seguem para o RAG de texto. |
| Interface | `interface.py` + `catalogo/apresentacao.py` | Upload, processamento, histórico e grade de cartões com paginação. |

## Processar catálogos pelo terminal

```bash
python -m catalogo.processamento --todos            # todos os PDFs de data/documents
python -m catalogo.processamento "data/documents/arquivo.pdf"
python -m catalogo.processamento --todos --forcar   # reprocessa mesmo sem mudanças
```

Documentos já processados com o mesmo conteúdo são ignorados, e uma nova versão
substitui a anterior (sem duplicar produtos, chunks ou imagens).

## Configuração opcional (`.env`)

| Variável | Padrão | Uso |
|----------|--------|-----|
| `INTELLIDOC_MODELO_VISAO` | `gpt-4.1` | Identificação dos produtos e localização das fotos |
| `INTELLIDOC_MODELO_VERIFICACAO` | `gpt-4.1-mini` | Confirmação de cada foto |
| `INTELLIDOC_MODELO_TEXTO` | `gpt-4o-mini` | Interpretação das perguntas |
| `INTELLIDOC_TRABALHADORES` | `6` | Páginas processadas em paralelo |

O `gpt-4o-mini` não é usado para localizar fotos: nos testes ele não acertou a
posição dos produtos na página.
