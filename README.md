# IntelliDoc — Assistente de Documentação Inteligente

Assistente RAG (Retrieval-Augmented Generation) para catálogos e documentos em PDF.
Responde perguntas sobre o conteúdo e apresenta os produtos encontrados em cartões
visuais, com a foto do produto isolada do fundo (a partir da imagem original do
catálogo), nome, preço, família olfativa, página e arquivo de origem.

A interface tem modo claro e modo noturno: por padrão segue o tema do sistema, e
pode ser trocado em ⋮ (canto superior direito) → Settings → *Choose app theme*.

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
| Localização das fotos | `catalogo/imagens.py` | A posição da foto de cada produto na página é encontrada pelo modelo de visão, refinada com zoom e confirmada por um segundo modelo. Sem confirmação, o cartão mostra "Imagem não localizada". |
| Fotos isoladas | `catalogo/fotos.py` | O produto é separado do fundo localmente (`rembg`, modelo `isnet-general-use`), sem textos, linhas ou objetos de cena, e centralizado sobre um fundo claro com sombra suave. O `gpt-4.1-mini` compara com o original para garantir que está inteiro; se não aprovar, usa o recorte da página clareado. |
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

Para refazer apenas as fotos dos produtos já armazenados (sem nova extração do PDF):

```bash
python -m catalogo.fotos            # fotos ainda no estilo antigo
python -m catalogo.fotos --forcar   # todas
```

Na primeira execução, o modelo de remoção de fundo (~180 MB) é baixado para
`~/.rembg/models`.

## Configuração opcional (`.env`)

| Variável | Padrão | Uso |
|----------|--------|-----|
| `INTELLIDOC_MODELO_VISAO` | `gpt-4.1` | Identificação dos produtos e localização das fotos |
| `INTELLIDOC_MODELO_VERIFICACAO` | `gpt-4.1-mini` | Confirmação de cada foto |
| `INTELLIDOC_MODELO_TEXTO` | `gpt-4o-mini` | Interpretação das perguntas |
| `INTELLIDOC_MODELO_RECORTE` | `isnet-general-use` | Modelo local que separa o produto do fundo |
| `INTELLIDOC_TRABALHADORES` | `6` | Páginas processadas em paralelo |

O `gpt-4o-mini` não é usado para localizar fotos: nos testes ele não acertou a
posição dos produtos na página.
