"""Configurações do catálogo.

Todos os caminhos são relativos à pasta do projeto. Os modelos podem ser
alterados no arquivo .env sem mexer no código. A chave da OpenAI continua
sendo lida somente do .env (OPENAI_API_KEY).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

PASTA_DADOS = BASE_DIR / "data"
PASTA_DOCUMENTOS = PASTA_DADOS / "documents"
PASTA_IMAGENS = PASTA_DADOS / "imagens"
CAMINHO_BANCO_PRODUTOS = PASTA_DADOS / "produtos.sqlite3"

# Modelo com visão usado para identificar os produtos e localizar as fotos.
MODELO_VISAO = os.getenv("INTELLIDOC_MODELO_VISAO", "gpt-4.1")

# Modelo usado para confirmar se o recorte mostra o produto correto.
MODELO_VERIFICACAO = os.getenv("INTELLIDOC_MODELO_VERIFICACAO", "gpt-4.1-mini")

# Modelo usado para interpretar perguntas e filtrar produtos.
MODELO_TEXTO = os.getenv("INTELLIDOC_MODELO_TEXTO", "gpt-4o-mini")

# Quantidade de páginas processadas em paralelo.
TRABALHADORES = max(1, int(os.getenv("INTELLIDOC_TRABALHADORES", "6")))

# Resolução usada para enviar as páginas ao modelo de visão.
DPI_ANALISE = 144

# Tamanho padronizado das imagens exibidas nos cartões (proporção 4:5).
LARGURA_CARTAO = 600
ALTURA_CARTAO = 750
