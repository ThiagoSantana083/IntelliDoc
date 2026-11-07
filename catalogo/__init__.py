"""Catálogo inteligente do IntelliDoc.

Módulos:
    config         Caminhos e modelos (configuráveis pelo .env).
    texto          Normalização de texto e conversão de preços.
    imagens        Renderização das páginas e recorte das fotos dos produtos.
    extracao       Identificação e estruturação dos produtos de cada página.
    repositorio    Armazenamento dos produtos em SQLite.
    processamento  Processamento completo de um catálogo PDF.
    busca          Interpretação das perguntas e recuperação de produtos.
    apresentacao   Cartões e grade de produtos (Streamlit).
"""
