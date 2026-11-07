"""Processamento completo de um catálogo PDF (produtos + fotos).

Uso pelo terminal (a partir da pasta do projeto):
    python -m catalogo.processamento "data/documents/arquivo.pdf"
    python -m catalogo.processamento --todos
    python -m catalogo.processamento --todos --forcar
"""

import hashlib
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pymupdf

from catalogo import config, repositorio
from catalogo.extracao import processar_pagina
from catalogo.texto import normalizar


def hash_arquivo(caminho):
    sha = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1024 * 1024), b""):
            sha.update(bloco)
    return sha.hexdigest()


def _pasta_imagens(caminho, hash_documento):
    nome = re.sub(r"[^a-z0-9]+", "-", normalizar(Path(caminho).stem)).strip("-") or "documento"
    return config.PASTA_IMAGENS / f"{nome[:50]}-{hash_documento[:10]}"


def _remover_pasta_segura(pasta):
    """Remove uma pasta de imagens somente se estiver dentro de data/imagens."""
    if not pasta:
        return
    pasta = Path(pasta)
    if not pasta.is_absolute():
        pasta = config.BASE_DIR / pasta
    pasta = pasta.resolve()
    raiz = config.PASTA_IMAGENS.resolve()
    if pasta != raiz and raiz in pasta.parents and pasta.is_dir():
        shutil.rmtree(pasta, ignore_errors=True)


def _consolidar(produtos):
    """Remove repetições do mesmo produto em páginas diferentes.

    Mantém a ocorrência mais completa (com foto, depois com preço, depois a
    primeira página) e registra as demais páginas em "outras_paginas".
    """
    grupos = {}
    for produto in produtos:
        chave = produto["codigo"] or normalizar(produto["nome"])
        grupos.setdefault(chave, []).append(produto)

    consolidados = []
    for ocorrencias in grupos.values():
        ocorrencias.sort(
            key=lambda p: (p["imagem"] is None, p["preco_valor"] is None, p["pagina"])
        )
        principal = dict(ocorrencias[0])
        outras = sorted({p["pagina"] for p in ocorrencias[1:]} - {principal["pagina"]})
        principal["outras_paginas"] = ", ".join(str(p) for p in outras) or None
        consolidados.append(principal)

    consolidados.sort(key=lambda p: p["pagina"])
    return consolidados


def processar_catalogo(caminho, forcar=False, progresso=None):
    """Extrai e armazena os produtos de um PDF.

    progresso: função opcional chamada como progresso(concluidas, total).
    Retorna um resumo com status: processado | parcial | ignorado | erro.
    """
    caminho = Path(caminho).resolve()
    documento = str(caminho)
    nome_arquivo = caminho.name

    if not caminho.is_file() or caminho.suffix.lower() != ".pdf":
        return {"status": "erro", "nome": nome_arquivo, "mensagem": "Arquivo PDF não encontrado."}

    hash_documento = hash_arquivo(caminho)
    if not forcar and repositorio.documento_atualizado(documento, hash_documento):
        return {"status": "ignorado", "nome": nome_arquivo}

    try:
        with pymupdf.open(caminho) as pdf:
            total_paginas = pdf.page_count
    except Exception as erro:  # noqa: BLE001
        return {"status": "erro", "nome": nome_arquivo, "mensagem": f"Não foi possível abrir o PDF ({erro})."}

    anterior = repositorio.obter_documento(documento)
    pasta = _pasta_imagens(caminho, hash_documento)
    _remover_pasta_segura(pasta)  # resto de uma execução interrompida
    pasta.mkdir(parents=True, exist_ok=True)

    encontrados = []
    erros = []
    concluidas = 0

    with ThreadPoolExecutor(max_workers=config.TRABALHADORES) as executor:
        tarefas = {
            executor.submit(processar_pagina, str(caminho), indice, pasta): indice
            for indice in range(total_paginas)
        }
        for tarefa in as_completed(tarefas):
            indice = tarefas[tarefa]
            try:
                encontrados.extend(tarefa.result())
            except Exception as erro:  # noqa: BLE001
                print(f"[IntelliDoc] Erro na página {indice + 1} de {nome_arquivo}: {erro!r}")
                erros.append({"pagina": indice + 1, "erro": type(erro).__name__})
            concluidas += 1
            if progresso:
                progresso(concluidas, total_paginas)

    if erros and not encontrados:
        _remover_pasta_segura(pasta)
        return {
            "status": "erro",
            "nome": nome_arquivo,
            "mensagem": f"Falha ao processar {len(erros)} página(s). Nenhum produto foi gravado.",
            "erros": erros,
        }

    produtos = _consolidar(encontrados)
    status = "parcial" if erros else "completo"
    repositorio.substituir_produtos(
        documento, nome_arquivo, hash_documento, total_paginas, status,
        pasta.relative_to(config.BASE_DIR).as_posix(), produtos,
    )

    if anterior and anterior.get("pasta_imagens") and anterior["pasta_imagens"] != pasta.relative_to(config.BASE_DIR).as_posix():
        _remover_pasta_segura(anterior["pasta_imagens"])

    return {
        "status": "parcial" if erros else "processado",
        "nome": nome_arquivo,
        "paginas": total_paginas,
        "produtos": len(produtos),
        "com_imagem": sum(1 for p in produtos if p["imagem"]),
        "erros": erros,
    }


def _main(argumentos):
    forcar = "--forcar" in argumentos
    caminhos = [a for a in argumentos if not a.startswith("--")]
    if "--todos" in argumentos:
        caminhos = sorted(str(p) for p in config.PASTA_DOCUMENTOS.glob("*.pdf"))
    if not caminhos:
        print(__doc__)
        return 1

    for caminho in caminhos:
        print(f"\nProcessando {Path(caminho).name}...")

        def mostrar(feitas, total):
            print(f"\r  páginas: {feitas}/{total}", end="", flush=True)

        resumo = processar_catalogo(caminho, forcar=forcar, progresso=mostrar)
        print()
        print("  resultado:", resumo)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
