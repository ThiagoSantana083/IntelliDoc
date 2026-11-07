"""Armazenamento estruturado dos produtos (SQLite).

Fica em data/produtos.sqlite3, separado do ChromaDB (data/db), que continua
guardando os chunks de texto do RAG sem nenhuma alteração.

Cada documento é identificado pelo mesmo caminho absoluto usado como
"source" no ChromaDB, o que mantém os dois bancos alinhados.
"""

import sqlite3
from contextlib import closing
from datetime import datetime

from catalogo import config

CAMPOS_PRODUTO = (
    "codigo", "nome", "tipo", "preco_texto", "preco_valor", "familia_olfativa",
    "publico", "linha", "descricao", "pagina", "outras_paginas", "imagem", "imagem_status",
    "imagem_tipo", "caixa",
)

# Colunas acrescentadas depois da primeira versão (migração automática).
_COLUNAS_NOVAS = {
    "imagem_tipo": "TEXT",  # isolada (fundo removido) | recorte (recorte da página)
    "caixa": "TEXT",        # posição da foto na página: "x0,y0,x1,y1" em pontos do PDF
}

_ESQUEMA = """
CREATE TABLE IF NOT EXISTS documentos (
    documento     TEXT PRIMARY KEY,
    nome_arquivo  TEXT NOT NULL,
    hash          TEXT NOT NULL,
    paginas       INTEGER NOT NULL,
    status        TEXT NOT NULL,          -- completo | parcial
    pasta_imagens TEXT,
    processado_em TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS produtos (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    documento        TEXT NOT NULL REFERENCES documentos(documento) ON DELETE CASCADE,
    nome_arquivo     TEXT NOT NULL,
    codigo           TEXT,
    nome             TEXT NOT NULL,
    tipo             TEXT,
    preco_texto      TEXT,
    preco_valor      REAL,
    familia_olfativa TEXT,
    publico          TEXT,
    linha            TEXT,
    descricao        TEXT,
    pagina           INTEGER NOT NULL,
    outras_paginas   TEXT,
    imagem           TEXT,
    imagem_status    TEXT
);

CREATE INDEX IF NOT EXISTS idx_produtos_documento ON produtos(documento);
CREATE INDEX IF NOT EXISTS idx_produtos_preco ON produtos(preco_valor);
"""


def _conectar():
    config.PASTA_DADOS.mkdir(parents=True, exist_ok=True)
    conexao = sqlite3.connect(config.CAMINHO_BANCO_PRODUTOS, timeout=30)
    conexao.row_factory = sqlite3.Row
    conexao.execute("PRAGMA foreign_keys = ON")
    return conexao


def inicializar():
    with closing(_conectar()) as conexao, conexao:
        conexao.executescript(_ESQUEMA)
        existentes = {linha["name"] for linha in conexao.execute("PRAGMA table_info(produtos)")}
        for coluna, tipo in _COLUNAS_NOVAS.items():
            if coluna not in existentes:
                conexao.execute(f"ALTER TABLE produtos ADD COLUMN {coluna} {tipo}")


def atualizar_foto(produto_id, imagem, imagem_tipo, caixa):
    """Atualiza somente a foto de um produto (sem tocar nos demais dados)."""
    inicializar()
    with closing(_conectar()) as conexao, conexao:
        conexao.execute(
            "UPDATE produtos SET imagem = ?, imagem_tipo = ?, caixa = ? WHERE id = ?",
            (imagem, imagem_tipo, caixa, produto_id),
        )


def obter_documento(documento):
    inicializar()
    with closing(_conectar()) as conexao:
        linha = conexao.execute(
            "SELECT * FROM documentos WHERE documento = ?", (documento,)
        ).fetchone()
        return dict(linha) if linha else None


def documento_atualizado(documento, hash_arquivo):
    """True se o documento já foi processado por completo com este conteúdo."""
    registro = obter_documento(documento)
    return bool(registro and registro["hash"] == hash_arquivo and registro["status"] == "completo")


def substituir_produtos(documento, nome_arquivo, hash_arquivo, paginas, status, pasta_imagens, produtos):
    """Grava os produtos do documento, substituindo a versão anterior.

    Tudo ocorre em uma única transação: ou a nova versão é gravada por
    inteiro, ou a anterior é mantida.
    """
    inicializar()
    with closing(_conectar()) as conexao, conexao:
        conexao.execute("DELETE FROM produtos WHERE documento = ?", (documento,))
        conexao.execute(
            """
            INSERT INTO documentos (documento, nome_arquivo, hash, paginas, status, pasta_imagens, processado_em)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(documento) DO UPDATE SET
                nome_arquivo = excluded.nome_arquivo, hash = excluded.hash,
                paginas = excluded.paginas, status = excluded.status,
                pasta_imagens = excluded.pasta_imagens, processado_em = excluded.processado_em
            """,
            (documento, nome_arquivo, hash_arquivo, paginas, status, pasta_imagens,
             datetime.now().isoformat(timespec="seconds")),
        )
        conexao.executemany(
            f"""
            INSERT INTO produtos (documento, nome_arquivo, {", ".join(CAMPOS_PRODUTO)})
            VALUES (?, ?, {", ".join("?" for _ in CAMPOS_PRODUTO)})
            """,
            [(documento, nome_arquivo, *(p.get(c) for c in CAMPOS_PRODUTO)) for p in produtos],
        )


def listar_produtos():
    inicializar()
    with closing(_conectar()) as conexao:
        linhas = conexao.execute("SELECT * FROM produtos ORDER BY nome_arquivo, pagina, id").fetchall()
        return [dict(linha) for linha in linhas]


def obter_produtos(ids):
    """Produtos pelos ids, na mesma ordem recebida."""
    if not ids:
        return []
    inicializar()
    with closing(_conectar()) as conexao:
        marcadores = ", ".join("?" for _ in ids)
        linhas = conexao.execute(f"SELECT * FROM produtos WHERE id IN ({marcadores})", list(ids)).fetchall()
    por_id = {linha["id"]: dict(linha) for linha in linhas}
    return [por_id[i] for i in ids if i in por_id]


def resumo_documentos():
    """{documento: {"produtos": n, "com_imagem": n, "status": ..., "nome_arquivo": ...}}"""
    inicializar()
    with closing(_conectar()) as conexao:
        linhas = conexao.execute(
            """
            SELECT d.documento, d.nome_arquivo, d.status,
                   COUNT(p.id) AS produtos,
                   SUM(CASE WHEN p.imagem IS NOT NULL THEN 1 ELSE 0 END) AS com_imagem
            FROM documentos d LEFT JOIN produtos p ON p.documento = d.documento
            GROUP BY d.documento
            """
        ).fetchall()
    return {linha["documento"]: dict(linha) for linha in linhas}
