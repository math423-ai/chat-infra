import os
from pathlib import Path

# Carrega variáveis do arquivo .env, se ele existir.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import pymupdf4llm

from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAIResponses
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate


# =========================
# CONFIGURAÇÕES
# =========================

BASE_DIR = Path(__file__).resolve().parent
DOCS_PATH = BASE_DIR / "docs"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

USE_OPENAI_EMBEDDINGS = (
    os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
)


# =========================
# PROMPT
# =========================

QA_PROMPT = PromptTemplate(
"""
Você é um assistente de helpdesk predial para escolas (Fatecs e Etecs).

Responda SEMPRE em português do Brasil (PT-BR), com linguagem clara, educada e profissional.

A resposta DEVE seguir EXATAMENTE este formato, quando relacionada a manutenção predial:

- Inicie com: "Prezado funcionário."
- Indique seu entendimento do problema a partir da correta interpretação da situação apresentada.
- Indique possíveis ações para sanar o problema.

Regras importantes:
- Baseie-se APENAS no contexto fornecido.
- NÃO invente informações fora do contexto.
- Respostas totalmente em português do Brasil, PT-BR, com linguagem clara, educada e profissional.
- Caso a pergunta não tenha relação com manutenção predial, você deve informar que não há relação e que não poderá ajudar.
- Se não houver informação suficiente no contexto, responda exatamente:
"Não encontrei informações suficientes nos documentos para responder com segurança."

Instruções para análise do contexto:
- Antes de responder, verifique cuidadosamente se algum trecho do contexto responde à pergunta.
- Use apenas as informações presentes no contexto.
- Se houver informações parciais, explique somente o que foi encontrado no contexto.
- Não cite procedimentos, contatos, prazos ou responsáveis que não estejam no contexto.

---------------------
Contexto:
{context_str}
---------------------

Pergunta:
{query_str}

Resposta:
"""
)


# =========================
# DOCUMENTOS
# =========================

def load_documents():
    if not DOCS_PATH.exists():
        raise RuntimeError(
            f"A pasta de documentos '{DOCS_PATH}' não foi encontrada. "
            "Verifique se a pasta docs existe na raiz do projeto."
        )

    documents = []

    for file_path in sorted(DOCS_PATH.rglob("*")):
        if file_path.is_dir():
            continue

        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            try:
                text = pymupdf4llm.to_markdown(str(file_path))
            except Exception as e:
                raise RuntimeError(
                    f"Erro ao extrair texto do PDF '{file_path.name}': {e}"
                )

            text = text.strip()

            if not text:
                raise RuntimeError(
                    f"O PDF '{file_path.name}' não retornou texto. "
                    "Ele pode ser um PDF escaneado como imagem. Nesse caso será necessário OCR."
                )

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                        "file_type": "pdf",
                    },
                )
            )

        elif suffix in [".txt", ".md"]:
            text = file_path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()

            if text:
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "file_name": file_path.name,
                            "file_path": str(file_path),
                            "file_type": suffix.replace(".", ""),
                        },
                    )
                )

    if not documents:
        raise RuntimeError(
            "Nenhum documento válido foi carregado. "
            "Verifique se a pasta docs contém PDFs, TXT ou MD com texto extraível."
        )

    return documents


# =========================
# EMBEDDINGS
# =========================

def load_embedding_model():
    api_key = os.getenv("OPENAI_API_KEY")

    if USE_OPENAI_EMBEDDINGS:
        if not api_key:
            raise RuntimeError(
                "USE_OPENAI_EMBEDDINGS=true foi definido, "
                "mas OPENAI_API_KEY não foi encontrada."
            )

        from llama_index.embeddings.openai import OpenAIEmbedding

        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=api_key
        )

        dimension = 1536

    else:
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        dimension = 384

    return embed_model, dimension


# =========================
# INDEXAÇÃO SEM FAISS
# =========================

def create_index():
    documents = load_documents()

    embed_model, _ = load_embedding_model()

    parser = SentenceSplitter(
        chunk_size=384,
        chunk_overlap=80
    )

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        transformations=[parser],
    )

    return index


# =========================
# LLM
# =========================

def load_llm():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "A variável de ambiente OPENAI_API_KEY não foi encontrada. "
            "Configure a chave da OpenAI antes de executar o projeto."
        )

    llm = OpenAIResponses(
        model=OPENAI_MODEL,
        api_key=api_key,
        temperature=0.0,
        max_output_tokens=1200,
        reasoning_options={
            "effort": "low"
        },
        timeout=120,
    )

    return llm


# =========================
# QUERY ENGINE
# =========================

def get_query_engine(debug=False):
    index = create_index()
    llm = load_llm()

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=8,
        text_qa_template=QA_PROMPT,
        response_mode="compact"
    )

    if debug:
        original_query = query_engine.query

        def debug_query(q):
            response = original_query(q)

            print("\n====================")
            print("PERGUNTA:", q)
            print("--------------------")
            print("FONTES RECUPERADAS:")

            for i, node in enumerate(response.source_nodes):
                print(f"\n--- Fonte {i + 1} ---")
                print(node.node.text[:700])

            print("====================\n")

            return response

        query_engine.query = debug_query

    return query_engine