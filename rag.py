import os
from pathlib import Path

import pymupdf4llm
from llama_index.core import VectorStoreIndex, Document

# Carrega variáveis do arquivo .env, se ele existir.
# Se você não usar .env, o programa continuará lendo as variáveis do Windows.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.storage.storage_context import StorageContext
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAIResponses

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate


# =========================
# CONFIGURAÇÕES
# =========================
DOCS_PATH = "./docs"

# Modelo OpenAI utilizado pelo RAG
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

# Por padrão, mantém embeddings locais.
# Se quiser melhorar a recuperação dos documentos usando embeddings da OpenAI,
# defina no .env:
# USE_OPENAI_EMBEDDINGS=true
USE_OPENAI_EMBEDDINGS = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"


# =========================
# PROMPT
# =========================
QA_PROMPT = PromptTemplate(
"""
Você é um assistente de helpdesk predial para escolas Fatecs, Etecs e Administração Central.

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
- Se não houver informações suficientes no contexto, responda exatamente:
"Não encontrei informações suficientes nos documentos para responder com segurança. Por favor, forneça mais detalhes."

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
    docs_path = Path(DOCS_PATH)

    if not docs_path.exists():
        raise RuntimeError(
            f"A pasta de documentos '{DOCS_PATH}' não foi encontrada. "
            "Verifique se a pasta docs existe na raiz do projeto."
        )

    documents = []

    for file_path in sorted(docs_path.rglob("*")):
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
            text = file_path.read_text(encoding="utf-8", errors="ignore").strip()

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
                "USE_OPENAI_EMBEDDINGS=true foi definido, mas OPENAI_API_KEY não foi encontrada."
            )

        # Embedding da OpenAI.
        # Melhora a recuperação semântica, mas envia os chunks dos documentos para a API.
        from llama_index.embeddings.openai import OpenAIEmbedding

        embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=api_key
        )

        dimension = 1536

    else:
        # Embedding local.
        # Não envia os documentos para a OpenAI na etapa de indexação.
        embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        dimension = 384

    return embed_model, dimension


# =========================
# INDEXAÇÃO COM CHUNKING
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
# LLM - OPENAI GPT-5-MINI
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

        # Para RAG, mantenha deterministicidade.
        temperature=0.0,

        # Em modelos GPT-5, use max_output_tokens.
        # Evita respostas vazias por limite baixo de saída.
        max_output_tokens=1200,

        # Raciocínio baixo para equilibrar custo/qualidade.
        # Se quiser mais precisão, teste "medium".
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

        # Aumentei de 5 para 8 para enviar mais trechos relevantes ao modelo.
        # Isso pode melhorar respostas quando a informação está espalhada no PDF.
        similarity_top_k=8,

        text_qa_template=QA_PROMPT,
        response_mode="compact"
    )

    # Wrapper opcional para debug no terminal
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