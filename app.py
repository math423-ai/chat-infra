import json
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from rag import get_query_engine


# =========================
# CONFIGURAÇÃO DE LOG
# =========================

# Cria a pasta logs na mesma pasta onde está o app.py
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "chatbot_messages.jsonl"

# Garante que a pasta e o arquivo existam ao iniciar o app
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE.touch(exist_ok=True)


def get_session_id():
    """
    Cria um ID único para cada sessão do usuário no Streamlit.
    """
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    return st.session_state.session_id


def log_message(role, content, sources=None):
    """
    Registra mensagens do usuário e respostas do chatbot em formato JSONL.
    Cada linha do arquivo é um registro JSON.
    """
    log_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "session_id": get_session_id(),
        "role": role,
        "content": content,
    }

    if sources is not None:
        log_record["sources"] = sources

    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(log_record, ensure_ascii=False, default=str) + "\n")


def extract_sources(response):
    """
    Extrai as fontes utilizadas pelo RAG para salvar no log.
    """
    sources = []

    for i, node in enumerate(getattr(response, "source_nodes", [])):
        node_text = ""
        metadata = {}

        if hasattr(node, "node"):
            node_text = getattr(node.node, "text", "")
            metadata = getattr(node.node, "metadata", {}) or {}

        sources.append({
            "source_number": i + 1,
            "text_preview": node_text[:300],
            "metadata": metadata,
        })

    return sources


# =========================
# STREAMLIT
# =========================

st.set_page_config(page_title="Chatbot RAG", layout="wide")

st.title("Helpdesk Predial")

# Opcional: mostra onde o log está sendo salvo
# st.sidebar.caption(f"Arquivo de log: {LOG_FILE}")


# =========================
# INICIALIZAÇÃO DO RAG
# =========================

@st.cache_resource
def load_engine():
    return get_query_engine()


query_engine = load_engine()


# =========================
# HISTÓRICO
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# Exibir histórico
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# =========================
# INPUT DO USUÁRIO
# =========================

user_input = st.chat_input("Digite sua pergunta...")


if user_input:
    # Salvar e exibir mensagem do usuário
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    log_message(
        role="user",
        content=user_input
    )

    with st.chat_message("user"):
        st.markdown(user_input)


    # Gerar resposta do chatbot
    with st.chat_message("assistant"):
        with st.spinner("Pensando..."):
            response = query_engine.query(user_input)
            answer = str(response)

            sources = extract_sources(response)

            # Mostrar fontes
            with st.expander("Fontes utilizadas"):
                for i, node in enumerate(response.source_nodes):
                    st.markdown(f"**Fonte {i + 1}:**")
                    st.write(node.node.text[:300])

            st.markdown(answer)


    # Salvar e registrar resposta do chatbot
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    log_message(
        role="assistant",
        content=answer,
        sources=sources
    )