import os
import uuid

import streamlit as st
from langchain_openai import ChatOpenAI

from graph import build_graph, stream_answer
from ingest import run as ingest_run
from retriever import list_all, search

LLM_HOST = os.environ.get("LLM_HOST")
LLM_MODEL_ALIAS = os.environ.get("LLM_MODEL_ALIAS")


@st.cache_resource
def _graph():
    return build_graph()


@st.cache_resource
def _db_status() -> str | None:
    """Return None if DB is healthy, or an error message string."""
    try:
        from store import check_connection
        check_connection()
        return None
    except RuntimeError as exc:
        return str(exc)


st.set_page_config(page_title="Sherlock Holmes stories", layout="wide")
st.title("Sherlock Holmes stories")

if db_error := _db_status():
    st.error(f"⚠️ {db_error}", icon="🛑")

# --- Sidebar ---
with st.sidebar:
    st.header("Admin")
    if st.button("Re-ingest documents", use_container_width=True):
        progress_bar = st.progress(0, text="Ingesting…")
        try:
            def _on_progress(current, total):
                progress_bar.progress(current / total, text=f"Ingesting… {current}/{total}")

            count = ingest_run(on_progress=_on_progress)
            progress_bar.empty()
            st.success(f"Done. Ingested {count} chunks.")
        except Exception as exc:
            progress_bar.empty()
            st.error(f"Ingest failed: {exc}")

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_debug = None
        st.session_state.free_messages = []
        st.rerun()

# Initialise session state
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_debug" not in st.session_state:
    st.session_state.last_debug = None
if "free_messages" not in st.session_state:
    st.session_state.free_messages = []  # list of {"role", "content"}

# --- Tabs ---
search_tab, chat_tab, free_chat_tab, all_tab = st.tabs(["Search Docs", "Chat about Docs", "Chat about anything", "All Docs"])

with search_tab:
    query = st.text_input("Search", placeholder="e.g. Hound of the Baskervilles")
    if st.button("Search", key="search_btn"):
        if not query.strip():
            st.warning("Enter a search term.")
        else:
            try:
                with st.spinner("Searching..."):
                    results = search(query, k=5)
                st.markdown(f"**{len(results)} results**")
                for i, doc in enumerate(results, 1):
                    topic = doc.metadata.get("topic") or doc.metadata.get("source", "")
                    with st.expander(f"{i}. {topic}"):
                        st.caption(doc.metadata.get("source", ""))
                        st.write(doc.page_content)
            except Exception as exc:
                st.error(f"Search failed: {exc}")

with chat_tab:
    debug_mode = st.toggle("Show debug panel")

    with st.container(height=500):
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if question := st.chat_input("Ask about Sherlock Holmes stories...", key="rag_input"):
        st.session_state.messages.append({"role": "user", "content": question})

        standalone = ""
        hypothesis = ""
        documents = []
        streamed = ""

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                with st.spinner("Thinking..."):
                    for event, payload in stream_answer(question, _graph(), st.session_state.thread_id):
                        if event == "standalone":
                            standalone = payload
                        elif event == "hypothesis":
                            hypothesis = payload
                        elif event == "docs":
                            documents = payload
                        elif event == "token":
                            streamed += payload
                            placeholder.markdown(streamed)
            except Exception as exc:
                placeholder.error(f"Error: {exc}")

        st.session_state.messages.append({"role": "assistant", "content": streamed})
        st.session_state.last_debug = {
            "standalone": standalone,
            "hypothesis": hypothesis,
            "documents": documents,
        }
        st.rerun()

    if debug_mode and st.session_state.last_debug:
        d = st.session_state.last_debug
        with st.expander("Debug — last turn", expanded=True):
            st.markdown("**Standalone question**")
            st.info(d.get("standalone", ""))
            st.markdown("**Hypothesis (HyDE)**")
            st.info(d.get("hypothesis", ""))
            st.markdown("**Retrieved chunks**")
            for i, doc in enumerate(d.get("documents", []), 1):
                title = doc.metadata.get("title", "")
                idx = doc.metadata.get("chunk_index", "")
                total = doc.metadata.get("chunk_total", "")
                st.markdown(f"*Chunk {i} — {title} [{idx}/{total}]*")
                st.code(doc.page_content, language=None)

with free_chat_tab:
    with st.container(height=500):
        for msg in st.session_state.free_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if question := st.chat_input("Ask anything...", key="free_input"):
        st.session_state.free_messages.append({"role": "user", "content": question})

        history = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.free_messages[:-1]
        ]

        with st.chat_message("assistant"):
            placeholder = st.empty()
            streamed = ""
            try:
                llm = ChatOpenAI(model=LLM_MODEL_ALIAS, base_url=LLM_HOST, api_key="sk-local", streaming=True)
                for chunk in llm.stream(history + [{"role": "user", "content": question}]):
                    if chunk.content:
                        streamed += chunk.content
                        placeholder.markdown(streamed)
            except Exception as exc:
                placeholder.error(f"Error: {exc}")

        st.session_state.free_messages.append({"role": "assistant", "content": streamed})
        st.rerun()

with all_tab:
    if st.button("Load all docs", key="load_all_btn"):
        try:
            with st.spinner("Loading..."):
                all_docs = list_all()
            st.markdown(f"**{len(all_docs)} documents**")
            for doc in all_docs:
                topic = doc.metadata.get("topic") or doc.metadata.get("source", "")
                with st.expander(topic):
                    st.caption(doc.metadata.get("source", ""))
                    st.write(doc.page_content)
        except Exception as exc:
            st.error(f"Failed to load docs: {exc}")
