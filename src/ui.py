import time

import streamlit as st

from graph import build_graph, stream_answer
from retriever import list_all, search


@st.cache_resource
def _graph():
    return build_graph()


st.set_page_config(page_title="Sherlock Holmes", layout="wide")
st.title("Sherlock Holmes")

# --- Sidebar ---
with st.sidebar:
    st.header("Admin")
    if st.button("Re-ingest documents", use_container_width=True):
        with st.spinner("Ingesting..."):
            time.sleep(2)  # TODO: call ingest.run()
        st.success("Done.")

    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.session_state.messages = []
        st.rerun()

# Initialise session state
if "history" not in st.session_state:
    st.session_state.history = []   # list of (question, answer) for graph context
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "debug"} for display

# --- Tabs ---
search_tab, chat_tab, all_tab = st.tabs(["Search Docs", "Chat", "All Docs"])

with search_tab:
    query = st.text_input("Search", placeholder="e.g. Hound of the Baskervilles")
    if st.button("Search", key="search_btn"):
        if not query.strip():
            st.warning("Enter a search term.")
        else:
            with st.spinner("Searching..."):
                results = search(query, k=5)
            st.markdown(f"**{len(results)} results**")
            for i, doc in enumerate(results, 1):
                topic = doc.metadata.get("topic") or doc.metadata.get("source", "")
                with st.expander(f"{i}. {topic}"):
                    st.caption(doc.metadata.get("source", ""))
                    st.write(doc.page_content)

with chat_tab:
    debug_mode = st.toggle("Show debug panel")

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if debug_mode and msg.get("debug"):
                with st.expander("Debug"):
                    d = msg["debug"]
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

    # New message input
    if question := st.chat_input("Ask about Sherlock Holmes..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        standalone = ""
        hypothesis = ""
        documents = []
        streamed = ""

        with st.chat_message("assistant"):
            placeholder = st.empty()
            with st.spinner("Thinking..."):
                for event, payload in stream_answer(question, st.session_state.history, _graph()):
                    if event == "standalone":
                        standalone = payload
                    elif event == "hypothesis":
                        hypothesis = payload
                    elif event == "docs":
                        documents = payload
                    elif event == "token":
                        streamed += payload
                        placeholder.markdown(streamed)

        # Persist to session state
        debug_data = {"standalone": standalone, "hypothesis": hypothesis, "documents": documents}
        st.session_state.messages.append({"role": "assistant", "content": streamed, "debug": debug_data})
        st.session_state.history.append((question, streamed))

with all_tab:
    if st.button("Load all docs", key="load_all_btn"):
        with st.spinner("Loading..."):
            all_docs = list_all()
        st.markdown(f"**{len(all_docs)} documents**")
        for doc in all_docs:
            topic = doc.metadata.get("topic") or doc.metadata.get("source", "")
            with st.expander(topic):
                st.caption(doc.metadata.get("source", ""))
                st.write(doc.page_content)
