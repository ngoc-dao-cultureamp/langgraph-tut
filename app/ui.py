import time  # used by re-ingest stub

import streamlit as st

from rag.graph import build_graph, stream_answer
from rag.retriever import search


@st.cache_resource
def _graph():
    return build_graph()


st.set_page_config(page_title="Knowledge Base", layout="wide")
st.title("Knowledge Base")

# --- Sidebar ---
with st.sidebar:
    st.header("Admin")
    if st.button("Re-ingest documents", use_container_width=True):
        with st.spinner("Ingesting..."):
            time.sleep(2)  # TODO: call ingest.pipeline.run()
        st.success("Done.")

# --- Tabs ---
search_tab, ask_tab = st.tabs(["Search Docs", "Ask a Question"])

with search_tab:
    query = st.text_input("Search", placeholder="e.g. performance reviews")
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

with ask_tab:
    question = st.text_input("Question", placeholder="e.g. What makes a good performance review?")
    if st.button("Ask", key="ask_btn"):
        if not question.strip():
            st.warning("Enter a question.")
        else:
            documents = []
            answer_container = st.container(border=True)
            answer_container.markdown("**Answer**")
            answer_placeholder = answer_container.empty()
            streamed = ""

            with st.spinner("Retrieving..."):
                for event, payload in stream_answer(question, _graph()):
                    if event == "docs":
                        documents = payload
                    elif event == "token":
                        streamed += payload
                        answer_placeholder.markdown(streamed)

            if documents:
                st.markdown("**Retrieved chunks**")
                for doc in documents:
                    topic = doc.metadata.get("topic") or doc.metadata.get("source", "")
                    with st.expander(f"{topic} — {doc.metadata.get('source', '')}"):
                        st.write(doc.page_content)
