import time

import streamlit as st

from rag.retriever import get_vector_store, search


@st.cache_resource
def _vector_store():
    return get_vector_store()


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
            with st.container(border=True):
                st.markdown("**Answer**")
                # TODO: replace with rag.graph streaming
                placeholder = st.empty()
                answer = "This is a placeholder answer. The RAG graph will stream the real answer here."
                streamed = ""
                for word in answer.split():
                    streamed += word + " "
                    placeholder.markdown(streamed)
                    time.sleep(0.05)

            st.markdown("**Retrieved chunks**")
            # TODO: replace with real retrieved docs from graph state
            chunks = [
                {"topic": "Performance Reviews", "source": "sample.txt", "document": "Sample retrieved chunk..."},
            ]
            for chunk in chunks:
                with st.expander(f"{chunk['topic']} — {chunk['source']}"):
                    st.write(chunk["document"])
