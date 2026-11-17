"""PDF RAG Chatbot: upload PDFs and chat with them.

Stack: Streamlit, LangChain agent, Groq LLM, OpenAI embeddings, in-memory vector store.
"""
import os
import tempfile
import uuid

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.memory import InMemorySaver

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 Chat with your PDFs")

# ---------- config ----------
EMBEDDING_MODEL = "text-embedding-3-large"
LLM_MODEL = "openai/gpt-oss-20b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 3

# ---------- session state ----------
st.session_state.setdefault("document_uploaded", False)
st.session_state.setdefault("agent", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("thread_id", str(uuid.uuid4()))  # unique per browser session


def check_keys():
    missing = [k for k in ("GROQ_API_KEY", "OPENAI_API_KEY") if not os.getenv(k)]
    if missing:
        st.error(f"Missing environment variable(s): {', '.join(missing)}. See README.")
        st.stop()


def process_document(folder: str):
    """Load PDFs from `folder`, build the vector store and the agent."""
    docs = PyPDFDirectoryLoader(folder).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [c for c in splitter.split_documents(docs) if c.page_content.strip()]
    if not chunks:
        raise ValueError(
            "No text could be extracted. The PDFs may be scanned images (OCR needed)."
        )

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_db = InMemoryVectorStore.from_documents(documents=chunks, embedding=embeddings)

    llm = ChatGroq(model=LLM_MODEL)

    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve documents relevant to a query from the knowledge base."""
        results = vector_db.similarity_search(query=query, k=TOP_K)
        return "\n\n".join(
            f"[{os.path.basename(d.metadata.get('source', 'doc'))}, "
            f"page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
            for d in results
        )

    system_prompt = (
        "You are a helpful assistant that answers questions using retrieved context. "
        "The knowledge base consists of the uploaded documents. "
        "ALWAYS use the `retrieve_context` tool for questions about the documents. "
        "If the answer is not in the retrieved context, say you could not find it."
    )

    st.session_state.agent = create_agent(
        model=llm,
        tools=[retrieve_context],
        system_prompt=system_prompt,
        checkpointer=InMemorySaver(),
    )
    st.session_state.document_uploaded = True


# ---------- upload UI ----------
if not st.session_state.document_uploaded:
    check_keys()
    uploaded = st.file_uploader("Select PDF files", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Processing..."):
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    for file in uploaded:
                        safe_name = os.path.basename(file.name)
                        with open(os.path.join(tmp, safe_name), "wb") as f:
                            f.write(file.getvalue())
                    process_document(tmp)
            except Exception as e:
                st.error(f"Could not process the documents: {e}")
                st.stop()
        st.rerun()

# ---------- chat UI ----------
if st.session_state.document_uploaded and st.session_state.agent:
    with st.sidebar:
        if st.button("Upload different PDFs"):
            for key in ("document_uploaded", "agent", "messages", "thread_id"):
                st.session_state.pop(key, None)
            st.rerun()

    for m in st.session_state.messages:
        st.chat_message(m["role"]).markdown(m["content"])

    query = st.chat_input("Ask anything related to the uploaded documents...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        st.chat_message("user").markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.agent.invoke(
                        {"messages": [{"role": "user", "content": query}]},
                        {"configurable": {"thread_id": st.session_state.thread_id}},
                    )
                    answer = response["messages"][-1].content
                except Exception as e:
                    answer = f"Sorry, something went wrong: {e}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
