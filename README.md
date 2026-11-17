# 📄 PDF RAG Chatbot

Upload one or more PDFs and chat with them. Built with **Streamlit**, a **LangChain agent**, **Groq** (LLM) and **OpenAI embeddings**.

## How it works

1. PDFs are uploaded and loaded with `PyPDFDirectoryLoader`.
2. Text is split into 1000-character chunks (200 overlap).
3. Chunks are embedded (`text-embedding-3-large`) and stored in an in-memory vector store.
4. A LangChain agent (`openai/gpt-oss-20b` on Groq) calls a `retrieve_context` tool to fetch the top-3 chunks and answers from them.
5. Conversation memory is kept per browser session with `InMemorySaver`.

## Features

- Multi-PDF upload
- Source file and page shown in retrieved context
- Per-session chat memory
- Friendly errors for missing API keys or scanned (image-only) PDFs

## Setup

```bash
git clone https://github.com/<your-username>/pdf-rag-chatbot.git
cd pdf-rag-chatbot
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env             # then add your API keys
streamlit run app.py
```

Get keys from [Groq Console](https://console.groq.com/keys) and [OpenAI Platform](https://platform.openai.com/api-keys).

## Configuration

Edit the constants at the top of `app.py`: `EMBEDDING_MODEL`, `LLM_MODEL`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.

## Limitations

- Vector store is in memory, so it is rebuilt each time you upload.
- Scanned PDFs need OCR first.
- Uses paid OpenAI embeddings.

## License

MIT
