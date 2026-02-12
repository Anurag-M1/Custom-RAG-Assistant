# Custom RAG Assistant (Qdrant + Groq + Streamlit)

A Streamlit-based RAG app that lets you chat with your own documents using:
- Qdrant (vector database)
- FastEmbed (embeddings)
- Groq (LLM)

## Features

- Query any custom knowledge base collection in Qdrant
- Works with `.txt`, `.md`, `.pdf`, `.json`, `.csv`
- Configurable embedding model and LLM via `.env`
- Handles common retrieval issues (timeouts, model-dimension mismatch hints)

## Project Files

- `app.py`: Streamlit chat application
- `index_custom_data.py`: Index local files into a Qdrant collection
- `requirements.txt`: Python dependencies
- `notebook/Bhagavad_Gita_Assistant.ipynb`: original notebook workflow

## Setup

```bash
cd "/Users/anurag/Desktop/RaG GpT"
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```env
GROQ_API_KEY=your_groq_api_key
QDRANT_URL=https://your-cluster-url
QDRANT_API_KEY=your_qdrant_api_key

QDRANT_COLLECTION=indian_constitution
ASSISTANT_DOMAIN=Indian Constitution

# Must match the embedding model used to index the same collection
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Must be a currently supported Groq model
GROQ_MODEL=llama-3.3-70b-versatile

QDRANT_PREFER_GRPC=false
QDRANT_TIMEOUT_SECONDS=30
APP_TITLE=Custom RAG Assistant
```

## Index Your Data

```bash
source .venv/bin/activate
python index_custom_data.py \
  --data-dir "/absolute/path/to/your/files" \
  --collection "my-custom-kb" \
  --embedding-model "sentence-transformers/all-MiniLM-L6-v2" \
  --recreate
```

After indexing, set `QDRANT_COLLECTION=my-custom-kb` in `.env`.

## Run App

```bash
cd "/Users/anurag/Desktop/RaG GpT"
source .venv/bin/activate
streamlit run app.py
```

If using an absolute path, quote it because the folder name contains spaces:

```bash
streamlit run "/Users/anurag/Desktop/RaG GpT/app.py"
```

## Troubleshooting

- `expected dim: X, got Y`:
  - Your `EMBEDDING_MODEL` does not match the model used to build that collection.
- `model ... decommissioned`:
  - Update `GROQ_MODEL` to a currently available Groq model.
- `I don't know. Not enough information received ...`:
  - Collection may be empty, wrong collection name, or payload format may not include retrievable text.

## Credits

Designed and developed by **Anurag Singh**  
GitHub: [github.com/Anurag-M1](https://github.com/Anurag-M1)  
Instagram: [instagram.com/ca_anuragsingh](https://instagram.com/ca_anuragsingh)
