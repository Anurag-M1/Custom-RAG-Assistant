# Custom RAG Assistant (Qdrant + Groq + Streamlit)

A Streamlit-based RAG app that lets you chat with your own documents using:
- Qdrant (vector database)
- FastEmbed (embeddings)
- Groq (LLM)
- 
<img width="1456" height="792" alt="CRA" src="https://github.com/user-attachments/assets/5cc34025-286d-414c-b2c3-823f27a5b6c0" />

## Features

- Query any custom knowledge base collection in Qdrant
- Works with `.txt`, `.md`, `.pdf`, `.json`, `.csv`
- Configurable embedding model and LLM via `.env`
- Handles common retrieval issues (timeouts, model-dimension mismatch hints)
- 
<img width="575" height="728" alt="RaG GpT" src="https://github.com/user-attachments/assets/dd3e6d0b-8a94-457e-abf0-b8d9203eccb8" />

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

## Deploy on Render

If Render uses Python 3.14, dependency installation may fail (for example `jiter` trying to build from source).
This repo includes deployment config to force Python 3.11.

### Option 1 (recommended)
Use `render.yaml` from this repo.

### Option 2 (manual Render settings)
1. Set environment variable `PYTHON_VERSION=3.11.11`
2. Build command:
   `pip install --upgrade pip && pip install -r requirements.txt`
3. Start command:
   `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`

### Required environment variables on Render
- `GROQ_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

Optional defaults already defined in `render.yaml`:
- `QDRANT_COLLECTION`
- `ASSISTANT_DOMAIN`
- `EMBEDDING_MODEL`
- `GROQ_MODEL`
- `QDRANT_PREFER_GRPC`
- `QDRANT_TIMEOUT_SECONDS`
- `APP_TITLE`
