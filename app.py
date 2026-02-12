import json
import os
import re
from time import sleep

import qdrant_client
import streamlit as st
from dotenv import load_dotenv
from llama_index.core import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq

load_dotenv()


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def get_collection_name():
    return os.getenv("QDRANT_COLLECTION", "custom-kb")


def get_domain_name():
    return os.getenv("ASSISTANT_DOMAIN", "your uploaded documents")


def get_embedding_model_name():
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def get_groq_model_name():
    return os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


class StaticResponse:
    def __init__(self, text):
        self.text = text


@st.cache_resource
def initialize_models():
    embedding_model_name = get_embedding_model_name()
    groq_model_name = get_groq_model_name()

    embed_model = FastEmbedEmbedding(model_name=embedding_model_name)
    llm = Groq(model=groq_model_name)
    request_timeout = int(float(os.getenv("QDRANT_TIMEOUT_SECONDS", "30")))
    client = qdrant_client.QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        prefer_grpc=_env_flag("QDRANT_PREFER_GRPC", default=False),
        timeout=request_timeout,
        check_compatibility=False,
    )
    return embed_model, llm, client


def build_message_templates(domain_name):
    return [
        ChatMessage(
            content=f"""
            You are an expert assistant focused on {domain_name}.
            You can respond in the same language as the user's query.

            Always structure your response in this format:
            <think>
            [Your step-by-step thinking process here]
            </think>

            [Your final answer here]
            """,
            role=MessageRole.SYSTEM,
        ),
        ChatMessage(
            content="""
            We have provided context information below.
            {context_str}
            ---------------------
            Given this information, please answer the question: {query}
            ---------------------
            If the question is not from the provided context, say `I don't know. Not enough information received.`
            """,
            role=MessageRole.USER,
        ),
    ]


def _show_dimension_mismatch_hint(error_text):
    pattern = r"expected dim:\s*(\d+)\s*,\s*got\s*(\d+)"
    match = re.search(pattern, error_text)
    if not match:
        return

    expected_dim, got_dim = match.group(1), match.group(2)
    st.error(
        "Embedding dimension mismatch. "
        f"Collection expects {expected_dim} but query embedding is {got_dim}. "
        "Set EMBEDDING_MODEL in .env to the same model used during indexing and restart Streamlit."
    )


def _extract_context_from_payload(payload):
    if not payload:
        return ""

    if payload.get("context"):
        return str(payload["context"])

    if payload.get("text"):
        return str(payload["text"])

    node_content = payload.get("_node_content")
    if isinstance(node_content, str) and node_content:
        try:
            parsed = json.loads(node_content)
            text_value = parsed.get("text")
            if text_value:
                return str(text_value)
        except Exception:
            return ""

    return ""


def search(query, client, embed_model, collection_name, k=5):
    request_timeout = int(float(os.getenv("QDRANT_TIMEOUT_SECONDS", "30")))
    query_embedding = embed_model.get_query_embedding(query)

    try:
        if not client.collection_exists(collection_name=collection_name):
            return []

        result = client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=k,
            timeout=request_timeout,
        )
        return result.points
    except Exception:
        try:
            fallback_client = qdrant_client.QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY"),
                prefer_grpc=False,
                timeout=request_timeout,
                check_compatibility=False,
            )
            result = fallback_client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=k,
                timeout=request_timeout,
            )
            return result.points
        except Exception as retrieval_error:
            error_text = str(retrieval_error)
            _show_dimension_mismatch_hint(error_text)
            st.warning(f"Vector retrieval is temporarily unavailable: {error_text}")
            return []


def pipeline(query, embed_model, llm, client):
    collection_name = get_collection_name()
    domain_name = get_domain_name()

    relevant_documents = search(query, client, embed_model, collection_name=collection_name)
    context = []
    for doc in relevant_documents:
        payload = getattr(doc, "payload", {}) or {}
        chunk = _extract_context_from_payload(payload)
        if chunk:
            context.append(chunk)

    if not context:
        return StaticResponse(
            "<think>\nNo supporting passage was retrieved from the selected index.\n</think>\n"
            f"I don't know. Not enough information received from collection `{collection_name}`."
        )

    chat_template = ChatPromptTemplate(message_templates=build_message_templates(domain_name))

    try:
        response = llm.complete(
            chat_template.format(
                context_str="\n".join(context),
                query=query,
            )
        )
        return response
    except Exception as llm_error:
        return StaticResponse(
            "<think>\nThe language model call failed while generating a response.\n</think>\n"
            f"Could not generate an answer right now: {llm_error}"
        )


def extract_thinking_and_answer(response_text):
    try:
        thinking = response_text[
            response_text.find("<think>") + 7 : response_text.find("</think>")
        ].strip()
        answer = response_text[response_text.find("</think>") + 8 :].strip()
        return thinking, answer
    except Exception:
        return "", response_text


def main():
    domain_name = get_domain_name()
    collection_name = get_collection_name()
    embedding_model_name = get_embedding_model_name()
    groq_model_name = get_groq_model_name()

    st.title(os.getenv("APP_TITLE", "Custom RAG Assistant"))
    st.markdown(
        """
        <style>
        div[data-testid="stChatInput"] {
            bottom: 3.1rem;
        }

        .main .block-container {
            padding-bottom: 8rem;
        }

        #anurag-footer {
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 0.45rem 1rem;
            border-top: 1px solid rgba(120, 120, 120, 0.25);
            background: rgba(255, 255, 255, 0.92);
            backdrop-filter: blur(6px);
            font-size: 0.88rem;
            z-index: 999;
        }

        #anurag-footer a {
            text-decoration: none;
            font-weight: 600;
        }

        @media (prefers-color-scheme: dark) {
            #anurag-footer {
                background: rgba(14, 17, 23, 0.92);
                border-top: 1px solid rgba(250, 250, 250, 0.15);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    embed_model, llm, client = initialize_models()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.caption(f"Domain: {domain_name}")
        st.caption(f"Collection: {collection_name}")
        st.caption(f"Embedding: {embedding_model_name}")
        st.caption(f"LLM: {groq_model_name}")
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                thinking, answer = extract_thinking_and_answer(message["content"])
                with st.expander("Show thinking process"):
                    st.markdown(thinking)
                st.markdown(answer)
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input(f"Ask your question about {domain_name}..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            with st.spinner("Thinking..."):
                full_response = pipeline(prompt, embed_model, llm, client)
                thinking, answer = extract_thinking_and_answer(full_response.text)

                with st.expander("Show thinking process"):
                    st.markdown(thinking)

                response = ""
                for chunk in answer.split():
                    response += chunk + " "
                    message_placeholder.markdown(response + "▌")
                    sleep(0.05)

                message_placeholder.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": full_response.text})

    st.markdown(
        """
        <div id="anurag-footer">
            Designed and developed by <strong>Anurag Singh</strong> |
            <a href="https://github.com/Anurag-M1" target="_blank">GitHub</a> |
            <a href="https://instagram.com/ca_anuragsingh" target="_blank">Instagram</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
