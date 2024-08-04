"""
This file contains all the needed Factory Method

author: L. Saetta
last update: 2024-08-04

"""

from langchain.chains import create_history_aware_retriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_qdrant import QdrantVectorStore

# for Hybrid Search
from langchain_qdrant import FastEmbedSparse, RetrievalMode

from langchain_community.chat_models.oci_generative_ai import ChatOCIGenAI

from oci_cohere_embeddings_utils import OCIGenAIEmbeddingsWithBatch

from prompts import CONTEXT_Q_PROMPT, QA_PROMPT

from config import (
    OCI_EMBED_MODEL,
    OCI_GENAI_MODEL,
    ENDPOINT,
    COLLECTION_NAME,
    TOP_K,
    TEMPERATURE,
    MAX_TOKENS,
)

from config_private import COMPARTMENT_ID


def get_embed_model():
    """
    get the Embeddings Model
    """

    embed_model = OCIGenAIEmbeddingsWithBatch(
        auth_type="API_KEY",
        model_id=OCI_EMBED_MODEL,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
    )

    return embed_model


def get_vector_store(docs, embed_model, collection_name="my_coll_name"):
    """
    Create a QDRANT Vector Store for the Documents provided.

    All in memory. Hybrid Search.
    """

    # BM25 implementation
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")

    qdrant = QdrantVectorStore.from_documents(
        docs,
        embedding=embed_model,
        sparse_embedding=sparse_embeddings,
        location=":memory:",
        collection_name=collection_name,
        retrieval_mode=RetrievalMode.HYBRID,
    )

    return qdrant


def get_llm(llm_model, temperature=TEMPERATURE):
    """
    return the llm model
    """
    chat = ChatOCIGenAI(
        model_id=llm_model,
        service_endpoint=ENDPOINT,
        compartment_id=COMPARTMENT_ID,
        is_stream=False,
        model_kwargs={"temperature": temperature, "max_tokens": MAX_TOKENS},
    )

    return chat


def build_chain(docs):
    """
    Build the entire chain

    create an initial set of docs, loading from the provided pdf
    """
    embed_model = get_embed_model()

    v_store = get_vector_store(docs, embed_model, collection_name=COLLECTION_NAME)

    retriever = v_store.as_retriever(search_kwargs={"k": TOP_K})

    llm = get_llm(llm_model=OCI_GENAI_MODEL)

    # steps to add chat_history
    # 1. create a retriever using chat history
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, CONTEXT_Q_PROMPT
    )

    # 2. create the chain for answering
    # we need to use a different prompt from the one used to
    # condense the standalone question

    question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)

    # 3, the entire chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain
