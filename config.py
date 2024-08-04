"""
Configuration file

Author: Luigi Saetta
Date created: 2024-04-27
Date last modified: 2024-08-04
Python Version: 3.11
"""

# title for the UI
TITLE = "AI Assistant with LangChain 🦜"
HELLO_MSG = "Ciao, come posso aiutarti?"

ADD_REFERENCES = True
VERBOSE = True

LANG_SUPPORTED = ["en", "it", "es", "fr", "de", "el", "nl", "ro"]


# for chunking
# in chars
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 100

# OCI GenAI model used for Embeddings
# to batch embedding with OCI
# with Cohere embeddings max is 96
# value: COHERE, OCI
EMBED_MODEL_TYPE = "OCI"
EMBED_BATCH_SIZE = 90
OCI_EMBED_MODEL = "cohere.embed-multilingual-v3.0"

# current endpoint for OCI GenAI (embed and llm) models
# switched to FRA (19/06)
ENDPOINT = "https://inference.generativeai.eu-frankfurt-1.oci.oraclecloud.com"


# retriever
TOP_K = 8
TOP_N = 4

# to limit chat_history
# probably in rfp can be kept low
MAX_MSGS_IN_CHAT = 2

# 23AI
# the name of the table with text and embeddings
COLLECTION_NAME = "MY_BOOKS"

# COHERE, OCI
LLM_MODEL_TYPE = "OCI"

# OCI
# OCI_GENAI_MODEL = "cohere.command"
# OCI_GENAI_MODEL = "meta.llama-3-70b-instruct"
# OCI_GENAI_MODEL = "cohere.command-r-16k"
OCI_GENAI_MODEL = "cohere.command-r-plus"

# params for LLM
TEMPERATURE = 0.1
MAX_TOKENS = 2048

# to enable streaming
DO_STREAMING = False
