"""
Test 01
"""

from factory import build_chain
from chunk_utils import load_books_and_split
from utils import get_console_logger

from config import DOCS_DIR

#
# Main
#
logger = get_console_logger()

# load pages and chunks
logger.info("")
logger.info("Loading documents...")

docs = load_books_and_split(DOCS_DIR)

rag_chain = build_chain(docs=docs)

QUESTIONS = ["Does the document contains a clear list of goals? List the goals."]

logger.info("Processing questions...")
logger.info("")

for i, question in enumerate(QUESTIONS):

    print(f"Question n. {i+1}")
    print("")

    input_msg = {
        "input": question,
        "chat_history": [],
    }

    response = rag_chain.invoke(input_msg)

    print(response["answer"])
    print("")
