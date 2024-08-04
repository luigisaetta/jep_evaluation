"""
Test 01
"""

from langchain_core.messages import HumanMessage, AIMessage

from factory import build_chain
from chunk_utils import load_books_and_split
from utils import get_console_logger
from chat_history_queue import MessageQueue

from config import DOCS_DIR, MAX_PAIRS_IN_CHAT

#
# Main
#
logger = get_console_logger()

#
# Setup
#

# load pages and chunks
logger.info("")
logger.info("Loading documents...")

docs = load_books_and_split(DOCS_DIR)

logger.info("Loading completed...")
logger.info("")

logger.info("Build RAG chain...")
logger.info("")

rag_chain = build_chain(docs=docs)

# to limit max msgs in history
chat_history_queue = MessageQueue(MAX_PAIRS_IN_CHAT)

#
# Processing
#

QUESTIONS = [
    "Create a one-page summary of the JEP document. Organize the summary in bullet points.",
    "Does the document contains a clear list of goals? List all the goals.",
    "The document contains a well defined timeline?",
    "What Oracle has to provide and has to do, as described in the document?",
    "List any spelling or grammatical error you find in the document.",
    "How would you rate the document based on clarity, completeness, grammar?",
]

logger.info("Processing questions...")
logger.info("")


for i, question in enumerate(QUESTIONS):

    print(f"--- Question n. {i+1}: {question}")
    print("")

    input_msg = {
        "input": question,
        "chat_history": chat_history_queue.get_messages(),
    }

    response = rag_chain.invoke(input_msg)

    print(response["answer"])
    print("")
    print("")

    # save request/response in chat history
    chat_history_queue.add_message(HumanMessage(content=question))
    chat_history_queue.add_message(AIMessage(content=response["answer"]))
