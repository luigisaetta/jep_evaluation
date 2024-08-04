"""
Prototype 01
"""

from langchain_core.messages import HumanMessage, AIMessage

from factory import build_chain
from chunk_utils import load_books_and_split
from utils import get_console_logger
from chat_history_queue import MessageQueue
from jep_questions import QUESTIONS

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

logger.info("Processing questions...")
logger.info("")

for i, question in enumerate(QUESTIONS):
    logger.info(f"Processing question: %s ...", i + 1)

    # using print so that we can redirect in a md file
    print(f"## Question n. {i+1}: {question}")
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
