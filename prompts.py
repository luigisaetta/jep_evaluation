"""
Prompts
"""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#
# The prompt for the condensed query on the Vector Store
#
CONTEXT_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

CONTEXT_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXT_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

#
# The prompt for the answer from the LLM
#
QA_SYSTEM_PROMPT = """You are an Oracle assistant and \
your role is to help in evaluating the JEP document provided. \
Use the following pieces of retrieved context to answer the question. \
Be clear, detailed and complete in your answers. \
If you don't know the answer, just say that you don't know. \

{context}"""

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
