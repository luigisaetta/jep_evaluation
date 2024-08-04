"""
Load and Chunk utils
"""

from tqdm import tqdm
from glob import glob

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import get_console_logger

from config import CHUNK_SIZE, CHUNK_OVERLAP


def get_recursive_text_splitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    return a recursive text splitter
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter


def load_books_and_split(books_dir) -> list:
    """
    load a set of books from books_dir and split in chunks
    """
    logger = get_console_logger()

    logger.info("Loading documents from %s...", books_dir)

    text_splitter = get_recursive_text_splitter()

    books_list = sorted(glob(books_dir + "/*.pdf"))

    logger.info("Loading books: ")
    for book in books_list:
        logger.info("* %s", book)

    docs = []

    for book in tqdm(books_list):
        loader = PyPDFLoader(file_path=book)

        docs += loader.load_and_split(text_splitter=text_splitter)

    logger.info("Loaded %s chunks of text...", len(docs))

    return docs
