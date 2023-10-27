import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFDirectoryLoader


def load_pdf_files(folder_path, chunk_size, overlap_size):
    """
    Load PDF files from a folder and split them into chunks of text.

    Args:
        folder_path (str): The path to the folder containing the PDF files.
        chunk_size (int): The size of each text chunk in characters.
        overlap_size (int): The size of the overlap between text chunks in characters.

    Returns:
        List[Document]: A list of Document objects, each representing a chunk of text from a PDF file.
    """
    loader = PyPDFDirectoryLoader(folder_path)
    
    documents = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = overlap_size,
    )

    docs = text_splitter.split_documents(documents)

    return docs