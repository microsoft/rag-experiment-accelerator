import numpy as np
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader


def load_pdf_files(folder_path, chunk_size, overlap_size):
    loader = PyPDFDirectoryLoader(folder_path)
    
    documents = loader.load()


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap  = overlap_size,
    )

    docs = text_splitter.split_documents(documents)

    return docs