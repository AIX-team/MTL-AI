# app/repositories/vectordb.py
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from config import settings

class VectorDBRepository:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma(
            persist_directory=settings.VECTOR_DB_PATH,
            embedding_function=self.embeddings
        )
    
    def add_document(self, content: str, metadata: dict):
        doc = Document(
            page_content=content,
            metadata=metadata
        )
        self.vectordb.add_documents([doc])
        self.vectordb.persist()
