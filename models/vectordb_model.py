from pydantic import BaseModel
from typing import List, Optional

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    result: str
    source_documents: List[dict]

class VectorDBDocument(BaseModel):
    content: str
    metadata: dict 