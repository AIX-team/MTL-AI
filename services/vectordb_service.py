from repository.vectordb_repository import VectorDBRepository
from models.vectordb_model import QueryRequest, QueryResponse

class VectorDBService:
    def __init__(self):
        self.repository = VectorDBRepository("vector_dbs")
    
    def process_query(self, query: QueryRequest) -> QueryResponse:
        # 쿼리 처리 및 응답 생성
        pass

    def add_chunks(self, chunks: List[ChunkData]):
        # 청크를 벡터 DB에 추가
        pass 