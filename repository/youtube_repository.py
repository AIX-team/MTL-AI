import os
import datetime
from typing import List
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from models.youtube_schemas import VideoInfo, PlaceInfo

class YouTubeRepository:
    def __init__(self):
        self.summary_dir = "summaries"
        self.chunks_dir = "chunks"
        self.vector_db_path = "vector_dbs/youtube_vectordb"
        self.embedding = OpenAIEmbeddings()
        self._ensure_directories()

    def _ensure_directories(self):
        """필요한 디렉토리들이 존재하는지 확인하고 없으면 생성"""
        for directory in [self.summary_dir, self.chunks_dir, self.vector_db_path]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def save_to_vectordb(self, final_summary: str, video_infos: List[VideoInfo], place_details: List[PlaceInfo]) -> None:
        """처리된 YouTube 데이터를 벡터 DB에 저장"""
        documents = []

        # 비디오 정보 저장
        for video_info in video_infos:
            video_metadata = {
                'url': video_info.url,
                'type': 'video_info',
                'title': video_info.title,
                'channel': video_info.channel
            }
            video_metadata = {k: v for k, v in video_metadata.items() if v is not None}  # None 값 제거
            
            doc = Document(
                page_content=f"제목: {video_info.title}\n채널: {video_info.channel}\n",
                metadata=video_metadata
            )
            documents.append(doc)

        # 장소 정보 저장
        # 장소 정보 저장 (수정된 코드)
        for place in place_details:
            place_metadata = {
                'type': 'place_info',
                'name': place.name if place.name is not None else "정보 없음",
                'rating': place.rating if place.rating is not None else "정보 없음",
                'address': place.formatted_address if place.formatted_address is not None else "정보 없음",
                'photos': ", ".join([photo.url for photo in place.photos]) if place.photos else "정보 없음"
            }
            
            place_doc = Document(
                page_content=f"장소명: {place.name}\n주소: {place.formatted_address or '정보 없음'}\n",
                metadata=place_metadata
            )
            documents.append(place_doc)



        # 최종 요약 저장
        summary_metadata = {'type': 'summary'}
        summary_doc = Document(
            page_content=final_summary,
            metadata=summary_metadata
        )
        documents.append(summary_doc)

        # Chroma DB에 저장
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.vector_db_path
        )
        vectordb.persist()

        print("✅ 벡터 DB 저장 완료")







    def query_vectordb(self, query: str, k: int = 3) -> List[Document]:
        """벡터 DB에서 검색"""
        if not os.path.exists(self.vector_db_path):
            raise ValueError("❌ 벡터 DB가 존재하지 않습니다.")

        vectordb = Chroma(
            persist_directory=self.vector_db_path,
            embedding_function=self.embedding
        )
        
        results = vectordb.similarity_search(query, k=k)
        
        # 검색된 결과 확인
        for doc in results:
            if not isinstance(doc, Document):
                print(f"⚠️ 검색 결과에 잘못된 데이터 포함됨: {type(doc)} - {doc}")
        
        return [doc for doc in results if isinstance(doc, Document)]
    
    





    def save_chunks(self, chunks: List[str]) -> None:
        """텍스트 청크들을 파일로 저장"""
        for idx, chunk in enumerate(chunks, 1):
            file_path = os.path.join(self.chunks_dir, f"chunk_{idx}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chunk)

    def save_final_summary(self, final_summary: str) -> str:
        """최종 요약을 파일로 저장하고 파일 경로 반환"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.summary_dir, f"final_summary_{timestamp}.txt")
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_summary)
        
        return file_path 
