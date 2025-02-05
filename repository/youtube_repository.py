import os
import datetime
from typing import List, Dict
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.vectorstores.utils import filter_complex_metadata
from models.youtube_schemas import VideoInfo, PlaceInfo, ContentInfo

class YouTubeRepository:
    def __init__(self, summary_dir: str = "./summaries"):
        """
        YouTubeRepository 초기화
        Args:
            summary_dir: 요약본을 저장할 디렉토리 경로 (기본값: "./summaries")
        """
        self.summary_dir = summary_dir
        os.makedirs(summary_dir, exist_ok=True)
        
        # 벡터 DB 초기화
        self.embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    def save_to_vectordb(self, final_summaries: Dict[str, str], content_infos: List[ContentInfo], place_details: List[PlaceInfo]) -> None:
        """벡터 DB에 최종 요약 저장"""
        try:
            documents = []
            
            for content in content_infos:
                if content.url in final_summaries:
                    summary = final_summaries[content.url]
                    metadata = {
                        "url": content.url,
                        "title": content.title,
                        "author": content.author,
                        "platform": content.platform.value,
                        "type": "summary"
                    }
                    documents.append(Document(page_content=summary, metadata=metadata))
            
            # 벡터 DB에 저장
            if documents:
                self.vectordb.add_documents(documents)
                print(f"✅ 벡터 DB 저장 완료: {len(documents)}개 문서")
            
        except Exception as e:
            print(f"벡터 DB 저장 중 오류 발생: {str(e)}")
            raise Exception(f"벡터 DB 저장 중 오류가 발생했습니다: {str(e)}")

    def query_vectordb(self, query: str, k: int = 3) -> List[Document]:
        """벡터 DB에서 검색"""
        try:
            results = self.vectordb.similarity_search(query, k=k)
            
            # 검색된 결과 확인
            for doc in results:
                if not isinstance(doc, Document):
                    print(f"⚠️ 검색 결과에 잘못된 데이터 포함됨: {type(doc)} - {doc}")
            
            return [doc for doc in results if isinstance(doc, Document)]
        except Exception as e:
            print(f"벡터 DB 검색 중 오류 발생: {str(e)}")
            return []
    
    def save_chunks(self, chunks: List[str]) -> None:
        """텍스트 청크들을 파일로 저장"""
        for idx, chunk in enumerate(chunks, 1):
            file_path = os.path.join(self.chunks_dir, f"chunk_{idx}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chunk)

    def save_final_summary(self, final_summaries: Dict[str, str], content_infos: List[ContentInfo]) -> List[str]:
        """URL별로 최종 요약을 파일로 저장하고 파일 경로 리스트 반환"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_paths = []
        
        for idx, content in enumerate(content_infos, 1):
            # URL의 플랫폼 타입과 인덱스를 파일명에 포함
            platform_type = content.platform.value
            file_name = f"final_summary_{platform_type}_{idx}_{timestamp}.txt"
            file_path = os.path.join(self.summary_dir, file_name)
            
            with open(file_path, "w", encoding="utf-8") as f:
                # 해당 URL에 대한 요약 정보만 추출하여 저장
                header = f"""=== {platform_type.upper()} 콘텐츠 요약 ===
URL: {content.url}
제목: {content.title}
작성자: {content.author}
==================================================\n"""
                f.write(header)
                
                # 해당 URL의 요약 내용만 저장
                if content.url in final_summaries:
                    summary_content = final_summaries[content.url]
                    if isinstance(summary_content, dict):
                        summary_content = str(summary_content)  # 딕셔너리인 경우 문자열로 변환
                    f.write(summary_content)
            
            saved_paths.append(file_path)
            print(f"✅ 요약본 저장 완료: {file_path}")
        
        return saved_paths 
