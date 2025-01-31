# -*- coding: utf-8 -*-
"""
문서 분석을 위한 벡터 저장소 및 QA 시스템
"""

import os
import time
from typing import List, Dict, Any
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # 새로운 임포트 경로
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.schema import Document
from datetime import datetime
from langchain_community.document_loaders import WebBaseLoader
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import VECTOR_DB_PATH

# User Agent 설정
os.environ['USER_AGENT'] = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.81 Safari/537.36'

class YouTubeProcessor:
    """유튜브 동영상 정보 처리를 위한 클래스"""
    
    def __init__(self, url: str):
        try:
            self.url = url
            self.video_id = self._extract_video_id(url)
            self.youtube = build('youtube', 'v3', 
                developerKey='AIzaSyCjDYLdVhLXj0TKjqJjYXpADY8lOa2OwhU')
        except Exception as e:
            print(f"YouTube API 초기화 중 오류 발생: {e}")
            print("필요한 라이브러리가 설치되어 있는지 확인하세요:")
            print("pip install google-api-python-client youtube-transcript-api")
            raise
    
    def get_video_info(self) -> Dict[str, Any]:
        """비디오 정보 수집"""
        try:
            # 비디오 정보 가져오기
            video_response = self.youtube.videos().list(
                part='snippet,statistics,contentDetails',
                id=self.video_id
            ).execute()

            if not video_response['items']:
                print("비디오를 찾을 수 없습니다.")
                return {}

            video_data = video_response['items'][0]
            snippet = video_data['snippet']
            statistics = video_data['statistics']

            info = {
                'title': snippet['title'],
                'author': snippet['channelTitle'],
                'channel_url': f"https://www.youtube.com/channel/{snippet['channelId']}",
                'description': snippet['description'],
                'thumbnail_url': snippet.get('thumbnails', {}).get('high', {}).get('url', ''),
                'publish_date': snippet['publishedAt'],
                'views': statistics.get('viewCount', 0),
                'url': self.url
            }

            # 자막 가져오기
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    self.video_id,
                    languages=['ko', 'en']
                )
                info['transcript'] = ' '.join([entry['text'] for entry in transcript])
            except Exception as e:
                print(f"자막 로딩 중 오류 발생: {e}")
                info['transcript'] = ''

            # 댓글 가져오기 (선택사항)
            try:
                comments_response = self.youtube.commentThreads().list(
                    part='snippet',
                    videoId=self.video_id,
                    maxResults=100,
                    order='relevance'
                ).execute()

                comments = []
                for item in comments_response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comments.append({
                        'author': comment['authorDisplayName'],
                        'text': comment['textDisplay'],
                        'likes': comment['likeCount'],
                        'published_at': comment['publishedAt']
                    })
                
                info['comments'] = comments
            except Exception as e:
                print(f"댓글 로딩 중 오류 발생: {e}")
                info['comments'] = []

            return info

        except HttpError as e:
            print(f"YouTube API 오류: {e}")
            return {}
        except Exception as e:
            print(f"비디오 정보 로딩 중 오류 발생: {e}")
            return {}

    def _extract_video_id(self, url: str) -> str:
        """URL에서 유튜브 비디오 ID 추출"""
        query = urlparse(url)
        if query.hostname == 'youtu.be':
            return query.path[1:]
        if query.hostname in ('www.youtube.com', 'youtube.com'):
            if query.path == '/watch':
                return parse_qs(query.query)['v'][0]
            if query.path[:7] == '/embed/':
                return query.path.split('/')[2]
            if query.path[:3] == '/v/':
                return query.path.split('/')[2]
        raise ValueError('유효하지 않은 YouTube URL입니다.')

class DocumentProcessor:
    """문서 처리를 위한 클래스
    
    계층 구조:
    1. 문서 로딩
    2. 텍스트 분할
    """
    def __init__(self, directory_path: str = None, url: str = None):
        """
        Args:
            directory_path (str, optional): 문서가 저장된 디렉토리 경로
            url (str, optional): 처리할 URL
        """
        self.directory_path = directory_path
        self.url = url
        self.documents = []
        self.texts = []
        
        if directory_path:
            os.makedirs(directory_path, exist_ok=True)
    
    def load_url(self) -> List[Document]:
        """URL에서 문서 로드"""
        try:
            loader = WebBaseLoader(self.url)
            documents = loader.load()
            
            # 수집된 정보 출력
            print("\n=== 수집된 URL 정보 ===")
            print(f"URL: {self.url}")
            print(f"콘텐츠 길이: {len(documents[0].page_content)} 문자")
            print("메타데이터:", documents[0].metadata)
            print("\n처음 500자:", documents[0].page_content[:500])
            
            self.documents = documents
            return self.documents
        except Exception as e:
            print(f"URL 로딩 중 오류 발생: {e}")
            return []

    def load_youtube(self) -> List[Document]:
        """YouTube URL에서 문서 로드"""
        try:
            processor = YouTubeProcessor(self.url)
            info = processor.get_video_info()
            
            if not info:
                return []
            
            # 메타데이터 구성
            metadata = {
                'source': self.url,
                'title': info.get('title', ''),
                'author': info.get('author', ''),
                'publish_date': info.get('publish_date', ''),
                'views': info.get('views', 0),
                'thumbnail_url': info.get('thumbnail_url', '')
            }
            
            # 본문 구성
            content_parts = [
                f"제목: {info.get('title', '')}",
                f"작성자: {info.get('author', '')}",
                f"채널 URL: {info.get('channel_url', '')}",
                f"설명:\n{info.get('description', '')}",
                f"자막:\n{info.get('transcript', '')}"
            ]
            
            content = '\n\n'.join(content_parts)
            
            # Document 객체 생성
            self.documents = [Document(
                page_content=content,
                metadata=metadata
            )]
            
            return self.documents
            
        except Exception as e:
            print(f"YouTube 처리 중 오류 발생: {e}")
            return []

    def load_documents(self) -> List[Document]:
        """문서 로드 (파일, URL, 또는 YouTube)"""
        if self.url:
            if 'youtube.com' in self.url or 'youtu.be' in self.url:
                return self.load_youtube()
            return self.load_url()
        
        if self.directory_path:
            loader = DirectoryLoader(
                self.directory_path, 
                glob="*.txt", 
                loader_cls=lambda x: TextLoader(x, encoding='utf-8')
            )
            self.documents = loader.load()
            return self.documents
        
        return []
    
    def split_texts(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """문서를 작은 텍스트 청크로 분할
        
        Args:
            chunk_size (int): 각 청크의 크기
            chunk_overlap (int): 청크 간 중복되는 문자 수
            
        Returns:
            List[Document]: 분할된 텍스트 청크 리스트
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        self.texts = text_splitter.split_documents(self.documents)
        return self.texts

class VectorStore:
    def __init__(self):
        self.main_db_path = VECTOR_DB_PATH
        os.makedirs(os.path.dirname(self.main_db_path), exist_ok=True)
        self.embedding = OpenAIEmbeddings()
        self.vectordb = None

    def initialize_main_db(self, texts: List[Document]) -> None:
        """메인 벡터 DB 초기화 (관리자 전용)"""
        if os.path.exists(self.main_db_path):
            raise ValueError("메인 DB가 이미 존재합니다. 기존 DB를 사용하세요.")
        
        # 메타데이터에 URL 소스 표시
        for text in texts:
            text.metadata['source_type'] = 'url'
        
        self.vectordb = Chroma.from_documents(
            documents=texts,
            embedding=self.embedding,
            persist_directory=self.main_db_path
        )
        self.vectordb.persist()
        print("메인 DB가 초기화되었습니다.")

    def add_to_main_db(self, texts: List[Document]) -> None:
        """메인 DB에 새로운 텍스트 추가"""
        if not os.path.exists(self.main_db_path):
            raise ValueError("메인 DB가 존재하지 않습니다. 관리자에게 문의하세요.")
        
        # 메타데이터에 URL 소스 표시
        for text in texts:
            text.metadata['source_type'] = 'url'
        
        self.vectordb = Chroma(
            persist_directory=self.main_db_path,
            embedding_function=self.embedding
        )
        self.vectordb.add_documents(texts)
        self.vectordb.persist()
        print(f"{len(texts)}개의 새로운 문서가 추가되었습니다.")

    def get_retriever(self, k: int = 3):
        """검색기 생성"""
        if not os.path.exists(self.main_db_path):
            raise ValueError("메인 DB가 존재하지 않습니다.")
            
        if self.vectordb is None:
            self.vectordb = Chroma(
                persist_directory=self.main_db_path,
                embedding_function=self.embedding
            )
        return self.vectordb.as_retriever(search_kwargs={"k": k})

    def delete_main_db(self) -> None:
        """메인 DB 삭제"""
        import shutil
        if os.path.exists(self.main_db_path):
            shutil.rmtree(self.main_db_path)
            print("기존 메인 DB가 삭제되었습니다.")
        else:
            print("삭제할 메인 DB가 없습니다.")

    def search_url_info(self, url):
        try:
            # 메타데이터 기반 검색 추가
            results = self.vectordb.get(
                where={"url": url}
            )
            if results and len(results['documents']) > 0:
                return results['documents'][0]
            return None
        except Exception as e:
            print(f"검색 중 오류 발생: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return None

class QASystem:
    """질의응답 시스템 클래스
    
    계층 구조:
    1. 질의 처리
    2. 응답 포맷팅
    """
    def __init__(self, retriever):
        """
        Args:
            retriever: 문서 검색기 객체
        """
        self.retriever = retriever
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """질의 처리 및 응답 반환"""
        response = self.qa_chain.invoke(query)
        
        # 참고 문서의 내용이 실제로 질문과 관련이 있는지 확인
        if not response["source_documents"] or not self._is_relevant_response(query, response["result"]):
            return {
                "result": "죄송합니다. 주어진 문서에서 관련 정보를 찾을 수 없습니다.",
                "source_documents": []
            }
        return response
    
    def _is_relevant_response(self, query: str, result: str) -> bool:
        """응답이 실제로 문서 내용에 기반하는지 확인"""
        irrelevant_responses = [
            "I am an AI",
            "I am a computer program",
            "I am an assistant",
            "I'm an AI",
            "I'm a computer program",
            "I'm an assistant"
        ]
        
        # 자기 소개하는 응답인 경우 관련 없는 것으로 판단
        for resp in irrelevant_responses:
            if resp.lower() in result.lower():
                return False
        return True
    
    @staticmethod
    def format_response(response: Dict[str, Any]) -> None:
        """응답 포맷팅 및 출력
        
        Args:
            response (Dict[str, Any]): QA 시스템의 응답
        """
        # 참고 문서가 있는 경우에만 답변 표시
        if response["source_documents"]:
            print("\n답변:")
            print(response['result'])
            print('\n참고 문서:')
            for source in response["source_documents"]:
                print(source.metadata['source'])
        else:
            print("\n죄송합니다. 주어진 문서에서 관련 정보를 찾을 수 없습니다.")

def query_mode():
    """메인 벡터 DB를 사용하여 질의 수행하는 모드"""
    # API 키 설정
    os.environ["OPENAI_API_KEY"] = "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"
    
    # 벡터 DB 초기화
    vectorstore = VectorStore()
    
    try:
        # QA 시스템 초기화
        retriever = vectorstore.get_retriever()
        qa_system = QASystem(retriever)
        
        # 대화형 질의 응답 시작
        print("\n=== 질문을 입력하세요 (종료하려면 'q' 또는 'quit' 입력) ===")
        while True:
            query = input("\n질문: ").strip()
            if query.lower() in ['q', 'quit', '종료']:
                break
            if not query:
                continue
                
            try:
                response = qa_system.process_query(query)
                qa_system.format_response(response)
            except Exception as e:
                print(f"오류 발생: {e}")
                
    except ValueError as e:
        print(f"\n오류: {e}")
        print("메인 DB가 존재하지 않습니다. 먼저 관리자 모드에서 DB를 초기화하세요.")

def admin_mode():
    """관리자 모드: 메인 DB 초기화"""
    print("\n=== 관리자 모드: 메인 DB 초기화 ===")
    
    # API 키 설정
    os.environ["OPENAI_API_KEY"] = "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"
    
    # 벡터 DB 초기화
    vectorstore = VectorStore()
    
    # 기존 DB 확인 및 삭제
    if os.path.exists(vectorstore.main_db_path):
        print("\n⚠️ 경고: 기존 메인 DB가 존재합니다!")
        confirm = input("기존 DB를 삭제하고 새로 만드시겠습니까? (y/n): ").lower()
        if confirm == 'y':
            vectorstore.delete_main_db()
            # 빈 벡터 DB 생성
            embeddings = OpenAIEmbeddings()
            Chroma(persist_directory=vectorstore.main_db_path, embedding_function=embeddings)
            print("\n새로운 메인 DB가 생성되었습니다.")
        else:
            print("\n작업이 취소되었습니다.")
            return
    else:
        # 빈 벡터 DB 생성
        embeddings = OpenAIEmbeddings()
        Chroma(persist_directory=vectorstore.main_db_path, embedding_function=embeddings)
        print("\n새로운 메인 DB가 생성되었습니다.")


def inspect_url_mode():
    """URL 정보 확인 모드: youtube_subtitle.py를 통해 벡터 DB에 저장된 정보 조회"""
    print("\n=== URL 정보 확인 모드 ===")
    
    # API 키 설정
    os.environ["OPENAI_API_KEY"] = "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"
    
    try:
        # Chroma DB 직접 연결
        embeddings = OpenAIEmbeddings()
        vectordb = Chroma(
            persist_directory="/Users/minkyeomkim/Desktop/MTL-FE/src/ai_vector/vector_dbs/main_vectordb",
            embedding_function=embeddings
        )
        
        # URL 입력 받기
        url = input("\n확인할 URL을 입력하세요: ").strip()
        if not url:
            print("URL이 입력되지 않았습니다.")
            return
            
        print("\n저장된 정보 검색 중...")
        
        # URL 관련 정보 검색
        results = vectordb.similarity_search(
            f"URL: {url}",
            k=5  # 관련된 여러 문서 검색
        )
        
        if results:
            print("\n=== 저장된 정보 ===")
            found = False
            for doc in results:
                content = doc.page_content
                if "=== 여행 정보 요약 ===" in content:
                    print(content)
                    found = True
                    break
            if not found:
                print("\n이 URL의 상세 분석 정보를 찾을 수 없습니다.")
                print("먼저 youtube_subtitle.py를 통해 URL을 처리하고 저장해주세요.")
        else:
            print("\n이 URL에 대한 정보가 벡터 DB에 저장되어 있지 않습니다.")
            print("먼저 youtube_subtitle.py를 통해 URL을 처리하고 저장해주세요.")
        
    except Exception as e:
        print(f"\n오류 발생: {e}")
        print("벡터 DB가 초기화되지 않았거나 접근할 수 없습니다.")

def main():
    """메인 실행 함수"""
    print("\n=== URL 기반 벡터 DB 시스템 ===")
    print("1. 메인 DB 초기화 (관리자 전용)")
    print("2. 질의하기")
    print("3. URL 정보 확인")  # 새로운 옵션 추가
    
    while True:
        try:
            choice = int(input("\n실행 모드를 선택하세요 (1-3): "))  # 선택 범위 수정
            if choice in [1, 2, 3]:  # 선택 가능한 옵션 수정
                if choice == 1:
                    print("\n⚠️ 경고: 관리자 전용 기능입니다!")
                    print("이 기능은 기존 DB를 초기화합니다!")
                    confirm = input("\n정말로 진행하시겠습니까? (y/n): ").lower()
                    if confirm != 'y':
                        print("\n작업이 취소되었습니다.")
                        continue
                break
            print("1-3 사이의 숫자를 입력하세요.")  # 안내 메시지 수정
        except ValueError:
            print("숫자를 입력하세요.")
    
    if choice == 1:
        admin_mode()
    elif choice == 2:
        query_mode()
    else:
        inspect_url_mode()

if __name__ == "__main__":
    main()
