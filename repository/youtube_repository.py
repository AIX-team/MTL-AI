from typing import List, Optional, Dict
import os
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from models.youtube_model import (
    YouTubeTranscript, 
    VideoInfo, 
    VideoMetadata,
    ChunkData,
    SubtitleChunk
)
from langchain.schema import Document
from langdetect import detect

class YouTubeRepository:
    def __init__(self):
        self.api_key = os.getenv("YOUTUBE_API_KEY")
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        self.embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    
    def extract_video_id(self, url: str) -> str:
        """URL에서 YouTube 비디오 ID 추출"""
        parsed_url = urlparse(url)
        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                return parse_qs(parsed_url.query)['v'][0]
        raise ValueError("유효하지 않은 YouTube URL입니다.")
    
    def get_transcript(self, video_id: str) -> List[YouTubeTranscript]:
        """자막 가져오기"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=['ko', 'en']
            )
            return [YouTubeTranscript(**t) for t in transcript]
        except Exception as e:
            raise Exception(f"자막을 가져오는데 실패했습니다: {str(e)}")
    
    def get_video_metadata(self, video_id: str) -> VideoMetadata:
        """비디오 메타데이터 가져오기"""
        try:
            response = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            ).execute()
            
            if not response['items']:
                raise ValueError("비디오를 찾을 수 없습니다.")
            
            video_data = response['items'][0]
            snippet = video_data['snippet']
            statistics = video_data['statistics']
            
            return VideoMetadata(
                video_id=video_id,
                title=snippet['title'],
                channel_name=snippet['channelTitle'],
                publish_date=datetime.fromisoformat(snippet['publishedAt'].replace('Z', '+00:00')),
                view_count=int(statistics.get('viewCount', 0)),
                description=snippet.get('description', '')
            )
        except Exception as e:
            raise Exception(f"비디오 정보를 가져오는데 실패했습니다: {str(e)}")

    def get_video_info(self, video_id: str) -> VideoInfo:
        # YouTube API를 사용하여 비디오 정보 가져오기
        pass

    def get_subtitles(self, video_id: str, languages: List[str] = ['ko', 'en']) -> List[SubtitleChunk]:
        """자막 가져오기 및 청크로 분할"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
            
            chunks = []
            for t in transcript:
                chunk = SubtitleChunk(
                    content=t['text'],
                    start_time=t['start'],
                    end_time=t['start'] + t['duration'],
                    language=detect(t['text'])
                )
                chunks.append(chunk)
                
            return chunks
            
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            raise Exception(f"자막을 가져올 수 없습니다: {str(e)}")
            
    def save_to_vectordb(self, chunks: List[SubtitleChunk], metadata: VideoMetadata):
        """벡터 DB에 저장"""
        try:
            documents = [
                Document(
                    page_content=chunk.content,
                    metadata={
                        "video_id": metadata.video_id,
                        "title": metadata.title,
                        "start_time": chunk.start_time,
                        "end_time": chunk.end_time,
                        "language": chunk.language
                    }
                )
                for chunk in chunks
            ]
            
            vectordb = Chroma(
                persist_directory="vector_dbs",
                embedding_function=self.embeddings
            )
            
            vectordb.add_documents(documents)
            
        except Exception as e:
            raise Exception(f"벡터 DB 저장 실패: {str(e)}") 