from pydantic import BaseModel, HttpUrl
from typing import List, Optional, Dict
from datetime import datetime

class YouTubeTranscript(BaseModel):
    text: str
    start: float
    duration: float
    
class VideoMetadata(BaseModel):
    video_id: str
    title: str
    channel_name: str
    publish_date: datetime
    view_count: Optional[int]
    description: Optional[str]
    
class VideoInfo(BaseModel):
    title: str
    channel: str
    url: HttpUrl
    metadata: Optional[Dict] = None
    
class ChunkData(BaseModel):
    content: str
    metadata: Dict
    timestamp: Optional[str]
    
class VideoProcessRequest(BaseModel):
    url: HttpUrl
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    
class VideoProcessResponse(BaseModel):
    success: bool
    chunks: Optional[List[ChunkData]]
    error: Optional[str]

class SubtitleRequest(BaseModel):
    urls: List[str]
    max_urls: Optional[int] = 5
    chunk_size: Optional[int] = 2048
    
class SubtitleChunk(BaseModel):
    content: str
    start_time: float
    end_time: float
    language: str
    
class SubtitleSummary(BaseModel):
    title: str
    channel: str
    url: str
    summary: str
    places: List[str]
    timestamps: List[Dict[str, str]]

class SubtitleResponse(BaseModel):
    success: bool
    summaries: List[SubtitleSummary]
    error: Optional[str] = None 