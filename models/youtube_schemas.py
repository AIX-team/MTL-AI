from typing import List, Optional, Dict, Union
from enum import Enum
from pydantic import BaseModel, HttpUrl

class ContentType(str, Enum):
    YOUTUBE = "youtube"
    NAVER_BLOG = "naver_blog"
    TISTORY = "tistory"
    TEXT_FILE = "text_file"
    WEBPAGE = "webpage"
    UNKNOWN = "unknown"

class ContentRequest(BaseModel):
    urls: List[HttpUrl]

class ContentInfo(BaseModel):
    url: str
    title: str
    author: str
    platform: ContentType
    published_date: Optional[str] = None

class VideoInfo(BaseModel):
    url: str
    title: Optional[str] = None
    channel: Optional[str] = None

class PlacePhoto(BaseModel):
    url: str

class PlaceInfo(BaseModel):
    name: str
    source_url: str  # 장소 정보의 출처 URL
    description: Optional[str] = None
    formatted_address: Optional[str] = None
    rating: Optional[float] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    price_level: Optional[int] = None
    opening_hours: Optional[List[str]] = None
    photos: Optional[List[PlacePhoto]] = None
    best_review: Optional[str] = None
    google_info: Dict = {}

class YouTubeResponse(BaseModel):
    summary: Dict[str, str]
    content_infos: List[ContentInfo]
    processing_time_seconds: float
    place_details: List[PlaceInfo]

class SearchResponse(BaseModel):
    """검색 결과 응답 모델"""
    content: str
    metadata: dict
