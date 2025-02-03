from typing import List, Optional
from pydantic import BaseModel, HttpUrl

class YouTubeRequest(BaseModel):
    urls: List[HttpUrl]

class VideoInfo(BaseModel):
    url: str
    title: str
    channel: str

class PlacePhoto(BaseModel):
    url: str
    title: Optional[str] = None  # 선택적 필드
    description: Optional[str] = None  # 선택적 필드

class PlaceInfo(BaseModel):
    name: str
    formatted_address: Optional[str] = None
    rating: Optional[float] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    price_level: Optional[int] = None
    opening_hours: Optional[List[str]] = None
    photos: Optional[List[PlacePhoto]] = None
    best_review: Optional[str] = None

class YouTubeResponse(BaseModel):
    final_summary: str
    video_infos: List[VideoInfo]
    processing_time_seconds: float
    place_details: Optional[List[PlaceInfo]] = None
