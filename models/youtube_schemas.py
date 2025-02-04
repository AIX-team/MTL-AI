from typing import List, Optional
from pydantic import BaseModel, HttpUrl

class YouTubeRequest(BaseModel):
    urls: List[HttpUrl]

class VideoInfo(BaseModel):
    url: str
    title: Optional[str] = None
    channel: Optional[str] = None

class PlacePhoto(BaseModel):
    url: str

class PlaceInfo(BaseModel):
    name: str
    description: Optional[str] = None
    formatted_address: Optional[str] = None
    rating: Optional[float] = None
    phone: Optional[str] = None
    website: Optional[str] = None
    price_level: Optional[int] = None
    opening_hours: Optional[List[str]] = None
    photos: Optional[List[PlacePhoto]] = None
    best_review: Optional[str] = None
    google_info: Optional[dict] = None

class YouTubeResponse(BaseModel):
    final_summary: str
    video_infos: List[VideoInfo]
    processing_time_seconds: float
    place_details: List[PlaceInfo]
