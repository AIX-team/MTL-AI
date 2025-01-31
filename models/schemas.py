# app/models/schemas.py
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict
from datetime import datetime

class ExtractRequest(BaseModel):
    urls: List[HttpUrl] = Field(
        ..., 
        min_items=1, 
        max_items=5, 
        description="List of YouTube or blog URLs (1-5)"
    )

class VideoInfo(BaseModel):
    url: HttpUrl
    title: Optional[str]
    channel: Optional[str]

class Photo(BaseModel):
    url: HttpUrl
    title: str
    description: str

class PlaceDetail(BaseModel):
    name: Optional[str]
    formatted_address: Optional[str]
    rating: Optional[float]
    phone: Optional[str]
    website: Optional[str]
    price_level: Optional[int]
    opening_hours: Optional[List[str]]
    photos: Optional[List[Photo]]
    best_review: Optional[str]
    editorial_summary: Optional[str]

class ExtractResponse(BaseModel):
    final_summary: str
    video_infos: List[VideoInfo]
    processing_time_seconds: float
