# app/routers/extract.py
from fastapi import APIRouter, HTTPException
from typing import List
from models.schemas import ExtractRequest, ExtractResponse, VideoInfo
from services.extractor import ExtractorService

router = APIRouter()
extractor_service = ExtractorService()

@router.post("/extractyoutubeinfo", response_model=ExtractResponse)
async def extract_youtube_info(request: ExtractRequest):
    # HttpUrl 객체를 문자열로 변환
    urls = [str(url) for url in request.urls]
    print(f"Converted URLs: {urls}")
    print(f"Types of Converted URLs: {[type(url) for url in urls]}")
    
    if not (1 <= len(urls) <= 5):
        raise HTTPException(status_code=400, detail="URL의 개수는 최소 1개에서 최대 5개여야 합니다.")
    
    try:
        result = extractor_service.process_urls(urls)
        video_infos = [VideoInfo(**info) for info in result['video_infos']]
        return ExtractResponse(
            final_summary=result['final_summary'],
            video_infos=video_infos,
            processing_time_seconds=result['processing_time_seconds']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
