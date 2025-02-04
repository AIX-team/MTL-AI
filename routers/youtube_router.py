from fastapi import APIRouter, HTTPException, status
from typing import List
from models.youtube_schemas import YouTubeRequest, YouTubeResponse
from services.youtube_service import YouTubeService
from pydantic import BaseModel

router = APIRouter()
youtube_service = YouTubeService()

class SearchRequest(BaseModel):
    query: str

class SearchResponse(BaseModel):
    content: str
    metadata: dict

@router.post("/contentanalysis", 
            response_model=YouTubeResponse,
            summary="YouTube 영상 정보 추출",
            description="YouTube URL을 받아 영상의 자막을 추출하고, 내용을 요약하며, 관련된 장소 정보를 수집합니다.")
async def process_youtube(request: YouTubeRequest):
    urls = [str(url) for url in request.urls]
    
    # URL 개수 검증
    if not (1 <= len(urls) <= 5):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL의 개수는 최소 1개에서 최대 5개여야 합니다."
        )
    
    try:
        result = youtube_service.process_urls(urls)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"처리 중 오류가 발생했습니다: {str(e)}"
        )

@router.post("/vectorsearch", response_model=List[SearchResponse])
async def search_content(request: SearchRequest):
    """벡터 DB에서 콘텐츠 검색"""
    try:
        results = youtube_service.search_content(request.query)
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"검색 중 오류가 발생했습니다: {str(e)}"
        ) 