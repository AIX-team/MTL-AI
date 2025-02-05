from fastapi import APIRouter, HTTPException, status
from typing import List
from models.youtube_schemas import ContentRequest, YouTubeResponse
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
            summary="콘텐츠 분석",
            description="YouTube 영상, 네이버 블로그, 티스토리 등의 URL을 받아 내용을 분석하고 요약합니다.")
async def process_content(request: ContentRequest):
    urls = [str(url) for url in request.urls]
    
    # URL 개수 검증
    if not (1 <= len(urls) <= 5):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL의 개수는 최소 1개에서 최대 5개여야 합니다."
        )
    
    try:
        result = youtube_service.process_urls(urls)
        # 결과가 올바른 형식인지 확인
        if not isinstance(result["summary"], dict):
            raise ValueError("최종 요약이 딕셔너리 형식이 아닙니다.")
        
        # YouTubeResponse 모델에 맞게 결과 구조화
        return YouTubeResponse(
            summary=result["summary"],
            content_infos=result["content_infos"],
            processing_time_seconds=result["processing_time_seconds"],
            place_details=result["place_details"]
        )
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
    

