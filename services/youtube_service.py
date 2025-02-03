import time
import os
from typing import List, Dict
from models.youtube_schemas import YouTubeResponse, VideoInfo, PlaceInfo, PlacePhoto
from repository.youtube_repository import YouTubeRepository
from langchain.schema import Document

from ai_api.youtube_subtitle import (
    get_video_info, process_link, split_text, summarize_text,
    extract_place_names, search_place_details, get_place_photo_google
)

class YouTubeService:
    def __init__(self):
        self.repository = YouTubeRepository()

    def process_urls(self, urls: List[str]) -> YouTubeResponse:
        try:
            start_time = time.time()
            all_text = ""
            video_infos = []
            
            # 1. URL 처리 및 텍스트 추출
            for idx, url in enumerate(urls, 1):
                print(f"\nURL {idx}/{len(urls)} 처리 중: {url}")
                
                # 1-1. 영상 정보 추출
                video_title, channel_name = get_video_info(url)
                if video_title and channel_name:
                    video_infos.append(VideoInfo(url=url, title=video_title, channel=channel_name))
                
                # 1-2. 자막 및 텍스트 추출
                text = process_link(url)
                all_text += f"\n\n--- URL {idx} 내용 ---\n{text}"

            # 2. 텍스트 분할 및 저장
            transcript_chunks = split_text(all_text)
            self.repository.save_chunks(transcript_chunks)
            
            # 3. 요약 생성
            final_summary = summarize_text(transcript_chunks)
            
            # 4. 장소 정보 수집
            place_details = []
            place_names = extract_place_names(final_summary)
            
            for place_name in place_names:
                details = search_place_details(place_name)
                if details:
                    print(f"장소 정보 검색 완료: {place_name}")
                    
                    # 사진 URL 수집
                    photo_url = get_place_photo_google(place_name, os.getenv("GOOGLE_PLACES_API_KEY", "dummy_key"))
                    if photo_url and photo_url not in ["사진을 찾을 수 없습니다.", "API 요청 실패."]:
                        if 'photos' not in details:
                            details['photos'] = []
                        details['photos'].append({
                            'url': photo_url,
                            'title': f'{place_name} 사진',
                            'description': f'{place_name}의 Google Places API를 통해 가져온 사진입니다.'
                        })
                    
                    # photos 필드 변환
                    photos = [PlacePhoto(**photo) for photo in details.get("photos", [])]
                    
                    # PlaceInfo 객체 생성
                    place_info = PlaceInfo(
                        name=details.get("name", ""),
                        formatted_address=details.get("formatted_address"),
                        rating=details.get("rating"),
                        phone=details.get("phone"),
                        website=details.get("website"),
                        price_level=details.get("price_level"),
                        opening_hours=details.get("opening_hours"),
                        photos=photos,
                        best_review=details.get("best_review"),
                    )
                    
                    place_details.append(place_info)

            # 5. 벡터 DB에 저장
            self.repository.save_to_vectordb(final_summary, video_infos, place_details)
            
            # 6. 최종 요약 파일 저장
            self.repository.save_final_summary(final_summary)
            
            # 7. 처리 시간 계산
            processing_time = time.time() - start_time
            
            return YouTubeResponse(
                final_summary=final_summary,
                video_infos=video_infos,
                processing_time_seconds=processing_time,
                place_details=place_details
            )
            
        except Exception as e:
            raise Exception(f"URL 처리 중 오류 발생: {str(e)}")

    def search_content(self, query: str) -> List[Dict]:
        """벡터 DB에서 콘텐츠 검색"""
        try:
            results = self.repository.query_vectordb(query)
            filtered_results = []

            for doc in results:
                # doc이 Document 객체인지 확인 후 필터링
                if isinstance(doc, Document) and hasattr(doc, "metadata") and hasattr(doc, "page_content"):
                    filtered_results.append({
                        'content': doc.page_content,
                        'metadata': doc.metadata
                    })
                else:
                    print(f"⚠️ 잘못된 데이터 타입 감지: {type(doc)} - {doc}")

            return filtered_results
        except Exception as e:
            raise Exception(f"검색 중 오류 발생: {str(e)}")




