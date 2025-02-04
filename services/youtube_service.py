import time
import os
import requests
import datetime
import tiktoken
from math import ceil
from langdetect import detect
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from bs4 import BeautifulSoup
import openai
from googleapiclient.discovery import build
import googlemaps
from typing import List, Dict, Tuple, Any
from models.youtube_schemas import YouTubeResponse, VideoInfo, PlaceInfo, PlacePhoto
from repository.youtube_repository import YouTubeRepository
from langchain.schema import Document

from ai_api.youtube_subtitle import (
    get_video_info, process_link, split_text, summarize_text,
    extract_place_names, search_place_details, get_place_photo_google
)

# 환경 변수 및 상수 설정
load_dotenv(dotenv_path=".env")

# API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
GEOCODING_API_KEY = os.getenv("GOOGLE_GEOCODING_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# 상수 설정
MAX_URLS = 5
CHUNK_SIZE = 2048
MODEL = "gpt-4o-mini"
FINAL_SUMMARY_MAX_TOKENS = 1500

class YouTubeService:
    """메인 YouTube 서비스"""
    
    def __init__(self):
        self.repository = YouTubeRepository()
        self.subtitle_service = YouTubeSubtitleService()
        self.text_service = TextProcessingService()
        self.place_service = PlaceService()

    def process_urls(self, urls: List[str]) -> YouTubeResponse:
        try:
            start_time = time.time()
            all_text = ""
            video_infos = []
            
            # 1. URL 처리 및 텍스트 추출
            for idx, url in enumerate(urls, 1):
                print(f"\nURL {idx}/{len(urls)} 처리 중: {url}")
                video_title, channel_name = self.subtitle_service.get_video_info(url)
                if video_title and channel_name:
                    video_infos.append(VideoInfo(url=url, title=video_title, channel=channel_name))
                text = self.subtitle_service.process_link(url)
                all_text += f"\n\n--- URL {idx} 내용 ---\n{text}"

            # 2. 텍스트 분할 및 요약
            chunks = self.text_service.split_text(all_text)
            summary = self.text_service.summarize_text(chunks)
            
            # 3. 장소 추출 및 정보 수집
            place_names = self.place_service.extract_place_names(summary)
            print(f"추출된 장소: {place_names}")
            
            # 4. 장소 정보 수집
            place_details = []
            for place_name in place_names:
                try:
                    # Google Places API로 장소 정보 검색
                    place_info = self.place_service.search_place_details(place_name)
                    
                    # 장소 설명 추출 (유튜버 리뷰)
                    description = self._extract_place_description(summary, place_name)
                    
                    if not place_info:
                        # 구글 정보가 없는 경우
                        place_details.append(PlaceInfo(
                            name=place_name,
                            description=description,
                            google_info={}  # 빈 딕셔너리로 설정
                        ))
                        continue
                    
                    # 사진 URL 가져오기
                    photo_url = self.place_service.get_place_photo_google(place_name)
                    photos = [PlacePhoto(url=photo_url)] if photo_url else []
                    
                    # PlaceInfo 객체 생성
                    place_details.append(PlaceInfo(
                        name=place_name,
                        description=description,
                        formatted_address=place_info.get('formatted_address'),
                        rating=place_info.get('rating'),
                        phone=place_info.get('formatted_phone_number'),
                        website=place_info.get('website'),
                        price_level=place_info.get('price_level'),
                        opening_hours=place_info.get('opening_hours', []),
                        photos=photos,
                        best_review=place_info.get('best_review'),
                        google_info=place_info  # 전체 구글 정보를 딕셔너리로 저장
                    ))
                    print(f"장소 정보 추가 완료: {place_name}")
                except Exception as e:
                    print(f"장소 정보 처리 중 오류 발생 ({place_name}): {str(e)}")
                    # 에러가 발생한 경우
                    place_details.append(PlaceInfo(
                        name=place_name,
                        description=description,
                        google_info={}  # 빈 딕셔너리로 설정
                    ))
                    continue

            # 5. 최종 결과 생성
            processing_time = time.time() - start_time
            final_result = self._format_final_result(
                video_infos=video_infos,
                final_summary=summary,
                place_details=place_details,
                processing_time=processing_time,
                urls=urls
            )
            
            # 6. 결과 저장
            self.repository.save_final_summary(final_result)
            
            return YouTubeResponse(
                final_summary=final_result,
                video_infos=video_infos,
                processing_time_seconds=processing_time,
                place_details=place_details
            )
            
        except Exception as e:
            raise Exception(f"URL 처리 중 오류 발생: {str(e)}")

    def _extract_place_description(self, summary: str, place_name: str) -> str:
        """요약 텍스트에서 특정 장소에 대한 설명을 추출"""
        try:
            # 장소 이름을 포함하는 문장들을 찾음
            sentences = summary.split('.')
            relevant_sentences = [s.strip() for s in sentences if place_name.lower() in s.lower()]
            
            if relevant_sentences:
                return ' '.join(relevant_sentences)
            return "장소 설명을 찾을 수 없습니다."
        except Exception:
            return "장소 설명 추출 중 오류가 발생했습니다."

    def _format_final_result(self, video_infos: List[VideoInfo], final_summary: str, 
                           place_details: List[PlaceInfo], processing_time: float,
                           urls: List[str]) -> str:
        """최종 결과 문자열을 포맷팅하는 메서드"""
        
        # 1. 기본 정보 헤더
        final_result = f"""
=== 여행 정보 요약 ===
처리 시간: {processing_time:.2f}초

분석한 영상:
{'='*50}"""
        
        # 2. 비디오 정보
        if video_infos:
            for info in video_infos:
                final_result += f"""
제목: {info.title}
채널: {info.channel}
URL: {info.url}"""
        else:
            final_result += f"\nURL: {chr(10).join(urls)}"
        
        final_result += f"\n{'='*50}\n\n=== 장소별 상세 정보 ===\n"

        # 3. 장소별 정보
        for idx, place in enumerate(place_details, 1):
            final_result += f"""
{idx}. {place.name}
{'='*50}

[유튜버의 리뷰]
장소설명: {place.description or '장소 설명을 찾을 수 없습니다.'}
"""
            # 구글 장소 정보가 있는 경우에만 추가
            if place.google_info:
                final_result += f"""
[구글 장소 정보]
🏠 주소: {place.formatted_address or '정보 없음'}
⭐ 평점: {place.rating or 'None'}
📞 전화: {place.phone or 'None'}
🌐 웹사이트: {place.website or 'None'}
💰 가격대: {'₩' * place.price_level if place.price_level else '정보 없음'}
⏰ 영업시간:
{chr(10).join(place.opening_hours if place.opening_hours else ['정보 없음'])}

[사진 및 리뷰]"""
                
                if place.photos:
                    for photo_idx, photo in enumerate(place.photos, 1):
                        final_result += f"""
📸 사진 {photo_idx}: {photo.url}"""
                
                final_result += f"""
⭐ 베스트 리뷰: {place.best_review or '리뷰 없음'}"""
            
            final_result += f"\n{'='*50}"
        
        return final_result

    def search_content(self, query: str) -> List[Dict]:
        """벡터 DB에서 콘텐츠 검색"""
        try:
            results = self.repository.query_vectordb(query)
            filtered_results = []

            for doc in results:
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

class YouTubeSubtitleService:
    """YouTube 자막 및 비디오 정보 처리 서비스"""
    
    @staticmethod
    def get_video_info(video_url: str) -> Tuple[str, str]:
        try:
            video_id = video_url.split("v=")[-1].split("&")[0]
            api_url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
            response = requests.get(api_url)
            if response.status_code == 200:
                data = response.json()
                title = data.get('title')
                author_name = data.get('author_name')
                print(f"[get_video_info] 제목: {title}, 채널: {author_name}")
                return title, author_name
            print(f"[get_video_info] API 응답 상태 코드: {response.status_code}")
            return None, None
        except Exception as e:
            print(f"영상 정보를 가져오는데 실패했습니다: {e}")
            return None, None

    @staticmethod
    def process_link(url: str) -> str:
        link_type = YouTubeSubtitleService._detect_link_type(url)
        print(f"[process_link] 링크 유형 감지: {link_type}")
        
        if link_type == "youtube":
            text = YouTubeSubtitleService._get_youtube_transcript(url)
        elif link_type == "text_file":
            text = YouTubeSubtitleService._get_text_from_file(url)
        else:
            text = YouTubeSubtitleService._get_text_from_webpage(url)
        
        print(f"[process_link] 추출된 텍스트 길이: {len(text)}")
        return text

    @staticmethod
    def _detect_link_type(url: str) -> str:
        if "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        elif url.endswith(".txt"):
            return "text_file"
        elif url.startswith("http"):
            return "webpage"
        else:
            return "unknown"

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    @staticmethod
    def _get_youtube_transcript(video_url: str) -> str:
        video_id = video_url.split("v=")[-1].split("&")[0]
        print(f"[get_youtube_transcript] 비디오 ID: {video_id}")
        
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # 1. 먼저 한국어 자막 시도
            try:
                transcript = transcripts.find_transcript(['ko'])
                transcript_text = "\n".join([f"[{YouTubeSubtitleService._format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript.fetch()])
                print(f"[get_youtube_transcript] 한국어 자막 추출 완료. 길이: {len(transcript_text)}")
                return transcript_text
            except (TranscriptsDisabled, NoTranscriptFound):
                print("[get_youtube_transcript] 한국어 자막 없음.")

            # 2. 영어 자막 시도
            try:
                transcript = transcripts.find_transcript(['en'])
                transcript_text = "\n".join([f"[{YouTubeSubtitleService._format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript.fetch()])
                print(f"[get_youtube_transcript] 영어 자막 추출 완료. 길이: {len(transcript_text)}")
                return transcript_text
            except (TranscriptsDisabled, NoTranscriptFound):
                print("[get_youtube_transcript] 영어 자막 없음.")

            # 3. 사용 가능한 첫 번째 자막 시도
            try:
                transcript = transcripts.find_generated_transcript()
                transcript_text = "\n".join([f"[{YouTubeSubtitleService._format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript.fetch()])
                print(f"[get_youtube_transcript] 생성된 자막 추출 완료. 길이: {len(transcript_text)}")
                return transcript_text
            except Exception as e:
                print(f"[get_youtube_transcript] 생성된 자막 추출 실패: {e}")

            raise ValueError("사용 가능한 자막을 찾을 수 없습니다.")

        except Exception as e:
            raise ValueError(f"비디오 {video_id}의 자막을 가져오는데 실패했습니다: {e}")

    @staticmethod
    def _get_text_from_file(url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            text = response.text.strip()
            print(f"[get_text_from_file] 텍스트 파일 추출 완료. 길이: {len(text)}")
            return text
        except Exception as e:
            raise ValueError(f"텍스트 파일 내용을 가져오는데 오류가 발생했습니다: {e}")

    @staticmethod
    def _get_text_from_webpage(url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator="\n").strip()
            text = text[:10000]  # 길이 제한 10000자
            print(f"[get_text_from_webpage] 웹페이지 텍스트 추출 완료. 길이: {len(text)}")
            return text
        except Exception as e:
            raise ValueError(f"웹페이지 내용을 가져오는데 오류가 발생했습니다: {e}")

class TextProcessingService:
    """텍스트 처리 서비스"""
    
    @staticmethod
    def split_text(text: str, max_chunk_size: int = CHUNK_SIZE) -> List[str]:
        words = text.split()
        total_words = len(words)
        num_chunks = ceil(total_words / (max_chunk_size // 5))
        chunks = []
        for i in range(num_chunks):
            start = i * (max_chunk_size // 5)
            end = start + (max_chunk_size // 5)
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
        print(f"[split_text] 총 단어 수: {total_words}, 청크 수: {num_chunks}")
        return chunks

    @staticmethod
    def summarize_text(transcript_chunks: List[str], model: str = MODEL) -> str:
        summaries = []
        for idx, chunk in enumerate(transcript_chunks):
            prompt = TextProcessingService._generate_prompt(chunk)
            try:
                response = openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a travel expert who provides detailed recommendations for places to visit, foods to eat, precautions, and suggestions based on transcripts."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=1500
                )
                summary = response.choices[0].message.content
                summaries.append(summary)
                print(f"청크 {idx+1}/{len(transcript_chunks)} 요약 완료.")
                print(f"[청크 {idx+1} 요약 내용 일부]")
                print(summary[:500])
            except Exception as e:
                raise ValueError(f"요약 중 오류 발생: {e}")

        # 개별 요약을 합쳐서 최종 요약
        combined_summaries = "\n".join(summaries)
        final_prompt = f"""
아래는 여러 청크로 나뉜 요약입니다. 이 요약들을 통합하여 다음의 형식으로 최종 요약을 작성해 주세요. 반드시 아래 형식을 따르고, 빠지는 내용 없이 모든 정보를 포함해 주세요.
**요구 사항:**
1. 장소, 음식, 유의 사항, 추천 사항 등 각각의 정보를 세부적으로 작성해 주세요.
2. 만약 해당 장소에서 먹은 음식, 유의 사항, 추천 사항이 없다면 작성하지 않고 넘어가도 됩니다.
3. 방문한 장소가 없거나 유의 사항만 있을 때, 유의 사항 섹션에 모아주세요.
4. 추천 사항만 있는 것들은 추천 사항 섹션에 모아주세요.
5. 가능한 장소 이름을 알고 있다면 실제 주소를 포함해 주세요.
**결과 형식:**

결과는 아래 형식으로 작성해 주세요
아래는 예시입니다. 

방문한 장소: 스미다 타워 (주소) 타임스탬프: [HH:MM:SS]
- 장소설명: [유튜버의 설명] 도쿄 스카이트리를 대표하는 랜드마크로, 전망대에서 도쿄 시내를 한눈에 볼 수 있습니다. 유튜버가 방문했을 때는 날씨가 좋아서 후지산까지 보였고, 야경이 특히 아름다웠다고 합니다.
- 먹은 음식: 라멘 이치란
    - 설명: 진한 국물과 쫄깃한 면발로 유명한 라멘 체인점으로, 개인실에서 편안하게 식사할 수 있습니다.
- 유의 사항: 혼잡한 시간대 피하기
    - 설명: 관광지 주변은 특히 주말과 휴일에 매우 혼잡할 수 있으므로, 가능한 평일이나 이른 시간에 방문하는 것이 좋습니다.
- 추천 사항: 스카이 트리 전망대 방문
    - 설명: 도쿄의 아름다운 야경을 감상할 수 있으며, 사진 촬영 하기에 최적의 장소입니다.

방문한 장소: 유니버셜 스튜디오 일본 (주소) 타임스탬프: [HH:MM:SS]
- 장소설명: [유튜버의 설명] 유튜버가 방문했을 때는 평일임에도 사람이 많았지만, 싱글라이더를 이용해서 대기 시간을 많이 줄일 수 있었습니다. 특히 해리포터 구역의 분위기가 실제 영화의 한 장면에 들어온 것 같았고, 버터맥주도 맛있었다고 합니다.
- 유의 사항: 짧은 옷 착용 
    - 설명: 팀랩 플래닛의 일부 구역에서는 물이 높고 거울이 있으므로, 짧은 옷을 입는 것이 좋다.

**요약 청크:**
{combined_summaries}

**최종 요약:**
"""
        try:
            final_response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert summary writer who strictly adheres to the provided format."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.1,
                max_tokens=4096
            )
            final_summary = final_response.choices[0].message.content
            print("\n[최종 요약 내용 일부]")
            print(final_summary[:1000])
            return final_summary
        except Exception as e:
            raise ValueError(f"최종 요약 중 오류 발생: {e}")

    @staticmethod
    def _generate_prompt(transcript_chunk: str) -> str:
        language = detect(transcript_chunk)
        if language != 'ko':
            translation_instruction = "이 텍스트는 한국어가 아닙니다. 한국어로 번역해 주세요.\n\n"
        else:
            translation_instruction = ""

        base_prompt = f"""
{translation_instruction}
아래는 여행 유튜버가 촬영한 영상의 자막입니다. 이 자막에서 방문한 장소, 먹은 음식, 유의 사항, 추천 사항을 분석하여 정리해 주세요.

**요구 사항:**
1. 장소, 음식, 유의 사항, 추천 사항 등 각각의 정보를 세부적으로 작성해 주세요.
2. 만약 해당 장소에서 먹은 음식, 유의 사항, 추천 사항이 없다면 작성하지 않고 넘어가도 됩니다.
3. 방문한 장소가 없거나 유의 사항만 있을 때, 유의 사항 섹션에 모아주세요.
4. 추천 사항만 있는 것들은 추천 사항 섹션에 모아주세요.
5. 가능한 장소 이름을 알고 있다면 실제 주소를 포함해 주세요.
6. 장소 설명은 반드시 유튜버가 언급한 내용을 바탕으로 작성해 주세요. 유튜버의 실제 경험과 평가를 포함해야 합니다.
**결과 형식:**

결과는 아래 형식으로 작성해 주세요
아래는 예시입니다. 

방문한 장소: 스미다 타워 (주소) 타임스탬프: [HH:MM:SS]
- 장소설명: [유튜버의 설명] 도쿄 스카이트리를 대표하는 랜드마크로, 전망대에서 도쿄 시내를 한눈에 볼 수 있습니다. 유튜버가 방문했을 때는 날씨가 좋아서 후지산까지 보였고, 야경이 특히 아름다웠다고 합니다.
- 먹은 음식: 라멘 이치란
    - 설명: 진한 국물과 쫄깃한 면발로 유명한 라멘 체인점으로, 개인실에서 편안하게 식사할 수 있습니다.
- 유의 사항: 혼잡한 시간대 피하기
    - 설명: 관광지 주변은 특히 주말과 휴일에 매우 혼잡할 수 있으므로, 가능한 평일이나 이른 시간에 방문하는 것이 좋습니다.
- 추천 사항: 스카이 트리 전망대 방문
    - 설명: 도쿄의 아름다운 야경을 감상할 수 있으며, 사진 촬영 하기에 최적의 장소입니다.

방문한 장소: 유니버셜 스튜디오 일본 (주소) 타임스탬프: [HH:MM:SS]
- 장소설명: [유튜버의 설명] 유튜버가 방문했을 때는 평일임에도 사람이 많았지만, 싱글라이더를 이용해서 대기 시간을 많이 줄일 수 있었습니다. 특히 해리포터 구역의 분위기가 실제 영화의 한 장면에 들어온 것 같았고, 버터맥주도 맛있었다고 합니다.
- 유의 사항: 짧은 옷 착용 
    - 설명: 팀랩 플래닛의 일부 구역에서는 물이 높고 거울이 있으므로, 짧은 옷을 입는 것이 좋다.

**자막:**
{transcript_chunk}

위 자막을 바탕으로 위의 요구 사항에 맞는 정보를 작성해 주세요. 특히 장소 설명은 반드시 유튜버가 실제로 언급한 내용과 경험을 바탕으로 작성해 주세요.
"""
        print("\n[generate_prompt] 생성된 프롬프트 일부:")
        print(base_prompt[:500])
        return base_prompt

class PlaceService:
    """장소 정보 처리 서비스"""
    
    @staticmethod
    def extract_place_names(summary: str) -> List[str]:
        """요약 텍스트에서 장소 이름을 추출"""
        place_names = []
        lines = summary.split("\n")
        
        for line in lines:
            if "장소설명:" in line or "방문한 장소:" in line:
                try:
                    # "장소설명:" 또는 "방문한 장소:" 이후의 텍스트에서 장소 이름 추출
                    place_info = line.split(":", 1)[1].strip()
                    # 괄호가 있는 경우 괄호 앞의 텍스트만 추출
                    place_name = place_info.split("(")[0].strip()
                    if place_name and place_name not in place_names:
                        place_names.append(place_name)
                except Exception as e:
                    print(f"장소 이름 추출 중 오류 발생: {e}")
                    continue
        
        print(f"추출된 장소 목록: {place_names}")
        return place_names

    @staticmethod
    def search_place_details(place_name: str) -> Dict[str, Any]:
        """Google Places API를 사용하여 장소 정보를 검색"""
        try:
            gmaps = googlemaps.Client(key=os.getenv("GOOGLE_PLACES_API_KEY"))
            
            # 장소 검색
            places_result = gmaps.places(place_name)
            
            if not places_result['results']:
                print(f"[search_place_details] 장소를 찾을 수 없음: {place_name}")
                return None
                
            place = places_result['results'][0]
            place_id = place['place_id']
            
            # 상세 정보 검색
            details_result = gmaps.place(place_id, language='ko')
            if not details_result.get('result'):
                return None
                
            details = details_result['result']
            
            # 리뷰 정보 가져오기
            reviews = details.get('reviews', [])
            best_review = reviews[0]['text'] if reviews else None
            
            # 결과 딕셔너리 생성
            return {
                'name': details.get('name', ''),
                'formatted_address': details.get('formatted_address', ''),
                'rating': details.get('rating'),
                'formatted_phone_number': details.get('formatted_phone_number', ''),
                'website': details.get('website', ''),
                'price_level': details.get('price_level'),
                'opening_hours': details.get('opening_hours', {}).get('weekday_text', []),
                'photos': details.get('photos', []),
                'best_review': best_review
            }
            
        except Exception as e:
            print(f"[search_place_details] 장소 정보 검색 중 오류 발생 ({place_name}): {str(e)}")
            return None

    @staticmethod
    def get_place_photo_google(place_name: str) -> str:
        """Google Places API를 사용하여 장소 사진 URL을 가져옴"""
        try:
            gmaps = googlemaps.Client(key=os.getenv("GOOGLE_PLACES_API_KEY"))
            places_result = gmaps.places(place_name)
            
            if not places_result['results']:
                print(f"[get_place_photo_google] 사진을 찾을 수 없음: {place_name}")
                return None
                
            place = places_result['results'][0]
            if not place.get('photos'):
                return None
                
            photo_reference = place['photos'][0]['photo_reference']
            photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={os.getenv('GOOGLE_PLACES_API_KEY')}"
            
            print(f"[get_place_photo_google] 사진 URL 생성 완료: {photo_url}")
            return photo_url
            
        except Exception as e:
            print(f"[get_place_photo_google] 사진 URL 생성 중 오류 발생: {str(e)}")
            return None
