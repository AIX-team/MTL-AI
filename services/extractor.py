# app/services/extractor.py
import os
import requests
import datetime
import time
from math import ceil
from langdetect import detect
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from bs4 import BeautifulSoup
import openai
from config import settings
from repository.vectordb_repository import VectorDBRepository
from models.schemas import PlaceDetail
import tiktoken
from typing import List

class ExtractorService:
    def __init__(self):
        openai.api_key = settings.OPENAI_API_KEY
        self.vectordb_repo = VectorDBRepository()
    
    def process_urls(self, urls: List[str]):
        # 1. URL 정리 및 video_id 추출
        for url in urls:
            video_id = self._extract_video_id(url)
            
            # 2. YouTube 자막 추출
            transcript = self._get_transcript(video_id)
            
            # 3. Vector Store에 저장
            self._store_in_vectordb(video_id, transcript)
            
            # 4. 자막 요약 (LLM 사용)
            summary = self._summarize_transcript(transcript)
            
            # 5. 결과 데이터 구성
            video_info = {
                'title': video_title,
                'url': url,
                'transcript': transcript
            }
    
    def get_video_info(self, video_url: str):
        try:
            video_id = self.extract_video_id(video_url)
            api_url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
            response = requests.get(api_url)
            if response.status_code == 200:
                data = response.json()
                return data.get('title'), data.get('author_name')
            return None, None
        except Exception as e:
            print(f"영상 정보를 가져오는데 실패했습니다: {e}")
            return None, None
    
    def extract_video_id(self, url: str) -> str:
        print(f"Extracting video ID from URL: {url} (Type: {type(url)})")
        if "youtu.be" in url:
            return url.split("youtu.be/")[-1].split("?")[0]
        elif "v=" in url:
            return url.split("v=")[-1].split("&")[0]
        else:
            raise ValueError("유효하지 않은 YouTube URL입니다.")
    
    def process_link(self, url: str) -> str:
        link_type = self.detect_link_type(url)
        
        if link_type == "youtube":
            return self.get_youtube_transcript(url)
        elif link_type == "text_file":
            return self.get_text_from_file(url)
        else:
            return self.get_text_from_webpage(url)
    
    def detect_link_type(self, url: str) -> str:
        if "youtube.com" in url or "youtu.be" in url:
            return "youtube"
        elif url.endswith(".txt"):
            return "text_file"
        elif url.startswith("http"):
            return "webpage"
        else:
            return "unknown"
    
    def get_youtube_transcript(self, video_url: str) -> str:
        video_id = self.extract_video_id(video_url)
        try:
            transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
            # 한국어 우선
            transcript = None
            for lang in ['ko', 'en']:
                try:
                    transcript = transcripts.find_transcript([lang]).fetch()
                    break
                except (TranscriptsDisabled, NoTranscriptFound):
                    continue
            if not transcript:
                # 사용 가능한 모든 자막 중 첫 번째 사용
                transcript = transcripts.find_transcript(transcripts._languages).fetch()
            
            transcript_text = "\n".join([f"[{self.format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript])
            return transcript_text
        except Exception as e:
            raise ValueError(f"비디오 {video_id}의 자막을 가져오는데 실패했습니다: {e}")
    
    def format_timestamp(self, seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def get_text_from_file(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.text.strip()
        except Exception as e:
            raise ValueError(f"텍스트 파일 내용을 가져오는데 오류가 발생했습니다: {e}")
    
    def get_text_from_webpage(self, url: str) -> str:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(separator="\n").strip()
            return text[:10000]  # 10,000자 제한
        except Exception as e:
            raise ValueError(f"웹페이지 내용을 가져오는데 오류가 발생했습니다: {e}")
    
    def split_text(self, text: str, max_chunk_size: int = None) -> list:
        max_chunk_size = max_chunk_size or settings.CHUNK_SIZE
        words = text.split()
        total_words = len(words)
        num_chunks = ceil(total_words / (max_chunk_size // 5))
        chunks = []
        for i in range(num_chunks):
            start = i * (max_chunk_size // 5)
            end = start + (max_chunk_size // 5)
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
        return chunks
    
    def save_chunks(self, chunks: list, directory: str = "chunks"):
        os.makedirs(directory, exist_ok=True)
        for idx, chunk in enumerate(chunks, 1):
            file_path = os.path.join(directory, f"chunk_{idx}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(chunk)
        print(f"{len(chunks)}개의 청크가 '{directory}' 디렉토리에 저장되었습니다.")
    
    def summarize_text(self, transcript_chunks: list) -> str:
        summaries = []
        for idx, chunk in enumerate(transcript_chunks):
            prompt = self.generate_prompt(chunk)
            try:
                response = openai.chat.completions.create(
                    model=settings.MODEL,
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
            except Exception as e:
                raise ValueError(f"요약 중 오류 발생: {e}")
        
        combined_summaries = "\n".join(summaries)
        final_prompt = f"""
        아래는 여러 청크로 나뉜 요약입니다. 이 요약들을 통합하여 다음의 형식으로 최종 요약을 작성해 주세요. 반드시 아래 형식을 따르고, 빠지는 내용 없이 모든 정보를 포함해 주세요.
        **요약 청크:**
        {combined_summaries}
        
        **최종 요약:**
        """
        try:
            final_response = openai.chat.completions.create(
                model=settings.MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert summary writer who strictly adheres to the provided format."},
                    {"role": "user", "content": final_prompt}
                ],
                temperature=0.1,
                max_tokens=4096
            )
            final_summary = final_response.choices[0].message.content
            return final_summary
        except Exception as e:
            raise ValueError(f"최종 요약 중 오류 발생: {e}")
    
    def generate_prompt(self, transcript_chunk: str) -> str:
        language = detect(transcript_chunk)
        translation_instruction = ""
        if language != 'ko':
            translation_instruction = "이 텍스트는 한국어가 아닙니다. 한국어로 번역해 주세요.\n\n"
        
        base_prompt = f"""
        {translation_instruction}
        아래는 여행 유튜버가 촬영한 영상의 자막입니다. 
        이 자막에서 방문한 장소, 먹은 음식, 유의 사항, 추천 사항을 분석하여 정리해 주세요.
        
        **요구 사항:**
        1. 장소, 음식, 유의 사항, 추천 사항 등 각각의 정보를 세부적으로 작성해 주세요.
        2. 만약 해당 장소에서 먹은 음식, 유의 사항, 추천 사항이 없다면 작성하지 않고 넘어가도 됩니다.
        3. 방문한 장소가 없거나 유의 사항만 있을 때, 유의 사항 섹션에 모아주세요.
        4. 추천 사항만 있는 것들은 추천 사항 섹션에 모아주세요.
        5. 가능한 장소 이름을 알고 있다면 실제 주소를 포함해 주세요.
        6. 장소 설명은 반드시 유튜버가 언급한 내용을 바탕으로 작성해 주세요. 유튜버의 실제 경험과 평가를 포함해야 합니다.
        
        **결과 형식:**
        아래는 예시입니다. 

        방문한 장소: 스미다 타워 (주소) 타임스탬프: [HH:MM:SS]
        - 장소설명: [유튜버의 설명] 도쿄 스카이트리를 대표하는 랜드마크로, 전망대에서 도쿄 시내를 한눈에 볼 수 있습니다. 유튜버가 방문했을 때는 날씨가 좋아서 후지산까지 보였고, 야경이 특히 아름다웠다고 합니다.
        - 먹은 음식: 라멘 이치란
            - 설명: 진한 국물과 쫄깃한 면발로 유명한 라멘 체인점으로, 개인실에서 편안하게 식사할 수 있습니다.
        - 유의 사항: 혼잡한 시간대 피하기
            - 설명: 관광지 주변은 특히 주말과 휴일에 매우 혼잡할 수 있으므로, 가능한 평일이나 이른 시간에 방문하는 것이 좋습니다.
        - 추천 사항: 스카이 트리 전망대 방문
            - 설명: 도쿄의 아름다운 야경을 감상할 수 있으며, 사진 촬영 하기에 최적의 장소입니다.

        **자막:**
        {transcript_chunk}
        
        이 자막을 바탕으로 위의 요구 사항에 맞는 정보를 작성해 주세요. 특히 장소 설명은 반드시 유튜버가 실제로 언급한 내용과 경험을 바탕으로 작성해 주세요.
        """
        return base_prompt
    
    def extract_place_names(self, summary: str) -> list:
        place_names = []
        lines = summary.split("\n")
        for line in lines:
            if line.startswith("방문한 장소:"):
                try:
                    place_info = line.replace("방문한 장소:", "").strip()
                    place_name = place_info.split("(")[0].strip()
                    if place_name and place_name not in place_names:
                        place_names.append(place_name)
                except Exception as e:
                    print(f"장소 이름 추출 중 오류 발생: {e}")
                    continue
        return place_names
    
    def search_place_details(self, place_name: str) -> PlaceDetail:
        try:
            search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
            search_params = {
                "query": place_name,
                "key": settings.GOOGLE_PLACES_API_KEY,
                "language": "ko",
                "region": "jp"
            }
            response = requests.get(search_url, params=search_params)
            data = response.json()
            if data.get('results'):
                place = data['results'][0]
                place_id = place.get('place_id')
                details = self.get_place_details(place_id)
                return details
            else:
                print(f"장소를 찾을 수 없음: {place_name}")
                return PlaceDetail()
        except Exception as e:
            print(f"장소 상세 정보 검색 중 오류 발생: {e}")
            return PlaceDetail()
    
    def get_place_details(self, place_id: str) -> PlaceDetail:
        details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        details_params = {
            "place_id": place_id,
            "fields": "name,formatted_address,formatted_phone_number,website,opening_hours,price_level,reviews,photos,editorial_summary",
            "key": settings.GOOGLE_PLACES_API_KEY,
            "language": "ko"
        }
        response = requests.get(details_url, params=details_params)
        details_data = response.json()
        result = details_data.get('result', {})
        
        best_review = None
        if 'reviews' in result:
            five_star_reviews = [r for r in result['reviews'] if r.get('rating', 0) == 5]
            if five_star_reviews:
                best_review = max(five_star_reviews, key=lambda x: len(x.get('text', '')))
            else:
                best_review = max(result['reviews'], key=lambda x: (x.get('rating', 0), len(x.get('text', ''))))
        
        photos = []
        if 'photos' in result:
            for photo in result['photos'][:5]:
                photo_url = self.get_place_photo_google(photo.get('photo_reference'))
                if photo_url:
                    photos.append({
                        'url': photo_url,
                        'title': f"{result.get('name', '사진')}",
                        'description': f"{result.get('name', '사진')}의 Google Places API를 통해 가져온 사진입니다."
                    })
        
        return PlaceDetail(
            name=result.get('name'),
            formatted_address=result.get('formatted_address'),
            rating=result.get('rating'),
            phone=result.get('formatted_phone_number'),
            website=result.get('website'),
            price_level=result.get('price_level'),
            opening_hours=result.get('opening_hours', {}).get('weekday_text'),
            photos=photos,
            best_review=best_review.get('text') if best_review else None,
            editorial_summary=result.get('editorial_summary', {}).get('overview')
        )
    
    def get_place_photo_google(self, photo_reference: str) -> str:
        if not photo_reference:
            return "사진을 찾을 수 없습니다."
        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={settings.GOOGLE_PLACES_API_KEY}"
        return photo_url
    
    def compose_final_result(self, video_infos: list, final_summary: str, place_details: list) -> str:
        processing_time = 0  # Placeholder, 필요 시 조정 가능
        final_result = f"""
=== 여행 정보 요약 ===
처리 시간: {processing_time:.2f}초

분석한 영상:
{'='*50}"""
        
        if video_infos:
            for info in video_infos:
                final_result += f"""
제목: {info['title']}
채널: {info['channel']}
URL: {info['url']}"""
        else:
            final_result += f"""
URL: {chr(10).join(info['url'] for info in video_infos)}"""
        
        final_result += f"\n{'='*50}\n"
        
        final_result += "\n=== 장소별 상세 정보 ===\n"
        for idx, place in enumerate(place_details, 1):
            final_result += f"\n{idx}. {place.name}\n{'='*50}\n"
            # 원본 스크립트에 따른 세부 정보 추가
            if place.formatted_address:
                final_result += f"주소: {place.formatted_address}\n"
            if place.rating:
                final_result += f"평점: {place.rating}\n"
            if place.phone:
                final_result += f"전화: {place.phone}\n"
            if place.website:
                final_result += f"웹사이트: {place.website}\n"
            if place.opening_hours:
                final_result += f"영업시간:\n" + "\n".join(place.opening_hours) + "\n"
            if place.photos:
                final_result += "[사진 및 리뷰]\n"
                for photo in place.photos:
                    final_result += f"📸 {photo.title}: {photo.url}\n"
                if place.best_review:
                    final_result += f"⭐ 베스트 리뷰: {place.best_review}\n"
            final_result += f"{'='*50}"
        
        return final_result
    
    def save_final_summary(self, final_summary: str):
        os.makedirs(settings.SUMMARY_DIR, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(settings.SUMMARY_DIR, f"final_summary_{timestamp}.txt")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(final_summary)
            print(f"최종 요약이 '{file_path}' 파일에 저장되었습니다.")
        except Exception as e:
            print(f"최종 요약을 저장하는데 오류가 발생했습니다: {e}")
    
    def get_place_photos(self, place_name: str, api_key: str) -> list:
        search_url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"
        search_params = {
            "input": place_name,
            "inputtype": "textquery",
            "fields": "photos,place_id",
            "key": api_key
        }
        search_response = requests.get(search_url, params=search_params)
        if search_response.status_code == 200:
            search_data = search_response.json()
            if search_data['candidates']:
                place_id = search_data['candidates'][0]['place_id']
                details_url = "https://maps.googleapis.com/maps/api/place/details/json"
                details_params = {
                    "place_id": place_id,
                    "fields": "photos",
                    "key": api_key
                }
                details_response = requests.get(details_url, params=details_params)
                if details_response.status_code == 200:
                    details_data = details_response.json()
                    if 'photos' in details_data['result']:
                        photo_reference = details_data['result']['photos'][0]['photo_reference']
                        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={api_key}"
                        return [{
                            'url': photo_url,
                            'title': f'{place_name} 사진',
                            'description': f'{place_name}의 Google Places API를 통해 가져온 사진입니다.'
                        }]
            return []
        else:
            return []
