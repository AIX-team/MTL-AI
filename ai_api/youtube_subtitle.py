import os
import requests
import datetime
import time
import tiktoken  # 토큰 수 계산을 위해 tiktoken 사용
from math import ceil
from langdetect import detect
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from bs4 import BeautifulSoup
import openai
from googleapiclient.discovery import build
import googlemaps
from typing import Dict, Any

# -------------------
# 0. 환경 변수 및 상수 설정
# -------------------

# .env 파일 로드
load_dotenv(dotenv_path=".env")  # .env 파일 경로 확인

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"OpenAI API Key: {'설정됨' if openai.api_key else '설정되지 않음'}")  # 디버깅용

# 구글 API 키 설정
GEOCODING_API_KEY = os.getenv("GOOGLE_GEOCODING_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")
print(f"Google Geocoding API Key: {'설정됨' if GEOCODING_API_KEY else '설정되지 않음'}")  # 디버깅용
print(f"Google Places API Key: {'설정됨' if GOOGLE_PLACES_API_KEY else '설정되지 않음'}")  # 디버깅용

# 사용할 상수들
MAX_URLS = 5  # 최대 URL 개수
CHUNK_SIZE = 2048  # 각 텍스트 청크의 최대 토큰 수 (조정 가능)
MODEL = "gpt-4o-mini"  # 사용할 OpenAI 모델
FINAL_SUMMARY_MAX_TOKENS = 1500  # 최종 요약의 최대 토큰 수

# -------------------
# 1. 메인 실행 흐름
# -------------------
def process_urls(urls):
    start_time = time.time()
    all_text = ""
    video_infos = []
    
    # (1) 입력받은 URL을 순회하면서 텍스트/자막을 추출합니다.
    for idx, url in enumerate(urls, 1):
        print(f"\nURL {idx}/{len(urls)} 처리 중: {url}")
        try:
            # 1-1) 영상 정보 가져오기
            video_title, channel_name = get_video_info(url)
            if video_title and channel_name:
                video_infos.append({
                    'url': url,
                    'title': video_title,
                    'channel': channel_name
                })
                print(f"영상 정보 추출 완료: 제목='{video_title}', 채널='{channel_name}'")
            else:
                print("영상 정보 추출 실패.")
            
            # 1-2) URL을 처리하여 텍스트를 얻습니다.
            text = process_link(url)
            all_text += f"\n\n--- URL {idx} 내용 ---\n{text}"
            print(f"URL {idx} 텍스트 추출 완료.")
        except Exception as e:
            print(f"URL {idx} 처리 중 오류 발생: {e}")
    
    # (2) 모든 URL에서 텍스트를 제대로 추출하지 못했다면 에러 발생
    if not all_text.strip():
        raise ValueError("모든 URL에서 텍스트를 추출하는데 실패했습니다.")
    
    # (3) 추출된 전체 텍스트를 CHUNK_SIZE에 맞게 분할합니다.
    print("\n텍스트를 청크로 분할 중...")
    transcript_chunks = split_text(all_text)
    print(f"텍스트가 {len(transcript_chunks)}개의 청크로 분할되었습니다.")
    
    # (4) 분할된 청크를 파일로 저장(디버깅/검증 용도)
    save_chunks(transcript_chunks)
    
    # (5) 나눠진 청크들을 요약합니다.
    print("\n요약을 생성 중...")
    final_summary = summarize_text(transcript_chunks)
    
    # (6) 요약에서 방문 장소명을 추출하고, 추가 정보를 수집합니다.
    print("\n장소 상세 정보 수집 중...")
    place_details = []
    place_names = extract_place_names(final_summary)
    
    print(f"추출된 장소 이름: {place_names}")  # 디버깅 출력
    
    for place_name in place_names:
        print(f"\n{place_name} 정보 수집 중...")
        details = {}
        
        # (6-1) Google Places API로 장소 정보를 가져옵니다.
        try:
            google_details = search_place_details(place_name)
            if google_details:
                details.update(google_details)
                print(f"{place_name}의 Google Places API 정보 수집 완료.")
                
                # (6-2) 가져온 place_name으로 사진 URL도 함께 수집합니다.
                photo_url = get_place_photo_google(place_name, GOOGLE_PLACES_API_KEY)
                if photo_url and photo_url not in ["사진을 찾을 수 없습니다.", "API 요청 실패."]:
                    details['photos'] = [{
                        'url': photo_url,
                        'title': f'{place_name} 사진',
                        'description': f'{place_name}의 Google Places API를 통해 가져온 사진입니다.'
                    }]
                    print(f"{place_name}의 사진 URL 수집 완료.")
                else:
                    print(f"{place_name}의 사진 URL 수집 실패: {photo_url}")
        except Exception as e:
            print(f"Google Places API 오류: {e}")
        
        if details:
            place_details.append(details)
            print(f"{place_name}의 상세 정보가 place_details에 추가되었습니다.")
        else:
            print(f"{place_name}의 상세 정보가 수집되지 않았습니다.")
    
    # (7) 처리 시간 계산
    end_time = time.time()
    processing_time = end_time - start_time
    
    # (8) 최종 결과를 문자열 형태로 구성
    final_result = f"""
=== 여행 정보 요약 ===
처리 시간: {processing_time:.2f}초

분석한 영상:
{'='*50}"""
    
    # 8-1) 수집된 유튜브 영상 정보 출력용
    if video_infos:
        for info in video_infos:
            final_result += f"""
제목: {info['title']}
채널: {info['channel']}
URL: {info['url']}"""
    else:
        final_result += f"""
URL: {chr(10).join(urls)}"""
    
    final_result += f"\n{'='*50}\n"

    # (9) 장소별 정보 통합(유튜브 요약내용 + 구글 정보)
    places_info = {}
    for line in final_summary.split('\n'):
        # 방문한 장소 파싱
        if line.startswith('방문한 장소:'):
            place_name = line.split('(')[0].replace('방문한 장소:', '').strip()
            if place_name not in places_info:
                places_info[place_name] = {'youtuber_info': [], 'google_info': None}
            
            start_idx = final_summary.find(line)
            end_idx = final_summary.find("\n\n방문한 장소:", start_idx)
            if end_idx == -1:
                end_idx = len(final_summary)
            place_section = final_summary[start_idx:end_idx]
            
            places_info[place_name]['youtuber_info'] = place_section.split('\n')
    
    # (10) Google Places API 정보와 매칭
    for place in place_details:
        place_name = place.get('name')
        if place_name in places_info:
            places_info[place_name]['google_info'] = place
            print(f"{place_name}의 Google Places 정보가 places_info에 매칭되었습니다.")
    
    # (11) 장소별 상세 정보 문자열에 추가
    final_result += "\n=== 장소별 상세 정보 ===\n"
    
    for idx, (place_name, info) in enumerate(places_info.items(), 1):
        final_result += f"\n{idx}. {place_name}\n{'='*50}\n"
        
        # 11-1) 유튜버 정보
        if info['youtuber_info']:
            final_result += "\n[유튜버의 리뷰]\n"
            
            place_desc = ""
            foods = []
            precautions = []
            recommendations = []
            
            for line in info['youtuber_info']:
                line = line.strip()
                if not line or line.startswith('방문한 장소:'):
                    continue
                    
                if line.startswith('- 장소설명:'):
                    place_desc = line.replace('- 장소설명:', '').strip()
                elif line.startswith('- 먹은 음식:'):
                    foods.append(line.replace('- 먹은 음식:', '').strip())
                elif line.startswith('- 유의 사항:'):
                    precautions.append(line.replace('- 유의 사항:', '').strip())
                elif line.startswith('- 추천 사항:'):
                    recommendations.append(line.replace('- 추천 사항:', '').strip())
                elif line.startswith('\t- 설명:'):
                    description = line.replace('\t- 설명:', '').strip()
                    if foods and not description.startswith('- '):
                        foods[-1] += f"\n  설명: {description}"
                    elif precautions and not description.startswith('- '):
                        precautions[-1] += f"\n  설명: {description}"
                    elif recommendations and not description.startswith('- '):
                        recommendations[-1] += f"\n  설명: {description}"
            
            # 카테고리별로 출력
            if place_desc:
                final_result += f"장소설명: {place_desc}\n"
            
            if foods:
                final_result += "\n[먹은 음식]\n"
                for food in foods:
                    final_result += f"- {food}\n"
            
            if precautions:
                final_result += "\n[유의 사항]\n"
                for precaution in precautions:
                    final_result += f"- {precaution}\n"
            
            if recommendations:
                final_result += "\n[추천 사항]\n"
                for recommendation in recommendations:
                    final_result += f"- {recommendation}\n"
        
        # 11-2) 구글 정보
        if info['google_info']:
            google_info = info['google_info']
            opening_hours = google_info.get('opening_hours', ['정보 없음'])
            if not isinstance(opening_hours, list):
                opening_hours = ['정보 없음']

            final_result += f"""
[구글 장소 정보]
🏠 주소: {google_info.get('formatted_address', '정보 없음')}
⭐ 평점: {google_info.get('rating', '정보 없음')}
📞 전화: {google_info.get('phone', '정보 없음')}
🌐 웹사이트: {google_info.get('website', '정보 없음')}
💰 가격대: {'₩' * google_info.get('price_level', 0) if google_info.get('price_level') else '정보 없음'}
⏰ 영업시간:
{chr(10).join(opening_hours)}

[사진 및 리뷰]"""
                    
            if 'photos' in google_info and google_info['photos']:
                for photo_idx, photo in enumerate(google_info['photos'], 1):
                    final_result += f"""
📸 사진 {photo_idx}: {photo['url']}
⭐ 베스트 리뷰: {google_info.get('best_review', {}).get('text', '리뷰 없음') if google_info.get('best_review') else '리뷰 없음'}"""
            else:
                final_result += "\n사진을 찾을 수 없습니다."
        
        final_result += f"\n{'='*50}"
    
    # (12) 최종 결과를 파일에 저장
    save_final_summary(final_result)
    return {
        'final_summary': final_result,
        'video_infos': video_infos,
        'processing_time_seconds': processing_time
    }


# -------------------
# 2. 메인에서 호출되는 핵심 함수: process_urls
# -------------------
def get_video_info(video_url):
    """
    YouTube 영상의 기본 정보를 추출합니다. 
    영상 제목, 채널명 등을 가져오기 위해 noembed API를 사용합니다.
    """
    try:
        video_id = video_url.split("v=")[-1].split("&")[0]
        api_url = f"https://noembed.com/embed?url=https://www.youtube.com/watch?v={video_id}"
        response = requests.get(api_url)
        if response.status_code == 200:
            data = response.json()
            title = data.get('title')
            author_name = data.get('author_name')
            print(f"[get_video_info] 제목: {title}, 채널: {author_name}")  # 디버깅 출력
            return title, author_name
        print(f"[get_video_info] API 응답 상태 코드: {response.status_code}")
        return None, None
    except Exception as e:
        print(f"영상 정보를 가져오는데 실패했습니다: {e}")
        return None, None


# -------------------
# 3. 메인 -> process_urls -> process_link
# -------------------
def process_link(url):
    """
    링크 유형에 따라(유튜브, 텍스트 파일, 웹페이지) 
    적절한 방법으로 텍스트를 추출해서 반환합니다.
    """
    link_type = detect_link_type(url)
    print(f"[process_link] 링크 유형 감지: {link_type}")  # 디버깅 출력
    
    if link_type == "youtube":
        text = get_youtube_transcript(url)
    elif link_type == "text_file":
        text = get_text_from_file(url)
    else:  # 웹페이지
        text = get_text_from_webpage(url)
    
    print(f"[process_link] 추출된 텍스트 길이: {len(text)}")  # 디버깅 출력
    return text


# -------------------
# 4. process_link -> detect_link_type
# -------------------
def detect_link_type(url):
    """링크 유형 감지"""
    if "youtube.com" in url or "youtu.be" in url:
        return "youtube"
    elif url.endswith(".txt"):
        return "text_file"
    elif url.startswith("http"):
        return "webpage"
    else:
        return "unknown"


# -------------------
# 5. process_link -> get_youtube_transcript
# -------------------
def get_youtube_transcript(video_url):
    """
    YouTube 자막 추출 및 타임스탬프 포함.
    한국어 자막 우선 -> 영어 자막 -> 기타 언어 자막 순으로 시도합니다.
    """
    video_id = video_url.split("v=")[-1].split("&")[0]  # 비디오 ID 추출
    print(f"[get_youtube_transcript] 비디오 ID: {video_id}")  # 디버깅 출력
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        # 우선 한국어 자막 시도
        if transcripts.find_transcript(['ko']):
            transcript = transcripts.find_transcript(['ko']).fetch()
            transcript_text = "\n".join([f"[{format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript])
            print(f"[get_youtube_transcript] 한국어 자막 추출 완료. 길이: {len(transcript_text)}")
            return transcript_text
    except (TranscriptsDisabled, NoTranscriptFound):
        print("[get_youtube_transcript] 한국어 자막 없음.")
        pass
    except Exception as e:
        raise ValueError(f"비디오 {video_id}의 자막을 가져오는데 실패했습니다: {e}")

    try:
        # 영어 자막 시도
        if transcripts.find_transcript(['en']):
            transcript = transcripts.find_transcript(['en']).fetch()
            transcript_text = "\n".join([f"[{format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript])
            print(f"[get_youtube_transcript] 영어 자막 추출 완료. 길이: {len(transcript_text)}")
            return transcript_text
    except (TranscriptsDisabled, NoTranscriptFound):
        print("[get_youtube_transcript] 영어 자막 없음.")
        pass
    except Exception as e:
        raise ValueError(f"비디오 {video_id}의 자막을 가져오는데 실패했습니다: {e}")

    try:
        # 기타 언어 자막 시도
        transcript = transcripts.find_transcript(transcripts._languages).fetch()
        transcript_text = "\n".join([f"[{format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript])
        print(f"[get_youtube_transcript] 기타 언어 자막 추출 완료. 길이: {len(transcript_text)}")
        return transcript_text
    except Exception as e:
        raise ValueError(f"비디오 {video_id}의 자막을 가져오는데 실패했습니다: {e}")


# -------------------
# 6-1. get_youtube_transcript -> format_timestamp
# -------------------
def format_timestamp(seconds):
    """초를 HH:MM:SS 형식으로 변환"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


# -------------------
# 7. process_link -> get_text_from_file
# -------------------
def get_text_from_file(url):
    """텍스트 파일 내용 읽기"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        text = response.text.strip()
        print(f"[get_text_from_file] 텍스트 파일 추출 완료. 길이: {len(text)}")
        return text
    except Exception as e:
        raise ValueError(f"텍스트 파일 내용을 가져오는데 오류가 발생했습니다: {e}")


# -------------------
# 8. process_link -> get_text_from_webpage
# -------------------
def get_text_from_webpage(url):
    """웹페이지 텍스트 추출"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text(separator="\n").strip()
        # 길이 제한 10000자
        text = text[:10000]
        print(f"[get_text_from_webpage] 웹페이지 텍스트 추출 완료. 길이: {len(text)}")
        return text
    except Exception as e:
        raise ValueError(f"웹페이지 내용을 가져오는데 오류가 발생했습니다: {e}")


# -------------------
# 9. process_urls -> split_text
# -------------------
def split_text(text, max_chunk_size=CHUNK_SIZE):
    """
    텍스트를 최대 크기에 맞게 분할합니다.
    대략적인 단어 수 기준으로 분할.
    """
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


# -------------------
# 10. process_urls -> save_chunks
# -------------------
def save_chunks(chunks, directory="chunks"):
    """텍스트 청크를 개별 파일로 저장합니다."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[save_chunks] '{directory}' 디렉토리 생성.")
    
    for idx, chunk in enumerate(chunks, 1):
        file_path = os.path.join(directory, f"chunk_{idx}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk)
        print(f"[save_chunks] 청크 {idx} 저장: {file_path}")
    print(f"[save_chunks] {len(chunks)}개의 청크가 '{directory}' 디렉토리에 저장되었습니다.")


# -------------------
# 11. process_urls -> summarize_text
# -------------------
def summarize_text(transcript_chunks, model=MODEL):
    """
    사용자 정의 프롬프트를 사용하여 ChatGPT로 세분화된 요약 작업 수행.
    각 청크별로 요약을 받고, 최종적으로 통합 요약을 생성합니다.
    """
    summaries = []
    # (1) 각 청크를 순회하며 요약
    for idx, chunk in enumerate(transcript_chunks):
        prompt = generate_prompt(chunk)
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
            print(summary[:500])  # 첫 500자 출력
        except Exception as e:
            raise ValueError(f"요약 중 오류 발생: {e}")
    
    # (2) 개별 요약을 합쳐서 최종 요약
    combined_summaries = "\n".join(summaries)
    final_prompt = f"""
아래는 여러 청크로 나뉜 요약입니다. 이 요약들을 통합하여 다음의 형식으로 최종 요약을 작성해 주세요. 반드시 아래 형식을 따르고, 빠지는 내용 없이 모든 정보를 포함해 주세요.
**요구 사항:**
1. 장소, 음식, 유의 사항, 추천 사항 등 각각의 정보를 세부적으로 작성해 주세요.
2. 만약 해당 장소에서 먹은 음식, 유의 사항, 추천 사항이 없다면 작성하지 않고 넘어가도 됩니다.
3. 방문한 장소가 없거나 유의 사항만 있을 때, 유의 사항 섹션에 모아주세요.
4. 추천 사항만 있는 것들은 추천 사항 섹션에 모아주세요.
5. 가능한 장소 이름을 알고 있다면 실제 주소를 포함해 주세요.

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
        print(final_summary[:1000])  # 첫 1000자 출력
        return final_summary
    except Exception as e:
        raise ValueError(f"최종 요약 중 오류 발생: {e}")


# -------------------
# 11-1. summarize_text -> generate_prompt
# -------------------
def generate_prompt(transcript_chunk):
    """
    사용자 정의 프롬프트를 생성하여 OpenAI API에 전달.
    한국어가 아니면 번역 안내를 추가.
    """
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
    print(base_prompt[:500])  # 첫 500자 출력
    return base_prompt


# -------------------
# 12. process_urls -> extract_place_names
# -------------------
def extract_place_names(summary):
    """요약에서 '방문한 장소:' 라인을 찾아 장소 이름을 리스트로 반환"""
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


# -------------------
# 13. process_urls -> search_place_details
# -------------------
def search_place_details(place_name: str) -> Dict[str, Any]:
    """Google Places API를 사용하여 장소 상세 정보 검색"""
    try:
        # Places API 클라이언트 초기화
        gmaps = googlemaps.Client(key=os.getenv("GOOGLE_PLACES_API_KEY"))
        
        # 장소 검색
        places_result = gmaps.places(place_name)
        
        if places_result['results']:
            place = places_result['results'][0]
            
            # 상세 정보 구성
            details = {
                'name': place.get('name', ''),
                'formatted_address': place.get('formatted_address', ''),
                'rating': place.get('rating'),
                'phone': place.get('formatted_phone_number', ''),
                'website': place.get('website', ''),
                'price_level': place.get('price_level'),
                'opening_hours': place.get('opening_hours', {}).get('weekday_text', []),
                'photos': []
            }
            
            return details
            
    except Exception as e:
        print(f"장소 정보 검색 중 오류 발생: {e}")
        return None


# -------------------
# 14. process_urls -> get_place_photo_google
# -------------------
def get_place_photo_google(place_name, api_key):
    """
    Google Places API를 사용하여 장소 ID를 검색한 뒤,
    place_id로 사진의 photoreference를 얻고
    최종적으로 사진 URL을 반환합니다.
    """
    try:
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
            if search_data.get('candidates'):
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
                    if 'result' in details_data and 'photos' in details_data['result']:
                        photo_reference = details_data['result']['photos'][0]['photo_reference']
                        photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth=400&photoreference={photo_reference}&key={api_key}"
                        print(f"[get_place_photo_google] 사진 URL 생성 완료: {photo_url}")  # 디버깅 출력
                        return photo_url
            print(f"[get_place_photo_google] 사진을 찾을 수 없음: {place_name}")
            return "사진을 찾을 수 없습니다."
        else:
            print(f"[get_place_photo_google] API 요청 실패: 상태 코드 {search_response.status_code}")
            return "API 요청 실패."
    except Exception as e:
        print(f"[get_place_photo_google] 오류 발생: {e}")
        return "API 요청 실패."


# -------------------
# 15. process_urls -> save_final_summary
# -------------------
def save_final_summary(final_summary, file_path="final_summary.txt"):
    """최종 요약을 파일로 저장합니다."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = f"final_summary_{timestamp}.txt"
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_summary)
        print(f"최종 요약이 '{file_path}' 파일에 저장되었습니다.")
        print("\n[save_final_summary] 최종 요약 내용 일부:")
        print(final_summary[:1000])  # 첫 1000자 출력
    except Exception as e:
        print(f"최종 요약을 저장하는데 오류가 발생했습니다: {e}")


# -------------------
# (기타) 사용되지 않았으나 원본 코드에 포함된 함수들
# -------------------
def count_tokens(text, model=MODEL):
    """
    텍스트가 몇 개의 토큰으로 이루어져 있는지 계산하는 함수.
    원본 코드에 포함되어 있으나 실제로는 사용되지 않습니다.
    """
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_address_google(place_name, api_key):
    """
    Google Geocoding API를 통해 주소를 찾는 함수.
    원본 코드에 포함되어 있으나 실제로는 사용되지 않습니다.
    """
    try:
        base_url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": place_name,
            "key": api_key
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                address = data['results'][0]['formatted_address']
                print(f"[get_address_google] 주소 찾음: {address}")  # 디버깅 출력
                return address
            else:
                print(f"[get_address_google] 주소를 찾을 수 없음: {place_name}")
                return "주소를 찾을 수 없습니다."
        else:
            print(f"[get_address_google] API 요청 실패: 상태 코드 {response.status_code}")
            return "API 요청 실패."
    except Exception as e:
        print(f"[get_address_google] 오류 발생: {e}")
        return "API 요청 실패."


# 메인 실행 코드를 함수 정의 뒤로 이동
if __name__ == "__main__":
    print("최대 5개의 URL을 입력할 수 있습니다. 입력을 마치려면 빈 줄을 입력하세요.")
    input_urls = []
    for i in range(MAX_URLS):
        url = input(f"URL {i+1}: ").strip()
        if not url:
            break
        input_urls.append(url)
    
    if not input_urls:
        print("입력된 URL이 없습니다. 프로그램을 종료합니다.")
    else:
        try:
            summary = process_urls(input_urls)
            print("\n[최종 요약]")
            print(summary)
        except Exception as e:
            print(f"오류: {e}")
