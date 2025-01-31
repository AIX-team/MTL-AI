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
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# -------------------
# 0. 환경 변수 및 상수 설정
# -------------------

# .env 파일 로드
load_dotenv()  # 기본적으로 프로젝트 루트의 .env 파일을 찾습니다

# 환경변수 로드 확인을 위한 디버깅 출력
print("OPENAI_API_KEY2:", os.getenv("OPENAI_API_KEY2"))

# OpenAI API 키 설정
openai.api_key = os.getenv("OPENAI_API_KEY2")

# 구글 API 키 설정
GEOCODING_API_KEY = os.getenv("GOOGLE_GEOCODING_API_KEY")
GOOGLE_PLACES_API_KEY = os.getenv("GOOGLE_PLACES_API_KEY")

# 사용할 상수들
MAX_URLS = 5  # 최대 URL 개수
CHUNK_SIZE = 2048  # 각 텍스트 청크의 최대 토큰 수 (조정 가능)
MODEL = "gpt-4o-mini"  # 사용할 OpenAI 모델
FINAL_SUMMARY_MAX_TOKENS = 1500  # 최종 요약의 최대 토큰 수

# 벡터 DB 경로
MAIN_DB_PATH = os.getenv("MAIN_DB_PATH", "main_db")

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
            
            # 1-2) URL을 처리하여 텍스트를 얻습니다.
            text = process_link(url)
            all_text += f"\n\n--- URL {idx} 내용 ---\n{text}"
            print(f"URL {idx} 처리 완료.")
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
    
    for place_name in place_names:
        print(f"\n{place_name} 정보 수집 중...")
        details = {}
        
        # (6-1) Google Places API로 장소 정보를 가져옵니다.
        try:
            google_details = search_place_details(place_name)
            if google_details:
                details.update(google_details)
                
                # (6-2) 가져온 place_name으로 사진 URL도 함께 수집합니다.
                photo_url = get_place_photo_google(place_name, GOOGLE_PLACES_API_KEY)
                if photo_url and photo_url != "사진을 찾을 수 없습니다." and photo_url != "API 요청 실패.":
                    details['photos'] = [{
                        'url': photo_url,
                        'title': f'{place_name} 사진',
                        'description': f'{place_name}의 Google Places API를 통해 가져온 사진입니다.'
                    }]
        except Exception as e:
            print(f"Google Places API 오류: {e}")
        
        if details:
            place_details.append(details)
    
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
                    place_desc = line
                elif line.startswith('- 먹은 음식:'):
                    foods.append(line)
                elif line.startswith('\t- 설명:') and foods:
                    foods[-1] += f"\n{line}"
                elif line.startswith('- 유의 사항:'):
                    precautions.append(line)
                elif line.startswith('\t- 설명:') and precautions:
                    precautions[-1] += f"\n{line}"
                elif line.startswith('- 추천 사항:'):
                    recommendations.append(line)
                elif line.startswith('\t- 설명:') and recommendations:
                    recommendations[-1] += f"\n{line}"
            
            # 카테고리별로 출력
            if place_desc:
                final_result += f"{place_desc}\n"
            
            if foods:
                final_result += "\n[먹은 음식]\n"
                for food in foods:
                    final_result += f"{food}\n"
            
            if precautions:
                final_result += "\n[유의 사항]\n"
                for precaution in precautions:
                    final_result += f"{precaution}\n"
            
            if recommendations:
                final_result += "\n[추천 사항]\n"
                for recommendation in recommendations:
                    final_result += f"{recommendation}\n"
        
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

    # 벡터 DB에 저장
    try:
        print("\n벡터 DB에 저장 시도 중...")
        embeddings = OpenAIEmbeddings()
        
        # config.py에서 정의된 통합 경로 사용
        vectordb = Chroma(
            persist_directory=VECTOR_DB_PATH,  # /Users/minkyeomkim/Desktop/MTL-FE/src/ai_vector/vector_dbs/main_vectordb
            embedding_function=embeddings
        )
        
        metadata = {"url": urls[0] if urls else ""}
        doc = Document(
            page_content=final_result,
            metadata=metadata
        )
        
        vectordb.add_documents([doc])
        vectordb.persist()
        print(f"벡터 DB가 {VECTOR_DB_PATH}에 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"벡터 DB에 저장하는데 오류가 발생했습니다: {str(e)}")
        import traceback
        print(traceback.format_exc())

    return final_result


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
            return data.get('title'), data.get('author_name')
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
    
    if link_type == "youtube":
        text = get_youtube_transcript(url)
    elif link_type == "text_file":
        text = get_text_from_file(url)
    else:  # 웹페이지
        text = get_text_from_webpage(url)
    
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
    try:
        transcripts = YouTubeTranscriptApi.list_transcripts(video_id)
        # 우선 한국어 자막 시도
        if transcripts.find_transcript(['ko']):
            transcript = transcripts.find_transcript(['ko']).fetch()
            transcript_text = "\n".join([f"[{format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript])
            return transcript_text
    except (TranscriptsDisabled, NoTranscriptFound):
        pass
    except Exception as e:
        raise ValueError(f"비디오 {video_id}의 자막을 가져오는데 실패했습니다: {e}")

    try:
        # 영어 자막 시도
        if transcripts.find_transcript(['en']):
            transcript = transcripts.find_transcript(['en']).fetch()
            transcript_text = "\n".join([f"[{format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript])
            return transcript_text
    except (TranscriptsDisabled, NoTranscriptFound):
        pass
    except Exception as e:
        raise ValueError(f"비디오 {video_id}의 자막을 가져오는데 실패했습니다: {e}")

    try:
        # 기타 언어 자막 시도
        transcript = transcripts.find_transcript(transcripts._languages).fetch()
        transcript_text = "\n".join([f"[{format_timestamp(entry['start'])}] {entry['text']}" for entry in transcript])
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
    return chunks


# -------------------
# 10. process_urls -> save_chunks
# -------------------
def save_chunks(chunks, directory="chunks"):
    """텍스트 청크를 개별 파일로 저장합니다."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for idx, chunk in enumerate(chunks, 1):
        file_path = os.path.join(directory, f"chunk_{idx}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(chunk)
    print(f"{len(chunks)}개의 청크가 '{directory}' 디렉토리에 저장되었습니다.")


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

        이 자막을 바탕으로 위의 요구 사항에 맞는 정보를 작성해 주세요. 특히 장소 설명은 반드시 유튜버가 실제로 언급한 내용과 경험을 바탕으로 작성해 주세요.
        """
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
def search_place_details(place_name):
    """
    Google Places API를 사용하여 장소의 상세 정보를 검색합니다.
    - 가장 평점이 높은(혹은 5점 리뷰가 있다면 가장 긴) 리뷰를 'best_review'로 설정
    - 사진, 연락처, 영업시간, 웹사이트 등 정보를 함께 가져옵니다.
    """
    try:
        # 장소 검색 (일본 지역 우선)
        search_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        search_params = {
            "query": f"{place_name} ",
            "key": GOOGLE_PLACES_API_KEY,
            "language": "ko",
            "region": "jp"
        }
        
        response = requests.get(search_url, params=search_params)
        data = response.json()
        
        if data.get('results'):
            place = data['results'][0]
            details = {
                'name': place.get('name'),
                'formatted_address': place.get('formatted_address'),
                'rating': place.get('rating'),
                'place_id': place.get('place_id')
            }
            
            # 장소 상세 정보 가져오기
            details_url = "https://maps.googleapis.com/maps/api/place/details/json"
            details_params = {
                "place_id": details['place_id'],
                "fields": "formatted_phone_number,website,opening_hours,price_level,reviews,photos,editorial_summary,price_level,current_opening_hours",
                "key": GOOGLE_PLACES_API_KEY,
                "language": "ko",
                "reviews_sort": "rating"
            }
            
            details_response = requests.get(details_url, params=details_params)
            details_data = details_response.json()
            
            if 'result' in details_data:
                result = details_data['result']
                
                # 평점이 가장 높은 리뷰 가져오기
                best_review = None
                if 'reviews' in result:
                    reviews = result['reviews']
                    five_star_reviews = [r for r in reviews if r.get('rating', 0) == 5]
                    if five_star_reviews:
                        best_review = max(five_star_reviews, key=lambda x: len(x.get('text', '')))
                    else:
                        best_review = max(reviews, key=lambda x: (x.get('rating', 0), len(x.get('text', ''))))
                
                details.update({
                    'phone': result.get('formatted_phone_number'),
                    'website': result.get('website'),
                    'opening_hours': result.get('opening_hours', {}).get('weekday_text'),
                    'current_opening_hours': result.get('current_opening_hours', {}).get('weekday_text'),
                    'price_level': result.get('price_level'),
                    'best_review': best_review,
                    'photos': result.get('photos', [])[:5],
                    'editorial_summary': result.get('editorial_summary', {}).get('overview')
                })
            return details
        else:
            print(f"장소를 찾을 수 없음: {place_name}")
            return None
    except Exception as e:
        print(f"장소 상세 정보 검색 중 오류 발생: {e}")
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
                    return photo_url
        return "사진을 찾을 수 없습니다."
    else:
        return "API 요청 실패."


# -------------------
# 15. process_urls -> save_final_summary
# -------------------
def save_final_summary(final_summary):
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(SUMMARY_DIR, f"final_summary_{timestamp}.txt")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_summary)
        print(f"최종 요약이 '{file_path}' 파일에 저장되었습니다.")
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
    base_url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "address": place_name,
        "key": api_key
    }
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            return data['results'][0]['formatted_address']
        else:
            return "주소를 찾을 수 없습니다."
    else:
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
            
            print("\n벡터 DB 확인을 위해 run_vectordb.py를 실행합니다...")
            run_vectordb_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'ai_vector',
                'run_vectordb.py'
            )
            os.system(f"python3 {run_vectordb_path}")
        except Exception as e:
            print(f"오류: {e}")
