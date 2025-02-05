from fastapi import APIRouter, HTTPException
import requests
from bs4 import BeautifulSoup
from cachetools import TTLCache

router = APIRouter()

# TTL 캐시 (5분 유지)
cache = TTLCache(maxsize=10, ttl=300)

def get_blog_content(url: str) -> str:
    """네이버 블로그에서 본문을 가져오는 함수"""
    if url in cache:
        return cache[url]

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        # 첫 번째 요청으로 iframe URL 가져오기
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # iframe 찾기
        iframe = soup.find('iframe', id='mainFrame')
        if iframe and iframe.get('src'):
            # iframe URL로 실제 컨텐츠 가져오기
            real_url = f"https://blog.naver.com{iframe['src']}"
            response = requests.get(real_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
        
        # 새로운 버전 블로그
        content = soup.find('div', {'class': 'se-main-container'})
        if content:
            text = content.get_text(separator='\n').strip()
            cache[url] = text
            return text
            
        # 구버전 블로그
        content = soup.find('div', {'class': 'post-view'})
        if content:
            text = content.get_text(separator='\n').strip()
            cache[url] = text
            return text
            
        raise HTTPException(status_code=404, detail="본문을 찾을 수 없습니다.")
        
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"블로그 데이터를 가져오는 중 오류 발생: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"처리 중 오류 발생: {str(e)}")

@router.get("/blog", summary="블로그 본문 가져오기", description="네이버 블로그 URL을 입력하면 본문 내용을 반환합니다.")
def read_blog(url: str):
    return {"content": get_blog_content(url)}
