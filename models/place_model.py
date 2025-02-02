from pydantic import BaseModel
from typing import List, Optional

class Place(BaseModel):
    name: str
    placeId: str
    lat:float
    lng:float
    address: Optional[str] = None  # 주소는 선택적 필드로 설정
    phone_number: Optional[str] = None  # 전화번호도 선택적 필드로 설정
    rating: Optional[float] = None  # 평점도 선택적 필드로 설정
    reviews: List[dict] = []  # 리뷰는 리스트 형태로 처리
    types: List[str] = []  # 장소 유형도 리스트 형태로 처리