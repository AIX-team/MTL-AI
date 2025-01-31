# 임시로 작성한 위치 정보를 데이터 베이스에 저장하는 로직
from models.googleMap_model import Location
from typing import List

def get_locations() -> List[Location]:
    return [
        # Location(name="Location 1", lat=37.4783, lng=126.9512),
        # Location(name="Location 2", lat=34.0522, lng=-118.2437),
        # Location(name="Location 3", lat=40.7128, lng=-74.0060),
    ]