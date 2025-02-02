from repository.place_repository import fetch_place
from models.place_model import Place

async def get_place(placeId):
    place_data = await fetch_place(placeId)
    place_result = place_data.get("result")
    
    if not place_result:
        return [] # 장소 데이터 없는 경우
    
    place = Place(
        name=place_result.get("name"),
        lat=place_result["geometry"]["location"]["lat"],
        lng=place_result["geometry"]["location"]["lng"],
        placeId=place_result.get("place_id"),  # place_id를 'result'에서 가져옴
        address=place_result.get("formatted_address"),  # 주소를 'formatted_address'에서 가져옴
        phone_number=place_result.get("formatted_phone_number"),  # 전화번호를 'formatted_phone_number'에서 가져옴
        rating=place_result.get("rating"),  # 평점을 'rating'에서 가져옴
        reviews=place_result.get("reviews", []),  # 리뷰는 'reviews' 리스트에서 가져옴, 없으면 빈 리스트
        types=place_result.get("types", []),  # 장소 유형은 'types'에서 가져옴
    )
       
    return [place]
