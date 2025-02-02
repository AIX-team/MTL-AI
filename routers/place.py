from fastapi import APIRouter
from services.place_service import get_place

router = APIRouter()

@router.get("/place")
async def place(placeId: str):
    return await get_place(placeId)
