from pydantic import BaseModel
class Location(BaseModel):
    name: str
    lat: float
    lng: float