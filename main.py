from fastapi import FastAPI, Request, APIRouter
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from routers.googleMap import router as google_map_router 
from routers.place import router as place
import os
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI()
router = APIRouter()
templates = Jinja2Templates(directory="templates")

#라우터 설정
app.include_router(google_map_router)
app.include_router(place)

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:8080"], # react 서버, spirng 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# api요청
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    api_key = os.getenv("api_key") # google api키
    return templates.TemplateResponse("index.html",{"request":request, "api_key":api_key})

# 실행 환경 설정
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000,reload=True)