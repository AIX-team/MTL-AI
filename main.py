# app/main.py
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers.googleMap import router as google_map_router  # 라우터 설정
from routers.youtube_router import router as youtube_router


app = FastAPI(
    title="YouTube Info Extractor API",
    description="Extracts information from YouTube or blog URLs and summarizes travel information.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

# 라우터 설정
app.include_router(google_map_router)
app.include_router(youtube_router, prefix="/youtube", tags=["YouTube"])

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:8080"],  # React 서버, Spring 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 실행 환경 설정
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
