# app/main.py
from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers.youtube_router import router as youtube_router
from routers.testrouters import router as test_router  # test 라우터 추가

app = FastAPI(
    title="YouTube Info Extractor API",
    description="Extracts information from YouTube or blog URLs and summarizes travel information.",
    version="1.0.0"
)

templates = Jinja2Templates(directory="templates")

# 라우터 설정
app.include_router(youtube_router, prefix="/api/v1", tags=["YouTube"])
app.include_router(test_router, prefix="/api/v1", tags=["Test"])  # test 라우터 추가

# 정적 파일 서빙
app.mount("/static", StaticFiles(directory="static"), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 실행 환경 설정
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
