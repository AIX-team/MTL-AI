from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers.googleMap import router as google_map_router # 라우터 설정

app = FastAPI()
templates = Jinja2Templates(directory="templates")

#라우터 설정
app.include_router(google_map_router)

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

# 실행 환경 설정
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000,reload=True)