from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from run_vectordb import VectorStore, QASystem, DocumentProcessor
from dotenv import load_dotenv
import os
import logging
from typing import Optional

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="문서 QA 시스템")

# CORS 설정 수정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메서드 허용
    allow_headers=["*"],  # 모든 헤더 허용
    expose_headers=["*"]  # 모든 응답 헤더 노출
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 전역 변수
qa_system: Optional[QASystem] = None
vectorstore: Optional[VectorStore] = None

def ensure_vectordb_exists():
    """벡터 DB가 없으면 생성"""
    global vectorstore
    try:
        if not vectorstore:
            vectorstore = VectorStore('vector_dbs')
        
        available_dbs = vectorstore.list_available_dbs()
        if not available_dbs:
            logger.info("벡터 DB 생성 시작")
            processor = DocumentProcessor('./Aufsaetze')
            documents = processor.load_documents()
            texts = processor.split_texts()
            db_name = "vectordb_initial"
            vectorstore.create_vectordb(texts, db_name)
            logger.info(f"벡터 DB 생성 완료: {db_name}")
            return db_name
        return available_dbs[0]
    except Exception as e:
        logger.error(f"벡터 DB 생성/확인 중 오류: {str(e)}")
        raise

def initialize_qa_system():
    """QA 시스템 초기화"""
    global qa_system, vectorstore
    try:
        logger.info("QA 시스템 초기화 시작")
        
        # 벡터 DB 확인/생성
        db_name = ensure_vectordb_exists()
        
        # 벡터스토어 초기화
        if not vectorstore:
            vectorstore = VectorStore('vector_dbs')
        
        # DB 로드
        vectorstore.load_vectordb(db_name)
        
        # QA 시스템 초기화
        qa_system = QASystem(vectorstore.get_retriever())
        logger.info("QA 시스템 초기화 완료")
        return True
    except Exception as e:
        logger.error(f"QA 시스템 초기화 중 오류: {str(e)}")
        return False

# 앱 시작 시 초기화
initialize_qa_system()

class Question(BaseModel):
    question: str

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.post("/query")
async def query(question: Question):
    global qa_system
    logger.info(f"받은 질문: {question.question}")
    
    if not qa_system:
        if not initialize_qa_system():
            return JSONResponse(
                content={
                    "error": "QA 시스템을 초기화할 수 없습니다.",
                    "result": "",
                    "source_documents": []
                },
                status_code=500
            )
    
    try:
        logger.info("질문 처리 시작")
        response = qa_system.process_query(question.question)
        logger.info("질문 처리 완료")
        
        # Document 객체를 직렬화 가능한 형태로 변환
        serializable_response = {
            "result": response.get("result", ""),
            "source_documents": [
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in response.get("source_documents", [])
            ]
        }
            
        return JSONResponse(content=serializable_response)
        
    except Exception as e:
        logger.error(f"질문 처리 중 오류: {str(e)}", exc_info=True)
        return JSONResponse(
            content={
                "error": str(e),
                "result": "",
                "source_documents": []
            },
            status_code=500
        )

@app.get("/status")
async def check_status():
    """시스템 상태 확인"""
    global qa_system, vectorstore
    return {
        "status": "ready" if qa_system else "not_ready",
        "vectordb_available": bool(vectorstore and vectorstore.list_available_dbs()),
        "qa_system_initialized": bool(qa_system)
    }

if __name__ == "__main__":
    import uvicorn
    # 모든 네트워크 인터페이스에서 접속 허용
    uvicorn.run(app, host="0.0.0.0", port=8000) 