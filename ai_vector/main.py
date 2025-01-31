from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import os
from run_vectordb import VectorStore, DocumentProcessor, QASystem

app = FastAPI()

# 템플릿 설정
templates = Jinja2Templates(directory="templates")

# API 키 설정
os.environ["OPENAI_API_KEY"] = "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"

# 벡터 DB 초기화
vectorstore = VectorStore('vector_dbs')

class URLRequest(BaseModel):
    url: str

class QueryRequest(BaseModel):
    query: str

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/add-url")
async def add_url(request: URLRequest):
    """URL을 벡터 DB에 추가"""
    try:
        # 문서 처리
        processor = DocumentProcessor(url=request.url)
        documents = processor.load_documents()
        if not documents:
            return {"success": False, "message": "URL에서 문서를 로드할 수 없습니다."}
        
        texts = processor.split_texts()
        
        # 벡터 DB에 추가
        vectorstore.add_to_main_db(texts)
        return {
            "success": True, 
            "message": "URL이 성공적으로 추가되었습니다."
        }
            
    except Exception as e:
        return {"success": False, "message": f"오류 발생: {str(e)}"}

@app.post("/query")
async def query(request: QueryRequest):
    """벡터 DB에 질의"""
    try:
        retriever = vectorstore.get_retriever()
        qa_system = QASystem(retriever)
        response = qa_system.process_query(request.query)
        
        # 응답 형식화
        result = {
            "answer": response["result"],
            "sources": [doc.metadata['source'] for doc in response["source_documents"]]
        }
        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "message": f"오류 발생: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 