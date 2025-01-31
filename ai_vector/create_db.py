from run_vectordb import DocumentProcessor, VectorStore
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

def create_vectordb():
    # 문서 처리
    processor = DocumentProcessor('./Aufsaetze')
    documents = processor.load_documents()
    texts = processor.split_texts()
    
    # 벡터 DB 생성
    vectorstore = VectorStore('vector_dbs')
    current_date = datetime.now().strftime('%Y%m%d')
    db_name = f"vectordb_{current_date}"
    
    try:
        vectorstore.create_vectordb(texts, db_name)
        print(f"벡터 DB가 성공적으로 생성되었습니다: {db_name}")
    except Exception as e:
        print(f"벡터 DB 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    create_vectordb() 