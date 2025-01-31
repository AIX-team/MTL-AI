from run_vectordb import VectorStore
import os

def check_vector_db():
    vectorstore = VectorStore('vector_dbs')
    available_dbs = vectorstore.list_available_dbs()
    
    if not available_dbs:
        print("사용 가능한 벡터 DB가 없습니다!")
        return False
        
    print("\n=== 사용 가능한 벡터 DB ===")
    for idx, db in enumerate(available_dbs, 1):
        db_path = os.path.join('vector_dbs', db)
        print(f"{idx}. {db}")
        if os.path.exists(os.path.join(db_path, 'chroma.sqlite3')):
            print(f"   - SQLite DB 존재함")
        else:
            print(f"   - SQLite DB 없음")
    return True

if __name__ == "__main__":
    check_vector_db() 