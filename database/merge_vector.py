import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# 사용할 임베딩 모델 (폴더 생성할 때 사용한 것과 동일해야 함)
embedding_model = HuggingFaceEmbeddings(
    model_name="./dragonkue_model",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

# 병합 대상 폴더 목록
folders = [
    "faiss_db/merged_blog_summary_1",
    "faiss_db/merged_blog_summary_2",
    "faiss_db/merged_blog_summary_3",
    "faiss_db/merged_blog_summary_4"
]

# 첫 번째 폴더에서 벡터 DB 불러오기
merged_db = FAISS.load_local(folders[0], embedding_model, allow_dangerous_deserialization=True)

# 나머지 폴더들을 병합
for folder in folders[1:]:
    db = FAISS.load_local(folder, embedding_model, allow_dangerous_deserialization=True)
    merged_db.merge_from(db)   # 병합

# 병합된 결과 저장 (예: merged_blog_summary_final)
output_folder = "final_db"
os.makedirs(output_folder, exist_ok=True)
merged_db.save_local(output_folder)

print(f"모든 벡터 DB가 {output_folder} 폴더에 병합 완료되었습니다.")
