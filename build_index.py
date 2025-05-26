import openai
import pandas as pd
import numpy as np
import faiss
import pickle
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Embeddingを取得する関数
def get_embedding(text: str) -> np.ndarray:
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(response["data"][0]["embedding"], dtype=np.float32)

# データ読み込み
df = pd.read_csv("question_answers.csv")

# 各質問に対して埋め込みを取得
embeddings = []
metadata = []

for _, row in df.iterrows():
    text = row["question"]
    embedding = get_embedding(text)
    embeddings.append(embedding)
    metadata.append(f"{row['category']}|{row['question']}|{row['answer']}")

# FAISSインデックスを作成
dimension = len(embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.vstack(embeddings))

# インデックスの保存
faiss.write_index(index, "index.faiss")

# メタデータの保存
with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Embeddingとインデックス、メタデータの保存が完了しました。")