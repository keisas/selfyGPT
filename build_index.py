from openai import OpenAI
import pandas as pd
import numpy as np
import faiss
import pickle
import os
from dotenv import load_dotenv

# ----------------------------
# 環境変数読み込み
# ----------------------------
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ----------------------------
# Embedding取得関数
# ----------------------------
def get_embedding(text: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# ----------------------------
# データ読み込み
# ----------------------------
df = pd.read_csv("question_answers.csv")

embeddings = []
metadata = []

# ----------------------------
# Embedding生成
# ----------------------------
for _, row in df.iterrows():
    text = row["question"]
    embedding = get_embedding(text)

    embeddings.append(embedding)
    metadata.append({
        "category": row["category"],
        "question": row["question"],
        "answer": row["answer"]
    })

# ----------------------------
# FAISS index 作成
# ----------------------------
dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)
index.add(np.vstack(embeddings))

# ----------------------------
# 保存
# ----------------------------
faiss.write_index(index, "index.faiss")

with open("metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print("✅ Embedding / FAISS index / metadata の保存が完了しました")