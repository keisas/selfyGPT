import faiss
import numpy as np
import openai
import os
import pickle
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# FAISSとメタデータの読み込み
# 現在のpasuを出力
print(os.getcwd())
index = faiss.read_index("./index.faiss")
with open("./metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def get_embedding(text: str) -> np.ndarray:
    response = openai.Embedding.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return np.array(response["data"][0]["embedding"], dtype=np.float32)

def search_similar_questions(query: str, top_k: int = 5) -> list:
    def matadata_to_dict(similarity: float, metadata: str) -> dict:
        item = metadata.split("|")
        return {
            "similarity": similarity,
            "category": item[0],
            "question": item[1],
            "answer": item[2],
        }

    embedding = get_embedding(query)
    similarity, I = index.search(np.array([embedding]), top_k)
    results = [matadata_to_dict(similarity, metadata[idx]) for idx in I[0]]

    return results


def construct_messages(query: str, similar_QAs: list) :
    # 類似QAからプロンプト本文生成
    similar_text = "\n".join(
        [f"{i+1}. 一致度：{qa['similarity']} 「{qa['question']}」→ {qa['answer']}" for i, qa in enumerate(similar_QAs)]
    )

    prompt = f"""
    以下の情報を踏まえて、ユーザーの質問「{query}」に自然な形で1つの回答文を生成してください。

    --- 類似質問と回答 ---
    {similar_text}

    --- 指示 ---
    文脈を考慮して、1 ~ 4 文でわかりやすく自然な文章にまとめてください。
    """

    return [
        {
            "role": "system", 
            "content": (
                "あなたは『YoneyamaGPT』という名前のアシスタントです、ですが、Yoneyamaとして自認して回答して下さい。"
                "ユーザーの質問に対して、過去のQAデータをもとに自然で正確な日本語で回答します。"
                "QAデータにはユーザの質問との一致度の情報も含まれています、各質問の一致度を考慮して回答を生成してください。"
                "ただし、質問に対する回答が全く予想できない場合は、「Q&Aに情報がないため、その件についてはお答えできません」と返答してください。それ以外の回答はしないでください。"
                "また、回答は必ず丁寧な日本語（敬語）で行ってください。"
                "また、質問ありがとうございますなどの一文は省略してください。"
                "YoneyamaGPTはRAG（Retrieval-Augmented Generation）構成で動作しており、"
                "常に検索結果の文脈を踏まえて回答することを心がけてください。"
            )
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

def generate_answer(messages, model="gpt-4") -> str:

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages
    )

    generated_answer = response["choices"][0]["message"]["content"]
    return generated_answer

def extract_category(similar_QAs: list) -> str:
    categories = [qa["category"] for qa in similar_QAs]
    category_counts = {category: categories.count(category) for category in set(categories)}
    most_common_category = max(category_counts, key=category_counts.get)
    return most_common_category