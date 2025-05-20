# selfyGPT（バックエンド）

FastAPI + FAISS による、RAGベース質問応答API。


## 構成

- **FastAPI**: REST API エンドポイント
- **sentence-transformers**: 事前ベクトル化済み
- **FAISS**: 高速ベクトル検索
- **Docker対応**: 軽量な `python:3.10-slim` ベース

## ディレクトリ構成
```
.
.
├── backend
│   ├── .dockerignore
│   ├── .env
│   ├── Dockerfile
│   ├── app
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── vector_search.py
│   │   ├── index.faiss
│   │   └── metadata.pkl
│   └── requirements.txt
├── README.md
└── .gitignore

```

## セットアップに必要なファイル

以下のファイルを準備してください：

### 1. `QA.csv`  
事前に質問と回答のデータをCSV形式で用意してください。  
このファイルをもとにベクトル検索用の `index.faiss` および `metadata.pkl` を生成します。

（生成スクリプトは別途提供予定）

### 2. `.env`  
OpenAI APIキーを含む環境変数ファイルです。以下のような内容を含めてください：

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## API 利用方法

### エンドポイント
`POST /ask`

### リクエスト形式 (JSON)

```json
{
  "question": "あなたは誰ですか？"
}
```

### レスポンス形式 (JSON)

```json
{
  "question": "あなたは誰ですか？",
  "answer": "私の名前はselfyGPTです。AIアシスタントとして、過去のQAデータをもとに最適な回答を生成します。",
  "category": "自己紹介"
}
```

### 使用例 (curl)

```bash
curl -X POST https://api.yourdomain.com/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "あなたは誰ですか？"}'
```

## Docker ビルド例

このプロジェクトは Docker を用いて簡単にデプロイできます。

### イメージのビルド

```bash
cd backend
docker build -t selfy-gpt-backend .
```

### コンテナの起動

```bash
cd backend
docker run --env-file .env -p 8000:8000 selfy-gpt-backend
```

---

## 公開サンプル

本番環境で動作しているサンプルを以下からご覧いただけます。

- Web フロントエンド: [https://yonecoding.com](https://yonecoding.com)
- API エンドポイント: [https://api.yonecoding.com/ask](https://api.yonecoding.com/ask)
