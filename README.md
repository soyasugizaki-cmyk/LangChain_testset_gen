# RAG POC

Azure OpenAIを使用したRAG（Retrieval-Augmented Generation）の実証実験プロジェクトです。

## セットアップ

### 1. 依存関係のインストール

```bash
uv sync
```

### 2. 環境変数の設定

プロジェクトルートに `.env` ファイルを作成し、以下の環境変数を設定してください：

```env
# Azure OpenAI Settings
AZURE_OPENAI_API_KEY=your_azure_openai_api_key
# 重要: エンドポイントはベースURLのみを指定（デプロイメント名やパスは含めない）
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview

# Azure OpenAI Deployment Names（Azure Portalで作成したデプロイメント名）
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o
AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002

# LangSmith (オプション - 設定しない場合はLangSmithへの保存と評価がスキップされます)
# LANGSMITH_API_KEY=your_langsmith_api_key
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=rag-poc

# ChromaDB テレメトリー無効化（オプション）
ANONYMIZED_TELEMETRY=False
```

**注意**: 
- **`LANGSMITH_API_KEY`** (推奨) または `LANGCHAIN_API_KEY` (旧版) を設定してください
- 設定しない場合、LangSmithへのデータ保存と評価機能は自動的にスキップされます

### 3. データの準備

`data/` ディレクトリに `.txt` ファイルを配置してください。

## 実行

### 基本的な実行

初回実行（テストデータ生成 + 評価）:
```bash
python rag.py
```

### コストと時間を節約するオプション

**1. テストデータをキャッシュから読み込んで評価のみ実行**
```bash
python rag.py --skip-generation
```
テストデータ生成（コストがかかる）をスキップし、キャッシュから読み込みます。

**2. テストデータの生成のみ実行（評価はスキップ）**
```bash
python rag.py --only-generate
```
テストデータを生成してキャッシュに保存し、評価は後で実行できます。

**3. キャッシュを無視してテストデータを再生成**
```bash
python rag.py --regenerate
```
既存のキャッシュを無視して、新しいテストデータを生成します。

### 推奨ワークフロー

1. **初回**: テストデータを生成してキャッシュに保存
   ```bash
   python rag.py --only-generate
   ```

2. **2回目以降**: キャッシュから読み込んで評価を実行
   ```bash
   python rag.py --skip-generation
   ```

これにより、Azure OpenAI APIの呼び出し回数を大幅に削減できます。

## 機能

- `load_documents()`: dataディレクトリから.txtファイルを読み込み
- `create_synthesized_test_data()`: Ragasを使用してテストデータを生成
- `create_ls_dataset()`: LangSmithデータセットを作成
- `save_test_data()`: テストデータを保存
- `get_evaluator()`: Ragasの評価メトリクスを取得
- `infer()`: RAGによる推論と評価を実行

## 使用技術

- **LangChain**: RAGパイプラインの構築
- **Azure OpenAI**: LLMとEmbeddings
- **Chroma**: ベクトルデータベース
- **Ragas**: RAGシステムの評価
- **LangSmith**: 実験の追跡と評価

