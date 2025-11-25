import numpy as np
assert np.__version__ == "1.26.4"

import os
import sys
import pickle
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import pandas as pd
import tiktoken

from dotenv import load_dotenv
load_dotenv(verbose=True)

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.documents import Document

# ============================================================
# 定数
# ============================================================

# ディレクトリ設定
CACHE_DIR = Path("./cache")
TMP_DIR = Path("./tmp")
TESTSET_CACHE_FILE = CACHE_DIR / "testset.pkl"
TESTSET_CSV_FILE = CACHE_DIR / "testset.csv"
CHUNK_MAPPING_FILE = CACHE_DIR / "chunk_mapping.csv"

# チャンク分割設定
MIN_CHUNK_TOKENS = 100
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_EMBEDDING_TOKENS = 8191

# チャンクサイズの閾値と設定
CHUNK_SIZE_CONFIG = [
    (500, 2000, 400),    # 短い: (閾値, chunk_size, overlap)
    (2000, 3000, 600),   # 中程度
    (5000, 4000, 800),   # 長い
    (float('inf'), 4000, 800),  # 非常に長い
]

# Azure OpenAI設定
AZURE_API_VERSION = "2024-02-15-preview"
AZURE_TIMEOUT = 30
AZURE_MAX_RETRIES = 2

# RAG設定
DEFAULT_RETRIEVER_K = 2 # 取得するチャンク数
TESTSET_SIZE = 10 # 生成するテストセットの数

# Azure OpenAIの環境変数設定を確認
def validate_azure_env_vars():
    """Azure OpenAIの環境変数を検証"""
    required_vars = {
        "AZURE_OPENAI_API_KEY": "Azure OpenAI APIキー",
        "AZURE_OPENAI_ENDPOINT": "Azure OpenAIエンドポイント",
        "AZURE_OPENAI_DEPLOYMENT_NAME": "Azure OpenAIデプロイメント名（LLM）",
        "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME": "Azure OpenAIデプロイメント名（Embedding）",
    }
    
    missing_vars = []
    warnings = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if not value:
            missing_vars.append(f"  - {var}: {description}")
        else:
            # エンドポイントのフォーマットチェック
            if var == "AZURE_OPENAI_ENDPOINT":
                if "/deployments/" in value or "/chat/completions" in value:
                    warnings.append(
                        f"⚠️  {var}は完全なURLではなく、ベースURLのみを指定してください\n"
                        f"   誤: {value}\n"
                        f"   正: https://your-resource.openai.azure.com/"
                    )
                else:
                    print(f"✓ {var}: {value}")
            else:
                print(f"✓ {var}: {value[:30]}..." if len(value) > 30 else f"✓ {var}: {value}")
    
    if warnings:
        print("\n" + "\n".join(warnings))
    
    if missing_vars:
        print("\n❌ 以下の環境変数が設定されていません:")
        print("\n".join(missing_vars))
        print("\n.envファイルを確認してください。")
        sys.exit(1)
    
    if warnings:
        print("\n.envファイルを修正してください。")
        sys.exit(1)
    
    print()

# 毎回エンコーダーを作成しなくてもいいようにキャッシュしておく
def get_encoding():
    """トークンエンコーダーを取得（キャッシュ）"""
    if not hasattr(get_encoding, '_cache'):
        get_encoding._cache = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    return get_encoding._cache

# 適切なチャンクサイズの決定とチャンクサイズの分析
def get_optimal_chunk_size(documents: List[Document]) -> Tuple[int, int]:
    """ドキュメントの特性に基づいて最適なチャンクサイズを決定"""
    encoding = get_encoding()
    
    # 全ドキュメントのトークン数を計算
    doc_tokens = [len(encoding.encode(doc.page_content)) for doc in documents]
    avg_tokens = sum(doc_tokens) / len(doc_tokens) if doc_tokens else 0
    max_tokens = max(doc_tokens) if doc_tokens else 0
    
    # 設定から適切なサイズを選択
    chunk_size, chunk_overlap = CHUNK_SIZE_CONFIG[-1][1:]  # デフォルト
    for threshold, size, overlap in CHUNK_SIZE_CONFIG:
        if avg_tokens < threshold:
            chunk_size, chunk_overlap = size, overlap
            break
    
    print(f"   ドキュメント分析:")
    print(f"     - 平均トークン数: {avg_tokens:.0f}")
    print(f"     - 最大トークン数: {max_tokens}")
    print(f"     - 選択されたチャンクサイズ: {chunk_size}トークン")
    print(f"     - オーバーラップ: {chunk_overlap}トークン")
    
    return chunk_size, chunk_overlap

# Azure OpenAIのLLMクライアントを作成
def create_azure_llm(**kwargs) -> AzureChatOpenAI:
    """Azure OpenAI LLMクライアントを作成"""
    return AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        timeout=AZURE_TIMEOUT,
        max_retries=AZURE_MAX_RETRIES,
        **kwargs
    )

# Azure OpenAIのEmbeddingsクライアントを作成
def create_azure_embeddings(**kwargs) -> AzureOpenAIEmbeddings:
    """Azure OpenAI Embeddingsクライアントを作成"""
    return AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", AZURE_API_VERSION),
        timeout=AZURE_TIMEOUT,
        max_retries=AZURE_MAX_RETRIES,
        **kwargs
    )

# テキスト分割器を作成
def create_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """テキスト分割器を作成"""
    encoding = get_encoding()
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(encoding.encode(text)),
        is_separator_regex=False,
        separators=["\n# ", "\n", "。", "．", ". ", "！", "？", "、", "，", ", ", " ", ""],
    )

def add_chunk_metadata(chunks: List[Document]) -> Tuple[List[Document], List[Dict]]:
    """チャンクにメタデータとIDを付与（page_contentにマーカーも埋め込み）"""
    encoding = get_encoding()
    filtered_chunks = []
    chunk_mapping = []
    chunk_counter = 0
    
    for chunk in chunks:
        tokens = len(encoding.encode(chunk.page_content))
        # トークン数が少なければ除外
        if tokens < MIN_CHUNK_TOKENS:
            continue
        
        source_file = chunk.metadata.get("source", "unknown") # ソースファイル名取得
        source_filename = Path(source_file).name # ソースファイル名のみ取得
        chunk_id = f"{source_filename}_chunk_{chunk_counter:03d}" # チャンクID生成
        
        # page_contentの先頭にマーカーを埋め込む
        chunk.page_content = f"[CHUNK_ID:{chunk_id}]\n{chunk.page_content}"
        
        # メタデータ追加
        chunk.metadata.update({
            "chunk_id": chunk_id,
            "chunk_index": chunk_counter,
            "chunk_tokens": tokens,
            "source_file": source_filename,
        })
        
        filtered_chunks.append(chunk)
        chunk_mapping.append({
            "chunk_id": chunk_id,
            "chunk_index": chunk_counter,
            "source_file": source_filename,
            "source_path": source_file,
            "tokens": tokens,
            "content_preview": chunk.page_content[:100].replace("\n", " ")
        })
        chunk_counter += 1
    
    return filtered_chunks, chunk_mapping

def save_chunk_mapping(chunk_mapping: List[Dict]):
    """チャンクマッピングをCSVに保存"""
    CACHE_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(chunk_mapping)
    df.to_csv(CHUNK_MAPPING_FILE, index=False, encoding="utf-8")

def save_chunks_to_tmp(filtered_chunks: List[Document]):
    """チャンクを個別のtxtファイルとしてtmpフォルダに保存"""
    # tmpディレクトリをクリーンアップして再作成
    if TMP_DIR.exists():
        import shutil
        shutil.rmtree(TMP_DIR)
    TMP_DIR.mkdir(exist_ok=True)
    
    for chunk in filtered_chunks:
        chunk_id = chunk.metadata.get("chunk_id", "unknown")
        chunk_index = chunk.metadata.get("chunk_index", -1)
        source_file = chunk.metadata.get("source_file", "unknown")
        chunk_tokens = chunk.metadata.get("chunk_tokens", 0)
        
        # ファイル名を生成
        filename = TMP_DIR / f"{chunk_id}.txt"
        
        # チャンクの内容をファイルに書き込み
        with open(filename, "w", encoding="utf-8") as f:
            # ヘッダー情報
            f.write("=" * 80 + "\n")
            f.write(f"チャンクID: {chunk_id}\n")
            f.write(f"ソースファイル: {source_file}\n")
            f.write(f"チャンクインデックス: {chunk_index}\n")
            f.write(f"トークン数: {chunk_tokens}\n")
            f.write("=" * 80 + "\n\n")
            
            # チャンクの内容
            f.write(chunk.page_content)
            f.write("\n")

def print_chunk_stats(original_count: int, filtered_chunks: List[Document], chunk_mapping: List[Dict]):
    """チャンク統計を表示"""
    tokens = [m["tokens"] for m in chunk_mapping]
    removed = original_count - len(filtered_chunks)
    
    print(f"   {original_count}個のチャンクに分割")
    if removed > 0:
        print(f"     - {removed}個の小さいチャンク（<{MIN_CHUNK_TOKENS}トークン）を除外")
    print(f"     - 使用するチャンク: {len(filtered_chunks)}個")
    print(f"     - 平均チャンクサイズ: {sum(tokens)/len(tokens):.0f}トークン")
    if tokens:
        print(f"     - 最小/最大: {min(tokens)}/{max(tokens)}トークン")
    print(f"     - チャンクマッピング保存: {CHUNK_MAPPING_FILE}")

# ドキュメントを読み込み、チャンク分割してメタデータを付与
def load_documents() -> List[Document]:
    """ドキュメントを読み込み、チャンク分割してメタデータを付与"""
    # ドキュメント読み込み
    loader = DirectoryLoader(
        path="./data",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    
    # チャンクサイズ決定とテキスト分割
    chunk_size, chunk_overlap = get_optimal_chunk_size(documents)
    text_splitter = create_text_splitter(chunk_size, chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    
    # メタデータ付与とフィルタリング
    filtered_chunks, chunk_mapping = add_chunk_metadata(chunks)
    
    # 保存と統計表示
    save_chunk_mapping(chunk_mapping)
    save_chunks_to_tmp(filtered_chunks)
    print(f"   {len(documents)}個のファイルを", end="")
    print_chunk_stats(len(chunks), filtered_chunks, chunk_mapping)
    print(f"     - チャンク内容保存: {TMP_DIR}")
    
    return filtered_chunks

# テストセットのデータ構造（LangChain版）
class TestSet:
    """LangChain版のテストセットデータ構造"""
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    
    def to_pandas(self):
        """テストセットをDataFrameに変換"""
        data = []
        for sample in self.samples:
            data.append({
                "user_input": sample.get("user_input", ""),
                "reference_contexts": sample.get("reference_contexts", []),
                "reference": sample.get("reference", ""),
                "synthesizer_name": sample.get("synthesizer_name", "langchain"),
            })
        return pd.DataFrame(data)

# テストデータセットを生成（LangChain版）
def create_synthesized_test_data(documents: List[Document], max_retries: int = 3):
    """テストデータセットを生成（LangChain版、エラー時は自動リトライ）"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import json
    import random
    
    # Azure OpenAIクライアント作成
    llm = create_azure_llm(
        temperature=0.7,  # 多様な質問を生成するため少し高めに設定
        model_kwargs={
            "response_format": {"type": "json_object"},  # JSONモードを有効化
        }
    )
    
    # プロンプトテンプレート（日本語対応）
    qa_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", """あなたはRAGシステムの評価用テストケースを生成する専門家です。
与えられたドキュメントの内容を基に、以下のJSON形式で質問と回答のペアを生成してください。

出力形式:
{{
    "question": "ドキュメントの内容に関する具体的な質問（日本語）",
    "answer": "質問に対する正確な回答（日本語、ドキュメントの内容に基づく）"
}}

質問は以下の要件を満たす必要があります:
- ドキュメントの内容に直接関連している
- 明確で具体的である
- 回答がドキュメントから導き出せる
- 日本語で記述されている"""),
        ("human", "以下のドキュメントから質問と回答のペアを生成してください:\n\n{document}")
    ])
    
    # チェーン構築
    chain = qa_generation_prompt | llm | StrOutputParser()
    
    # 段階的にサイズを減らしてリトライ
    testset_sizes = [TESTSET_SIZE, max(1, TESTSET_SIZE - 1), 1]
    
    for attempt, size in enumerate(testset_sizes, 1):
        try:
            print(f"   試行 {attempt}/{len(testset_sizes)}: testset_size={size}")
            
            # ドキュメントからランダムに選択（重複を避ける）
            selected_docs = random.sample(documents, min(size, len(documents)))
            
            test_samples = []
            for idx, doc in enumerate(selected_docs):
                try:
                    # チャンクIDを抽出
                    chunk_id_match = re.search(r'\[CHUNK_ID:([^\]]+)\]', doc.page_content)
                    chunk_id = chunk_id_match.group(1) if chunk_id_match else doc.metadata.get("chunk_id", f"chunk_{idx}")
                    
                    # ドキュメント内容からマーカーを除去（プロンプトに含めるため）
                    doc_content = re.sub(r'\[CHUNK_ID:[^\]]+\]\n?', '', doc.page_content)
                    
                    # LLMで質問と回答を生成
                    response = chain.invoke({"document": doc_content})
                    
                    # JSONをパース
                    try:
                        qa_data = json.loads(response)
                        question = qa_data.get("question", "")
                        answer = qa_data.get("answer", "")
                        
                        if not question or not answer:
                            print(f"   ⚠️  サンプル {idx+1}: 質問または回答が空です。スキップします。")
                            continue
                        
                        # テストサンプルを作成
                        test_samples.append({
                            "user_input": question,
                            "reference_contexts": [doc_content],  # 元のドキュメント内容
                            "reference": answer,
                            "synthesizer_name": "langchain",
                            "chunk_id": chunk_id,
                            "source_file": doc.metadata.get("source_file", "unknown"),
                        })
                        
                        print(f"   ✓ サンプル {idx+1}/{size} を生成しました")
                        
                    except json.JSONDecodeError as e:
                        print(f"   ⚠️  サンプル {idx+1}: JSONパースエラー - {str(e)[:100]}")
                        continue
                        
                except Exception as e:
                    print(f"   ⚠️  サンプル {idx+1}: エラー - {str(e)[:100]}")
                    continue
            
            if len(test_samples) == 0:
                raise ValueError("テストサンプルが1つも生成できませんでした")
            
            if attempt > 1:
                print(f"   ✓ リトライ成功（サイズ: {len(test_samples)}）")
            
            # TestSetオブジェクトを作成
            testset = TestSet(test_samples)
            return testset
            
        except ValueError as e:
            error_msg = str(e)
            print(f"⚠️  エラー発生: {error_msg}")
            if attempt < len(testset_sizes):
                print(f"   → サイズを {testset_sizes[attempt]} に減らして再試行します")
            else:
                print(f"   → すべての試行が失敗しました")
                raise
        
        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  予期しないエラー: {error_msg[:150]}...")
            if attempt < len(testset_sizes):
                print(f"   → サイズを {testset_sizes[attempt]} に減らして再試行します")
                continue
            else:
                print(f"   → すべての試行が失敗しました")
                raise

def find_chunk_ids_for_contexts(contexts: List[str], documents: List[Document]) -> List[str]:
    """コンテキストテキストに対応するchunk_idを見つける"""
    chunk_ids = []
    
    for context in contexts:# contextはテスト生成の際に用いたコンテキスト
        # まず[CHUNK_ID:xxx]パターンを正規表現で抽出
        chunk_id_match = re.search(r'\[CHUNK_ID:([^\]]+)\]', context)
        if chunk_id_match:
            chunk_id = chunk_id_match.group(1)
            chunk_ids.append(chunk_id)
            continue
        
        # マーカーが見つからない場合は従来の方法でマッチング
        matched = False
        context_normalized = context.strip()
        
        for doc in documents:# documentsは分割されたテキスト
            doc_content = doc.page_content.strip()
            # 完全一致または高い類似度でマッチング
            if context_normalized == doc_content or context_normalized in doc_content:
                chunk_id = doc.metadata.get("chunk_id", "unknown")
                chunk_ids.append(chunk_id)
                matched = True
                break
        
        if not matched:
            chunk_ids.append("unknown")
    
    return chunk_ids

def save_testset_to_cache(testset, documents: List[Document]):
    """テストセットをキャッシュに保存（期待されるchunk_id付き、LangChain版）"""
    CACHE_DIR.mkdir(exist_ok=True)
    with open(TESTSET_CACHE_FILE, "wb") as f:
        pickle.dump(testset, f)
    print(f"💾 テストセットをキャッシュに保存しました: {TESTSET_CACHE_FILE}")

    # DataFrameに変換
    df_test = testset.to_pandas()
    
    # 各サンプルの期待されるchunk_idを抽出
    expected_chunk_ids_list = []
    
    for sample in testset.samples:
        # LangChain版では、chunk_idが直接サンプルに含まれている
        chunk_id = sample.get("chunk_id")
        if chunk_id:
            chunk_ids = [chunk_id]
        else:
            # フォールバック: reference_contextsから検索
            contexts = sample.get("reference_contexts", [])
            chunk_ids = find_chunk_ids_for_contexts(contexts, documents)
        
        # リスト形式で保存（空の場合は空リスト）
        expected_chunk_ids_list.append(chunk_ids if chunk_ids else [])
    
    # 新しい列を追加
    df_test['expected_chunk_ids'] = expected_chunk_ids_list
    
    # 列の順序を調整（見やすくするため）
    columns_order = ['user_input', 'expected_chunk_ids', 'reference_contexts', 'reference', 'synthesizer_name']
    df_test = df_test[columns_order]
    
    df_test.to_csv(TESTSET_CSV_FILE, index=False)
    print(f"💾 テストセットをCSVに保存しました: {TESTSET_CSV_FILE}")

def load_testset_from_cache():
    """キャッシュからテストセットを読み込み"""
    if TESTSET_CACHE_FILE.exists():
        with open(TESTSET_CACHE_FILE, "rb") as f:
            testset = pickle.load(f)
        print(f"📦 キャッシュからテストセットを読み込みました: {TESTSET_CACHE_FILE}")
        return testset
    return None

def generate_run_id() -> str:
    """実行ごとのユニークなIDを生成"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def create_ls_dataset(run_id: str):
    """実行IDを含むデータセットを作成"""
    from langsmith import Client
    
    # LangSmith APIキーが設定されているか確認（新旧両方をサポート）
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("⚠️  LANGSMITH_API_KEY が設定されていません。LangSmithへの保存をスキップします。")
        return None, None

    # 実行ごとにユニークなデータセット名を生成
    dataset_name = f"agent-book_{run_id}"

    try:
        client = Client()
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=f"RAG評価データセット (実行ID: {run_id})"
        )
        return dataset, dataset_name
    except Exception as e:
        print(f"⚠️  LangSmithデータセットの作成に失敗しました: {e}")
        print("   評価は続行しますが、LangSmithへの保存はスキップされます。")
        return None, None

def save_test_data(testset, dataset, run_id: str):
    """テストデータをLangSmithに保存（実行IDとタイムスタンプ付き、LangChain版）"""
    if dataset is None:
        print("   LangSmithデータセットが利用できないため、スキップします。")
        return
    
    from langsmith import Client
    
    try:
        client = Client()
        inputs = []
        outputs = []
        metadatas = []
        
        timestamp = datetime.now().isoformat()

        for idx, testset_record in enumerate(testset.samples):
            # LangChain版では、サンプルは辞書形式
            contexts = testset_record.get("reference_contexts", [])
            
            inputs.append(
                {
                    "question": testset_record.get("user_input", ""),
                }
            )
            outputs.append(
                {
                    "contexts": contexts,
                    "ground_truth": testset_record.get("reference", ""),
                }
            )
            metadatas.append(
                {
                    "run_id": run_id,
                    "timestamp": timestamp,
                    "example_index": idx,
                    "source": "synthesized",
                    "synthesizer": testset_record.get("synthesizer_name", "langchain"),
                }
            )
        
        client.create_examples(
            inputs=inputs,
            outputs=outputs,
            metadata=metadatas,
            dataset_id=dataset.id,
        )
        print(f"✓ テストデータを保存しました (実行ID: {run_id})\n")
    except Exception as e:
        print(f"⚠️  テストデータの保存に失敗しました: {e}")
        print("   評価は続行します。")


def get_evaluator():
    """RAG評価器を作成"""
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from evaluator import RagasMetricEvaluator

    llm = create_azure_llm(temperature=0)
    embeddings = create_azure_embeddings()
    metrics = [context_precision, answer_relevancy, context_recall, faithfulness] # 　金かかるから一旦抜きで

    return [RagasMetricEvaluator(m, llm, embeddings).evaluate for m in metrics]


def create_rag_chain(documents: List[Document]):
    """RAGチェーンを作成"""
    from langchain_chroma import Chroma
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    
    # ChromaDBのテレメトリーを無効化
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    
    # ベクトルDB構築
    embeddings = create_azure_embeddings()
    chunk_ids = [doc.metadata.get("chunk_id", f"chunk_{i}") for i, doc in enumerate(documents)]
    db = Chroma.from_documents(documents, embeddings, ids=chunk_ids)
    
    # プロンプトとモデル
    prompt = ChatPromptTemplate.from_template(
        "以下の文脈だけを踏まえて質問に回答してください。\n\n"
        "文脈: \"\"\"\n{context}\n\"\"\"\n\n質問: {question}"
    )
    model = create_azure_llm(temperature=0)
    
    # Retriever設定
    k = min(DEFAULT_RETRIEVER_K, len(documents))
    retriever = db.as_retriever(search_kwargs={"k": k})
    
    # チェーン構築
    return RunnableParallel({
        "question": RunnablePassthrough(),
        "context": retriever,
    }).assign(answer=prompt | model | StrOutputParser())

def extract_context_metadata(contexts: List[Document]) -> List[Dict]:
    """コンテキストからメタデータを抽出"""
    return [{
        "chunk_id": doc.metadata.get("chunk_id", "unknown"),
        "source_file": doc.metadata.get("source_file", "unknown"),
        "chunk_index": doc.metadata.get("chunk_index", -1),
        "content": doc.page_content
    } for doc in contexts]

def infer(evaluators, documents: List[Document], dataset_name: str, run_id: str):
    """推論と評価を実行"""
    # LangSmith APIキー確認
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("⚠️  LANGSMITH_API_KEY が設定されていないため、評価をスキップします。")
        return None
    
    # RAGチェーン作成
    chain = create_rag_chain(documents)
    
    def predict(inputs: Dict[str, Any]) -> Dict[str, Any]:
        output = chain.invoke(inputs["question"])
        context_metadata = extract_context_metadata(output["context"])
        
        return {
            "contexts": output["context"],
            "answer": output["answer"],
            "retrieved_chunk_ids": [m["chunk_id"] for m in context_metadata],
            "context_metadata": context_metadata,
            "run_id": run_id,  # 実行IDを含める
        }
    
    try:
        from langsmith.evaluation import evaluate
        return evaluate(
            predict, 
            data=dataset_name, 
            evaluators=evaluators,
            experiment_prefix=f"rag-eval-{run_id}"  # 実験名にもrun_idを含める
        )
    except Exception as e:
        print(f"⚠️  評価の実行に失敗しました: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="RAG評価システム")
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="テストデータ生成をスキップし、キャッシュから読み込む",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="キャッシュを無視してテストデータを再生成",
    )
    parser.add_argument(
        "--only-generate",
        action="store_true",
        help="テストデータの生成のみを実行（評価は実行しない）",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="評価をスキップ（テストデータの生成・保存のみ実行）",
    )
    args = parser.parse_args()
    validate_azure_env_vars()
    
    # 実行ごとのユニークなIDを生成
    run_id = generate_run_id()
    print(f"🆔 実行ID: {run_id}\n")
    
    print("📄 ドキュメントを読み込んでいます...")
    documents = load_documents()
    print(f"✓ {len(documents)}個のドキュメントを読み込みました\n")
    
    # テストデータの生成または読み込み
    testset = None
    if args.skip_generation and not args.regenerate:
        testset = load_testset_from_cache()
        if testset:
            print(f"✓ {len(testset.samples)}個のテストケースを読み込みました\n")
        else:
            print("⚠️  キャッシュが見つかりません。テストデータを生成します...\n")
    
    if testset is None or args.regenerate:
        print("🔬 テストデータを生成しています...")
        try:
            testset = create_synthesized_test_data(documents)
            print(f"✓ {len(testset.samples)}個のテストケースを生成しました\n")
            
            # キャッシュに保存（chunk_id情報も含める）
            save_testset_to_cache(testset, documents)
            print()
        except Exception as e:
            print(f"\n❌ テストデータ生成に失敗しました: {e}")
            
            # 既存のキャッシュを確認
            cached_testset = load_testset_from_cache()
            if cached_testset:
                print("   → 既存のキャッシュを使用します")
                testset = cached_testset
            else:
                print("   → キャッシュも見つかりません。評価をスキップします。")
                print("\n" + "=" * 50)
                print("⚠️  テストデータがないため、処理を終了します")
                print("=" * 50)
                return
    
    # LangSmithへの保存
    dataset = None
    dataset_name = None
    if not args.only_generate:
        print("📊 LangSmithデータセットを作成しています...")
        dataset, dataset_name = create_ls_dataset(run_id)
        if dataset:
            print(f"✓ データセット '{dataset.name}' を作成しました\n")
        else:
            print("   LangSmithデータセットの作成をスキップしました\n")
        
        print("💾 テストデータをLangSmithに保存しています...")
        save_test_data(testset, dataset, run_id)
    
    if args.only_generate:
        print("=" * 50)
        print("✓ テストデータ生成が完了しました")
        print("=" * 50)
        return
    
    # 評価の実行
    if not args.skip_evaluation:
        if dataset_name is None:
            print("⚠️  データセットが作成されていないため、評価をスキップします\n")
        else:
            print("⚙️  評価器を初期化しています...")
            evaluators = get_evaluator()
            print("✓ 評価器を初期化しました\n")
            
            print("🚀 推論と評価を実行しています...")
            result = infer(evaluators, documents, dataset_name, run_id)
            
            if result:
                print("✓ 評価が完了しました\n")
                print("=" * 50)
                print("📈 評価結果:")
                print("=" * 50)
                
                # 結果をDataFrameに変換して表示
                import pandas as pd
                df = result.to_pandas()
                # print(df.to_string())
                
                # CSV保存
                #result_csv = CACHE_DIR / f"evaluation_results_{run_id}.csv"
                #df.to_csv(result_csv, index=False)
                #print(f"\n💾 評価結果を保存しました: {result_csv}")
                
                # サマリー表示
                if 'feedback.context_precision' in df.columns and 'feedback.answer_relevancy' in df.columns:
                    print("\n📊 スコアサマリー:")
                    print(f"  - Context Precision 平均: {df['feedback.context_precision'].mean():.3f}")
                    print(f"  - Answer Relevancy 平均: {df['feedback.answer_relevancy'].mean():.3f}")
            else:
                print("⚠️  評価をスキップしました（LangSmith APIキーが未設定）\n")
    else:
        print("=" * 50)
        print("✓ 評価をスキップしました")
        print("=" * 50)

if __name__ == "__main__":
    main()