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

# テスト生成時の多様性確保設定
QUESTION_SIMILARITY_THRESHOLD = 0.85  # 質問の類似度閾値（これ以上は除外）
MAX_CHUNK_USAGE_COUNT = 2  # 同じチャンクから生成できる最大回数

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

# ============================================================
# LangChain版テストセット生成
# ============================================================

# テストセットのデータ構造（LangChain版）
# ここにはexpected_chunk_idsがないが、後で追加するので問題ない
class TestSet:
    """LangChain版のテストセットデータ構造"""
    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples
    
    def to_pandas(self):
        """テストセットをDataFrameに変換"""
        data = []
        for sample in self.samples:
            data.append({
                "user_input": sample.get("user_input", ""), # 質問
                "reference_contexts": sample.get("reference_contexts", []), # 参照コンテキスト
                "reference": sample.get("reference", ""), # 正解
                "synthesizer_name": sample.get("synthesizer_name", "langchain"), # 生成方法
            })
        return pd.DataFrame(data)

# テストデータセットを生成
# 【変更点】RAGASのTestsetGeneratorからLangChainのプロンプトテンプレート+LLMに変更
def create_synthesized_test_data(
    documents: List[Document], 
    max_retries: int = 3,
    question_types: List[str] = None
):
    """テストデータセットを生成（LangChain版、エラー時は自動リトライ）
    
    RAGAS版からの主な変更:
    - TestsetGeneratorの代わりにChatPromptTemplateとLLMを使用
    - 各ドキュメントから直接質問と回答を生成
    - JSON形式で出力を受け取り、パースしてテストサンプルを作成
    
    Args:
        documents: ドキュメントのリスト
        max_retries: 最大リトライ回数
        question_types: 質問の傾向のリスト
            - "single_hop": 単一ホップ質問（デフォルト）
            - "multi_hop": マルチホップ質問（複数の情報を組み合わせる）
            - "synonym": 同義語で言い換えた質問
            - "typo": 誤字を含む質問
            - "negation": 否定形の質問
    """
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    import json
    import random
    
    # デフォルト値の設定
    if question_types is None:
        question_types = ["single_hop"]
    
    # Azure OpenAIクライアント作成（LangChain版）
    llm = create_azure_llm(
        temperature=0.7,  # 多様な質問を生成するため少し高めに設定
        model_kwargs={
            "response_format": {"type": "json_object"},  # JSONモードを有効化
        }
    )
    
    # プロンプトテンプレート（日本語対応、質問の傾向を動的に組み込む）
    # LangChainのChatPromptTemplateを使用して質問と回答を生成
    question_instructions = []
    if "multi_hop" in question_types:
        question_instructions.append("- 複数の情報を組み合わせて推論が必要な質問（マルチホップ）")
    if "synonym" in question_types:
        question_instructions.append("- 同義語や類義語を使って言い換えた質問")
    if "typo" in question_types:
        question_instructions.append("- 意図的な誤字やタイプミスを含む質問（例：「蓋然性」→「蓮然性」、「契約」→「k約」、「利用」→「理容」など、よくある誤字や変換ミスを含める。質問文に必ず1つ以上の誤字を含めること）")
    if "negation" in question_types:
        question_instructions.append("- 否定形や反対の意味を問う質問")
    if "single_hop" in question_types or not question_instructions:
        question_instructions.append("- ドキュメントから直接答えられる単純な質問（シングルホップ）")
    
    question_instructions_text = "\n".join(question_instructions)
    
    # プロンプトテンプレートの構築
    # JSONの例の中の{question}と{answer}は変数として解釈されないように、
    # 文字列リテラルとして表現（"question"と"answer"をそのまま使用）
    system_prompt_template = """あなたはRAGシステムの評価用テストケースを生成する専門家です。
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
- 日本語で記述されている

質問の傾向:
{question_instructions_text}"""
    
    # question_instructions_textを先に置換（formatで置換）
    system_prompt_intermediate = system_prompt_template.format(question_instructions_text=question_instructions_text)
    
    
    # ChatPromptTemplateが{question}と{answer}を変数として解釈する問題を回避するため、
    # 直接LLMを呼び出す関数を定義
    def generate_qa(doc_content: str) -> str:
        """ドキュメントから質問と回答を生成"""
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt_intermediate),
            HumanMessage(content=f"以下のドキュメントから質問と回答のペアを生成してください:\n\n{doc_content}")
        ]
        # LLMを直接呼び出し
        response = llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)
    
    # Embeddingsクライアントを作成（質問の類似度チェック用）
    embeddings = create_azure_embeddings()
    
    # 質問の類似度をチェックする関数
    def is_question_similar(new_question: str, existing_questions: List[str], threshold: float = QUESTION_SIMILARITY_THRESHOLD) -> bool:
        """新しい質問が既存の質問と類似しているかチェック"""
        if not existing_questions:
            return False
        
        try:
            # 既存の質問のベクトルを取得
            existing_vectors = embeddings.embed_documents(existing_questions)
            # 新しい質問のベクトルを取得
            new_vector = embeddings.embed_query(new_question)
            
            # コサイン類似度を計算
            existing_vectors_np = np.array(existing_vectors)
            new_vector_np = np.array(new_vector)
            
            # 正規化
            existing_norms = np.linalg.norm(existing_vectors_np, axis=1)
            new_norm = np.linalg.norm(new_vector_np)
            
            # コサイン類似度 = (A・B) / (|A| * |B|)
            similarities = np.dot(existing_vectors_np, new_vector_np) / (existing_norms * new_norm)
            
            # 最大類似度が閾値を超えているかチェック
            return float(np.max(similarities)) >= threshold
        except Exception as e:
            print(f"   ⚠️  類似度チェックエラー: {str(e)[:100]}")
            # エラー時は類似とみなさない（安全側に倒す）
            return False
    
    # 段階的にサイズを減らしてリトライ
    testset_sizes = [TESTSET_SIZE, max(1, TESTSET_SIZE - 1), 1]
    
    for attempt, size in enumerate(testset_sizes, 1):
        try:
            print(f"   試行 {attempt}/{len(testset_sizes)}: testset_size={size}")
            
            # ドキュメントからランダムに選択（重複あり）
            # LangChain版では明示的にランダムサンプリング、
            # selected_docs = random.sample(documents, min(size, len(documents))) # sampleでは重複なしなので、chunk数が少ないとテストも少なくなる
            selected_docs = random.choices(documents, k=size) # こうすることで重複ありでテストを生成することができる  
            
            test_samples = []
            existing_questions = []  # 既存の質問を保持（類似度チェック用）
            chunk_usage_count = {}  # チャンクの使用回数をカウント
            
            for idx, doc in enumerate(selected_docs):
                try:
                    # チャンクIDを抽出
                    chunk_id_match = re.search(r'\[CHUNK_ID:([^\]]+)\]', doc.page_content)
                    chunk_id = chunk_id_match.group(1) if chunk_id_match else doc.metadata.get("chunk_id", f"chunk_{idx}")
                    
                    # チャンクの使用頻度をチェック
                    chunk_usage_count[chunk_id] = chunk_usage_count.get(chunk_id, 0)
                    if chunk_usage_count[chunk_id] >= MAX_CHUNK_USAGE_COUNT:
                        print(f"   ⚠️  サンプル {idx+1}: チャンク {chunk_id} の使用回数が上限に達しています。スキップします。")
                        continue
                    
                    # ドキュメント内容からマーカーを除去（プロンプトに含めるため）
                    doc_content = re.sub(r'\[CHUNK_ID:[^\]]+\]\n?', '', doc.page_content)
                    
                    # LLMで質問と回答を生成（LangChain版）
                    # 【変更点】RAGAS版ではTestsetGenerator.generate_with_langchain_docs()を使用していたが、
                    # LangChain版では各ドキュメントに対して個別にLLMを呼び出し
                    # ChatPromptTemplateの代わりに、直接LLMを呼び出す関数を使用
                    response = generate_qa(doc_content)
                    
                    # JSONをパース（LangChain版）
                    # 【変更点】RAGAS版ではTestsetGeneratorが自動的にパースしていたが、
                    # LangChain版では明示的にJSONをパース
                    try:
                        qa_data = json.loads(response)
                        question = qa_data.get("question", "")
                        answer = qa_data.get("answer", "")
                        
                        if not question or not answer:
                            print(f"   ⚠️  サンプル {idx+1}: 質問または回答が空です。スキップします。")
                            continue
                        
                        # 質問の類似度をチェック（既存の質問と似すぎていないか）
                        if is_question_similar(question, existing_questions):
                            print(f"   ⚠️  サンプル {idx+1}: 既存の質問と類似度が高すぎます。スキップします。")
                            continue
                        
                        # テストサンプルを作成（LangChain版のデータ構造）
                        # 【変更点】RAGAS版ではTestsetGeneratorが自動的にTestsetオブジェクトを作成していたが、
                        # LangChain版では辞書形式で明示的に作成
                        test_samples.append({
                            "user_input": question,
                            "reference_contexts": [doc_content],  # 元のドキュメント内容
                            "reference": answer,
                            "synthesizer_name": "langchain",
                            "chunk_id": chunk_id,
                            "source_file": doc.metadata.get("source_file", "unknown"),
                        })
                        
                        # 既存の質問リストとチャンク使用回数を更新
                        existing_questions.append(question)
                        chunk_usage_count[chunk_id] = chunk_usage_count.get(chunk_id, 0) + 1
                        
                        print(f"   ✓ サンプル {idx+1}/{size} を生成しました（チャンク {chunk_id} 使用回数: {chunk_usage_count[chunk_id]}）")
                        
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
            
            # TestSetオブジェクトを作成（LangChain版）
            # 【変更点】RAGAS版ではTestsetGeneratorがTestsetオブジェクトを返していたが、
            # LangChain版ではカスタムのTestSetクラスを使用
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
    """テストセットをキャッシュに保存（期待されるchunk_id付き、LangChain版）
    
    【変更点】RAGAS版のTestset.samplesから、LangChain版のTestSet.samples（辞書リスト）に対応
    """
    CACHE_DIR.mkdir(exist_ok=True)
    with open(TESTSET_CACHE_FILE, "wb") as f:
        pickle.dump(testset, f)
    print(f"💾 テストセットをキャッシュに保存しました: {TESTSET_CACHE_FILE}")

    # DataFrameに変換
    df_test = testset.to_pandas()
    
    # 各サンプルの期待されるchunk_idを抽出
    expected_chunk_ids_list = []
    
    for sample in testset.samples:
        # 【LangChain版変更】RAGAS版ではsample.eval_sample.reference_context_idsを使用していたが、
        # LangChain版ではchunk_idが直接サンプル辞書に含まれている
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
    """テストデータをLangSmithに保存（実行IDとタイムスタンプ付き、LangChain版）
    
    【変更点】RAGAS版のtestset_record.eval_sampleから、LangChain版の辞書形式（testset_record.get()）に対応
    """
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
            # 【LangChain版変更】RAGAS版ではtestset_record.eval_sample.reference_contextsを使用していたが、
            # LangChain版ではtestset_record.get("reference_contexts")で辞書から直接取得
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
    parser.add_argument(
        "--question-types",
        nargs="+",
        default=["single_hop"],
        choices=["single_hop", "multi_hop", "synonym", "typo", "negation"],
        help="質問の傾向を指定（複数指定可）: single_hop, multi_hop, synonym, typo, negation",
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
        if args.question_types:
            print(f"   質問の傾向: {', '.join(args.question_types)}")
        try:
            testset = create_synthesized_test_data(documents, question_types=args.question_types)
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