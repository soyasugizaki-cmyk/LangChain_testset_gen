from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langsmith.schemas import Example, Run
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.base import Metric, MetricWithEmbeddings, MetricWithLLM


class RagasMetricEvaluator:
    def __init__(self, metric: Metric, llm: BaseChatModel, embeddings: Embeddings):
        self.metric = metric

        # LLMとEmbeddingsをMetricに設定
        if isinstance(self.metric, MetricWithLLM):
            self.metric.llm = LangchainLLMWrapper(llm)
        if isinstance(self.metric, MetricWithEmbeddings):
            self.metric.embeddings = LangchainEmbeddingsWrapper(embeddings)

    def evaluate(self, run: Run, example: Example) -> dict[str, Any]:
        context_strs = [doc.page_content for doc in run.outputs["contexts"]]

        # Ragas 0.3.8の新しいデータ形式に合わせる
        score = self.metric.score(
            {
                "user_input": example.inputs["question"],  # question → user_input
                "response": run.outputs["answer"],  # answer → response
                "retrieved_contexts": context_strs,  # contexts → retrieved_contexts
                "reference": example.outputs["ground_truth"],  # ground_truth → reference
            },
        )
        return {"key": self.metric.name, "score": score}