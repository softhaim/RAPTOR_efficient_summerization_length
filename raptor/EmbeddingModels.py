import logging
from abc import ABC, abstractmethod

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-ada-002"):
        self.client = OpenAI()
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)



class LLaMAEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.pipeline = pipeline(
            "feature-extraction",
            model=model_name,
            tokenizer=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        try:
            # 임베딩 생성
            embedding_vector = self.pipeline(text, return_tensors=False)[0][0]
            
            # torch.Tensor를 list[float]로 변환하여 반환
            if isinstance(embedding_vector, torch.Tensor):
                embedding_vector = embedding_vector.tolist()

            return embedding_vector  # list[float] 형식으로 반환
        except Exception as e:
            print(f"LLaMA 임베딩 생성 오류: {e}")
            return []
