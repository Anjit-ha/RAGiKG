"""
Check GraphRAG response structure
"""

import os
import json
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import LLMBase, LLMResponse
from neo4j_graphrag.retrievers import VectorRetriever
import requests

# ==================== CONFIG ====================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Anjitha@2002"
INDEX_NAME = "vector-index-name"

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_CHAT_MODEL = "meta/llama-3.2-90b-vision-instruct"

class NVIDIAEmbeddings(Embedder):
    def __init__(self, model: str = NVIDIA_EMBEDDING_MODEL):
        self.model = model
        self.api_key = NVIDIA_API_KEY
    
    def embed_query(self, text: str) -> list:
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        payload = {
            "model": self.model, "input": text, "input_type": "query",
            "encoding_format": "float", "truncate": "NONE"
        }
        response = requests.post(NVIDIA_EMBEDDINGS_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    async def async_embed_query(self, text: str) -> list:
        return self.embed_query(text)

class NVIDIALLM(LLMBase):
    def __init__(self, model: str = NVIDIA_CHAT_MODEL, temperature: float = 0.0, model_params=None):
        super().__init__(model_name=model, model_params=model_params or {"temperature": temperature})
        self.model = model
        self.temperature = temperature
        self.api_key = NVIDIA_API_KEY
    
    def invoke(self, input, message_history=None, system_instruction=None, response_format=None, **kwargs):
        if isinstance(input, str):
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": input})
        else:
            messages = input
        
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        payload = {
            "model": self.model, "messages": messages, "max_tokens": 512,
            "temperature": self.temperature, "top_p": 1.0,
        }
        response = requests.post(NVIDIA_CHAT_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return LLMResponse(content=content, response_format=response_format, prompt_tokens=0, completion_tokens=0)
    
    async def ainvoke(self, input, message_history=None, system_instruction=None, response_format=None, **kwargs):
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.invoke(input=input, message_history=message_history, system_instruction=system_instruction, response_format=response_format, **kwargs)
        )

# ==================== TEST ====================

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
embedder = NVIDIAEmbeddings()
retriever = VectorRetriever(driver, INDEX_NAME, embedder)
llm = NVIDIALLM()
rag = GraphRAG(retriever=retriever, llm=llm)

query_text = "Who is Hadi Bin Noor"
response = rag.search(query_text=query_text, retriever_config={"top_k": 3})

print("Response type:", type(response))
print("Response attributes:", dir(response))
print("\nResponse object:", response)
print("\nResponse dict:", response.__dict__)

driver.close()
