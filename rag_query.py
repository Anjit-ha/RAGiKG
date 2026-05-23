"""
GraphRAG Query using NVIDIA NIM APIs instead of OpenAI.
Adapted for the current itext2kg-main codebase.
"""

import os
import json
import asyncio
import requests
from typing import Union, List, Optional, Any, Dict
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.llm import LLMBase, LLMResponse
from neo4j_graphrag.retrievers import VectorRetriever
from openai import OpenAI

# ==================== NEO4J CONFIG ====================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Anjitha@2002"
INDEX_NAME = "vector-index-name"

# ==================== NVIDIA CONFIG ====================
def load_local_env() -> None:
    env_path = ".env"
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise EnvironmentError("Set NVIDIA_API_KEY environment variable before running this script.")

nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_CHAT_MODEL = "meta/llama-3.3-70b-instruct"
NVIDIA_EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# ==================== CUSTOM NVIDIA EMBEDDER ====================
class NVIDIAEmbeddings(Embedder):
    """Custom embeddings class for NVIDIA NIM API."""
    
    def __init__(self, model: str = NVIDIA_EMBEDDING_MODEL):
        self.model = model
    
    def embed_query(self, text: str) -> list:
        """Generate embedding for a query."""
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
        }
        payload = {
            "model": self.model,
            "input": text,
            "input_type": "query",
            "encoding_format": "float",
            "truncate": "NONE",
        }
        response = requests.post(
            NVIDIA_EMBEDDINGS_URL,
            headers=headers,
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    
    async def async_embed_query(self, text: str) -> list:
        """Async version of embed_query (delegates to sync for now)."""
        return self.embed_query(text)

# ==================== CUSTOM NVIDIA LLM ====================
class NVIDIALLM(LLMBase):
    """Custom LLM class for NVIDIA NIM API."""
    
    def __init__(self, model: str = NVIDIA_CHAT_MODEL, temperature: float = 0.0, model_params: Optional[dict] = None):
        super().__init__(
            model_name=model,
            model_params=model_params or {"temperature": temperature}
        )
        self.model = model
        self.temperature = temperature
    
    def invoke(
        self,
        input: Union[str, List[Dict[str, str]]],
        message_history: Optional[Union[List[Dict[str, str]]]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate response from NVIDIA LLM."""
        
        # Handle different input types
        if isinstance(input, str):
            # Simple string input
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": input})
        else:
            # List of message dicts
            messages = input
        
        response = nvidia_client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=self.temperature,
        )
        
        content = response.choices[0].message.content
        
        return LLMResponse(
            content=content,
            response_format=response_format,
            prompt_tokens=0,  # NVIDIA API doesn't return token counts
            completion_tokens=0
        )
    
    async def ainvoke(
        self,
        input: Union[str, List[Dict[str, str]]],
        message_history: Optional[Union[List[Dict[str, str]]]] = None,
        system_instruction: Optional[str] = None,
        response_format: Optional[Any] = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Async version of invoke."""
        # For now, just call the sync version in an executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.invoke(
                input=input,
                message_history=message_history,
                system_instruction=system_instruction,
                response_format=response_format,
                **kwargs
            )
        )

# ==================== INITIALIZE COMPONENTS ====================
print("Initializing GraphRAG with NVIDIA NIM APIs...")

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
print(f"✅ Connected to Neo4j at {NEO4J_URI}")

# Create embedder using NVIDIA
embedder = NVIDIAEmbeddings(model=NVIDIA_EMBEDDING_MODEL)
print(f"✅ Initialized NVIDIA embedder ({NVIDIA_EMBEDDING_MODEL})")

# Initialize retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)
print(f"✅ Initialized retriever with index '{INDEX_NAME}'")

# Instantiate NVIDIA LLM
llm = NVIDIALLM(model=NVIDIA_CHAT_MODEL, temperature=0.0)
print(f"✅ Initialized NVIDIA LLM ({NVIDIA_CHAT_MODEL})")

# Instantiate GraphRAG pipeline
rag = GraphRAG(retriever=retriever, llm=llm)
print(f"✅ GraphRAG pipeline ready!\n")

# ==================== QUERY THE GRAPH ====================
print("=" * 60)
print("QUERYING KNOWLEDGE GRAPH")
print("=" * 60)

# ==================== QUERY THE GRAPH ====================
print("=" * 60)
print("QUERYING KNOWLEDGE GRAPH")
print("=" * 60)

query_text = "Who is Hadi Bin Noor and what are their relationships?"
print(f"\n❓ Query: {query_text}")

try:
    # Use GraphRAG with enriched query that asks for relationships
    response = rag.search(query_text=query_text, retriever_config={"top_k": 5})
    print(f"\n📝 Answer:\n{response.answer}")
except Exception as e:
    print(f"⚠️  GraphRAG search failed: {e}")
    print("\nNote: This may fail if the vector index is not properly set up.")
    print("Run the recreate_vector_index.py script first to create and populate the vector index.")

driver.close()
print("\n✅ Done!")
