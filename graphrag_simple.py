"""
Simplified GraphRAG Query with better error handling and relationship context.
"""

import os
import requests
from neo4j import GraphDatabase
from openai import OpenAI

# ==================== CONFIG ====================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Anjitha@2002"


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


# Initialize NVIDIA OpenAI client
load_local_env()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise EnvironmentError(
        "Set NVIDIA_API_KEY environment variable. In PowerShell: "
        "$env:NVIDIA_API_KEY='your_key_here'"
    )

nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_CHAT_MODEL = "meta/llama-3.3-70b-instruct"
NVIDIA_EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"

# ==================== HELPER FUNCTIONS ====================

def get_nvidia_embedding(text: str) -> list:
    """Get a query embedding from NVIDIA NIM API."""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "model": NVIDIA_EMBEDDING_MODEL,
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

def get_nvidia_response(prompt: str) -> str:
    """Get response from NVIDIA LLM using OpenAI client."""
    response = nvidia_client.chat.completions.create(
        model=NVIDIA_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return response.choices[0].message.content

def get_entity_context(driver, entity_name: str) -> str:
    """Get relationships for an entity."""
    with driver.session() as session:
        # Get outgoing relationships
        out_results = session.run("""
            MATCH (a:Entity {name: $name})-[r]->(b:Entity)
            RETURN type(r) AS rel_type, b.name AS target
        """, name=entity_name)
        
        # Get incoming relationships
        in_results = session.run("""
            MATCH (a:Entity)-[r]->(b:Entity {name: $name})
            RETURN type(r) AS rel_type, a.name AS source
        """, name=entity_name)
        
        lines = [f"Entity: {entity_name}\n"]
        
        for record in out_results:
            rel = record['rel_type'].replace('_', ' ').lower()
            lines.append(f"  - {entity_name} {rel} {record['target']}")
        
        for record in in_results:
            rel = record['rel_type'].replace('_', ' ').lower()
            lines.append(f"  - {record['source']} {rel} {entity_name}")
        
        return "\n".join(lines)

def retrieve_relevant_entities(driver, query: str, top_k: int = 3) -> list:
    """Vector search for relevant entities."""
    query_embedding = get_nvidia_embedding(query)
    
    with driver.session() as session:
        result = session.run(f"""
            CALL db.index.vector.queryNodes('vector-index-name', $k, $embedding)
            YIELD node, score
            RETURN node.name AS entity, score
            ORDER BY score DESC
            LIMIT $k
        """, embedding=query_embedding, k=top_k)
        
        return [record['entity'] for record in result]

# ==================== MAIN GRAPHRAG FLOW ====================

print("=" * 70)
print("GRAPHRAG WITH NVIDIA NIM APIS")
print("=" * 70)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
print(f"✅ Connected to Neo4j at {NEO4J_URI}")

# User query
query_text = "Who founded Bagh?"
print(f"\n❓ Query: {query_text}\n")

try:
    # Step 1: Vector retrieval
    print("📌 Step 1: Retrieving relevant entities via vector search...")
    relevant_entities = retrieve_relevant_entities(driver, query_text, top_k=3)
    print(f"   Retrieved: {relevant_entities}")
    
    # Step 2: Build context from entities and relationships
    print("\n📌 Step 2: Building knowledge graph context...")
    context_parts = []
    for entity in relevant_entities:
        entity_context = get_entity_context(driver, entity)
        context_parts.append(entity_context)
        #print(f"   Entity '{entity}' context:\n{entity_context}\n")
    
    context = "\n\n".join(context_parts)
    print(f"   ✅ Context built with {len(relevant_entities)} entities")
    #print(f"\n🔍 DEBUG - Full context being sent to LLM:\n{context}\n")
    
    # Step 3: Generate answer using LLM with context
    print("\n📌 Step 3: Generating answer using NVIDIA LLM...")
    
    if not context or context.strip() == "":
        print("⚠️ WARNING: Context is empty! Relationships may not exist in Neo4j.")
    
    rag_prompt = f"""You are a knowledge graph Q&A assistant with reasoning capabilities.

TASK: Answer the user's question using the knowledge graph context provided.
- Reason about entity relationships to construct your answer
- If entities are connected, use those connections to infer the answer
- Be specific and reference the relationships you found
- If the exact information is not present, infer from related entities and their relationships

KNOWLEDGE GRAPH CONTEXT:
{context}

USER QUESTION:
{query_text}

ANSWER:"""
    
    answer = get_nvidia_response(rag_prompt)
    
    print(f"\n📝 ANSWER:")
    print(f"{answer}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

driver.close()
print("\n✅ Done!")
