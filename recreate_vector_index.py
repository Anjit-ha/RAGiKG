"""
Recreate Neo4j vector index with correct dimensions for NVIDIA embeddings.
The old index was 3072D (OpenAI), we need 1024D (NVIDIA).
"""

import os
import requests
from neo4j import GraphDatabase

# ==================== CONFIG ====================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Anjitha@2002"
INDEX_NAME = "vector-index-name"
VECTOR_DIMENSION = 1024  # NVIDIA embedding dimension

# NVIDIA API
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise EnvironmentError("Set NVIDIA_API_KEY environment variable")

NVIDIA_EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"

# ==================== HELPER FUNCTIONS ====================

def get_nvidia_embedding(text: str) -> list:
    """Get embedding from NVIDIA NIM API."""
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "model": NVIDIA_EMBEDDING_MODEL,
        "input": text,
        "input_type": "passage",
        "encoding_format": "float",
        "truncate": "NONE"
    }
    
    response = requests.post(NVIDIA_EMBEDDINGS_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def drop_vector_index(tx):
    """Drop the old vector index if it exists."""
    query = f"DROP INDEX `{INDEX_NAME}` IF EXISTS"
    tx.run(query)
    print(f"✅ Dropped old index '{INDEX_NAME}'")

def create_vector_index(tx):
    """Create a new vector index with correct dimensions."""
    query = f"""
    CREATE VECTOR INDEX `{INDEX_NAME}` FOR (n:Entity)
    ON (n.embedding)
    OPTIONS {{
        indexConfig: {{
            `vector.dimensions`: {VECTOR_DIMENSION},
            `vector.similarity_function`: 'cosine'
        }}
    }}
    """
    tx.run(query)
    print(f"✅ Created new vector index '{INDEX_NAME}' with {VECTOR_DIMENSION} dimensions")

def get_entities(tx) -> list:
    """Get all entities from the graph."""
    query = "MATCH (n:Entity) RETURN n.name AS name"
    result = tx.run(query)
    return [record["name"] for record in result]

def upsert_embeddings(tx, entity_name: str, embedding: list):
    """Add or update embedding for an entity."""
    query = """
    MATCH (n:Entity {name: $name})
    SET n.embedding = $embedding
    """
    tx.run(query, name=entity_name, embedding=embedding)

# ==================== MAIN ====================

print("=" * 60)
print("RECREATING VECTOR INDEX FOR NVIDIA EMBEDDINGS")
print("=" * 60)

# Connect to Neo4j
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
print(f"\n✅ Connected to Neo4j at {NEO4J_URI}")

# Step 1: Drop old index
print("\n📌 Step 1: Dropping old vector index...")
with driver.session() as session:
    session.execute_write(drop_vector_index)

# Step 2: Create new index
print("\n📌 Step 2: Creating new vector index...")
with driver.session() as session:
    session.execute_write(create_vector_index)

# Step 3: Get all entities
print("\n📌 Step 3: Fetching entities from knowledge graph...")
with driver.session() as session:
    entities = session.execute_read(get_entities)

print(f"   Found {len(entities)} entities:")
for e in entities:
    print(f"      - {e}")

# Step 4: Generate embeddings and populate index
print(f"\n📌 Step 4: Generating embeddings using NVIDIA API...")
print(f"   Model: {NVIDIA_EMBEDDING_MODEL}")

with driver.session() as session:
    for i, entity_name in enumerate(entities, 1):
        try:
            print(f"   [{i}/{len(entities)}] Embedding '{entity_name}'...", end=" ", flush=True)
            embedding = get_nvidia_embedding(entity_name)
            session.execute_write(upsert_embeddings, entity_name, embedding)
            print(f"✅ ({len(embedding)}D)")
        except Exception as e:
            print(f"❌ Error: {e}")

# Step 5: Verify the index
print(f"\n📌 Step 5: Verifying vector index...")
with driver.session() as session:
    result = session.run(f"SHOW INDEXES WHERE name = '{INDEX_NAME}'")
    indexes = list(result)
    if indexes:
        idx = indexes[0]
        print(f"   ✅ Index exists: {idx['name']}")
        print(f"      Type: {idx.get('type', 'unknown')}")
        print(f"      State: {idx.get('state', 'unknown')}")
    else:
        print(f"   ⚠️  Index not found")

# Step 6: Test vector search
print(f"\n📌 Step 6: Testing vector search...")
try:
    test_query = "What is mathematics?"
    test_embedding = get_nvidia_embedding(test_query)
    print(f"   Generated query embedding: {len(test_embedding)}D")
    
    with driver.session() as session:
        result = session.run(f"""
            CALL db.index.vector.queryNodes('{INDEX_NAME}', 3, $embedding)
            YIELD node, score
            RETURN node.name AS entity, score
            ORDER BY score DESC
            LIMIT 5
        """, embedding=test_embedding)
        
        results = list(result)
        if results:
            print(f"   ✅ Vector search working! Top results:")
            for record in results:
                print(f"      - {record['entity']}: {record['score']:.4f}")
        else:
            print(f"   ⚠️  No results from vector search")
except Exception as e:
    print(f"   ❌ Vector search error: {e}")

driver.close()

print("\n" + "=" * 60)
print("✅ VECTOR INDEX RECREATION COMPLETE!")
print("=" * 60)
print(f"\nYou can now run rag_query.py successfully:")
print(f"  $env:NVIDIA_API_KEY='...'")
print(f"  ..\venv310\Scripts\python.exe rag_query.py")
