"""
Diagnostic script to check vector embeddings and search performance.
"""

import os
import requests
from neo4j import GraphDatabase

# ==================== CONFIG ====================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Anjitha@2002"
INDEX_NAME = "vector-index-name"

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
        "input_type": "query",
        "encoding_format": "float",
        "truncate": "NONE"
    }
    
    response = requests.post(NVIDIA_EMBEDDINGS_URL, headers=headers, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

# ==================== MAIN ====================

print("=" * 70)
print("VECTOR EMBEDDING DIAGNOSTIC")
print("=" * 70)

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Step 1: Check entities with embeddings
print("\n📌 Step 1: Entities WITH embeddings in database:")
with driver.session() as session:
    result = session.run("""
        MATCH (n:Entity) 
        WHERE n.embedding IS NOT NULL
        RETURN n.name AS name, size(n.embedding) AS dim
        ORDER BY n.name
    """)
    entities_with_embeddings = list(result)
    for record in entities_with_embeddings:
        print(f"   ✅ {record['name']} ({record['dim']}D)")

# Step 2: Check entities WITHOUT embeddings
print("\n📌 Step 2: Entities WITHOUT embeddings in database:")
with driver.session() as session:
    result = session.run("""
        MATCH (n:Entity) 
        WHERE n.embedding IS NULL
        RETURN n.name AS name
        ORDER BY n.name
    """)
    entities_without_embeddings = list(result)
    if entities_without_embeddings:
        for record in entities_without_embeddings:
            print(f"   ❌ {record['name']}")
    else:
        print("   ✅ All entities have embeddings!")

# Step 3: Test vector search for the query
print("\n📌 Step 3: Testing vector search:")
test_queries = [
    "Who is Hadi Bin Noor",
    "Hadi Bin Noor",
    "Tech Innovators",
    "founder CEO"
]

for query in test_queries:
    print(f"\n   Query: '{query}'")
    try:
        query_embedding = get_nvidia_embedding(query)
        print(f"      Generated {len(query_embedding)}D embedding")
        
        with driver.session() as session:
            result = session.run(f"""
                CALL db.index.vector.queryNodes('{INDEX_NAME}', 10, $embedding)
                YIELD node, score
                RETURN node.name AS entity, score
                ORDER BY score DESC
                LIMIT 5
            """, embedding=query_embedding)
            
            results = list(result)
            if results:
                for record in results:
                    print(f"      - {record['entity']}: {record['score']:.4f}")
            else:
                print("      ⚠️ No results found")
    except Exception as e:
        print(f"      ❌ Error: {e}")

# Step 4: Show all entities and their relationships
print("\n📌 Step 4: All entities in database:")
with driver.session() as session:
    result = session.run("""
        MATCH (n:Entity) 
        RETURN DISTINCT n.name AS name
        ORDER BY n.name
    """)
    all_entities = [record['name'] for record in result]
    for name in all_entities:
        print(f"   - {name}")

driver.close()

print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)
if entities_without_embeddings:
    print(f"⚠️  {len(entities_without_embeddings)} entities are missing embeddings!")
    print("   Run: python recreate_vector_index.py")
else:
    print("✅ All entities have embeddings and should be searchable.")
    print("\nIf search results are still unsatisfactory, try:")
    print("   1. Run: python .\free_kg.py  (to regenerate embeddings)")
    print("   2. Run: python query.py  (to update embeddings)")
    print("   3. Restart Neo4j to clear any caches")
