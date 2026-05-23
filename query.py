import os

import requests
from neo4j import GraphDatabase
from neo4j_graphrag.indexes import upsert_vectors
from neo4j_graphrag.types import EntityType

NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "Anjitha@2002")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_EMBED_MODEL = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embedqa-e5-v5")
NVIDIA_EMBED_URL = os.getenv("NVIDIA_EMBED_URL", "https://integrate.api.nvidia.com/v1/embeddings")

if not NVIDIA_API_KEY:
    raise EnvironmentError("Set NVIDIA_API_KEY before running query.py")


def embed_query_with_nvidia(text: str) -> list[float]:
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "model": NVIDIA_EMBED_MODEL,
        "input": text,
        "input_type": "query",
        "encoding_format": "float",
        "truncate": "NONE",
    }
    response = requests.post(NVIDIA_EMBED_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()

    response_json = response.json()
    return response_json["data"][0]["embedding"]


# Connect to the Neo4j database
driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
)

# Generate an embedding for some text
text = (
    "The son of Duke Leto Atreides and the Lady Jessica, Paul is the heir of House "
    "Atreides, an aristocratic family that rules the planet Caladan."
)
vector = embed_query_with_nvidia(text)

# Upsert the vector
upsert_vectors(
    driver,
    ids=["1234"],
    embedding_property="vectorProperty",
    embeddings=[vector],
    entity_type=EntityType.NODE,
)

print("Vector upserted successfully.")

driver.close()
