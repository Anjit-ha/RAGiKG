import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
from openai import OpenAI

# -------- GLOBAL MEMORY --------
GLOBAL_ENTITY_EMB = {}   # name -> embedding
GLOBAL_REL_EMB = {}      # (subj, rel, obj) -> embedding

SIM_THRESHOLD_ENTITY = 0.8
SIM_THRESHOLD_REL = 0.75

import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from neo4j import GraphDatabase

# ==================== NVIDIA CONFIG ====================
def _load_local_env() -> None:
    env_path = Path(".env")
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_local_env()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY not set. In PowerShell, run: "
        "$env:NVIDIA_API_KEY='your_key_here'  or create a .env file with NVIDIA_API_KEY=your_key_here"
    )

# Create OpenAI client pointing to NVIDIA endpoint
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)

NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_CHAT_MODEL = "meta/llama-3.3-70b-instruct"


def _message_content_to_text(content: Any) -> str:
    """Normalize OpenAI/NVIDIA message content into plain text."""
    if content is None:
        raise ValueError("LLM returned empty message content.")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                item_type = item.get("type")
                if item_type in {"text", "output_text"}:
                    parts.append(str(item.get("text", "")))
                elif "content" in item:
                    parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        text = "\n".join(part for part in parts if part.strip())
        if text:
            return text
    return str(content)


def _extract_json_block(text: Any) -> Dict:
    """Extract the first JSON object from an LLM response."""
    cleaned = _message_content_to_text(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    if cleaned.startswith("{"):
        return json.loads(cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("LLM response did not contain valid JSON.")


def _normalize_relation_label(label: str) -> str:
    """Convert a free-form relation into a Neo4j-safe relationship type."""
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip().upper())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return "RELATED_TO"
    if not normalized[0].isalpha():
        normalized = f"R_{normalized}"
    return normalized


def _get_nvidia_embedding(text: str) -> List[float]:
    """Generate embedding for text using NVIDIA NIM API."""
    try:
        # Use direct API call for embeddings (needs input_type parameter)
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Accept": "application/json",
        }
        payload = {
            "model": NVIDIA_EMBEDDING_MODEL,
            "input": text,
            "input_type": "passage",  # Required for asymmetric models
            "encoding_format": "float",
            "truncate": "NONE"
        }
        response = requests.post(
            "https://integrate.api.nvidia.com/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        raise

def extract_graph_data_with_llm(text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    """LLM extraction + incremental entity & relation matching"""

    global GLOBAL_ENTITY_EMB, GLOBAL_REL_EMB

    system_prompt = (
        "Extract a small knowledge graph from the user text. "
        "Return JSON only, with exactly this shape: "
        '{"entities":["..."],"relations":[{"subject":"...","relation":"...","object":"..."}]}. '
        "Entities must be short names copied from the text. "
        "Every relation subject and object must also appear in entities. "
        "Extract at least one relation when the text states any connection, action, description, or event. "
        "Use short factual relation names such as FRIEND_OF, DESCRIBED_AS, SAID, PROTESTED, or SUPPORTED. "
        "Do not include markdown, comments, or text outside the JSON object."
    )

    print(f"  🤖 Calling NVIDIA LLM...")
    try:
        response = nvidia_client.chat.completions.create(
            model=NVIDIA_CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        llm_content = _message_content_to_text(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        raise

    parsed = _extract_json_block(llm_content)

    raw_entities = parsed.get("entities", [])
    raw_relations = parsed.get("relations", [])

    # Normalize entities: if they're dicts, extract the 'name' field; if they're strings, keep as-is
    normalized_entities = []
    for entity in raw_entities:
        if isinstance(entity, dict):
            entity_name = entity.get("name", str(entity))
        else:
            entity_name = str(entity)
        normalized_entities.append(entity_name)
    
    raw_entities = normalized_entities

    # ================= ENTITY MATCHING =================
    final_entities = []

    for entity in raw_entities:
        emb = _get_nvidia_embedding(entity)

        best_match = None
        best_score = 0

        for g_name, g_emb in GLOBAL_ENTITY_EMB.items():
            score = cosine_similarity([emb], [g_emb])[0][0]

            if score > best_score:
                best_score = score
                best_match = g_name

        if best_score > SIM_THRESHOLD_ENTITY:
            final_entities.append(best_match)
        else:
            GLOBAL_ENTITY_EMB[entity] = emb
            final_entities.append(entity)

    # ================= RELATION MATCHING =================
    final_relations = []

    for rel in raw_relations:
        # Handle case where relation might be a list or have non-string fields
        if not isinstance(rel, dict):
            continue
        
        s = str(rel.get("subject") or rel.get("source") or rel.get("from") or "").strip()
        r = str(rel.get("relation") or rel.get("predicate") or rel.get("type") or "").strip()
        o = str(rel.get("object") or rel.get("target") or rel.get("to") or "").strip()
        
        # Normalize the relation label
        r = _normalize_relation_label(r)

        if not s or not r or not o:
            continue

        # map to matched entities
        s = next((e for e in final_entities if e.lower() == s.lower()), s)
        o = next((e for e in final_entities if e.lower() == o.lower()), o)

        rel_text = f"{s} {r} {o}"
        emb = _get_nvidia_embedding(rel_text)

        best_match = None
        best_score = 0

        for g_rel, g_emb in GLOBAL_REL_EMB.items():
            score = cosine_similarity([emb], [g_emb])[0][0]

            if score > best_score:
                best_score = score
                best_match = g_rel

        if best_score > SIM_THRESHOLD_REL:
            final_relations.append(best_match)
        else:
            GLOBAL_REL_EMB[(s, r, o)] = emb
            final_relations.append((s, r, o))

    return final_entities, final_relations


# def extract_graph_data_with_llm(text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
#     """Ask the LLM to extract entities and triples from text."""
#     global invoke_url, model_name, provider_name

#     api_key = os.getenv("NVIDIA_API_KEY")
#     if api_key:
#         invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
#         model_name = "meta/llama-3.2-90b-vision-instruct"
#         provider_name = "nvidia"
#     else:
#         api_key = os.getenv("OPENAI_API_KEY")
#         if api_key:
#             invoke_url = "https://api.openai.com/v1/chat/completions"
#             model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
#             provider_name = "openai"

#     if not api_key:
#         raise EnvironmentError(
#             "Set NVIDIA_API_KEY or OPENAI_API_KEY before running this script."
#         )

#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Accept": "text/event-stream" if stream else "application/json",
#     }

#     system_prompt = (
#         "Extract a small knowledge graph from the user text. "
#         "First, extract a list of key entities (persons, organizations, concepts, events). "
#         "Then, extract ONLY relations that connect two of those entities together. "
#         "Do NOT create relations to entities outside your extracted entity list. "
#         "Return JSON only with this shape: "
#         '{"entities": ["..."], "relations": [{"subject": "...", "relation": "...", "object": "..."}]}. '
#         "Use short, factual relation names. Both subject and object MUST be in your entities list. "
#         "Do not include markdown or commentary."
#     )

#     payload = {
#         "model": model_name,
#         "messages": [
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": text},
#         ],
#         "max_tokens": 512,
#         "temperature": 0.0,
#         "top_p": 1.0,
#         "frequency_penalty": 0.0,
#         "presence_penalty": 0.0,
#         "stream": stream,
#     }

#     response = requests.post(invoke_url, headers=headers, json=payload, timeout=120)
#     response.raise_for_status()

#     response_data = response.json()
#     llm_content = response_data["choices"][0]["message"]["content"]
#     parsed = _extract_json_block(llm_content)

#     entities = parsed.get("entities", [])
#     relations_raw = parsed.get("relations", [])
#     relations: List[Tuple[str, str, str]] = []

#     # Normalize entity names for matching
#     entity_set = {str(e).strip().lower() for e in entities}

#     for relation in relations_raw:
#         subject = str(relation.get("subject", "")).strip()
#         rel = str(relation.get("relation", "")).strip()
#         obj = str(relation.get("object", "")).strip()
#         if subject and rel and obj:
#             # Only keep relations where both subject and object are in the entity list
#             if subject.lower() in entity_set and obj.lower() in entity_set:
#                 relations.append((subject, _normalize_relation_label(rel), obj))

#     return entities, relations

# ---------------- NEO4J CONNECTION ----------------
uri = "bolt://localhost:7687"
username = "neo4j"
password = "Anjitha@2002"   # 🔴 replace if needed

driver = GraphDatabase.driver(uri, auth=(username, password))

# ---------------- PUSH TO NEO4J ----------------
def create_graph(tx, relations):
    for subj, rel, obj in relations:
        query = f"""
        MERGE (a:Entity {{name: $subj}})
        MERGE (b:Entity {{name: $obj}})
        MERGE (a)-[r:{rel}]->(b)
        ON CREATE SET r.relation = $rel
        ON MATCH SET r.relation = $rel
        """
        tx.run(query, subj=subj, obj=obj, rel=rel)


def generate_and_store_embeddings(driver, entities: List[str]):
    """Generate embeddings for entities and store them in Neo4j."""
    print(f"\n🔄 Generating embeddings for {len(entities)} entities...")
    
    with driver.session() as session:
        for i, entity_name in enumerate(entities, 1):
            try:
                print(f"   [{i}/{len(entities)}] Embedding '{entity_name}'...", end=" ", flush=True)
                embedding = _get_nvidia_embedding(entity_name)
                
                # Store embedding in Neo4j
                session.run(
                    "MATCH (n:Entity {name: $name}) SET n.embedding = $embedding",
                    name=entity_name,
                    embedding=embedding
                )
                print(f"✅ ({len(embedding)}D)")
            except Exception as e:
                print(f"❌ Error: {e}")


def fetch_created_relations(tx):
    query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.name AS subject, type(r) AS relation_type, r.relation AS relation_name, b.name AS object
    ORDER BY subject, relation_type, object
    """
    return list(tx.run(query))


def process_text(text: str, source_name: str) -> None:
    """Extract graph data from one text file and persist its updated graph snapshot."""
    print(f"\n📄 Processing: {source_name}")
    print(f"   Characters: {len(text)}")

    entities, relations = extract_graph_data_with_llm(text)

    print("\n✅ Entities:")
    for entity in entities:
        print(entity)

    print("\n✅ Relations:")
    for relation in relations:
        print(relation)

    if not relations:
        print("\n⚠️ No relations to push")
        return

    with driver.session() as session:
        session.execute_write(create_graph, relations)
        created_relations = session.execute_read(fetch_created_relations)

    print("\n✅ Data pushed to Neo4j with proper relationship labels!")
    print("\n✅ Neo4j relationships:")
    for record in created_relations:
        print((record["subject"], record["relation_type"], record["relation_name"], record["object"]))

    all_entities = list(set(entities + [rel[0] for rel in relations] + [rel[2] for rel in relations]))
    generate_and_store_embeddings(driver, all_entities)

    graph_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_name).stem).strip("_") or "knowledge_graph"
    graph_image = export_graph_visualization(driver, graph_name=graph_slug)
    if graph_image:
        print(f"\n✅ Graph visualization saved to {graph_image}")


def process_directory(input_directory: Path, watch_for_new_files: bool = True, poll_interval: int = 5) -> None:
    """Process each .txt file in a directory in order and keep watching for new files."""
    seen_files = set()

    print(f"\n📁 Watching directory: {input_directory}")
    print("   Files are processed one by one in sorted order.")
    print("   A graph snapshot is saved for every file.")
    print("   Press Ctrl+C to stop.\n")

    while True:
        text_files = sorted(
            [path for path in input_directory.iterdir() if path.is_file() and path.suffix.lower() == ".txt"],
            key=lambda path: path.name.lower(),
        )
        new_files = [path for path in text_files if path not in seen_files]

        if not new_files:
            if watch_for_new_files:
                time.sleep(poll_interval)
                continue
            break

        for file_path in new_files:
            try:
                with file_path.open("r", encoding="utf-8") as handle:
                    text = handle.read()
                process_text(text, file_path.name)
                seen_files.add(file_path)
            except Exception as error:
                print(f"\n❌ Failed to process {file_path.name}: {error}")

        if not watch_for_new_files:
            break


def export_graph_visualization(driver, output_dir: str = "docs/graph_visualizations", graph_name: str = "knowledge_graph") -> str:
    """Export the current Neo4j entity graph to PNG images."""
    os.makedirs(output_dir, exist_ok=True)

    graph = nx.DiGraph()

    with driver.session() as session:
        node_result = session.run("""
        MATCH (n:Entity)
        RETURN DISTINCT n.name AS name
        ORDER BY name
        """)
        for record in node_result:
            if record["name"]:
                graph.add_node(record["name"])

        edge_result = session.run("""
        MATCH (a:Entity)-[r]->(b:Entity)
        RETURN a.name AS source, type(r) AS relation_type, coalesce(r.relation, type(r)) AS relation_name, b.name AS target
        ORDER BY source, relation_type, target
        """)
        for record in edge_result:
            source = record["source"]
            target = record["target"]
            relation_name = record["relation_name"] or record["relation_type"]
            if source and target:
                graph.add_edge(source, target, label=relation_name)

    if graph.number_of_nodes() == 0:
        print("⚠️ No graph data found to visualize")
        return ""

    plt.figure(figsize=(16, 11))
    positions = nx.spring_layout(graph, seed=42, k=1.0 / max(1, (graph.number_of_nodes() ** 0.5)))
    degrees = dict(graph.degree())
    node_sizes = [1800 + degrees[node] * 250 for node in graph.nodes()]

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes,
        node_color="#8ecae6",
        edgecolors="#1d3557",
        linewidths=1.2,
        alpha=0.95,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        edge_color="#6c757d",
        width=1.4,
        connectionstyle="arc3,rad=0.08",
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=9,
        font_weight="bold",
        font_color="#0b1f33",
    )

    edge_labels = nx.get_edge_attributes(graph, "label")
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.5,
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#b0bec5", alpha=0.9),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = os.path.join(output_dir, f"{graph_name}_latest.png")
    snapshot_path = os.path.join(output_dir, f"{graph_name}_{timestamp}.png")

    plt.title(f"Knowledge Graph Snapshot: {graph_name}", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(latest_path, dpi=180, bbox_inches="tight")
    plt.savefig(snapshot_path, dpi=180, bbox_inches="tight")
    plt.close()

    return snapshot_path

input_directory = Path("text_data")
if not input_directory.exists():
    print(f"❌ Input directory not found: {input_directory}")
    driver.close()
    raise SystemExit(1)

if __name__ == "__main__":
    try:
        process_directory(input_directory, watch_for_new_files=True, poll_interval=5)
    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user.")
    finally:
        driver.close()
