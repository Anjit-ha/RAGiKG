"""
Process text files one by one:

1. Extract a knowledge graph from each .txt file and store it in Neo4j.
2. Store NVIDIA embeddings for graph entities.
3. Run the simple GraphRAG query flow against the updated graph.
4. Continue until every .txt file in the input folder is processed.

This combines the main behavior from free_kg.py and graphrag_simple.py without
changing either original file.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
import networkx as nx
import requests
from neo4j import GraphDatabase
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from pyvis.network import Network
except ImportError:
    Network = None


# ==================== CONFIG ====================
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIRECTORY = BASE_DIR / "text_data"
WATCH_FOR_NEW_FILES = False
POLL_INTERVAL_SECONDS = 5

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "Anjitha@2002"
INDEX_NAME = "vector-index-name"
VECTOR_DIMENSION = 1024

NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_CHAT_MODEL = "meta/llama-3.3-70b-instruct"
NVIDIA_EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"

SIM_THRESHOLD_ENTITY = 0.8
SIM_THRESHOLD_REL = 0.75

GLOBAL_ENTITY_EMB: Dict[str, List[float]] = {}
GLOBAL_REL_EMB: Dict[Tuple[str, str, str], List[float]] = {}


# ==================== NVIDIA / ENV ====================
def load_local_env() -> None:
    env_path = BASE_DIR / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise EnvironmentError(
        "NVIDIA_API_KEY not set. In PowerShell, run: "
        "$env:NVIDIA_API_KEY='your_key_here' or create .env with NVIDIA_API_KEY=your_key_here"
    )

nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)


def message_content_to_text(content: Any) -> str:
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


def extract_json_block(text: Any) -> Dict:
    cleaned = message_content_to_text(text).strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    if cleaned.startswith("{"):
        return json.loads(cleaned)

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    raise ValueError("LLM response did not contain valid JSON.")


def normalize_relation_label(label: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", label.strip().upper())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    if not normalized:
        return "RELATED_TO"
    if not normalized[0].isalpha():
        normalized = f"R_{normalized}"
    return normalized


def get_nvidia_embedding(text: str, input_type: str) -> List[float]:
    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Accept": "application/json",
    }
    payload = {
        "model": NVIDIA_EMBEDDING_MODEL,
        "input": text,
        "input_type": input_type,
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
    response = nvidia_client.chat.completions.create(
        model=NVIDIA_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0.0,
    )
    return message_content_to_text(response.choices[0].message.content)


# ==================== KG EXTRACTION ====================
def extract_graph_data_with_llm(text: str) -> Tuple[List[str], List[Tuple[str, str, str]]]:
    global GLOBAL_ENTITY_EMB, GLOBAL_REL_EMB

    system_prompt = (
        "Extract a comprehensive knowledge graph from the user text. "
        "Return JSON only, with exactly this shape: "
        '{"entities":["..."],"relations":[{"subject":"...","relation":"...","object":"..."}]}. '
        "Entities must be short names copied from the text. Include all important people, places, "
        "organizations, events, concepts, interests, descriptions, and attributes mentioned in the text. "
        "Every relation subject and object must also appear in entities. "
        "Extract every clear factual relation, including friendships, locations, activities, interests, "
        "descriptions, preferences, actions, and group membership. "
        "When a sentence says several people were friends, create pairwise FRIEND_OF relations for every pair. "
        "For example, if A, B, and C were friends, include A FRIEND_OF B, A FRIEND_OF C, and B FRIEND_OF C. "
        "Use short factual relation names such as FRIEND_OF, STUDIED_IN, INTERESTED_IN, LIKED, "
        "DESCRIBED_AS, SAID, PROTESTED, or SUPPORTED. "
        "Do not include markdown, comments, or text outside the JSON object. "
        "Do not repeat duplicate relations."
    )

    print("  Calling NVIDIA LLM for KG extraction...")
    response = nvidia_client.chat.completions.create(
        model=NVIDIA_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    parsed = extract_json_block(response.choices[0].message.content)

    raw_entities = parsed.get("entities", [])
    raw_relations = parsed.get("relations", [])

    # -------- ENTITY NORMALIZATION --------
    normalized_entities = []
    for entity in raw_entities:
        if isinstance(entity, dict):
            name = str(entity.get("name", entity)).strip()
        else:
            name = str(entity).strip()

        if name:
            name = name.title()
            normalized_entities.append(name)

    # -------- ENTITY MATCHING --------
    final_entities = []
    for entity in normalized_entities:
        emb = get_nvidia_embedding(entity, input_type="passage")
        best_match = None
        best_score = 0.0

        for g_name, g_emb in GLOBAL_ENTITY_EMB.items():
            score = cosine_similarity([emb], [g_emb])[0][0]
            if score > best_score:
                best_score = score
                best_match = g_name

        if best_match and best_score > SIM_THRESHOLD_ENTITY:
            final_entities.append(best_match)
        else:
            GLOBAL_ENTITY_EMB[entity] = emb
            final_entities.append(entity)

    # Deduplicate entities
    final_entities = list(dict.fromkeys(final_entities))

    # -------- RELATION MATCHING --------
    relation_set = set()
    final_relations = []

    for rel in raw_relations:
        if not isinstance(rel, dict):
            continue

        subject = str(rel.get("subject") or rel.get("source") or rel.get("from") or "").strip().title()
        relation = normalize_relation_label(
            str(rel.get("relation") or rel.get("predicate") or rel.get("type") or "").strip()
        )
        obj = str(rel.get("object") or rel.get("target") or rel.get("to") or "").strip().title()

        if not subject or not relation or not obj:
            continue

        subject = next((e for e in final_entities if e.lower() == subject.lower()), subject)
        obj = next((e for e in final_entities if e.lower() == obj.lower()), obj)

        relation_tuple = (subject, relation, obj)
        if relation_tuple not in GLOBAL_REL_EMB:
            rel_text = f"{subject} {relation} {obj}"
            GLOBAL_REL_EMB[relation_tuple] = get_nvidia_embedding(rel_text, input_type="passage")

        # -------- REMOVE DUPLICATES --------
        if relation_tuple not in relation_set:
            relation_set.add(relation_tuple)
            final_relations.append(relation_tuple)

    return final_entities, final_relations


# ==================== NEO4J KG STORAGE ====================
def create_entities(tx, entities: List[str]) -> None:
    for entity in entities:
        tx.run("MERGE (:Entity {name: $name})", name=entity)


def create_graph(tx, relations: List[Tuple[str, str, str]]) -> None:
    for subj, rel, obj in relations:
        query = f"""
        MERGE (a:Entity {{name: $subj}})
        MERGE (b:Entity {{name: $obj}})
        MERGE (a)-[r:{rel}]->(b)
        ON CREATE SET r.relation = $rel
        ON MATCH SET r.relation = $rel
        """
        tx.run(query, subj=subj, obj=obj, rel=rel)


def fetch_created_relations(tx) -> list:
    query = """
    MATCH (a:Entity)-[r]->(b:Entity)
    RETURN a.name AS subject, type(r) AS relation_type, r.relation AS relation_name, b.name AS object
    ORDER BY subject, relation_type, object
    """
    return list(tx.run(query))


def generate_and_store_embeddings(driver, entities: List[str]) -> None:
    print(f"\nGenerating embeddings for {len(entities)} entities...")
    with driver.session() as session:
        for i, entity_name in enumerate(entities, 1):
            try:
                print(f"   [{i}/{len(entities)}] Embedding '{entity_name}'...", end=" ", flush=True)
                embedding = get_nvidia_embedding(entity_name, input_type="passage")
                session.run(
                    "MATCH (n:Entity {name: $name}) SET n.embedding = $embedding",
                    name=entity_name,
                    embedding=embedding,
                )
                print(f"OK ({len(embedding)}D)")
            except Exception as error:
                print(f"ERROR: {error}")


def ensure_vector_index(driver) -> None:
    with driver.session() as session:
        result = session.run("SHOW INDEXES YIELD name WHERE name = $name RETURN name", name=INDEX_NAME)
        if list(result):
            return

        print(f"\nCreating Neo4j vector index '{INDEX_NAME}'...")
        session.run(
            f"""
            CREATE VECTOR INDEX `{INDEX_NAME}` FOR (n:Entity)
            ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {VECTOR_DIMENSION},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )


def export_graph_visualization(
    driver,
    output_dir: Path = BASE_DIR / "text_data" / "graph",
    graph_name: str = "knowledge_graph",
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = nx.DiGraph()

    with driver.session() as session:
        node_result = session.run(
            """
            MATCH (n:Entity)
            RETURN DISTINCT n.name AS name
            ORDER BY name
            """
        )
        for record in node_result:
            if record["name"]:
                graph.add_node(record["name"])

        edge_result = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            RETURN a.name AS source, type(r) AS relation_type,
                   coalesce(r.relation, type(r)) AS relation_name, b.name AS target
            ORDER BY source, relation_type, target
            """
        )
        for record in edge_result:
            if record["source"] and record["target"]:
                graph.add_edge(record["source"], record["target"], label=record["relation_name"])

    if graph.number_of_nodes() == 0:
        print("No graph data found to visualize.")
        return ""

    plt.figure(figsize=(16, 11))
    positions = nx.spring_layout(graph, seed=42, k=1.0 / max(1, graph.number_of_nodes() ** 0.5))
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
    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=nx.get_edge_attributes(graph, "label"),
        font_size=8,
        label_pos=0.5,
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#b0bec5", alpha=0.9),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latest_path = output_dir / f"{graph_name}_latest.png"
    snapshot_path = output_dir / f"{graph_name}_{timestamp}.png"
    latest_html_path = output_dir / f"{graph_name}_latest.html"
    snapshot_html_path = output_dir / f"{graph_name}_{timestamp}.html"

    plt.title(f"Knowledge Graph Snapshot: {graph_name}", fontsize=16, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(latest_path, dpi=180, bbox_inches="tight")
    plt.savefig(snapshot_path, dpi=180, bbox_inches="tight")
    plt.close()

    graph_payload = {
        "nodes": [
            {
                "id": node,
                "label": node,
                "degree": degrees.get(node, 0),
            }
            for node in graph.nodes()
        ],
        "edges": [
            {
                "source": source,
                "target": target,
                "label": data.get("label", ""),
            }
            for source, target, data in graph.edges(data=True)
        ],
    }
    interactive_html = build_interactive_graph_html(graph_name, graph_payload)
    latest_html_path.write_text(interactive_html, encoding="utf-8")
    snapshot_html_path.write_text(interactive_html, encoding="utf-8")

    return str(snapshot_path)


def build_interactive_graph_html(graph_name: str, graph_payload: Dict[str, Any]) -> str:
    payload_json = json.dumps(graph_payload)
    title = json.dumps(f"Knowledge Graph: {graph_name}")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{graph_name}</title>
<style>
html, body {{
    height: 100%;
    margin: 0;
    background: #ffffff;
    color: #0b1f33;
    font-family: Arial, sans-serif;
}}
#toolbar {{
    height: 42px;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0 12px;
    border-bottom: 1px solid #d7dee8;
    box-sizing: border-box;
}}
#title {{
    font-size: 15px;
    font-weight: 700;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}}
button {{
    border: 1px solid #b8c4d4;
    background: #f7fafc;
    color: #0b1f33;
    border-radius: 6px;
    padding: 6px 10px;
    cursor: pointer;
}}
#graph {{
    width: 100%;
    height: calc(100% - 42px);
    cursor: grab;
    touch-action: none;
}}
#graph:active {{
    cursor: grabbing;
}}
.edge {{
    stroke: #7b8794;
    stroke-width: 1.6;
    fill: none;
}}
.edge-label {{
    fill: #4b5563;
    font-size: 10px;
    paint-order: stroke;
    stroke: #ffffff;
    stroke-width: 4px;
    stroke-linejoin: round;
}}
.node circle {{
    fill: #8ecae6;
    stroke: #1d3557;
    stroke-width: 1.6;
}}
.node text {{
    fill: #0b1f33;
    font-size: 12px;
    font-weight: 700;
    text-anchor: middle;
    dominant-baseline: middle;
    pointer-events: none;
}}
</style>
</head>
<body>
<div id="toolbar">
  <div id="title"></div>
  <button id="fit" type="button">Fit</button>
  <button id="pause" type="button">Pause</button>
</div>
<svg id="graph" role="img" aria-label="Interactive knowledge graph">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="3" orient="auto" markerUnits="strokeWidth">
      <path d="M0,0 L0,6 L9,3 z" fill="#7b8794"></path>
    </marker>
  </defs>
  <g id="viewport">
    <g id="edges"></g>
    <g id="edgeLabels"></g>
    <g id="nodes"></g>
  </g>
</svg>
<script>
const payload = {payload_json};
const graphTitle = {title};
const svg = document.getElementById("graph");
const viewport = document.getElementById("viewport");
const edgesGroup = document.getElementById("edges");
const edgeLabelsGroup = document.getElementById("edgeLabels");
const nodesGroup = document.getElementById("nodes");
document.getElementById("title").textContent = graphTitle + "  Drag nodes. Wheel to zoom. Drag background to pan.";

const width = () => svg.clientWidth || 1200;
const height = () => svg.clientHeight || 720;
let transform = {{x: 0, y: 0, k: 1}};
let paused = false;
let draggingNode = null;
let panning = null;

const nodes = payload.nodes.map((node, index) => {{
  const angle = (index / Math.max(1, payload.nodes.length)) * Math.PI * 2;
  return {{
    ...node,
    x: width() / 2 + Math.cos(angle) * Math.min(width(), height()) * 0.32,
    y: height() / 2 + Math.sin(angle) * Math.min(width(), height()) * 0.32,
    vx: 0,
    vy: 0,
    r: 24 + Math.min((node.degree || 0) * 3, 16)
  }};
}});
const nodeById = new Map(nodes.map(node => [node.id, node]));
const edges = payload.edges
  .map(edge => ({{...edge, sourceNode: nodeById.get(edge.source), targetNode: nodeById.get(edge.target)}}))
  .filter(edge => edge.sourceNode && edge.targetNode);

function setViewport() {{
  viewport.setAttribute("transform", `translate(${{transform.x}} ${{transform.y}}) scale(${{transform.k}})`);
}}

function screenToGraph(event) {{
  const rect = svg.getBoundingClientRect();
  return {{
    x: (event.clientX - rect.left - transform.x) / transform.k,
    y: (event.clientY - rect.top - transform.y) / transform.k
  }};
}}

function makeSvg(name, attrs = {{}}) {{
  const el = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
  return el;
}}

const edgeEls = edges.map(edge => {{
  const line = makeSvg("line", {{"class": "edge", "marker-end": "url(#arrow)"}});
  edgesGroup.appendChild(line);
  return line;
}});
const edgeLabelEls = edges.map(edge => {{
  const text = makeSvg("text", {{"class": "edge-label"}});
  text.textContent = edge.label || "";
  edgeLabelsGroup.appendChild(text);
  return text;
}});
const nodeEls = nodes.map(node => {{
  const group = makeSvg("g", {{"class": "node"}});
  const circle = makeSvg("circle", {{r: node.r}});
  const text = makeSvg("text");
  text.textContent = node.label;
  group.append(circle, text);
  group.addEventListener("pointerdown", event => {{
    draggingNode = node;
    draggingNode.fixed = true;
    svg.setPointerCapture(event.pointerId);
    event.stopPropagation();
  }});
  nodesGroup.appendChild(group);
  return group;
}});

function tick() {{
  if (!paused && !draggingNode) {{
    for (let i = 0; i < nodes.length; i++) {{
      for (let j = i + 1; j < nodes.length; j++) {{
        const a = nodes[i], b = nodes[j];
        let dx = b.x - a.x, dy = b.y - a.y;
        let dist2 = dx * dx + dy * dy || 1;
        let force = Math.min(2200 / dist2, 2.5);
        let dist = Math.sqrt(dist2);
        dx /= dist; dy /= dist;
        a.vx -= dx * force; a.vy -= dy * force;
        b.vx += dx * force; b.vy += dy * force;
      }}
    }}
    for (const edge of edges) {{
      const a = edge.sourceNode, b = edge.targetNode;
      let dx = b.x - a.x, dy = b.y - a.y;
      let dist = Math.sqrt(dx * dx + dy * dy) || 1;
      let force = (dist - 170) * 0.012;
      dx /= dist; dy /= dist;
      a.vx += dx * force; a.vy += dy * force;
      b.vx -= dx * force; b.vy -= dy * force;
    }}
    for (const node of nodes) {{
      node.vx += (width() / 2 - node.x) * 0.0008;
      node.vy += (height() / 2 - node.y) * 0.0008;
      node.vx *= 0.86;
      node.vy *= 0.86;
      node.x += node.vx;
      node.y += node.vy;
    }}
  }}
  render();
  requestAnimationFrame(tick);
}}

function render() {{
  edges.forEach((edge, index) => {{
    const a = edge.sourceNode, b = edge.targetNode;
    const line = edgeEls[index];
    line.setAttribute("x1", a.x);
    line.setAttribute("y1", a.y);
    line.setAttribute("x2", b.x);
    line.setAttribute("y2", b.y);
    const label = edgeLabelEls[index];
    label.setAttribute("x", (a.x + b.x) / 2);
    label.setAttribute("y", (a.y + b.y) / 2 - 6);
  }});
  nodes.forEach((node, index) => {{
    nodeEls[index].setAttribute("transform", `translate(${{node.x}} ${{node.y}})`);
  }});
}}

svg.addEventListener("pointerdown", event => {{
  panning = {{x: event.clientX, y: event.clientY, tx: transform.x, ty: transform.y}};
  svg.setPointerCapture(event.pointerId);
}});
svg.addEventListener("pointermove", event => {{
  if (draggingNode) {{
    const pos = screenToGraph(event);
    draggingNode.x = pos.x;
    draggingNode.y = pos.y;
    draggingNode.vx = 0;
    draggingNode.vy = 0;
  }} else if (panning) {{
    transform.x = panning.tx + event.clientX - panning.x;
    transform.y = panning.ty + event.clientY - panning.y;
    setViewport();
  }}
}});
svg.addEventListener("pointerup", event => {{
  if (draggingNode) draggingNode.fixed = false;
  draggingNode = null;
  panning = null;
  try {{ svg.releasePointerCapture(event.pointerId); }} catch (error) {{}}
}});
svg.addEventListener("wheel", event => {{
  event.preventDefault();
  const rect = svg.getBoundingClientRect();
  const mx = event.clientX - rect.left;
  const my = event.clientY - rect.top;
  const nextK = Math.max(0.25, Math.min(4, transform.k * (event.deltaY < 0 ? 1.1 : 0.9)));
  transform.x = mx - ((mx - transform.x) / transform.k) * nextK;
  transform.y = my - ((my - transform.y) / transform.k) * nextK;
  transform.k = nextK;
  setViewport();
}}, {{passive: false}});
document.getElementById("pause").addEventListener("click", () => {{
  paused = !paused;
  document.getElementById("pause").textContent = paused ? "Resume" : "Pause";
}});
document.getElementById("fit").addEventListener("click", () => {{
  transform = {{x: 0, y: 0, k: 1}};
  setViewport();
}});

setViewport();
tick();
</script>
</body>
</html>
"""


def process_text_file(driver, file_path: Path) -> bool:
    print("\n" + "=" * 70)
    print(f"Processing KG input file: {file_path.name}")
    print("=" * 70)

    text = file_path.read_text(encoding="utf-8")
    print(f"Characters: {len(text)}")

    entities, relations = extract_graph_data_with_llm(text)

    print("\nEntities:")
    for entity in entities:
        print(f"  - {entity}")

    print("\nRelations:")
    for relation in relations:
        print(f"  - {relation}")

    with driver.session() as session:
        session.execute_write(create_entities, entities)
        session.execute_write(create_graph, relations)
        created_relations = session.execute_read(fetch_created_relations)

    if not relations:
        print("\nNo relations extracted, but standalone entities were pushed to Neo4j.")
    else:
        print("\nData pushed to Neo4j.")
    print(f"Total Neo4j relationships now: {len(created_relations)}")

    all_entities = list(dict.fromkeys(entities + [rel[0] for rel in relations] + [rel[2] for rel in relations]))
    generate_and_store_embeddings(driver, all_entities)
    ensure_vector_index(driver)

    graph_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", file_path.stem).strip("_") or "knowledge_graph"
    graph_image = export_graph_visualization(driver, graph_name=graph_slug)
    if graph_image:
        print(f"Graph visualization saved to {graph_image}")
        print(
            "Interactive graph saved to "
            f"{BASE_DIR / 'text_data' / 'graph' / f'{graph_slug}_latest.html'}"
        )

    return True


# ==================== GRAPHRAG SIMPLE FLOW ====================
def get_entity_context(driver, entity_name: str) -> str:
    with driver.session() as session:
        out_results = session.run(
            """
            MATCH (a:Entity {name: $name})-[r]->(b:Entity)
            RETURN type(r) AS rel_type, b.name AS target
            """,
            name=entity_name,
        )
        in_results = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity {name: $name})
            RETURN type(r) AS rel_type, a.name AS source
            """,
            name=entity_name,
        )

        lines = [f"Entity: {entity_name}"]
        for record in out_results:
            rel = record["rel_type"].replace("_", " ").lower()
            lines.append(f"  - {entity_name} {rel} {record['target']}")
        for record in in_results:
            rel = record["rel_type"].replace("_", " ").lower()
            lines.append(f"  - {record['source']} {rel} {entity_name}")

        return "\n".join(lines)


def retrieve_relevant_entities(driver, query: str, top_k: int = 3) -> List[str]:
    query_embedding = get_nvidia_embedding(query, input_type="query")
    with driver.session() as session:
        result = session.run(
            f"""
            CALL db.index.vector.queryNodes('{INDEX_NAME}', $k, $embedding)
            YIELD node, score
            RETURN node.name AS entity, score
            ORDER BY score DESC
            LIMIT $k
            """,
            embedding=query_embedding,
            k=top_k,
        )
        return [record["entity"] for record in result]


def find_entities_mentioned_in_query(driver, query: str) -> List[str]:
    """Return graph entities whose names are directly mentioned in the user question."""
    query_lower = query.lower()
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n:Entity)
            RETURN n.name AS entity
            ORDER BY size(n.name) DESC
            """
        )
        return [
            record["entity"]
            for record in result
            if record["entity"] and record["entity"].lower() in query_lower
        ]


def run_graphrag_query(driver, query_text: str, source_name: str) -> None:
    print("\n" + "-" * 70)
    print(f"Running GraphRAG after file: {source_name}")
    print(f"Query: {query_text}")
    print("-" * 70)

    try:
        answer, relevant_entities, _context = answer_graphrag_query(driver, query_text)
        print(f"Retrieved entities: {relevant_entities}")
        print("\nGraphRAG answer:")
        print(answer)
    except Exception as error:
        print(f"GraphRAG failed after {source_name}: {error}")


def answer_graphrag_query(driver, query_text: str) -> Tuple[str, List[str], str]:
    """Return a concise GraphRAG answer plus retrieved entities and context."""
    mentioned_entities = find_entities_mentioned_in_query(driver, query_text)
    vector_entities = retrieve_relevant_entities(driver, query_text, top_k=5)
    relevant_entities = list(dict.fromkeys(mentioned_entities + vector_entities))

    context_parts = [get_entity_context(driver, entity) for entity in relevant_entities]
    context = "\n\n".join(part for part in context_parts if part.strip())

    if not context.strip():
        return "Not found in the graph.", relevant_entities, context

    rag_prompt = f"""You are a concise knowledge graph Q&A assistant.

TASK: Answer the user's question using the knowledge graph context provided.
- Give only the direct answer.
- Be concise, but include all direct relevant facts from the graph for the asked entity.
- Do not explain extra background.
- If the question asks who/what an entity is, include its direct friendships, interests, studies, locations, likes, roles, and other direct relations when present.
- If the question asks for a relation between two entities, answer only that relation.
- Do not summarize unrelated relationships.
- If the graph does not contain the answer, say: Not found in the graph.

KNOWLEDGE GRAPH CONTEXT:
{context}

USER QUESTION:
{query_text}

ANSWER:"""

    return get_nvidia_response(rag_prompt), relevant_entities, context


def run_interactive_graphrag_session(driver, source_name: str) -> bool:
    """Ask GraphRAG questions until the user chooses to continue."""
    while True:
        query_text = input("\nEnter your GraphRAG question: ").strip()
        if not query_text:
            print("Please enter a question.")
            continue

        run_graphrag_query(driver, query_text, source_name)

        while True:
            choice = input("\nShall we continue to the next file? (yes/no): ").strip().lower()
            if choice in {"yes", "y"}:
                return True
            if choice in {"no", "n"}:
                print("Okay, ask another question for the current graph.")
                break
            print("Please type yes or no.")


# ==================== DIRECTORY PIPELINE ====================
def process_directory(
    input_directory: Path,
    watch_for_new_files: bool = WATCH_FOR_NEW_FILES,
    poll_interval: int = POLL_INTERVAL_SECONDS,
) -> None:
    seen_files = set()

    print(f"\nInput directory: {input_directory}")
    print("Each .txt file is processed in sorted order.")
    print("For each file: KG extraction -> Neo4j update -> embeddings -> GraphRAG query.")

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
    try:
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {NEO4J_URI}")

        while True:
            text_files = sorted(
                [
                    path
                    for path in input_directory.iterdir()
                    if path.is_file() and path.suffix.lower() == ".txt"
                ],
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
                    graph_updated = process_text_file(driver, file_path)
                    if graph_updated:
                        run_interactive_graphrag_session(driver, file_path.name)
                    seen_files.add(file_path)
                except Exception as error:
                    print(f"\nFailed to process {file_path.name}: {error}")

            if not watch_for_new_files:
                break
    finally:
        driver.close()


if __name__ == "__main__":
    if not INPUT_DIRECTORY.exists():
        raise SystemExit(f"Input directory not found: {INPUT_DIRECTORY}")

    try:
        process_directory(INPUT_DIRECTORY)
    except KeyboardInterrupt:
        print("\nStopped by user.")
