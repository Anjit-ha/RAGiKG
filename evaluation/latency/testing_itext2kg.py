import os
import json
import time
import asyncio
import logging
from datetime import datetime
from typing import Tuple, Optional
from itext2kg import iText2KG
from itext2kg.atom.models import KnowledgeGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# ---------------- CONFIG ----------------
CACHE_DIR = "./batch_cache_itext2kg"
BATCH_SIZE = 5   # small for testing
ENT_THRESHOLD = 0.8
REL_THRESHOLD = 0.7

# ⚠️ IMPORTANT: use environment variable instead
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------- MODELS ----------------
openai_llm_model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    model="gpt-4.1-2025-04-14",
    temperature=0,
)

openai_embeddings_model = OpenAIEmbeddings(
    api_key=OPENAI_API_KEY,
    model="text-embedding-3-large",
)

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- SIMPLE INPUT DATA ----------------
def get_sample_data():
    return [
        "COVID-19 spread rapidly across the world in 2020.",
        "Pfizer developed a vaccine for COVID-19.",
        "Moderna also produced an mRNA vaccine.",
        "The World Health Organization declared COVID-19 a pandemic.",
        "Vaccination campaigns reduced infection rates."
    ]

# ---------------- MAIN PIPELINE ----------------
async def run_pipeline():
    data = get_sample_data()

    itext2kg_instance = iText2KG(
        llm_model=openai_llm_model,
        embeddings_model=openai_embeddings_model
    )

    logger.info(f"Processing {len(data)} text samples...")

    kg = await itext2kg_instance.build_graph(
        sections=data,
        existing_knowledge_graph=None,
        ent_threshold=ENT_THRESHOLD,
        rel_threshold=REL_THRESHOLD,
    )

    # -------- OUTPUT --------
    print("\n✅ FINAL KNOWLEDGE GRAPH")
    print("=" * 40)

    print("\nEntities:")
    for e in kg.entities:
        print("-", e)

    print("\nRelationships:")
    for r in kg.relationships:
        print("-", r)

    return kg

# ---------------- RUN ----------------
if __name__ == "__main__":
    asyncio.run(run_pipeline())