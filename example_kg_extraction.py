"""
Simple Example: Knowledge Graph Extraction with iText2KG

This script demonstrates how to:
1. Extract entities and relationships from text
2. Build a knowledge graph
3. Visualize the results

Prerequisites:
- Python 3.10+
- OpenAI API Key (or other LangChain-supported LLM)
- pip install -r requirements.txt
"""

import asyncio
import json
from typing import List, Dict, Any

# NOTE: This example requires proper setup of dependencies
# For now, showing the structure and what output looks like

SAMPLE_TEXT = """
Steve Jobs was the founder and CEO of Apple Inc. Apple is a technology company 
that designs and manufactures computers and consumer electronics. In 2011, Steve Jobs 
resigned as CEO of Apple. Tim Cook became the new CEO of Apple. Apple was founded 
in 1976 in Cupertino, California.
"""

SAMPLE_TEXT_2 = """
Tim Cook is the current CEO of Apple Inc. He was born in 1960. Apple Inc. 
develops the iPhone, iPad, and Mac computers. The company is headquartered 
in Cupertino, California. Apple is known for innovation in technology.
"""


# ============================================================================
# SETUP: Import after installing requirements
# ============================================================================
async def setup_and_run():
    """
    This is how you would run the actual iText2KG library.
    Uncomment after installing: pip install -r requirements.txt
    """
    
    # Step 1: Initialize LLM and Embeddings
    # from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    # import os
    # 
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    # 
    # llm = ChatOpenAI(model="gpt-4", temperature=0)
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    
    # Step 2: Initialize iText2KG
    # from itext2kg.itext2kg_star import iText2KG_Star
    # 
    # itext2kg = iText2KG_Star(llm_model=llm, embeddings_model=embeddings)
    
    
    # Step 3: Build Knowledge Graph
    # kg = await itext2kg.build_graph(
    #     sections=[SAMPLE_TEXT, SAMPLE_TEXT_2],
    #     ent_threshold=0.7,
    #     rel_threshold=0.7,
    #     entity_name_weight=0.6,
    #     entity_label_weight=0.4
    # )
    
    
    # Step 4: Display Results
    # print("\n" + "="*80)
    # print("EXTRACTED KNOWLEDGE GRAPH")
    # print("="*80)
    # print(f"\n📊 Entities Found: {len(kg.entities)}")
    # for entity in kg.entities:
    #     print(f"   • {entity.name} (Label: {entity.label})")
    # 
    # print(f"\n🔗 Relationships Found: {len(kg.relationships)}")
    # for rel in kg.relationships:
    #     print(f"   • {rel.startNode.name} --[{rel.name}]--> {rel.endNode.name}")
    
    
    # Step 5: Store in Neo4j (optional)
    # from itext2kg.graph_integration.neo4j_storage import Neo4jStorage
    # 
    # neo4j_storage = Neo4jStorage(
    #     uri="bolt://localhost:7687",
    #     user="neo4j",
    #     password="password"
    # )
    # neo4j_storage.store_knowledge_graph(kg)
    # print("\n✅ Knowledge graph stored in Neo4j!")
    
    
    print("Setup example. Uncomment code above after installing requirements.")


# ============================================================================
# EXPECTED OUTPUT STRUCTURE
# ============================================================================

def show_expected_output():
    """Shows what the graph output would look like"""
    
    expected_kg = {
        "entities": [
            {"name": "Steve Jobs", "label": "Person"},
            {"name": "Apple Inc.", "label": "Organization"},
            {"name": "Tim Cook", "label": "Person"},
            {"name": "Cupertino", "label": "Location"},
            {"name": "iPhone", "label": "Product"},
            {"name": "Mac", "label": "Product"}
        ],
        "relationships": [
            {"subject": "Steve Jobs", "predicate": "founded", "object": "Apple Inc."},
            {"subject": "Steve Jobs", "predicate": "was_ceo_of", "object": "Apple Inc."},
            {"subject": "Tim Cook", "predicate": "is_ceo_of", "object": "Apple Inc."},
            {"subject": "Apple Inc.", "predicate": "headquartered_in", "object": "Cupertino"},
            {"subject": "Apple Inc.", "predicate": "manufactures", "object": "iPhone"},
            {"subject": "Apple Inc.", "predicate": "manufactures", "object": "Mac"},
            {"subject": "Tim Cook", "predicate": "born_in", "object": "1960"},
            {"subject": "Apple Inc.", "predicate": "founded_in", "object": "1976"}
        ]
    }
    
    return expected_kg


def visualize_graph_ascii(kg_data: Dict[str, Any]):
    """ASCII visualization of the knowledge graph"""
    print("\n" + "="*80)
    print("KNOWLEDGE GRAPH VISUALIZATION (ASCII)")
    print("="*80)
    
    print("\n📊 ENTITIES:")
    for entity in kg_data["entities"]:
        print(f"   [{entity['label']}] {entity['name']}")
    
    print("\n🔗 RELATIONSHIPS:")
    for rel in kg_data["relationships"]:
        print(f"   {rel['subject']} --[{rel['predicate']}]--> {rel['object']}")
    
    print("\n📈 GRAPH STATISTICS:")
    print(f"   Total Entities: {len(kg_data['entities'])}")
    print(f"   Total Relationships: {len(kg_data['relationships'])}")
    print(f"   Entity Types: {set(e['label'] for e in kg_data['entities'])}")


# ============================================================================
# DATASET EXAMPLES
# ============================================================================

def list_available_datasets():
    """Show available sample datasets in the project"""
    
    print("\n" + "="*80)
    print("AVAILABLE DATASETS IN PROJECT")
    print("="*80)
    
    datasets = {
        "CVs": {
            "path": "datasets/itext2kg/cvs/",
            "files": ["Emily_Davis.txt", "Jane_Smith.txt", "John_Doe.txt"],
            "description": "Curriculum Vitae documents for entity/relation extraction"
        },
        "Scientific Articles": {
            "path": "datasets/itext2kg/scientific_articles/",
            "files": ["bertology.txt", "bioclip.txt", "building_age.txt"],
            "description": "Research papers for knowledge extraction"
        },
        "News": {
            "path": "datasets/atom/nyt_news/",
            "description": "New York Times articles for temporal KG construction"
        }
    }
    
    for dataset_name, info in datasets.items():
        print(f"\n📁 {dataset_name}:")
        print(f"   Path: {info['path']}")
        print(f"   Description: {info['description']}")
        if "files" in info:
            for file in info["files"]:
                print(f"      • {file}")


# ============================================================================
# COMPLETE USAGE EXAMPLE
# ============================================================================

def print_complete_example():
    """Print complete usage code"""
    
    code = '''
# Complete Usage Example

import asyncio
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from itext2kg.itext2kg_star import iText2KG_Star

async def main():
    # Set your OpenAI API key
    os.environ["OPENAI_API_KEY"] = "sk-..."
    
    # Initialize models
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create iText2KG instance
    kg_builder = iText2KG_Star(llm_model=llm, embeddings_model=embeddings)
    
    # Load your documents
    text1 = "Steve Jobs founded Apple Inc. in 1976..."
    text2 = "Apple is headquartered in Cupertino, California..."
    
    # Build knowledge graph
    kg = await kg_builder.build_graph(
        sections=[text1, text2],
        ent_threshold=0.7,
        rel_threshold=0.7,
        entity_name_weight=0.6,
        entity_label_weight=0.4
    )
    
    # Display results
    print(f"Extracted {len(kg.entities)} entities and {len(kg.relationships)} relationships")
    
    # Save to Neo4j (optional)
    from itext2kg.graph_integration.neo4j_storage import Neo4jStorage
    
    storage = Neo4jStorage(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="password"
    )
    storage.store_knowledge_graph(kg)

# Run
asyncio.run(main())
'''
    
    print("\n" + "="*80)
    print("COMPLETE USAGE CODE")
    print("="*80)
    print(code)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("iText2KG Knowledge Graph Extraction Example\n")
    
    # Show datasets
    list_available_datasets()
    
    # Show expected output
    expected_kg = show_expected_output()
    visualize_graph_ascii(expected_kg)
    
    # Show complete example
    print_complete_example()
    
    # Show steps to run
    print("\n" + "="*80)
    print("STEPS TO RUN THIS PROJECT")
    print("="*80)
    print("""
1. Install Python 3.10+
   - Current: Python 3.8.10 (TOO OLD)
   - Required: Python 3.10 or higher

2. Install dependencies:
   $ pip install -r requirements.txt

3. Set up API credentials:
   export OPENAI_API_KEY="your-key-here"

4. Run the example:
   $ python example_kg_extraction.py

5. (Optional) Set up Neo4j for visualization:
   - Install Neo4j Desktop or use Docker
   - Update connection details in the script

6. View the extracted knowledge graph:
   - In console output
   - In Neo4j Browser (if using Neo4j)
    """)
