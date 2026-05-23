"""
Demo: Knowledge Graph Extraction Without Dependencies
Shows what the output would look like
"""

import json
from typing import List, Dict, Any
from dataclasses import dataclass, asdict

# ============================================================================
# Mock Classes (What the actual library returns)
# ============================================================================

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    name: str
    label: str  # Person, Organization, Location, Product, etc.
    
    def __repr__(self):
        return f"Entity(name='{self.name}', label='{self.label}')"

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    subject: Entity
    predicate: str
    obj: Entity
    
    def __repr__(self):
        return f"{self.subject.name} --[{self.predicate}]--> {self.obj.name}"

@dataclass
class KnowledgeGraph:
    """Represents a knowledge graph"""
    entities: List[Entity]
    relationships: List[Relationship]
    
    def summary(self):
        return f"KG({len(self.entities)} entities, {len(self.relationships)} relationships)"


# ============================================================================
# Example 1: Steve Jobs & Apple
# ============================================================================

def example_steve_jobs_apple():
    """Extract KG from Apple/Steve Jobs text"""
    
    print("\n" + "="*80)
    print("EXAMPLE 1: Steve Jobs & Apple Knowledge Graph")
    print("="*80)
    
    text1 = """
    Steve Jobs was the founder and CEO of Apple Inc. 
    Apple is a technology company that designs computers.
    Apple is headquartered in Cupertino, California.
    """
    
    text2 = """
    Tim Cook became the CEO of Apple in 2011.
    Steve Jobs resigned from Apple in 2011.
    Apple manufactures the iPhone, iPad, and Mac computers.
    """
    
    # Simulated extracted entities
    entities = [
        Entity(name="Steve Jobs", label="Person"),
        Entity(name="Apple Inc.", label="Organization"),
        Entity(name="Tim Cook", label="Person"),
        Entity(name="Cupertino", label="Location"),
        Entity(name="iPhone", label="Product"),
        Entity(name="iPad", label="Product"),
        Entity(name="Mac", label="Product"),
        Entity(name="2011", label="Date"),
    ]
    
    # Simulated extracted relationships
    relationships = [
        Relationship(entities[0], "founded", entities[1]),
        Relationship(entities[0], "was_ceo_of", entities[1]),
        Relationship(entities[1], "headquartered_in", entities[3]),
        Relationship(entities[1], "manufactures", entities[4]),
        Relationship(entities[1], "manufactures", entities[5]),
        Relationship(entities[1], "manufactures", entities[6]),
        Relationship(entities[2], "became_ceo_of", entities[1]),
        Relationship(entities[0], "resigned_from", entities[1]),
    ]
    
    kg = KnowledgeGraph(entities, relationships)
    
    print(f"\n📝 Input Texts:")
    print(f"   Text 1: {len(text1)} chars")
    print(f"   Text 2: {len(text2)} chars")
    
    print(f"\n📊 Extracted Entities ({len(entities)}):")
    for entity in entities:
        print(f"   • {entity.name:20} [Label: {entity.label}]")
    
    print(f"\n🔗 Extracted Relationships ({len(relationships)}):")
    for rel in relationships:
        print(f"   • {rel.subject.name:20} --[{rel.predicate:15}]--> {rel.obj.name}")
    
    return kg


# ============================================================================
# Example 2: Scientific Article
# ============================================================================

def example_research_article():
    """Extract KG from scientific article"""
    
    print("\n" + "="*80)
    print("EXAMPLE 2: Scientific Article Knowledge Graph")
    print("="*80)
    
    text = """
    The transformer architecture was introduced by Vaswani et al. in 2017.
    BERT is a transformer-based model developed by Google for natural language processing.
    BERT uses bidirectional training of transformers.
    The model was trained on Wikipedia and BookCorpus datasets.
    GPT-3 is a language model developed by OpenAI.
    Both BERT and GPT-3 are based on the transformer architecture.
    """
    
    entities = [
        Entity(name="Transformer", label="Technology"),
        Entity(name="Vaswani et al.", label="Person"),
        Entity(name="BERT", label="Model"),
        Entity(name="Google", label="Organization"),
        Entity(name="NLP", label="Technology"),
        Entity(name="Wikipedia", label="Dataset"),
        Entity(name="BookCorpus", label="Dataset"),
        Entity(name="GPT-3", label="Model"),
        Entity(name="OpenAI", label="Organization"),
        Entity(name="2017", label="Date"),
    ]
    
    relationships = [
        Relationship(entities[0], "introduced_by", entities[1]),
        Relationship(entities[0], "introduced_in", entities[9]),
        Relationship(entities[2], "is_based_on", entities[0]),
        Relationship(entities[2], "developed_by", entities[3]),
        Relationship(entities[2], "used_for", entities[4]),
        Relationship(entities[2], "trained_on", entities[5]),
        Relationship(entities[2], "trained_on", entities[6]),
        Relationship(entities[7], "developed_by", entities[8]),
        Relationship(entities[7], "is_based_on", entities[0]),
        Relationship(entities[2], "similar_to", entities[7]),
    ]
    
    kg = KnowledgeGraph(entities, relationships)
    
    print(f"\n📝 Input Text: {len(text)} chars")
    
    print(f"\n📊 Extracted Entities ({len(entities)}):")
    entity_types = {}
    for entity in entities:
        if entity.label not in entity_types:
            entity_types[entity.label] = []
        entity_types[entity.label].append(entity.name)
    
    for label, names in sorted(entity_types.items()):
        print(f"   [{label:15}] {', '.join(names)}")
    
    print(f"\n🔗 Extracted Relationships ({len(relationships)}):")
    for rel in relationships:
        print(f"   • {rel.subject.name:15} --[{rel.predicate:15}]--> {rel.obj.name}")
    
    return kg


# ============================================================================
# Example 3: Temporal Knowledge Graph (ATOM)
# ============================================================================

@dataclass
class TemporalRelationship:
    """5-tuple: (subject, predicate, object, t_start, t_end)"""
    subject: Entity
    predicate: str
    obj: Entity
    t_start: str
    t_end: str
    
    def __repr__(self):
        return f"({self.subject.name}, {self.predicate}, {self.obj.name}, {self.t_start}, {self.t_end})"

def example_temporal_kg():
    """Extract temporal KG (ATOM)"""
    
    print("\n" + "="*80)
    print("EXAMPLE 3: Temporal Knowledge Graph (ATOM)")
    print("="*80)
    
    # Simulated temporal text
    observations = [
        ("2007-01-09", "Steve Jobs was the CEO of Apple."),
        ("2011-08-24", "Steve Jobs is no longer CEO of Apple."),
        ("2011-08-25", "Tim Cook became the new CEO of Apple."),
    ]
    
    jobs = Entity(name="Steve Jobs", label="Person")
    cook = Entity(name="Tim Cook", label="Person")
    apple = Entity(name="Apple Inc.", label="Organization")
    
    # Temporal relationships
    temporal_rels = [
        TemporalRelationship(jobs, "is_ceo_of", apple, "2007-01-09", "2011-08-24"),
        TemporalRelationship(cook, "is_ceo_of", apple, "2011-08-25", "."),  # . = ongoing
    ]
    
    print(f"\n📝 Observations Over Time:")
    for date, text in observations:
        print(f"   [{date}] {text}")
    
    print(f"\n📊 Entities ({3}):")
    for entity in [jobs, cook, apple]:
        print(f"   • {entity.name:20} [Label: {entity.label}]")
    
    print(f"\n📅 Temporal Relationships (5-tuples):")
    print(f"   Format: (subject, predicate, object, t_start, t_end)")
    print(f"   Note: '.' means 'ongoing' or 'unknown'")
    for rel in temporal_rels:
        print(f"   • {rel}")
    
    return temporal_rels


# ============================================================================
# Visualization
# ============================================================================

def print_graph_ascii(kg: KnowledgeGraph):
    """ASCII art visualization"""
    
    print("\n" + "="*80)
    print("GRAPH VISUALIZATION (ASCII)")
    print("="*80)
    
    # Build adjacency
    edges = {}
    for rel in kg.relationships:
        if rel.subject.name not in edges:
            edges[rel.subject.name] = []
        edges[rel.subject.name].append((rel.predicate, rel.obj.name))
    
    # Simple tree visualization
    visited = set()
    
    def print_node(name, indent=0, parent=None):
        if name in visited:
            return
        visited.add(name)
        
        prefix = "  " * indent + "├─ " if indent > 0 else ""
        entity_label = next((e.label for e in kg.entities if e.name == name), "?")
        print(f"{prefix}{name} [{entity_label}]")
        
        if name in edges:
            for predicate, target in edges[name]:
                if target != parent:
                    rel_prefix = "  " * (indent + 1) + "└─ "
                    print(f"{'  ' * (indent + 1)}--[{predicate}]-->")
                    print_node(target, indent + 2, name)
    
    # Start from first entity
    if kg.entities:
        print_node(kg.entities[0].name)


def export_to_json(kg: KnowledgeGraph) -> str:
    """Export KG to JSON format"""
    
    data = {
        "entities": [
            {"name": e.name, "label": e.label} 
            for e in kg.entities
        ],
        "relationships": [
            {
                "subject": r.subject.name,
                "predicate": r.predicate,
                "object": r.obj.name
            }
            for r in kg.relationships
        ]
    }
    
    return json.dumps(data, indent=2)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n" + "█"*80)
    print("ITEXT2KG/ATOM - KNOWLEDGE GRAPH EXTRACTION DEMO")
    print("█"*80)
    
    # Run examples
    kg1 = example_steve_jobs_apple()
    kg2 = example_research_article()
    temporal_rels = example_temporal_kg()
    
    # Visualize first graph
    print_graph_ascii(kg1)
    
    # Export to JSON
    print("\n" + "="*80)
    print("EXPORT TO JSON FORMAT")
    print("="*80)
    json_output = export_to_json(kg1)
    print(json_output)
    
    # Summary
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. ✅ You now understand the KG structure (entities + relationships)

2. 📦 Install Python 3.10+ and dependencies:
   pip install -r requirements.txt

3. 🔑 Set your OpenAI API key:
   export OPENAI_API_KEY="sk-..."

4. 🚀 Run the actual extraction:
   python your_script.py

5. 📊 View in Neo4j (optional):
   docker run -p 7687:7687 -p 7474:7474 neo4j
   # Then navigate to http://localhost:7474
    """)
