# 🚀 Quick Start Guide: iText2KG Knowledge Graph Extraction

## ⚠️ Current Environment Issue

**Python 3.8.10** → You have this (TOO OLD)  
**Python 3.10+** → You need this

**Solution**: Upgrade Python or create a new virtual environment with Python 3.10+

---

## 📋 Installation Steps

### Step 1: Install Python 3.10+
```bash
# Check your current Python version
python --version

# Download and install from https://www.python.org/downloads/
# Ensure "Add Python to PATH" is checked during installation
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Navigate to project directory
cd c:\Users\USER\Downloads\prjt\itext2kg-main

# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
# Install project dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Step 4: Set API Key
```bash
# For OpenAI (required)
export OPENAI_API_KEY="sk-your-key-here"

# Or set it in Python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
```

---

## 💡 Complete Example: Build a Knowledge Graph

### Option 1: Using iText2KG_Star (Simpler, Faster)

```python
import asyncio
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from itext2kg.itext2kg_star import iText2KG_Star

async def main():
    # Setup
    os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
    
    llm = ChatOpenAI(model="gpt-4", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create KG builder
    kg_builder = iText2KG_Star(llm_model=llm, embeddings_model=embeddings)
    
    # Your text documents
    text1 = """
    Steve Jobs was the founder and CEO of Apple Inc. 
    Apple is a technology company headquartered in Cupertino, California.
    """
    
    text2 = """
    Tim Cook became CEO of Apple in 2011. 
    Apple manufactures iPhones, iPads, and Mac computers.
    """
    
    # Build knowledge graph
    kg = await kg_builder.build_graph(
        sections=[text1, text2],
        ent_threshold=0.7,      # Entity matching threshold
        rel_threshold=0.7,      # Relationship matching threshold
        entity_name_weight=0.6, # Weight entity name in embeddings
        entity_label_weight=0.4 # Weight entity label in embeddings
    )
    
    # Display results
    print(f"\n📊 Extracted {len(kg.entities)} entities:")
    for entity in kg.entities:
        print(f"   • {entity.name} (Label: {entity.label})")
    
    print(f"\n🔗 Extracted {len(kg.relationships)} relationships:")
    for rel in kg.relationships:
        print(f"   • {rel.startNode.name} --[{rel.name}]--> {rel.endNode.name}")
    
    return kg

# Run
kg = asyncio.run(main())
```

### Option 2: Using ATOM (Temporal Knowledge Graphs)

```python
import asyncio
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from itext2kg.atom import Atom

async def main():
    os.environ["OPENAI_API_KEY"] = "sk-..."
    
    llm = ChatOpenAI(model="gpt-4")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    # Create ATOM instance
    atom = Atom(llm_model=llm, embeddings_model=embeddings)
    
    text = "Steve Jobs was CEO of Apple on January 9, 2007"
    
    # Extract with temporal information
    kg = await atom.build_graph(
        documents=[text],
        observation_timestamp="2007-01-09"
    )
    
    print(f"Temporal KG with {len(kg.relationships)} quintuples:")
    for rel in kg.relationships:
        print(f"   ({rel.startNode.name}, {rel.name}, {rel.endNode.name}, "
              f"{rel.properties.t_start}, {rel.properties.t_end})")
    
    return kg

asyncio.run(main())
```

---

## 📊 Expected Output Example

```
📊 Extracted 6 entities:
   • Steve Jobs (Label: Person)
   • Apple Inc. (Label: Organization)
   • Tim Cook (Label: Person)
   • Cupertino (Label: Location)
   • iPhone (Label: Product)
   • Mac (Label: Product)

🔗 Extracted 7 relationships:
   • Steve Jobs --[founded]--> Apple Inc.
   • Steve Jobs --[was_ceo_of]--> Apple Inc.
   • Tim Cook --[became_ceo_of]--> Apple Inc.
   • Apple Inc. --[headquartered_in]--> Cupertino
   • Apple Inc. --[manufactures]--> iPhone
   • Apple Inc. --[manufactures]--> Mac
```

---

## 🗄️ Store in Neo4j (Optional)

```python
from itext2kg.graph_integration.neo4j_storage import Neo4jStorage

# Initialize storage
storage = Neo4jStorage(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="your-password"
)

# Store the knowledge graph
storage.store_knowledge_graph(kg)

# View in Neo4j Browser: http://localhost:7474
```

---

## 📁 Available Sample Datasets

The project includes sample data in `datasets/`:

### 1. **CVs** (Curriculum Vitae)
```
datasets/itext2kg/cvs/
  ├── John_Doe.txt
  ├── Jane_Smith.txt
  ├── Emily_Davis.txt
  ├── Michael_Brown.txt
  └── Robert_Johnson.txt
```

### 2. **Scientific Articles**
```
datasets/itext2kg/scientific_articles/
  ├── bertology.txt
  ├── bioclip.txt
  ├── building_age.txt
  ├── pollmgraph.txt
  └── seasonal.txt
```

### 3. **News Articles**
```
datasets/atom/nyt_news/
  └── [New York Times COVID articles]
```

---

## 🧪 Running with Sample Data

```python
import asyncio
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from itext2kg.itext2kg_star import iText2KG_Star

async def main():
    # Load sample CV
    with open("datasets/itext2kg/cvs/John_Doe.txt") as f:
        text = f.read()
    
    llm = ChatOpenAI(model="gpt-4")
    embeddings = OpenAIEmbeddings()
    
    kg_builder = iText2KG_Star(llm_model=llm, embeddings_model=embeddings)
    kg = await kg_builder.build_graph([text])
    
    print(f"✅ Extracted graph with {len(kg.entities)} entities")

asyncio.run(main())
```

---

## 🧪 Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/itext2kg/test_itext2kg_matching.py

# Run with verbose output
pytest -v tests/
```

---

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ent_threshold` | 0.7 | Entity matching threshold (higher = stricter matching) |
| `rel_threshold` | 0.7 | Relationship matching threshold |
| `entity_name_weight` | 0.6 | Weight for entity name in embeddings |
| `entity_label_weight` | 0.4 | Weight for entity label in embeddings |
| `max_tries` | 5 | Max LLM attempts for extraction |
| `observation_date` | "" | Timestamp for temporal KGs |

---

## 🐛 Troubleshooting

### Error: `ModuleNotFoundError: No module named 'pydantic'`
```bash
pip install pydantic pydantic-settings
```

### Error: `No module named 'langchain_openai'`
```bash
pip install langchain-openai
```

### Error: `OPENAI_API_KEY not found`
```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-actual-key"
```

### Error: Python version incompatible
```bash
# Check version
python --version

# Need 3.10+, upgrade or use virtual environment with newer Python
python3.10 -m venv venv
```

---

## 📚 Key Concepts

### Entity Embedding
- Entities are embedded using **name** (0.6 weight) + **label** (0.4 weight)
- Prevents merging entities like "Python:Language" vs "Python:Snake"

### Graph Matching
- Uses **cosine similarity** for entity/relationship resolution
- Threshold-based matching (default 0.7)

### Temporal KGs (ATOM)
- Extracts **5-tuples**: (subject, predicate, object, t_start, t_end)
- Example: (Steve Jobs, CEO, Apple, 2007-01-09, 2011-08-24)

### Incremental Building
- Processes multiple documents sequentially
- Merges entities and relationships across documents
- Can expand existing knowledge graphs

---

## 📖 References

- **Paper**: https://arxiv.org/abs/2510.22590 (ATOM)
- **Paper**: https://arxiv.org/abs/2409.03284 (iText2KG)
- **LangChain Docs**: https://python.langchain.com/
- **Neo4j Docs**: https://neo4j.com/docs/
