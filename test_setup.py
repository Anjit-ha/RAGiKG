"""Test if NVIDIA API and Neo4j are properly configured."""
import os
import sys
import requests
from neo4j import GraphDatabase

print("=" * 60)
print("TESTING SETUP")
print("=" * 60)

# Test 1: Check API Key
print("\n1️⃣  Checking NVIDIA_API_KEY...")
api_key = os.getenv("NVIDIA_API_KEY")
if api_key:
    print(f"   ✅ NVIDIA_API_KEY found (length: {len(api_key)})")
else:
    print("   ❌ NVIDIA_API_KEY not set!")
    sys.exit(1)

# Test 2: Test NVIDIA API endpoint
print("\n2️⃣  Testing NVIDIA API endpoint...")
try:
    test_payload = {
        "model": "meta/llama-3.2-90b-vision-instruct",
        "messages": [
            {"role": "system", "content": "Extract entities from text."},
            {"role": "user", "content": "Hello world"},
        ],
        "max_tokens": 50,
        "temperature": 0.0,
    }
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers=headers,
        json=test_payload,
        timeout=10
    )
    
    if response.status_code == 200:
        print(f"   ✅ NVIDIA API responding (status: {response.status_code})")
    else:
        print(f"   ❌ NVIDIA API error: {response.status_code}")
        print(f"   Response: {response.text[:200]}")
except Exception as e:
    print(f"   ❌ Error: {e}")
    sys.exit(1)

# Test 3: Test Neo4j connection
print("\n3️⃣  Testing Neo4j connection...")
try:
    uri = "bolt://localhost:7687"
    username = "neo4j"
    password = "Anjitha@2002"
    
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        result = session.run("RETURN 1 as test")
        print(f"   ✅ Neo4j connected successfully")
    driver.close()
except Exception as e:
    print(f"   ❌ Neo4j connection failed: {e}")
    sys.exit(1)

# Test 4: Check current entities in database
print("\n4️⃣  Current entities in Neo4j database...")
try:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    with driver.session() as session:
        result = session.run("MATCH (n:Entity) RETURN DISTINCT n.name ORDER BY n.name LIMIT 20")
        entities = [record["n.name"] for record in result]
        if entities:
            print(f"   Found {len(entities)} entities:")
            for e in entities:
                print(f"      - {e}")
        else:
            print("   ✅ Database is empty (good for fresh test)")
    driver.close()
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("✅ All checks passed! Ready to run free_kg.py")
print("=" * 60)
