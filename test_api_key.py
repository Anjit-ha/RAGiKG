"""
Test Script: Verify OpenAI API Key Configuration
Run this before proceeding with knowledge graph extraction
"""

import os
import sys

print("\n" + "="*80)
print("🔍 STEP 3 VERIFICATION: Testing OpenAI API Configuration")
print("="*80)

# Check if API key exists
api_key = os.environ.get("OPENAI_API_KEY")

if not api_key:
    print("\n❌ ERROR: OPENAI_API_KEY environment variable not found!")
    print("\n   How to fix:")
    print("   1. Get your API key from: https://platform.openai.com/api-keys")
    print("   2. Set it in PowerShell:")
    print("      [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-your-key', 'User')")
    print("   3. Restart PowerShell/Python")
    sys.exit(1)

print(f"\n✅ API Key found: {api_key[:20]}...{api_key[-4:]}")

# Try to import and test
try:
    print("\n🤖 Testing LangChain imports...")
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    print("   ✅ langchain_openai imported successfully")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    print("   Run: pip install langchain-openai")
    sys.exit(1)

# Test LLM
try:
    print("\n🤖 Testing ChatOpenAI connection...")
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    response = llm.invoke("Say 'Hello' briefly (1 word only)")
    print(f"   ✅ LLM Response: {response.content}")
except Exception as e:
    print(f"   ❌ LLM Error: {e}")
    sys.exit(1)

# Test Embeddings
try:
    print("\n📊 Testing OpenAI Embeddings...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    result = embeddings.embed_query("Test embedding")
    print(f"   ✅ Embedding created: {len(result)} dimensions")
except Exception as e:
    print(f"   ❌ Embeddings Error: {e}")
    sys.exit(1)

print("\n" + "="*80)
print("🎉 ALL TESTS PASSED! API is configured correctly.")
print("="*80)
print("\nYou can now proceed to Step 4: Run the full extraction")
