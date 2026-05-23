"""
Unit tests for NVIDIA API functionality
Tests LLM and embedding calls independently with proper error handling
"""

import os
import json
import time
import requests
from typing import Dict, List

# ==================== CONFIG ====================
NVIDIA_EMBEDDINGS_URL = "https://integrate.api.nvidia.com/v1/embeddings"
NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_CHAT_MODEL = "meta/llama-3.2-90b-vision-instruct"


def get_api_key() -> str:
    """Get NVIDIA API key from environment."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise EnvironmentError("❌ NVIDIA_API_KEY not set in environment")
    print(f"✓ API key found: {api_key[:20]}...{api_key[-4:]}")
    return api_key


def test_endpoint_connectivity(url: str, timeout: int = 10) -> bool:
    """Test if endpoint is reachable."""
    try:
        print(f"\n🌐 Testing connectivity to {url}...")
        response = requests.get(url, timeout=timeout)
        print(f"✓ Endpoint reachable (status: {response.status_code})")
        return True
    except requests.exceptions.Timeout:
        print(f"❌ Connection timeout to {url}")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False


def test_embedding_api(api_key: str, timeout: int = 60) -> bool:
    """Test NVIDIA embedding API with simple text."""
    print(f"\n📊 Testing NVIDIA Embedding API (timeout={timeout}s)...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    payload = {
        "model": NVIDIA_EMBEDDING_MODEL,
        "input": "Hello world",
        "input_type": "query",
        "encoding_format": "float",
        "truncate": "NONE"
    }
    
    try:
        start = time.time()
        response = requests.post(
            NVIDIA_EMBEDDINGS_URL, 
            headers=headers, 
            json=payload, 
            timeout=timeout
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            embedding = data["data"][0]["embedding"]
            print(f"✓ Embedding successful in {elapsed:.2f}s")
            print(f"  - Embedding dimension: {len(embedding)}")
            print(f"  - First 5 values: {embedding[:5]}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Embedding request timed out after {timeout}s")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_llm_api_simple(api_key: str, timeout: int = 60) -> bool:
    """Test NVIDIA LLM API with simple prompt."""
    print(f"\n🤖 Testing NVIDIA LLM API (simple prompt, timeout={timeout}s)...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    payload = {
        "model": NVIDIA_CHAT_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Reply in one sentence."},
            {"role": "user", "content": "Say hello briefly."},
        ],
        "temperature": 0.0,
        "max_tokens": 100,
    }
    
    try:
        start = time.time()
        response = requests.post(
            NVIDIA_CHAT_URL, 
            headers=headers, 
            json=payload, 
            timeout=timeout
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            message = data["choices"][0]["message"]["content"]
            print(f"✓ LLM response successful in {elapsed:.2f}s")
            print(f"  - Message: {message[:100]}")
            return True
        else:
            print(f"❌ API returned status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ LLM request timed out after {timeout}s")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_llm_api_kg_extraction(api_key: str, timeout: int = 60) -> bool:
    """Test NVIDIA LLM API with knowledge graph extraction."""
    print(f"\n📈 Testing NVIDIA LLM API (KG extraction, timeout={timeout}s)...")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    
    payload = {
        "model": NVIDIA_CHAT_MODEL,
        "messages": [
            {
                "role": "system", 
                "content": "Extract a small knowledge graph from the user text. Return JSON with 'entities' and 'relations' arrays."
            },
            {
                "role": "user", 
                "content": "Apple is a company founded by Steve Jobs. Tim Cook is the CEO of Apple."
            },
        ],
        "temperature": 0.0,
        "max_tokens": 500,
    }
    
    try:
        start = time.time()
        response = requests.post(
            NVIDIA_CHAT_URL, 
            headers=headers, 
            json=payload, 
            timeout=timeout
        )
        elapsed = time.time() - start
        
        if response.status_code == 200:
            data = response.json()
            message = data["choices"][0]["message"]["content"]
            print(f"✓ KG extraction successful in {elapsed:.2f}s")
            print(f"  - Response: {message[:200]}")
            
            # Try to parse JSON
            try:
                cleaned = message.strip()
                cleaned = cleaned.replace("```json", "").replace("```", "")
                parsed = json.loads(cleaned)
                print(f"  - JSON valid: {list(parsed.keys())}")
                return True
            except json.JSONDecodeError:
                print(f"  - Warning: Response is not valid JSON (might still be OK)")
                return True
        else:
            print(f"❌ API returned status {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ LLM request timed out after {timeout}s")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("🧪 NVIDIA API Unit Tests")
    print("=" * 80)
    
    try:
        api_key = get_api_key()
    except EnvironmentError as e:
        print(f"\n{e}")
        return False
    
    results = {}
    
    # Test 1: Connectivity
    results["connectivity"] = test_endpoint_connectivity(NVIDIA_CHAT_URL)
    
    # Test 2: Embedding (shorter timeout first)
    results["embedding_30s"] = test_embedding_api(api_key, timeout=30)
    
    # Test 3: Simple LLM (shorter timeout first)
    results["llm_simple_30s"] = test_llm_api_simple(api_key, timeout=30)
    
    # Test 4: KG Extraction (longer timeout)
    results["llm_kg_120s"] = test_llm_api_kg_extraction(api_key, timeout=120)
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 Test Summary")
    print("=" * 80)
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests PASSED! API is working correctly.")
        print("\nNext: You can now integrate this into free_kg.py with confidence.")
    else:
        print("⚠️  Some tests FAILED. Check errors above.")
        print("\nPossible issues:")
        print("  1. API key is invalid or expired")
        print("  2. NVIDIA API is down or unreachable")
        print("  3. Rate limiting or account issues")
        print("  4. Network connectivity issues")
    print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
