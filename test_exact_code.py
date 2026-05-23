"""Test the exact OpenAI client code provided by the user"""

from openai import OpenAI
import os

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise EnvironmentError("NVIDIA_API_KEY not set")

print(f"Using API key: {api_key[:20]}...")

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = api_key
)

print("Created OpenAI client successfully")
print("Testing chat completion...")

try:
    completion = client.chat.completions.create(
      model="meta/llama-3.3-70b-instruct",
      messages=[{"role":"user","content":"Say hello briefly in one word"}],
      temperature=0.2,
      top_p=0.7,
      max_tokens=100,
      stream=False
    )
    
    print("✓ Success!")
    print(f"Response: {completion.choices[0].message.content}")
except Exception as e:
    print(f"❌ Failed: {e}")
    print(f"Error type: {type(e).__name__}")
