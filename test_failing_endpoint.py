import json
import os
import re
from typing import Any

from openai import OpenAI

def load_local_env() -> None:
    env_path = ".env"
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_local_env()
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise EnvironmentError("NVIDIA_API_KEY not set")

NVIDIA_CHAT_MODEL = "meta/llama-3.3-70b-instruct"


def message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part.strip())
    return str(content)


def extract_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    if cleaned.startswith("{"):
        return json.loads(cleaned)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response")
    return json.loads(match.group(0))


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY,
)

system_prompt = (
    "Extract a small knowledge graph from the user text. "
    "Return JSON only, with exactly this shape: "
    '{"entities":["..."],"relations":[{"subject":"...","relation":"...","object":"..."}]}. '
    "Every relation subject and object must also appear in entities. "
    "Extract at least one relation when the text states any connection, action, description, or event. "
    "Do not include markdown or text outside JSON."
)

text = (
    "Arjun, Meera, and Riya are friends from college. "
    "Meera suggested an unplanned trip. Arjun protested the idea. "
    "Riya supported Meera."
)

response = client.chat.completions.create(
    model=NVIDIA_CHAT_MODEL,
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ],
    temperature=0.0,
    max_tokens=512,
)

content = message_content_to_text(response.choices[0].message.content)
print("Raw content:")
print(content)

parsed = extract_json_object(content)
print("\nParsed:")
print(json.dumps(parsed, indent=2))
print(f"\nEntities: {len(parsed.get('entities', []))}")
print(f"Relations: {len(parsed.get('relations', []))}")
