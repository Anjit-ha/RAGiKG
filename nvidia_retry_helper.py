"""
Retry helper for NVIDIA API with exponential backoff
Handles timeouts and rate limiting gracefully
"""

import time
import requests
from typing import Dict, Any, Callable


class NVIDIAAPIError(Exception):
    """Base exception for NVIDIA API errors."""
    pass


def call_nvidia_api_with_retries(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    max_retries: int = 3,
    initial_timeout: int = 120,
    backoff_factor: float = 1.5,
) -> Dict[str, Any]:
    """
    Call NVIDIA API with exponential backoff retry logic.
    
    Args:
        url: API endpoint
        headers: Request headers (with Authorization)
        payload: Request payload
        max_retries: Number of retries on timeout
        initial_timeout: Initial timeout in seconds
        backoff_factor: Multiply timeout by this factor on retry
        
    Returns:
        Response JSON dict
        
    Raises:
        NVIDIAAPIError: If all retries exhausted or non-retryable error
    """
    timeout = initial_timeout
    last_error = None
    
    for attempt in range(max_retries + 1):
        try:
            print(f"  Attempt {attempt + 1}/{max_retries + 1} (timeout={timeout}s)...", end=" ", flush=True)
            
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
            response.raise_for_status()
            
            print("✓")
            return response.json()
            
        except requests.exceptions.Timeout as e:
            last_error = e
            if attempt < max_retries:
                wait_time = backoff_factor ** attempt
                print(f"⏱ timeout, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                timeout = int(initial_timeout * (backoff_factor ** (attempt + 1)))
            else:
                print("❌ all retries exhausted")
                
        except requests.exceptions.HTTPError as e:
            # Non-retryable HTTP error (4xx, 5xx)
            error_msg = f"HTTP {response.status_code}: {response.text[:200]}"
            print(f"❌ {error_msg}")
            raise NVIDIAAPIError(error_msg) from e
            
        except requests.exceptions.RequestException as e:
            # Network error, connection error, etc.
            error_msg = f"Request failed: {str(e)[:200]}"
            print(f"❌ {error_msg}")
            raise NVIDIAAPIError(error_msg) from e
    
    # All retries exhausted
    raise NVIDIAAPIError(f"Request timed out after {max_retries} retries. Last error: {last_error}")
