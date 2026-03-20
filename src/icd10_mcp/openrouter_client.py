"""
OpenRouter API client.
Unchanged from your original RAG — drop-in copy.
"""
import os
import requests
from typing import Any, Dict, List, Optional


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    pass


def chat_completion(
    *,
    model: str,
    api_key: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 900,
    extra_headers: Optional[Dict[str, str]] = None,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    site_url = os.getenv("OPENROUTER_SITE_URL")
    app_name = os.getenv("OPENROUTER_APP_NAME")
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name

    if extra_headers:
        headers.update(extra_headers)

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        resp = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=60)
    except requests.RequestException as e:
        raise OpenRouterError(f"OpenRouter request failed: {e}") from e

    if resp.status_code >= 400:
        raise OpenRouterError(f"OpenRouter error {resp.status_code}: {resp.text}")

    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise OpenRouterError(f"Unexpected OpenRouter response: {data}")


OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"


class OpenRouterEmbedder:
    """
    Drop-in replacement for SentenceTransformer.
    encode() returns a numpy ndarray — .tolist() works without changes.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "openai/text-embedding-3-small",
        batch_size: int = 64,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.batch_size = batch_size

    def encode(self, texts, normalize_embeddings: bool = True, **_kwargs):
        import numpy as np

        single = isinstance(texts, str)
        if single:
            texts = [texts]

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            payload = {"model": self.model, "input": batch}
            resp = None
            for attempt in range(3):
                try:
                    resp = requests.post(
                        OPENROUTER_EMBEDDINGS_URL,
                        headers=headers,
                        json=payload,
                        timeout=120,
                    )
                    break
                except requests.exceptions.ReadTimeout:
                    if attempt < 2:
                        import time as _t
                        _t.sleep(5 * (attempt + 1))
                        continue
                    raise OpenRouterError(
                        f"Embeddings request timed out after 3 attempts (batch {i//self.batch_size + 1})"
                    )
                except requests.RequestException as exc:
                    raise OpenRouterError(f"Embeddings request failed: {exc}") from exc

            if resp is None:
                raise OpenRouterError("Embeddings request returned no response")

            if resp.status_code >= 400:
                raise OpenRouterError(f"Embeddings error {resp.status_code}: {resp.text}")

            data = resp.json()
            try:
                items = sorted(data["data"], key=lambda x: x["index"])
                all_embeddings.extend(item["embedding"] for item in items)
            except Exception as exc:
                raise OpenRouterError(f"Unexpected embeddings response format: {data}") from exc

        arr = np.array(all_embeddings, dtype=np.float32)

        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            arr = arr / norms

        return arr[0] if single else arr