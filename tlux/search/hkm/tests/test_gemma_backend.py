import json

import numpy as np

from tlux.search.hkm.libs.gemma import inference


class _Response:
    def __init__(self, body: object) -> None:
        self.body = json.dumps(body).encode("utf-8")

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_gemma_backend_posts_openai_embeddings(monkeypatch) -> None:
    monkeypatch.setattr(inference, "ENDPOINT", "http://example.test/v1/embeddings")
    monkeypatch.setattr(inference.drama, "detokenize", lambda rows: ["a query"])
    seen = {}

    def fake_urlopen(request, timeout):
        seen["url"] = request.full_url
        seen["payload"] = json.loads(request.data)
        return _Response({"data": [{"index": 0, "embedding": [3.0, 4.0]}]})

    monkeypatch.setattr(inference.urllib.request, "urlopen", fake_urlopen)
    values = inference.embed([[1]], role="query")
    assert seen["url"].endswith("/v1/embeddings")
    assert seen["payload"]["input"] == ["task: search result | query: a query"]
    assert np.allclose(values, [[0.6, 0.8]])


def test_gemma_backend_uses_document_title(monkeypatch) -> None:
    monkeypatch.setattr(inference, "ENDPOINT", "http://example.test/v1/embeddings")
    monkeypatch.setattr(inference.drama, "detokenize", lambda rows: ["A useful title\nThe body text"])
    seen = {}

    def fake_urlopen(request, timeout):
        seen["payload"] = json.loads(request.data)
        return _Response({"data": [{"index": 0, "embedding": [1.0, 0.0]}]})

    monkeypatch.setattr(inference.urllib.request, "urlopen", fake_urlopen)
    inference.embed([[1]], role="doc")
    assert seen["payload"]["input"] == ["title: A useful title | text: The body text"]
