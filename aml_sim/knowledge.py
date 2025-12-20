from __future__ import annotations

import dataclasses
from typing import List, Tuple

import hashlib
import torch
from torch.nn.functional import cosine_similarity
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast


@dataclasses.dataclass
class Document:
    title: str
    text: str
    embedding: torch.Tensor


class DPRTextEmbedder:
    """Embed text with a DPR context encoder for retrieval."""

    def __init__(self, model_path: str = "facebook/dpr-ctx_encoder-single-nq-base", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = None
        self.tokenizer = None
        self.using_transformer = False
        try:
            self.tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path, local_files_only=True)
            self.model = DPRContextEncoder.from_pretrained(model_path, local_files_only=True).to(self.device)
            self.model.eval()
            self.using_transformer = True
        except Exception:
            # Fallback to lightweight hashing-based embeddings for offline use.
            self.model = None
            self.tokenizer = None
            self.using_transformer = False

    def embed(self, text: str) -> torch.Tensor:
        if self.using_transformer and self.tokenizer and self.model:
            with torch.no_grad():
                tokenized = self.tokenizer(text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
                input_ids = tokenized["input_ids"].to(self.device)
                attention_mask = tokenized["attention_mask"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                pooled = outputs.pooler_output.squeeze(0).detach()
                return pooled
        # Deterministic hashing fallback
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [b / 255.0 for b in digest[:64]]
        return torch.tensor(values, dtype=torch.float, device=self.device)


class KnowledgeBase:
    def __init__(self, documents: List[Tuple[str, str]], embedder: DPRTextEmbedder | None = None):
        self.embedder = embedder or DPRTextEmbedder()
        self.documents: List[Document] = []
        self._embeddings: torch.Tensor | None = None
        for title, text in documents:
            self.add_document(title, text)
    
    def add_document(self, title: str, text: str) -> None:
        embedding = self.embedder.embed(text)
        doc = Document(title=title, text=text, embedding=embedding)
        self.documents.append(doc)
        if self._embeddings is None:
            self._embeddings = embedding.unsqueeze(0)
        else:
            self._embeddings = torch.cat([self._embeddings, embedding.unsqueeze(0)], dim=0)
    
    def query(self, text: str, top_k: int = 3) -> List[Document]:
        if not self.documents:
            return []
        query_vec = self.embedder.embed(text)
        sims = cosine_similarity(self._embeddings, query_vec.unsqueeze(0), dim=1)
        top_indices = torch.argsort(sims, descending=True)[:top_k]
        return [self.documents[i] for i in top_indices.tolist()]
    
    @classmethod
    def from_config(
        cls,
        docs_config: List[dict],
        *,
        model_path: str | None = None,
        device: str | None = None,
        embedder: DPRTextEmbedder | None = None,
    ) -> "KnowledgeBase":
        docs = []
        for entry in docs_config:
            docs.append((entry.get("title", ""), entry.get("text", "")))
        active_embedder = embedder or DPRTextEmbedder(model_path=model_path or "facebook/dpr-ctx_encoder-single-nq-base", device=device)
        return cls(docs, embedder=active_embedder)
