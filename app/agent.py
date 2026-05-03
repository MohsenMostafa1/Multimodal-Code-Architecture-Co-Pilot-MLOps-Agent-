import base64
import os
from typing import Optional
import ollama
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "mlops-agent-knowledge"
VISION_MODEL = "deepseek-vl:7b"   # or "llava:13b"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

class MLOpsCoPilot:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = self.pc.Index(INDEX_NAME)
        self.embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)

    def _query_knowledge(self, question: str, top_k: int = 3) -> str:
        """Retrieve relevant MLOps patterns."""
        vec = self.embeddings.embed_query(question)
        results = self.index.query(vector=vec, top_k=top_k, include_metadata=True)
        contexts = [match.metadata["text"] for match in results.matches]
        return "\n\n".join(contexts)

    def _encode_image(self, image_path: str) -> str:
        """Convert image to base64 for Ollama."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze(self, image_path: str, config_text: str, user_question: str) -> str:
        # 1. Retrieve relevant knowledge
        knowledge = self._query_knowledge(user_question)

        # 2. Build prompt for VLM
        prompt = f"""You are Mohsen's MLOps Co-Pilot, an expert in production AI systems.
You analyze architecture diagrams (provided as image) and CI/CD configurations.

## Diagram (attached)
## Configuration file content:
{config_text}

## Expert MLOps patterns (from Mohsen's knowledge base):
{knowledge}

## User question:
{user_question}

Provide a detailed, actionable answer. Reference specific technologies used in the diagram and config.
Mention latency budgets, specific tooling, and code patterns from Mohsen's expertise.
If the question is about latency, propose exact changes to reach <100ms p99."""

        # 3. Call Ollama multimodal
        image_b64 = self._encode_image(image_path)
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_b64]
            }]
        )
        return response["message"]["content"]
