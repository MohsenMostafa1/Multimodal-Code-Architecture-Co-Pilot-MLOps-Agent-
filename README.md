# Multimodal-Code-Architecture-Co-Pilot-MLOps-Agent-

# MLOps Best Practices – Mohsen Mostafa

## FastAPI Latency Optimisation
- Use async endpoints and httpx for asynchronous downstream calls.
- Stream responses with StreamingResponse for LLM token generation.
- Deploy with gunicorn + uvicorn workers behind nginx.
- Cache frequent model calls with Redis; TTL based on data freshness.
- Always profile with py-spy or Prometheus metrics.

## RAG System <100ms p99 Latency
- Embed with small, quantised models (e.g., BAAI/bge-small-en-v1.5).
- Use Pinecone with metadata filtering to avoid full index scans.
- Chunk documents in 256-token segments, overlap 20%.
- Pre-compute and cache frequently asked query embeddings.

## CI/CD Pipeline Tuning
- Parallelise Docker layer caching in multistage builds.
- Use BuildKit for faster builds.
- Limit resource per service in docker-compose (mem_limit, cpus).
- Monitor build times with Datadog/Grafana.

## Multimodal LLM Deployment
- Serve vision-language models (LLaVA, DeepSeek-VL) via Ollama with GPU.
- Resize images to ≤ 512px before sending to VLM to reduce latency.
