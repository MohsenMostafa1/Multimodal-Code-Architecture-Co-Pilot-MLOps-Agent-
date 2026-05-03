import os
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "mlops-agent-knowledge"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

def ingest():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    index = pc.Index(INDEX_NAME)
    
    loader = UnstructuredMarkdownLoader("knowledge_base.md")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=30)
    chunks = splitter.split_documents(docs)
    
    embeddings = HuggingFaceBgeEmbeddings(model_name=EMBED_MODEL)
    vectors = []
    for i, chunk in enumerate(chunks):
        vec = embeddings.embed_query(chunk.page_content)
        vectors.append((f"chunk-{i}", vec, {"text": chunk.page_content}))
    
    index.upsert(vectors)
    print(f"Ingested {len(vectors)} chunks into Pinecone.")

if __name__ == "__main__":
    ingest()
