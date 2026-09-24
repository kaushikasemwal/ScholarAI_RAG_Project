"""
retrieve_test.py — Retrieval Debug Script
==========================================
Usage:
    python -m backend.rag.retrieve_test --user-id <user_id> --document-id <doc_id> --query "What is the main concept?"

Environment:
    Requires chroma_db to exist with indexed documents.
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.rag import (
    retrieve,
    RetrievalConfig,
    VectorStoreConfig,
    EmbeddingConfig,
    LocalVectorStore,
    BGEEmbeddings,
)


def main():
    parser = argparse.ArgumentParser(description="Test RAG retrieval")
    parser.add_argument("--user-id", required=True, help="User ID for isolation")
    parser.add_argument("--document-id", help="Document ID to search (optional)")
    parser.add_argument("--query", required=True, help="Query string")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    parser.add_argument("--vector-store-dir", default="./chroma_db", help="Chroma persist directory")
    parser.add_argument("--collection", default="scholarai_documents", help="Collection name")
    parser.add_argument("--threshold", type=float, help="Similarity threshold (0-1)")

    args = parser.parse_args()

    # Configure
    vector_store_config = VectorStoreConfig(
        persist_directory=args.vector_store_dir,
        collection_name=args.collection,
    )

    embedding_config = EmbeddingConfig()
    retrieval_config = RetrievalConfig(
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )

    # Create components
    vector_store = LocalVectorStore(vector_store_config)
    embeddings = BGEEmbeddings(embedding_config)

    print(f"\n{'='*60}")
    print(f"QUERY: {args.query}")
    print(f"USER: {args.user_id}")
    print(f"DOCUMENT: {args.document_id or 'ALL'}")
    print(f"TOP_K: {args.top_k}")
    print(f"{'='*60}\n")

    # Retrieve
    result = retrieve(
        query=args.query,
        user_id=args.user_id,
        document_id=args.document_id,
        top_k=args.top_k,
        vector_store=vector_store,
        embeddings=embeddings,
        retrieval_config=retrieval_config,
    )

    # Display results
    if not result.documents:
        print("NO RESULTS FOUND")
        return

    for i, doc in enumerate(result.documents):
        score = doc.metadata.get("similarity_score", 0)
        page = doc.metadata.get("page_number", "?")
        chunk_id = doc.metadata.get("chunk_id", "?")
        chunk_type = doc.metadata.get("chunk_type", "?")
        filename = doc.metadata.get("source_filename", "?")

        print(f"RESULT {i+1}")
        print(f"  Score:        {score:.4f}")
        print(f"  Document:     {filename}")
        print(f"  Page/Slide:   {page}")
        print(f"  Chunk ID:     {chunk_id}")
        print(f"  Chunk Type:   {chunk_type}")
        print(f"  Text Preview: {doc.page_content[:200]}...")
        print()

    print(f"Total results: {result.total_found}")


if __name__ == "__main__":
    main()