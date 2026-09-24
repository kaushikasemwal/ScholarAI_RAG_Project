"""
generate_test.py — Grounded Generation Demo
============================================
Demonstrates the full Phase 2 pipeline:

Query → Retriever → Top-K Chunks → Prompt → FLAN-T5 → Structured Question

Usage:
    python -m backend.rag.generate_test \
        --user-id user-123 \
        --document-id doc-456 \
        --query "What is machine learning?" \
        --top-k 5
"""

import argparse
import sys
import json
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
    generate_grounded_question,
    GeneratorConfig,
    GroundedQuizGenerator,
)


def main():
    parser = argparse.ArgumentParser(description="Test RAG grounded generation")
    parser.add_argument("--user-id", required=True, help="User ID for isolation")
    parser.add_argument("--document-id", help="Document ID to search (optional)")
    parser.add_argument("--query", required=True, help="Query/question specification")
    parser.add_argument("--top-k", type=int, default=5, help="Number of retrieved chunks")
    parser.add_argument("--vector-store-dir", default="./chroma_db", help="Chroma persist directory")
    parser.add_argument("--collection", default="scholarai_documents", help="Collection name")
    parser.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--topic-focus", default="the key concepts", help="Question focus")
    parser.add_argument("--show-prompt", action="store_true", help="Display the full prompt")
    parser.add_argument("--show-context", action="store_true", help="Display retrieved context")

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"QUERY: {args.query}")
    print(f"USER: {args.user_id}")
    print(f"DOCUMENT: {args.document_id or 'ALL'}")
    print(f"TOP_K: {args.top_k}")
    print(f"DIFFICULTY: {args.difficulty}")
    print(f"TOPIC FOCUS: {args.topic_focus}")
    print(f"{'='*70}\n")

    # Configure components
    vector_store_config = VectorStoreConfig(
        persist_directory=args.vector_store_dir,
        collection_name=args.collection,
    )

    embedding_config = EmbeddingConfig()
    retrieval_config = RetrievalConfig(top_k=args.top_k)

    # Create components
    vector_store = LocalVectorStore(vector_store_config)
    embeddings = BGEEmbeddings(embedding_config)

    # Step 1: Retrieve
    print("📥 RETRIEVING EVIDENCE...")
    result = retrieve(
        query=args.query,
        user_id=args.user_id,
        document_id=args.document_id,
        top_k=args.top_k,
        vector_store=vector_store,
        embeddings=embeddings,
    )

    if not result.documents:
        print("❌ NO DOCUMENTS RETRIEVED")
        return

    print(f"✅ Retrieved {result.total_found} chunks\n")

    if args.show_context:
        print("📄 RETRIEVED CONTEXT:")
        print("-" * 50)
        for i, doc in enumerate(result.documents, 1):
            score = doc.metadata.get("similarity_score", 0)
            page = doc.metadata.get("page_number", "?")
            chunk_id = doc.metadata.get("chunk_id", "?")
            filename = doc.metadata.get("source_filename", "?")
            preview = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
            print(f"  [{i}] Score: {score:.3f} | Page: {page} | Chunk: {chunk_id} | File: {filename}")
            print(f"      {preview}")
            print()

    # Step 2: Generate grounded question
    print("🧠 GENERATING GROUNDED QUESTION...")
    gen_config = GeneratorConfig(
        max_context_chunks=args.top_k,
        difficulty=args.difficulty,
        topic_focus=args.topic_focus,
    )
    generator = GroundedQuizGenerator(gen_config)

    gen_result = generator.generate_grounded_question(
        retrieved_documents=result.documents,
        topic_focus=args.topic_focus,
        difficulty=args.difficulty,
        user_id=args.user_id,
        document_id=args.document_id,
    )

    # Step 3: Display results
    print(f"\n{'='*70}")
    print("GENERATION RESULT")
    print(f"{'='*70}\n")

    if not gen_result.success:
        print(f"❌ GENERATION FAILED: {gen_result.error}")
        if gen_result.raw_model_output:
            print(f"\nRaw model output:\n{gen_result.raw_model_output}")
        if gen_result.prompt_used and args.show_prompt:
            print(f"\nPrompt used:\n{gen_result.prompt_used}")
        return

    q = gen_result.question
    print("✅ GENERATION SUCCESSFUL\n")
    print(f"QUESTION: {q.question}\n")
    print("OPTIONS:")
    for i, opt in enumerate(q.options):
        marker = " ✓" if opt == q.correct_answer else ""
        print(f"  {chr(65+i)}) {opt}{marker}")
    print(f"\nCORRECT ANSWER: {q.correct_answer}")
    print(f"\nEXPLANATION: {q.explanation}\n")

    # Provenance
    print("📍 SOURCE PROVENANCE:")
    print("-" * 50)
    for i, prov in enumerate(gen_result.provenance, 1):
        print(f"  [{i}] Document: {prov.get('document_id')}")
        print(f"      File: {prov.get('source_filename')}")
        print(f"      Page/Slide: {prov.get('page_or_slide')}")
        print(f"      Chunk ID: {prov.get('chunk_id')}")
        print(f"      Score: {prov.get('similarity_score', 'N/A')}")
        print()

    if args.show_prompt and gen_result.prompt_used:
        print("📝 PROMPT USED:")
        print("-" * 50)
        print(gen_result.prompt_used[:2000] + ("..." if len(gen_result.prompt_used) > 2000 else ""))

    # Retrieval metadata
    print("\n📊 RETRIEVAL METADATA:")
    print("-" * 50)
    for k, v in gen_result.retrieval_metadata.items():
        print(f"  {k}: {v}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()