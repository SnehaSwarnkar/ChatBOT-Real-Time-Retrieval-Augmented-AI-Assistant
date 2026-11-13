
import ollama
import chromadb
import re

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ==========================================
# 1. CONFIGURATION
# ==========================================

conversation_history = []

MAX_HISTORY = 10

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2"

VECTOR_TOP_K = 20
FINAL_TOP_K = 5

RRF_K = 60

# ==========================================
# 2. CONNECT TO CHROMADB
# ==========================================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db_ollama"
)

# Use get_or_create_collection so the server can start
# even before documents have been ingested.
collection = chroma_client.get_or_create_collection(
    name="multi_format_documents",
    metadata={"hnsw:space": "cosine"}
)

print("\n================================")
print("       INITIALIZING RAG")
print("================================")

# ==========================================
# 3. TOKENIZATION
# ==========================================

def tokenize(text):

    return re.findall(
        r"\b\w+\b",
        text.lower()
    )


# ==========================================
# 4. INITIALIZE BM25 SEARCH
# ==========================================

# Module-level mutable state; populated by _init_bm25().
corpus_documents = []
corpus_metadata = []
corpus_ids = []
tokenized_corpus = []
bm25 = None
id_to_index = {}

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

print("\nLoading Cross-Encoder model...")
reranker = CrossEncoder(RERANKER_MODEL)
print("Cross-Encoder initialized successfully!")


def _init_bm25():
    """Load all ChromaDB documents into the BM25 index.

    Called lazily at retrieval time so the module imports
    cleanly even when the collection is empty.
    """
    global bm25, corpus_documents, corpus_metadata
    global corpus_ids, tokenized_corpus, id_to_index

    total_chunks = collection.count()

    if total_chunks == 0:
        return False

    print("\nInitializing BM25 keyword search...")

    all_records = collection.get(
        limit=total_chunks,
        include=["documents", "metadatas"]
    )

    corpus_documents = all_records["documents"]
    corpus_metadata = all_records["metadatas"]
    corpus_ids = all_records["ids"]

    if not corpus_documents:
        return False

    tokenized_corpus = [
        tokenize(doc)
        for doc in corpus_documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    id_to_index = {
        doc_id: index
        for index, doc_id in enumerate(corpus_ids)
    }

    print("BM25 initialized successfully!")
    print("Total indexed chunks:", len(corpus_documents))
    return True


# Try to initialise immediately (works if documents
# have already been ingested); silently deferred otherwise.
_init_bm25()

# ==========================================
# 5. HYBRID DOCUMENT RETRIEVAL
# ==========================================


def retrieve_documents(question, top_k=FINAL_TOP_K):

    if collection.count() == 0:
        return [], []

    # Lazily initialise BM25 if not done yet
    # (handles the case where documents were ingested
    # after the server started).
    if bm25 is None:
        if not _init_bm25():
            return [], []

    candidate_k = min(
        VECTOR_TOP_K,
        len(corpus_documents)
    )

    # --------------------------------------
    # A. VECTOR SEARCH
    # --------------------------------------

    print("\nPerforming vector search...")

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )

    query_embedding = response.embeddings[0]

    vector_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    vector_ids = vector_results["ids"][0]

    print("Vector candidates:", len(vector_ids))

    # --------------------------------------
    # B. BM25 KEYWORD SEARCH
    # --------------------------------------

    print("Performing BM25 keyword search...")

    tokenized_query = tokenize(question)

    bm25_scores = bm25.get_scores(tokenized_query)

    bm25_indices = sorted(
        [
            i
            for i, score in enumerate(bm25_scores)
            if score > 0
        ],
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:candidate_k]

    bm25_ids = [
        corpus_ids[i]
        for i in bm25_indices
    ]

    print("BM25 candidates:", len(bm25_ids))

    # --------------------------------------
    # C. RECIPROCAL RANK FUSION (RRF)
    # --------------------------------------

    print("Combining search results using RRF...")

    rrf_scores = {}

    # Vector rankings
    for rank, doc_id in enumerate(vector_ids, start=1):

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (RRF_K + rank)
        )

    # BM25 rankings
    for rank, doc_id in enumerate(bm25_ids, start=1):

        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + 1 / (RRF_K + rank)
        )

    # --------------------------------------
    # D. SELECT RRF CANDIDATES
    # --------------------------------------

    ranked_ids = sorted(
        rrf_scores.keys(),
        key=lambda doc_id: rrf_scores[doc_id],
        reverse=True
    )[:VECTOR_TOP_K]

    candidate_documents = []
    candidate_metadata = []

    for doc_id in ranked_ids:

        index = id_to_index.get(doc_id)

        if index is None:
            continue

        candidate_documents.append(
            corpus_documents[index]
        )

        candidate_metadata.append(
            corpus_metadata[index] or {}
        )

    print(
        "RRF candidates:",
        len(candidate_documents)
    )

    if not candidate_documents:
        return [], []

    # --------------------------------------
    # E. CROSS-ENCODER RE-RANKING
    # --------------------------------------

    print("Applying Cross-Encoder re-ranking...")

    pairs = [
        (question, document)
        for document in candidate_documents
    ]

    rerank_scores = reranker.predict(pairs)

    reranked_results = sorted(
        zip(
            rerank_scores,
            candidate_documents,
            candidate_metadata
        ),
        key=lambda item: item[0],
        reverse=True
    )

    # --------------------------------------
    # F. SELECT FINAL TOP DOCUMENTS
    # --------------------------------------

    final_results = reranked_results[:top_k]

    documents = [
        item[1]
        for item in final_results
    ]

    metadata = [
        item[2]
        for item in final_results
    ]

    print(
        "Cross-Encoder candidates:",
        len(rerank_scores)
    )

    print(
        "Final selected chunks:",
        len(documents)
    )

    return documents, metadata

# ==========================================
# 6. GENERATE ANSWERS USING LLAMA
# ==========================================

def generate_answer(question, documents, metadata):

    if not documents:

        return (
            "I couldn't find relevant information "
            "in your uploaded documents."
        )

    # --------------------------------------
    # PREPARE DOCUMENT CONTEXT
    # --------------------------------------

    context = ""

    for i, (doc, meta) in enumerate(
        zip(documents, metadata),
        start=1
    ):

        source = meta.get(
            "source",
            "Unknown document"
        )

        page = meta.get(
            "page",
            meta.get("slide", "Unknown")
        )

        context += f"""
Source {i}: {source}

Page/Slide: {page}

Content:
{doc}

--------------------------------
"""

    # --------------------------------------
    # CREATE PROMPT
    # --------------------------------------

    prompt = f"""
You are an intelligent document-based AI assistant.

Your task is to answer questions using ONLY
the information available in the provided context.

INSTRUCTIONS:

1. Give accurate and relevant answers.
2. Never invent information.
3. Do not use outside knowledge as document evidence.
4. If information is unavailable, clearly state:
   "I couldn't find this information in the uploaded documents."
5. Explain technical concepts in simple language.
6. Use headings, bullet points and examples when useful.
7. For comparisons, organize the answer clearly.
8. Use the conversation history to understand follow-up
   questions, but base factual answers on the context.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    # --------------------------------------
    # LLAMA RESPONSE
    # --------------------------------------

    response = ollama.chat(
        model=LLM_MODEL,

        messages=[
            {
                "role": "system",
                "content": (
                    "You are a reliable RAG document assistant. "
                    "Use the supplied document context as the "
                    "only source of factual evidence. "
                    "Never fabricate document information."
                )
            },

            *conversation_history,

            {
                "role": "user",
                "content": prompt
            }
        ],

        stream=False
    )

    return response.message.content


# ==========================================
# 7. QUERY REWRITING
# ==========================================

def rewrite_question(question):

    if not conversation_history:
        return question

    response = ollama.chat(

        model=LLM_MODEL,

        messages=[

            {
                "role": "system",

                "content": """
You are a search query rewriting assistant.

Convert the user's latest question into a
standalone search query.

Use conversation history to understand references such as:

- It
- They
- The second one
- Explain more
- That concept
- Compare both

Preserve the original meaning.

Return ONLY the rewritten search query.
Do not answer the question.
"""
            },

            *conversation_history,

            {
                "role": "user",
                "content": question
            }
        ],

        stream=False
    )

    rewritten = (
        response.message.content.strip()
    )

    print(
        "\nRewritten search query:",
        rewritten
    )

    return rewritten or question


# ==========================================
# 8. COMPLETE RAG PIPELINE
# ==========================================

def ask_question(question):

    print("\nUnderstanding your question...")

    # Step 1: Query rewriting
    search_query = rewrite_question(question)

    # Step 2: Hybrid retrieval
    print("\nSearching your documents...")

    documents, metadata = retrieve_documents(
        search_query
    )

    print(
        f"\nRetrieved {len(documents)} relevant chunks."
    )

    # Step 3: Answer generation
    answer = generate_answer(
        question,
        documents,
        metadata
    )

    # Step 4: Save conversation history
    conversation_history.append({

        "role": "user",
        "content": question

    })

    conversation_history.append({

        "role": "assistant",
        "content": answer

    })

    # Keep only recent messages
    if len(conversation_history) > MAX_HISTORY:

        conversation_history[:] = (
            conversation_history[-MAX_HISTORY:]
        )

    return answer, metadata


# ==========================================
# 9. DISPLAY SOURCES
# ==========================================

def display_sources(metadata):

    print("\n--------------- SOURCES ---------------")

    if not metadata:

        print("No sources found.")
        return

    seen_sources = set()

    for meta in metadata:

        source = meta.get(
            "source",
            "Unknown"
        )

        page = meta.get(
            "page",
            meta.get("slide", "Unknown")
        )

        source_key = (source, page)

        if source_key not in seen_sources:

            print(
                f"Document: {source} | "
                f"Page/Slide: {page}"
            )

            seen_sources.add(source_key)


# ==========================================
# 10. CHAT INTERFACE
# ==========================================

if __name__ == "__main__":

    print("\n================================")
    print("       RAG DOCUMENT CHATBOT")
    print("================================")

    print(
        "\nDocuments loaded:",
        collection.count()
    )

    print(
        "LLM:",
        LLM_MODEL
    )

    print(
        "Embedding model:",
        EMBEDDING_MODEL
    )

    print(
        "Retrieval: Vector + BM25 + RRF"
    )

    print("\nAsk questions about your uploaded PDFs and PPTs.")

    print("Type 'exit' to stop.")

    print("\n================================\n")

    while True:

        question = input("You: ").strip()

        if question.lower() in [
            "exit",
            "quit"
        ]:

            print("\nChatbot closed.")

            break

        if not question:
            continue

        try:

            answer, metadata = ask_question(
                question
            )

            print("\nAI:", answer)

            display_sources(metadata)

            print("\n" + "=" * 50 + "\n")

        except KeyboardInterrupt:

            print("\nChatbot interrupted.")

            break

        except Exception as e:

            print(
                "\nError:",
                str(e)
            )
