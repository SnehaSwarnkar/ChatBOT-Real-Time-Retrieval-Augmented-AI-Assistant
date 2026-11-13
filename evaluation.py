
import os

from chat import retrieve_documents, FINAL_TOP_K


# ==========================================
# 1. EVALUATION DATASET
# ==========================================

EVALUATION_DATASET = [

    {
        "question": "What are the main topics covered in Unit 1?",
        "relevant": ["doc1.pptx"]
    },

    {
        "question": "Explain the important concepts in Unit 2.",
        "relevant": ["doc2.pptx"]
    },

    {
        "question": "What concepts are discussed in Unit 3?",
        "relevant": ["doc3.pptx"]
    },

    {
        "question": "Explain the important topics in Unit 4.",
        "relevant": ["doc4.pptx"]
    },

    {
        "question": "Summarize the main concepts of Unit 5.",
        "relevant": ["doc5.pptx"]
    }

]


# ==========================================
# 2. NORMALIZE SOURCE METADATA
# ==========================================

def normalize_source(source):

    return os.path.basename(
        str(source)
    ).strip().lower()


# ==========================================
# 3. PRECISION@K
# ==========================================

def precision_at_k(retrieved, relevant, k):

    retrieved = retrieved[:k]

    if not retrieved or k <= 0:
        return 0.0

    hits = sum(
        1 for source in retrieved
        if source in relevant
    )

    return hits / len(retrieved)


# ==========================================
# 4. RECALL@K
# ==========================================

def recall_at_k(retrieved, relevant, k):

    retrieved = retrieved[:k]

    if not relevant:
        return 0.0

    # Count unique relevant documents retrieved
    hits = len(
        set(retrieved) & relevant
    )

    return hits / len(relevant)


# ==========================================
# 5. MEAN RECIPROCAL RANK
# ==========================================

def reciprocal_rank(retrieved, relevant):

    for rank, source in enumerate(
        retrieved,
        start=1
    ):

        if source in relevant:
            return 1 / rank

    return 0.0


# ==========================================
# 6. COMPLETE RAG EVALUATION
# ==========================================

def evaluate_rag():

    precision_scores = []
    recall_scores = []
    reciprocal_scores = []

    print("\n" + "=" * 50)
    print("          RAG EVALUATION")
    print("=" * 50)

    print(
        "\nTotal evaluation questions:",
        len(EVALUATION_DATASET)
    )

    print(
        "Evaluation cutoff K:",
        FINAL_TOP_K
    )

    # Process each evaluation question
    for index, test in enumerate(
        EVALUATION_DATASET,
        start=1
    ):

        question = test["question"]

        # Ground-truth relevant documents
        relevant = {
            normalize_source(source)
            for source in test["relevant"]
        }

        print("\n" + "-" * 50)
        print(f"TEST QUESTION {index}")
        print("-" * 50)

        print("Question:", question)

        print(
            "Expected documents:",
            relevant
        )

        # ----------------------------------
        # RETRIEVE DOCUMENTS
        # ----------------------------------

        documents, metadata = retrieve_documents(
            question,
            top_k=FINAL_TOP_K
        )

        # Extract unique source names
        retrieved = [
            normalize_source(
                meta.get("source", "Unknown")
            )
            for meta in metadata
        ]

        # ----------------------------------
        # CALCULATE METRICS
        # ----------------------------------

        precision = precision_at_k(
            retrieved,
            relevant,
            FINAL_TOP_K
        )

        recall = recall_at_k(
            retrieved,
            relevant,
            FINAL_TOP_K
        )

        mrr = reciprocal_rank(
            retrieved,
            relevant
        )

        # Store scores
        precision_scores.append(precision)
        recall_scores.append(recall)
        reciprocal_scores.append(mrr)

        # ----------------------------------
        # DISPLAY RESULTS
        # ----------------------------------

        print("\nRetrieved documents:")

        for rank, source in enumerate(
            retrieved,
            start=1
        ):

            print(f"{rank}. {source}")

        print("\nEvaluation metrics:")

        print(
            f"Precision@{FINAL_TOP_K}: {precision:.4f}"
        )

        print(
            f"Recall@{FINAL_TOP_K}: {recall:.4f}"
        )

        print(
            f"MRR: {mrr:.4f}"
        )

    # ======================================
    # 7. FINAL AVERAGE RESULTS
    # ======================================

    total = len(EVALUATION_DATASET)

    if total == 0:
        print("\nNo evaluation questions available.")
        return

    average_precision = (
        sum(precision_scores) / total
    )

    average_recall = (
        sum(recall_scores) / total
    )

    mean_reciprocal_rank = (
        sum(reciprocal_scores) / total
    )

    print("\n" + "=" * 50)
    print("           FINAL RESULTS")
    print("=" * 50)

    print(
        f"\nAverage Precision@{FINAL_TOP_K}: "
        f"{average_precision:.4f}"
    )

    print(
        f"Average Recall@{FINAL_TOP_K}: "
        f"{average_recall:.4f}"
    )

    print(
        f"Mean Reciprocal Rank: "
        f"{mean_reciprocal_rank:.4f}"
    )

    print("\n" + "=" * 50)
    print("       EVALUATION COMPLETED")
    print("=" * 50)


# ==========================================
# 8. EXECUTION
# ==========================================

if __name__ == "__main__":

    evaluate_rag()
