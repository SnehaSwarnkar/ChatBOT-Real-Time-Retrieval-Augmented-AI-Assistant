
import ollama
import json

from chat import (
    retrieve_documents,
    generate_answer,
    FINAL_TOP_K,
    LLM_MODEL
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

EVALUATION_DATASET = [
    "What are the main topics covered in Unit 1?",
    "Explain the important concepts in Unit 2.",
    "What concepts are discussed in Unit 3?",
    "Explain the important topics in Unit 4.",
    "Summarize the main concepts of Unit 5."
]


# ==========================================
# 2. FAITHFULNESS EVALUATION
# ==========================================

def evaluate_faithfulness(question, answer, documents, metadata):

    if not documents or not answer.strip():
        return {
            "score": 0,
            "supported": 0,
            "total": 0,
            "unsupported": []
        }

    # Prepare retrieved context
    context = ""

    for i, (doc, meta) in enumerate(
        zip(documents, metadata), start=1
    ):
        source = meta.get("source", "Unknown")
        page = meta.get(
            "page",
            meta.get("slide", "Unknown")
        )

        context += f"""
Source: {source}
Page/Slide: {page}
Content:
{doc}

"""

    # Ask Llama to evaluate answer faithfulness
    prompt = f"""
You are an expert RAG evaluation system.

Evaluate whether the answer is supported by
the retrieved document context.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

INSTRUCTIONS:

1. Break the generated answer into individual
   factual claims.
2. Check each claim against the retrieved context.
3. Mark a claim supported only if the context
   provides evidence for it.
4. Do not use outside knowledge.
5. Do not assume unsupported information is true.
6. Return valid JSON only.

Required JSON format:

{{
    "claims": [
        {{
            "claim": "Example factual claim",
            "supported": true,
            "evidence": "Evidence from context"
        }}
    ],
    "overall_reason": "Short explanation"
}}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        stream=False
    )

    try:
        result = json.loads(
            response.message.content
        )
    except json.JSONDecodeError:
        # Fallback: attempt to extract a JSON object from
        # the response using a simple regex.
        import re
        match = re.search(
            r"\{.*\}",
            response.message.content,
            re.DOTALL
        )
        if match:
            result = json.loads(match.group(0))
        else:
            raise ValueError(
                "LLM returned invalid JSON for faithfulness evaluation."
            )

    claims = result.get("claims", [])

    total = len(claims)

    supported = sum(
        1 for claim in claims
        if claim.get("supported") is True
    )

    unsupported = [
        claim for claim in claims
        if claim.get("supported") is not True
    ]

    score = (
        supported / total
        if total > 0 else 0
    )

    return {
        "score": score,
        "supported": supported,
        "total": total,
        "unsupported": unsupported,
        "reason": result.get("overall_reason", "")
    }


# ==========================================
# 3. COMPLETE EVALUATION PIPELINE
# ==========================================

def run_faithfulness_evaluation():

    scores = []

    print("\n======================================")
    print("       FAITHFULNESS EVALUATION")
    print("======================================")

    print("LLM:", LLM_MODEL)
    print("Top K:", FINAL_TOP_K)

    for index, question in enumerate(
        EVALUATION_DATASET, start=1
    ):

        print(f"\nTest Question {index}: {question}")

        # Retrieve relevant documents
        documents, metadata = retrieve_documents(
            question,
            top_k=FINAL_TOP_K
        )

        print("Retrieved chunks:", len(documents))

        if not documents:
            print("No relevant documents found.")
            scores.append(0)
            continue

        # Generate answer
        answer = generate_answer(
            question,
            documents,
            metadata
        )

        print("\nGenerated Answer:")
        print(answer)

        # Evaluate faithfulness
        print("\nEvaluating faithfulness...")

        result = evaluate_faithfulness(
            question,
            answer,
            documents,
            metadata
        )

        scores.append(result["score"])

        print("\n--- FAITHFULNESS RESULTS ---")

        print(
            f"Supported Claims: "
            f"{result['supported']}/{result['total']}"
        )

        print(
            f"Faithfulness Score: "
            f"{result['score']:.4f}"
        )

        if result["unsupported"]:

            print("\nUnsupported Claims:")

            for claim in result["unsupported"]:
                print("-", claim.get("claim", ""))

        print("\nReason:", result.get("reason", ""))

        print("\n" + "=" * 50)

    # ======================================
    # 4. FINAL RESULTS
    # ======================================

    if not scores:
        print("No questions evaluated.")
        return

    average_score = sum(scores) / len(scores)

    print("\n======================================")
    print("       FINAL FAITHFULNESS RESULTS")
    print("======================================")

    print("Questions Evaluated:", len(scores))

    print(
        "Average Faithfulness:",
        round(average_score, 4)
    )

    print(
        "Faithfulness Percentage:",
        round(average_score * 100, 2),
        "%"
    )

    print("\nEVALUATION COMPLETED!")


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    run_faithfulness_evaluation()
