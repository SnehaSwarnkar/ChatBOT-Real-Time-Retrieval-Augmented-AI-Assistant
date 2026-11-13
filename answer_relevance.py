
import json
import re
import ollama

from chat import (
    retrieve_documents,
    generate_answer,
    LLM_MODEL
)


# ==========================================
# 1. EVALUATION DATASET
# ==========================================

EVALUATION_QUESTIONS = [

    "What are the main topics covered in Unit 1?",

    "Explain the important concepts in Unit 2.",

    "What concepts are discussed in Unit 3?",

    "Explain the important topics in Unit 4.",

    "Summarize the main concepts of Unit 5."

]


# ==========================================
# 2. JSON EXTRACTION
# ==========================================

def extract_json(text):

    cleaned = re.sub(
        r"```(?:json)?|```",
        "",
        text
    ).strip()

    try:
        return json.loads(cleaned)

    except json.JSONDecodeError:

        match = re.search(
            r"\{.*\}",
            cleaned,
            re.DOTALL
        )

        if match:
            return json.loads(match.group(0))

        raise ValueError(
            "Invalid JSON response from evaluator"
        )


# ==========================================
# 3. ANSWER RELEVANCE EVALUATION
# ==========================================

def evaluate_answer_relevance(question, answer):

    prompt = f"""
You are an expert AI response evaluator.

Your task is to evaluate how relevant an AI-generated
answer is to the user's original question.

Evaluate ONLY whether the answer addresses the
question directly and meaningfully.

Use this scoring scale:

5 = Completely relevant and directly answers the question.
4 = Mostly relevant with minor missing information.
3 = Partially relevant but misses important aspects.
2 = Mostly irrelevant or only loosely related.
1 = Completely irrelevant or does not answer the question.

Return ONLY valid JSON in this format:

{{
    "score": 5,
    "reason": "The answer directly addresses the question."
}}

QUESTION:
{question}

GENERATED ANSWER:
{answer}
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

        options={
            "temperature": 0
        },

        stream=False
    )

    result = extract_json(
        response.message.content
    )

    score = result.get("score")

    if (
        isinstance(score, bool)
        or not isinstance(score, (int, float))
        or not 1 <= score <= 5
    ):
        raise ValueError("Invalid relevance score")

    return score, result.get("reason", "")


# ==========================================
# 4. COMPLETE EVALUATION
# ==========================================

def run_answer_relevance_evaluation():

    scores = []

    print("\n======================================")
    print("       ANSWER RELEVANCE EVALUATION")
    print("======================================")

    for index, question in enumerate(
        EVALUATION_QUESTIONS,
        start=1
    ):

        print(f"\nQuestion {index}: {question}")

        try:

            # Retrieve relevant documents
            documents, metadata = retrieve_documents(
                question
            )

            if not documents:
                print("No documents retrieved. Skipping.")
                continue

            # Generate chatbot answer
            answer = generate_answer(
                question,
                documents,
                metadata
            )

            print("\nGenerated Answer:")
            print(answer)

            # Evaluate answer relevance
            score, reason = evaluate_answer_relevance(
                question,
                answer
            )

            scores.append(score)

            print("\nRelevance Score:", score, "/ 5")
            print("Reason:", reason)

        except Exception as error:

            print("\nEvaluation Error:", error)

    # ======================================
    # 5. FINAL RESULTS
    # ======================================

    print("\n======================================")
    print("       FINAL RELEVANCE RESULTS")
    print("======================================")

    if scores:

        average_score = sum(scores) / len(scores)

        percentage = (
            average_score / 5
        ) * 100

        print(
            "Questions Evaluated:",
            len(scores)
        )

        print(
            "Average Relevance Score:",
            round(average_score, 2),
            "/ 5"
        )

        print(
            "Answer Relevance Percentage:",
            round(percentage, 2),
            "%"
        )

    else:

        print("No valid evaluation results.")

    print("\nEVALUATION COMPLETED!")


# ==========================================
# 6. MAIN
# ==========================================

if __name__ == "__main__":

    run_answer_relevance_evaluation()
