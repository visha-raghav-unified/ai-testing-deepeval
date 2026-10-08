
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric
)


def test_rag_metrics():

    # Step 1: Customer question
    question = "Can I return my shoes after 45 days?"

    # Step 2: Simulated chatbot response
    actual_answer = (
        "Yes, you can return your shoes within 60 days."
    )

    # Step 3: Simulated RAG retrieved context
    retrieved_documents = [
        "Eligible products can be returned within "
        "30 days of purchase."
    ]

    # Step 4: Create the evaluation test case
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        retrieval_context=retrieved_documents
    )

    # Step 5: Configure metrics
    metrics = [
        AnswerRelevancyMetric(
            threshold=0.8,
            include_reason=True
        ),
        FaithfulnessMetric(
            threshold=0.8,
            include_reason=True
        )
    ]

    # Step 6: Execute evaluations
    evaluate(
        test_cases=[test_case],
        metrics=metrics
    )


if __name__ == "__main__":
    test_rag_metrics()
