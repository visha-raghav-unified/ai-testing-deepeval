
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
    ContextualRecallMetric
)


def test_return_policy_retrieval():

    # Step 1: User question
    question = (
        "What is the return period, what proof is needed, "
        "and how will my refund be issued?"
    )

    # Step 2: Expected answer based on business rules
    expected_answer = (
        "Eligible products can be returned within 30 days. "
        "Proof of purchase is required. "
        "Refunds are issued to the original payment method."
    )

    # Step 3: Actual chatbot response (simulated)
    actual_answer = expected_answer

    # Step 4: Documents retrieved by the RAG system
    # IMPORTANT: List order represents retrieval ranking
    retrieved_documents = [
        "Eligible products can be returned within "
        "30 days of purchase.",

        "Proof of purchase is required for returns.",

        "Refunds are issued to the original "
        "payment method."
    ]

    # Step 5: Build DeepEval test case
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=retrieved_documents
    )

    # Step 6: Define retriever metrics
    metrics = [

        # Are useful documents ranked first?
        ContextualPrecisionMetric(
            threshold=0.8,
            include_reason=True
        ),

        # Is retrieved content relevant?
        ContextualRelevancyMetric(
            threshold=0.8,
            include_reason=True
        ),

        # Are all expected facts retrieved?
        ContextualRecallMetric(
            threshold=0.8,
            include_reason=True
        )
    ]

    # Step 7: Evaluate and assert all metrics
    assert_test(test_case, metrics)
