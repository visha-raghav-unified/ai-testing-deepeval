
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams


def test_return_policy_correctness():

    # Step 1: Prepare the test input
    question = "Can I return shoes after 45 days?"

    # Step 2: Simulated chatbot response
    actual_answer = (
        "Yes, you can return shoes within 60 days."
    )

    # Step 3: Expected business response
    expected_answer = (
        "Returns after 30 days are not allowed "
        "under the standard policy."
    )

    # Step 4: Create DeepEval test case
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer
    )

    # Step 5: Define correctness evaluation
    correctness_metric = GEval(
        name="Answer Correctness",
        criteria=(
            "Determine whether the actual answer "
            "is factually correct based on the "
            "expected answer."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        threshold=0.8
    )

    # Step 6: Assert quality
    assert_test(test_case, [correctness_metric])
