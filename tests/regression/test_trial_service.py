import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.my_agent.patient_collector import create_agent_state
from backend.my_agent.trial_service import grade_trials_node, trial_search_node


# Mock a minimal trial object for grading test
class MockTrial:
    def __init__(self, page_content, diseases, nctid):
        self.page_content = page_content
        self.metadata = {"diseases": diseases, "nctid": nctid}


def test_trial_search_valid():
    print("Test: trial_search_node with valid patient profile")
    state = create_agent_state()
    state["patient_profile"] = "Patient has leukemia and is taking imatinib."
    result = trial_search_node(state)
    print("Result:", result)
    assert "trials" in result
    assert isinstance(result["trials"], list)
    print("✅ trial_search_node valid profile test passed!\n")


def test_trial_search_empty():
    print("Test: trial_search_node with empty patient profile")
    state = create_agent_state()
    state["patient_profile"] = ""
    result = trial_search_node(state)
    print("Result:", result)
    assert result["trials"] == []
    print("✅ trial_search_node empty profile test passed!\n")


def test_grade_trials_node():
    print("Test: grade_trials_node with mock trial")
    state = create_agent_state()
    state["patient_profile"] = "Patient has leukemia and is taking imatinib."
    # Use a mock trial object
    trial = MockTrial(
        page_content="Inclusion: leukemia. Exclusion: none.",
        diseases="leukemia",
        nctid="NCT00000001",
    )
    state["trials"] = [trial]
    result = grade_trials_node(state)
    print("Result:", result)
    assert "relevant_trials" in result
    assert isinstance(result["relevant_trials"], list)
    print("✅ grade_trials_node test passed!\n")


if __name__ == "__main__":
    test_trial_search_valid()
    test_trial_search_empty()
    test_grade_trials_node()
    print("All regression tests for trial_service passed!")
