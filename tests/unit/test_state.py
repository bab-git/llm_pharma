"""
Unit tests for the State module.
"""

from backend.my_agent.State import AgentState, create_agent_state


class TestAgentState:
    """Test cases for AgentState TypedDict."""

    def test_agent_state_creation(self):
        """Test that AgentState can be created with valid data."""
        state: AgentState = {
            "last_node": "test_node",
            "patient_prompt": "test prompt",
            "patient_id": 123,
            "patient_data": {"name": "John Doe"},
            "patient_profile": "test profile",
            "policy_eligible": True,
            "policies": [{"id": 1, "name": "Policy A"}],
            "checked_policy": {"id": 1, "name": "Policy A"},
            "unchecked_policies": [{"id": 2, "name": "Policy B"}],
            "policy_qs": "test questions",
            "rejection_reason": "",
            "revision_number": 1,
            "max_revisions": 3,
            "trial_searches": 0,
            "max_trial_searches": 2,
            "trials": [],
            "relevant_trials": [],
            "ask_expert": "",
            "trial_found": False,
            "error_message": "",
            "selected_model": "test-model",
        }

        assert state["patient_id"] == 123
        assert state["policy_eligible"] is True
        assert len(state["policies"]) == 1


class TestCreateAgentState:
    """Test cases for create_agent_state function."""

    def test_create_agent_state_returns_default_values(self):
        """Test that create_agent_state returns expected default values."""
        state = create_agent_state()

        # Test default string values
        assert state["last_node"] == ""
        assert state["patient_prompt"] == ""
        assert state["patient_profile"] == ""
        assert state["policy_qs"] == ""
        assert state["rejection_reason"] == ""
        assert state["ask_expert"] == ""
        assert state["error_message"] == ""
        assert state["selected_model"] == "llama-3.3-70b-versatile"

        # Test default numeric values
        assert state["patient_id"] == 0
        assert state["revision_number"] == 0
        assert state["max_revisions"] == 3
        assert state["trial_searches"] == 0
        assert state["max_trial_searches"] == 2

        # Test default boolean values
        assert state["policy_eligible"] is False
        assert state["trial_found"] is False

        # Test default collection values
        assert state["patient_data"] == {}
        assert state["policies"] == []
        assert state["checked_policy"] is None
        assert state["unchecked_policies"] == []
        assert state["trials"] == []
        assert state["relevant_trials"] == []

    def test_create_agent_state_returns_new_instance(self):
        """Test that create_agent_state returns a new instance each time."""
        state1 = create_agent_state()
        state2 = create_agent_state()

        # Modify one state
        state1["patient_id"] = 999

        # Verify the other state is unchanged
        assert state2["patient_id"] == 0
        assert state1["patient_id"] == 999
