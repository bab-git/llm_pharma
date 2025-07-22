from typing import TypedDict, List

class AgentState(TypedDict):
    """State definition for the LLM Pharma workflow agent."""

    last_node: str
    patient_prompt: str
    patient_id: int
    patient_data: dict
    patient_profile: str
    policy_eligible: bool
    policies: List
    checked_policy: dict
    unchecked_policies: List
    policy_qs: str
    rejection_reason: str
    revision_number: int
    max_revisions: int
    trial_searches: int
    max_trial_searches: int
    trials: List
    relevant_trials: list[dict]
    ask_expert: str
    trial_found: bool
    error_message: str
    selected_model: str 


def create_agent_state() -> AgentState:
    """
    Create the initial agent state for the LLM Pharma workflow.

    Returns:
        AgentState: Initial state with default values
    """
    return {
        "last_node": "",
        "patient_prompt": "",
        "patient_id": 0,
        "patient_data": {},
        "patient_profile": "",
        "policy_eligible": False,
        "policies": [],
        "checked_policy": None,
        "unchecked_policies": [],
        "policy_qs": "",
        "rejection_reason": "",
        "revision_number": 0,
        "max_revisions": 3,
        "trial_searches": 0,
        "max_trial_searches": 2,
        "trials": [],
        "relevant_trials": [],
        "ask_expert": "",
        "trial_found": False,
        "error_message": "",
        "selected_model": "llama-3.3-70b-versatile",
    } 