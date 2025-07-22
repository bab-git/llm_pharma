#!/usr/bin/env python3
"""
Demo Graph Module for LLM Pharma Frontend

This module provides dummy graph functionality for testing and demonstration purposes.
It can be imported by the main app when running in demo mode.
"""

import sqlite3
from typing import List

from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict


def create_demo_agent_state():
    """
    Create the AgentState class for demo purposes.
    Based on the notebook code snippet.
    """

    class AgentState(TypedDict):
        last_node: str
        patient_prompt: str
        patient_id: int
        patient_data: dict
        patient_profile: str
        policy_eligible: bool
        policies: List[Document]
        checked_policy: Document
        unchecked_policies: List[Document]
        policy_qs: str
        rejection_reason: str
        revision_number: int
        max_revisions: int
        trial_searches: int
        max_trial_searches: int
        trials: List[Document]
        relevant_trials: list[dict]
        ask_expert: str

    return AgentState


def create_demo_node_functions():
    """
    Create dummy node functions for testing the dashboard GUI.
    """

    def dummy_patient_collector(state):
        """Dummy patient collector node"""
        return {
            "last_node": "patient_collector",
            "patient_data": {"age": 45, "condition": "test"},
            "patient_profile": "Test patient profile",
            "patient_id": 1,
            "revision_number": 1,
            "policy_eligible": "N/A",
        }

    def dummy_policy_search(state):
        """Dummy policy search node"""
        return {
            "last_node": "policy_search",
            "policies": [],
            "unchecked_policies": [],
        }

    def dummy_policy_evaluator(state):
        """Dummy policy evaluator node"""
        return {
            "last_node": "policy_evaluator",
            "policy_eligible": True,
            "rejection_reason": "N/A",
            "revision_number": 1,
            "checked_policy": None,
            "policy_qs": "",
            "unchecked_policies": [],
        }

    def dummy_trial_search(state):
        """Dummy trial search node"""
        return {
            "last_node": "trial_search",
            "trials": [],
            "trial_searches": 1,
        }

    def dummy_grade_trials(state):
        """Dummy grade trials node"""
        return {"last_node": "grade_trials", "relevant_trials": []}

    def dummy_profile_rewriter(state):
        """Dummy profile rewriter node"""
        return {
            "last_node": "profile_rewriter",
            "patient_profile": "Updated test patient profile",
        }

    return {
        "patient_collector": dummy_patient_collector,
        "policy_search": dummy_policy_search,
        "policy_evaluator": dummy_policy_evaluator,
        "trial_search": dummy_trial_search,
        "grade_trials": dummy_grade_trials,
        "profile_rewriter": dummy_profile_rewriter,
    }


def create_demo_conditional_functions():
    """
    Create dummy conditional functions for workflow transitions.
    """

    def dummy_should_continue_patient(state):
        return "policy_search"

    def dummy_should_continue_policy(state):
        return "trial_search"

    def dummy_should_continue_trials(state):
        return END

    return {
        "should_continue_patient": dummy_should_continue_patient,
        "should_continue_policy": dummy_should_continue_policy,
        "should_continue_trials": dummy_should_continue_trials,
    }


def create_demo_graph():
    """
    Create a complete dummy graph for testing the dashboard GUI.
    Based on the notebook code snippets.
    """
    try:
        # Create AgentState
        AgentState = create_demo_agent_state()
        print("✅ Demo AgentState created")

        # Create node functions
        node_functions = create_demo_node_functions()
        print("✅ Demo node functions created")

        # Create conditional functions
        conditional_functions = create_demo_conditional_functions()
        print("✅ Demo conditional functions created")

        # Create StateGraph builder (from notebook snippet)
        builder = StateGraph(AgentState)
        builder.set_entry_point("patient_collector")

        # Add nodes
        for node_name, node_func in node_functions.items():
            builder.add_node(node_name, node_func)

        # Add edges
        builder.add_conditional_edges(
            "patient_collector",
            conditional_functions["should_continue_patient"],
            {END: END, "policy_search": "policy_search"},
        )

        builder.add_conditional_edges(
            "policy_evaluator",
            conditional_functions["should_continue_policy"],
            {
                "trial_search": "trial_search",
                "policy_evaluator": "policy_evaluator",
                END: END,
            },
        )

        builder.add_edge("policy_search", "policy_evaluator")
        builder.add_edge("trial_search", "grade_trials")
        builder.add_edge("profile_rewriter", "trial_search")

        builder.add_conditional_edges(
            "grade_trials",
            conditional_functions["should_continue_trials"],
            {"profile_rewriter": "profile_rewriter", END: END},
        )

        print("✅ Demo StateGraph builder created with nodes and edges")

        # Setup SQLite memory (from notebook snippet)
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        memory = SqliteSaver(conn)
        print("✅ Demo SQLite memory initialized")

        # Compile graph (from notebook snippet)
        graph = builder.compile(
            checkpointer=memory,
            interrupt_after=[
                "patient_collector",
                "policy_search",
                "trial_search",
                "grade_trials",
                "profile_rewriter",
            ],
        )

        print("✅ Demo graph compiled successfully")
        return graph

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please ensure LangGraph and LangChain are installed")
        return None
    except Exception as e:
        print(f"❌ Error creating demo graph: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test the demo graph creation
    print("🧪 Testing demo graph creation...")
    graph = create_demo_graph()
    if graph:
        print("✅ Demo graph test successful!")
    else:
        print("❌ Demo graph test failed!")
