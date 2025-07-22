"""
WorkflowManager for LLM Pharma Clinical Trial System

USAGE EXAMPLE (with Hydra):
--------------------------
import hydra
from omegaconf import DictConfig
from backend.my_agent.workflow_manager import WorkflowManager

@hydra.main(version_base=None, config_path='../../config', config_name='config')
def main(cfg: DictConfig):
    workflow = WorkflowManager.from_config(cfg)
    # ... use workflow ...

if __name__ == "__main__":
    main()

This module manages the LangGraph workflow for patient screening and trial matching.
It handles graph creation, state management, and workflow execution.
"""

import sqlite3
from typing import Any, Dict, List, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from omegaconf import DictConfig

from .database_manager import DatabaseManager
from .llm_manager import LLMManager

# Import functions from helper_functions - these will be imported when needed to avoid circular imports
from .patient_collector import AgentState, create_agent_state
from .trial_service import grade_trials_node, trial_search_node


class WorkflowManager:
    """
    Manages the LangGraph workflow for the LLM Pharma clinical trial system.

    This class handles:
    - Graph creation and configuration
    - State management and persistence
    - Workflow execution
    - Result processing
    """

    def __init__(
        self,
        llm_manager: LLMManager = None,
        llm_manager_tool: LLMManager = None,
        config: Optional[DictConfig] = None,
    ):
        """
        Initialize the WorkflowManager.

        Args:
            llm_manager: LLM manager for general completions
            llm_manager_tool: LLM manager for tool calls
            config: Optional Hydra config for models and paths
        """
        if config is not None:
            if llm_manager is None:
                from .llm_manager import LLMManager

                llm_manager = LLMManager.from_config(config, use_tool_models=False)
            if llm_manager_tool is None:
                from .llm_manager import LLMManager

                llm_manager_tool = LLMManager.from_config(config, use_tool_models=True)
            self.db_manager = DatabaseManager(config=config)
        else:
            self.llm_manager = llm_manager
            self.llm_manager_tool = llm_manager_tool
            self.db_manager = DatabaseManager()
        self.graph = None
        self.memory = None
        self.app = None
        self._setup_workflow()

    @classmethod
    def from_config(cls, config: DictConfig) -> "WorkflowManager":
        return cls(config=config)

    def _setup_workflow(self):
        """Setup the workflow graph and memory."""
        # Create the state graph
        self.graph = self._create_workflow_graph()

        # Setup memory for checkpointing
        self.memory = self._setup_memory()

        # Compile the graph with interrupts for interactive workflow
        self.app = self.graph.compile(
            checkpointer=self.memory,
            interrupt_after=[
                "patient_collector",
                "policy_search",
                "policy_evaluator",
                "trial_search",
                "grade_trials",
                "profile_rewriter",
            ],
        )

    def _create_workflow_graph(self) -> StateGraph:
        """
        Create the workflow graph with all nodes and edges.

        Returns:
            StateGraph: The configured workflow graph
        """
        # Import functions here to avoid circular imports
        import os
        import sys

        from .patient_collector import (
            AgentState,
            patient_collector_node,
            profile_rewriter_node,
        )

        backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if backend_path not in sys.path:
            sys.path.append(backend_path)
        from .policy_service import policy_evaluator_node, policy_search_node

        # Create the state graph
        builder = StateGraph(AgentState)

        # Set entry point
        builder.set_entry_point("patient_collector")

        # Add nodes
        builder.add_node("patient_collector", patient_collector_node)
        builder.add_node("policy_search", policy_search_node)
        builder.add_node("policy_evaluator", policy_evaluator_node)
        builder.add_node("trial_search", trial_search_node)
        builder.add_node("grade_trials", grade_trials_node)
        builder.add_node("profile_rewriter", profile_rewriter_node)

        # Add conditional edges
        builder.add_conditional_edges(
            "patient_collector",
            self._should_continue_patient,
            {END: END, "policy_search": "policy_search"},
        )

        builder.add_conditional_edges(
            "policy_evaluator",
            self._should_continue_policy,
            {
                "trial_search": "trial_search",
                "policy_evaluator": "policy_evaluator",
                END: END,
            },
        )

        builder.add_conditional_edges(
            "trial_search",
            self._should_continue_trial_search,
            {"grade_trials": "grade_trials", END: END},
        )

        builder.add_edge("policy_search", "policy_evaluator")
        builder.add_edge("profile_rewriter", "trial_search")

        builder.add_conditional_edges(
            "grade_trials",
            self._should_continue_trials,
            {"profile_rewriter": "profile_rewriter", END: END},
        )

        return builder

    def _setup_memory(self) -> SqliteSaver:
        """
        Setup SQLite memory for checkpointing the workflow state.

        Returns:
            SqliteSaver: Configured SQLite saver for state persistence
        """
        # Create in-memory SQLite connection for checkpoints
        conn = sqlite3.connect(":memory:", check_same_thread=False)
        memory = SqliteSaver(conn)
        return memory

    def run_workflow(
        self, patient_prompt: str, thread_id: str = None
    ) -> Dict[str, Any]:
        """
        Run the complete workflow for a patient.

        Args:
            patient_prompt: The patient prompt/query
            thread_id: Optional thread ID for state persistence

        Returns:
            Dict containing the workflow results
        """
        try:
            # Use create_agent_state from .patient_collector
            state = create_agent_state()
            state["patient_prompt"] = patient_prompt

            # Run the workflow
            if thread_id:
                result = self.app.invoke(
                    state, config={"configurable": {"thread_id": thread_id}}
                )
            else:
                result = self.app.invoke(state)

            # Extract key results
            workflow_result = {
                "success": True,
                "patient_id": result.get("patient_id", 0),
                "patient_profile": result.get("patient_profile", ""),
                "policy_eligible": result.get("policy_eligible", False),
                "rejection_reason": result.get("rejection_reason", ""),
                "relevant_trials": result.get("relevant_trials", []),
                "trial_found": result.get("trial_found", False),
                "last_node": result.get("last_node", ""),
                "error_message": result.get("error_message", ""),
                "full_state": result,
            }

            return workflow_result

        except Exception as e:
            print(f"❌ Error in workflow execution: {e}")
            return {
                "success": False,
                "error_message": str(e),
                "patient_id": 0,
                "patient_profile": "",
                "policy_eligible": False,
                "relevant_trials": [],
                "trial_found": False,
            }

    def get_workflow_status(self, thread_id: str) -> Dict[str, Any]:
        """
        Get the current status of a workflow thread.

        Args:
            thread_id: The thread ID to check

        Returns:
            Dict containing the current state
        """
        try:
            # Get the current state from memory
            current_state = self.memory.get({"configurable": {"thread_id": thread_id}})
            return {"success": True, "state": current_state, "thread_id": thread_id}
        except Exception as e:
            return {"success": False, "error_message": str(e), "thread_id": thread_id}

    def reset_workflow(self, thread_id: str = None) -> Dict[str, Any]:
        """
        Reset the workflow state for a thread.

        Args:
            thread_id: The thread ID to reset

        Returns:
            Dict containing the reset result
        """
        try:
            if thread_id:
                # Clear the thread state
                self.memory.clear({"configurable": {"thread_id": thread_id}})
                return {
                    "success": True,
                    "message": f"Workflow state reset for thread {thread_id}",
                    "thread_id": thread_id,
                }
            else:
                # Clear all states
                self.memory.clear()
                return {"success": True, "message": "All workflow states reset"}
        except Exception as e:
            return {"success": False, "error_message": str(e), "thread_id": thread_id}

    def get_workflow_summary(self, result: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the workflow results.

        Args:
            result: The workflow result dictionary

        Returns:
            String summary of the results
        """
        if not result.get("success", False):
            return f"❌ Workflow failed: {result.get('error_message', 'Unknown error')}"

        summary_parts = []

        # Patient information
        patient_id = result.get("patient_id", 0)
        summary_parts.append(f"👤 Patient ID: {patient_id}")

        # Policy eligibility
        policy_eligible = result.get("policy_eligible", False)
        if policy_eligible:
            summary_parts.append("✅ Patient is eligible for clinical trials")
        else:
            rejection_reason = result.get("rejection_reason", "No reason provided")
            summary_parts.append(f"❌ Patient is not eligible: {rejection_reason}")

        # Trial results
        relevant_trials = result.get("relevant_trials", [])
        trial_found = result.get("trial_found", False)

        if trial_found:
            summary_parts.append(f"🎯 Found {len(relevant_trials)} relevant trials")
            for i, trial in enumerate(relevant_trials[:3], 1):  # Show first 3
                nctid = trial.get("nctid", "Unknown")
                score = trial.get("relevance_score", "Unknown")
                summary_parts.append(f"   {i}. Trial {nctid}: {score}")
        else:
            summary_parts.append("🔍 No relevant trials found")

        # Last node executed
        last_node = result.get("last_node", "Unknown")
        summary_parts.append(f"📍 Last executed: {last_node}")

        return "\n".join(summary_parts)

    def validate_patient_data(self, patient_id: int) -> Dict[str, Any]:
        """
        Validate that patient data exists in the database.

        Args:
            patient_id: The patient ID to validate

        Returns:
            Dict containing validation result
        """
        try:
            patient_data = self.db_manager.get_patient_data(patient_id)
            if patient_data:
                return {
                    "valid": True,
                    "patient_data": patient_data,
                    "message": f"Patient {patient_id} found in database",
                }
            else:
                return {
                    "valid": False,
                    "message": f"Patient {patient_id} not found in database",
                }
        except Exception as e:
            return {
                "valid": False,
                "error_message": str(e),
                "message": f"Error validating patient {patient_id}",
            }

    def get_available_patients(self) -> List[int]:
        """
        Get list of available patient IDs in the database.

        Returns:
            List of patient IDs
        """
        try:
            # This would need to be implemented in DatabaseManager
            # For now, return a range of IDs that should exist
            return list(range(1, 101))  # Assuming 100 demo patients
        except Exception as e:
            print(f"Error getting available patients: {e}")
            return []

    # MOVE THESE FROM helper_functions.py:
    def _should_continue_patient(self, state: AgentState) -> str:
        """Determine if patient collection should continue."""
        if state.get("patient_data"):
            return "policy_search"
        else:
            return END

    def _should_continue_policy(self, state: AgentState) -> str:
        """Determine if policy evaluation should continue."""
        if state.get("revision_number", 0) > state.get("max_revisions", 3):
            return END

        more_policies = len(state.get("unchecked_policies", [])) > 0
        if state.get("policy_eligible", False):
            if more_policies:
                return "policy_evaluator"
            else:
                return "trial_search"

    def _should_continue_trials(self, state: AgentState) -> str:
        """Determine if trial search should continue."""
        if state.get("revision_number", 0) > state.get("max_revisions", 3):
            return END

        more_policies = len(state.get("unchecked_policies", [])) > 0
        if state.get("policy_eligible", False):
            if more_policies:
                return "policy_evaluator"
            else:
                return "trial_search"
        else:
            return END

    def _should_continue_trial_search(self, state: AgentState) -> str:
        """Determine if trial search should continue."""
        trials = state.get("trials", [])
        has_potential_trial = trials != []

        if has_potential_trial:
            return "grade_trials"
        else:
            return END

    def _should_continue_trials(self, state: AgentState) -> str:
        """Determine if trial search should continue."""
        relevant_trials = state.get("relevant_trials", [])
        has_trial_math = any(
            trial.get("relevance_score") == "Yes" for trial in relevant_trials
        )

        if state.get("trial_searches", 0) > state.get("max_trial_searches", 3):
            return END
        elif not has_trial_math:
            return "profile_rewriter"
        else:
            return END
