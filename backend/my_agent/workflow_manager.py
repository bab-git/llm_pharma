"""
WorkflowManager for LLM Pharma Clinical Trial System

USAGE EXAMPLE (with Hydra):
--------------------------
import hydra
from omegaconf import DictConfig
from backend.my_agent.workflow_manager import WorkflowManager

@hydra.main(version_base=None, config_path='../../config', config_name='config')
def main(cfg: DictConfig):
    workflow = WorkflowManager(configs=cfg)
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
from .State import AgentState, create_agent_state
# Remove the old trial service imports - we'll use the new TrialService class
# from .trial_service import grade_trials_node, trial_search_node


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
        configs: Optional[DictConfig] = None,
    ):
        """
        Initialize the WorkflowManager.

        Args:
            configs: Optional Hydra config for models and paths
        """
        from .llm_manager import LLMManager

        # Create shared dependencies once - single source of truth
        if configs is not None:
            self.llm_manager = LLMManager.from_config(configs, use_tool_models=False)
            self.llm_manager_tool = LLMManager.from_config(configs, use_tool_models=True)
            self.db_manager = DatabaseManager(configs=configs)
        else:
            self.llm_manager, self.llm_manager_tool = LLMManager.get_default_managers()
            self.db_manager = DatabaseManager()
        
        # Initialize service instances with injected dependencies
        from .policy_service import PolicyService
        from .patient_collector import PatientService
        from .trial_service import TrialService
        
        self.policy_service = PolicyService(
            llm_manager=self.llm_manager,
            llm_manager_tool=self.llm_manager_tool,
            db_manager=self.db_manager,
            configs=configs
        )
        self.patient_service = PatientService(
            llm_manager=self.llm_manager,
            llm_manager_tool=self.llm_manager_tool,
            db_manager=self.db_manager,
            configs=configs
        )
        self.trial_service = TrialService(
            llm_manager=self.llm_manager,
            llm_manager_tool=self.llm_manager_tool,
            db_manager=self.db_manager,
            configs=configs
        )
        
        self.graph = None
        self.memory = None
        self.app = None
        self._setup_workflow()

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

        backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if backend_path not in sys.path:
            sys.path.append(backend_path)

        # Create the state graph
        builder = StateGraph(AgentState)

        # Set entry point
        builder.set_entry_point("patient_collector")

        # Add nodes - use service instance methods
        builder.add_node("patient_collector", self.patient_service.patient_collector_node)
        builder.add_node("policy_search", self.policy_service.policy_search_node)
        builder.add_node("policy_evaluator", self.policy_service.policy_evaluator_node)
        builder.add_node("trial_search", self.trial_service.trial_search_node)
        builder.add_node("grade_trials", self.trial_service.grade_trials_node)
        builder.add_node("profile_rewriter", self.patient_service.profile_rewriter_node)

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
        # Use check_same_thread=False to allow multiple sessions to access the database
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
            thread_id: Optional thread ID for state persistence (should be unique per session)

        Returns:
            Dict containing the workflow results
        """
        try:
            # Use create_agent_state from .patient_collector
            state = create_agent_state()
            state["patient_prompt"] = patient_prompt

            # Run the workflow with session-specific thread_id
            if thread_id is None:
                import uuid

                thread_id = str(uuid.uuid4())
            result = self.app.invoke(
                state, config={"configurable": {"thread_id": thread_id}}
            )

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

    # Workflow control methods
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
        else:
            return END

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
