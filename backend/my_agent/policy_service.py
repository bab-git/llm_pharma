"""
Policy Service Module

This module encapsulates policy knowledge for the LLM Pharma clinical trial workflow system:
- Policy vector store builder and retrieval
- Policy search functionality
- Date/number comparison tool wrappers plus ReAct agent for eligibility scoring
- Policy evaluator state machine logic

FEATURES:
=========

1. Policy Search - Retrieves relevant institutional policies based on patient profile
   - Uses vector search to find matching policy documents
   - Configurable retrieval parameters

2. Policy Evaluation - Evaluates patient eligibility against institutional policies
   - Converts policy documents into yes/no questions
   - Uses structured tools for date and number comparisons
   - Provides detailed rejection reasons for ineligible patients
   - Uses ReAct agent for complex policy evaluation

3. Policy Tools - Date and number comparison utilities
   - Date calculation and threshold checking
   - Number comparison operations
   - Structured evaluation with fallback handling

USAGE EXAMPLE:
==============

    from my_agent.policy_service import PolicyService
    from my_agent.patient_collector import create_agent_state

    # Create policy service
    policy_service = PolicyService()

    # Create initial state with patient profile
    state = create_agent_state()
    state['patient_profile'] = "Patient profile text..."

    # Run policy search and evaluation
    search_result = policy_service.policy_search_node(state)
    state.update(search_result)

    eval_result = policy_service.policy_evaluator_node(state)
    print(f"Policy Eligible: {eval_result['policy_eligible']}")

"""

import logging
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from omegaconf import DictConfig

# Import logging configuration
from .logging_config import get_logger

# Import the extracted policy tools
from .policy_tool_helpers import POLICY_TOOLS

# Import required components
from backend.my_agent.database_manager import DatabaseManager
from backend.my_agent.llm_manager import LLMManager
from .policy import PolicySearcher, PolicyEvaluator

from .State import AgentState

_ = load_dotenv(find_dotenv())  # read local .env file





class PolicyService:
    """
    Service class for policy-related operations in the clinical trial workflow.

    This class encapsulates all policy knowledge: retrieval, question-generation,
    tool calls, and final yes/no eligibility decisions.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        llm_manager_tool: LLMManager,
        db_manager: DatabaseManager,
        configs: Optional[DictConfig] = None,
    ):
        """
        Initialize the PolicyService.
        Args:
            llm_manager: LLM manager for general completions
            llm_manager_tool: LLM manager for tool calls
            db_manager: Database manager for policy data operations
            configs: Optional Hydra config for additional configuration
        """
        # Use injected dependencies
        self.llm_manager = llm_manager
        self.llm_manager_tool = llm_manager_tool
        self.db_manager = db_manager
        
        # Initialize policy tools once
        self.tools = POLICY_TOOLS
        self.logger = get_logger(__name__)
        
        # Initialize the searcher and evaluator components
        self.searcher = PolicySearcher(self.db_manager, self.logger)
        self.evaluator = PolicyEvaluator(
            self.llm_manager, self.llm_manager_tool, self.tools, self.logger
        )


    def policy_search_node(self, state: AgentState) -> dict:
        """
        Policy search node that retrieves relevant institutional policies based on patient profile.

        Args:
            state: Current agent state containing patient profile

        Returns:
            Updated state with retrieved policies
        """
        try:
            # Get patient profile from state
            patient_profile = state.get("patient_profile", "")
            
            # Use the searcher component
            docs_retrieved = self.searcher.run(patient_profile)
            
            return {
                "last_node": "policy_search",
                "policies": docs_retrieved,
                "unchecked_policies": docs_retrieved.copy(),
                "policy_eligible": state.get("policy_eligible", False),
            }

        except Exception as e:
            self.logger.error(f"❌ Error in policy search: {e}")
            return {
                "last_node": "policy_search",
                "policies": [],
                "unchecked_policies": [],
                "policy_eligible": state.get("policy_eligible", False),
                "error_message": str(e) if e else "",
            }

    def policy_evaluator_node(self, state: AgentState) -> dict:
        """
        Policy evaluator node that evaluates patient eligibility against institutional policies.

        Args:
            state: Current agent state containing patient profile and unchecked policies

        Returns:
            Updated state with evaluation results
        """
        try:
            unchecked_policies = state.get("unchecked_policies", [])

            if not unchecked_policies:
                self.logger.warning("No unchecked policies available for evaluation")
                return {
                    "last_node": "policy_evaluator",
                    "policy_eligible": state.get("policy_eligible", False),
                    "rejection_reason": state.get("rejection_reason", ""),
                    "revision_number": state.get("revision_number", 0) + 1,
                    "checked_policy": None,
                    "policy_qs": "",
                }

            policy_doc = unchecked_policies[0]
            patient_profile = state.get("patient_profile", "")

            # Use the evaluator component
            is_eligible, rejection_reason, policy_qs = self.evaluator.run(patient_profile, policy_doc)

            unchecked_policies.pop(0)
            self.logger.info(f"Remaining unchecked policies: {len(unchecked_policies)}")

            return {
                "last_node": "policy_evaluator",
                "policy_eligible": is_eligible,
                "rejection_reason": rejection_reason,
                "checked_policy": policy_doc,
                "policy_qs": policy_qs,
                "unchecked_policies": unchecked_policies,
            }

        except Exception as e:
            self.logger.error(f"❌ Error in policy evaluation: {e}")
            return {
                "last_node": "policy_evaluator",
                "policy_eligible": False,
                "rejection_reason": f"Error during evaluation: {str(e)}",
                "revision_number": state.get("revision_number", 0) + 1,
                "checked_policy": None,
                "policy_qs": "",
                "error_message": str(e) if e else "",
            }