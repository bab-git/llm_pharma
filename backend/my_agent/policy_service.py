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

from datetime import datetime
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from omegaconf import DictConfig
from pydantic import BaseModel, Field

# Import required components
from backend.my_agent.database_manager import DatabaseManager
from backend.my_agent.patient_collector import AgentState, PatientCollectorConfig
from .State import AgentState, create_agent_state

_ = load_dotenv(find_dotenv())  # read local .env file


class PolicyEligibility(BaseModel):
    """Give the patient's eligibility result."""

    eligibility: str = Field(
        description="Patient's eligibility for the clinical trial. 'yes' or 'no'"
    )
    reason: str = Field(
        description="The reason(s) only if the patient is not eligible for clinical trials. Otherwise use N/A"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "eligibility": "yes",
                "reason": "N/A",
            },
            "example 2": {
                "eligibility": "no",
                "reason": "The patient is pregnant at the moment.",
            },
        }


class PolicyService:
    """
    Service class for policy-related operations in the clinical trial workflow.

    This class encapsulates all policy knowledge: retrieval, question-generation,
    tool calls, and final yes/no eligibility decisions.
    """

    def __init__(
        self,
        llm_manager=None,
        llm_manager_tool=None,
        configs: Optional[DictConfig] = None,
    ):
        """
        Initialize the PolicyService.
        Args:
            llm_manager: LLM manager for completions (optional, will create default if not provided)
            llm_manager_tool: LLM manager for tool calls (optional, will create default if not provided)
            configs: Optional Hydra config for models and paths
        """
        from backend.my_agent.llm_manager import LLMManager

        if configs is not None:
            if llm_manager is None:
                llm_manager = LLMManager.from_config(configs, use_tool_models=False)
            if llm_manager_tool is None:
                llm_manager_tool = LLMManager.from_config(configs, use_tool_models=True)
            self.llm_manager = llm_manager
            self.llm_manager_tool = llm_manager_tool
            self.db_manager = DatabaseManager(configs=configs)
        else:
            if llm_manager is None or llm_manager_tool is None:
                self.llm_manager, self.llm_manager_tool = (
                    LLMManager.get_default_managers()
                )
            else:
                self.llm_manager = llm_manager
                self.llm_manager_tool = llm_manager_tool
            self.db_manager = DatabaseManager()

    @classmethod
    def from_config(cls, configs: DictConfig) -> "PolicyService":
        return cls(configs=configs)

    def policy_tools(
        self,
        policy_qs: str,
        patient_profile: str,
        model_agent=None,
        llm_manager_tool=None,
    ):
        """
        Policy evaluation tools for clinical trial eligibility assessment.

        Args:
            policy_qs: Policy questions to evaluate
            patient_profile: Patient profile document
            model_agent: LLM model for evaluation (deprecated, kept for compatibility)
            llm_manager_tool: LLM manager for tool calls with fallback (optional)

        Returns:
            str: Evaluation result
        """
        # Use instance managers if not provided
        if llm_manager_tool is None:
            llm_manager_tool = self.llm_manager_tool

        # Simplified date input schema
        class DateInput(BaseModel):
            past_date: str = Field(description="A past date in YYYY-MM-DD format")
            threshold_months: int = Field(
                description="Number of months to compare against"
            )

        @tool("get_today_date", return_direct=False)
        def get_today_date() -> str:
            """Returns today's date in YYYY-MM-DD format."""
            return datetime.today().date().strftime("%Y-%m-%d")

        @tool("check_months_since_date", args_schema=DateInput, return_direct=False)
        def check_months_since_date(past_date: str, threshold_months: int) -> str:
            """Calculate months between a past date and today, and check if within threshold."""
            try:
                today = datetime.today().date()
                parsed_date = datetime.strptime(past_date, "%Y-%m-%d").date()
                months_diff = (
                    (today.year - parsed_date.year) * 12
                    + today.month
                    - parsed_date.month
                )
                is_within_threshold = months_diff <= threshold_months
                return f"Months since {past_date}: {months_diff}. Within {threshold_months} months: {is_within_threshold}"
            except ValueError:
                return f"Invalid date format: {past_date}. Please use YYYY-MM-DD."

        # Simple number comparison
        class NumberInput(BaseModel):
            num1: float = Field(description="First number")
            num2: float = Field(description="Second number")

        @tool("compare_numbers", args_schema=NumberInput, return_direct=False)
        def compare_numbers(num1: float, num2: float) -> str:
            """Compare if first number is less than second number."""
            result = num1 < num2
            return f"Is {num1} less than {num2}? {result}"

        # Keep only the essential tools
        tools = [get_today_date, check_months_since_date, compare_numbers]
        tool_names = ", ".join([tool.name for tool in tools])

        system_message = f"""You are a Principal Investigator (PI) evaluating patients for clinical trials.
Compare the patient profile to the policy questions and determine eligibility.

PATIENT PROFILE:
{patient_profile}

EVALUATION RULES:
- If ANY policy question answer is "yes", the patient is NOT eligible
- If information is missing from the profile, answer "no" to that question
- Give a final binary 'yes' or 'no' for patient eligibility
- If not eligible, include the specific reason

TOOLS AVAILABLE:
- get_today_date: Get current date
- check_months_since_date: Check if event was within X months of today
- compare_numbers: Compare two numbers

TOOL USAGE:
- Use ONE tool at a time
- Wait for results before using another tool
- Use actual dates from the patient profile
- For date checks: call check_months_since_date with the specific date and month threshold

Available tools: {tool_names}
"""

        user_message_content = f"Policy Questions to Evaluate:\n{policy_qs}"

        # Use simpler message format that works better with Groq
        try:

            def run_react_agent():
                # Create the react_agent inside the runnable using the current model
                current_model = llm_manager_tool.current
                react_agent = create_react_agent(current_model, tools, debug=False)
                return react_agent.invoke(
                    {
                        "messages": [
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_message_content},
                        ]
                    }
                )

            result = llm_manager_tool.invoke_with_fallback(run_react_agent, reset=False)
            return result["messages"][-1].content
        except Exception as e:
            print(f"Error in policy_tools: {e}")
            # Fallback: evaluate without tools
            fallback_prompt = f"""
            As a Principal Investigator, evaluate this patient's eligibility:

            Patient Profile: {patient_profile}

            Policy Questions: {policy_qs}

            Answer with 'yes' if eligible, 'no' if not eligible, and include reasoning.
            """

            try:
                # Use the current model from the manager for fallback too
                current_model = llm_manager_tool.current
                response = current_model.invoke(
                    [{"role": "user", "content": fallback_prompt}]
                )
                return response.content
            except Exception as fallback_error:
                print(f"Fallback also failed: {fallback_error}")
                return "Error: Unable to evaluate policy. Patient marked as not eligible for safety."

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

            if not patient_profile:
                print("⚠️ No patient profile available for policy search")
                return {
                    "last_node": "policy_search",
                    "policies": [],
                    "unchecked_policies": [],
                    "policy_eligible": state.get("policy_eligible", False),
                }

            # Create or load policy vector store
            policy_vectorstore = self.db_manager.create_policy_vectorstore()

            # Create retriever
            retriever = policy_vectorstore.as_retriever(search_kwargs={"k": 5})

            # Retrieve relevant policies
            docs_retrieved = retriever.get_relevant_documents(patient_profile)
            print(f"Retrieved policies to be evaluated: {len(docs_retrieved)}")

            print(f"✅ Retrieved {len(docs_retrieved)} relevant policy sections")

            return {
                "last_node": "policy_search",
                "policies": docs_retrieved,
                "unchecked_policies": docs_retrieved.copy(),
                "policy_eligible": state.get("policy_eligible", False),
            }

        except Exception as e:
            print(f"❌ Error in policy search: {e}")

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
            config = PatientCollectorConfig(
                llm_manager=self.llm_manager, llm_manager_tool=self.llm_manager_tool
            )
            unchecked_policies = state.get("unchecked_policies", [])

            if not unchecked_policies:
                print("⚠️ No unchecked policies available for evaluation")
                return {
                    "last_node": "policy_evaluator",
                    "policy_eligible": state.get("policy_eligible", False),
                    "rejection_reason": state.get("rejection_reason", ""),
                    "revision_number": state.get("revision_number", 0) + 1,
                    "checked_policy": None,
                    "policy_qs": "",
                }

            policy_doc = unchecked_policies[0]
            policy_header = (
                policy_doc.page_content.split("\n", 2)[1]
                if len(policy_doc.page_content.split("\n")) > 1
                else "Policy"
            )
            print(f"Evaluating Policy:\n {policy_header}")

            policy = policy_doc.page_content
            patient_profile = state.get("patient_profile", "")

            if not patient_profile:
                print("⚠️ No patient profile available for policy evaluation")
                return {
                    "last_node": "policy_evaluator",
                    "policy_eligible": False,
                    "rejection_reason": "No patient profile available",
                    "revision_number": state.get("revision_number", 0) + 1,
                    "checked_policy": policy_doc,
                    "policy_qs": "",
                }

            # Generate policy questions
            def run_policy_qs():
                current_model = self.llm_manager.current
                prompt_rps = PromptTemplate(
                    template=""" You are a Principal Investigator (PI) for clinical trials.
                        The following document contains a policy document about participation in clinical trials:\n\n{policy}\n\n
                        Your task is to rephrase each single policy from the document above into a single yes/no question.
                        Form each question so that a yes answer indicates the patient's ineligibility.
                        Do not create more questions than given number of policies.

                        Example: Patients who have had accidents in the past 10 months are not eligible
                        rephrased: Did patient have an accident in the past 10 months?

                        Example: Patients with active tuberculosis, hepatitis B or C, or HIV are excluded unless the trial is specifically designed for these conditions.
                        rephrased: Did patient have active tuberculosis, hepatitis B or C, or HIV?
                        """,
                    input_variables=["policy"],
                )
                policy_rps_chain = prompt_rps | current_model | StrOutputParser()
                return policy_rps_chain.invoke({"policy": policy})

            policy_qs = self.llm_manager.invoke_with_fallback(run_policy_qs, reset=True)
            print(f"✅ Generated policy questions: {policy_qs}")

            # Evaluate using policy tools
            def run_policy_tools():
                return self.policy_tools(
                    policy_qs, patient_profile, config.model_tool, self.llm_manager_tool
                )

            result = self.llm_manager_tool.invoke_with_fallback(
                run_policy_tools, reset=False
            )
            print(f"✅ Policy evaluation result: {result}")

            # Get final eligibility decision
            message = f"""Evaluation of the patient's eligibility:\n{result}\n\nIs the patient eligible according to this policy?"""

            def run_eligibility():
                current_model = self.llm_manager_tool.current
                llm_with_tools = current_model.bind_tools([PolicyEligibility])
                return llm_with_tools.invoke(message)

            response = self.llm_manager_tool.invoke_with_fallback(
                run_eligibility, reset=False
            )

            if response.tool_calls and len(response.tool_calls) > 0:
                tool_call = response.tool_calls[0]
                if "args" in tool_call:
                    policy_eligible = tool_call["args"].get("eligibility", "no")
                    rejection_reason = tool_call["args"].get("reason", "N/A")
                else:
                    policy_eligible = "no"
                    rejection_reason = "Unable to parse evaluation result"
            else:
                policy_eligible = "no"
                rejection_reason = "No evaluation result available"

            unchecked_policies.pop(0)
            print(f"Remaining unchecked policies: {len(unchecked_policies)}")

            return {
                "last_node": "policy_evaluator",
                "policy_eligible": policy_eligible.lower() == "yes",
                "rejection_reason": rejection_reason,
                "revision_number": state.get("revision_number", 0) + 1,
                "checked_policy": policy_doc,
                "policy_qs": policy_qs,
                "unchecked_policies": unchecked_policies,
            }

        except Exception as e:
            print(f"❌ Error in policy evaluation: {e}")
            return {
                "last_node": "policy_evaluator",
                "policy_eligible": False,
                "rejection_reason": f"Error during evaluation: {str(e)}",
                "revision_number": state.get("revision_number", 0) + 1,
                "checked_policy": None,
                "policy_qs": "",
                "error_message": str(e) if e else "",
            }


# Standalone functions for backward compatibility
def get_default_policy_service(config: Optional[DictConfig] = None):
    if config is not None:
        return PolicyService.from_config(config)
    return PolicyService()


def policy_search_node(state: AgentState) -> dict:
    """
    Standalone policy search node function for backward compatibility.

    Args:
        state: Current agent state containing patient profile

    Returns:
        Updated state with retrieved policies
    """
    policy_service = get_default_policy_service()
    return policy_service.policy_search_node(state)


def policy_evaluator_node(state: AgentState) -> dict:
    """
    Standalone policy evaluator node function for backward compatibility.

    Args:
        state: Current agent state containing patient profile and unchecked policies

    Returns:
        Updated state with evaluation results
    """
    policy_service = get_default_policy_service()
    return policy_service.policy_evaluator_node(state)


def policy_tools(policy_qs: str, patient_profile: str, model_agent, llm_manager_tool):
    """
    Standalone policy tools function for backward compatibility.

    Args:
        policy_qs: Policy questions to evaluate
        patient_profile: Patient profile document
        model_agent: LLM model for evaluation (deprecated)
        llm_manager_tool: LLM manager for tool calls with fallback

    Returns:
        str: Evaluation result
    """
    policy_service = get_default_policy_service()
    return policy_service.policy_tools(
        policy_qs, patient_profile, model_agent, llm_manager_tool
    )
