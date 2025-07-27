"""
Policy Evaluator Module

This module contains the PolicyEvaluator class responsible for evaluating patient
eligibility against institutional policies using LLM-based reasoning and tools.
"""

import logging
from functools import cached_property
from typing import List, Tuple

from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field

from backend.my_agent.llm_manager import LLMManager


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


class PolicyEvaluator:
    """
    Handles policy evaluation using LLM-based reasoning and tools.

    This class is responsible for evaluating patient eligibility against
    institutional policies using structured tools and ReAct agent.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        llm_manager_tool: LLMManager,
        tools: List,
        logger: logging.Logger,
    ):
        """
        Initialize the PolicyEvaluator.

        Args:
            llm_manager: LLM manager for general text generation
            llm_manager_tool: LLM manager for tool-based operations
            tools: List of tools available for evaluation
            logger: Logger instance for this component
        """
        self.llm_manager = llm_manager
        self.llm_manager_tool = llm_manager_tool
        self.tools = tools
        self.logger = logger

    @cached_property
    def react_agent(self):
        """Build the ReAct agent once and cache it."""
        return create_react_agent(
            self.llm_manager_tool.current, self.tools, debug=False
        )

    def _invalidate_react_agent_cache(self):
        """Invalidate the cached ReAct agent when model changes."""
        if hasattr(self, "_react_agent"):
            del self._react_agent

    def _invoke_react_agent(self, system_message: str, user_message: str) -> str:
        """Invoke the cached ReAct agent with system and user messages."""
        # Store the current model index to detect changes
        initial_model_index = self.llm_manager_tool.current_index

        def invoke_agent():
            # Check if model changed during fallback
            if self.llm_manager_tool.current_index != initial_model_index:
                self._invalidate_react_agent_cache()
            return self.react_agent.invoke(
                {
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_message},
                    ]
                }
            )

        return self.llm_manager_tool.invoke_with_fallback(
            invoke_agent,
            reset=False,
        )

    def _generate_policy_questions(self, policy: str) -> str:
        """Generate policy questions from a policy document."""
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

    def _decide_final_eligibility(self, message: str):
        """Decide final eligibility using structured output."""
        current_model = self.llm_manager_tool.current
        llm_with_tools = current_model.bind_tools([PolicyEligibility])
        return llm_with_tools.invoke(message)

    def _evaluate_with_tools(self, policy_qs: str, patient_profile: str) -> str:
        """Evaluate policy using ReAct agent and tools."""
        tool_names = ", ".join([tool.name for tool in self.tools])

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

        try:
            result = self._invoke_react_agent(system_message, user_message_content)
            if isinstance(result, dict) and "messages" in result:
                return result["messages"][-1].content
            else:
                self.logger.warning(
                    f"Unexpected result format from ReAct agent: {type(result)}"
                )
                return str(result)
        except Exception as e:
            self.logger.error(f"Error in policy_tools: {e}")
            self.logger.error(f"Error type: {type(e)}")
            # Fallback: evaluate without tools
            fallback_prompt = f"""
            As a Principal Investigator, evaluate this patient's eligibility:

            Patient Profile: {patient_profile}

            Policy Questions: {policy_qs}

            Answer with 'yes' if eligible, 'no' if not eligible, and include reasoning.
            """
            try:
                current_model = self.llm_manager_tool.current
                response = current_model.invoke(
                    [{"role": "user", "content": fallback_prompt}]
                )
                return response.content
            except Exception as fallback_error:
                self.logger.error(f"Fallback also failed: {fallback_error}")
                return "Error: Unable to evaluate policy. Patient marked as not eligible for safety."

    def run(self, patient_profile: str, policy_doc: Document) -> Tuple[bool, str, str]:
        """
        Evaluate a single policy against patient profile.

        Args:
            patient_profile: Patient profile text
            policy_doc: Policy document to evaluate

        Returns:
            Tuple of (is_eligible, rejection_reason, policy_questions)
        """
        try:
            if not patient_profile:
                self.logger.warning(
                    "No patient profile available for policy evaluation"
                )
                return False, "No patient profile available", ""

            policy_header = (
                policy_doc.page_content.split("\n", 2)[1]
                if len(policy_doc.page_content.split("\n")) > 1
                else "Policy"
            )
            self.logger.info(f"Evaluating Policy:\n {policy_header}")

            policy = policy_doc.page_content

            # Generate policy questions
            try:
                policy_qs = self.llm_manager.invoke_with_fallback(
                    lambda: self._generate_policy_questions(policy), reset=True
                )
                self.logger.info(f"Generated policy questions: {policy_qs}")
            except Exception as e:
                self.logger.error(f"Error generating policy questions: {e}")
                policy_qs = ""

            # Evaluate using policy tools
            try:
                result = self.llm_manager_tool.invoke_with_fallback(
                    lambda: self._evaluate_with_tools(policy_qs, patient_profile),
                    reset=False,
                )
                self.logger.info(f"Policy evaluation result: {result}")
            except Exception as e:
                self.logger.error(f"Error in policy_tools: {e}")
                result = ""

            # Get final eligibility decision
            message = f"Evaluation of the patient's eligibility:\n{result}\n\nIs the patient eligible according to this policy?"
            try:
                response = self.llm_manager_tool.invoke_with_fallback(
                    lambda: self._decide_final_eligibility(message), reset=False
                )
            except Exception as e:
                self.logger.error(f"Error in eligibility decision: {e}")
                response = None

            if (
                response
                and hasattr(response, "tool_calls")
                and len(response.tool_calls) > 0
            ):
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

            return policy_eligible.lower() == "yes", rejection_reason, policy_qs

        except Exception as e:
            self.logger.error(f"❌ Error in policy evaluation: {e}")
            return False, f"Error during evaluation: {str(e)}", ""
