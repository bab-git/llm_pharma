"""
Patient Collector Module

This module handles patient data collection, profile generation, and profile rewriting
for the LLM Pharma clinical trial workflow system.
"""

import logging
from typing import Any, Dict, Optional

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from omegaconf import DictConfig
from pydantic import BaseModel

from .database_manager import DatabaseManager
from .llm_manager import LLMManager
from .State import AgentState

# Load environment variables
_ = load_dotenv(find_dotenv())

# Constants
# Note: MAX_RETRIES and PROFILE_SENTENCES are handled by LLMManager.invoke_with_fallback

# Prompts
PATIENT_ID_PROMPT = """You are a helpful assistant in extracting patient's medical history.
Based on the following request identify and return the patient's ID number."""

PROFILE_PROMPT = """You are the Clinical Research Coordinator in the screening phase of a clinical trial. 
Use the following patient data to write the patient profile for the screening phase.
The patient profile is a summary of the patient's information in continuous text form.    
If they had no previous trial participation, exclude trial status and trial completion date.
Do not ignore any available information. 
Also suggest medical trials that can be related to patient's disease history.    
Write the patient profile in 3 to 4 short sentences.

{patient_data}"""

PROFILE_REWRITE_PROMPT = """A trial cross match resulted in no trials for the patient.
As a clinical specialist write a medical profile for this patient and see if their disease(s) can be relevant to any of these categories of mental_health, cancer, or leukemia.
If yes, then suggest relevant medical trial categories for the agent.
If no, then do not add anything there.

Your output must be as below:
<a text summary of original profile>
Suggested relevant trials:
<bullet points of relevant medical trial categories from the above with a one line reason>

Only include categories which can be related to patient diseases in more often cases.
Disregard categories which occasionally or in some cases can be relevant to patient diseases.

example:
The patient is a X-year-old with a medical history of Y. They have participated ........  previous trials, and their trial status and completion date .........
Suggested relevant trials:
category X: [patient's disease] can be related to X due to Y."""


class PatientId(BaseModel):
    """Model for extracting patient ID from user prompt."""

    patient_id: int


class PatientService:
    """
    Service class for patient-related operations in the clinical trial workflow.

    This class encapsulates all patient operations: ID extraction, data fetching,
    profile generation, and profile rewriting.
    """

    def __init__(
        self,
        llm_manager: LLMManager,
        llm_manager_tool: LLMManager,
        db_manager: DatabaseManager,
        configs: Optional[DictConfig] = None,
    ):
        """
        Initialize the PatientService.

        Args:
            llm_manager: LLM manager for general completions
            llm_manager_tool: LLM manager for tool calls
            db_manager: Database manager for patient data operations
            configs: Optional Hydra config for additional configuration
        """
        self.logger = logging.getLogger(__name__)

        # Use injected dependencies
        self.llm_manager = llm_manager
        self.llm_manager_tool = llm_manager_tool
        self.db_manager = db_manager

        # Setup chains directly in constructor
        # Profile generation chain
        prompt_profile = PromptTemplate(
            template=PROFILE_PROMPT, input_variables=["patient_data"]
        )
        self.profile_chain = (
            prompt_profile | self.llm_manager.current | StrOutputParser()
        )

        # Profile rewrite chain
        prompt_rewrite = ChatPromptTemplate.from_messages(
            [
                ("system", PROFILE_REWRITE_PROMPT),
                (
                    "human",
                    "Here is a patient data:\n\n {patient_data} \n write a patient profile.",
                ),
            ]
        )
        self.profile_rewrite_chain = (
            prompt_rewrite | self.llm_manager.current | StrOutputParser()
        )

    def extract_patient_id(self, prompt: str) -> int:
        """
        Extract patient ID from user prompt.

        Args:
            prompt: User prompt containing patient information

        Returns:
            Extracted patient ID
        """
        try:

            def run_id_extraction():
                current_model = self.llm_manager_tool.current
                return current_model.with_structured_output(PatientId).invoke(
                    [
                        SystemMessage(content=PATIENT_ID_PROMPT),
                        HumanMessage(content=prompt),
                    ]
                )

            response = self.llm_manager_tool.invoke_with_fallback(
                run_id_extraction, reset=True
            )
            patient_id = response.patient_id
            self.logger.info(f"Extracted Patient ID: {patient_id}")
            return patient_id

        except Exception as e:
            self.logger.error(f"Error extracting patient ID: {e}")
            raise

    def fetch_patient_data(self, patient_id: int) -> Optional[Dict[str, Any]]:
        """
        Fetch patient data from database.

        Args:
            patient_id: Patient ID to fetch

        Returns:
            Patient data dictionary or None if not found
        """
        try:
            patient_data = self.db_manager.get_patient_data(patient_id)
            if patient_data is not None:
                self.logger.info(f"Fetched patient data for ID {patient_id}")
                # Log patient details at debug level to avoid sensitive data in production logs
                self.logger.debug(f"Patient details: {patient_data}")
                # Log a redacted summary for info level
                age = patient_data.get("age", "N/A")
                medical_history = patient_data.get("medical_history", "N/A")
                previous_trials = patient_data.get("previous_trials", "N/A")
                self.logger.info(
                    f"Patient summary: Age {age}, Medical History: {medical_history}, Previous Trials: {previous_trials}"
                )
            else:
                self.logger.warning(f"No patient found with ID: {patient_id}")
            return patient_data

        except Exception as e:
            self.logger.error(f"Error fetching patient data: {e}")
            return None

    def build_profile(self, patient_data: Dict[str, Any]) -> str:
        """
        Build patient profile from patient data.

        Args:
            patient_data: Patient data dictionary

        Returns:
            Generated patient profile text
        """
        try:
            patient_profile = self.llm_manager.invoke_with_fallback(
                lambda: self.profile_chain.invoke({"patient_data": patient_data}),
                reset=False,
            )
            self.logger.info("Generated patient profile")
            self.logger.info(f"Profile content: {patient_profile}")
            return patient_profile

        except Exception as e:
            self.logger.error(f"Error building patient profile: {e}")
            return ""

    def rewrite_profile(self, patient_data: Dict[str, Any]) -> str:
        """
        Rewrite patient profile when no trials are found.

        Args:
            patient_data: Patient data dictionary

        Returns:
            Rewritten patient profile text
        """
        try:
            patient_profile_rewritten = self.llm_manager.invoke_with_fallback(
                lambda: self.profile_rewrite_chain.invoke(
                    {"patient_data": patient_data}
                ),
                reset=True,
            )
            self.logger.info("Rewrote patient profile")
            self.logger.info(f"Rewritten profile content: {patient_profile_rewritten}")
            return patient_profile_rewritten

        except Exception as e:
            self.logger.error(f"Error rewriting patient profile: {e}")
            return ""

    def patient_collector_node(self, state: AgentState) -> AgentState:
        """
        Patient collector node that extracts patient ID, fetches data, and generates profile.

        Args:
            state: Current agent state containing patient prompt

        Returns:
            Updated state with patient data and profile
        """
        try:
            # Extract patient ID
            patient_id = self.extract_patient_id(state["patient_prompt"])

            # Fetch patient data
            patient_data = self.fetch_patient_data(patient_id)

            # Build profile if data exists
            if patient_data is not None:
                patient_profile = self.build_profile(patient_data)
            else:
                patient_profile = ""

            return {
                "last_node": "patient_collector",
                "patient_data": patient_data or {},
                "patient_profile": patient_profile,
                "patient_id": patient_id,
                "revision_number": state.get("revision_number", 0) + 1,
                "policy_eligible": False,
            }

        except Exception as e:
            self.logger.error(f"Error in patient collection: {e}")
            return {
                "last_node": "patient_collector",
                "patient_data": {},
                "patient_profile": "",
                "patient_id": 0,
                "revision_number": state.get("revision_number", 0) + 1,
                "policy_eligible": False,
                "error_message": str(e) if e else "",
            }

    def profile_rewriter_node(self, state: AgentState) -> AgentState:
        """
        Profile rewriter node that rewrites patient profiles when no trials are found.

        Args:
            state: Current agent state containing patient data

        Returns:
            Updated state with rewritten patient profile
        """
        try:
            patient_data = state.get("patient_data", {})
            if not patient_data:
                self.logger.warning("No patient data available for profile rewriting")
                return {
                    "last_node": "profile_rewriter",
                    "patient_profile": state.get("patient_profile", ""),
                    "policy_eligible": state.get("policy_eligible", False),
                }

            patient_profile_rewritten = self.rewrite_profile(patient_data)

            return {
                "last_node": "profile_rewriter",
                "patient_profile": patient_profile_rewritten,
                "policy_eligible": state.get("policy_eligible", False),
            }

        except Exception as e:
            self.logger.error(f"Error in profile rewriting: {e}")
            return {
                "last_node": "profile_rewriter",
                "patient_profile": state.get("patient_profile", ""),
                "policy_eligible": state.get("policy_eligible", False),
                "error_message": str(e) if e else "",
            }


# Public API
__all__ = [
    "PatientService",
    "PatientId",
]
