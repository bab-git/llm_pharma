"""
Trial Service Module

This module handles trial matching and relevance scoring for the LLM Pharma clinical trial workflow system.
"""

import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from omegaconf import DictConfig
from pydantic import BaseModel, Field

from .database_manager import DatabaseManager
from .llm_manager import LLMManager
from .State import AgentState


class Relevance(str, Enum):
    """Enum for trial relevance scores."""
    YES = "yes"
    NO = "no"


class TrialGrade(BaseModel):
    """The result of the trial's relevance check as relevance score and explanation."""
    relevance_score: str = Field(description="Relevance score: 'Yes' or 'No'")
    explanation: str = Field(description="Reasons to the given relevance score.")
    further_information: Optional[str] = Field(
        default="Not applicable",
        description="Additional information needed from patient's medical history",
    )


class GradeHallucinations(BaseModel):
    """Binary score and explanation for whether the LLM's generated answer is grounded in the patient's medical profile."""
    binary_score: str = Field(
        description="Answer is grounded in the patient's medical profile, 'yes' or 'no'"
    )
    reason: str = Field(description="Reasons to the given relevance score.")


class TrialService:
    """
    Service class for trial-related operations in the clinical trial workflow.
    
    This class encapsulates all trial operations: search, retrieval, grading, and hallucination checking.
    """

    def __init__(self, configs: Optional[DictConfig] = None):
        """
        Initialize the TrialService.
        
        Args:
            configs: Optional Hydra config for models and paths. If not provided,
                    default LLM managers will be used.
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM managers once - single source of truth
        if configs is not None:
            self.llm_manager = LLMManager.from_config(configs, use_tool_models=False)
            self.llm_manager_tool = LLMManager.from_config(configs, use_tool_models=True)
        else:
            self.llm_manager, self.llm_manager_tool = LLMManager.get_default_managers()
        
        # Initialize database manager once
        self.db_manager = DatabaseManager(configs=configs)
        
        # Cache heavy objects
        self._vectorstore = None
        self._retriever = None
        
        # Setup prompts and chains once
        self._setup_prompts()
        self._setup_metadata()

    @classmethod
    def from_config(cls, configs: DictConfig) -> "TrialService":
        """Create TrialService from Hydra config."""
        return cls(configs=configs)

    def _setup_prompts(self):
        """Setup prompt templates for trial grading and hallucination checking."""
        # Trial grading prompt
        self.trial_grade_prompt = PromptTemplate(
            template="""
            You are a Principal Investigator (PI) for evaluating patients for clinical trials.
            Your task is to evaluate the relevance of a clinical trial to the given patient's medical profile.
            The clinical trial is related to these diseases: {trial_diseases}
            Here are the inclusion and exclusion criteria of the trial:

            {document}

            ===============
            Use the following steps to determine relevance and provide the necessary fields in your response:
            1- If the patient's profile meets any exclusion criteria, then the trial is not relevant --> relevance_score = 'No'.
            2- If the patient has or had the trial's inclusion diseases, then it is relevant --> relevance_score = 'Yes'.
            3- If the patient did not have the trial's inclusion diseases, then it is not relevant --> relevance_score = 'No'.
            
            Example 1: The patient has Arthritis and the trial is related to pancreatic cancer. --> relevance_score = 'No'
            Example 2: The patient has pancreatic cancer and the trial is also related to carcinoma pancreatic cancer. --> relevance_score = 'Yes'
            Example 3: The patient has pancreatic cancer and the trial is related to breast cancer or ovarian cancer. --> relevance_score = 'No'.
            
            Bring your justification in the explanation.
            Mention further information that is needed from the patient's medical history related to the trial's criteria
            ===============
            Here is the patient's medical profile: {patient_profile}

            Respond with:
            - relevance_score: "Yes" or "No"
            - explanation: Your reasoning
            - further_information: What additional info is needed
            """,
            input_variables=["document", "patient_profile", "trial_diseases"],
        )
        
        # Hallucination checking prompt
        self.hallucination_prompt = PromptTemplate(
            template="""
            You are a grader assessing whether an LLM generation is grounded in / supported by the facts in the patient's medical profile.
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the facts in the patient's medical profile.
            ===============
            Here is the patient's medical profile: {patient_profile}

            ===============
            Here is the LLM generated answer: {explanation}

            Respond with:
            - binary_score: "yes" or "no"
            - reason: Your reasoning
            """,
            input_variables=["patient_profile", "explanation"],
        )

    def _setup_metadata(self):
        """Setup metadata field information for trial retrieval."""
        self.metadata_field_info = [
            AttributeInfo(
                name="disease_category",
                description="Defines the disease group of patients related to this trial. One of ['cancer', 'leukemia', 'mental_health']",
                type="string",
            ),
            AttributeInfo(
                name="drugs",
                description="List of drug names used in the trial",
                type="str",
            ),
        ]
        self.document_content_description = (
            "The list of patient conditions to include or exclude them from the trial"
        )

    @property
    def vectorstore(self):
        """Lazy load and cache the trial vectorstore."""
        if self._vectorstore is None:
            self.logger.info("Creating trial vectorstore...")
            self._vectorstore = self.db_manager.create_trial_vectorstore()
            self.logger.info(f"Trial vectorstore created with {self._vectorstore._collection.count()} trials")
        return self._vectorstore

    @property
    def retriever(self):
        """Lazy load and cache the self-query retriever."""
        if self._retriever is None:
            self.logger.info("Creating self-query retriever...")
            self._retriever = SelfQueryRetriever.from_llm(
                self.llm_manager.current,
                self.vectorstore,
                self.document_content_description,
                self.metadata_field_info,
            )
            self.logger.info("Self-query retriever created")
        return self._retriever

    def retry_invoke_json(self, retriever, question: str, retries: int = 2) -> List[Any]:
        """
        Retry JSON parsing for retriever invocations.
        
        Args:
            retriever: The retriever to invoke
            question: The question to ask
            retries: Number of retry attempts
            
        Returns:
            Retrieved documents
            
        Raises:
            RuntimeError: If all retries fail
        """
        for i in range(retries):
            try:
                return retriever.invoke(question)
            except ValueError as e:
                self.logger.warning(f"JSON parse failed (attempt {i+1}), retrying... Error: {e}")
        raise RuntimeError("Failed to parse JSON after retries")

    def search_relevant_trials(self, patient_profile: str) -> List[Any]:
        """
        Search for relevant trials based on patient profile.
        
        Args:
            patient_profile: Patient's medical profile
            
        Returns:
            List of relevant trial documents
        """
        try:
            question = f"""
            Which trials are relevant to the patient with the following medical history?
            patient_profile: {patient_profile}
            """
            
            def run_trial_retrieval():
                return self.retry_invoke_json(self.retriever, question)
            
            docs_retrieved = self.llm_manager.invoke_with_fallback(
                run_trial_retrieval, reset=True
            )
            
            self.logger.info(f"Retrieved {len(docs_retrieved)} relevant trials")
            return docs_retrieved
            
        except Exception as e:
            self.logger.error(f"Error in trial search: {e}")
            return []

    def _score_trial(self, trial_doc, patient_profile: str) -> TrialGrade:
        """
        Score a single trial for relevance to patient profile.
        
        Args:
            trial_doc: Trial document
            patient_profile: Patient's medical profile
            
        Returns:
            TrialGrade with relevance score and explanation
        """
        try:
            doc_txt = trial_doc.page_content
            trial_diseases = trial_doc.metadata["diseases"]
            nctid = trial_doc.metadata["nctid"]
            
            self.logger.info(f"Scoring trial {nctid} for diseases: {trial_diseases}")
            
            def run_trial_score():
                current_model = self.llm_manager_tool.current
                llm_with_tool = current_model.with_structured_output(TrialGrade)
                retrieval_grader = self.trial_grade_prompt | llm_with_tool
                return retrieval_grader.invoke({
                    "patient_profile": patient_profile,
                    "document": doc_txt,
                    "trial_diseases": trial_diseases,
                })
            
            result = self.llm_manager_tool.invoke_with_fallback(run_trial_score, reset=False)
            
            # Log detailed scoring results
            relevance_score = result.relevance_score
            explanation = result.explanation
            further_info = result.further_information
            
            if relevance_score.lower() == Relevance.YES:
                self.logger.info(f"Trial {nctid} ACCEPTED - Diseases: {trial_diseases}")
                self.logger.info(f"  Reason: {explanation}")
                if further_info and further_info != "Not applicable":
                    self.logger.info(f"  Additional info needed: {further_info}")
            else:
                self.logger.info(f"Trial {nctid} REJECTED - Diseases: {trial_diseases}")
                self.logger.info(f"  Reason: {explanation}")
                if further_info and further_info != "Not applicable":
                    self.logger.info(f"  Additional info needed: {further_info}")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Structured output failed for trial {trial_doc.metadata.get('nctid', 'Unknown')}, using fallback: {e}")
            # Fallback to text-based scoring
            text_response = (
                self.trial_grade_prompt | self.llm_manager_tool.current | StrOutputParser()
            ).invoke({
                "patient_profile": patient_profile,
                "document": doc_txt,
                "trial_diseases": trial_diseases,
            })
            
            # Default to No for safety
            relevance = Relevance.NO
            if "yes" in text_response.lower() and "relevance" in text_response.lower():
                relevance = Relevance.YES
                
            self.logger.warning(f"Fallback scoring for trial {trial_doc.metadata.get('nctid', 'Unknown')}: {relevance.value.upper()}")
            self.logger.warning(f"  Fallback explanation: {text_response[:200]}...")
                
            return TrialGrade(
                relevance_score=relevance.value.title(),
                explanation=text_response[:500],
                further_information="Additional patient history review needed",
            )

    def _hallucination_guard(self, explanation: str, patient_profile: str, trial_nctid: str = "Unknown") -> bool:
        """
        Check if the explanation is grounded in the patient profile.
        
        Args:
            explanation: The explanation to check
            patient_profile: Patient's medical profile
            trial_nctid: Trial NCT ID for logging context
            
        Returns:
            True if explanation is grounded, False if hallucination detected
        """
        try:
            self.logger.info(f"Checking hallucination for trial {trial_nctid}")
            
            def run_hallucination():
                current_model = self.llm_manager_tool.current
                llm_with_tool = current_model.with_structured_output(GradeHallucinations)
                hallucination_grader = self.hallucination_prompt | llm_with_tool
                return hallucination_grader.invoke({
                    "patient_profile": patient_profile,
                    "explanation": explanation,
                })
            
            result = self.llm_manager_tool.invoke_with_fallback(run_hallucination, reset=False)
            
            # Log detailed hallucination check results
            binary_score = result.binary_score
            reason = result.reason
            
            if binary_score.lower() == Relevance.YES:
                self.logger.info(f"Trial {trial_nctid} - Hallucination check PASSED")
                self.logger.info(f"  Grounding reason: {reason}")
            else:
                self.logger.warning(f"Trial {trial_nctid} - Hallucination check FAILED")
                self.logger.warning(f"  Hallucination reason: {reason}")
                self.logger.warning(f"  Explanation being checked: {explanation[:200]}...")
            
            return binary_score.lower() == Relevance.YES
            
        except Exception as e:
            self.logger.error(f"Error in hallucination check for trial {trial_nctid}: {e}")
            # Default to allowing the explanation if hallucination check fails
            return True

    def grade_trials(self, trials: List[Any], patient_profile: str) -> List[Dict[str, Any]]:
        """
        Grade multiple trials for relevance to patient profile.
        
        Args:
            trials: List of trial documents
            patient_profile: Patient's medical profile
            
        Returns:
            List of graded trial results
        """
        relevant_trials = []
        
        for trial in trials:
            nctid = trial.metadata["nctid"]
            self.logger.info(f"Grading trial {nctid}")
            
            # Score the trial
            trial_score = self._score_trial(trial, patient_profile)
            trial_score_dict = dict(trial_score)
            trial_score_dict["nctid"] = nctid
            
            # Check for hallucination if trial is relevant
            if trial_score.relevance_score.lower() == Relevance.YES:
                if not self._hallucination_guard(trial_score.explanation, patient_profile):
                    self.logger.warning(f"Hallucination detected in trial {nctid} - rejecting")
                    trial_score_dict["relevance_score"] = Relevance.NO
                    trial_score_dict["explanation"] = "Agent's Hallucination"
                else:
                    self.logger.info(f"Trial {nctid} is relevant")
            else:
                self.logger.info(f"Trial {nctid} is not relevant")
            
            relevant_trials.append(trial_score_dict)
        
        return relevant_trials

    def trial_search_node(self, state: AgentState) -> AgentState:
        """
        Trial search node that retrieves relevant trials based on patient profile.
        
        Args:
            state: Current agent state containing patient profile
            
        Returns:
            Updated state with retrieved trials
        """
        try:
            patient_profile = state.get("patient_profile", "")
            if not patient_profile:
                self.logger.warning("No patient profile available for trial search")
                return {
                    "last_node": "trial_search",
                    "trials": [],
                    "trial_searches": state.get("trial_searches", 0) + 1,
                    "policy_eligible": state.get("policy_eligible", False),
                }
            
            # Search for relevant trials
            trials = self.search_relevant_trials(patient_profile)
            
            return {
                "last_node": "trial_search",
                "trials": trials,
                "trial_searches": state.get("trial_searches", 0) + 1,
                "policy_eligible": state.get("policy_eligible", False),
            }
            
        except Exception as e:
            self.logger.error(f"Error in trial search: {e}")
            return {
                "last_node": "trial_search",
                "trials": [],
                "trial_searches": state.get("trial_searches", 0) + 1,
                "policy_eligible": state.get("policy_eligible", False),
                "error_message": str(e) if e else "",
            }

    def grade_trials_node(self, state: AgentState) -> AgentState:
        """
        Grade trials node that evaluates trial relevance to patient profile.
        
        Args:
            state: Current agent state containing trials and patient profile
            
        Returns:
            Updated state with graded trials
        """
        try:
            self.logger.info("Checking trials relevance to patient profile")
            
            trials = state.get("trials", [])
            patient_profile = state.get("patient_profile", "")
            
            if not trials:
                self.logger.warning("No trials available for grading")
                return {
                    "last_node": "grade_trials",
                    "relevant_trials": [],
                    "policy_eligible": state.get("policy_eligible", False),
                }
            
            if not patient_profile:
                self.logger.warning("No patient profile available for trial grading")
                return {
                    "last_node": "grade_trials",
                    "relevant_trials": [],
                    "policy_eligible": state.get("policy_eligible", False),
                }
            
            # Grade all trials
            relevant_trials = self.grade_trials(trials, patient_profile)
            
            # Check if any trials were found relevant
            trial_found = any(
                trial.get("relevance_score", "").lower() == Relevance.YES 
                for trial in relevant_trials
            )
            
            return {
                "last_node": "grade_trials",
                "relevant_trials": relevant_trials,
                "policy_eligible": state.get("policy_eligible", False),
                "trial_found": trial_found,
            }
            
        except Exception as e:
            self.logger.error(f"Error in trial grading: {e}")
            return {
                "last_node": "grade_trials",
                "relevant_trials": [],
                "policy_eligible": state.get("policy_eligible", False),
                "error_message": str(e) if e else "",
            }


# Standalone functions for backward compatibility
def get_default_trial_service(config: Optional[DictConfig] = None) -> TrialService:
    """Get default trial service instance."""
    if config is not None:
        return TrialService.from_config(config)
    return TrialService()


def trial_search_node(state: AgentState, configs: Optional[DictConfig] = None) -> AgentState:
    """
    Standalone trial search node function for backward compatibility.
    
    Args:
        state: Current agent state containing patient profile
        configs: Optional config for service initialization
        
    Returns:
        Updated state with retrieved trials
    """
    trial_service = get_default_trial_service(configs)
    return trial_service.trial_search_node(state)


def grade_trials_node(state: AgentState) -> AgentState:
    """
    Standalone grade trials node function for backward compatibility.
    
    Args:
        state: Current agent state containing trials and patient profile
        
    Returns:
        Updated state with graded trials
    """
    trial_service = get_default_trial_service()
    return trial_service.grade_trials_node(state)


# Public API
__all__ = [
    "TrialService",
    "TrialGrade", 
    "GradeHallucinations",
    "Relevance",
    "trial_search_node",
    "grade_trials_node",
]
