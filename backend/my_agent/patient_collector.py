"""
Patient Collector Module

This module handles all patient data collection, profile generation, and profile rewriting functionality
for the LLM Pharma clinical trial workflow system.

COMPLETED FEATURES:
==================

1. PatientCollectorConfig - Configuration for patient collector operations
2. Patient_ID schema - For extracting patient ID from natural language prompts
3. patient_collector_node - Main node for patient data collection and profile generation
4. profile_rewriter_node - Node for rewriting patient profiles when no trials are found
5. Helper functions for patient data processing

USAGE EXAMPLE:
==============

    from my_agent.patient_collector import (
        PatientCollectorConfig,
        patient_collector_node,
        profile_rewriter_node,
        create_agent_state
    )
    
    # Initialize the system
    llm_manager, llm_manager_tool = get_default_llm_managers()
    config = PatientCollectorConfig(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool)
    
    # Create initial state
    state = create_agent_state()
    state['patient_prompt'] = "I need information about patient 1"
    
    # Run patient collector
    result = patient_collector_node(state)
    
    # Run profile rewriter if needed
    if not result.get('trials_found', False):
        state.update(result)
        rewrite_result = profile_rewriter_node(state)
    
    print(f"Patient ID: {result['patient_id']}")
    print(f"Profile: {result['patient_profile']}")

REQUIREMENTS:
=============

Make sure you have these environment variables set:
- GROQ_API_KEY (for free Groq model usage)
- OPENAI_API_KEY (if using OpenAI models)

Install required packages:
    pip install langchain-groq langchain-openai langchain-core langgraph
"""

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import TypedDict, List, Optional
import sqlite3
import os
from dotenv import load_dotenv, find_dotenv
import hydra
from omegaconf import DictConfig

# Load environment variables
_ = load_dotenv(find_dotenv())

# Import from existing modules
from .llm_manager import LLMManager
from .database_manager import DatabaseManager

class Patient_ID(BaseModel):
    """Model for extracting patient ID from user prompt."""
    patient_id: int

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
        "selected_model": "llama-3.3-70b-versatile"
    }

def get_default_llm_managers():
    """Helper to get default LLMManagers for completions and tool calls."""
    from .llm_manager import LLMManager
    return LLMManager.get_default_managers()

class PatientCollectorConfig:
    """Configuration for patient collector node."""
    def __init__(self, llm_manager: LLMManager, llm_manager_tool: LLMManager = None, db_path="sql_server/patients.db", config: Optional[DictConfig] = None):
        self.llm_manager = llm_manager
        self.llm_manager_tool = llm_manager_tool or llm_manager
        if config is not None:
            self.db_path = os.path.join(config.directories.sql_server, "patients.db")
            self.model = llm_manager.current
            self.model_tool = self.llm_manager_tool.current
        else:
            self.db_path = db_path
            self.model = llm_manager.current
            self.model_tool = self.llm_manager_tool.current
        self._setup_profile_chain()
    @classmethod
    def from_config(cls, config: DictConfig) -> 'PatientCollectorConfig':
        from .llm_manager import LLMManager
        llm_manager = LLMManager.from_config(config, use_tool_models=False)
        llm_manager_tool = LLMManager.from_config(config, use_tool_models=True)
        return cls(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool, config=config)

    def _setup_profile_chain(self):
        """Setup the chain for patient profile generation."""
        parser = StrOutputParser()
        prompt_profile = PromptTemplate(
            template="""
            You are the Clinical Research Coordinator in the screening phase of a clinical trial. 
            Use the following patient data to write the patient profile for the screening phase.
            The patient profile is a summary of the patient's information in continuous text form.    
            If they had no previous trial participation, exclude trial status and trial completion date.\n
            Do not ignore any available information.\n 
            Also suggest medical trials that can be related to patient's disease history.\n    
            Write the patient profile in 3 to 4 short sentences.\n\n
            {patient_data}""",
            input_variables=["patient_data"],
        )
        self.chain_profile = prompt_profile | self.model | parser

def patient_collector_node(state: AgentState) -> dict:
    """
    Patient collector node that extracts patient ID from prompt, fetches patient data,
    and generates patient profile.
    
    Args:
        state: Current agent state containing patient prompt
        
    Returns:
        Updated state with patient data and profile
    """
    try:
        llm_manager, llm_manager_tool = get_default_llm_managers()
        config = PatientCollectorConfig(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool)

        patient_data_prompt = """You are a helpful assistance in extracting patient's medical history.\nBased on the following request identify and return the patient's ID number.\n"""

        def run_id_extraction():
            current_model = llm_manager_tool.current
            return current_model.with_structured_output(Patient_ID).invoke([
                SystemMessage(content=patient_data_prompt),
                HumanMessage(content=state['patient_prompt'])
            ])
        response = llm_manager_tool.invoke_with_fallback(run_id_extraction, reset=True)
        patient_id = response.patient_id
        print(f"Patient ID: {patient_id}")

        db_manager = DatabaseManager()
        patient_data = db_manager.get_patient_data(patient_id)
        print(patient_data)

        if patient_data is not None:
            if patient_data.get('name'):
                del patient_data['patient_id']
                del patient_data['name']
            def run_profile_chain():
                current_model = llm_manager.current
                parser = StrOutputParser()
                prompt_profile = PromptTemplate(
                    template="""
                    You are the Clinical Research Coordinator in the screening phase of a clinical trial. 
                    Use the following patient data to write the patient profile for the screening phase.
                    The patient profile is a summary of the patient's information in continuous text form.    
                    If they had no previous trial participation, exclude trial status and trial completion date.\n
                    Do not ignore any available information.\n 
                    Also suggest medical trials that can be related to patient's disease history.\n    
                    Write the patient profile in 3 to 4 short sentences.\n\n
                    {patient_data}""",
                    input_variables=["patient_data"],
                )
                profile_chain = prompt_profile | current_model | parser
                return profile_chain.invoke({'patient_data': patient_data})
            patient_profile = llm_manager.invoke_with_fallback(run_profile_chain, reset=False)
        else:
            patient_profile = ""
            print(f"No patient found with ID: {patient_id}")

        return {
            "last_node": "patient_collector",
            "patient_data": patient_data or {},
            "patient_profile": patient_profile,
            "patient_id": patient_id,
            "revision_number": state.get("revision_number", 0) + 1,
            "policy_eligible": False
        }
    except Exception as e:
        print(f"❌ Error in patient collection: {e}")
        return {
            "last_node": "patient_collector",
            "patient_data": {},
            "patient_profile": "",
            "patient_id": 0,
            "revision_number": state.get("revision_number", 0) + 1,
            "policy_eligible": False,
            "error_message": str(e) if e else ""
        }

def profile_rewriter_node(state: AgentState) -> dict:
    """
    Profile rewriter node that rewrites patient profiles when no trials are found.
    
    Args:
        state: Current agent state containing patient data
        
    Returns:
        Updated state with rewritten patient profile
    """
    try:
        llm_manager, llm_manager_tool = get_default_llm_managers()
        config = PatientCollectorConfig(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool)
        patient_data = state.get("patient_data", {})
        if not patient_data:
            print("⚠️ No patient data available for profile rewriting")
            return {
                'last_node': 'profile_rewriter',
                'patient_profile': state.get("patient_profile", ""),
                "policy_eligible": state.get("policy_eligible", False)
            }
        def run_profile_rewrite():
            current_model = llm_manager.current
            system = """
A trial cross match resulted in no trials for the patient.
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
category X: [patient's disease] can be related to X due to Y.
"""
            re_write_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system),
                    (
                        "human",
                        "Here is a patient data:\n\n {patient_data} \n write a patient profile.",
                    ),
                ]
            )
            profile_rewriter_chain = re_write_prompt | current_model | StrOutputParser()
            return profile_rewriter_chain.invoke({"patient_data": patient_data})
        patient_profile_rewritten = llm_manager.invoke_with_fallback(run_profile_rewrite, reset=True)
        print("--- PROFILE REWRITER: PATIENT'S PROFILE REWRITTEN ---")
        return {
            'last_node': 'profile_rewriter',
            'patient_profile': patient_profile_rewritten,
            "policy_eligible": state.get("policy_eligible", False)
        }
    except Exception as e:
        print(f"❌ Error in profile rewriting: {e}")
        return {
            'last_node': 'profile_rewriter',
            'patient_profile': state.get("patient_profile", ""),
            "policy_eligible": state.get("policy_eligible", False),
            "error_message": str(e) if e else ""
        }

def initialize_patient_collector_system(use_free_model: bool = True, config: Optional[DictConfig] = None) -> PatientCollectorConfig:
    """
    Initialize the patient collector system with appropriate LLM managers.
    Args:
        use_free_model: Whether to use free Groq model (default: True)
        config: Optional Hydra config
    Returns:
        PatientCollectorConfig: Configured patient collector system
    """
    if config is not None:
        return PatientCollectorConfig.from_config(config)
    llm_manager, llm_manager_tool = get_default_llm_managers()
    return PatientCollectorConfig(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool) 