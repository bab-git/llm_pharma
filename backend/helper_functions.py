"""
LLM Pharma Helper Functions

This module contains the complete implementation of the patient collector and policy evaluator nodes
and supporting functions for the LLM Pharma clinical trial workflow system.

COMPLETED FEATURES:
==================

1. Patient Collector Node - COMPLETED
   - Extracts patient ID from natural language prompts
   - Fetches patient data from SQLite database
   - Generates patient profile for clinical trial screening
   - Uses Groq model for free LLM inference

2. Policy Evaluator Node - COMPLETED
   - Evaluates patient eligibility against institutional policies
   - Converts policy documents into yes/no questions
   - Uses structured tools for date and number comparisons
   - Provides detailed rejection reasons for ineligible patients

3. Demo Patient Database - COMPLETED
   - Pre-populated SQLite database with 100 sample patients
   - Includes medical history, trial participation, demographics
   - Automatic database creation and management

4. Configuration System - COMPLETED
   - PatientCollectorConfig class for model and database setup
   - Support for both OpenAI and Groq models
   - Flexible database path configuration

5. Policy Tools - COMPLETED
   - Date comparison and calculation tools
   - Number comparison tools
   - Structured evaluation with ReAct agent

USAGE EXAMPLE:
==============

    from helper_functions import (
        initialize_patient_collector_system,
        patient_collector_node,
        policy_evaluator_node,
        create_agent_state
    )
    
    # Initialize the system
    config = initialize_patient_collector_system(use_free_model=True)
    
    # Create initial state
    state = create_agent_state()
    state['patient_prompt'] = "I need information about patient 1"
    
    # Run patient collector
    result = patient_collector_node(state)
    
    # Run policy evaluator
    state.update(result)
    state['unchecked_policies'] = [policy_document]  # Add policy documents
    policy_result = policy_evaluator_node(state)
    
    print(f"Patient ID: {result['patient_id']}")
    print(f"Profile: {result['patient_profile']}")
    print(f"Policy Eligible: {policy_result['policy_eligible']}")

TESTING:
========

Run the test script:
    python backend/test_patient_collector.py

REQUIREMENTS:
=============

Make sure you have these environment variables set:
- GROQ_API_KEY (for free Groq model usage)
- OPENAI_API_KEY (if using OpenAI models)

Install required packages:
    pip install langchain-groq langchain-openai langchain-core langgraph

TODO - PLACEHOLDER NODES:
=========================

The following nodes still need implementation:
- trial_search_node
- grade_trials_node
- profile_rewriter_node

"""

from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from pydantic import BaseModel, Field
from operator import itemgetter
from typing import Literal
from langgraph.graph import StateGraph, END
import sqlite3
from langchain_core.tools import StructuredTool
from datetime import date, datetime
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool

import os
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
import pandas as pd
import ast

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

# Vector store imports moved to DatabaseManager
# from langchain_community.vectorstores import Chroma
# from langchain_nomic import NomicEmbeddings
# import chromadb
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from backend.my_agent.llm_manager import LLMManager
from backend.my_agent.database_manager import DatabaseManager

class Patient_ID(BaseModel):
    """Model for extracting patient ID from user prompt."""
    patient_id: int

class eligibility(BaseModel):
    """Give the patient's eligibility result."""
    eligibility: str = Field(description="Patient's eligibility for the clinical trial. 'yes' or 'no'")
    reason: str = Field(description="The reason(s) only if the patient is not eligible for clinical trials. Otherwise use N/A")

    class Config:
        schema_extra = {
            "example": {
                "eligibility": 'yes',
                "reason": "N/A",
            },
            "example 2": {
                "eligibility": 'no',
                "reason": "The patient is pregnant at the moment.",
            },                
        }

class grade(BaseModel):
    """The result of the trial's relevance check as relevance score and explanation."""
    relevance_score: str = Field(description="Relevance score: 'Yes' or 'No'")
    explanation: str = Field(description="Reasons to the given relevance score.")        
    further_information: str = Field(description="Additional information needed from patient's medical history")
    
    class Config:
        schema_extra = {
            "example": {
                "relevance_score": "Yes",
                "explanation": "The patient has the target disease condition for this trial.",
                "further_information": "Need to verify patient's current treatment status."
            }
        }

class GradeHallucinations(BaseModel):
    """Binary score and explanation for whether the LLM's generated answer is grounded in / supported by the facts in the patient's medical profile."""
    binary_score: str = Field(
        description="Answer is grounded in the patient's medical profile, 'yes' or 'no'"
    )
    Reason: str = Field(description="Reasons to the given relevance score.")

# Helper to get default LLMManagers for completions and tool calls
# This function is now deprecated - use LLMManager.get_default_managers() instead
def get_default_llm_managers():
    from .my_agent.llm_manager import LLMManager
    return LLMManager.get_default_managers()

class PatientCollectorConfig:
    """Configuration for patient collector node."""
    def __init__(self, llm_manager: LLMManager, llm_manager_tool: LLMManager = None, db_path="sql_server/patients.db"):
        self.llm_manager = llm_manager
        self.llm_manager_tool = llm_manager_tool or llm_manager
        self.db_path = db_path
        self.model = llm_manager.current
        self.model_tool = self.llm_manager_tool.current
        self._setup_profile_chain()

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

# Database functions moved to DatabaseManager class
# Use DatabaseManager().create_demo_patient_database() instead

# Database functions moved to DatabaseManager class
# Use DatabaseManager().get_patient_data(patient_id) instead

class AgentState(TypedDict):
    """State definition for the LLM Pharma workflow agent."""
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

def policy_tools(policy_qs: str, patient_profile: str, model_agent, llm_manager_tool):
    """
    Policy evaluation tools for clinical trial eligibility assessment.
    
    Args:
        policy_qs: Policy questions to evaluate
        patient_profile: Patient profile document
        model_agent: LLM model for evaluation (not used directly anymore)
        llm_manager_tool: LLM manager for tool calls with fallback
        
    Returns:
        str: Evaluation result
    """
    # Simplified date input schema
    class DateInput(BaseModel):
        past_date: str = Field(description="A past date in YYYY-MM-DD format")
        threshold_months: int = Field(description="Number of months to compare against")

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
            months_diff = (today.year - parsed_date.year) * 12 + today.month - parsed_date.month
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
            return react_agent.invoke({
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message_content}
                ]
            })
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
            response = current_model.invoke([{"role": "user", "content": fallback_prompt}])
            return response.content
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            return "Error: Unable to evaluate policy. Patient marked as not eligible for safety."

# Workflow creation functions moved to WorkflowManager class
# Use WorkflowManager() instead of these functions



# Placeholder node functions (to be implemented with actual logic)
# --- Update all node functions to use both managers ---
def patient_collector_node(state: AgentState) -> dict:
    """
    Patient collector node that extracts patient ID from prompt, fetches patient data,
    and generates patient profile.
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
        # error_message = extract_error_message(e, "patient collection")
        return {
            "last_node": "patient_collector",
            "patient_data": {},
            "patient_profile": "",
            "patient_id": 0,
            "revision_number": state.get("revision_number", 0) + 1,
            "policy_eligible": False,
            "error_message": str(e) if e else ""
        }

# Policy vector store function moved to DatabaseManager class
# Use DatabaseManager().create_policy_vectorstore() instead

# Trial vector store function moved to DatabaseManager class
# Use DatabaseManager().create_trial_vectorstore() instead

def policy_search_node(state: AgentState) -> dict:
    """
    Policy search node that retrieves relevant institutional policies based on patient profile.
    
    Args:
        state: Current agent state containing patient profile
        
    Returns:
        Updated state with retrieved policies
    """
    try:
        llm_manager, llm_manager_tool = get_default_llm_managers()
        config = PatientCollectorConfig(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool)
        # Get patient profile from state
        patient_profile = state.get("patient_profile", "")
        
        if not patient_profile:
            print("⚠️ No patient profile available for policy search")
            return {
                "last_node": "policy_search",
                "policies": [],
                "unchecked_policies": [],
                "policy_eligible": state.get("policy_eligible", False)
            }
        
        # Create or load policy vector store
        db_manager = DatabaseManager()
        policy_vectorstore = db_manager.create_policy_vectorstore()
        
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
            "policy_eligible": state.get("policy_eligible", False)
        }
        
    except Exception as e:
        print(f"❌ Error in policy search: {e}")
        # error_message = extract_error_message(e, "policy search")
        
        return {
            "last_node": "policy_search",
            "policies": [],
            "unchecked_policies": [],
            "policy_eligible": state.get("policy_eligible", False),
            "error_message": str(e) if e else ""
        }

# --- Fix policy_evaluator_node ---
def policy_evaluator_node(state: AgentState) -> dict:
    try:
        llm_manager, llm_manager_tool = get_default_llm_managers()
        config = PatientCollectorConfig(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool)
        unchecked_policies = state.get("unchecked_policies", [])
        if not unchecked_policies:
            print("⚠️ No unchecked policies available for evaluation")
            return {
                "last_node": "policy_evaluator",
                "policy_eligible": state.get("policy_eligible", False),
                "rejection_reason": state.get("rejection_reason", ""),
                "revision_number": state.get("revision_number", 0) + 1,
                'checked_policy': None,
                'policy_qs': ""
            }
        policy_doc = unchecked_policies[0]
        policy_header = policy_doc.page_content.split('\n', 2)[1] if len(policy_doc.page_content.split('\n')) > 1 else "Policy"
        print(f'Evaluating Policy:\n {policy_header}')
        policy = policy_doc.page_content
        patient_profile = state.get("patient_profile", "")
        if not patient_profile:
            print("⚠️ No patient profile available for policy evaluation")
            return {
                "last_node": "policy_evaluator",
                "policy_eligible": False,
                "rejection_reason": "No patient profile available",
                "revision_number": state.get("revision_number", 0) + 1,
                'checked_policy': policy_doc,
                'policy_qs': ""
            }
        def run_policy_qs():
            current_model = llm_manager.current
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
        policy_qs = llm_manager.invoke_with_fallback(run_policy_qs, reset=True)
        print(f"✅ Generated policy questions: {policy_qs}")
        def run_policy_tools():
            return policy_tools(policy_qs, patient_profile, config.model_tool, llm_manager_tool)
        result = llm_manager_tool.invoke_with_fallback(run_policy_tools, reset=False)
        print(f"✅ Policy evaluation result: {result}")
        message = f"""Evaluation of the patient's eligibility:\n{result}\n\nIs the patient eligible according to this policy?"""
        def run_eligibility():
            current_model = llm_manager_tool.current
            llm_with_tools = current_model.bind_tools([eligibility])
            return llm_with_tools.invoke(message)
        response = llm_manager_tool.invoke_with_fallback(run_eligibility, reset=False)
        
        if response.tool_calls and len(response.tool_calls) > 0:
            tool_call = response.tool_calls[0]
            if 'args' in tool_call:
                policy_eligible = tool_call['args'].get('eligibility', 'no')
                rejection_reason = tool_call['args'].get('reason', 'N/A')
            else:
                policy_eligible = 'no'
                rejection_reason = 'Unable to parse evaluation result'
        else:
            policy_eligible = 'no'
            rejection_reason = 'No evaluation result available'
        unchecked_policies.pop(0)
        print(f"Remaining unchecked policies: {len(unchecked_policies)}")
        return {
            "last_node": "policy_evaluator",
            "policy_eligible": policy_eligible.lower() == 'yes',
            "rejection_reason": rejection_reason,
            "revision_number": state.get("revision_number", 0) + 1,
            'checked_policy': policy_doc,
            'policy_qs': policy_qs,
            'unchecked_policies': unchecked_policies
        }
    except Exception as e:
        print(f"❌ Error in policy evaluation: {e}")
        # error_message = extract_error_message(e, "policy evaluation")
        return {
            "last_node": "policy_evaluator",
            "policy_eligible": False,
            "rejection_reason": f"Error during evaluation: {str(e)}",
            "revision_number": state.get("revision_number", 0) + 1,
            'checked_policy': None,
            'policy_qs': "",
            "error_message": str(e) if e else ""
        }

# --- Fix trial_search_node ---
def trial_search_node(state: AgentState) -> dict:
    try:
        llm_manager, llm_manager_tool = get_default_llm_managers()
        config = PatientCollectorConfig(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool)
        patient_profile = state.get("patient_profile", "")
        if not patient_profile:
            print("⚠️ No patient profile available for trial search")
            return {
                'last_node': 'trial_search',
                'trials': [],
                'trial_searches': state.get('trial_searches', 0) + 1,
                "policy_eligible": state.get("policy_eligible", False)
            }
        db_manager = DatabaseManager()
        trial_vectorstore = db_manager.create_trial_vectorstore()
        print(f"Number of trials in the vector store: {trial_vectorstore._collection.count()}")
        metadata_field_info = [
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
        document_content_description = "The list of patient conditions to include or exclude them from the trial"
        print(f"patient_profile: {patient_profile}")
        question = f"""
        Which trials are relevant to the patient with the following medical history?\n
        patient_profile: {patient_profile}
        """
        def run_trial_retrieval():
            current_model = llm_manager.current
            retriever_trial_sq = SelfQueryRetriever.from_llm(
                current_model,
                trial_vectorstore,
                document_content_description,
                metadata_field_info
            )
            return retriever_trial_sq.get_relevant_documents(question)
        docs_retrieved = llm_manager.invoke_with_fallback(run_trial_retrieval, reset=True)
        print(f"✅ Retrieved {len(docs_retrieved)} relevant trials")
        trial_searches = state.get('trial_searches', 0) + 1
        return {
            'last_node': 'trial_search',
            'trials': docs_retrieved,
            'trial_searches': trial_searches,
            "policy_eligible": state.get("policy_eligible", False)
        }
    except Exception as e:
        print(f"❌ Error in trial search: {e}")
        # error_message = extract_error_message(e, "trial search")
        return {
            'last_node': 'trial_search',
            'trials': [],
            'trial_searches': state.get('trial_searches', 0) + 1,
            "policy_eligible": state.get("policy_eligible", False),
            "error_message": str(e) if e else ""
        }

# --- Fix grade_trials_node ---
def grade_trials_node(state: AgentState) -> dict:
    try:
        print("----- CHECKING THE TRIALS RELEVANCE TO PATIENT PROFILE ----- ")
        trial_found = False
        trials = state.get('trials', [])
        patient_profile = state.get('patient_profile', '')
        if not trials:
            print("⚠️ No trials available for grading")
            return {
                'last_node': 'grade_trials',
                "relevant_trials": [],
                "policy_eligible": state.get("policy_eligible", False)
            }
        if not patient_profile:
            print("⚠️ No patient profile available for trial grading")
            return {
                'last_node': 'grade_trials',
                "relevant_trials": [],
                "policy_eligible": state.get("policy_eligible", False)
            }
        llm_manager, llm_manager_tool = get_default_llm_managers()
        config = PatientCollectorConfig(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool)
        relevant_trials = []
        for trial in trials:
            doc_txt = trial.page_content
            trial_diseases = trial.metadata['diseases']
            nctid = trial.metadata['nctid']
            print(f"---GRADER: TRIAL {nctid}: ---")
            def run_trial_score():
                current_model = llm_manager_tool.current
                prompt_grader = PromptTemplate(
                    template=""" 
                    You are a Principal Investigator (PI) for evaluating patients for clinical trials.\n
                    Your task is to evaluate the relevance of a clinical trial to the given patient's medical profile. \n
                    The clinical trial is related to these diseases: {trial_diseases} \n
                    Here are the inclusion and exclusion criteria of the trial: \n\n {document} \n\n
                    ===============                
                    Use the following steps to determine relevance and provide the necessary fields in your response: \n
                    1- If the patient's profile meets any exclusion criteria, then the trial is not relevant --> relevance_score = 'No'. \n
                    2- If the patient has or had the trial's inclusion diseases, then it is relevant --> relevance_score = 'Yes'.\n        
                    3- If the patient did not have the trial's inclusion diseases, then it is not relevant --> relevance_score = 'No'.\n
                    Example 1: 
            The patient has Arthritis and the trial is related to pancreatic cancer. --> relevance_score = 'No' \n
                    Example 2: 
            The patient has pancreatic cancer and the trial is also related to carcinoma pancreatic cancer. --> relevance_score = 'Yes' \n
                    Example 3: 
            The patient has pancreatic cancer and the trial is related to breast cancer or ovarian cancer. --> relevance_score = 'No'. \n 
                    Bring your justification in the explanation. \n
                    Mention further information that is needed from the patient's medical history related to the trial's criteria \n
                    ===============
                    Here is the patient's medical profile: {patient_profile} \n\n
                    
                    Respond with:
                    - relevance_score: "Yes" or "No"
                    - explanation: Your reasoning
                    - further_information: What additional info is needed
                    """,
                    input_variables=["document", "patient_profile", "trial_diseases"],
                )
                try:
                    llm_with_tool = current_model.with_structured_output(grade)
                    retrieval_grader = prompt_grader | llm_with_tool
                    return retrieval_grader.invoke({
                        "patient_profile": patient_profile, 
                        "document": doc_txt, 
                        "trial_diseases": trial_diseases
                    })
                except Exception as e:
                    print(f"Structured output failed, using fallback: {e}")
                    # Fallback: parse from text response
                    text_response = (prompt_grader | current_model | StrOutputParser()).invoke({
                        "patient_profile": patient_profile, 
                        "document": doc_txt, 
                        "trial_diseases": trial_diseases
                    })
                    # Create a fallback grade object
                    relevance = "No"  # Default to No for safety
                    if "yes" in text_response.lower() and "relevance" in text_response.lower():
                        relevance = "Yes"
                    
                    return type('Grade', (), {
                        'relevance_score': relevance,
                        'explanation': text_response[:500],  # Truncate if too long
                        'further_information': "Additional patient history review needed"
                    })()
            trial_score = llm_manager_tool.invoke_with_fallback(run_trial_score, reset=False)
            relevance_score = trial_score.relevance_score
            trial_score_dic = dict(trial_score)
            trial_score_dic['nctid'] = nctid
            if relevance_score.lower() == "yes":
                explanation = trial_score.explanation
                def run_hallucination():
                    current_model = llm_manager_tool.current
                    system = """You are a grader assessing whether an LLM generation is grounded in / supported by the facts in the patient's medical profile. \n 
                         Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the facts in the patient's medical profile."""
                    hallucination_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", system),
                            ("human", "Patient's medical profile: \n\n {patient_profile} \n\n LLM generation: {explanation}"),
                        ]
                    )
                    llm_with_tool_hallucination = current_model.with_structured_output(GradeHallucinations)
                    hallucination_grader = hallucination_prompt | llm_with_tool_hallucination
                    return hallucination_grader.invoke({"patient_profile": patient_profile, "explanation": explanation})
                factual_score = llm_manager_tool.invoke_with_fallback(run_hallucination, reset=False)
                factual_score_grade = factual_score.binary_score
                if factual_score_grade == "no":
                    print("--- HALLUCINATION: MODEL'S EXPLANATION IS NOT GROUNDED IN PATIENT PROFILE --> REJECTED---")
                    trial_score_dic['relevance_score'] = 'no'
                    trial_score_dic['explanation'] = "Agent's Hallucination"
            if relevance_score.lower() == "yes" and trial_score_dic.get('relevance_score', '').lower() == "yes":
                print(f"---TRIAL RELEVANT---")
                trial_found = True
            else:
                print(f"--- TRIAL NOT RELEVANT---")
            relevant_trials.append(trial_score_dic)
        return {
            'last_node': 'grade_trials',
            "relevant_trials": relevant_trials,
            "policy_eligible": state.get("policy_eligible", False),
            "trial_found": trial_found
        }
    except Exception as e:
        print(f"❌ Error in trial grading: {e}")
        # error_message = extract_error_message(e, "trial grading")
        return {
            'last_node': 'grade_trials',
            "relevant_trials": [],
            "policy_eligible": state.get("policy_eligible", False),
            "error_message": str(e) if e else ""
        }

# --- Fix profile_rewriter_node ---
def profile_rewriter_node(state: AgentState) -> dict:
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
        # error_message = extract_error_message(e, "profile rewriting")
        return {
            'last_node': 'profile_rewriter',
            'patient_profile': state.get("patient_profile", ""),
            "policy_eligible": state.get("policy_eligible", False),
            "error_message": str(e) if e else ""
        }

# Conditional edge functions
def should_continue_patient(state: AgentState) -> str:
    """Determine if patient collection should continue."""
    if state.get("patient_data"):
        return "policy_search"
    else:
        return END

def should_continue_policy(state: AgentState) -> str:
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

def should_continue_trial_search(state: AgentState) -> str:
    """Determine if trial search should continue."""
    # relevant_trials = state.get("relevant_trials", [])
    # has_trial_math = any(trial.get('relevance_score') == 'Yes' for trial in relevant_trials)
    trials = state.get("trials", [])
    has_potential_trial = trials != []
    
    # if state.get("trial_searches", 0) > state.get("max_trial_searches", 2):
    #     print("--- TRIAL SEARCH: MAX TRIAL SEARCHES REACHED --> END ---")
    #     return END
    if has_potential_trial:        
        return "grade_trials"
    else:
        return END

def should_continue_trials(state: AgentState) -> str:
    """Determine if trial search should continue."""
    relevant_trials = state.get("relevant_trials", [])
    has_trial_math = any(trial.get('relevance_score') == 'Yes' for trial in relevant_trials)    
    
    if state.get("trial_searches", 0) > state.get("max_trial_searches", 3):
        return END
    elif not has_trial_math:
        return "profile_rewriter"
    else:
        return END


# Dataset creation function moved to DatabaseManager class
# Use DatabaseManager().create_trials_dataset(status) instead



# Disease mapping function moved to DatabaseManager class
# Use DatabaseManager().disease_map(disease_list) instead
