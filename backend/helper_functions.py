"""
LLM Pharma Helper Functions

This module contains the complete implementation of the policy evaluator, trial search, and trial grading nodes
and supporting functions for the LLM Pharma clinical trial workflow system.

COMPLETED FEATURES:
==================

1. Policy Evaluator Node - COMPLETED
   - Evaluates patient eligibility against institutional policies
   - Converts policy documents into yes/no questions
   - Uses structured tools for date and number comparisons
   - Provides detailed rejection reasons for ineligible patients

2. Policy Search Node - COMPLETED
   - Retrieves relevant institutional policies based on patient profile
   - Uses vector search to find matching policy documents

3. Trial Search Node - COMPLETED
   - Searches for relevant clinical trials based on patient profile
   - Uses self-query retriever for intelligent trial matching

4. Trial Grading Node - COMPLETED
   - Evaluates trial relevance to patient profile
   - Checks for LLM hallucinations in trial assessments
   - Provides detailed relevance scores and explanations

5. Policy Tools - COMPLETED
   - Date comparison and calculation tools
   - Number comparison tools
   - Structured evaluation with ReAct agent

USAGE EXAMPLE:
==============

    from helper_functions import (
        policy_evaluator_node,
        policy_search_node,
        trial_search_node,
        grade_trials_node
    )
    from patient_collector import (
        patient_collector_node,
        create_agent_state
    )
    
    # Create initial state
    state = create_agent_state()
    state['patient_prompt'] = "I need information about patient 1"
    
    # Run patient collector (from patient_collector module)
    result = patient_collector_node(state)
    
    # Run policy search and evaluation
    state.update(result)
    policy_search_result = policy_search_node(state)
    state.update(policy_search_result)
    policy_result = policy_evaluator_node(state)
    
    print(f"Policy Eligible: {policy_result['policy_eligible']}")

TESTING:
========

Run the test scripts:
    python backend/test_policy_evaluator.py
    python backend/test_policy_search.py
    python backend/test_trial_search.py

REQUIREMENTS:
=============

Make sure you have these environment variables set:
- GROQ_API_KEY (for free Groq model usage)
- OPENAI_API_KEY (if using OpenAI models)

Install required packages:
    pip install langchain-groq langchain-openai langchain-core langgraph

NOTE:
=====

Patient collector functionality has been moved to the patient_collector.py module.
Import patient_collector_node and related functions from that module instead.

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
from my_agent.llm_manager import LLMManager
from my_agent.database_manager import DatabaseManager

# Import patient collector functionality from the new module
from my_agent.patient_collector import (
    PatientCollectorConfig,
    patient_collector_node,
    profile_rewriter_node,
    create_agent_state,
    Patient_ID,
    AgentState
)

# Patient_ID moved to patient_collector.py module

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
def get_default_llm_managers():
    from my_agent.llm_manager import LLMManager
    return LLMManager.get_default_managers()

# PatientCollectorConfig moved to patient_collector.py module

# Database functions moved to DatabaseManager class
# Use DatabaseManager().create_demo_patient_database() instead

# Database functions moved to DatabaseManager class
# Use DatabaseManager().get_patient_data(patient_id) instead

# AgentState and create_agent_state moved to patient_collector.py module

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

# Patient collector node moved to patient_collector.py module

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

# Profile rewriter node moved to patient_collector.py module



# Dataset creation function moved to DatabaseManager class
# Use DatabaseManager().create_trials_dataset(status) instead



# Disease mapping function moved to DatabaseManager class
# Use DatabaseManager().disease_map(disease_list) instead
