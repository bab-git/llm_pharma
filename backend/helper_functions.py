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

    from my_agent.policy_service import (
        policy_evaluator_node,
        policy_search_node
    )
    from helper_functions import (
        trial_search_node,
        grade_trials_node
    )
    from my_agent.patient_collector import (
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

# eligibility class moved to policy_service.py as PolicyEligibility

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

# policy_tools function moved to policy_service.py

# Workflow creation functions moved to WorkflowManager class
# Use WorkflowManager() instead of these functions

# Patient collector node moved to patient_collector.py module

# Policy vector store function moved to DatabaseManager class
# Use DatabaseManager().create_policy_vectorstore() instead

# Trial vector store function moved to DatabaseManager class
# Use DatabaseManager().create_trial_vectorstore() instead

# policy_search_node function moved to policy_service.py

# policy_evaluator_node function moved to policy_service.py

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
