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

# --- Fix trial_search_node ---
# (Moved to my_agent.trial_service)

# --- Fix grade_trials_node ---
# (Moved to my_agent.trial_service)

# Profile rewriter node moved to patient_collector.py module



# Dataset creation function moved to DatabaseManager class
# Use DatabaseManager().create_trials_dataset(status) instead



# Disease mapping function moved to DatabaseManager class
# Use DatabaseManager().disease_map(disease_list) instead
