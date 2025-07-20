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

from langchain_community.vectorstores import Chroma
from langchain_nomic import NomicEmbeddings
import chromadb
import json
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever

class Patient_ID(BaseModel):
    """Model for extracting patient ID from user prompt."""
    patient_id: int

class policy_relevance(BaseModel):
    """Policy relevance score"""
    relevant: str = Field(description="is policy relevant? 'yes' or 'no'.")
    reason: str

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
    relevance_score: str        
    explanation: str = Field(description="Reasons to the given relevance score.")        
    further_information: str

class GradeHallucinations(BaseModel):
    """Binary score and explanation for whether the LLM's generated answer is grounded in / supported by the facts in the patient's medical profile."""
    binary_score: str = Field(
        description="Answer is grounded in the patient's medical profile, 'yes' or 'no'"
    )
    Reason: str = Field(description="Reasons to the given relevance score.")

class PatientCollectorConfig:
    """Configuration for patient collector node."""
    def __init__(self, use_free_model=True, db_path="sql_server/patients.db"):
        self.use_free_model = use_free_model
        self.db_path = db_path
        self.modelID_groq = "llama-3.3-70b-versatile"
        self.modelID = "gpt-3.5-turbo"
        
        # Initialize models
        if use_free_model:
            self.model = ChatGroq(model=self.modelID_groq, temperature=0)
            print(f"Using Groq model: {self.modelID_groq}")
        else:
            self.model = ChatOpenAI(temperature=0.0, model=self.modelID)
            print(f"Using OpenAI model: {self.modelID}")
        
        # Initialize chain for patient profile generation
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

def create_demo_patient_database(db_path="sql_server/patients.db"):
    """
    Create a demo patient database with randomly generated patient data.
    
    Args:
        db_path (str): Path where the database file will be created
    
    Returns:
        pandas.DataFrame: DataFrame containing the generated patient data
    """
    import pandas as pd
    from datetime import datetime, timedelta
    import numpy as np
    import json
    import random
    import os
    
    # Convert to absolute path relative to the project root
    if not os.path.isabs(db_path):
        project_root = os.path.dirname(os.path.dirname(__file__))
        db_path = os.path.join(project_root, db_path)
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Remove existing database if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Define columns for the database
    columns = ["patient_id", "name", "age", "medical_history", "previous_trials", "trial_status", "trial_completion_date"]
    data = []

    # Given names and surnames
    names = ["John", "Jane", "Alice", "Michael", "Emily", "Daniel", "Sophia", "James", "Emma", "Oliver"]
    surnames = ["Doe", "Smith", "Johnson", "Brown", "Davis", "Garcia", "Martinez", "Anderson", "Thomas", "Wilson"]

    # Generate all possible unique combinations of names and surnames
    combinations = [(name, surname) for name in names for surname in surnames]

    # Shuffle the combinations to ensure randomness
    random.shuffle(combinations)

    # Select the first 100 unique combinations
    unique_names = combinations[:100]

    # Generate the full names
    full_names = [f"{name} {surname}" for name, surname in unique_names]

    # Load diseases from the JSON file
    diseases_file_path = os.path.join(os.path.dirname(__file__), '..', 'source_data', 'diseases_list.json')
    try:
        with open(diseases_file_path, 'r') as file:
            trial_diseases = json.load(file)
        
        list_trial_diseases = list({disease for diseases in trial_diseases.values() for disease in diseases})
    except FileNotFoundError:
        # Fallback if diseases file not found
        list_trial_diseases = ["myelomonocytic leukemia", "myeloid leukemia", "lymphoblastic leukemia", 
                              "colorectal cancer", "esophageal cancer", "gastric cancer"]

    other_medical_conditions = ["Hypertension", "Diabetes", "Asthma", "Heart Disease", "Arthritis",
                          "Chronic Pain", "Anxiety", "Depression", "Obesity"]

    all_conditions = list(set(list_trial_diseases + other_medical_conditions))

    trial_statuses = ["Completed", "Ongoing", "Withdrawn"]

    def random_date(start, end):
        return start + timedelta(days=random.randint(0, int((end - start).days)))

    # start_date must be 2 years before now
    start_date = datetime.now() - timedelta(days=365 * 2)

    # end_date must be a month before now
    end_date = datetime.now() - timedelta(days=10)

    # Generate 100 patients
    for i in range(1, 101):
        name = random.choice(full_names)
        age = random.randint(20, 80)
        medical_history = random.choice(all_conditions)
        
        # 50% chance of having previous trials
        if random.choice([True, False]):
            previous_trials = f"NCT0{random.randint(1000000, 9999999)}"
            trial_status = random.choice(trial_statuses)
            trial_completion_date = random_date(start_date, end_date).strftime('%Y-%m-%d')
        else:
            previous_trials = ""
            trial_status = ""
            trial_completion_date = ""
        
        # If trial is ongoing, no completion date
        if trial_status == "Ongoing":
            trial_completion_date = ""

        data.append((i, name, age, medical_history, previous_trials, trial_status, trial_completion_date))

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Save DataFrame to CSV in the same directory as the database
    csv_path = db_path.replace('.db', '.csv').replace('sql_server', 'data')
    df.to_csv(csv_path, index=False)
    
    # Create SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the patients table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER PRIMARY KEY,
        name TEXT,
        age INTEGER,
        medical_history TEXT,
        previous_trials TEXT,
        trial_status TEXT,
        trial_completion_date TEXT
    )
    ''')

    # Insert DataFrame into SQLite table
    df.to_sql('patients', conn, if_exists='append', index=False)

    # Commit and close the connection
    conn.commit()
    conn.close()
    
    print(f"Demo patient database created at: {db_path}")
    print(f"CSV export created at: {csv_path}")
    print(f"Total patients created: {len(df)}")
    
    return df

def get_patient_data(patient_id: int, db_path="sql_server/patients.db") -> dict:
    """
    Fetch all fields for the patient based on the given patient_id as an integer.

    Args:
        patient_id: The patient ID to fetch data for
        db_path: Path to the SQLite database file

    Returns:
        A dictionary containing the patient's medical history, or None if not found.        
    """
    # Convert to absolute path relative to the project root
    if not os.path.isabs(db_path):
        project_root = os.path.dirname(os.path.dirname(__file__))
        db_path = os.path.join(project_root, db_path)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()    
    query = 'SELECT * FROM patients WHERE patient_id=?'
    cursor.execute(query, (patient_id,))
    patient_data = cursor.fetchone()
    column_names = [column[0] for column in cursor.description]
    conn.close()
    
    if patient_data is None:
        return None
    else:    
        results = dict(zip(column_names, patient_data))    
    return results

# Not used yet
def add_patient_data(patient_data: dict, db_path="../data/patients.db"):    
    """
    Adds a new patient to the SQLite database.
    
    Args:
        patient_data: Dictionary containing patient information
        db_path: Path to the SQLite database file
    """
    name = patient_data['name']
    age = patient_data['age']
    medical_history = patient_data['medical_history']
    previous_trials = patient_data['previous_trials']
    trial_status = patient_data['trial_status']
    last_trial_dates = patient_data['last_trial_dates']

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Insert the new patient data into the database
    cursor.execute('''
    INSERT INTO patients (name, age, medical_history, previous_trials, trial_status, last_trial_dates)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (name, age, medical_history, previous_trials, trial_status, last_trial_dates))
    
    conn.commit()
    conn.close()

# def initialize_patient_collector_system(use_free_model=True, db_path="sql_server/patients.db", force_recreate_db=False):
#     """
#     Initialize the patient collector system by creating database and configuration.
    
#     Args:
#         use_free_model: Whether to use free Groq model or OpenAI
#         db_path: Path to the SQLite database file
#         force_recreate_db: Whether to recreate the database even if it exists
        
#     Returns:
#         PatientCollectorConfig: Configured patient collector
#     """
#     # Create database if it doesn't exist or if forced
#     if not os.path.exists(db_path) or force_recreate_db:
#         print("Creating demo patient database...")
#         create_demo_patient_database(db_path)
#     else:
#         print(f"Using existing database at: {db_path}")
    
    # Create and return configuration
    config = PatientCollectorConfig(use_free_model=use_free_model, db_path=db_path)
    print("Patient collector system initialized successfully!")
    return config

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
        "trial_found": False
    }

def policy_tools(policy_qs: str, patient_profile: str, model_agent):
    """
    Policy evaluation tools for clinical trial eligibility assessment.
    
    Args:
        policy_qs: Policy questions to evaluate
        patient_profile: Patient profile document
        model_agent: LLM model for evaluation
        
    Returns:
        str: Evaluation result
    """
    class CalculatorInput(BaseModel):
        num1: float = Field(description="first number")
        num2: float = Field(description="second number")

    def multiply(num1: float, num2: float) -> float:
        "multiplies two input numbers together, num1 and num2"
        return (num1 * num2)

    multiply_tool = StructuredTool.from_function(
        func=multiply,
        name="multiply",
        description="multiply numbers",
        args_schema=CalculatorInput,
    )

    @tool("date_today-tool")
    def date_today() -> datetime.date:
        "returns today date"
        return datetime.today().date()    

    def date_difference(date1: date, date2: date) -> int:
        "The number of months date1 is before date2"
        month_difference = (date2.year - date1.year) * 12 + date2.month - date1.month
        return f'{month_difference} months'

    class dates(BaseModel):
        date1: date = Field(description="first date")
        date2: date = Field(description="second date")

    date_difference_tool = StructuredTool.from_function(
        func=date_difference,
        name="date_difference",
        description="The number of months first date is before second date",
        args_schema=dates,
    )

    class date_class(BaseModel):
        date: str = Field(description="A date string in the format YYYY-MM-DD")    

    @tool("date_convert-tool", args_schema=date_class)
    def date_convert(date: str) -> date:
        "Converts a date string to a date object"
        date = datetime.strptime(date, "%Y-%m-%d").date()
        return date

    @tool("date_split-tool", args_schema=date_class)
    def date_split(date: str) -> date:
        "Extracts the year and month from a date string"
        date = datetime.strptime(date, "%Y-%m-%d").date()
        year = date.year
        month = date.month
        return f'year: {year}, month: {month}'

    @tool("number_comparison-tool", args_schema=CalculatorInput)
    def number_compare(num1: float, num2: float) -> bool:
        "Determines if first number is less than the second number"
        num1_less_num1 = num1 < num2
        return num1_less_num1

    tools = [multiply_tool, date_today, date_difference_tool, date_split, number_compare]

    tool_names=", ".join([tool.name for tool in tools])

    system_message = f"""
    You are a Principal Investigator (PI) for evaluating patients for clinical trials.
    You are asked to compare the patient profile document to the institution policy questions.
    You must determine if the patient is eligible based on the following documents.

    \n #### Here is the patient profile document: \n {patient_profile}\n\n

    If the answer to any policy question is yes, then the patient is not eligible.\n
    If the answer to the question is not provided in the patient profile, answer 'no'.\n

    Give a binary 'yes' or 'no' score in the response to indicate whether the patient is eligible according to the given policy ONLY.
    If the patient is not eligible then also include the reason in your response.

    You have access to the following tools:
    {tool_names}
    """

    user_message_content = f""" Here are the policy questions: \n{policy_qs}"""
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_content}
    ]

    react_agent = create_react_agent(model_agent, tools, debug=False)    
    messages = react_agent.invoke({"messages": messages})    
    result = messages["messages"][-1].content    
    return result

def create_workflow_builder(agent_state: AgentState) -> StateGraph:
    """
    Create the workflow builder with all nodes and edges for the LLM Pharma system.
    
    Args:
        agent_state: The agent state definition
        
    Returns:
        StateGraph: The configured workflow graph
    """
    # Create the state graph
    builder = StateGraph(AgentState)
    
    # Set entry point
    builder.set_entry_point("patient_collector")
    
    # Add nodes (placeholder implementations)
    builder.add_node("patient_collector", patient_collector_node)
    builder.add_node("policy_search", policy_search_node)
    builder.add_node("policy_evaluator", policy_evaluator_node)
    builder.add_node("trial_search", trial_search_node)
    builder.add_node("grade_trials", grade_trials_node)
    builder.add_node("profile_rewriter", profile_rewriter_node)
    
    # Add conditional edges
    builder.add_conditional_edges(
        "patient_collector", 
        should_continue_patient, 
        {
            END: END,
            "policy_search": "policy_search"
        }
    )
    
    builder.add_conditional_edges(
        "policy_evaluator", 
        should_continue_policy, 
        {
            "trial_search": "trial_search",
            "policy_evaluator": "policy_evaluator",
            END: END
        }
    )
    
    builder.add_conditional_edges(
        "trial_search", 
        should_continue_trial_search, 
        {
            "grade_trials": "grade_trials",
            # "profile_rewriter": "profile_rewriter",
            END: END
        }
    )

    builder.add_edge("policy_search", "policy_evaluator")
    # builder.add_edge("trial_search", "grade_trials")
    builder.add_edge("profile_rewriter", "trial_search")
    
    builder.add_conditional_edges(
        "grade_trials", 
        should_continue_trials, 
        {
            "profile_rewriter": "profile_rewriter",
            END: END
        }
    )
    
    return builder

def setup_sqlite_memory() -> SqliteSaver:
    """
    Setup SQLite memory for checkpointing the workflow state.
    
    Returns:
        SqliteSaver: Configured SQLite saver for state persistence
    """
    # Create in-memory SQLite connection for checkpoints
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    memory = SqliteSaver(conn)
    return memory



# Placeholder node functions (to be implemented with actual logic)
def patient_collector_node(state: AgentState) -> dict:
    """
    Patient collector node that extracts patient ID from prompt, fetches patient data,
    and generates patient profile.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with patient data and profile
    """
    # Create configuration for this node
    config = PatientCollectorConfig(use_free_model=True)
    
    patient_data_prompt = """You are a helpful assistance in extracting patient's medical history.
Based on the following request identify and return the patient's ID number.
"""

    response = config.model.with_structured_output(Patient_ID).invoke([
        SystemMessage(content=patient_data_prompt),
        HumanMessage(content=state['patient_prompt'])
    ])
    patient_id = response.patient_id
    print(f"Patient ID: {patient_id}")
    
    patient_data = get_patient_data(patient_id, config.db_path)
    print(patient_data)
    
    if patient_data is not None:        
        if patient_data.get('name'):
            del patient_data['patient_id']
            del patient_data['name']
        patient_profile = config.chain_profile.invoke({'patient_data': patient_data})
    else:
        patient_profile = ""
        print(f"No patient found with ID: {patient_id}")

    return {
        "last_node": "patient_collector",
        "patient_data": patient_data or {},
        "patient_profile": patient_profile,
        "patient_id": patient_id,
        "revision_number": state.get("revision_number", 0) + 1,
        "policy_eligible": False  # Initialize this key to prevent KeyError
    }

def create_policy_vectorstore(policy_file_path="source_data/instut_trials_policy.md", 
                            vectorstore_path="vector_store", 
                            collection_name="policies"):
    """
    Create a vector store from the institutional policy document.
    
    Args:
        policy_file_path: Path to the policy markdown file
        vectorstore_path: Path to store the vector database
        collection_name: Name of the collection in the vector store
        
    Returns:
        Chroma: The created vector store
    """
    # Convert to absolute paths relative to the project root
    if not os.path.isabs(policy_file_path):
        project_root = os.path.dirname(os.path.dirname(__file__))
        policy_file_path = os.path.join(project_root, policy_file_path)
    
    if not os.path.isabs(vectorstore_path):
        project_root = os.path.dirname(os.path.dirname(__file__))
        vectorstore_path = os.path.join(project_root, vectorstore_path)
    
    # Ensure vector store directory exists
    os.makedirs(vectorstore_path, exist_ok=True)
    
    # Read the policy document
    with open(policy_file_path, 'r', encoding='utf-8') as file:
        policy_content = file.read()
    
    # Split the policy into sections (by headers)
    sections = []
    current_section = ""
    current_title = ""
    
    for line in policy_content.split('\n'):
        if line.startswith('####'):
            # Save previous section if exists
            if current_section.strip():
                sections.append({
                    'title': current_title,
                    'content': current_section.strip()
                })
            # Start new section
            current_title = line.replace('####', '').strip()
            current_section = ""
        else:
            current_section += line + "\n"
    
    # Add the last section
    if current_section.strip():
        sections.append({
            'title': current_title,
            'content': current_section.strip()
        })
    
    # Create documents for vector store
    policy_docs = []
    for section in sections:
        doc = Document(
            page_content=section['content'],
            metadata={
                "title": section['title'],
                "source": "institutional_policy"
            }
        )
        policy_docs.append(doc)
    
    # Create persistent client
    persistent_client = chromadb.PersistentClient(path=vectorstore_path)
    
    # Create or load vector store
    vectorstore = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode='local'),
    )
    
    # Check if collection is empty and add documents if needed
    if vectorstore._collection.count() == 0:
        vectorstore = Chroma.from_documents(
            documents=policy_docs,
            client=persistent_client,
            collection_name=collection_name,
            embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode='local'),
        )
        print(f"✅ Policy vector store created with {len(policy_docs)} sections")
    else:
        print(f"✅ Policy vector store loaded with {vectorstore._collection.count()} documents")
    
    return vectorstore

def create_trial_vectorstore(trials_csv_path="data/trials_data.csv",
                           vectorstore_path="vector_store",
                           collection_name="trials",
                           status_filter="recruiting",
                           vstore_delete=False):
    """
    Create a vector store from the clinical trials dataset.
    
    Args:
        trials_csv_path: Path to the trials CSV file
        vectorstore_path: Path to store the vector database
        collection_name: Name of the collection in the vector store
        status_filter: Filter trials by status (e.g., 'recruiting')
        
    Returns:
        Chroma: The created vector store
    """
    import ast
    
    # Convert to absolute paths relative to the project root
    if not os.path.isabs(trials_csv_path):
        project_root = os.path.dirname(os.path.dirname(__file__))
        trials_csv_path = os.path.join(project_root, trials_csv_path)
    
    if not os.path.isabs(vectorstore_path):
        project_root = os.path.dirname(os.path.dirname(__file__))
        vectorstore_path = os.path.join(project_root, vectorstore_path)
    
    # Ensure vector store directory exists
    os.makedirs(vectorstore_path, exist_ok=True)
    
    # Create persistent client
    persistent_client = chromadb.PersistentClient(path=vectorstore_path)

    if vstore_delete == True:
        try:
            persistent_client.delete_collection(collection_name)
            print(f"Collection {collection_name} is deleted")
        except Exception:
            print(f"Collection {collection_name} does not exist.")

    # Create or load vector store
    vectorstore = Chroma(
        client=persistent_client,
        collection_name=collection_name,
        embedding_function=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode='local'),
    )


    # if vstore_delete == True:
    #     vectorstore.delete_collection()
    #     vectorstore = Chroma(
    #         client=persistent_client,
    #         collection_name=collection_name,
    #         embedding_function=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode='local'),
    #     )    
    #     print("vstore deleted")

    if vectorstore._collection.count() > 0:        
        print(f"✅ Trial vector store loaded with {vectorstore._collection.count()} trials")
        return vectorstore
    
    
    # Read trials data
    df_trials = pd.read_csv(trials_csv_path)
    # Convert 'diseases' and 'drugs' columns from string to list

    df_trials['diseases'] = df_trials['diseases'].apply(ast.literal_eval)
    # df_trials['drugs'] = df_trials['drugs'].apply(ast.literal_eval)

    print(trials_csv_path)
    print(f"loaded trials: {len(df_trials)}")
    
    # Filter by status if specified
    if status_filter:
        df_trials = df_trials[df_trials['status'] == status_filter].reset_index(drop=True)
        print(f"✅ Filtered trials to status '{status_filter}': {len(df_trials)} trials")
    
    # Create documents for vector store
    trial_docs = []
    for i, row in df_trials.iterrows():
        disease = disease_map(row['diseases'])
        if disease == 'other_conditions':
            continue
        doc = Document(
            page_content=row['criteria'],
            metadata={
                "nctid": row['nctid'],
                "status": row['status'],
                # "why_stop": row['why_stop'],
                # "label": row['label'],
                # "phase": row['phase'],
                "diseases": str(row['diseases']),
                "disease_category": disease[0],
                "drugs": row['drugs'],            
            }
        )
        trial_docs.append(doc)
    print(f"sample trial doc metadata:\n {trial_docs[0].metadata}")

    # Remove documents with very long content (>10000 characters)
    # trial_docs = [doc for doc in trial_docs if len(doc.page_content) <= 10000]

    list_remove = set()
    for i, doc in enumerate(trial_docs):
        if len(doc.page_content)>10000:
            print(f"removing trial {i} because it's too long")
            list_remove.add(i)
            # print(doc.metadata)
        if doc.metadata['disease_category'] == 'other_conditions':
            print(f"removing trial {i} because it's for other conditions")
            list_remove.add(i)
            # print(doc.metadata)
    # remove list_remove indexes from trial_docs
    trial_docs = [doc for i, doc in enumerate(trial_docs) if i not in list_remove]

    print(f"Number of trial docs to be added to the vector store: {len(trial_docs)}")
    if len(trial_docs) == 0:
        print(f"No trials to add to the vector store")
        return None
        
    

    # Check if collection is empty and add documents if needed
    # if vectorstore._collection.count() == 0:
    vectorstore = Chroma.from_documents(
        documents=trial_docs,
        client=persistent_client,
        collection_name=collection_name,
        # persist_directory=vectorstore_path,
        embedding=NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode='local'),
    )
    print(f"✅ Trial vector store created with {len(trial_docs)} trials")
    # else:
    #     print(f"✅ Trial vector store loaded with {vectorstore._collection.count()} trials")
    
    return vectorstore

def policy_search_node(state: AgentState) -> dict:
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
                "policy_eligible": state.get("policy_eligible", False)
            }
        
        # Create or load policy vector store
        policy_vectorstore = create_policy_vectorstore()
        
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
        return {
            "last_node": "policy_search",
            "policies": [],
            "unchecked_policies": [],
            "policy_eligible": state.get("policy_eligible", False)
        }

def policy_evaluator_node(state: AgentState) -> dict:
    """
    Policy evaluator node that evaluates patient eligibility against institutional policies.
    
    Args:
        state: Current agent state containing patient profile and policies
        
    Returns:
        Updated state with policy evaluation results
    """
    try:
        # Check if there are unchecked policies
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
        
        # Get the first unchecked policy
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
        
        # Create configuration for this node
        config = PatientCollectorConfig(use_free_model=True)
        
        # Create policy questions prompt
        prompt_rps = PromptTemplate(
            template=""" You are a Principal Investigator (PI) for clinical trials. 
                The following document contains a policy document about participation in clinical trials:
                \n\n{policy}\n\n

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
        
        # Create policy questions chain
        policy_rps_chain = prompt_rps | config.model | StrOutputParser()
        
        # Generate policy questions
        policy_qs = policy_rps_chain.invoke({"policy": policy})
        print(f"✅ Generated policy questions: {policy_qs}")
        
        # Evaluate policy using tools
        result = policy_tools(policy_qs, patient_profile, config.model)
        print(f"✅ Policy evaluation result: {result}")
        
        # Parse the evaluation result using structured output
        llm_with_tools = config.model.bind_tools([eligibility])
        message = f"""Evaluation of the patient's eligibility:
        {result}\n\n
        Is the patient eligible according to this policy?"""
        response = llm_with_tools.invoke(message)
        
        # Extract eligibility from tool calls
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
        
        # Update state
        unchecked_policies.pop(0)  # Remove the evaluated policy
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
        return {
            "last_node": "policy_evaluator",
            "policy_eligible": False,
            "rejection_reason": f"Error during evaluation: {str(e)}",
            "revision_number": state.get("revision_number", 0) + 1,
            'checked_policy': None,
            'policy_qs': ""
        }

def trial_search_node(state: AgentState) -> dict:
    """
    Trial search node that searches the trial database to retrieve clinical trials 
    that match the patient's medical history using a self-query retriever.
    
    Args:
        state: Current agent state containing patient profile
        
    Returns:
        Updated state with retrieved trials
    """
    try:
        # Get patient profile from state
        patient_profile = state.get("patient_profile", "")
        
        if not patient_profile:
            print("⚠️ No patient profile available for trial search")
            return {
                'last_node': 'trial_search',
                'trials': [],
                'trial_searches': state.get('trial_searches', 0) + 1,
                "policy_eligible": state.get("policy_eligible", False)
            }
        
        # Create configuration for this node
        config = PatientCollectorConfig(use_free_model=True)
        
        # Create or load trial vector store
        trial_vectorstore = create_trial_vectorstore()
        print(f"Number of trials in the vector store: {trial_vectorstore._collection.count()}")
        
        metadata_field_info = [
            AttributeInfo(
                name="disease_category",
                description="Defines the disease group of patients related to this trial. One of ['cancer', 'leukemia', 'mental_health']",
                # description="The trial is for patients when their disease is related to this category. One of ['cancer', 'leukemia', 'mental_health']",
                type="string",
            ),
            AttributeInfo(
                name="drugs",
                description="List of drug names used in the trial",
                type="str",
            ),    
        ]

        document_content_description = "The list of patient conditions to include or exclude them from the trial"
        retriever_trial_sq = SelfQueryRetriever.from_llm(
            config.model,
            trial_vectorstore,
            # vectorstore_trials_mpnet,
            document_content_description,
            metadata_field_info
            # enable_limit=True
        )
        print(f"patient_profile: {patient_profile}")
        # Create search question
        question = f"""
        Which trials are relevant to the patient with the following medical history?\n
        patient_profile: {patient_profile}
        """
        
        # Retrieve relevant trials
        docs_retrieved = retriever_trial_sq.get_relevant_documents(question)
        print(f"✅ Retrieved {len(docs_retrieved)} relevant trials")
        
        # Update trial searches counter
        trial_searches = state.get('trial_searches', 0) + 1
        
        return {
            'last_node': 'trial_search',
            'trials': docs_retrieved,
            'trial_searches': trial_searches,
            "policy_eligible": state.get("policy_eligible", False)
        }
        
    except Exception as e:
        print(f"❌ Error in trial search: {e}")
        return {
            'last_node': 'trial_search',
            'trials': [],
            'trial_searches': state.get('trial_searches', 0) + 1,
            "policy_eligible": state.get("policy_eligible", False)
        }

def grade_trials_node(state: AgentState) -> dict:
    """
    Trial grading node that evaluates the relevance of retrieved trials to the patient profile.
    
    Args:
        state: Current agent state containing trials and patient profile
        
    Returns:
        Updated state with graded trials
    """
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
        
        # Create configuration for this node
        config = PatientCollectorConfig(use_free_model=True)
        
        # Create prompt for trial grading
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
            """,
            input_variables=["document", "patient_profile", "trial_diseases"],
        )
        
        # Create hallucination detection prompt
        system = """You are a grader assessing whether an LLM generation is grounded in / supported by the facts in the patient's medical profile. \n 
             Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the facts in the patient's medical profile."""
        hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Patient's medical profile: \n\n {patient_profile} \n\n LLM generation: {explanation}"),
            ]
        )
        
        # Create chains
        llm_with_tool = config.model.with_structured_output(grade)
        retrieval_grader = prompt_grader | llm_with_tool
        
        llm_with_tool_hallucination = config.model.with_structured_output(GradeHallucinations)
        hallucination_grader = hallucination_prompt | llm_with_tool_hallucination
        
        # Score each trial
        relevant_trials = []
        for trial in trials:
            doc_txt = trial.page_content
            trial_diseases = trial.metadata['diseases']
            nctid = trial.metadata['nctid']
            print(f"---GRADER: TRIAL {nctid}: ---") 
            
            trial_score = retrieval_grader.invoke(
                {
                    "patient_profile": patient_profile, 
                    "document": doc_txt, 
                    "trial_diseases": trial_diseases
                }
            )
                
            relevance_score = trial_score.relevance_score
            trial_score_dic = dict(trial_score)
            trial_score_dic['nctid'] = nctid                    

            if relevance_score.lower() == "yes":   
                # Hallucination check         
                explanation = trial_score.explanation            
                factual_score = hallucination_grader.invoke({"patient_profile": patient_profile, "explanation": explanation})
                factual_score_grade = factual_score.binary_score            
                if factual_score_grade == "no":
                    print("--- HALLUCINATION: MODEL'S EXPLANATION IS NOT GROUNDED IN PATIENT PROFILE --> REJECTED---")
                    trial_score_dic['relevance_score'] = 'no'
                    trial_score_dic['explanation'] = "Agent's Hallucination"

            if relevance_score.lower() == "yes" and factual_score_grade == "yes":
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
        return {
            'last_node': 'grade_trials',
            "relevant_trials": [],
            "policy_eligible": state.get("policy_eligible", False)
        }

def profile_rewriter_node(state: AgentState) -> dict:
    """
    Profile rewriter node that rewrites patient profile when no relevant trials are found.
    
    Args:
        state: Current agent state containing patient data
        
    Returns:
        Updated state with rewritten patient profile
    """
    try:
        # Get patient data from state
        patient_data = state.get("patient_data", {})
        
        if not patient_data:
            print("⚠️ No patient data available for profile rewriting")
            return {
                'last_node': 'profile_rewriter',
                'patient_profile': state.get("patient_profile", ""),
                "policy_eligible": state.get("policy_eligible", False)
            }
        
        # Create configuration for this node
        config = PatientCollectorConfig(use_free_model=True)
        
        # Create system prompt for profile rewriting
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
        
        # Create profile rewriter prompt
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is a patient data:\n\n {patient_data} \n write a patient profile.",
                ),
            ]
        )
        
        # Create profile rewriter chain
        profile_rewriter_chain = re_write_prompt | config.model | StrOutputParser()
        
        # Generate rewritten patient profile
        patient_profile_rewritten = profile_rewriter_chain.invoke({"patient_data": patient_data})
        
        # Print in capitals
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
            "policy_eligible": state.get("policy_eligible", False)
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


def llm_completion(prompt, model=None, temperature=0, sys_content="You are a helpful assistant."):
    if model == None:
        raise ValueError('Please specify a model ID from openai')
    client = OpenAI()
    messages=[
        {"role": "system", "content": sys_content},
        {"role": "user", "content": prompt}
    ]
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature, 
    )
    output = completion.choices[0].message.content
    return output

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )


def _print_event(event: dict, _printed: set, max_length=1500):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


def dataset_create_trials(status = None):
    """
    Creates a dataset of clinical trials by downloading a CSV file from a GitHub repository and preprocessing it.

    Args:
        dataset_name (str): The name of the dataset. This argument is not used in the function.

    Returns:
        tuple: A tuple containing two elements:
            - df_trials (pandas.DataFrame): The preprocessed dataset of clinical trials.
            - csv_path (str): The path to the CSV file where the dataset is saved.

    Raises:
        None

    Notes:
        - The function downloads a CSV file from the GitHub repository 'futianfan/clinical-trial-outcome-prediction'
          located at 'https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/main/data/raw_data.csv'.
        - The downloaded CSV file is read into a pandas DataFrame called 'df_trials'.
        - The 'diseases' column of the DataFrame is converted from a string representation of lists to actual lists using the 'ast.literal_eval' function.
        - The 'label' column of the DataFrame is mapped to the strings 'success' if the value is 1, and 'failure' if the value is 0.
        - The 'why_stop' column of the DataFrame is filled with the string 'not stopped' where the value is null.
        - The 'smiless' and 'icdcodes' columns of the DataFrame are dropped.
        - The preprocessed DataFrame is saved to a CSV file located at '../data/trials_data.csv'.
        - The path to the saved CSV file is stored in the 'csv_path' variable.
        - The function prints the path to the saved CSV file and the number of rows in the DataFrame.
    """

    import pandas as pd
    import ast

    # URL to the raw_data.csv file
    url = 'https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/main/data/raw_data.csv'

    # Read the CSV file directly into a pandas DataFrame
    df_trials = pd.read_csv(url)

    if status is not None:
        df_trials = df_trials[df_trials['status'] == status].reset_index(drop=True)
        print(f'Only trials with status {status} are selected.')

    # Convert the string representation of lists to actual lists
    df_trials['diseases'] = df_trials['diseases'].apply(ast.literal_eval)
    # df_trials['drugs'] = df_trials['drugs'].apply(ast.literal_eval) 

    # map lable = 1 to success and lable = 0 to failure
    df_trials['label'] = df_trials['label'].map({1: 'success', 0: 'failure'})

    # map why_stop null to not_stopped
    df_trials['why_stop'] = df_trials['why_stop'].fillna('not stopped')
    # df_trials.head()
    
    df_trials = df_trials.drop(columns=['smiless','icdcodes'])

    # create ../data if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        
    df_trials.to_csv('data/trials_data.csv', index=False)
    csv_path = 'data/trials_data.csv'
    print(f'The database for trials is saved to {csv_path} \n It has {len(df_trials)} rows.')
    
    return df_trials, csv_path



def disease_map(disease_list):
    # read disease_mapping from a file    
    import json
    
    # Convert to absolute path relative to the project root
    disease_mapping_path = 'source_data/disease_mapping.json'
    if not os.path.isabs(disease_mapping_path):
        project_root = os.path.dirname(os.path.dirname(__file__))
        disease_mapping_path = os.path.join(project_root, disease_mapping_path)
    
    with open(disease_mapping_path, 'r') as file:
        disease_mapping =  json.load(file)

    categories = set()
    for disease in disease_list:
        if disease in disease_mapping:
            mapped = disease_mapping[disease]
            if mapped != 'other_conditions':
                # mapped = disease 
                categories.add(mapped)
            elif 'cancer' in disease:
                mapped = 'cancer'
            elif 'leukemia' in disease:
                mapped = 'leukemia'            
        # else:
        #     mapped = 'other_conditions'
            # categories.add(disease)
    if len(categories) == 0:
        categories.add('other_conditions')
    return list(categories)

def fn_get_state(graph, thread, vernose = False, next = None):
    states = []
    state_next = []
    for state in graph.get_state_history(thread):
        if vernose:
            print(state)
            print('--')
        states.append(state)
    state_last = states[0]
    
    if next is not None:
        for state in graph.get_state_history(thread):
            if len(state.next)>0 and state.next[0] == next:
                state_next = state
                break
    return states, state_next, state_last

def resume_from_state(graph, state, key = None, value = None , as_node = None):
    if key is not None:
        state.values[key] = value
    
    if as_node is not None:
        branch_state = graph.update_state(state.config, state.values, as_node = as_node)
    else:
        branch_state = graph.update_state(state.config, state.values)
    print("--------- continue from modified state ---------")
    events = []
            
    for event in graph.stream(None, branch_state):
        events.append(event)
        print(event)
    return events
