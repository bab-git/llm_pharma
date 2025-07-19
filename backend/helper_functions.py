"""
LLM Pharma Helper Functions

This module contains the complete implementation of the patient collector node
and supporting functions for the LLM Pharma clinical trial workflow system.

COMPLETED FEATURES:
==================

1. Patient Collector Node - COMPLETED
   - Extracts patient ID from natural language prompts
   - Fetches patient data from SQLite database
   - Generates patient profile for clinical trial screening
   - Uses Groq model for free LLM inference

2. Demo Patient Database - COMPLETED
   - Pre-populated SQLite database with 5 sample patients
   - Includes medical history, trial participation, demographics
   - Automatic database creation and management

3. Configuration System - COMPLETED
   - PatientCollectorConfig class for model and database setup
   - Support for both OpenAI and Groq models
   - Flexible database path configuration

USAGE EXAMPLE:
==============

    from helper_functions import (
        initialize_patient_collector_system,
        patient_collector_node,
        create_agent_state
    )
    
    # Initialize the system
    config = initialize_patient_collector_system(use_free_model=True)
    
    # Create initial state
    state = create_agent_state()
    state['patient_prompt'] = "I need information about patient 1"
    
    # Run patient collector
    result = patient_collector_node(state)
    
    print(f"Patient ID: {result['patient_id']}")
    print(f"Profile: {result['patient_profile']}")

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
- policy_search_node
- policy_evaluator_node  
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

class Patient_ID(BaseModel):
    """Model for extracting patient ID from user prompt."""
    patient_id: int

class PatientCollectorConfig:
    """Configuration for patient collector node."""
    def __init__(self, use_free_model=True, db_path="sql_server/patients.db"):
        self.use_free_model = use_free_model
        self.db_path = db_path
        self.modelID_groq = "gemma2-9b-it"
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
    csv_path = db_path.replace('.db', '.csv')
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
        "max_trial_searches": 3,
        "trials": [],
        "relevant_trials": [],
        "ask_expert": ""
    }

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
    
    builder.add_edge("policy_search", "policy_evaluator")
    builder.add_edge("trial_search", "grade_trials")
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

def policy_search_node(state: AgentState) -> dict:
    """Placeholder for policy search node."""
    # TODO: Implement actual policy search logic
    return {
        "last_node": "policy_search",
        "policies": [],
        "unchecked_policies": [],
        "policy_eligible": state.get("policy_eligible", False)  # Preserve existing value
    }

def policy_evaluator_node(state: AgentState) -> dict:
    """Placeholder for policy evaluator node."""
    # TODO: Implement actual policy evaluation logic
    return {
        "last_node": "policy_evaluator",
        "policy_eligible": True,
        "rejection_reason": "",
        "revision_number": state.get("revision_number", 0) + 1,
        'checked_policy': None,
        'policy_qs': ""
    }

def trial_search_node(state: AgentState) -> dict:
    """Placeholder for trial search node."""
    # TODO: Implement actual trial search logic
    trial_searches = state.get('trial_searches', 0)
    return {
        'last_node': 'trial_search',
        'trials': [],
        'trial_searches': trial_searches + 1,
        "policy_eligible": state.get("policy_eligible", False)  # Preserve existing value
    }

def grade_trials_node(state: AgentState) -> dict:
    """Placeholder for trial grading node."""
    # TODO: Implement actual trial grading logic
    return {
        'last_node': 'grade_trials',
        "relevant_trials": [],
        "policy_eligible": state.get("policy_eligible", False)  # Preserve existing value
    }

def profile_rewriter_node(state: AgentState) -> dict:
    """Placeholder for profile rewriter node."""
    # TODO: Implement actual profile rewriting logic
    return {
        'last_node': 'profile_rewriter',
        'patient_profile': state.get("patient_profile", ""),
        "policy_eligible": state.get("policy_eligible", False)  # Preserve existing value
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

def should_continue_trials(state: AgentState) -> str:
    """Determine if trial search should continue."""
    relevant_trials = state.get("relevant_trials", [])
    has_relevant_trial = any(trial.get('relevance_score') == 'Yes' for trial in relevant_trials)
    
    if state.get("trial_searches", 0) > state.get("max_trial_searches", 3):
        return END
    elif not has_relevant_trial:
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
    if not os.path.exists('../data'):
        os.makedirs('../data')
        
    df_trials.to_csv('../data/trials_data.csv', index=False)
    csv_path = '../data/trials_data.csv'
    print(f'The database for trials is saved to {csv_path} \n It has {len(df_trials)} rows.')
    
    return df_trials, csv_path



def disease_map(disease_list):
    # read disease_mapping from a file    
    import json
    
    with open('../source_data/disease_mapping.json', 'r') as file:
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
