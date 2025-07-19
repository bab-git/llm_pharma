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
from langchain_core.pydantic_v1 import BaseModel, Field
from operator import itemgetter
from typing import Literal
from langgraph.graph import StateGraph, END
import sqlite3

import os
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
# openai.api_key = os.environ['OPENAI_API_KEY']

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
    """Placeholder for patient collector node."""
    # TODO: Implement actual patient data collection logic
    return {
        "last_node": "patient_collector",
        "patient_data": state.get("patient_data", {}),
        "patient_profile": state.get("patient_profile", ""),
        "patient_id": state.get("patient_id", 0),
        "revision_number": state.get("revision_number", 0) + 1,
        'policy_eligible': 'N/A'
    }

def policy_search_node(state: AgentState) -> dict:
    """Placeholder for policy search node."""
    # TODO: Implement actual policy search logic
    return {
        "last_node": "policy_search",
        "policies": [],
        "unchecked_policies": [],
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
    }

def grade_trials_node(state: AgentState) -> dict:
    """Placeholder for trial grading node."""
    # TODO: Implement actual trial grading logic
    return {
        'last_node': 'grade_trials',
        "relevant_trials": []
    }

def profile_rewriter_node(state: AgentState) -> dict:
    """Placeholder for profile rewriter node."""
    # TODO: Implement actual profile rewriting logic
    return {
        'last_node': 'profile_rewriter',
        'patient_profile': state.get("patient_profile", "")
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
