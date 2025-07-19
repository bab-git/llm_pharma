from langchain_core.messages import ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import ToolNode

import os
from openai import OpenAI

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
# openai.api_key = os.environ['OPENAI_API_KEY']

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
