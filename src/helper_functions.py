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
        - The preprocessed DataFrame is saved to a CSV file located at '../../data/trials_data.csv'.
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
    df_trials.to_csv('../../data/trials_data.csv', index=False)
    csv_path = '../../data/trials_data.csv'
    print(f'The database for trials is saved to {csv_path} \n It has {len(df_trials)} rows.')
    
    return df_trials, csv_path



def disease_map(disease_list):
    # read disease_mapping from a file    
    import json
    
    with open('../../source_data/disease_mapping.json', 'r') as file:
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


from langgraph.graph import StateGraph, END
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




# GRAPH
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.documents import Document
from langchain.agents import AgentExecutor, create_openai_tools_agent


# from typing import TypedDict

class AgentState(TypedDict):
    # messages: Annotated[Sequence[BaseMessage], operator.add]
    patient_prompt: str
    patient_id: int    
    policy_eligible: bool
    rejection_reason: str
    patient_data: dict
    patient_profile: str
    revision_number: int
    max_revisions: int
    policies: List[Document]
    checked_policy: Document
    unchecked_policies: List[Document]
    trials: List[Document]
    relevant_trials: list[dict]
    ask_expert: str


# nodes
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain_core.pydantic_v1 import BaseModel, Field
from operator import itemgetter
from typing import Literal
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime, timedelta
from langchain_core.tools import Tool
from langchain_experimental.utilities import PythonREPL
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI





# def policy_evaluator(patient_profile: str, policy: str):
def policy_evaluator(state: AgentState, policy_doc: Document = None):
    if policy_doc is None:
        policy_doc = state['unchecked_policies'][0]
    
    policy_header = policy_doc.page_content.split('\n', 2)[1]
    print(f'Evaluating Policy:\n {policy_header}')
    
    policy = policy_doc.page_content

    model = ChatOpenAI(temperature = 0.0, model="gpt-3.5-turbo")

    
    class codesc(BaseModel):
        code: str = Field(
            description="The code to execute. IMPORTANT: the last line of code must always finish with a print(...)."
        )
    python_repl = PythonREPL()
    python_repl_tool = Tool(
        name="python_repl",
        description="""
        A Python shell. Use this to execute python commands. Input should be a valid python command. IMPORTANT: alwyas print the output value out with `print(...)`.

        Wrong code example:
        ```python
        A = 12
        B = 20
        A_bigger = A > B    
        ```

        Correct code example:
        ```python
        A = 12
        B = 20
        A_bigger = A > B
        print("A_bigger")
        ```
        """,
        func=python_repl.run,
        args_schema=codesc,
    )
    
    # promptA = PromptTemplate(
    #     template=""" You are a Principal Investigator (PI) for clinical trials. 
    #         Use the following clinial trials policy and simplify it.
    #         Use simpler language to define the regulations.
    #         Remove headers. Remove topic titles.
    #         Only keep the regulations that may prevent patients from being eligible for trials.
            
    #         The retrieved policies: \n\n {context} \n\n        
    #         """,
    #     input_variables=["context"],
    # )


    # promptB = PromptTemplate(
    #     template=""" You are a Principal Investigator (PI) for clinical trials. 
    #         You are given the following patient's profile:\n
    #         {patient_profile}\n
            
    #         You also have access to the following trial policies:\n
    #         {policies}\n
            
    #         Simplify the policies.\n
    #         Determine which policy is relevant to which information in patient's profile.        
    #         """,
    #     input_variables=["patient_profile", "policies"],
    # )
    # For each policy, if the information you need does not exist in the patient's profile remove the policy it.\n
    # At the end, suggest how to determine whether the patient is eligible according to the remaining policies.
    

    # outputparse = StrOutputParser()

    # chainA = promptA | model | outputparse
    # policy_simplified = chainA.invoke(
    #     {"context": policy}
    # )
    
    prompt_repl = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Principal Investigator (PI) for evaluating patients for clinical trials." 
                "You are asked to compare the patient profile document to the institution policies document."
                "You must determine if the patient is eligible based on the following documnents or not."
                # "Use the following institution policy(s) to find if the patient is eligibile for trials."
                "\n #### Here is the patient profile document: \n {patient_profile}\n\n"
                # "\n Here is/are the retrieved policy(s) document: \n {context}\n\n"

                "-------------------\n"
                
                "Today date is {date}.\n\n"

                # "EXCLUSION terms in the patient profile should be removed from policies.\n"                                                
                "Do not use your own knowledege regarding clinical trials. Restrict your judgment Only to the information given in documents.\n"
                "You have access to the following tools: {tool_names}."
                # "\n You may generate only safe python code."
                "\n IMPORTANT: Do not use this tool {tool_names} when thre is no numerical or date related information in the policy document.\n\n"
                
                "Give a binary 'yes' or 'no' score in the response to indicate whether the patient is eligible according to the given policy ONLY."
                "If the patient is not eligible then also include the reason in your response.", 
            ),
            # MessagesPlaceholder(variable_name="examples"),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            # (
            #     "system",
            #     # "Note: Current date is {date}.\n"
    
            #     "Give a binary 'yes' or 'no' score in the response to indicate whether the patient is eligible according to the given policy ONLY."
            #     "If the patient is not eligible then also include the reason in your response."
            #     # Use it as the reference date to evaluate time related regulations."
            # ),
            # MessagesPlaceholder(variable_name="examples"),
        ]
    )   

    # examples = """
    # Example of date comparison:
    # query: is November 25, 2023 within the 5 months before the current date, 25-03-2024?

    # generated code:
    # ```
    # gap_months = 5
    # today = datetime.strptime('25-03-2024', '%d-%m-%Y')
    # trial_completion_date = datetime.strptime('25-11-2023', '%d-%m-%Y')
    # days_difference = (today - trial_completion_date).days
    # if days_difference < gap_months*30:
    #     trial_within_5_months = True
    # else:
    #     trial_within_5_months = False
    # print(trial_within_5_months)
    # ```
    # """
    # tools = [python_repl]
    
    tools = [python_repl_tool]
    patient_profile = state["patient_profile"]
    
    date = datetime.today().date()
    prompt_repl = prompt_repl.partial(tool_names=", ".join([tool.name for tool in tools]))
    prompt_repl = prompt_repl.partial(date=date)
    # prompt_repl = prompt_repl.partial(context=docs)
    prompt_repl = prompt_repl.partial(patient_profile=patient_profile)
    # prompt_repl = prompt_repl.partial(examples= [HumanMessage(content=examples)])

    agent = create_openai_tools_agent(model, tools, prompt_repl)
    agent_with_tool = AgentExecutor(agent=agent, tools=tools)
    
    message = f"Here is the retrieved policy document: \n {policy}\n\n"
    # nprint(message)    
    result = agent_with_tool.invoke({"messages": [HumanMessage(content=message)]} )
    # nprint(result['output'])

    class eligibility(BaseModel):
        """ Give the patient's eligibility result."""

        eligibility: str = Field(description="Patient's eligibility for the clinical trial. 'yes' or 'no'")
        reason: str = Field(description="The reason(s) only if the patient is not eligible for clinical trials. Othereise use N/A")

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


    # prompt_policy = PromptTemplate(
    #     template=""" 
    #     You are a Principal Investigator (PI) for evaluating patients for clinical trials.
    #     You compare the patient profile document to the trials policy document to determine if the patient is eligible for the clinical trials or not.        

    #     \n #### Here is the patient profile document: \n {patient_profile}\n\n

    #     Here are the retrieved eligibility policies: \n\n {policy} \n\n
        
    #     Today date is {date}\n\n.

    #     Compare policy document to patient profile document to determine if the patient is eligible for the clinical trials or not\n.
    #     Restrict your judgment Only to the information given in patient's profile and the policy document given by the user\n"        
    #     Give a binary 'yes' or 'no' score in the response to indicate whether the patient is eligible according to the given policy ONLY.\n
    #     If the patient is not eligible then also include the reason in your response.
    #     """,
    #     input_variables=["patient_profile", "policy", "date"],
    # )
    
    # llm_with_tool = model.with_structured_output(eligibility)    
    # chain_eval = prompt_policy | llm_with_tool
    # response = chain_eval.invoke(
    #     {
    #         "patient_profile": patient_profile, 
    #         "policy": policy, 
    #         "date": date
    #         # "patient_data": patient_data
    #     }
    # )

    llm_with_tools = model.bind_tools([eligibility])
    message = f"""Evaluation of the patient's eligibility:
    {result['output']}\n\n
    Is the patient eligible according to this policy?"""
    response = llm_with_tools.invoke(message)
        

    
    
    state["policy_eligible"] = response.tool_calls[0]['args']['eligibility'] == 'yes'
    state["rejection_reason"] = response.tool_calls[0]['args']['reason']

    # state["policy_eligible"] = response.eligibility == 'yes'
    # state["rejection_reason"] = response.reason
    
    # remove the first element of unchecked_policies
    state['unchecked_policies'].pop(0)
    state["revision_number"] = state.get("revision_number", 1) + 1
    state['checked_policy'] = policy_doc
    state['last_node'] = 'policy_evaluator'

    return state