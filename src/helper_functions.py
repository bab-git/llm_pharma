from openai import OpenAI
import os

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