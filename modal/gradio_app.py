import modal
from pathlib import Path

parent_dir = Path(__file__).parent
project_root = parent_dir.parent

web_image = (
    modal.Image.debian_slim(python_version="3.12").pip_install(
        "fastapi[standard]==0.115.4",
        "gradio~=5.7.1",
        # "pillow~=10.2.0",
        # "gradio==5.38.0",    
        "pillow~=10.4.0",    
        "chromadb==1.0.15",
        "gpt4all==2.8.2",
        "hydra-core==1.3.2",
        "langchain==0.3.26",
        "langchain-community==0.3.27",
        "langchain-core==0.3.69",
        "langchain-nomic==0.1.4",
        "langchain-openai==0.3.28",
        "langchain-groq==0.3.6",
        "langgraph-checkpoint-sqlite==2.0.10",
        "langgraph==0.4.5",
        "lark==1.2.2",
        "nomic==3.5.3",
        "numpy==2.3.1",
        "pandas==2.3.1",
        "python-dotenv==1.1.0",
    )
    # .add_local_dir(project_root / "frontend", "/root/frontend")
    .add_local_dir(project_root / "backend", "/root/backend")
    .add_local_dir(project_root / "config", "/root/config")    
    .add_local_dir(project_root / "vector_store", "/root/vector_store")
    .add_local_dir(project_root / "source_data", "/root/source_data")
    .add_local_file(project_root / "data/trials_data.csv", "/root/data/trials_data.csv")
    .add_local_file(project_root / "frontend/static/image_source.jpg", "/root/frontend/static/image_source.jpg")
    .add_local_file(project_root / "frontend/helper_gui.py", "/root/frontend/helper_gui.py")
    .add_local_file(project_root / "frontend/app.py", "/root/frontend/app.py")
    .add_local_file(project_root / "frontend/style.css", "/root/frontend/style.css")
    .add_local_file(project_root / "frontend/__init__.py", "/root/frontend/__init__.py")
    .add_local_file(project_root / "sql_server/patients.db", "/root/sql_server/patients.db")
)

app = modal.App("llm-pharma-agent")

from frontend.helper_gui import trials_gui
# from frontend.app import create_workflow_manager  # adjust import path if necessary
from backend.my_agent.workflow_manager import WorkflowManager


@app.function(
    # gpu="A10G:1",
    image=web_image,
    min_containers=0,
    scaledown_window=60 * 3,
    secrets=[
        modal.Secret.from_name("groq-secret-pharma"),
        modal.Secret.from_name("openai_dummy"),
    ],
    # gradio requires sticky sessions
    # so we limit the number of concurrent containers to 1
    # and allow it to scale to 100 concurrent inputs
    max_containers=1,
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """A simple Gradio interface for a greeting function."""

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    import time
    from omegaconf import OmegaConf

    # from frontend.helper_gui import trials_gui
    # from frontend.app import create_workflow_manager  # adjust import path if necessary


    configs = OmegaConf.load("config/config.yaml")
    workflow_manager = WorkflowManager(configs=configs)

    # workflow_manager = create_workflow_manager(demo=False, configs=configs)

    import os

    # Prepare the output to be shown at page load
    frontend_files = os.listdir("frontend")
    workflow_manager_str = "\n\n".join(frontend_files) + "\n\n" + str(workflow_manager)
    greeting = "Hello, User!"  # Default greeting
    output_text = f"{greeting}\n\n---\nWorkflow Manager:\n{workflow_manager_str}"

    # with gr.Blocks() as demo:
    #     gr.Markdown(output_text)
    # demo.queue(max_size=5)  # Enable queue for handling multiple request

    # mount the app at the root path    # build the Gradio Blocks
    # demo: "gradio.Blocks" = trials_gui(workflow_manager, share=False)
    trials_gui_instance = trials_gui(workflow_manager, share=False)
    demo = trials_gui_instance.demo  # Get the actual Gradio interface

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")