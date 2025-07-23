#!/usr/bin/env python3
"""
Simplified LLM Pharma Frontend App
"""

import argparse
import sys
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from helper_gui import trials_gui

# Add OmegaConf import for config loading
from omegaconf import OmegaConf


# Optional backend imports; deferred until needed
def ensure_path_exists(path: Path, factory, *args, **kwargs):
    """
    Create the file/database at `path` if it doesn't exist by calling `factory`.
    """
    if not path.exists():
        print(f"🔄 Creating {path.name}...")
        factory(*args, **kwargs)
    else:
        print(f"✅ {path.name} already exists")


def make_absolute(relative: str) -> Path:
    """
    Convert a project-relative path into an absolute Path.
    """
    p = Path(relative)
    return p if p.is_absolute() else Path(__file__).parent.parent / p


def create_workflow_manager(demo: bool, configs=None):
    """
    Build and return the WorkflowManager (demo or production).
    """
    if demo:
        from demo_graph import create_demo_graph
        # Demo returns a compiled graph directly, not a WorkflowManager
        return create_demo_graph()

    # import os
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from backend.my_agent.llm_manager import LLMManager
    from backend.my_agent.workflow_manager import WorkflowManager

    # Use passed configs
    llm_manager = LLMManager.from_config(configs, use_tool_models=False)
    llm_manager_tool = LLMManager.from_config(configs, use_tool_models=True)

    # Create and return workflow manager
    workflow_manager = WorkflowManager(
        llm_manager=llm_manager, llm_manager_tool=llm_manager_tool, configs=configs
    )
    return workflow_manager


def main():
    # Load environment and parse CLI
    load_dotenv(find_dotenv())
    print(f"OpenAI API Key: {os.environ.get('OPENAI_API_KEY', 'Not found')}")
    print(f"Groq API Key: {os.environ.get('GROQ_API_KEY', 'Not found')[-20:]}")

    parser = argparse.ArgumentParser(description="LLM Pharma Frontend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7958)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    # Load configs at the top
    # import os
    from omegaconf import OmegaConf
    configs_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    configs = OmegaConf.load(configs_path)

    # Ensure necessary databases
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
    from backend.my_agent.database_manager import DatabaseManager

    db_manager = DatabaseManager(configs=configs)
    patients_db = make_absolute(os.path.join(configs.directories.sql_server, "patients.db"))
    ensure_path_exists(patients_db, db_manager.create_demo_patient_database)

    trials_csv = make_absolute(configs.files.trials_csv)
    ensure_path_exists(
        trials_csv, db_manager.create_trials_dataset, status="recruiting"
    )

    # Build workflow manager
    workflow_manager = create_workflow_manager(demo=args.demo, configs=configs)
    if not workflow_manager:
        sys.exit("❌ Failed to create workflow manager. Try --demo for testing.")

    # Launch Gradio dashboard with workflow manager
    app = trials_gui(workflow_manager, share=args.share)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
