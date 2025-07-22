#!/usr/bin/env python3
"""
Simplified LLM Pharma Frontend App
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from helper_gui import trials_gui

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


def create_workflow(demo: bool):
    """
    Build and return the LangGraph workflow graph (demo or production).
    """
    if demo:
        from demo_graph import create_demo_graph
        return create_demo_graph()

    from backend.helper_functions import (
        create_agent_state,
        create_workflow_builder,
        setup_sqlite_memory
    )
    state = create_agent_state()
    builder = create_workflow_builder(state)
    memory = setup_sqlite_memory()
    return builder.compile(
        checkpointer=memory,
        interrupt_after=[
            'patient_collector', 'policy_search', 'policy_evaluator',
            'trial_search', 'grade_trials', 'profile_rewriter'
        ]
    )


def main():
    # Load environment and parse CLI
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(description="LLM Pharma Frontend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7958)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    # Ensure necessary databases
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from backend.my_agent.database_manager import DatabaseManager
    
    db_manager = DatabaseManager()
    patients_db = make_absolute("sql_server/patients.db")
    ensure_path_exists(patients_db, db_manager.create_demo_patient_database)

    trials_csv = make_absolute("data/trials_data.csv")
    ensure_path_exists(trials_csv, db_manager.create_trials_dataset, status='recruiting')

    # Build workflow graph
    graph = create_workflow(demo=args.demo)
    if not graph:
        sys.exit("❌ Failed to create workflow graph. Try --demo for testing.")

    # Launch Gradio dashboard
    app = trials_gui(graph, share=args.share)
    app.launch(share=args.share)


if __name__ == "__main__":
    main()
