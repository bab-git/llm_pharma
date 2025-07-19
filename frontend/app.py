#!/usr/bin/env python3
"""
LLM Pharma Frontend App

A clean Gradio dashboard for the LLM Pharma clinical trial management system.
This app creates a dummy graph object to run the dashboard GUI.

Usage:
    python frontend/app.py [--port PORT] [--host HOST] [--share]
"""

import os
import sys
import sqlite3
import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def create_dummy_graph():
    """
    Create a dummy graph object using the notebook code snippets.
    This allows the dashboard GUI to run without the full backend setup.
    """
    try:
        # Import required components
        from typing import Annotated, List
        from typing_extensions import TypedDict
        from langgraph.graph.message import AnyMessage, add_messages
        from langchain_core.messages import AIMessage, SystemMessage, HumanMessage, ToolMessage
        from langchain_core.documents import Document
        from langgraph.graph import StateGraph, END
        from langgraph.checkpoint.sqlite import SqliteSaver
        
        print("✅ Successfully imported LangGraph components")
        
        # Define AgentState (from notebook snippet)
        class AgentState(TypedDict):
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
        
        print("✅ AgentState defined")
        
        # Create dummy node functions
        def dummy_patient_collector(state):
            """Dummy patient collector node"""
            return {
                "last_node": "patient_collector",
                "patient_data": {"age": 45, "condition": "test"},
                "patient_profile": "Test patient profile",
                "patient_id": 1,
                "revision_number": 1,
                'policy_eligible': 'N/A'
            }
        
        def dummy_policy_search(state):
            """Dummy policy search node"""
            return {
                "last_node": "policy_search",
                "policies": [],
                "unchecked_policies": [],
            }
        
        def dummy_policy_evaluator(state):
            """Dummy policy evaluator node"""
            return {
                "last_node": "policy_evaluator",
                "policy_eligible": True,
                "rejection_reason": "N/A",
                "revision_number": 1,
                'checked_policy': None,
                'policy_qs': "",
                'unchecked_policies': []
            }
        
        def dummy_trial_search(state):
            """Dummy trial search node"""
            return {
                'last_node': 'trial_search',
                'trials': [],
                'trial_searches': 1,
            }
        
        def dummy_grade_trials(state):
            """Dummy grade trials node"""
            return {
                'last_node': 'grade_trials',
                "relevant_trials": []
            }
        
        def dummy_profile_rewriter(state):
            """Dummy profile rewriter node"""
            return {
                'last_node': 'profile_rewriter',
                'patient_profile': "Updated test patient profile"
            }
        
        # Dummy conditional functions
        def dummy_should_continue_patient(state):
            return "policy_search"
        
        def dummy_should_continue_policy(state):
            return "trial_search"
        
        def dummy_should_continue_trials(state):
            return END
        
        print("✅ Dummy node functions created")
        
        # Create StateGraph builder (from notebook snippet)
        builder = StateGraph(AgentState)
        builder.set_entry_point("patient_collector")
        
        # Add nodes
        builder.add_node("patient_collector", dummy_patient_collector)
        builder.add_node("policy_search", dummy_policy_search)
        builder.add_node("policy_evaluator", dummy_policy_evaluator)
        builder.add_node("trial_search", dummy_trial_search)
        builder.add_node("grade_trials", dummy_grade_trials)
        builder.add_node("profile_rewriter", dummy_profile_rewriter)
        
        # Add edges
        builder.add_conditional_edges(
            "patient_collector", 
            dummy_should_continue_patient, 
            {END: END, "policy_search": "policy_search"}
        )
        
        builder.add_conditional_edges(
            "policy_evaluator", 
            dummy_should_continue_policy, 
            {"trial_search": "trial_search", "policy_evaluator": "policy_evaluator", END: END}
        )
        
        builder.add_edge("policy_search", "policy_evaluator")
        builder.add_edge("trial_search", "grade_trials")
        builder.add_edge("profile_rewriter", "trial_search")
        
        builder.add_conditional_edges(
            "grade_trials", 
            dummy_should_continue_trials, 
            {"profile_rewriter": "profile_rewriter", END: END}
        )
        
        print("✅ StateGraph builder created with nodes and edges")
        
        # Setup SQLite memory (from notebook snippet)
        conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
        memory = SqliteSaver(conn)
        print("✅ SQLite memory initialized")
        
        # Compile graph (from notebook snippet)
        graph = builder.compile(
            checkpointer=memory,
            interrupt_after=['patient_collector', 'policy_search', 'trial_search', 'grade_trials', 'profile_rewriter']
        )
        
        print("✅ Graph compiled successfully")
        return graph
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please ensure LangGraph and LangChain are installed")
        return None
    except Exception as e:
        print(f"❌ Error creating dummy graph: {e}")
        import traceback
        traceback.print_exc()
        return None

def launch_dashboard(host="127.0.0.1", port=7958, share=False):
    """
    Launch the Gradio dashboard using the trials_gui class.
    """
    print("🚀 Initializing LLM Pharma Dashboard...")
    
    try:
        # Import the GUI class
        from src.helper_gui import trials_gui
        print("✅ Imported trials_gui class")
        
        # Create dummy workflow graph
        graph = create_dummy_graph()
        if graph is None:
            print("❌ Failed to create dummy graph")
            return
        
        print("✅ Dummy workflow graph created")
        
        # Create the GUI application
        app = trials_gui(graph, share=share)
        print("✅ Dashboard created successfully")
        
        # Launch the interface
        print(f"🌐 Launching dashboard on http://{host}:{port}")
        print("🔔 Press Ctrl+C to stop the server")
        print("💡 Note: This is running with dummy data for demonstration")
        
        # Launch with custom host/port if different from defaults
        if port != 7958 or host != "127.0.0.1":
            app.demo.launch(server_name=host, server_port=port, share=share)
        else:
            app.launch(share=share)
            
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please ensure helper_gui.py is in the src/ directory")
    except Exception as e:
        print(f"❌ Failed to launch dashboard: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="LLM Pharma Frontend App (Dummy Mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python frontend/app.py
  python frontend/app.py --port 8080
  python frontend/app.py --host 0.0.0.0 --share

Note: This runs with dummy data for demonstration purposes.
        """
    )
    
    parser.add_argument(
        "--host", 
        default="127.0.0.1", 
        help="Host to bind the server to (default: 127.0.0.1)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=7958, 
        help="Port to bind the server to (default: 7958)"
    )
    
    parser.add_argument(
        "--share", 
        action="store_true", 
        help="Create a public shareable link"
    )
    
    args = parser.parse_args()
    
    print("🏥 LLM Pharma - Clinical Trial Management Dashboard")
    print("🎭 Running in DUMMY MODE with test data")
    print("=" * 60)
    
    try:
        launch_dashboard(host=args.host, port=args.port, share=args.share)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 