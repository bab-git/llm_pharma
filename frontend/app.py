#!/usr/bin/env python3
"""
LLM Pharma Frontend App

A clean Gradio dashboard for the LLM Pharma clinical trial management system.
This app can run in production mode (with real backend) or demo mode (with dummy data).

Usage:
    python frontend/app.py [--port PORT] [--host HOST] [--share] [--demo]
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# Add project paths for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

def create_workflow_graph():
    """
    Create the workflow graph using backend helper functions.
    This is the production mode that uses real backend components.
    """
    try:
        # Import backend helper functions
        from backend.helper_functions import (
            create_agent_state,
            create_workflow_builder,
            setup_sqlite_memory
        )
        
        # Create the workflow components
        agent_state = create_agent_state()
        builder = create_workflow_builder(agent_state)
        memory = setup_sqlite_memory()
        
        # Compile the graph directly with interrupts
        graph = builder.compile(
            checkpointer=memory,
            interrupt_after=['patient_collector', 'policy_search', 'trial_search', 'grade_trials', 'profile_rewriter']
        )
        
        return graph
        
    except ImportError as e:
        print(f"❌ Backend components not found: {e}")
        print("💡 Please ensure backend.helper_functions is available")
        return None
    except Exception as e:
        print(f"❌ Error creating workflow graph: {e}")
        return None

def ensure_patient_database_exists():
    """
    Ensure the patient database exists by creating it if it doesn't.
    This function calls create_demo_patient_database from helper functions.
    """
    try:
        from backend.helper_functions import create_demo_patient_database
        
        print("📊 Checking patient database...")
        
        # Check if database already exists
        db_path = "sql_server/patients.db"
        if not os.path.isabs(db_path):
            project_root = Path(__file__).parent.parent
            db_path = project_root / db_path
        
        if db_path.exists():
            print("✅ Patient database already exists")
        else:
            print("🔄 Creating demo patient database...")
            create_demo_patient_database(str(db_path))
            print("✅ Demo patient database created successfully")
            
    except ImportError as e:
        print(f"❌ Could not import create_demo_patient_database: {e}")
        print("💡 Please ensure backend.helper_functions is available")
    except Exception as e:
        print(f"❌ Error creating patient database: {e}")
        print(" Continuing without patient database...")

def ensure_trial_database_exists():
    """
    Ensure the trial database exists by creating it if it doesn't.
    This function calls dataset_create_trials from helper functions.
    """
    try:
        from backend.helper_functions import dataset_create_trials
        
        print("📊 Checking trial database...")
        
        # Check if trial database already exists
        trials_path = "data/trials_data.csv"
        if not os.path.isabs(trials_path):
            project_root = Path(__file__).parent.parent
            trials_path = project_root / trials_path
        
        if trials_path.exists():
            print("✅ Trial database already exists")
        else:
            print("🔄 Creating trial database...")
            df_trials, csv_path = dataset_create_trials(status='recruiting')
            print(f"✅ Trial database created successfully at {csv_path}")
            print(f"📈 Created {len(df_trials)} trials")
            
    except ImportError as e:
        print(f"❌ Could not import dataset_create_trials: {e}")
        print("💡 Please ensure backend.helper_functions is available")
    except Exception as e:
        print(f"❌ Error creating trial database: {e}")
        print(" Continuing without trial database...")

def ensure_vector_stores_exist():
    """
    Ensure the vector stores exist by creating them if they don't.
    This function creates both policy and trial vector stores.
    """
    try:
        from backend.helper_functions import create_policy_vectorstore, create_trial_vectorstore
        
        print("🔍 Checking vector stores...")
        
        # Create policy vector store
        print("📋 Creating policy vector store...")
        policy_vectorstore = create_policy_vectorstore()
        
        # Create trial vector store
        print("🧪 Creating trial vector store...")
        trial_vectorstore = create_trial_vectorstore()
        
        print("✅ Vector stores ready")
        
    except ImportError as e:
        print(f"❌ Could not import vector store functions: {e}")
        print("💡 Please ensure backend.helper_functions is available")
    except Exception as e:
        print(f"❌ Error creating vector stores: {e}")
        print(" Continuing without vector stores...")

def launch_dashboard(host="127.0.0.1", port=7958, share=False, demo_mode=False):
    """
    Launch the Gradio dashboard using the trials_gui class.
    """
    print("🚀 Initializing LLM Pharma Dashboard...")
    
    # Ensure all databases and vector stores exist before creating the workflow
    ensure_patient_database_exists()
    ensure_trial_database_exists()
    # ensure_vector_stores_exist()
    
    try:
        # Import the GUI class
        from src.helper_gui import trials_gui
        print("✅ Imported trials_gui class")
        
        # Create workflow graph based on mode
        if demo_mode:
            print("🎭 Running in DEMO MODE with test data")
            try:
                from demo_graph import create_demo_graph
                graph = create_demo_graph()
                if graph is None:
                    print("❌ Failed to create demo graph")
                    return
                print("✅ Demo workflow graph created")
            except ImportError as e:
                print(f"❌ Demo module not found: {e}")
                print(" Please ensure demo_graph.py is in the frontend directory")
                return
        else:
            print("🏭 Running in PRODUCTION MODE with real backend")
            graph = create_workflow_graph()
            if graph is None:
                print("❌ Failed to create workflow graph")
                print("💡 Try running with --demo for testing")
                return
            print("✅ Production workflow graph created")
        
        # Create the GUI application
        app = trials_gui(graph, share=share)
        print("✅ Dashboard created successfully")
        
        # Launch the interface
        print(f"🌐 Launching dashboard on http://{host}:{port}")
        print("🔔 Press Ctrl+C to stop the server")
        
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
        description="LLM Pharma Frontend App",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python frontend/app.py                    # Production mode
  python frontend/app.py --demo             # Demo mode with test data
  python frontend/app.py --port 8080        # Custom port
  python frontend/app.py --host 0.0.0.0 --share  # Public sharing
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
    
    parser.add_argument(
        "--demo", 
        action="store_true", 
        help="Run in demo mode with test data (for testing GUI)"
    )
    
    args = parser.parse_args()
    
    print("🏥 LLM Pharma - Clinical Trial Management Dashboard")
    print("=" * 60)
    
    try:
        launch_dashboard(host=args.host, port=args.port, share=args.share, demo_mode=args.demo)
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 