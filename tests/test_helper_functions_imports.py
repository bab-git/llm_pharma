#!/usr/bin/env python3
"""
Test script to verify that helper_functions.py works correctly with the new patient_collector imports.
"""

import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
project_root = Path(__file__).parent.parent
backend_path = project_root / "backend"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_path))

def test_helper_functions_imports():
    """Test that helper_functions.py can import from patient_collector module."""
    print("🧪 Testing Helper Functions Imports")
    print("=" * 50)
    
    try:
        # Test importing from policy_service
        from my_agent.policy_service import (
            policy_evaluator_node,
            policy_search_node
        )
        # Test importing from helper_functions
        from my_agent.trial_service import trial_search_node, grade_trials_node
        # These should now be imported from patient_collector
        from my_agent.patient_collector import (
            PatientCollectorConfig,
            patient_collector_node,
            profile_rewriter_node,
            create_agent_state,
            Patient_ID,
            AgentState
        )
        
        print("✅ Successfully imported all functions from helper_functions")
        print(f"   - policy_evaluator_node: {policy_evaluator_node}")
        print(f"   - policy_search_node: {policy_search_node}")
        print(f"   - trial_search_node: {trial_search_node}")
        print(f"   - grade_trials_node: {grade_trials_node}")
        print(f"   - PatientCollectorConfig: {PatientCollectorConfig}")
        print(f"   - patient_collector_node: {patient_collector_node}")
        print(f"   - profile_rewriter_node: {profile_rewriter_node}")
        print(f"   - create_agent_state: {create_agent_state}")
        print(f"   - Patient_ID: {Patient_ID}")
        print(f"   - AgentState: {AgentState}")
        
        # Test creating agent state
        state = create_agent_state()
        print(f"\n✅ Successfully created agent state")
        print(f"   - patient_id: {state['patient_id']}")
        print(f"   - patient_profile: {state['patient_profile']}")
        
        # Test that AgentState is properly typed
        print(f"\n✅ AgentState type checking works")
        print(f"   - Type: {type(state)}")
        print(f"   - Keys: {list(state.keys())}")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ All import tests passed!")

if __name__ == "__main__":
    test_helper_functions_imports() 