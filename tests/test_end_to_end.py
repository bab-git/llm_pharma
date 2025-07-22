#!/usr/bin/env python3
"""
End-to-end test for the patient collection workflow.
"""

import sys
import os

# Add the backend directory to the Python path
backend_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backend")
sys.path.append(backend_path)

def test_end_to_end_workflow():
    """Test the complete patient collection workflow."""
    print("🧪 Testing End-to-End Patient Collection Workflow")
    print("=" * 60)
    
    try:
        # Import all necessary components
        from my_agent.patient_collector import (
            patient_collector_node,
            create_agent_state,
            AgentState
        )
        from my_agent.policy_service import (
            policy_search_node,
            policy_evaluator_node
        )
        from my_agent.trial_service import trial_search_node, grade_trials_node
        from my_agent.workflow_manager import WorkflowManager
        from my_agent.llm_manager import LLMManager
        
        print("✅ All imports successful")
        
        # Test 1: Create workflow manager
        print("\n1. Testing WorkflowManager creation...")
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        workflow_manager = WorkflowManager(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool)
        print("✅ WorkflowManager created successfully")
        
        # Test 2: Test patient collection
        print("\n2. Testing patient collection...")
        state = create_agent_state()
        state['patient_prompt'] = "I need information about patient 1"
        
        result = patient_collector_node(state)
        print(f"✅ Patient collection completed")
        print(f"   - Patient ID: {result.get('patient_id', 'N/A')}")
        print(f"   - Patient Profile: {result.get('patient_profile', 'N/A')[:100]}...")
        print(f"   - Last Node: {result.get('last_node', 'N/A')}")
        
        # Test 3: Test policy search
        print("\n3. Testing policy search...")
        state.update(result)
        policy_result = policy_search_node(state)
        print(f"✅ Policy search completed")
        print(f"   - Policies found: {len(policy_result.get('policies', []))}")
        print(f"   - Last Node: {policy_result.get('last_node', 'N/A')}")
        
        # Test 4: Test policy evaluation
        print("\n4. Testing policy evaluation...")
        state.update(policy_result)
        if state.get('unchecked_policies'):
            eval_result = policy_evaluator_node(state)
            print(f"✅ Policy evaluation completed")
            print(f"   - Policy Eligible: {eval_result.get('policy_eligible', 'N/A')}")
            print(f"   - Last Node: {eval_result.get('last_node', 'N/A')}")
        else:
            print("⚠️ No policies to evaluate")
        
        # Test 5: Test trial search
        print("\n5. Testing trial search...")
        state.update(policy_result)
        trial_result = trial_search_node(state)
        print(f"✅ Trial search completed")
        print(f"   - Trials found: {len(trial_result.get('trials', []))}")
        print(f"   - Last Node: {trial_result.get('last_node', 'N/A')}")
        
        # Test 6: Test trial grading
        print("\n6. Testing trial grading...")
        state.update(trial_result)
        if state.get('trials'):
            grade_result = grade_trials_node(state)
            print(f"✅ Trial grading completed")
            print(f"   - Relevant trials: {len(grade_result.get('relevant_trials', []))}")
            print(f"   - Trial found: {grade_result.get('trial_found', 'N/A')}")
            print(f"   - Last Node: {grade_result.get('last_node', 'N/A')}")
        else:
            print("⚠️ No trials to grade")
        
        print("\n" + "=" * 60)
        print("✅ End-to-end workflow test completed successfully!")
        print("\nSummary:")
        print(f"   - Patient ID: {result.get('patient_id', 'N/A')}")
        print(f"   - Policies found: {len(policy_result.get('policies', []))}")
        print(f"   - Trials found: {len(trial_result.get('trials', []))}")
        print(f"   - Workflow completed without errors")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error in workflow: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_manager_integration():
    """Test the workflow manager integration."""
    print("\n🧪 Testing WorkflowManager Integration")
    print("=" * 60)
    
    try:
        from my_agent.workflow_manager import WorkflowManager
        from my_agent.llm_manager import LLMManager
        
        # Create workflow manager
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        workflow_manager = WorkflowManager(llm_manager=llm_manager, llm_manager_tool=llm_manager_tool)
        
        # Test workflow execution
        patient_prompt = "I need information about patient 1"
        result = workflow_manager.run_workflow(patient_prompt)
        
        print("✅ WorkflowManager integration test successful")
        print(f"   - Workflow completed: {result.get('completed', False)}")
        print(f"   - Final state: {result.get('final_state', {}).get('last_node', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in workflow manager integration: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting comprehensive app testing...")
    
    # Test end-to-end workflow
    workflow_success = test_end_to_end_workflow()
    
    # Test workflow manager integration
    integration_success = test_workflow_manager_integration()
    
    if workflow_success and integration_success:
        print("\n🎉 All tests passed! The app is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 