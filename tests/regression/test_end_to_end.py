#!/usr/bin/env python3
"""
End-to-end test for the patient collection workflow.
"""

import os
import sys

# Add the backend directory to the Python path
# Add the parent directory to the path to import helper_functions
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def test_end_to_end_workflow():
    """Test the complete patient collection workflow."""
    print("🧪 Testing End-to-End Patient Collection Workflow")
    print("=" * 60)

    try:
        # Import all necessary components
        from backend.my_agent.database_manager import DatabaseManager
        from backend.my_agent.llm_manager import LLMManager
        from backend.my_agent.patient_collector import PatientService
        from backend.my_agent.policy_service import PolicyService
        from backend.my_agent.State import create_agent_state
        from backend.my_agent.trial_service import TrialService

        print("✅ All imports successful")

        # Test 1: Create shared dependencies and services
        print("\n1. Testing service creation with dependency injection...")

        # Create shared dependencies
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        db_manager = DatabaseManager()

        # Create service instances with injected dependencies
        policy_service = PolicyService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager,
        )
        patient_service = PatientService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager,
        )
        trial_service = TrialService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager,
        )
        print("✅ Services created successfully with dependency injection")

        # Test 2: Test patient collection
        print("\n2. Testing patient collection...")
        state = create_agent_state()
        state["patient_prompt"] = "I need information about patient 41"

        result = patient_service.patient_collector_node(state)
        print("✅ Patient collection completed")
        print(f"   - Patient ID: {result.get('patient_id', 'N/A')}")
        print(f"   - Patient Profile: {result.get('patient_profile', 'N/A')[:100]}...")
        print(f"   - Last Node: {result.get('last_node', 'N/A')}")

        # Test 3: Test policy search
        print("\n3. Testing policy search...")
        state.update(result)
        policy_result = policy_service.policy_search_node(state)
        print("✅ Policy search completed")
        print(f"   - Policies found: {len(policy_result.get('policies', []))}")
        print(f"   - Last Node: {policy_result.get('last_node', 'N/A')}")

        # Test 4: Test policy evaluation
        print("\n4. Testing policy evaluation...")
        state.update(policy_result)
        if state.get("unchecked_policies"):
            eval_result = policy_service.policy_evaluator_node(state)
            print("✅ Policy evaluation completed")
            print(f"   - Policy Eligible: {eval_result.get('policy_eligible', 'N/A')}")
            print(f"   - Last Node: {eval_result.get('last_node', 'N/A')}")
        else:
            print("⚠️ No policies to evaluate")

        # Test 5: Test trial search
        print("\n5. Testing trial search...")
        state.update(policy_result)
        trial_result = trial_service.trial_search_node(state)
        print("✅ Trial search completed")
        print(f"   - Trials found: {len(trial_result.get('trials', []))}")
        print(f"   - Last Node: {trial_result.get('last_node', 'N/A')}")

        # Test 6: Test trial grading
        print("\n6. Testing trial grading...")
        state.update(trial_result)
        if state.get("trials"):
            grade_result = trial_service.grade_trials_node(state)
            print("✅ Trial grading completed")
            print(
                f"   - Relevant trials: {len(grade_result.get('relevant_trials', []))}"
            )
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
        print("   - Workflow completed without errors")

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
        from backend.my_agent.workflow_manager import WorkflowManager

        # Create workflow manager
        workflow_manager = WorkflowManager()

        # Test workflow execution
        patient_prompt = "I need information about patient 1"
        result = workflow_manager.run_workflow(patient_prompt)

        print("✅ WorkflowManager integration test successful")
        print(f"   - Success: {result.get('success', False)}")
        print(f"   - Patient ID: {result.get('patient_id', 'N/A')}")
        print(f"   - Last Node: {result.get('last_node', 'N/A')}")

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
