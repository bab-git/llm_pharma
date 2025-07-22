#!/usr/bin/env python3
"""
Test script for the policy service module functionality.

This script tests the PolicyService class and related functions to ensure
they work correctly after being moved from helper_functions.py.
"""

import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
backend_path = project_root / "backend"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_path))


def test_policy_service():
    """Test the policy service module functionality."""

    print("🧪 Testing Policy Service Module")
    print("=" * 50)

    # Load environment variables
    _ = load_dotenv(find_dotenv())

    # Check for required API keys
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment variables")
        print("Please set your GROQ_API_KEY to run this test")
        return False

    try:
        # Test 1: Import PolicyService
        print("\n1. Testing PolicyService imports...")
        from backend.my_agent.policy_service import (  # get_default_policy_service,; policy_tools,
            PolicyEligibility,
            PolicyService,
            policy_evaluator_node,
            policy_search_node,
        )

        print("✅ Successfully imported all PolicyService components")

        # Test 2: Create PolicyService instance
        print("\n2. Testing PolicyService initialization...")
        policy_service = PolicyService()
        print("✅ PolicyService instance created")
        print(f"   - LLM Manager: {type(policy_service.llm_manager)}")
        print(f"   - LLM Manager Tool: {type(policy_service.llm_manager_tool)}")
        print(f"   - Database Manager: {type(policy_service.db_manager)}")

        # Test 3: Test standalone functions
        print("\n3. Testing standalone policy functions...")
        from my_agent.database_manager import DatabaseManager
        from backend.my_agent.State import create_agent_state

        # Create demo database first
        db_manager = DatabaseManager()
        db_manager.create_demo_patient_database()

        # Create sample agent state
        state = create_agent_state()
        state[
            "patient_profile"
        ] = """
        Patient is a 45-year-old male with hypertension and diabetes. 
        He has participated in one clinical trial in the past 6 months and is currently 
        taking medication for his conditions. He has a BMI of 28 and lives locally.
        No history of cancer, tuberculosis, hepatitis, or HIV.
        """

        # Test policy search
        print("\n   a. Testing policy_search_node...")
        search_result = policy_search_node(state)
        print("   ✅ Policy search completed")
        print(f"      - Last node: {search_result['last_node']}")
        print(f"      - Policies found: {len(search_result['policies'])}")
        print(f"      - Unchecked policies: {len(search_result['unchecked_policies'])}")

        # Test policy evaluator (if policies were found)
        if search_result["unchecked_policies"]:
            print("\n   b. Testing policy_evaluator_node...")
            state.update(search_result)
            eval_result = policy_evaluator_node(state)
            print("   ✅ Policy evaluation completed")
            print(f"      - Last node: {eval_result['last_node']}")
            print(f"      - Policy eligible: {eval_result['policy_eligible']}")
            print(f"      - Rejection reason: {eval_result['rejection_reason']}")
            print(f"      - Revision number: {eval_result['revision_number']}")
        else:
            print("\n   b. ⚠️ No policies found for evaluation")

        # Test 4: Test PolicyService class methods directly
        print("\n4. Testing PolicyService class methods...")

        # Reset state for class method testing
        state = create_agent_state()
        state[
            "patient_profile"
        ] = """
        Patient is a 35-year-old female with breast cancer. 
        She completed chemotherapy 8 months ago and is currently in remission.
        No other significant medical history. BMI of 24.
        """

        # Test instance methods
        print("\n   a. Testing PolicyService.policy_search_node...")
        search_result_class = policy_service.policy_search_node(state)
        print("   ✅ Class method policy search completed")
        print(f"      - Policies found: {len(search_result_class['policies'])}")

        if search_result_class["unchecked_policies"]:
            print("\n   b. Testing PolicyService.policy_evaluator_node...")
            state.update(search_result_class)
            eval_result_class = policy_service.policy_evaluator_node(state)
            print("   ✅ Class method policy evaluation completed")
            print(f"      - Policy eligible: {eval_result_class['policy_eligible']}")
        else:
            print("\n   b. ⚠️ No policies found for class method evaluation")

        # Test 5: Test PolicyEligibility model
        print("\n5. Testing PolicyEligibility model...")
        eligibility = PolicyEligibility(eligibility="yes", reason="N/A")
        print("✅ PolicyEligibility model works")
        print(f"   - Eligibility: {eligibility.eligibility}")
        print(f"   - Reason: {eligibility.reason}")

        print("\n" + "=" * 50)
        print("✅ All PolicyService tests passed!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_policy_service()
    if not success:
        sys.exit(1)
    print("\n🎉 Policy Service Module Test Completed Successfully!")
