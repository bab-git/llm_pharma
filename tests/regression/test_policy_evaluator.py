#!/usr/bin/env python3
"""
Test script for the policy evaluator node functionality.

This script tests the policy evaluator node with sample data to ensure
it correctly evaluates patient eligibility against institutional policies.
"""

import os
import sys

from dotenv import find_dotenv, load_dotenv

# Add the parent directory to the path to import helper_functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from backend.my_agent.database_manager import DatabaseManager
from backend.my_agent.patient_collector import (
    create_agent_state,
    patient_collector_node,
)
from backend.my_agent.policy_service import policy_evaluator_node


def test_policy_evaluator():
    """Test the policy evaluator node with sample data."""

    print("🧪 Testing Policy Evaluator Node")
    print("=" * 50)

    # Load environment variables
    _ = load_dotenv(find_dotenv())

    # Check for required API keys
    if not os.getenv("GROQ_API_KEY"):
        print("❌ GROQ_API_KEY not found in environment variables")
        print("Please set your GROQ_API_KEY to run this test")
        return False

    try:
        # Create demo database
        print("📊 Creating demo patient database...")
        db_manager = DatabaseManager()
        db_manager.create_demo_patient_database()

        # Create initial state
        print("🔧 Creating initial agent state...")
        state = create_agent_state()
        state["patient_prompt"] = "I need information about patient 1"

        # Run patient collector to get patient profile
        print("👤 Running patient collector...")
        patient_result = patient_collector_node(state)
        state.update(patient_result)

        print(f"✅ Patient ID: {state['patient_id']}")
        print(f"✅ Patient Profile: {state['patient_profile'][:200]}...")

        # Create sample policy document for testing
        print("📋 Creating sample policy document...")
        sample_policy = Document(
            page_content="""#### Age Restriction Policy
            Patients must be between 18 and 75 years of age to participate in clinical trials.
            Patients who are under 18 or over 75 years of age are not eligible for participation.
            This policy ensures patient safety and compliance with regulatory requirements.""",
            metadata={
                "title": "Age Restriction Policy",
                "source": "institutional_policy",
            },
        )

        # Add policy to unchecked policies
        state["unchecked_policies"] = [sample_policy]

        # Run policy evaluator
        print("🔍 Running policy evaluator...")
        policy_result = policy_evaluator_node(state)

        # Display results
        print("\n📊 Policy Evaluation Results:")
        print(f"✅ Policy Eligible: {policy_result['policy_eligible']}")
        print(f"✅ Rejection Reason: {policy_result['rejection_reason']}")
        print(f"✅ Policy Questions: {policy_result['policy_qs']}")
        print(f"✅ Revision Number: {policy_result['revision_number']}")
        print(
            f"✅ Remaining Unchecked Policies: {len(policy_result['unchecked_policies'])}"
        )

        print("\n🎉 Policy evaluator test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during policy evaluator test: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_policy_vectorstore():
    """Test the policy vector store creation."""

    print("\n🧪 Testing Policy Vector Store")
    print("=" * 50)

    try:
        # Create policy vector store
        print("📚 Creating policy vector store...")
        db_manager = DatabaseManager()
        vectorstore = db_manager.create_policy_vectorstore()

        print("✅ Vector store created successfully")
        print(f"✅ Collection count: {vectorstore._collection.count()}")

        return True

    except Exception as e:
        print(f"❌ Error during vector store test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 Starting LLM Pharma Policy Evaluator Tests")
    print("=" * 60)

    # Test policy vector store
    vectorstore_success = test_policy_vectorstore()

    # Test policy evaluator
    evaluator_success = test_policy_evaluator()

    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"✅ Policy Vector Store: {'PASS' if vectorstore_success else 'FAIL'}")
    print(f"✅ Policy Evaluator: {'PASS' if evaluator_success else 'FAIL'}")

    if vectorstore_success and evaluator_success:
        print("\n🎉 All tests passed! Policy evaluator is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
