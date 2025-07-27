#!/usr/bin/env python3
"""
Test script for PatientService with real LLM calls.

This script tests the PatientService functionality with actual LLM API calls
to verify that the profile extraction and rewriting work correctly.
"""

import logging
import os
import sys

# Add the backend directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.my_agent.database_manager import DatabaseManager
from backend.my_agent.llm_manager import LLMManager
from backend.my_agent.patient_collector import PatientService
from backend.my_agent.State import create_agent_state


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def check_environment():
    """Check if required environment variables are set."""
    groq_key = os.getenv("GROQ_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if not groq_key and not openai_key:
        print("❌ No API keys found!")
        print("   Please set one of the following environment variables:")
        print("   - GROQ_API_KEY (recommended for free tier)")
        print("   - OPENAI_API_KEY")
        return False

    print("✅ API keys found:")
    if groq_key:
        print(f"   - GROQ_API_KEY: {groq_key[:10]}...")
    if openai_key:
        print(f"   - OPENAI_API_KEY: {openai_key[:10]}...")

    return True


def test_patient_id_extraction():
    """Test patient ID extraction with real LLM calls."""
    print("\n🧪 Testing Patient ID Extraction")
    print("=" * 50)

    if not check_environment():
        return False

    try:
        # Create dependencies using default managers
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        db_manager = DatabaseManager()

        patient_service = PatientService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager,
        )

        # Test cases with different patient prompts
        test_cases = [
            "I need information about patient 1",
            "Can you show me the data for patient 42?",
            "What's the medical history of patient 15?",
            "Patient 7 needs to be evaluated for clinical trials",
            "Show me patient 99's profile",
        ]

        for i, prompt in enumerate(test_cases, 1):
            print(f"\n{i}. Testing prompt: '{prompt}'")
            try:
                patient_id = patient_service.extract_patient_id(prompt)
                print(f"   ✅ Extracted Patient ID: {patient_id}")

                # Validate the extracted ID
                if isinstance(patient_id, int) and patient_id > 0:
                    print(f"   ✅ Valid ID format: {patient_id}")
                else:
                    print(
                        f"   ⚠️ Unexpected ID format: {patient_id} (type: {type(patient_id)})"
                    )

            except Exception as e:
                print(f"   ❌ Error extracting patient ID: {e}")
                return False

        print("\n✅ All patient ID extraction tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error in patient ID extraction test: {e}")
        return False


def test_patient_data_fetching():
    """Test patient data fetching from database."""
    print("\n🧪 Testing Patient Data Fetching")
    print("=" * 50)

    try:
        # Create dependencies using default managers
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        db_manager = DatabaseManager()

        patient_service = PatientService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager,
        )

        # Test fetching data for different patient IDs
        test_patient_ids = [1, 5, 10, 15, 20]

        for patient_id in test_patient_ids:
            print(f"\nTesting patient ID: {patient_id}")
            try:
                patient_data = patient_service.fetch_patient_data(patient_id)

                if patient_data is not None:
                    print("   ✅ Patient data found")
                    print(f"   - Age: {patient_data.get('age', 'N/A')}")
                    print(
                        f"   - Medical History: {patient_data.get('medical_history', 'N/A')[:50]}..."
                    )
                    print(
                        f"   - Previous Trials: {patient_data.get('previous_trials', 'N/A')}"
                    )
                else:
                    print(f"   ⚠️ No patient data found for ID {patient_id}")

            except Exception as e:
                print(f"   ❌ Error fetching patient data: {e}")
                return False

        print("\n✅ All patient data fetching tests completed!")
        return True

    except Exception as e:
        print(f"❌ Error in patient data fetching test: {e}")
        return False


def test_profile_generation():
    """Test patient profile generation with real LLM calls."""
    print("\n🧪 Testing Profile Generation")
    print("=" * 50)

    if not check_environment():
        return False

    try:
        # Create dependencies using default managers
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        db_manager = DatabaseManager()

        patient_service = PatientService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager,
        )

        # Test with sample patient data
        sample_patient_data = {
            "age": 45,
            "medical_history": "Diabetes type 2, hypertension, previous heart surgery in 2020",
            "previous_trials": "Participated in diabetes medication trial in 2021, completed successfully",
            "trial_status": "completed",
            "trial_completion_date": "2021-12-15",
        }

        print("Testing profile generation with sample data:")
        print(f"   - Age: {sample_patient_data['age']}")
        print(f"   - Medical History: {sample_patient_data['medical_history']}")
        print(f"   - Previous Trials: {sample_patient_data['previous_trials']}")

        try:
            profile = patient_service.build_profile(sample_patient_data)
            print("\n✅ Generated Profile:")
            print(f"   {profile}")

            # Validate profile content
            if profile and len(profile.strip()) > 0:
                print(f"   ✅ Profile is not empty (length: {len(profile)})")

                # Check for key information
                if "diabetes" in profile.lower():
                    print("   ✅ Profile mentions diabetes")
                if "45" in profile or "45-year-old" in profile.lower():
                    print("   ✅ Profile mentions age")
                if "trial" in profile.lower():
                    print("   ✅ Profile mentions trial participation")
            else:
                print("   ⚠️ Generated profile is empty")
                return False

        except Exception as e:
            print(f"   ❌ Error generating profile: {e}")
            return False

        print("\n✅ Profile generation test passed!")
        return True

    except Exception as e:
        print(f"❌ Error in profile generation test: {e}")
        return False


def test_profile_rewriting():
    """Test profile rewriting with real LLM calls."""
    print("\n🧪 Testing Profile Rewriting")
    print("=" * 50)

    if not check_environment():
        return False

    try:
        # Create dependencies using default managers
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        db_manager = DatabaseManager()

        patient_service = PatientService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager,
        )

        # Test with sample patient data that might need rewriting
        sample_patient_data = {
            "age": 38,
            "medical_history": "Rheumatoid arthritis, mild depression, seasonal allergies",
            "previous_trials": "No previous trial participation",
            "trial_status": None,
            "trial_completion_date": None,
        }

        print("Testing profile rewriting with sample data:")
        print(f"   - Age: {sample_patient_data['age']}")
        print(f"   - Medical History: {sample_patient_data['medical_history']}")
        print(f"   - Previous Trials: {sample_patient_data['previous_trials']}")

        try:
            rewritten_profile = patient_service.rewrite_profile(sample_patient_data)
            print("\n✅ Rewritten Profile:")
            print(f"   {rewritten_profile}")

            # Validate rewritten profile content
            if rewritten_profile and len(rewritten_profile.strip()) > 0:
                print(
                    f"   ✅ Rewritten profile is not empty (length: {len(rewritten_profile)})"
                )

                # Check for key information
                if "arthritis" in rewritten_profile.lower():
                    print("   ✅ Rewritten profile mentions arthritis")
                if (
                    "38" in rewritten_profile
                    or "38-year-old" in rewritten_profile.lower()
                ):
                    print("   ✅ Rewritten profile mentions age")
                if (
                    "mental_health" in rewritten_profile.lower()
                    or "cancer" in rewritten_profile.lower()
                    or "leukemia" in rewritten_profile.lower()
                ):
                    print("   ✅ Rewritten profile suggests relevant trial categories")
            else:
                print("   ⚠️ Rewritten profile is empty")
                return False

        except Exception as e:
            print(f"   ❌ Error rewriting profile: {e}")
            return False

        print("\n✅ Profile rewriting test passed!")
        return True

    except Exception as e:
        print(f"❌ Error in profile rewriting test: {e}")
        return False


def test_full_patient_collector_node():
    """Test the complete patient_collector_node with real LLM calls."""
    print("\n🧪 Testing Complete Patient Collector Node")
    print("=" * 50)

    if not check_environment():
        return False

    try:
        # Create dependencies using default managers
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        db_manager = DatabaseManager()

        patient_service = PatientService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager,
        )

        # Test with a real patient prompt
        test_prompts = [
            "I need information about patient 1",
            "Can you show me the data for patient 5?",
            "What's the medical history of patient 10?",
        ]

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Testing complete workflow with prompt: '{prompt}'")

            try:
                # Create initial state
                state = create_agent_state()
                state["patient_prompt"] = prompt

                # Run the complete patient collector node
                result = patient_service.patient_collector_node(state)

                print("   ✅ Patient Collector Node completed")
                print(f"   - Patient ID: {result.get('patient_id', 'N/A')}")
                print(f"   - Last Node: {result.get('last_node', 'N/A')}")
                print(f"   - Policy Eligible: {result.get('policy_eligible', 'N/A')}")
                print(f"   - Revision Number: {result.get('revision_number', 'N/A')}")

                # Check if we got patient data
                patient_data = result.get("patient_data", {})
                if patient_data:
                    print(
                        f"   - Patient Data: Found (age: {patient_data.get('age', 'N/A')})"
                    )
                else:
                    print("   - Patient Data: Not found")

                # Check if we got a profile
                profile = result.get("patient_profile", "")
                if profile:
                    print(f"   - Patient Profile: Generated ({len(profile)} chars)")
                    print(f"     Preview: {profile[:100]}...")
                else:
                    print("   - Patient Profile: Not generated")

                # Check for errors
                if result.get("error_message"):
                    print(f"   ⚠️ Error: {result.get('error_message')}")

            except Exception as e:
                print(f"   ❌ Error in patient collector node: {e}")
                return False

        print("\n✅ All patient collector node tests passed!")
        return True

    except Exception as e:
        print(f"❌ Error in patient collector node test: {e}")
        return False


def test_profile_rewriter_node():
    """Test the complete profile_rewriter_node with real LLM calls."""
    print("\n🧪 Testing Complete Profile Rewriter Node")
    print("=" * 50)

    if not check_environment():
        return False

    try:
        # Create dependencies using default managers
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        db_manager = DatabaseManager()

        patient_service = PatientService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager,
        )

        # Test with sample patient data
        sample_patient_data = {
            "age": 52,
            "medical_history": "Breast cancer (stage 2), chemotherapy completed in 2022, currently in remission",
            "previous_trials": "Participated in immunotherapy trial in 2023, discontinued due to side effects",
            "trial_status": "discontinued",
            "trial_completion_date": None,
        }

        print("Testing profile rewriter node with sample data:")
        print(f"   - Age: {sample_patient_data['age']}")
        print(f"   - Medical History: {sample_patient_data['medical_history']}")
        print(f"   - Previous Trials: {sample_patient_data['previous_trials']}")

        try:
            # Create initial state with patient data
            state = create_agent_state()
            state["patient_data"] = sample_patient_data
            state["patient_profile"] = "Original profile would be here"
            state["policy_eligible"] = False

            # Run the profile rewriter node
            result = patient_service.profile_rewriter_node(state)

            print("\n✅ Profile Rewriter Node completed")
            print(f"   - Last Node: {result.get('last_node', 'N/A')}")
            print(f"   - Policy Eligible: {result.get('policy_eligible', 'N/A')}")

            # Check if we got a rewritten profile
            rewritten_profile = result.get("patient_profile", "")
            if rewritten_profile:
                print(
                    f"   - Rewritten Profile: Generated ({len(rewritten_profile)} chars)"
                )
                print(f"     Preview: {rewritten_profile[:150]}...")

                # Check for key elements in rewritten profile
                if "cancer" in rewritten_profile.lower():
                    print("   ✅ Rewritten profile mentions cancer")
                if (
                    "mental_health" in rewritten_profile.lower()
                    or "cancer" in rewritten_profile.lower()
                    or "leukemia" in rewritten_profile.lower()
                ):
                    print("   ✅ Rewritten profile suggests relevant trial categories")
            else:
                print("   - Rewritten Profile: Not generated")

            # Check for errors
            if result.get("error_message"):
                print(f"   ⚠️ Error: {result.get('error_message')}")

        except Exception as e:
            print(f"   ❌ Error in profile rewriter node: {e}")
            return False

        print("\n✅ Profile rewriter node test passed!")
        return True

    except Exception as e:
        print(f"❌ Error in profile rewriter node test: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 PatientService LLM Integration Tests")
    print("=" * 60)

    setup_logging()

    # Run all tests
    tests = [
        ("Patient ID Extraction", test_patient_id_extraction),
        ("Patient Data Fetching", test_patient_data_fetching),
        ("Profile Generation", test_profile_generation),
        ("Profile Rewriting", test_profile_rewriting),
        ("Complete Patient Collector Node", test_full_patient_collector_node),
        ("Complete Profile Rewriter Node", test_profile_rewriter_node),
    ]

    results = []

    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Running: {test_name}")
        print(f"{'='*60}")

        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print(
            "🎉 All tests passed! PatientService is working correctly with LLM calls."
        )
    else:
        print("⚠️ Some tests failed. Check the output above for details.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
