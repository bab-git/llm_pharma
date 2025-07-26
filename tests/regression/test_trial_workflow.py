#!/usr/bin/env python3
"""
Test script for trial-focused workflow (patient collection -> trial search -> trial evaluation).
This test skips policy search and evaluation steps.

Usage:
    python tests/test_trial_workflow.py --patient-id 41
    python tests/test_trial_workflow.py -p 41
"""

import os
import sys
import argparse
from dotenv import load_dotenv, find_dotenv

# Load environment variables from a .env file if present
_ = load_dotenv(find_dotenv())


# Add the project root directory to the Python path
# The test is in tests/regression/, so we need to go up 2 levels to reach the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)


def test_trial_workflow(patient_id: int = 41):
    """Test the trial-focused workflow: patient collection -> trial search -> trial evaluation."""
    print("🧪 Testing Trial-Focused Workflow")
    print("=" * 60)
    print(f"🎯 Testing with Patient ID: {patient_id}")

    try:
        # Check for required API keys
        groq_key = os.getenv("GROQ_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not groq_key and not openai_key:
            print("❌ No API keys found in environment variables")
            print("Please set one of the following to run this test:")
            print("   - GROQ_API_KEY (recommended for free tier)")
            print("   - OPENAI_API_KEY")
            print("\nExample:")
            print("   export GROQ_API_KEY=your_groq_api_key_here")
            return False
        
        if groq_key:
            print(f"✅ Using GROQ_API_KEY: {groq_key[:10]}...")
        if openai_key:
            print(f"✅ Using OPENAI_API_KEY: {openai_key[:10]}...")

        # Import all necessary components
        from backend.my_agent.State import create_agent_state
        from backend.my_agent.trial_service import TrialService
        from backend.my_agent.llm_manager import LLMManager
        from backend.my_agent.database_manager import DatabaseManager

        print("✅ All imports successful")

        # Create required dependencies for TrialService
        llm_manager, llm_manager_tool = LLMManager.get_default_managers()
        db_manager = DatabaseManager()
        
        # Create TrialService instance with required dependencies
        trial_service = TrialService(
            llm_manager=llm_manager,
            llm_manager_tool=llm_manager_tool,
            db_manager=db_manager
        )

        # Test 1: Patient collection for specified patient
        print(f"\n1. Testing patient collection for patient {patient_id}...")
        state = create_agent_state()
        state["patient_prompt"] = f"I need information about patient {patient_id}"

        # Create PatientService with required dependencies
        try:
            from backend.my_agent.patient_collector import PatientService
            print("   Creating PatientService with dependencies...")
            patient_service = PatientService(
                llm_manager=llm_manager,
                llm_manager_tool=llm_manager_tool,
                db_manager=db_manager
            )
            result = patient_service.patient_collector_node(state)
            print("   ✅ PatientService approach successful")
        except Exception as e:
            print(f"   ❌ PatientService approach failed: {e}")
            raise e
        
#         result['patient_profile'] = """The patient is a 43-year-old individual with a medical history of generalized anxiety disorder. 
# They previously participated in the clinical trial identified by NCT03081690 but withdrew on February 28, 2025. 
# Given their medical history, relevant clinical trials could include studies focused on anxiety disorders, mental health interventions, or related psychiatric conditions."""
        
        print("✅ Patient collection completed")
        print(f"   - Patient ID: {result.get('patient_id', 'N/A')}")
        print(f"   - Patient Profile: {result.get('patient_profile', 'N/A')}")
        print(f"   - Last Node: {result.get('last_node', 'N/A')}")

        # Test 2: Trial search (skip policy steps)
        print("\n2. Testing trial search...")
        state.update(result)
        
        print("✅ ======= all state keys:")
        for key, value in state.items():
            print(f"   - {key}: {value}")

        # Skip policy search and evaluation - go directly to trial search
        trial_result = trial_service.trial_search_node(state)
        print("✅ Trial search completed")
        print(f"   - Trials found: {len(trial_result.get('trials', []))}")
        print(f"   - Last Node: {trial_result.get('last_node', 'N/A')}")
        
        # Show all found trials with details
        if trial_result.get('trials'):
            print("\n   All Found Trials:")
            for i, trial in enumerate(trial_result.get('trials', []), 1):
                # Handle both Document objects and dictionaries
                if hasattr(trial, 'metadata'):
                    # Document object
                    nctid = trial.metadata.get('nctid', 'N/A')
                    diseases = trial.metadata.get('diseases', 'N/A')
                    content = trial.page_content[:200]
                else:
                    # Dictionary - check if it's a serialized Document
                    if 'metadata' in trial:
                        nctid = trial['metadata'].get('nctid', 'N/A')
                        diseases = trial['metadata'].get('diseases', 'N/A')
                        content = trial.get('page_content', 'N/A')[:200] if trial.get('page_content') else 'N/A'
                    else:
                        nctid = trial.get('nctid', 'N/A')
                        diseases = trial.get('diseases', 'N/A')
                        content = trial.get('content', 'N/A')[:200] if trial.get('content') else 'N/A'
                
                print(f"   {i}. NCT ID: {nctid}")
                print(f"      Diseases: {diseases}")
                print(f"      Content: {content}...")
                print()

        # Test 3: Trial evaluation
        print("\n3. Testing trial evaluation...")
        state.update(trial_result)
        if state.get("trials"):
            grade_result = trial_service.grade_trials_node(state)
            print("✅ Trial evaluation completed")
            print(f"   - Relevant trials: {len(grade_result.get('relevant_trials', []))}")
            print(f"   - Trial found: {grade_result.get('trial_found', 'N/A')}")
            print(f"   - Last Node: {grade_result.get('last_node', 'N/A')}")
            
            # Count trials by relevance score
            yes_count = 0
            no_count = 0
            
            # Show detailed relevant trial information
            if grade_result.get('relevant_trials'):
                print("\n   Relevant Trials (Detailed):")
                for i, trial in enumerate(grade_result.get('relevant_trials', []), 1):
                    relevance_score = trial.get('relevance_score', 'N/A')
                    
                    # Count by relevance score
                    if relevance_score.lower() == 'yes':
                        yes_count += 1
                    elif relevance_score.lower() == 'no':
                        no_count += 1
                    
                    print(f"   {i}. NCT ID: {trial.get('nctid', 'N/A')}")
                    print(f"      Relevance Score: {relevance_score}")
                    print(f"      Explanation: {trial.get('explanation', 'N/A')[:150]}...")
                    
                    # Handle None values for further_information
                    further_info = trial.get('further_information')
                    if further_info:
                        print(f"      Further Information: {further_info[:100]}...")
                    else:
                        print(f"      Further Information: None")
                    print()
                
                # Show relevance score summary
                print(f"   📊 Relevance Score Summary:")
                print(f"      - Trials graded as 'Yes': {yes_count}")
                print(f"      - Trials graded as 'No': {no_count}")
                print(f"      - Total trials evaluated: {len(grade_result.get('relevant_trials', []))}")
            else:
                print("   ⚠️ No trials were found relevant for this patient")
        else:
            print("⚠️ No trials to evaluate")

        print("\n" + "=" * 60)
        print("✅ Trial-focused workflow test completed successfully!")
        print("\nSummary:")
        print(f"   - Patient ID: {result.get('patient_id', 'N/A')}")
        print(f"   - Trials found: {len(trial_result.get('trials', []))}")
        print(f"   - Relevant trials: {len(grade_result.get('relevant_trials', [])) if 'grade_result' in locals() else 0}")
        print(f"   - Trials graded as 'Yes': {yes_count if 'yes_count' in locals() else 0}")
        print(f"   - Trials graded as 'No': {no_count if 'no_count' in locals() else 0}")
        print(f"   - Trial found: {grade_result.get('trial_found', 'N/A') if 'grade_result' in locals() else 'N/A'}")

        return True

    except Exception as e:
        print(f"❌ Error in trial workflow test: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test trial-focused workflow with specified patient ID",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_trial_workflow.py --patient-id 41
  python tests/test_trial_workflow.py -p 41
  python tests/test_trial_workflow.py  # Uses default patient ID 41
        """
    )
    parser.add_argument(
        "--patient-id", "-p",
        type=int,
        default=41,
        help="Patient ID to test (default: 41)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"Starting trial-focused workflow testing for Patient ID: {args.patient_id}...")
    
    # Run main test with specified patient ID
    success1 = test_trial_workflow(patient_id=args.patient_id)
    
    # Run edge case test
    # success2 = test_trial_workflow_with_empty_profile()
    
    if success1:
        print("\n🎉 All trial workflow tests passed!")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1) 