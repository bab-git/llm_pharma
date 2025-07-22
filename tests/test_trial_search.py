#!/usr/bin/env python3
"""
Test script for trial_search_node function

This script tests the trial_search_node function with a sample patient profile
to verify that it correctly retrieves relevant clinical trials.

Usage:
    python test_trial_search.py
"""

import os
import sys
from dotenv import load_dotenv, find_dotenv

# Load environment variables
_ = load_dotenv(find_dotenv())

# Add the backend directory to the path so we can import helper_functions
# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.my_agent.trial_service import trial_search_node
from backend.my_agent.patient_collector import create_agent_state, AgentState

def test_trial_search_node():
    """
    Test the trial_search_node function with a sample patient profile.
    """
    print("🧪 Testing trial_search_node function...")
    print("=" * 50)
    
    # Create initial agent state
    state = create_agent_state()
    
    # Set up the sample patient profile
    sample_patient_profile = """The patient is a 48-year-old individual with a medical history of solid tumors. They have no prior participation in clinical trials. This patient may be eligible for trials related to cancer treatment, such as immunotherapy trials for solid tumors or clinical trials evaluating new chemotherapy agents.

Possible related medical trials:
- A phase III trial evaluating the efficacy of a new immunotherapy agent in treating solid tumors.
- A phase II trial assessing the safety and efficacy of a novel chemotherapy regimen for patients with solid tumors.
"""
    
    # Update state with the patient profile
    state.update({
        'patient_profile': sample_patient_profile,
        'policy_eligible': True,  # Assume patient passed policy evaluation
        'trial_searches': 0
    })
    
    print(f"📋 Sample Patient Profile:")
    print(f"{sample_patient_profile}")
    print("=" * 50)
    
    try:
        # Call the trial_search_node function
        print("🔍 Executing trial_search_node...")
        result = trial_search_node(state)
        
        # Display results
        print("\n✅ Trial Search Results:")
        print(f"Last Node: {result.get('last_node', 'N/A')}")
        print(f"Trial Searches: {result.get('trial_searches', 0)}")
        print(f"Policy Eligible: {result.get('policy_eligible', False)}")
        print(f"Number of Trials Retrieved: {len(result.get('trials', []))}")
        
        # Display trial details
        trials = result.get('trials', [])
        if trials:
            print(f"\n📊 Retrieved Trials Details:")
            for i, trial in enumerate(trials[:3], 1):  # Show first 3 trials
                print(f"\n--- Trial {i} ---")
                print(f"Metadata: {trial.metadata}")
                print(f"Content Preview: {trial.page_content[:200]}...")
        else:
            print("\n❌ No trials were retrieved.")
            
        print("\n" + "=" * 50)
        print("✅ Test completed successfully!")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error during trial search test: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return None

def test_trial_search_with_empty_profile():
    """
    Test the trial_search_node function with an empty patient profile.
    """
    print("\n🧪 Testing trial_search_node with empty profile...")
    print("=" * 50)
    
    # Create initial agent state
    state = create_agent_state()
    
    # Update state with empty patient profile
    state.update({
        'patient_profile': "",
        'policy_eligible': True,
        'trial_searches': 0
    })
    
    try:
        # Call the trial_search_node function
        result = trial_search_node(state)
        
        print(f"Last Node: {result.get('last_node', 'N/A')}")
        print(f"Trial Searches: {result.get('trial_searches', 0)}")
        print(f"Number of Trials Retrieved: {len(result.get('trials', []))}")
        
        if len(result.get('trials', [])) == 0:
            print("✅ Correctly handled empty profile - no trials retrieved")
        else:
            print("⚠️ Unexpected: trials retrieved with empty profile")
            
        return result
        
    except Exception as e:
        print(f"❌ Error during empty profile test: {e}")
        return None

def main():
    """
    Main function to run all tests.
    """
    print("🚀 Starting Trial Search Node Tests")
    print("=" * 60)
    
    # Check if required environment variables are set
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        print("⚠️ Warning: GROQ_API_KEY not found in environment variables")
        print("   The test may fail if the API key is required for trial search")
    
    # Test 1: Normal trial search
    result1 = test_trial_search_node()
    
    # Test 2: Empty profile test
    # result2 = test_trial_search_with_empty_profile()
    
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print(f"Test 1 (Normal Profile): {'✅ PASSED' if result1 else '❌ FAILED'}")
    # print(f"Test 2 (Empty Profile): {'✅ PASSED' if result2 else '❌ FAILED'}")
    
    if result1: # and result2:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main() 