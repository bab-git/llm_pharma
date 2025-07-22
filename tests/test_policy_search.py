#!/usr/bin/env python3
"""
Test script for the policy search node functionality.

This script tests the policy_search_node function and related vector store creation.
"""

import os
import sys
from pathlib import Path

# Add project paths for imports
project_root = Path(__file__).parent.parent
backend_path = project_root / "backend"
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(backend_path))

def test_policy_search():
    """Test the policy search functionality."""
    try:
        from my_agent.patient_collector import create_agent_state
        from my_agent.policy_service import policy_search_node
        from my_agent.database_manager import DatabaseManager
        
        print("🧪 Testing Policy Search Node")
        print("=" * 50)
        
        # Test 1: Create vector stores
        print("\n1. Testing vector store creation...")
        
        # Create policy vector store
        db_manager = DatabaseManager()
        policy_vs = db_manager.create_policy_vectorstore()
        print(f"✅ Policy vector store created with {policy_vs._collection.count()} documents")
               
        # Test 2: Test policy search with sample patient profile
        print("\n2. Testing policy search with sample patient profile...")
        
        # Create sample agent state
        state = create_agent_state()
        state['patient_profile'] = """
        Patient is a 45-year-old male with hypertension and diabetes. 
        He has participated in one clinical trial in the past 6 months and is currently 
        taking medication for his conditions. He has a BMI of 28 and lives locally.
        """
        
        # Run policy search
        result = policy_search_node(state)
        
        print(f"✅ Policy search completed")
        print(f"📋 Retrieved {len(result['policies'])} policy sections")
        print(f"📝 Unchecked policies: {len(result['unchecked_policies'])}")
        
        # Display first policy section
        if result['policies']:
            print(f"\n📄 First policy section:")
            print(f"Title: {result['policies'][0].metadata.get('title', 'N/A')}")
            print(f"Content preview: {result['policies'][0].page_content[:200]}...")
        
        print("\n✅ All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🏥 LLM Pharma - Policy Search Test")
    print("=" * 50)
    
    # Run tests
    policy_test = test_policy_search()
    
    if policy_test:
        print("\n🎉 All tests passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1) 