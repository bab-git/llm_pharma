"""
Test script for DatabaseManager

This script tests the basic functionality of the DatabaseManager class.
"""

import os
import tempfile
import shutil
from my_agent.database_manager import DatabaseManager

def test_database_manager():
    """Test the DatabaseManager functionality."""
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Testing DatabaseManager in temporary directory: {temp_dir}")
        
        # Initialize DatabaseManager with custom project root
        db_manager = DatabaseManager(project_root=temp_dir)
        
        # Test 1: Create demo patient database
        print("\n1. Testing demo patient database creation...")
        try:
            df = db_manager.create_demo_patient_database()
            print(f"✅ Created {len(df)} patients successfully")
            
            # Test 2: Get patient data
            print("\n2. Testing patient data retrieval...")
            patient_data = db_manager.get_patient_data(1)
            if patient_data:
                print(f"✅ Retrieved patient data: {patient_data['name']}, age {patient_data['age']}")
            else:
                print("❌ Failed to retrieve patient data")
                
        except Exception as e:
            print(f"❌ Error in patient database operations: {e}")
        
        # Test 3: Create trials dataset
        print("\n3. Testing trials dataset creation...")
        try:
            df_trials, csv_path = db_manager.create_trials_dataset(status="recruiting")
            print(f"✅ Created trials dataset with {len(df_trials)} trials")
            print(f"✅ Saved to: {csv_path}")
        except Exception as e:
            print(f"❌ Error in trials dataset creation: {e}")
        
        # Test 4: Disease mapping
        print("\n4. Testing disease mapping...")
        try:
            test_diseases = ["colorectal cancer", "diabetes", "hypertension"]
            mapped = db_manager.disease_map(test_diseases)
            print(f"✅ Mapped diseases {test_diseases} to categories: {mapped}")
        except Exception as e:
            print(f"❌ Error in disease mapping: {e}")
        
        # Test 5: Policy vector store (if policy file exists)
        print("\n5. Testing policy vector store creation...")
        try:
            # Check if policy file exists in the original project
            original_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            original_policy_path = os.path.join(original_project_root, "source_data", "instut_trials_policy.md")
            
            if os.path.exists(original_policy_path):
                # Copy policy file to temp directory
                temp_policy_path = os.path.join(temp_dir, "source_data", "instut_trials_policy.md")
                os.makedirs(os.path.dirname(temp_policy_path), exist_ok=True)
                shutil.copy2(original_policy_path, temp_policy_path)
                
                # Create policy vector store
                policy_store = db_manager.create_policy_vectorstore()
                print(f"✅ Created policy vector store with {policy_store._collection.count()} documents")
            else:
                print("⚠️ Policy file not found, skipping policy vector store test")
                
        except Exception as e:
            print(f"❌ Error in policy vector store creation: {e}")
        
        # Test 6: Trial vector store (if trials data exists)
        print("\n6. Testing trial vector store creation...")
        try:
            # Check if trials data exists
            if os.path.exists(csv_path):
                trial_store = db_manager.create_trial_vectorstore()
                if trial_store:
                    print(f"✅ Created trial vector store with {trial_store._collection.count()} trials")
                else:
                    print("⚠️ No trials to add to vector store")
            else:
                print("⚠️ Trials data not found, skipping trial vector store test")
                
        except Exception as e:
            print(f"❌ Error in trial vector store creation: {e}")
        
        print("\n🎉 DatabaseManager test completed!")

if __name__ == "__main__":
    test_database_manager() 