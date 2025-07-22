#!/usr/bin/env python3
"""
Test script for the patient collector node functionality.
Demonstrates how to use the completed patient_collector_node with demo data.
"""

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the helper functions from backend
from my_agent.patient_collector import (
    initialize_patient_collector_system, 
    patient_collector_node, 
    create_agent_state,
    PatientCollectorConfig
)
from backend.my_agent.database_manager import DatabaseManager


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        db_path = tmp_file.name
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def patient_collector_config(temp_db_path):
    """Create a patient collector configuration for testing."""
    # Create demo database in temp location
    db_manager = DatabaseManager()
    db_manager.create_demo_patient_database(temp_db_path)
    
    # Create config with temp database
    config = PatientCollectorConfig(use_free_model=True, db_path=temp_db_path)
    return config


def test_create_demo_patient_database(temp_db_path):
    """Test that demo patient database is created correctly."""
    print("Testing demo database creation...")
    
    # Create the database
    db_manager = DatabaseManager()
    df = db_manager.create_demo_patient_database(temp_db_path)
    
    # Verify database exists
    assert os.path.exists(temp_db_path), "Database file should be created"
    
    # Test that patients were added
    for patient_id in range(1, 6):
        patient_data = db_manager.get_patient_data(patient_id, temp_db_path)
        assert patient_data is not None, f"Patient {patient_id} should exist"
        assert 'name' in patient_data, f"Patient {patient_id} should have name"
        assert 'age' in patient_data, f"Patient {patient_id} should have age"
        assert 'medical_history' in patient_data, f"Patient {patient_id} should have medical history"
    
    print("✓ Demo database creation test passed")


def test_get_patient_data(patient_collector_config):
    """Test patient data retrieval."""
    print("Testing patient data retrieval...")
    
    db_manager = DatabaseManager()
    
    # Test existing patient
    patient_data = db_manager.get_patient_data(1, patient_collector_config.db_path)
    assert patient_data is not None, "Should retrieve existing patient"
    assert 'name' in patient_data, "Should have name"
    assert 'age' in patient_data, "Should have age"
    
    # Test non-existing patient
    patient_data = db_manager.get_patient_data(99, patient_collector_config.db_path)
    assert patient_data is None, "Should return None for non-existing patient"
    
    print("✓ Patient data retrieval test passed")


@patch('backend.helper_functions.ChatGroq')
def test_patient_collector_node_with_mock_llm(mock_groq, patient_collector_config):
    """Test patient collector node with mocked LLM."""
    print("Testing patient collector node with mock LLM...")
    
    # Mock the LLM response
    mock_model = MagicMock()
    mock_response = MagicMock()
    mock_response.patient_id = 2
    mock_model.with_structured_output.return_value.invoke.return_value = mock_response
    
    # Mock the chain_profile
    mock_chain = MagicMock()
    mock_chain.invoke.return_value = "32-year-old female with breast cancer stage II, completed chemotherapy. Suitable for oncology trials."
    
    # Set up the mock to return our mock model
    mock_groq.return_value = mock_model
    
    # Create test state
    state = create_agent_state()
    state['patient_prompt'] = "I need information about patient 2"
    
    # Run patient collector
    result = patient_collector_node(state)
    
    # Verify results
    assert result['patient_id'] == 2, "Should extract correct patient ID"
    assert result['last_node'] == "patient_collector", "Should set correct last node"
    assert result['revision_number'] == 1, "Should increment revision number"
    assert 'patient_data' in result, "Should have patient_data key"
    assert 'patient_profile' in result, "Should have patient_profile key"
    assert 'policy_eligible' in result, "Should have policy_eligible key"
    
    print("✓ Patient collector node test passed")


def test_patient_collector_node_integration(patient_collector_config):
    """Integration test for patient collector node."""
    print("Testing patient collector node integration...")
    
    # Test with different patient prompts
    test_cases = [
        ("I need information about patient 1", 1),
        ("Can you look up patient ID 2?", 2),
        ("Please find patient number 3 for trial screening", 3),
        ("Get details for patient 4", 4),
        ("Screen patient 5", 5),
    ]
    
    for prompt, expected_id in test_cases:
        print(f"  Testing: '{prompt}' -> Expected ID: {expected_id}")
        
        # Create state
        state = create_agent_state()
        state['patient_prompt'] = prompt
        
        try:
            # Run patient collector
            result = patient_collector_node(state)
            
            # Verify patient ID extraction
            assert result['patient_id'] == expected_id, f"Should extract ID {expected_id} from '{prompt}'"
            
            # Verify required keys exist
            assert 'patient_data' in result, f"Should have patient_data key for patient {expected_id}"
            assert 'patient_profile' in result, f"Should have patient_profile key for patient {expected_id}"
            assert 'policy_eligible' in result, f"Should have policy_eligible key for patient {expected_id}"
            
            # Verify patient data exists (for patients 1-5)
            if expected_id <= 5:
                assert result['patient_data'] is not None, f"Should have data for patient {expected_id}"
                assert len(result['patient_profile']) > 0, f"Profile should not be empty for patient {expected_id}"
            
            print(f"    ✓ Passed: ID {result['patient_id']}, Profile length: {len(result.get('patient_profile', ''))}")
            
        except Exception as e:
            print(f"    ✗ Failed: {e}")
            raise
    
    print("✓ Patient collector integration test passed")


def test_patient_collector_with_non_existent_patient(patient_collector_config):
    """Test handling of non-existent patients."""
    print("Testing patient collector with non-existent patient...")
    
    # Create state with non-existent patient
    state = create_agent_state()
    state['patient_prompt'] = "Get details for patient 99"
    
    try:
        result = patient_collector_node(state)
        
        # Should still extract the ID
        assert result['patient_id'] == 99, "Should extract ID even for non-existent patient"
        
        # Should have empty patient data
        assert result['patient_data'] == {}, "Should have empty data for non-existent patient"
        
        # Should have empty profile
        assert result['patient_profile'] == "", "Should have empty profile for non-existent patient"
        
        # Should have policy_eligible key
        assert 'policy_eligible' in result, "Should have policy_eligible key"
        
        print("✓ Non-existent patient handling test passed")
        
    except Exception as e:
        print(f"✗ Failed: {e}")
        raise


def test_initialize_patient_collector_system(temp_db_path):
    """Test the initialization function."""
    print("Testing patient collector system initialization...")
    
    # Test initialization with new database
    config = initialize_patient_collector_system(
        use_free_model=True,
        db_path=temp_db_path,
        force_recreate_db=True
    )
    
    # Verify config
    assert isinstance(config, PatientCollectorConfig), "Should return PatientCollectorConfig"
    assert config.db_path == temp_db_path, "Should use provided database path"
    assert config.use_free_model is True, "Should use free model setting"
    
    # Verify database was created
    assert os.path.exists(temp_db_path), "Database should be created"
    
    print("✓ System initialization test passed")


def run_all_tests():
    """Run all patient collector tests."""
    print("🏥 LLM Pharma - Patient Collector Tests")
    print("=" * 50)
    
    # Create temp database for testing
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_file:
        temp_db_path = tmp_file.name
    
    try:
        # Run tests
        test_create_demo_patient_database(temp_db_path)
        test_get_patient_data(PatientCollectorConfig(use_free_model=True, db_path=temp_db_path))
        test_initialize_patient_collector_system(temp_db_path)
        
        # Integration test - these now don't need config parameter
        config = PatientCollectorConfig(use_free_model=True, db_path=temp_db_path)
        create_demo_patient_database(temp_db_path)
        test_patient_collector_node_integration(config)
        test_patient_collector_with_non_existent_patient(config)
        
        print("\n🎉 All tests passed! Patient collector is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        try:
            os.unlink(temp_db_path)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    run_all_tests() 