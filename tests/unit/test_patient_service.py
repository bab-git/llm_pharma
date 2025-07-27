"""
Test for the refactored PatientService class.
"""

from unittest.mock import Mock, patch

import pytest

from backend.my_agent.patient_collector import PatientService
from backend.my_agent.State import create_agent_state


class TestPatientService:
    """Test cases for PatientService class."""

    def test_patient_service_initialization(self, mock_llm, mock_database_manager):
        """Test that PatientService initializes correctly."""
        service = PatientService(
            llm_manager=mock_llm,
            llm_manager_tool=mock_llm,
            db_manager=mock_database_manager,
        )

        # Check that all required attributes are present
        assert hasattr(service, "logger")
        assert hasattr(service, "llm_manager")
        assert hasattr(service, "llm_manager_tool")
        assert hasattr(service, "db_manager")
        assert hasattr(service, "profile_chain")
        assert hasattr(service, "profile_rewrite_chain")

    def test_extract_patient_id(self, mock_llm, mock_database_manager):
        """Test patient ID extraction."""
        service = PatientService(
            llm_manager=mock_llm,
            llm_manager_tool=mock_llm,
            db_manager=mock_database_manager,
        )

        # Mock the LLM response
        mock_response = Mock()
        mock_response.patient_id = 42

        with patch.object(
            service.llm_manager_tool, "invoke_with_fallback"
        ) as mock_invoke:
            mock_invoke.return_value = mock_response

            result = service.extract_patient_id("I need information about patient 42")

            assert result == 42
            mock_invoke.assert_called_once()

    def test_fetch_patient_data(self, mock_llm, mock_database_manager):
        """Test patient data fetching."""
        service = PatientService(
            llm_manager=mock_llm,
            llm_manager_tool=mock_llm,
            db_manager=mock_database_manager,
        )

        # Mock patient data (only non-PII columns)
        mock_patient_data = {
            "age": 35,
            "medical_history": "diabetes, hypertension",
            "previous_trials": 2,
            "trial_status": "completed",
            "trial_completion_date": "2023-06-15",
        }

        with patch.object(service.db_manager, "get_patient_data") as mock_get:
            mock_get.return_value = mock_patient_data

            result = service.fetch_patient_data(42)

            # Check that only non-PII data is returned
            assert result["age"] == 35
            assert result["medical_history"] == "diabetes, hypertension"
            assert result["previous_trials"] == 2
            assert result["trial_status"] == "completed"
            assert result["trial_completion_date"] == "2023-06-15"
            # Verify no PII fields are present
            assert "patient_id" not in result
            assert "name" not in result

    def test_build_profile(self, mock_llm, mock_database_manager):
        """Test profile building."""
        service = PatientService(
            llm_manager=mock_llm,
            llm_manager_tool=mock_llm,
            db_manager=mock_database_manager,
        )

        patient_data = {"age": 35, "disease": "diabetes"}
        expected_profile = "Patient is a 35-year-old with diabetes."

        with patch.object(service.llm_manager, "invoke_with_fallback") as mock_invoke:
            mock_invoke.return_value = expected_profile

            result = service.build_profile(patient_data)

            assert result == expected_profile
            mock_invoke.assert_called_once()

    def test_patient_collector_node(self, mock_llm, mock_database_manager):
        """Test the patient collector node method."""
        service = PatientService(
            llm_manager=mock_llm,
            llm_manager_tool=mock_llm,
            db_manager=mock_database_manager,
        )

        state = create_agent_state()
        state["patient_prompt"] = "I need information about patient 1"

        # Mock the service methods
        with patch.object(service, "extract_patient_id") as mock_extract:
            with patch.object(service, "fetch_patient_data") as mock_fetch:
                with patch.object(service, "build_profile") as mock_build:
                    mock_extract.return_value = 1
                    mock_fetch.return_value = {"age": 35}
                    mock_build.return_value = "Patient profile"

                    result = service.patient_collector_node(state)

                    assert result["patient_id"] == 1
                    assert result["patient_profile"] == "Patient profile"
                    assert result["last_node"] == "patient_collector"

    def test_profile_rewriter_node(self, mock_llm, mock_database_manager):
        """Test the profile rewriter node method."""
        service = PatientService(
            llm_manager=mock_llm,
            llm_manager_tool=mock_llm,
            db_manager=mock_database_manager,
        )

        state = create_agent_state()
        state["patient_data"] = {"age": 35, "disease": "diabetes"}

        # Mock the service methods
        with patch.object(service, "rewrite_profile") as mock_rewrite:
            mock_rewrite.return_value = "Rewritten profile"

            result = service.profile_rewriter_node(state)

            assert result["patient_profile"] == "Rewritten profile"
            assert result["last_node"] == "profile_rewriter"


if __name__ == "__main__":
    pytest.main([__file__])
