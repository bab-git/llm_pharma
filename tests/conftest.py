"""
Shared test fixtures for Clinical Trial Management System tests.

This module provides common fixtures and test utilities used across
all test modules.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def test_config():
    """Create a test configuration object."""
    config = {
        "app": {
            "name": "Test LLM Pharma",
            "version": "0.1.0",
            "debug": True,
            "log_level": "DEBUG",
        },
        "environment": {
            "use_dotenv": False,
            "langchain_tracing": False,
            "langchain_project": "Test Project",
        },
        "paths": {
            "data_dir": "test_data",
            "source_data_dir": "test_source_data",
            "database_file": "test_patients.db",
            "chroma_db_path": "test_chroma_db",
            "trials_data": "test_trials.csv",
            "policy_file": "test_policy.md",
        },
        "collections": {
            "policy_collection": "test-policy-chroma",
            "trial_collection": "test-trial-chroma",
        },
        "model": {
            "provider": "openai",
            "model_id": "gpt-3.5-turbo",
            "agent_model_id": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 1024,
            "api_settings": {"timeout": 30, "max_retries": 2, "retry_delay": 1},
            "model_params": {
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
            },
            "embeddings": {
                "provider": "nomic",
                "model": "nomic-embed-text-v1.5",
                "inference_mode": "local",
            },
        },
        "database": {
            "type": "sqlite",
            "connection": {"database": "test_patients.db", "timeout": 30},
        },
        "vectorstore": {"type": "chroma", "persistent": True, "path": "test_chroma_db"},
        "retrieval": {
            "policy_retrieval": {"type": "vector", "k": 3, "search_type": "similarity"},
            "trial_retrieval": {
                "type": "self_query",
                "k": 4,
                "search_type": "similarity",
                "enable_limit": True,
                "metadata_fields": [
                    {
                        "name": "disease_category",
                        "description": "Disease category",
                        "type": "string",
                    }
                ],
                "document_content_description": "Test description",
            },
        },
        "agent": {
            "workflow": {
                "max_revisions": 5,
                "max_trial_searches": 2,
                "interrupt_points": [],
            },
            "memory": {"type": "sqlite", "connection_string": ":memory:"},
            "grading": {"hallucination_check": True, "relevance_threshold": 0.7},
        },
        "frontend": {
            "server": {
                "host": "127.0.0.1",
                "port": 7999,
                "share": False,
                "debug": True,
            },
            "interface": {
                "title": "Test Clinical Trial System",
                "description": "Test interface",
            },
            "components": {
                "patient_input": {
                    "label": "Test Patient Input",
                    "placeholder": "Test placeholder",
                    "lines": 2,
                },
                "output_display": {"label": "Test Output", "lines": 5},
                "status_display": {"label": "Test Status", "lines": 3},
            },
        },
    }
    return OmegaConf.create(config)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_llm():
    """Create a mock LLM for testing."""
    mock = Mock()
    mock.invoke.return_value = "Mock LLM response"
    mock.with_structured_output.return_value = mock
    mock.bind_tools.return_value = mock
    return mock


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model for testing."""
    mock = Mock()
    mock.embed_documents.return_value = [[0.1, 0.2, 0.3] * 384]  # Mock embedding
    mock.embed_query.return_value = [0.1, 0.2, 0.3] * 384
    return mock


@pytest.fixture
def sample_patient_data():
    """Create sample patient data for testing."""
    return {
        "patient_id": 1,
        "age": 45,
        "medical_history": "diabetes",
        "previous_trials": "NCT12345",
        "trial_status": "completed",
        "trial_completion_date": "2023-01-15",
    }


@pytest.fixture
def sample_patient_data_no_name():
    """Create sample patient data without sensitive fields."""
    return {
        "age": 45,
        "medical_history": "diabetes",
        "previous_trials": "NCT12345",
        "trial_status": "completed",
        "trial_completion_date": "2023-01-15",
    }


@pytest.fixture
def sample_policy_document():
    """Create a sample policy document for testing."""
    from langchain_core.documents import Document

    return Document(
        page_content="""
        ### Age Restrictions
        - Patients must be between 18 and 75 years old.
        - Informed consent required.
        """,
        metadata={"source": "test_policy.md"},
    )


@pytest.fixture
def sample_trial_document():
    """Create a sample trial document for testing."""
    from langchain_core.documents import Document

    return Document(
        page_content="""
        Inclusion Criteria:
        - Patients with diabetes mellitus
        - Age 18-65 years
        
        Exclusion Criteria:
        - Pregnant women
        - Severe kidney disease
        """,
        metadata={
            "nctid": "NCT12345",
            "status": "recruiting",
            "diseases": "['diabetes']",
            "disease_category": "endocrine",
            "drugs": "['metformin']",
        },
    )


@pytest.fixture
def sample_agent_state():
    """Create a sample agent state for testing."""
    from src.llm_pharma.state import AgentState, initialize_state

    return initialize_state(
        patient_prompt="Is patient 1 eligible for trials?",
        max_revisions=5,
        max_trial_searches=2,
    )


@pytest.fixture
def mock_database_manager(test_config, sample_patient_data):
    """Create a mock database manager for testing."""
    mock = Mock()
    mock.config = test_config
    mock.get_patient_data.return_value = sample_patient_data
    mock.add_patient_data.return_value = 1
    mock.get_all_patients.return_value = [sample_patient_data]
    mock.get_database_stats.return_value = {
        "total_patients": 1,
        "patients_with_trials": 1,
        "database_size": 1024,
    }
    return mock


@pytest.fixture
def mock_vectorstore():
    """Create a mock vector store for testing."""
    mock = Mock()
    mock.similarity_search.return_value = []
    mock.as_retriever.return_value = Mock()
    mock._collection.count.return_value = 10
    mock.add_documents.return_value = None
    mock.delete_collection.return_value = None
    return mock


@pytest.fixture
def mock_retriever():
    """Create a mock retriever for testing."""
    mock = Mock()
    mock.get_relevant_documents.return_value = []
    return mock


@pytest.fixture
def set_test_env():
    """Set up test environment variables."""
    # Set required environment variables for testing
    os.environ["OPENAI_API_KEY"] = "test-key"
    os.environ["LANGCHAIN_API_KEY"] = "test-key"

    yield

    # Clean up after test
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    if "LANGCHAIN_API_KEY" in os.environ:
        del os.environ["LANGCHAIN_API_KEY"]


@pytest.fixture
def mock_workflow_result():
    """Create a mock workflow result for testing."""
    return {
        "patient_collector": {
            "last_node": "patient_collector",
            "patient_profile": "Test patient profile",
            "patient_data": {"age": 45, "medical_history": "diabetes"},
            "policy_eligible": True,
        },
        "grade_trials": {
            "last_node": "grade_trials",
            "relevant_trials": [
                {
                    "nctid": "NCT12345",
                    "relevance_score": "Yes",
                    "explanation": "Patient matches inclusion criteria",
                    "further_information": "No additional info needed",
                }
            ],
        },
    }


class MockChromaDB:
    """Mock ChromaDB client for testing."""

    def __init__(self):
        self.collections = {}

    def get_or_create_collection(self, name):
        if name not in self.collections:
            self.collections[name] = MockCollection()
        return self.collections[name]


class MockCollection:
    """Mock ChromaDB collection for testing."""

    def __init__(self):
        self.documents = []
        self.metadatas = []
        self.ids = []

    def count(self):
        return len(self.documents)

    def add(self, documents, metadatas=None, ids=None):
        self.documents.extend(documents)
        if metadatas:
            self.metadatas.extend(metadatas)
        if ids:
            self.ids.extend(ids)


@pytest.fixture
def mock_chromadb():
    """Create a mock ChromaDB for testing."""
    return MockChromaDB()
