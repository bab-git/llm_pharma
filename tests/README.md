# 🧪 LLM Pharma Test Suite

This directory contains comprehensive tests for the LLM Pharma project, organized into unit, integration, and regression test categories.

## 📁 Structure

```
tests/
├── __init__.py
├── conftest.py              # Pytest configuration and fixtures
├── README.md               # This file
├── unit/                   # Unit tests for individual components
│   ├── __init__.py
│   ├── test_patient_service.py    # PatientService class tests
│   ├── test_llm_manager.py        # LLMManager class tests
│   └── test_state.py              # State management tests
├── integration/            # Integration tests for workflows
│   ├── __init__.py
│   └── test_dashboard_demo.py     # Dashboard demo integration tests
└── regression/             # End-to-end regression tests
    ├── __init__.py
    ├── test_end_to_end.py         # Complete workflow testing
    ├── test_patient_service_llm.py # PatientService with real LLM calls
    ├── test_policy_service.py     # Policy service functionality
    └── test_trial_workflow.py     # Trial-focused workflow testing
```

## 🚀 Quick Start

### Run All Tests
```bash
# Run all tests with pytest
pytest tests/ -v

# Run specific test categories
pytest tests/unit/ -v           # Unit tests only
pytest tests/integration/ -v    # Integration tests only
pytest tests/regression/ -v     # Regression tests only
```

### Run Specific Test Files
```bash
# Unit tests
pytest tests/unit/test_patient_service.py -v
pytest tests/unit/test_llm_manager.py -v
pytest tests/unit/test_state.py -v

# Integration tests
pytest tests/integration/test_dashboard_demo.py -v

# Regression tests
pytest tests/regression/test_end_to_end.py -v
pytest tests/regression/test_patient_service_llm.py -v
pytest tests/regression/test_policy_service.py -v
pytest tests/regression/test_trial_workflow.py -v
```

### Run Individual Test Methods
```bash
# Example: Run specific test method
pytest tests/unit/test_patient_service.py::TestPatientService::test_patient_service_initialization -v
```

## 🧪 Test Categories

### 1. Unit Tests (`tests/unit/`)

**Purpose**: Test individual components in isolation with mocked dependencies.

#### `test_patient_service.py`
- **Purpose**: Test PatientService class functionality
- **Key Tests**:
  - `test_patient_service_initialization()`: Service initialization with dependencies
  - `test_extract_patient_id()`: Patient ID extraction from prompts
  - `test_fetch_patient_data()`: Patient data retrieval (non-PII only)
  - `test_build_profile()`: Patient profile generation
  - `test_patient_collector_node()`: Complete patient collection workflow
  - `test_profile_rewriter_node()`: Profile rewriting functionality

#### `test_llm_manager.py`
- **Purpose**: Test LLMManager class with multiple model providers
- **Key Tests**:
  - `test_llm_manager_initialization()`: Manager setup with model configs
  - `test_llm_manager_advance()`: Model switching functionality
  - `test_llm_manager_reset()`: Reset to first model
  - `test_llm_manager_invalid_provider()`: Error handling for invalid providers
  - `test_get_default_managers()`: Default manager creation

#### `test_state.py`
- **Purpose**: Test state management and AgentState functionality
- **Key Tests**:
  - `test_agent_state_creation()`: Valid state creation
  - `test_create_agent_state_returns_default_values()`: Default state values
  - `test_create_agent_state_returns_new_instance()`: State isolation

### 2. Integration Tests (`tests/integration/`)

**Purpose**: Test complete workflows and component interactions.

#### `test_dashboard_demo.py`
- **Purpose**: Test dashboard functionality with demo graph
- **Key Tests**:
  - `test_demo_graph_creation()`: Demo graph creation
  - `test_demo_graph_structure()`: Graph component validation
  - `test_dashboard_import()`: Dashboard component imports
  - `test_dashboard_creation()`: Dashboard initialization
  - `test_dashboard_launch_simulation()`: Launch preparation
  - `test_demo_mode_integration()`: Complete demo workflow

### 3. Regression Tests (`tests/regression/`)

**Purpose**: End-to-end testing with real dependencies and API calls.

#### `test_end_to_end.py`
- **Purpose**: Complete workflow testing from patient collection to trial grading
- **Key Tests**:
  - `test_end_to_end_workflow()`: Full patient → policy → trial workflow
  - `test_workflow_manager_integration()`: WorkflowManager integration

#### `test_patient_service_llm.py`
- **Purpose**: PatientService testing with real LLM API calls
- **Key Tests**:
  - `test_patient_id_extraction()`: Real LLM patient ID extraction
  - `test_patient_data_fetching()`: Database integration
  - `test_profile_generation()`: Profile creation with LLM
  - `test_profile_rewriting()`: Profile rewriting with LLM
  - `test_full_patient_collector_node()`: Complete patient collection
  - `test_profile_rewriter_node()`: Profile rewriting workflow

#### `test_policy_service.py`
- **Purpose**: Policy service functionality with real dependencies
- **Key Tests**:
  - `test_policy_service()`: Policy search and evaluation
  - Policy eligibility determination
  - Policy database integration

#### `test_trial_workflow.py`
- **Purpose**: Trial-focused workflow testing (patient → trial search → evaluation)
- **Key Tests**:
  - `test_trial_workflow()`: Complete trial workflow
  - Patient collection for specific patient IDs
  - Trial search and evaluation
  - Command-line argument support

## 🔧 Test Configuration

### Environment Setup
```bash
# Activate virtual environment
source .llmpenv/bin/activate

# Install test dependencies
pip install -r requirements.txt

# Set up API keys (for regression tests)
export GROQ_API_KEY=your_groq_api_key
export OPENAI_API_KEY=your_openai_api_key
```

### Pytest Configuration
The `conftest.py` file provides:
- **Mock Fixtures**: `mock_llm`, `mock_database_manager`, `mock_vectorstore`
- **Sample Data**: `sample_patient_data`, `sample_policy_data`
- **Test Configuration**: `test_config`, `set_test_env`

## 🎯 Test Scenarios

### Scenario 1: Development Testing
```bash
# Quick unit tests during development
pytest tests/unit/ -v
```

### Scenario 2: Integration Validation
```bash
# Test component interactions
pytest tests/integration/ -v
```

### Scenario 3: End-to-End Validation
```bash
# Full workflow testing (requires API keys)
pytest tests/regression/ -v
```

### Scenario 4: Specific Component Testing
```bash
# Test specific service
pytest tests/unit/test_patient_service.py -v
pytest tests/regression/test_patient_service_llm.py -v
```

## 🔍 What Tests Validate

### Unit Test Validation
- ✅ Component initialization with dependency injection
- ✅ Method functionality with mocked dependencies
- ✅ Error handling and edge cases
- ✅ State management and data flow

### Integration Test Validation
- ✅ Component interactions and workflows
- ✅ Dashboard functionality and demo mode
- ✅ Graph creation and structure validation

### Regression Test Validation
- ✅ Real API calls and responses
- ✅ Database integration and data persistence
- ✅ Complete end-to-end workflows
- ✅ Performance and reliability

## 🐛 Troubleshooting

### Common Issues

**❌ Import Errors**
```bash
# Ensure you're in the project root
cd /path/to/llm_pharma

# Check Python path
python -c "import sys; print(sys.path)"
```

**❌ API Key Errors (Regression Tests)**
```bash
# Set required API keys
export GROQ_API_KEY=your_groq_api_key
export OPENAI_API_KEY=your_openai_api_key

# Verify keys are set
echo $GROQ_API_KEY
echo $OPENAI_API_KEY
```

**❌ Database Connection Errors**
```bash
# Check if database files exist
ls -la data/
ls -la vector_store/

# Recreate demo database if needed
python scripts/create_patients_database.py
```

### Debug Mode
```bash
# Run with verbose output
pytest tests/ -v -s

# Run specific test with debug
pytest tests/regression/test_end_to_end.py -v -s --tb=long
```

## 📝 Adding New Tests

### Adding Unit Tests
1. Create test file in `tests/unit/`
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures from `conftest.py`
4. Mock external dependencies

### Adding Integration Tests
1. Create test file in `tests/integration/`
2. Test component interactions
3. Use real dependencies where appropriate

### Adding Regression Tests
1. Create test file in `tests/regression/`
2. Test with real APIs and databases
3. Include proper error handling and logging

### Test Structure
```python
def test_something():
    """Test description."""
    # Arrange
    # Act
    # Assert
    assert condition, "Error message"
```

## 🔄 Continuous Integration

These tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run Unit Tests
  run: pytest tests/unit/ -v

- name: Run Integration Tests
  run: pytest tests/integration/ -v

- name: Run Regression Tests
  run: |
    export GROQ_API_KEY=${{ secrets.GROQ_API_KEY }}
    pytest tests/regression/ -v
```

## 📚 Related Files

- `backend/my_agent/`: Main backend components
- `frontend/`: Dashboard and demo components
- `conftest.py`: Pytest configuration and fixtures
- `requirements.txt`: Test dependencies
- `scripts/`: Database and setup scripts
