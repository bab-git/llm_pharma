# LLM Pharma - Clinical Trial Management System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-green.svg)](https://langchain.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-orange.svg)](https://langchain.com/langgraph)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-purple.svg)](https://openai.com/)
[![Nomic](https://img.shields.io/badge/Nomic-GPT4All%20Embeddings-darkgreen.svg)](https://nomic.ai/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-red.svg)](https://chromadb.com/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20UI-yellow.svg)](https://gradio.app/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey.svg)](https://sqlite.org/)
[![Database Creation](https://img.shields.io/badge/Database-Creation%20Tools-brightgreen.svg)](https://sqlite.org/)
[![Hydra](https://img.shields.io/badge/Hydra-Config%20Management-9cf.svg)](https://hydra.cc/)
[![Poetry](https://img.shields.io/badge/Poetry-Dependency%20Management-cyan.svg)](https://python-poetry.org/)
[![Pytest](https://img.shields.io/badge/Pytest-Testing-green.svg)](https://pytest.org/)

A comprehensive LLM-powered system for evaluating patient eligibility for clinical trials using advanced agent-based workflows, vector databases, and interactive web interfaces.

## 🔬 Overview

LLM Pharma is an intelligent clinical trial management system that automates the evaluation of patients for potential clinical trials. The system utilizes Large Language Models (LLMs), vector databases, and agent-based workflows to:

- **Analyze patient medical histories** and generate comprehensive profiles
- **Evaluate eligibility** against institutional policies and trial criteria
- **Match patients** to relevant clinical trials with detailed explanations
- **Prevent hallucinations** through advanced grading and verification systems
- **Provide interactive dashboards** for clinical research coordinators

## 🛠️ Tech Stack

- **Python 3.9+**: Core programming language
- **LangChain**: Framework for building LLM applications
- **LangGraph**: Workflow orchestration and agent management
- **OpenAI GPT-4o**: Large Language Model provider
- **Nomic**: Local GPT4All-based embeddings for semantic search
- **ChromaDB**: Vector database for semantic search
- **Gradio**: Web interface framework
- **SQLite**: Relational database for patient data
- **Database Creation Tools**: Tools for creating and managing databases
- **Hydra**: Configuration management
- **Poetry**: Dependency management
- **Pytest**: Testing framework

## 🏗️ Architecture

The system is built using a modular architecture with the following key components:

### Backend Modules
- **WorkflowManager**: Orchestrates the LangGraph-based evaluation workflow
- **LLMManager**: Handles all LLM operations and prompt management
- **DatabaseManager**: Manages SQLite patient database operations
- **VectorStoreManager**: Handles ChromaDB vector stores for policies and trials
- **RetrievalManager**: Implements semantic search and self-query retrieval using Nomic GPT4All embeddings
- **GradingManager**: Provides trial relevance grading and hallucination detection
- **ToolManager**: Manages Python tools for date calculations and policy evaluation

### Frontend
- **Gradio Web Interface**: Interactive dashboard for patient evaluation
- **Real-time Processing**: Live workflow execution with status updates
- **Multi-tab Results**: Detailed views for profiles, policies, and trial matches

### Configuration
- **Hydra Configuration**: Centralized config management with YAML files
- **Environment Management**: Secure API key and settings management

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Poetry (for dependency management)
- OpenAI API key
- LangChain API key (optional, for tracing)

### Installation

1. **Clone the repository and navigate to the llm_pharma directory**:
   ```bash
   cd llm_pharma
   ```

2. **Install dependencies using Poetry**:
   ```bash
   make dev-install
   ```

3. **Set up environment variables**:
   ```bash
   # Create .env file with your API keys
   echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
   echo "LANGCHAIN_API_KEY=your_langchain_api_key_here" >> .env
   ```

4. **Initialize the database** (optional):
   ```bash
   make run --init-db
   ```

### Running the Application

#### Web Interface (Recommended)
```bash
make run-frontend
```
Then visit `http://127.0.0.1:7958` in your browser.

#### Command Line Interface
```bash
make run
```

#### With Custom Prompt
```bash
poetry run python -m llm_pharma.main --prompt "Is patient 2 eligible for any medical trial?"
```

## 📁 Project Structure

```
llm_pharma/
├── src/llm_pharma/           # Main application code
│   ├── __init__.py           # Package initialization
│   ├── main.py               # Main entry point
│   ├── state.py              # Agent state management
│   ├── database_manager.py   # SQLite database operations
│   ├── llm_manager.py        # LLM operations and chains
│   ├── workflow_manager.py   # LangGraph workflow orchestration
│   ├── helper_functions.py   # Utility functions
│   ├── agent_instructions.py # Prompts and instructions
│   ├── agents/               # Specialized agent modules
│   │   ├── __init__.py
│   │   ├── vectorstore_manager.py
│   │   ├── retrieval_manager.py
│   │   ├── grading_manager.py
│   │   └── tool_manager.py
│   └── frontend/             # Gradio web interface
│       ├── __init__.py
│       └── app.py
├── config/                   # Hydra configuration files
│   ├── config.yaml          # Main configuration
│   ├── model/               # Model configurations
│   ├── database/            # Database configurations
│   ├── vectorstore/         # Vector store configurations
│   ├── retrieval/           # Retrieval configurations
│   ├── agent/               # Agent configurations
│   └── frontend/            # Frontend configurations
├── tests/                   # Test suite
│   ├── conftest.py          # Shared test fixtures
│   ├── unit/                # Unit tests
│   └── integration/         # Integration tests
├── pyproject.toml           # Poetry configuration
├── Makefile                 # Development automation
└── README.md               # This file
```

## 🔧 Configuration

The system uses Hydra for configuration management. Key configuration files:

- `config/config.yaml`: Main configuration file
- `config/model/openai.yaml`: OpenAI model settings
- `config/database/sqlite.yaml`: Database configuration
- `config/vectorstore/chroma.yaml`: Vector store settings
- `config/agent/clinical_trial.yaml`: Workflow parameters

### Key Configuration Options

```yaml
# Model settings
model:
  model_id: "gpt-3.5-turbo"
  agent_model_id: "gpt-4o"
  temperature: 0.0

# Workflow limits
agent:
  workflow:
    max_revisions: 10
    max_trial_searches: 3

# Retrieval parameters  
retrieval:
  policy_retrieval:
    k: 5  # Number of policies to retrieve
  trial_retrieval:
    k: 6  # Number of trials to retrieve
```

## 🎯 Usage Examples

### Web Interface Usage

1. **Start the web interface**:
   ```bash
   make run-frontend
   ```

2. **Enter a patient query**:
   ```
   Is patient 15 eligible for any clinical trials?
   ```

3. **Review results** in the detailed tabs:
   - Patient Profile: Generated patient summary
   - Policy Evaluation: Institutional policy compliance
   - Trial Matches: Relevant clinical trials with explanations

### Programmatic Usage

```python
from omegaconf import OmegaConf
from llm_pharma.workflow_manager import WorkflowManager

# Load configuration
config = OmegaConf.load("config/config.yaml")

# Initialize workflow manager
workflow_manager = WorkflowManager(config)

# Run evaluation
result = workflow_manager.run_workflow(
    patient_prompt="Is patient 5 eligible for any medical trial?",
    thread_id="example_session"
)

# Process results
for node_name, node_result in result.items():
    print(f"{node_name}: {node_result}")
```

## 🧪 Testing

The project includes comprehensive unit and integration tests:

```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run integration tests only  
make test-integration

# Run tests with coverage report
make test
```

### Test Coverage

- **Helper Functions**: Disease mapping, data validation, formatting
- **Database Manager**: CRUD operations, query execution, statistics
- **LLM Manager**: Model initialization, chain creation, prompt management
- **Vector Store Manager**: Document processing, embedding operations
- **Workflow Manager**: Node execution, state management, error handling

## 🔄 Development Workflow

### Available Make Commands

```bash
make install       # Install production dependencies
make dev-install   # Install development dependencies
make run          # Run main application
make run-frontend # Run Gradio frontend
make lint         # Run linting (flake8, mypy)
make format       # Format code (black, isort)
make test         # Run tests with coverage
make check        # Run all checks (lint + format + test)
make clean        # Clean up build artifacts
```

### Code Quality Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing framework

## 📊 Features

### Core Capabilities

- **Patient Data Management**: SQLite database with full CRUD operations
- **Semantic Search**: ChromaDB vector stores with Nomic GPT4All embeddings for high-quality semantic retrieval
- **LLM-Powered Analysis**: OpenAI GPT models for profile generation and evaluation
- **Policy Evaluation**: Automated compliance checking with Python tools
- **Trial Matching**: Intelligent matching with relevance scoring
- **Hallucination Prevention**: Advanced grading to ensure factual responses
- **Interactive Interface**: Modern web UI with real-time processing

### Advanced Features

- **Self-Query Retrieval**: Metadata-aware trial search with GPT4All embeddings
- **Multi-step Workflows**: LangGraph-based agent orchestration  
- **Tool Integration**: Date calculations and numerical operations
- **Profile Rewriting**: Adaptive profile enhancement for better matches
- **Conversation Tracking**: Thread-based session management
- **Comprehensive Logging**: Detailed execution tracking
- **Local Embeddings**: Privacy-preserving semantic search using Nomic's GPT4All-based embedding models

## 🔒 Security & Privacy

- **Data Privacy**: Patient names and IDs are removed from processing
- **Secure Configuration**: Environment-based API key management
- **Input Validation**: Comprehensive data validation and sanitization
- **Error Handling**: Graceful error handling with informative messages

## 📈 Performance

- **Parallel Processing**: Concurrent tool execution where possible
- **Caching**: Vector store persistence for faster retrieval
- **Optimized Queries**: Efficient database operations
- **Memory Management**: SQLite checkpointing for workflow state

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Install development dependencies**: `make dev-install`
4. **Make your changes** and add tests
5. **Run the test suite**: `make check`
6. **Commit your changes**: `git commit -m 'Add amazing feature'`
7. **Push to the branch**: `git push origin feature/amazing-feature`
8. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive tests for new features
- Update documentation as needed
- Use descriptive commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain & LangGraph**: For the agent framework and workflow orchestration
- **OpenAI**: For the powerful LLM capabilities
- **ChromaDB**: For vector database functionality
- **Gradio**: For the interactive web interface
- **Hydra**: For configuration management

## 📞 Support

For questions, issues, or contributions:

1. **Check the Issues**: Look for existing issues or create a new one
2. **Documentation**: Review this README and inline documentation
3. **Tests**: Run the test suite to verify functionality
4. **Configuration**: Check your configuration files and environment variables

## 🚧 Future Enhancements

- **Enhanced RAG**: Graph-based retrieval with entity relationships
- **Multi-modal Support**: Image and document processing
- **Advanced Analytics**: Trial success prediction and patient outcome analysis
- **Integration APIs**: RESTful APIs for external system integration
- **Scalability**: Distributed processing and cloud deployment options
- **Embedding Optimization**: Fine-tuning of GPT4All embeddings for clinical domain specificity

---

**Note**: This system is designed for research and demonstration purposes. For production clinical trial management, consult with healthcare professionals and ensure compliance with relevant regulations and standards. 