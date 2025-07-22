# Backend Overview: `llm_pharma`

## 🎯 **Backend Architecture**

The backend provides the core business logic and AI components for the LLM Pharma clinical trial management system. It implements a LangGraph-based workflow for patient evaluation and trial matching.

## 📁 **Module Structure**

### **Core Location**: `backend/my_agent/`

| Module | File | Purpose | Key Components |
|--------|------|---------|----------------|
| **Workflow Manager** | `workflow_manager.py` | Main orchestration | LangGraph workflow, state management |
| **LLM Manager** | `llm_manager.py` | AI model management | Groq/OpenAI integration, fallback logic |
| **Database Manager** | `database_manager.py` | Data persistence | SQLite operations, ChromaDB vector stores |
| **Policy Service** | `policy_service.py` | Policy evaluation | Policy search, eligibility assessment |
| **Trial Service** | `trial_service.py` | Trial matching | Trial search, relevance scoring |
| **Patient Collector** | `patient_collector.py` | Patient data | Profile generation, data extraction |
| **State Management** | `State.py` | Workflow state | Agent state, memory management |

## 🔧 **Core Components**

### **1. Workflow Manager** (`workflow_manager.py`)
- **Purpose**: Orchestrates the entire patient evaluation workflow using LangGraph
- **Key Features**:
  - Defines workflow nodes and edges
  - Manages state transitions and persistence
  - Handles workflow execution and interruption
  - Provides workflow resumption capabilities
- **Workflow Nodes**:
  - `patient_collector`: Extracts patient information and generates profiles
  - `policy_search`: Searches relevant institutional policies
  - `policy_evaluator`: Evaluates patient eligibility against policies
  - `trial_search`: Searches for matching clinical trials
  - `grade_trials`: Scores trial relevance and eligibility
  - `profile_rewriter`: Refines patient profile when no trials found

### **2. LLM Manager** (`llm_manager.py`)
- **Purpose**: Manages multiple LLM models with fallback logic
- **Key Features**:
  - Support for Groq and OpenAI providers
  - Automatic fallback between models
  - Separate managers for completions and tool calls
  - Configuration via Hydra
- **Supported Models**:
  - Completion models: `mistral-saba-24b`, `meta-llama/llama-4-maverick-17b-128e-instruct`, `meta-llama/llama-4-scout-17b-16e-instruct`
  - Tool models: `llama-3.3-70b-versatile`, `llama3-70b-8192`, `deepseek-r1-distill-llama-70b`, `moonshotai/kimi-k2-instruct`, `qwen/qwen3-32b`

### **3. Database Manager** (`database_manager.py`)
- **Purpose**: Handles all database and vector store operations
- **Key Features**:
  - SQLite patient database management
  - ChromaDB vector stores for policies and trials
  - Nomic embeddings integration
  - Disease mapping utilities
- **Database Schema**:
  ```sql
  CREATE TABLE patients (
      patient_id INTEGER PRIMARY KEY,
      name TEXT,
      age INTEGER,
      medical_history TEXT,
      previous_trials TEXT,
      trial_status TEXT,
      trial_completion_date TEXT
  );
  ```
- **Vector Stores**:
  - Policy collection: Institutional policy documents
  - Trial collection: Clinical trial documents with metadata filtering

### **4. Policy Service** (`policy_service.py`)
- **Purpose**: Handles policy-related operations and eligibility assessment
- **Key Features**:
  - Policy document retrieval via vector search
  - Policy evaluation using ReAct agent
  - Date and number comparison tools
  - Structured eligibility assessment
- **Tools**:
  - `get_today_date`: Returns current date
  - `check_months_since_date`: Date threshold checking
  - `compare_numbers`: Numeric comparisons

### **5. Trial Service** (`trial_service.py`)
- **Purpose**: Manages trial matching and relevance scoring
- **Key Features**:
  - Self-query retrieval for trial matching
  - Metadata-based filtering (disease categories, drugs, status)
  - Trial relevance scoring with hallucination detection
  - Structured output for trial evaluation
- **Metadata Fields**:
  - `disease_category`: cancer, leukemia, mental_health
  - `drugs`: List of trial drugs
  - `status`: Trial recruitment status

### **6. Patient Collector** (`patient_collector.py`)
- **Purpose**: Handles patient data collection and profile generation
- **Key Features**:
  - Patient ID extraction from natural language
  - Patient profile generation from medical data
  - Profile rewriting for better trial matching
  - Configuration management
- **Components**:
  - `PatientCollectorConfig`: Configuration class
  - `Patient_ID`: Pydantic model for ID extraction
  - `patient_collector_node`: Main collection node
  - `profile_rewriter_node`: Profile refinement node

### **7. State Management** (`State.py`)
- **Purpose**: Defines and manages workflow state
- **Key Features**:
  - TypedDict-based state definition
  - State initialization and management
  - Workflow progress tracking
- **State Fields**:
  - Patient data: ID, profile, medical history
  - Policy data: eligibility, rejection reasons
  - Trial data: search results, relevance scores
  - Workflow control: node tracking, revision counts

## 🚀 **Usage**

### **Entry Point**: `workflow_manager.py`
```python
# Run backend test
from backend.my_agent.workflow_manager import WorkflowManager
from omegaconf import DictConfig

# Initialize with config
workflow = WorkflowManager.from_config(config)

# Run workflow
result = workflow.run_workflow("I need information about patient 1")
```

### **Configuration**
The backend uses Hydra for configuration management:
- Model settings in `config/models/`
- Database settings in `config/directories/`
- File paths in `config/files/`

### **Dependencies**
- **LangGraph**: Workflow orchestration
- **LangChain**: LLM integration and tools
- **ChromaDB**: Vector storage
- **SQLite**: Database management
- **Hydra**: Configuration management
- **Groq/OpenAI**: LLM providers

## 🔄 **Workflow Process**

1. **Patient Collection**: Extract patient ID and retrieve data from database
2. **Profile Generation**: Create comprehensive patient profile from medical data
3. **Policy Search**: Find relevant institutional policies using vector search
4. **Policy Evaluation**: Check patient eligibility against policies using ReAct agent
5. **Trial Search**: Find matching clinical trials with metadata filtering
6. **Trial Grading**: Score trial relevance and detect hallucinations
7. **Profile Refinement**: Update patient profile if no trials found
8. **Result Generation**: Compile final evaluation results

## 📊 **Performance**

- **Vector Search**: Policy and trial document retrieval
- **Response Time**: Varies by workflow complexity and model selection
- **Memory Usage**: Optimized for concurrent processing
- **Scalability**: Supports multiple concurrent evaluations
- **Fallback Logic**: Automatic model switching for reliability

## 🧪 **Testing**

Run backend tests:
```bash
cd tests/
pytest unit/
```

## 🔧 **Development**

### **Adding New Components**:
1. Create new module in `backend/my_agent/`
2. Update workflow manager if adding new nodes
3. Add configuration in `config/`
4. Update state definition if needed
5. Add tests in `tests/unit/`

### **Configuration Changes**:
1. Modify appropriate config file in `config/`
2. Update default settings in `config/config.yaml`
3. Test with different configurations
4. Update documentation

### **Model Management**:
1. Add new models to `LLMManager.get_default_managers()`
2. Configure provider (groq/openai) and model ID
3. Test fallback behavior
4. Update model lists in configuration