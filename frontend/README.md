# 🚀 LLM Pharma Frontend

A comprehensive Gradio dashboard for the LLM Pharma clinical trial management system with interactive workflow control.

## 📋 Quick Start

### Basic Usage
```bash
# Launch dashboard on default settings (127.0.0.1:7958)
python frontend/app.py

# Launch on a different port
python frontend/app.py --port 8080

# Launch with public sharing enabled
python frontend/app.py --share

# Launch on different host and port
python frontend/app.py --host 0.0.0.0 --port 8080

# Launch in demo mode (uses dummy data)
python frontend/app.py --demo
```

### Help
```bash
# Show command line options
python frontend/app.py --help
```

## 🏗️ Architecture

The frontend consists of two main components:

### **Main Application** (`app.py`)
- **Purpose**: Entry point and configuration management
- **Features**:
  - Command-line argument parsing
  - Environment and configuration loading
  - Database initialization and validation
  - Workflow manager creation (production or demo mode)
  - Gradio interface launching

### **GUI Interface** (`helper_gui.py`)
- **Purpose**: Comprehensive Gradio web interface
- **Features**:
  - Multi-tab dashboard with interactive controls
  - Real-time workflow state management
  - Patient profile editing and management
  - Policy conflict resolution tools
  - Trial matching and scoring visualization
  - Thread management and state persistence

## 🎯 Dashboard Features

### **Agent Control Tab**
- **Patient Query Input**: Natural language patient queries
- **Patient ID Dropdown**: Quick selection from available patients
- **Notification Center**: Real-time status updates and guidance
- **Workflow Control**: Start/Continue evaluation with interrupt points
- **Debug Mode**: Advanced controls for development
- **Thread Management**: Multi-session support

### **Status Panels**
- **Patient Profile**: Editable patient information with formatting
- **Policy Evaluation**: Policy conflicts and resolution tools
- **Trials Summary**: Overview of matched trials with relevance scores
- **Execution History**: Step-by-step workflow progress

### **Data Visualization Tabs**
- **Potential Trials**: Detailed trial information in interactive tables
- **Trials Scores**: Comprehensive scoring and ranking results

## 🔧 Required Backend Integration

The app integrates with the backend through:

```python
# Workflow Manager Integration
from backend.my_agent.workflow_manager import WorkflowManager
from backend.my_agent.llm_manager import LLMManager
from backend.my_agent.database_manager import DatabaseManager

# Configuration Management
from omegaconf import OmegaConf
configs = OmegaConf.load("config/config.yaml")

# Database Setup
db_manager = DatabaseManager(configs=configs)
db_manager.create_demo_patient_database()
db_manager.create_trials_dataset(status="recruiting")
```

## 📡 Accessing the Dashboard

Once launched successfully, access the dashboard at:
- **Local**: http://127.0.0.1:7958 (default)
- **Custom**: http://[host]:[port] (if you specified different values)
- **Public**: Shared URL (when using --share flag)

## 🔄 Workflow Integration

The dashboard provides full integration with the LangGraph workflow:

### **Workflow Nodes**
- `patient_collector`: Patient profile generation
- `policy_search`: Institutional policy retrieval
- `policy_evaluator`: Policy eligibility assessment
- `trial_search`: Clinical trial matching
- `grade_trials`: Trial relevance scoring
- `profile_rewriter`: Profile refinement

### **State Management**
- **Thread-based**: Multiple evaluation sessions
- **Persistent**: SQLite checkpointing
- **Interactive**: Real-time state updates
- **Resumable**: Continue interrupted workflows

## 📊 Data Management

### **Patient Data**
- SQLite database with 100+ demo patients
- Patient profile generation and editing
- Medical history and trial status tracking

### **Policy Data**
- Vector store integration for policy documents
- Policy conflict detection and resolution
- Eligibility assessment tools

### **Trial Data**
- Clinical trial database with metadata filtering
- Disease category mapping (cancer, leukemia, mental_health)
- Relevance scoring with hallucination detection

## 🎨 User Interface

### **Design Features**
- **Responsive Layout**: Multi-column design with proper scaling
- **Color-coded Sections**: Visual organization by function
- **Interactive Elements**: Buttons, dropdowns, and editable fields
- **Real-time Updates**: Live status and notification system
- **Custom Styling**: Enhanced CSS for better user experience

### **Navigation**
- **Tab-based Interface**: Organized workflow sections
- **Status Indicators**: Clear progress and state information
- **Help Text**: Contextual guidance throughout the interface
- **Debug Controls**: Advanced options for development

## 📚 Files

- `app.py`: Main frontend application and launcher
- `helper_gui.py`: Comprehensive Gradio interface implementation
- `demo_graph.py`: Demo mode with dummy workflow nodes
- `style.css`: Custom styling for enhanced UI
- `README.md`: This documentation
- `README_DASHBOARD.md`: Detailed dashboard documentation

## 🔧 Configuration

### **Environment Variables**
```bash
# Required for LLM integration
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Optional for deployment
PORT1=8080  # Custom port for deployment
```

### **Configuration Files**
- `config/config.yaml`: Main configuration
- `config/models/`: LLM model settings
- `config/directories/`: Path configurations
- `config/files/`: File path settings

## 🚀 Deployment

### **Local Development**
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export GROQ_API_KEY=your_key
export OPENAI_API_KEY=your_key

# Launch dashboard
python frontend/app.py
```

### **Production Deployment**
```bash
# Launch with public access
python frontend/app.py --host 0.0.0.0 --port 8080 --share

# Or use environment variable
export PORT1=8080
python frontend/app.py --share
```

## 🔄 Integration

This app integrates with:
- **LangGraph**: For workflow orchestration
- **Groq/OpenAI**: For language model operations
- **ChromaDB**: For vector storage and retrieval
- **SQLite**: For persistent state management
- **Gradio**: For the web interface
- **Hydra**: For configuration management 