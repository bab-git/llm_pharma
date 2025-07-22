# LLM Pharma - Data Setup Scripts

This folder contains scripts to set up all the required data for the LLM Pharma clinical trial workflow system.

## 📋 Overview

The LLM Pharma system requires three main data components:

1. **Patients Database** - SQLite database with patient information
2. **Policies Vector Store** - ChromaDB vector store for institutional policies
3. **Trials Vector Store** - ChromaDB vector store for clinical trials data

## 🚀 Quick Start

To set up all data at once, run the master script:

```bash
python scripts/setup_all_data.py
```

This will create all three data components in sequence.

## 📁 Individual Scripts

### 1. Patients Database Creator

Creates a demo patient database with randomly generated patient data.

```bash
python scripts/create_patients_database.py [--db-path PATH] [--force-recreate] [--config-path PATH] [--config-name NAME]
```

**Options:**
- `--db-path PATH` - Path where the database file will be created (default: `sql_server/patients.db`)
- `--force-recreate` - Force recreation of the database even if it exists
- `--config-path PATH` - Path to the config directory (default: `config`)
- `--config-name NAME` - Name of the config file without .yaml (default: `config`)

**Example:**
```bash
python scripts/create_patients_database.py --db-path data/patients.db --force-recreate
```

### 2. Policies Vector Store Creator

Creates a vector store from institutional policy documents.

```bash
python scripts/create_policies_vectorstore.py [--policy-file PATH] [--vectorstore-path PATH] [--collection-name NAME] [--force-recreate] [--config-path PATH] [--config-name NAME]
```

**Options:**
- `--policy-file PATH` - Path to the policy markdown file (default: `source_data/instut_trials_policy.md`)
- `--vectorstore-path PATH` - Path to store the vector database (default: `vector_store`)
- `--collection-name NAME` - Name of the collection in the vector store (default: `policies`)
- `--force-recreate` - Force recreation of the vector store even if it exists
- `--config-path PATH` - Path to the config directory (default: `config`)
- `--config-name NAME` - Name of the config file without .yaml (default: `config`)

**Example:**
```bash
python scripts/create_policies_vectorstore.py --policy-file data/policies.md --force-recreate
```

### 3. Trials Vector Store Creator

Downloads clinical trials data and creates a vector store for trial matching.

```bash
python scripts/create_trials_vectorstore.py [--trials-csv-path PATH] [--vectorstore-path PATH] [--collection-name NAME] [--status-filter STATUS] [--force-recreate] [--skip-data-download] [--config-path PATH] [--config-name NAME]
```

**Options:**
- `--trials-csv-path PATH` - Path to the trials CSV file (default: `data/trials_data.csv`)
- `--vectorstore-path PATH` - Path to store the vector database (default: `vector_store`)
- `--collection-name NAME` - Name of the collection in the vector store (default: `trials`)
- `--status-filter STATUS` - Filter trials by status (default: `recruiting`)
- `--force-recreate` - Force recreation of the vector store even if it exists
- `--skip-data-download` - Skip downloading trials data (use existing CSV file)
- `--config-path PATH` - Path to the config directory (default: `config`)
- `--config-name NAME` - Name of the config file without .yaml (default: `config`)

**Examples:**
```bash
# Download recruiting trials and create vector store
python scripts/create_trials_vectorstore.py --status-filter "recruiting"

# Use existing CSV file and create vector store
python scripts/create_trials_vectorstore.py --skip-data-download --force-recreate

# Create vector store for completed trials
python scripts/create_trials_vectorstore.py --status-filter "completed"
```

### 4. Master Setup Script

Runs all three scripts in sequence with common options.

```bash
python scripts/setup_all_data.py [--force-recreate] [--skip-patients] [--skip-policies] [--skip-trials] [--config-path PATH] [--config-name NAME]
```

**Options:**
- `--force-recreate` - Force recreation of all databases and vector stores
- `--skip-patients` - Skip patients database creation
- `--skip-policies` - Skip policies vector store creation
- `--skip-trials` - Skip trials data and vector store creation
- `--config-path PATH` - Path to the config directory (default: `config`)
- `--config-name NAME` - Name of the config file without .yaml (default: `config`)

**Examples:**
```bash
# Set up everything
python scripts/setup_all_data.py

# Force recreate everything
python scripts/setup_all_data.py --force-recreate

# Set up only trials data
python scripts/setup_all_data.py --skip-patients --skip-policies

# Use custom config
python scripts/setup_all_data.py --config-path custom_config --config-name my_config
```

## 📊 Data Structure

### Patients Database
- **Location**: `sql_server/patients.db` (SQLite)
- **CSV Export**: `sql_server/patients.csv`
- **Content**: 100 randomly generated patients with medical history, trial participation, demographics
- **Schema**: `patient_id`, `name`, `age`, `medical_history`, `previous_trials`, `trial_status`, `trial_completion_date`
- **Generation**: Uses `DatabaseManager.create_demo_patient_database()`

### Policies Vector Store
- **Location**: `vector_store/` (ChromaDB)
- **Collection**: `policies`
- **Source**: `source_data/instut_trials_policy.md`
- **Embedding Model**: `nomic-embed-text-v1.5`
- **Content**: Institutional policy sections for clinical trial eligibility
- **Creation**: Uses `DatabaseManager.create_policy_vectorstore()`

### Trials Vector Store
- **Location**: `vector_store/` (ChromaDB)
- **Collection**: `trials`
- **Source**: Downloaded from GitHub (clinical trial outcome prediction dataset)
- **Embedding Model**: `nomic-embed-text-v1.5`
- **Content**: Clinical trials with criteria, diseases, drugs, phases
- **Creation**: Uses `DatabaseManager.create_trial_vectorstore()`

## 🔧 Requirements

Before running the scripts, ensure you have:

1. **Python Dependencies**: Install required packages from `requirements.txt`
2. **Environment Variables**: Set up API keys if needed
3. **Disk Space**: Ensure sufficient space for databases and vector stores
4. **Configuration**: Optional Hydra configuration files in `config/` directory

## 🔧 Configuration Integration

All scripts support Hydra configuration management:

### **Config Options**
- `--config-path PATH` - Path to config directory (default: `config`)
- `--config-name NAME` - Config file name without .yaml (default: `config`)

### **Configuration Files**
- `config/config.yaml` - Main configuration
- `config/models/` - LLM model settings
- `config/directories/` - Path configurations
- `config/files/` - File path settings

### **Database Manager Integration**
All scripts use the `DatabaseManager` class from `backend.my_agent.database_manager`:
- Automatic path resolution and validation
- Configuration-aware initialization
- Error handling and progress reporting
- Sample data display and validation

## 🧪 Testing

After setting up the data, you can test the system:

```bash
# Test patient collector
python backend/test_patient_collector.py

# Test policy evaluator
python backend/test_policy_evaluator.py

# Test trial service
python backend/test_trial_service.py
```

## 📝 Implementation Details

### **Script Architecture**
- **Modular Design**: Each script handles one data component
- **Error Handling**: Comprehensive error handling with detailed messages
- **Progress Reporting**: Clear status messages and progress indicators
- **Path Management**: Automatic relative/absolute path conversion
- **Configuration Support**: Hydra integration for flexible configuration

### **Integration Points**
- **DatabaseManager**: All scripts use the centralized database manager
- **Configuration**: Hydra-based configuration management
- **Error Handling**: Consistent error handling across all scripts
- **Logging**: Detailed progress and error reporting

### **Data Validation**
- **Existence Checks**: Prevents accidental overwrites
- **Force Recreation**: Options to rebuild existing data
- **Sample Display**: Shows preview of created data
- **Collection Info**: Displays vector store statistics

## 🐛 Troubleshooting

If you encounter issues:

1. **Check file permissions** - Ensure write access to target directories
2. **Verify dependencies** - Make sure all required packages are installed
3. **Check API keys** - Some operations may require API keys for LLM services
4. **Review logs** - Scripts provide detailed error messages and stack traces
5. **Use force-recreate** - If data is corrupted, use `--force-recreate` to rebuild
6. **Check configuration** - Verify Hydra config files are properly formatted
7. **Validate paths** - Ensure all file paths are accessible and writable

## 📚 Files

- `setup_all_data.py` - Master script for complete data setup
- `create_patients_database.py` - Patients database creator
- `create_policies_vectorstore.py` - Policies vector store creator
- `create_trials_vectorstore.py` - Trials vector store creator
- `README.md` - This documentation
- `CHANGELOG.md` - Version history and changes

## 📞 Support

For issues or questions about the data setup scripts, please refer to the main project documentation or create an issue in the project repository. 