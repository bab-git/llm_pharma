# LLM Agents for Clinical Trial Management

<image src="images/dashboard_complete.jpg" alt="Pipeline Diagram" width="1000"/>

## Overview

This repository provides an agent-based Large Language Model (LLM) application designed to automate the evaluation of patients for clinical trials. 
By leveraging documents related to patients' medical histories, clinical policies, and trial inclusion criteria, this agentic system aims to enhance evaluation quality and reduce processing time, offering a glimpse into how advanced LLMs can revolutionize clinical trial management.

The project uses a limited demo dataset and simple agentic tools for demonstration purposes. 
To translate this concept into real-world clinical environments, comprehensive databases, clinical expertise, and advanced toolsets (such as Retrieval-Augmented Generation (RAG)) will be necessary.

### Objective
- **Automate clinical trial data management, analysis, compliance checks, and reporting.**

## Key Components

The application workflow involves the following core components:

1. **Patient Data Collection**: Gather and manage comprehensive medical histories and patient information.
2. **Patient Data Analysis**: Analyze patient data to identify key health indicators and relevant medical conditions.
3. **Clinical Compliance Verification**: Ensure patient data meets clinical policies and trial eligibility criteria.
4. **Trial Matching**: Match patients with suitable clinical trials based on their medical profiles.
5. **Hallucination Grader**: Guardrail the LLM outputs to ensure relevance and accuracy, preventing hallucinations.
6. **Human-in-the-Loop Interventions**: Enable clinical experts to review and refine intermediate and final outcomes.

## Visual Representation of the Pipeline:
<image src="images/diagram.png" alt="Pipeline Diagram" width="700"/>

## Pipeline Architecture

The pipeline is designed to leverage agentic workflows and provides the following functionality:

- **Local Embedding Models**: Patient data and trial information are embedded locally for enhanced privacy and accuracy.
- **Vector Stores for Information Retrieval**: Metadata-based retrieval ensures relevant information is found efficiently.
- **Chained LLM Calls**: Complex tasks are handled through a sequence of LLM prompts, each building on the previous output.
- **Agentic Graph Workflow**: Graph-based workflows evaluate patient eligibility in a structured and transparent manner.
- **Human-in-the-Loop Functionality**: Clinical experts can intervene at any stage, ensuring that automated evaluations are accurate and contextually relevant.
- **Hallucination Prevention**: Utilizes a hallucination grader to maintain high-quality LLM responses.

## Technologies and Tools

- **Local Models**: To embed policies and trial information securely.
- **Local Vector Stores**: Metadata-based retrieval, ensuring quick and contextually accurate data access.
- **Python Shell Integration**: For numerical calculations required during trial evaluation.
- **Langchain-based LLM Chains**: To handle complex, multi-step tasks.
- **Graph Workflows for Patient-Trial Matching**: Improves transparency and ensures structured decision-making.
- **Human-in-the-Loop for Validation**: Clinical experts are involved to provide oversight and make corrections where necessary.

## Demonstration Video

[Watch our demo videos on YouTube](https://www.youtube.com/playlist?list=PLMtE8Mev6Cct7n4NRsKVTgP8N5IzDSjbq) to see the application in action, including functionalities like:
- Patient Profile (Re-)generation
- Policy Evaluation
- Human in-the-Loop Interventions
- Hallucination Guardrails
- Retrieval Grader

## How to Run

### Requirements
- **Python 3.8+** or higher as needed to install langchain 0.2.3.
- **pip** for installing dependencies.
- Set up API keys for OpenAI, Pinecone, Hugging Face, and Tavily (if applicable).

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/bab-git/llm-pharma.git
   cd llm-pharma
   ```

2. Install dependencies using the provided `requirements.txt` file:
   ```bash
   pip install -r requirements.txt
   ```

3. Set the required environment variables for API access.

### Usage
- Run the notebook in the `notebooks/` directory to explore various components of the agent-based application step by step, including Patient Evaluation, Compliance Verification, and Trial Matching.
- At the end of the notebook, the complete solution can be run using a gradio dashboard.
  - Navigate to the provided URL (usually `http://127.0.0.1:7958`) to access the dashboard.


## Potential Future Improvements

To achieve a more robust real-world application, especially for managing diverse patient and trial conditions, the following enhancements are recommended:

1. **Graph-Based Retrieval-Augmented Generation (RAG)**
   - Implement graph-based RAG to leverage relationships between entities like patients, diseases, drugs, and trials, thus improving information retrieval.

2. **Advanced RAG Pipeline**
   - Use methods such as Adaptive RAG, Corrective RAG, and Self-RAG to enhance output quality and robustness, minimizing LLM hallucinations and inference errors.

3. **Advanced Chain of Thought Processing**
   - Implement more sophisticated multi-factor matching of patient profiles with trials to enable more nuanced decision-making.

4. **Fine-Tuning of LLMs**
   - Fine-tune the language model with clinical trial datasets to better understand nuances, enhancing both the accuracy and context relevance of its inferences.

5. **Cyclic Graphs for Iterative Evaluation**
   - Utilize cyclic graphs for iterative profile evaluations against multiple trials and policies, refining matches in multiple iterations to find the best fit for patients.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

We welcome contributions! Please feel free to open issues or submit pull requests for suggestions, improvements, or bug fixes. For significant contributions, please discuss the changes via an issue before submitting.

## Contact
For more information, questions, or feedback, please contact me at [https://www.linkedin.com/in/bhosseini/](https://www.linkedin.com/in/bhosseini/). 