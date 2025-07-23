"""
Database Manager for LLM Pharma

This module handles all database and vector store operations including:
- Patient database creation and management
- Policy vector store creation and retrieval
- Trial vector store creation and retrieval
- Database utilities and helpers

COMPLETED FEATURES:
==================

1. Patient Database Management - COMPLETED
   - Create demo patient database with 100 sample patients
   - Fetch patient data by ID
   - SQLite database operations

2. Policy Vector Store - COMPLETED
   - Create vector store from institutional policy documents
   - Retrieve relevant policies based on patient profile
   - ChromaDB integration with Nomic embeddings

3. Trial Vector Store - COMPLETED
   - Create vector store from clinical trials dataset
   - Filter trials by status and disease categories
   - Self-query retrieval for trial matching

4. Disease Mapping - COMPLETED
   - Map diseases to categories (cancer, leukemia, mental_health)
   - Support for custom disease mappings

USAGE EXAMPLE:
==============

    from backend.my_agent.database_manager import DatabaseManager

    # Initialize database manager
    db_manager = DatabaseManager()

    # Create demo patient database
    db_manager.create_demo_patient_database()

    # Get patient data
    patient_data = db_manager.get_patient_data(1)

    # Create policy vector store
    policy_store = db_manager.create_policy_vectorstore()

    # Create trial vector store
    trial_store = db_manager.create_trial_vectorstore()

REQUIREMENTS:
=============

Install required packages:
    pip install chromadb langchain-community langchain-nomic pandas sqlite3

"""

import ast
import json
import os
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import chromadb
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_nomic import NomicEmbeddings
from omegaconf import DictConfig


class DatabaseManager:
    """
    Manages all database and vector store operations for the LLM Pharma system.
    """

    def __init__(
        self, project_root: Optional[str] = None, configs: Optional[DictConfig] = None
    ):
        """
        Initialize the DatabaseManager.

        Args:
            project_root: Path to the project root directory. If None, will be inferred.
            configs: Optional Hydra config for overriding paths
        """
        if project_root is None:
            # Infer project root from current file location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.project_root = os.path.dirname(os.path.dirname(current_dir))
        else:
            self.project_root = project_root
        # If configs is provided, use it for paths
        if configs is not None:
            dirs = configs.directories
            files = configs.files
            self.default_db_path = os.path.join(
                self.project_root, dirs.sql_server, "patients.db"
            )
            self.default_vectorstore_path = os.path.join(
                self.project_root, dirs.vector_store
            )
            self.default_policy_path = os.path.join(
                self.project_root, files.policy_markdown
            )
            self.default_trials_path = os.path.join(self.project_root, files.trials_csv)
            self.default_disease_mapping_path = os.path.join(
                self.project_root, files.disease_mapping
            )
        else:
            self.default_db_path = os.path.join(
                self.project_root, "sql_server", "patients.db"
            )
            self.default_vectorstore_path = os.path.join(
                self.project_root, "vector_store"
            )
            self.default_policy_path = os.path.join(
                self.project_root, "source_data", "instut_trials_policy.md"
            )
            self.default_trials_path = os.path.join(
                self.project_root, "data", "trials_data.csv"
            )
            self.default_disease_mapping_path = os.path.join(
                self.project_root, "source_data", "disease_mapping.json"
            )

    def _ensure_directory(self, path: str) -> None:
        """Ensure directory exists for the given path."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def _get_absolute_path(self, path: str) -> str:
        """Convert relative path to absolute path relative to project root."""
        if os.path.isabs(path):
            return path
        return os.path.join(self.project_root, path)

    def create_demo_patient_database(
        self, db_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a demo patient database with randomly generated patient data.

        Args:
            db_path: Path where the database file will be created. If None, uses default.

        Returns:
            pandas.DataFrame: DataFrame containing the generated patient data
        """
        if db_path is None:
            db_path = self.default_db_path

        db_path = self._get_absolute_path(db_path)
        self._ensure_directory(db_path)

        # Remove existing database if it exists
        if os.path.exists(db_path):
            os.remove(db_path)

        # Define columns for the database
        columns = [
            "patient_id",
            "name",
            "age",
            "medical_history",
            "previous_trials",
            "trial_status",
            "trial_completion_date",
        ]
        data = []

        # Given names and surnames
        names = [
            "John",
            "Jane",
            "Alice",
            "Michael",
            "Emily",
            "Daniel",
            "Sophia",
            "James",
            "Emma",
            "Oliver",
        ]
        surnames = [
            "Doe",
            "Smith",
            "Johnson",
            "Brown",
            "Davis",
            "Garcia",
            "Martinez",
            "Anderson",
            "Thomas",
            "Wilson",
        ]

        # Generate all possible unique combinations of names and surnames
        combinations = [(name, surname) for name in names for surname in surnames]
        random.shuffle(combinations)
        unique_names = combinations[:100]
        full_names = [f"{name} {surname}" for name, surname in unique_names]

        # Load diseases from the JSON file
        try:
            with open(self.default_disease_mapping_path, "r") as file:
                trial_diseases = json.load(file)
            list_trial_diseases = list(trial_diseases.keys())
        except FileNotFoundError:
            # Fallback if diseases file not found
            list_trial_diseases = [
                "myelomonocytic leukemia",
                "myeloid leukemia",
                "lymphoblastic leukemia",
                "colorectal cancer",
                "esophageal cancer",
                "gastric cancer",
            ]

        other_medical_conditions = [
            "Hypertension",
            "Diabetes",
            "Asthma",
            "Heart Disease",
            "Arthritis",
            "Chronic Pain",
            "Anxiety",
            "Depression",
            "Obesity",
        ]

        all_conditions = list(set(list_trial_diseases + other_medical_conditions))
        trial_statuses = ["Completed", "Ongoing", "Withdrawn"]

        def random_date(start, end):
            return start + timedelta(days=random.randint(0, int((end - start).days)))

        # start_date must be 2 years before now
        start_date = datetime.now() - timedelta(days=365 * 2)
        # end_date must be a month before now
        end_date = datetime.now() - timedelta(days=10)

        # Generate 100 patients
        for i in range(1, 101):
            name = random.choice(full_names)
            age = random.randint(20, 80)
            
            # 30% chance of having 2 or 3 diseases, 70% chance of having 1 disease
            if random.random() < 0.3:
                # Patient has 2 or 3 diseases
                num_diseases = random.choice([2, 3])
                selected_conditions = random.sample(all_conditions, num_diseases)
                medical_history = "; ".join(selected_conditions)
            else:
                # Patient has 1 disease
                medical_history = random.choice(all_conditions)

            # 50% chance of having previous trials
            if random.choice([True, False]):
                previous_trials = f"NCT0{random.randint(1000000, 9999999)}"
                trial_status = random.choice(trial_statuses)
                trial_completion_date = random_date(start_date, end_date).strftime(
                    "%Y-%m-%d"
                )
            else:
                previous_trials = ""
                trial_status = ""
                trial_completion_date = ""

            # If trial is ongoing, no completion date
            if trial_status == "Ongoing":
                trial_completion_date = ""

            data.append(
                (
                    i,
                    name,
                    age,
                    medical_history,
                    previous_trials,
                    trial_status,
                    trial_completion_date,
                )
            )

        # Create DataFrame
        df = pd.DataFrame(data, columns=columns)

        # Save DataFrame to CSV in the same directory as the database
        csv_path = db_path.replace(".db", ".csv").replace("sql_server", "data")
        self._ensure_directory(csv_path)
        df.to_csv(csv_path, index=False)

        # Create SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create the patients table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id INTEGER PRIMARY KEY,
            name TEXT,
            age INTEGER,
            medical_history TEXT,
            previous_trials TEXT,
            trial_status TEXT,
            trial_completion_date TEXT
        )
        """
        )

        # Insert DataFrame into SQLite table
        df.to_sql("patients", conn, if_exists="append", index=False)

        # Commit and close the connection
        conn.commit()
        conn.close()

        print(f"Demo patient database created at: {db_path}")
        print(f"CSV export created at: {csv_path}")
        print(f"Total patients created: {len(df)}")

        return df

    def get_patient_data(
        self, patient_id: int, db_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Fetch all fields for the patient based on the given patient_id as an integer.

        Args:
            patient_id: The patient ID to fetch data for
            db_path: Path to the SQLite database file. If None, uses default.

        Returns:
            A dictionary containing the patient's medical history, or None if not found.
        """
        if db_path is None:
            db_path = self.default_db_path

        db_path = self._get_absolute_path(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = "SELECT * FROM patients WHERE patient_id=?"
        cursor.execute(query, (patient_id,))
        patient_data = cursor.fetchone()
        column_names = [column[0] for column in cursor.description]
        conn.close()

        if patient_data is None:
            return None
        else:
            results = dict(zip(column_names, patient_data))
        return results

    def create_policy_vectorstore(
        self,
        policy_file_path: Optional[str] = None,
        vectorstore_path: Optional[str] = None,
        collection_name: str = "policies",
    ) -> Chroma:
        """
        Create a vector store from the institutional policy document.

        Args:
            policy_file_path: Path to the policy markdown file. If None, uses default.
            vectorstore_path: Path to store the vector database. If None, uses default.
            collection_name: Name of the collection in the vector store

        Returns:
            Chroma: The created vector store
        """
        if policy_file_path is None:
            policy_file_path = self.default_policy_path
        if vectorstore_path is None:
            vectorstore_path = self.default_vectorstore_path

        policy_file_path = self._get_absolute_path(policy_file_path)
        vectorstore_path = self._get_absolute_path(vectorstore_path)

        # Ensure vector store directory exists
        os.makedirs(vectorstore_path, exist_ok=True)

        # Read the policy document
        with open(policy_file_path, "r", encoding="utf-8") as file:
            policy_content = file.read()

        # Split the policy into sections (by headers)
        sections = []
        current_section = ""
        current_title = ""

        for line in policy_content.split("\n"):
            if line.startswith("####"):
                # Save previous section if exists
                if current_section.strip():
                    sections.append(
                        {"title": current_title, "content": current_section.strip()}
                    )
                # Start new section
                current_title = line.replace("####", "").strip()
                current_section = ""
            else:
                current_section += line + "\n"

        # Add the last section
        if current_section.strip():
            sections.append(
                {"title": current_title, "content": current_section.strip()}
            )

        # Create documents for vector store
        policy_docs = []
        for section in sections:
            doc = Document(
                page_content=section["content"],
                metadata={"title": section["title"], "source": "institutional_policy"},
            )
            policy_docs.append(doc)

        # Create persistent client
        persistent_client = chromadb.PersistentClient(path=vectorstore_path)

        # Create or load vector store
        vectorstore = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=NomicEmbeddings(
                model="nomic-embed-text-v1.5", inference_mode="local"
            ),
        )

        # Check if collection is empty and add documents if needed
        if vectorstore._collection.count() == 0:
            vectorstore = Chroma.from_documents(
                documents=policy_docs,
                client=persistent_client,
                collection_name=collection_name,
                embedding=NomicEmbeddings(
                    model="nomic-embed-text-v1.5", inference_mode="local"
                ),
            )
            print(f"✅ Policy vector store created with {len(policy_docs)} sections")
        else:
            print(
                f"✅ Policy vector store loaded with {vectorstore._collection.count()} documents"
            )

        return vectorstore

    def create_trial_vectorstore(
        self,
        trials_csv_path: Optional[str] = None,
        vectorstore_path: Optional[str] = None,
        collection_name: str = "trials",
        status_filter: str = "recruiting",
        vstore_delete: bool = False,
    ) -> Optional[Chroma]:
        """
        Create a vector store from the clinical trials dataset.

        Args:
            trials_csv_path: Path to the trials CSV file. If None, uses default.
            vectorstore_path: Path to store the vector database. If None, uses default.
            collection_name: Name of the collection in the vector store
            status_filter: Filter trials by status (e.g., 'recruiting')
            vstore_delete: Whether to delete existing collection before creating new one

        Returns:
            Chroma: The created vector store, or None if no trials to add
        """
        if trials_csv_path is None:
            trials_csv_path = self.default_trials_path
        if vectorstore_path is None:
            vectorstore_path = self.default_vectorstore_path

        trials_csv_path = self._get_absolute_path(trials_csv_path)
        vectorstore_path = self._get_absolute_path(vectorstore_path)

        # Ensure vector store directory exists
        os.makedirs(vectorstore_path, exist_ok=True)

        # Create persistent client
        persistent_client = chromadb.PersistentClient(path=vectorstore_path)

        if vstore_delete:
            try:
                persistent_client.delete_collection(collection_name)
                print(f"Collection {collection_name} is deleted")
            except Exception:
                print(f"Collection {collection_name} does not exist.")

        # Create or load vector store
        vectorstore = Chroma(
            client=persistent_client,
            collection_name=collection_name,
            embedding_function=NomicEmbeddings(
                model="nomic-embed-text-v1.5", inference_mode="local"
            ),
        )

        if vectorstore._collection.count() > 0:
            print(
                f"✅ Trial vector store loaded with {vectorstore._collection.count()} trials"
            )
            return vectorstore

        # Read trials data
        df_trials = pd.read_csv(trials_csv_path)
        # Convert 'diseases' column from string to list
        df_trials["diseases"] = df_trials["diseases"].apply(ast.literal_eval)

        print(f"Loaded trials from: {trials_csv_path}")
        print(f"Total trials loaded: {len(df_trials)}")

        # Filter by status if specified
        if status_filter:
            df_trials = df_trials[df_trials["status"] == status_filter].reset_index(
                drop=True
            )
            print(
                f"✅ Filtered trials to status '{status_filter}': {len(df_trials)} trials"
            )

        # Create documents for vector store
        trial_docs = []
        for i, row in df_trials.iterrows():
            disease = self.disease_map(row["diseases"])
            if disease == "other_conditions":
                continue
            doc = Document(
                page_content=row["criteria"],
                metadata={
                    "nctid": row["nctid"],
                    "status": row["status"],
                    "diseases": str(row["diseases"]),
                    "disease_category": disease[0],
                    "drugs": row["drugs"],
                },
            )
            trial_docs.append(doc)

        print(f"Sample trial doc metadata:\n {trial_docs[0].metadata}")

        # Remove documents with very long content or other conditions
        list_remove = set()
        for i, doc in enumerate(trial_docs):
            if len(doc.page_content) > 10000:
                print(f"Removing trial {i} because it's too long")
                list_remove.add(i)
            if doc.metadata["disease_category"] == "other_conditions":
                print(f"Removing trial {i} because it's for other conditions")
                list_remove.add(i)

        # Remove list_remove indexes from trial_docs
        trial_docs = [doc for i, doc in enumerate(trial_docs) if i not in list_remove]

        print(
            f"Number of trial docs to be added to the vector store: {len(trial_docs)}"
        )
        if len(trial_docs) == 0:
            print("No trials to add to the vector store")
            return None

        # Create vector store with documents
        vectorstore = Chroma.from_documents(
            documents=trial_docs,
            client=persistent_client,
            collection_name=collection_name,
            embedding=NomicEmbeddings(
                model="nomic-embed-text-v1.5", inference_mode="local"
            ),
        )
        print(f"✅ Trial vector store created with {len(trial_docs)} trials")

        return vectorstore

    def disease_map(self, disease_list: List[str]) -> List[str]:
        """
        Map diseases to categories (cancer, leukemia, mental_health).

        Args:
            disease_list: List of disease names

        Returns:
            List of disease categories
        """
        # Read disease_mapping from file
        try:
            with open(self.default_disease_mapping_path, "r") as file:
                disease_mapping = json.load(file)
        except FileNotFoundError:
            print(
                f"Warning: Disease mapping file not found at {self.default_disease_mapping_path}"
            )
            return ["other_conditions"]

        categories = set()
        for disease in disease_list:
            if disease in disease_mapping:
                mapped = disease_mapping[disease]
                if mapped != "other_conditions":
                    categories.add(mapped)
                elif "cancer" in disease:
                    mapped = "cancer"
                elif "leukemia" in disease:
                    mapped = "leukemia"

        if len(categories) == 0:
            categories.add("other_conditions")
        return list(categories)

    def create_trials_dataset(self, status: Optional[str] = None) -> tuple:
        """
        Creates a dataset of clinical trials by downloading a CSV file from a GitHub repository and preprocessing it.

        Args:
            status: Filter trials by status (e.g., 'recruiting')

        Returns:
            tuple: A tuple containing:
                - df_trials (pandas.DataFrame): The preprocessed dataset of clinical trials.
                - csv_path (str): The path to the CSV file where the dataset is saved.
        """
        # URL to the raw_data.csv file
        url = "https://raw.githubusercontent.com/futianfan/clinical-trial-outcome-prediction/main/data/raw_data.csv"

        # Read the CSV file directly into a pandas DataFrame
        df_trials = pd.read_csv(url)

        if status is not None:
            df_trials = df_trials[df_trials["status"] == status].reset_index(drop=True)
            print(f"Only trials with status {status} are selected.")

        # Convert the string representation of lists to actual lists
        df_trials["diseases"] = df_trials["diseases"].apply(ast.literal_eval)

        # map label = 1 to success and label = 0 to failure
        df_trials["label"] = df_trials["label"].map({1: "success", 0: "failure"})

        # map why_stop null to not_stopped
        df_trials["why_stop"] = df_trials["why_stop"].fillna("not stopped")

        df_trials = df_trials.drop(columns=["smiless", "icdcodes"])

        # create data directory if it doesn't exist
        data_dir = os.path.join(self.project_root, "data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        csv_path = os.path.join(data_dir, "trials_data.csv")
        df_trials.to_csv(csv_path, index=False)
        print(
            f"The database for trials is saved to {csv_path} \n It has {len(df_trials)} rows."
        )

        return df_trials, csv_path

    def get_policy_retriever(self, patient_profile: str, k: int = 5):
        """
        Get a retriever for policy documents based on patient profile.

        Args:
            patient_profile: Patient profile to match against policies
            k: Number of policies to retrieve

        Returns:
            Retriever for policy documents
        """
        policy_vectorstore = self.create_policy_vectorstore()
        return policy_vectorstore.as_retriever(search_kwargs={"k": k})

    # def get_trial_retriever(self, llm_model, patient_profile: str):
    #     """
    #     Get a self-query retriever for trial documents based on patient profile.

    #     Args:
    #         llm_model: LLM model for self-query retrieval
    #         patient_profile: Patient profile to match against trials

    #     Returns:
    #         SelfQueryRetriever for trial documents
    #     """
    #     trial_vectorstore = self.create_trial_vectorstore()

    #     metadata_field_info = [
    #         AttributeInfo(
    #             name="disease_category",
    #             description="Defines the disease group of patients related to this trial. One of ['cancer', 'leukemia', 'mental_health']",
    #             type="string",
    #         ),
    #         AttributeInfo(
    #             name="drugs",
    #             description="List of drug names used in the trial",
    #             type="str",
    #         ),
    #     ]
    #     document_content_description = (
    #         "The list of patient conditions to include or exclude them from the trial"
    #     )

    #     question = f"""
    #     Which trials are relevant to the patient with the following medical history?\n
    #     patient_profile: {patient_profile}
    #     """

    #     return SelfQueryRetriever.from_llm(
    #         llm_model,
    #         trial_vectorstore,
    #         document_content_description,
    #         metadata_field_info,
    #     )
