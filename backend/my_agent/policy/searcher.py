"""
Policy Searcher Module

This module contains the PolicySearcher class responsible for retrieving relevant
institutional policies based on patient profiles using vector search.
"""

import logging
from typing import List

from langchain.schema import Document

from backend.my_agent.database_manager import DatabaseManager


class PolicySearcher:
    """
    Handles policy retrieval using vector search.

    This class is responsible for finding relevant institutional policies
    based on patient profiles using semantic search.
    """

    def __init__(self, db_manager: DatabaseManager, logger: logging.Logger):
        """
        Initialize the PolicySearcher.

        Args:
            db_manager: Database manager for vector store operations
            logger: Logger instance for this component
        """
        self.db_manager = db_manager
        self.logger = logger

    def run(self, patient_profile: str) -> List[Document]:
        """
        Retrieve relevant policies based on patient profile.

        Args:
            patient_profile: Patient profile text to search against

        Returns:
            List of relevant policy documents
        """
        try:
            if not patient_profile:
                self.logger.warning("No patient profile available for policy search")
                return []

            # Create or load policy vector store
            policy_vectorstore = self.db_manager.create_policy_vectorstore()

            # Create retriever
            retriever = policy_vectorstore.as_retriever(search_kwargs={"k": 5})

            # Retrieve relevant policies
            docs_retrieved = retriever.get_relevant_documents(patient_profile)
            self.logger.info(
                f"Retrieved policies to be evaluated: {len(docs_retrieved)}"
            )
            self.logger.info(
                f"✅ Retrieved {len(docs_retrieved)} relevant policy sections"
            )

            return docs_retrieved

        except Exception as e:
            self.logger.error(f"❌ Error in policy search: {e}")
            return []
