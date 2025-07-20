#!/usr/bin/env python3
"""
Script to create the policies vector store for the LLM Pharma system.

This script creates a vector store from institutional policy documents
for efficient retrieval during patient eligibility evaluation.

Usage:
    python scripts/create_policies_vectorstore.py [--policy-file PATH] [--vectorstore-path PATH] [--collection-name NAME] [--force-recreate]

Options:
    --policy-file PATH        Path to the policy markdown file (default: source_data/instut_trials_policy.md)
    --vectorstore-path PATH   Path to store the vector database (default: vector_store)
    --collection-name NAME    Name of the collection in the vector store (default: policies)
    --force-recreate          Force recreation of the vector store even if it exists
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import helper_functions
sys.path.append(str(Path(__file__).parent.parent))

from backend.helper_functions import create_policy_vectorstore

def main():
    """Main function to create the policies vector store."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create policies vector store for LLM Pharma system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/create_policies_vectorstore.py
    python scripts/create_policies_vectorstore.py --policy-file data/policies.md
    python scripts/create_policies_vectorstore.py --vectorstore-path data/vector_store
    python scripts/create_policies_vectorstore.py --force-recreate
        """
    )
    
    parser.add_argument(
        "--policy-file",
        default="source_data/instut_trials_policy.md",
        help="Path to the policy markdown file (default: source_data/instut_trials_policy.md)"
    )
    
    parser.add_argument(
        "--vectorstore-path",
        default="vector_store",
        help="Path to store the vector database (default: vector_store)"
    )
    
    parser.add_argument(
        "--collection-name",
        default="policies",
        help="Name of the collection in the vector store (default: policies)"
    )
    
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of the vector store even if it exists"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths if relative
    if not os.path.isabs(args.policy_file):
        project_root = Path(__file__).parent.parent
        policy_file = project_root / args.policy_file
    else:
        policy_file = Path(args.policy_file)
    
    if not os.path.isabs(args.vectorstore_path):
        project_root = Path(__file__).parent.parent
        vectorstore_path = project_root / args.vectorstore_path
    else:
        vectorstore_path = Path(args.vectorstore_path)
    
    print("📋 LLM Pharma - Policies Vector Store Creator")
    print("=" * 50)
    
    # Check if policy file exists
    if not policy_file.exists():
        print(f"❌ Policy file not found at: {policy_file}")
        print("Please ensure the policy markdown file exists")
        sys.exit(1)
    
    # Check if vector store already exists
    collection_path = vectorstore_path / "chroma.sqlite3"
    if collection_path.exists() and not args.force_recreate:
        print(f"⚠️  Vector store already exists at: {vectorstore_path}")
        print("Use --force-recreate to overwrite the existing vector store")
        return
    
    try:
        # Create the vector store
        print(f"📚 Creating policies vector store...")
        print(f"📄 Policy file: {policy_file}")
        print(f"🗂️  Vector store path: {vectorstore_path}")
        print(f"📦 Collection name: {args.collection_name}")
        
        vectorstore = create_policy_vectorstore(
            policy_file_path=str(policy_file),
            vectorstore_path=str(vectorstore_path),
            collection_name=args.collection_name
        )
        
        print("\n✅ Policies vector store created successfully!")
        print(f"📁 Vector store location: {vectorstore_path}")
        print(f"📊 Collection count: {vectorstore._collection.count()} documents")
        
        # Display collection info
        print(f"\n📋 Collection information:")
        print(f"   Name: {args.collection_name}")
        print(f"   Documents: {vectorstore._collection.count()}")
        print(f"   Embedding model: nomic-embed-text-v1.5")
        
        print("\n🎉 Policies vector store is ready for use!")
        
    except Exception as e:
        print(f"❌ Error creating policies vector store: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 