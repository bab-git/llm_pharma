#!/usr/bin/env python3
"""
Script to create the trials data and vector store for the LLM Pharma system.

This script downloads clinical trials data from GitHub and creates a vector store
for efficient trial matching during patient screening.

Usage:
    python scripts/create_trials_vectorstore.py [--trials-csv-path PATH] [--vectorstore-path PATH] [--collection-name NAME] [--status-filter STATUS] [--force-recreate]

Options:
    --trials-csv-path PATH    Path to the trials CSV file (default: data/trials_data.csv)
    --vectorstore-path PATH   Path to store the vector database (default: vector_store)
    --collection-name NAME    Name of the collection in the vector store (default: trials)
    --status-filter STATUS    Filter trials by status (default: recruiting)
    --force-recreate          Force recreation of the vector store even if it exists
"""

import argparse
import os
import sys
from pathlib import Path

# Add the parent directory to the path to import helper_functions
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import OmegaConf

from backend.my_agent.database_manager import DatabaseManager


def main():
    """Main function to create the trials data and vector store."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create trials data and vector store for LLM Pharma system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/create_trials_vectorstore.py
    python scripts/create_trials_vectorstore.py --status-filter "completed"
    python scripts/create_trials_vectorstore.py --trials-csv-path data/my_trials.csv
    python scripts/create_trials_vectorstore.py --vectorstore-path data/vector_store
    python scripts/create_trials_vectorstore.py --force-recreate
        """,
    )

    parser.add_argument(
        "--trials-csv-path",
        default="data/trials_data.csv",
        help="Path to the trials CSV file (default: data/trials_data.csv)",
    )

    parser.add_argument(
        "--vectorstore-path",
        default="vector_store",
        help="Path to store the vector database (default: vector_store)",
    )

    parser.add_argument(
        "--collection-name",
        default="trials",
        help="Name of the collection in the vector store (default: trials)",
    )

    parser.add_argument(
        "--status-filter",
        default="recruiting",
        help="Filter trials by status (default: recruiting)",
    )

    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of the vector store even if it exists",
    )

    parser.add_argument(
        "--skip-data-download",
        action="store_true",
        help="Skip downloading trials data (use existing CSV file)",
    )

    parser.add_argument(
        "--config-path",
        default="config",
        help="Path to the config directory (default: config)",
    )
    parser.add_argument(
        "--config-name",
        default="config",
        help="Name of the config file without .yaml (default: config)",
    )

    args = parser.parse_args()

    # Convert to absolute paths if relative
    if not os.path.isabs(args.trials_csv_path):
        project_root = Path(__file__).parent.parent
        trials_csv_path = project_root / args.trials_csv_path
    else:
        trials_csv_path = Path(args.trials_csv_path)

    if not os.path.isabs(args.vectorstore_path):
        project_root = Path(__file__).parent.parent
        vectorstore_path = project_root / args.vectorstore_path
    else:
        vectorstore_path = Path(args.vectorstore_path)

    print("🧪 LLM Pharma - Trials Data and Vector Store Creator")
    print("=" * 60)

    # Load config if available
    config = None
    config_file = Path(args.config_path) / f"{args.config_name}.yaml"
    if config_file.exists():
        config = OmegaConf.load(str(config_file))

    try:
        # Step 1: Try to load existing trials data, download if not available
        df_trials = None
        csv_path = None

        if df_trials is None:
            print("📥 Downloading trials data...")
            print(f"🔍 Status filter: {args.status_filter}")

            db_manager = (
                DatabaseManager(configs=config)
                if config is not None
                else DatabaseManager()
            )
            df_trials, csv_path = db_manager.create_trials_dataset(
                status=args.status_filter
            )

            print("✅ Trials data downloaded successfully!")
            print(f"📁 CSV file location: {csv_path}")
            print(f"📊 Total trials: {len(df_trials)}")

            # Display sample data
            print("\n📋 Sample trials data:")
            print(df_trials.head(3).to_string(index=False))

        # Step 2: Check if vector store already exists
        collection_path = vectorstore_path / "chroma.sqlite3"
        if collection_path.exists() and not args.force_recreate:
            print(f"\n⚠️  Vector store already exists at: {vectorstore_path}")
            print("Use --force-recreate to overwrite the existing vector store")
            # return

        # Step 3: Create the vector store
        print("\n📚 Creating trials vector store...")
        print(f"📄 Trials CSV file: {trials_csv_path}")
        print(f"🗂️  Vector store path: {vectorstore_path}")
        print(f"📦 Collection name: {args.collection_name}")
        print(f"🔍 Status filter: {args.status_filter}")

        vectorstore = db_manager.create_trial_vectorstore(
            trials_csv_path=str(trials_csv_path),
            vectorstore_path=str(vectorstore_path),
            collection_name=args.collection_name,
            status_filter=args.status_filter,
            vstore_delete=True,
        )

        print("\n✅ Trials vector store created successfully!")
        print(f"📁 Vector store location: {vectorstore_path}")
        print(f"📊 Collection count: {vectorstore._collection.count()} trials")

        # Display collection info
        print("\n📋 Collection information:")
        print(f"   Name: {args.collection_name}")
        print(f"   Documents: {vectorstore._collection.count()}")
        print(f"   Status filter: {args.status_filter}")
        print("   Embedding model: nomic-embed-text-v1.5")

        print("\n🎉 Trials vector store is ready for use!")

    except Exception as e:
        print(f"❌ Error creating trials vector store: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
