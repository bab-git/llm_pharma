#!/usr/bin/env python3
"""
Script to create the patients database for the LLM Pharma system.

This script creates a demo patient database with randomly generated patient data
including medical history, trial participation, and demographics.

Usage:
    python scripts/create_patients_database.py [--db-path PATH] [--force-recreate]

Options:
    --db-path PATH        Path where the database file will be created (default: sql_server/patients.db)
    --force-recreate      Force recreation of the database even if it exists
"""

import os
import sys
import argparse
from pathlib import Path

# Add the parent directory to the path to import helper_functions
sys.path.append(str(Path(__file__).parent.parent))

from backend.my_agent.database_manager import DatabaseManager

def main():
    """Main function to create the patients database."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Create patients database for LLM Pharma system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/create_patients_database.py
    python scripts/create_patients_database.py --db-path data/patients.db
    python scripts/create_patients_database.py --force-recreate
        """
    )
    
    parser.add_argument(
        "--db-path",
        default="sql_server/patients.db",
        help="Path where the database file will be created (default: sql_server/patients.db)"
    )
    
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of the database even if it exists"
    )
    
    args = parser.parse_args()
    
    # Convert to absolute path if relative
    if not os.path.isabs(args.db_path):
        project_root = Path(__file__).parent.parent
        db_path = project_root / args.db_path
    else:
        db_path = Path(args.db_path)
    
    print("🏥 LLM Pharma - Patients Database Creator")
    print("=" * 50)
    
    # Check if database already exists
    if db_path.exists() and not args.force_recreate:
        print(f"⚠️  Database already exists at: {db_path}")
        print("Use --force-recreate to overwrite the existing database")
        return
    
    try:
        # Create the database
        print(f"📊 Creating patients database at: {db_path}")
        db_manager = DatabaseManager()
        df = db_manager.create_demo_patient_database(str(db_path))
        
        print("\n✅ Database creation completed successfully!")
        print(f"📁 Database location: {db_path}")
        print(f"📁 CSV export location: {db_path.with_suffix('.csv')}")
        print(f"👥 Total patients created: {len(df)}")
        
        # Display sample data
        print("\n📋 Sample patient data:")
        print(df.head(3).to_string(index=False))
        
        print("\n🎉 Patients database is ready for use!")
        
    except Exception as e:
        print(f"❌ Error creating patients database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 