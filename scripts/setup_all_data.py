#!/usr/bin/env python3
"""
Master script to set up all data for the LLM Pharma system.

This script runs all three data creation scripts in sequence:
1. Create patients database
2. Create policies vector store
3. Create trials data and vector store

Usage:
    python scripts/setup_all_data.py [--force-recreate] [--skip-patients] [--skip-policies] [--skip-trials]

Options:
    --force-recreate    Force recreation of all databases and vector stores
    --skip-patients     Skip patients database creation
    --skip-policies     Skip policies vector store creation
    --skip-trials       Skip trials data and vector store creation
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_script(script_name, args=None, config_path=None, config_name=None):
    """Run a script and return success status."""
    script_path = Path(__file__).parent / script_name

    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)
    # Add config path/name if provided
    if config_path:
        cmd.extend(["--config-path", config_path])
    if config_name:
        cmd.extend(["--config-name", config_name])

    print(f"\n🚀 Running {script_name}...")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {script_name} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {script_name} failed with exit code {e.returncode}")
        return False


def main():
    """Main function to set up all data."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Set up all data for LLM Pharma system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/setup_all_data.py
    python scripts/setup_all_data.py --force-recreate
    python scripts/setup_all_data.py --skip-trials
    python scripts/setup_all_data.py --skip-patients --skip-policies
        """,
    )

    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Force recreation of all databases and vector stores",
    )

    parser.add_argument(
        "--skip-patients", action="store_true", help="Skip patients database creation"
    )

    parser.add_argument(
        "--skip-policies",
        action="store_true",
        help="Skip policies vector store creation",
    )

    parser.add_argument(
        "--skip-trials",
        action="store_true",
        help="Skip trials data and vector store creation",
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

    print("🏥 LLM Pharma - Complete Data Setup")
    print("=" * 50)
    print("This script will set up all required data for the LLM Pharma system:")
    print("1. Patients database (SQLite)")
    print("2. Policies vector store (ChromaDB)")
    print("3. Trials data and vector store (ChromaDB)")
    print()

    # Prepare common arguments
    common_args = []
    if args.force_recreate:
        common_args.append("--force-recreate")

    # Track success/failure
    results = {}

    # Step 1: Create patients database
    if not args.skip_patients:
        results["patients"] = run_script(
            "create_patients_database.py",
            common_args,
            args.config_path,
            args.config_name,
        )
    else:
        print("⏭️  Skipping patients database creation")
        results["patients"] = True

    # Step 2: Create policies vector store
    if not args.skip_policies:
        results["policies"] = run_script(
            "create_policies_vectorstore.py",
            common_args,
            args.config_path,
            args.config_name,
        )
    else:
        print("⏭️  Skipping policies vector store creation")
        results["policies"] = True

    # Step 3: Create trials data and vector store
    if not args.skip_trials:
        results["trials"] = run_script(
            "create_trials_vectorstore.py",
            common_args,
            args.config_path,
            args.config_name,
        )
    else:
        print("⏭️  Skipping trials data and vector store creation")
        results["trials"] = True

    # Summary
    print("\n" + "=" * 50)
    print("📋 Setup Summary:")
    print(f"✅ Patients Database: {'PASS' if results['patients'] else 'FAIL'}")
    print(f"✅ Policies Vector Store: {'PASS' if results['policies'] else 'FAIL'}")
    print(f"✅ Trials Vector Store: {'PASS' if results['trials'] else 'FAIL'}")

    all_success = all(results.values())

    if all_success:
        print("\n🎉 All data setup completed successfully!")
        print("The LLM Pharma system is ready to use!")

        # Display next steps
        print("\n📝 Next Steps:")
        print("1. Set up your environment variables (GROQ_API_KEY, OPENAI_API_KEY)")
        print("2. Run the test scripts to verify functionality:")
        print("   python backend/test_patient_collector.py")
        print("   python backend/test_policy_evaluator.py")
        print("3. Start the frontend application:")
        print("   python frontend/app.py")
    else:
        print("\n❌ Some setup steps failed. Please check the error messages above.")
        failed_steps = [step for step, success in results.items() if not success]
        print(f"Failed steps: {', '.join(failed_steps)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
