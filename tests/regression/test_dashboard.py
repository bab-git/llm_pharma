#!/usr/bin/env python3
"""
Dashboard Test Runner

A simple script to run dashboard tests easily.

Usage:
    python tests/test_dashboard.py                    # Run all tests
    python tests/test_dashboard.py --quick           # Quick test
    python tests/test_dashboard.py --manual          # Launch dashboard
    python tests/test_dashboard.py --demo-graph      # Test demo graph only
"""

import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_demo_graph_only():
    """Test just the demo graph functionality."""
    print("🧪 Testing demo graph only...")

    try:
        from frontend.demo_graph import create_demo_graph

        graph = create_demo_graph()
        if graph:
            print("✅ Demo graph test successful!")
            return True
        else:
            print("❌ Demo graph test failed!")
            return False

    except Exception as e:
        print(f"❌ Demo graph test error: {e}")
        return False


def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Dashboard Test Runner")
    parser.add_argument("--quick", action="store_true", help="Run quick test")
    parser.add_argument(
        "--manual", action="store_true", help="Launch dashboard for manual testing"
    )
    parser.add_argument(
        "--demo-graph", action="store_true", help="Test demo graph only"
    )

    args = parser.parse_args()

    if args.demo_graph:
        success = test_demo_graph_only()
        sys.exit(0 if success else 1)
    elif args.quick:
        from tests.integration.test_dashboard_demo import run_quick_test

        run_quick_test()
    elif args.manual:
        from tests.integration.test_dashboard_demo import run_manual_dashboard_test

        run_manual_dashboard_test()
    else:
        # Run full test suite
        from tests.integration.test_dashboard_demo import TestDashboardDemo

        print("🧪 Running full dashboard test suite...")
        test_instance = TestDashboardDemo()

        # Run all test methods
        test_methods = [
            method for method in dir(test_instance) if method.startswith("test_")
        ]

        passed = 0
        failed = 0

        for method_name in test_methods:
            print(f"\n{'='*50}")
            print(f"Running: {method_name}")
            print(f"{'='*50}")
            try:
                getattr(test_instance, method_name)()
                passed += 1
                print(f"✅ {method_name} PASSED")
            except Exception as e:
                failed += 1
                print(f"❌ {method_name} FAILED: {e}")

        print(f"\n{'='*50}")
        print(f"🎉 Test Results: {passed} passed, {failed} failed")
        print(f"{'='*50}")

        sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
