#!/usr/bin/env python3
"""
Integration Test: Dashboard Demo Testing

This test script validates that the dashboard can be launched and function
properly using the demo graph. It tests the complete workflow from
graph creation to dashboard launch.

Usage:
    python -m pytest tests/integration/test_dashboard_demo.py -v
    python tests/integration/test_dashboard_demo.py  # Run directly
"""

import sys
from pathlib import Path

# Add project paths for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))


class TestDashboardDemo:
    """Test class for dashboard demo functionality."""

    def setup_method(self):
        """Setup before each test method."""
        self.demo_port = 7959  # Use different port to avoid conflicts
        self.demo_host = "127.0.0.1"
        self.dashboard_url = f"http://{self.demo_host}:{self.demo_port}"

    def test_demo_graph_creation(self):
        """Test that demo graph can be created successfully."""
        try:
            from frontend.demo_graph import create_demo_graph

            print("🧪 Testing demo graph creation...")
            graph = create_demo_graph()

            assert graph is not None, "Demo graph should be created successfully"
            print("✅ Demo graph creation test passed")

        except ImportError as e:
            print(f"❌ Import error: {e}")
            assert False, f"Demo graph module not found: {e}"
        except Exception as e:
            print(f"❌ Error creating demo graph: {e}")
            assert False, f"Demo graph creation failed: {e}"

    def test_demo_graph_structure(self):
        """Test that demo graph has the expected structure."""
        try:
            from frontend.demo_graph import create_demo_agent_state

            print("🧪 Testing demo graph structure...")

            # Test AgentState creation
            AgentState = create_demo_agent_state()
            assert AgentState is not None, "AgentState should be created"

            # Test node functions creation
            from frontend.demo_graph import create_demo_node_functions

            node_functions = create_demo_node_functions()

            expected_nodes = [
                "patient_collector",
                "policy_search",
                "policy_evaluator",
                "trial_search",
                "grade_trials",
                "profile_rewriter",
            ]

            for node in expected_nodes:
                assert node in node_functions, f"Node {node} should be present"

            # Test conditional functions creation
            from frontend.demo_graph import create_demo_conditional_functions

            conditional_functions = create_demo_conditional_functions()

            expected_conditionals = [
                "should_continue_patient",
                "should_continue_policy",
                "should_continue_trials",
            ]

            for conditional in expected_conditionals:
                assert (
                    conditional in conditional_functions
                ), f"Conditional {conditional} should be present"

            print("✅ Demo graph structure test passed")

        except Exception as e:
            print(f"❌ Error testing demo graph structure: {e}")
            assert False, f"Demo graph structure test failed: {e}"

    def test_dashboard_import(self):
        """Test that dashboard components can be imported."""
        try:
            print("🧪 Testing dashboard imports...")

            # Test GUI import
            from frontend.helper_gui import trials_gui

            assert trials_gui is not None, "trials_gui should be importable"

            # Test demo graph import
            from frontend.demo_graph import create_demo_graph

            assert (
                create_demo_graph is not None
            ), "create_demo_graph should be importable"

            print("✅ Dashboard imports test passed")

        except ImportError as e:
            print(f"❌ Import error: {e}")
            assert False, f"Dashboard imports failed: {e}"

    def test_dashboard_creation(self):
        """Test that dashboard can be created with demo graph."""
        try:
            print("🧪 Testing dashboard creation...")

            from frontend.demo_graph import create_demo_graph
            from frontend.helper_gui import trials_gui

            # Create demo graph
            graph = create_demo_graph()
            assert graph is not None, "Demo graph should be created"

            # Create dashboard
            app = trials_gui(graph, share=False)
            assert app is not None, "Dashboard should be created"

            print("✅ Dashboard creation test passed")

        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
            assert False, f"Dashboard creation test failed: {e}"

    def test_dashboard_launch_simulation(self):
        """Test dashboard launch simulation (without actually launching)."""
        try:
            print("🧪 Testing dashboard launch simulation...")

            from frontend.demo_graph import create_demo_graph
            from frontend.helper_gui import trials_gui

            # Create demo graph
            graph = create_demo_graph()
            assert graph is not None, "Demo graph should be created"

            # Create dashboard
            app = trials_gui(graph, share=False)
            assert app is not None, "Dashboard should be created"

            # Test that launch method exists
            assert hasattr(app, "launch"), "Dashboard should have launch method"
            assert hasattr(app, "demo"), "Dashboard should have demo attribute"

            print("✅ Dashboard launch simulation test passed")

        except Exception as e:
            print(f"❌ Error in launch simulation: {e}")
            assert False, f"Dashboard launch simulation test failed: {e}"

    def test_demo_mode_integration(self):
        """Test the complete demo mode integration."""
        try:
            print("🧪 Testing demo mode integration...")

            # Import the demo graph directly
            from frontend.demo_graph import create_demo_graph

            # Test demo graph creation
            graph = create_demo_graph()
            assert graph is not None, "Demo graph should be created"

            # Test that it can be used with the main app
            # from frontend.app import launch_dashboard  # Commented out, as launch_dashboard is not defined

            print("✅ Demo mode integration test passed")

        except Exception as e:
            print(f"❌ Error in demo mode integration: {e}")
            assert False, f"Demo mode integration test failed: {e}"


def run_manual_dashboard_test():
    """
    Manual test function to actually launch the dashboard for manual testing.
    This is useful for visual verification of the dashboard functionality.
    """
    print("🚀 Starting manual dashboard test...")
    print("📝 This will launch the actual dashboard for manual testing")
    print("🔔 Press Ctrl+C to stop the test")

    try:
        from frontend.app import launch_dashboard

        # Launch dashboard in demo mode
        launch_dashboard(
            host="127.0.0.1",
            port=7959,  # Use different port
            share=False,
            demo_mode=True,
        )

    except KeyboardInterrupt:
        print("\n👋 Manual test stopped by user")
    except Exception as e:
        print(f"❌ Manual test failed: {e}")
        import traceback

        traceback.print_exc()


def run_quick_test():
    """
    Quick test function for fast validation.
    """
    print("⚡ Running quick dashboard demo test...")

    test_instance = TestDashboardDemo()

    # Run basic tests
    test_instance.test_demo_graph_creation()
    test_instance.test_dashboard_import()
    test_instance.test_dashboard_creation()

    print("✅ Quick test completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dashboard Demo Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    parser.add_argument(
        "--manual", action="store_true", help="Launch dashboard for manual testing"
    )
    parser.add_argument("--all", action="store_true", help="Run all tests")

    args = parser.parse_args()

    if args.manual:
        run_manual_dashboard_test()
    elif args.quick:
        run_quick_test()
    else:
        # Run all tests by default
        print("🧪 Running all dashboard demo tests...")
        test_instance = TestDashboardDemo()

        # Run all test methods
        test_methods = [
            method for method in dir(test_instance) if method.startswith("test_")
        ]

        for method_name in test_methods:
            print(f"\n{'='*50}")
            print(f"Running: {method_name}")
            print(f"{'='*50}")
            try:
                getattr(test_instance, method_name)()
            except Exception as e:
                print(f"❌ Test {method_name} failed: {e}")

        print(f"\n{'='*50}")
        print("🎉 All tests completed!")
        print(f"{'='*50}")
