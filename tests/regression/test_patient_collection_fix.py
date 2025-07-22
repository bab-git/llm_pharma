#!/usr/bin/env python3
"""
Quick test to verify patient collection is working after import fixes.
"""

import os
import sys

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_patient_collection():
    """Test that patient collection works without import errors."""
    print("🧪 Testing Patient Collection Fix")
    print("=" * 40)

    try:
        # Test importing from patient_collector
        from my_agent.patient_collector import (
            AgentState,
            create_agent_state,
            patient_collector_node,
        )

        print("✅ Successfully imported from patient_collector")

        # Test creating agent state
        state = create_agent_state()
        state["patient_prompt"] = "I need information about patient 1"
        print("✅ Successfully created agent state")

        # Test that the functions are the same
        if patient_collector_node == hf_patient_collector_node:
            print("✅ patient_collector_node functions are identical")
        else:
            print("❌ patient_collector_node functions are different")

        if create_agent_state == hf_create_agent_state:
            print("✅ create_agent_state functions are identical")
        else:
            print("❌ create_agent_state functions are different")

        print("\n" + "=" * 40)
        print("✅ Patient collection import fix successful!")

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

    return True


if __name__ == "__main__":
    success = test_patient_collection()
    sys.exit(0 if success else 1)
