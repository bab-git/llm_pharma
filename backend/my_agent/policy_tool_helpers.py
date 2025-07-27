"""
Policy Tool Helpers

This module contains the tool definitions and Pydantic models used for policy evaluation
in the clinical trial workflow. These tools are extracted from policy_service.py to
improve modularity and avoid recreating them on each function call.
"""

from datetime import datetime

from langchain_core.tools import tool
from pydantic import BaseModel, Field


class DateInput(BaseModel):
    """Input schema for date comparison tool."""

    past_date: str = Field(description="A past date in YYYY-MM-DD format")
    threshold_months: int = Field(description="Number of months to compare against")


class NumberInput(BaseModel):
    """Input schema for number comparison tool."""

    num1: float = Field(description="First number")
    num2: float = Field(description="Second number")


@tool("get_today_date", return_direct=False)
def get_today_date() -> str:
    """Returns today's date in YYYY-MM-DD format."""
    return datetime.today().date().strftime("%Y-%m-%d")


@tool("check_months_since_date", args_schema=DateInput, return_direct=False)
def check_months_since_date(past_date: str, threshold_months: int) -> str:
    """Calculate months between a past date and today, and check if within threshold."""
    try:
        today = datetime.today().date()
        parsed_date = datetime.strptime(past_date, "%Y-%m-%d").date()
        months_diff = (
            (today.year - parsed_date.year) * 12 + today.month - parsed_date.month
        )
        is_within_threshold = months_diff <= threshold_months
        return f"Months since {past_date}: {months_diff}. Within {threshold_months} months: {is_within_threshold}"
    except ValueError:
        return f"Invalid date format: {past_date}. Please use YYYY-MM-DD."


@tool("compare_numbers", args_schema=NumberInput, return_direct=False)
def compare_numbers(num1: float, num2: float) -> str:
    """Compare if first number is less than second number."""
    result = num1 < num2
    return f"Is {num1} less than {num2}? {result}"


# Export the tools as a list for easy import
POLICY_TOOLS = [get_today_date, check_months_since_date, compare_numbers]
