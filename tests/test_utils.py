import pytest
from app import PhysicsMetricsCalculator

class TestUtils:
    def test_find_equations(self):
        """Test finding equations in text"""
        calculator = PhysicsMetricsCalculator()
        text = "F = ma"
        equations = calculator._find_equations(text)
        assert len(equations) > 0, "No equations found, but expected at least one."
