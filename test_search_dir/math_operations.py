"""
This module provides basic mathematical operations.
"""

def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

class Calculator:
    """A simple calculator class."""
    
    def __init__(self, initial_value: float = 0):
        self.value = initial_value
    
    def add(self, x: float) -> 'Calculator':
        """Add a number to the current value."""
        self.value += x
        return self
    
    def multiply(self, x: float) -> 'Calculator':
        """Multiply the current value by a number."""
        self.value *= x
        return self
    
    def get_value(self) -> float:
        """Get the current value."""
        return self.value
