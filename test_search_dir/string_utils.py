"""
This module provides string manipulation utilities.
"""

def reverse_string(s: str) -> str:
    """Reverse the input string."""
    return s[::-1]

def count_vowels(s: str) -> int:
    """Count the number of vowels in a string."""
    vowels = 'aeiouAEIOU'
    return sum(1 for char in s if char in vowels)

class StringProcessor:
    """A class for processing strings with various operations."""
    
    def __init__(self, text: str = ""):
        self.text = text
    
    def to_uppercase(self) -> 'StringProcessor':
        """Convert the text to uppercase."""
        self.text = self.text.upper()
        return self
    
    def remove_whitespace(self) -> 'StringProcessor':
        """Remove all whitespace from the text."""
        self.text = ''.join(self.text.split())
        return self
    
    def get_result(self) -> str:
        """Get the processed text result."""
        return self.text
