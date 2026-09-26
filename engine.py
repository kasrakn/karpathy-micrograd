"""Value class
This class is the data structure for storing variables
of the code

Returns:
    Value: A new object with type Value
    
"""

from __future__ import annotations


class Value:
    
    def __init__(self, data: float):
        self.data = data
    
    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other: Value) -> Value:
        return Value(self.data + other.data)
    
    def __mul__(self, other: Value) -> Value:
        return Value(self.data * other.data)
    
    def __pow__(self, other: Value) -> Value:
        return Value(self.data ** other.data)