"""Value class
This class is the data structure for storing variables
of the code

Returns:
    Value: A new object with type Value
    
"""

from __future__ import annotations


class Value:
    
    def __init__(self, data: float, name='', children=[], tag=None, operator=None):
        self.data = data
        self.name = name
        self.tag = tag if tag else self.name
        self._op = operator
        self._prev = children
    
    def __repr__(self):
        return f"Value({self.name} | data={self.data}) | tag={self.tag} | op={self._op} | children={[c.name for c in self._prev] if self._prev else []}"
    
    def __add__(self, other: Value) -> Value:
        return Value(self.data + other.data, tag=f"{self.name}+{other.name}", operator='+', children=[self, other])
    
    def __mul__(self, other: Value) -> Value:
        return Value(self.data * other.data, tag=f"{self.name}.{other.name}", operator='*', children=[self, other])
    
    def __pow__(self, other: Value) -> Value:
        return Value(self.data ** other.data, tag=f"{self.name}**{other.name}", operator='**', children=[self, other])