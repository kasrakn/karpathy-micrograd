from __future__ import annotations

import math

"""Value class
This class is the data structure for storing variables
of the code

Returns:
    Value: A new object with type Value
    
"""

class Value:
    
    def __init__(self, data: float, name='', _prev=[], tag=None, _op=''):
        self.data = data
        self.grad = 0.0
        self.name = name
        self.tag = tag if tag else self.name
        self._op = _op
        self._prev = _prev
    
    def __repr__(self):
        return f"Value({self.name} | data={self.data}) | grad={self.grad} | tag={self.tag} | op={self._op} | children={[c.name for c in self._prev] if self._prev else []}"
    
    def __add__(self, other: Value) -> Value:
        self.grad += 1.0
        other.grad += 1.0
        return Value(self.data + other.data, tag=f"{self.name}+{other.name}", _op='+', _prev=[self, other])
    
    def __mul__(self, other: Value) -> Value:
        self.grad = other.data
        other.grad = self.data
        return Value(self.data * other.data, tag=f"{self.name}.{other.name}", _op='*', _prev=[self, other])
    
    def __pow__(self, other: Value) -> Value:
        return Value(self.data ** other.data, tag=f"{self.name}**{other.name}", _op='**', _prev=[self, other])
    
    def tanh(self):
        n = self.data
        out = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        self.grad = 1 - out**2
        return Value(out, tag=f"tanh({self.name})", _op="tanh", _prev=[self])
    
    def sigmoid(self):
        n = self.data
        out = 1 / (1 + math.exp(-n))
        self.grad = out * (1 - out)
        return Value(out, tag=f"sig({self.name})", _op="sig", _prev=[self])
    
    def relu(self):
        n = self.data
        if n > 0:
            out = n
            self.grad = 1
        elif n < 0:
            out = 0.0
            self.grad = 0
        else:
            out = 0.0
            self.grad = 0.5
        return Value(out, tag=f"relu({self.data})", _op="relu", _prev=[self])
            
    # backpropagation - apply chain-rule
    def backward(self): 
        for c in self._prev:
            c.grad = self.grad * c.grad
            c.backward()