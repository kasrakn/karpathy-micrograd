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
        self._backward = lambda: None
    
    def __repr__(self):
        return f"Value({self.name} | data={self.data}) | grad={self.grad} | tag={self.tag} | op={self._op} | children={[c.name for c in self._prev] if self._prev else []}"
    
    def __neg__(self):
        return self * -1
    
    def __add__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, tag=f"{self.name}+{other.name}", _op='+', _prev=[self, other])
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out
    
    def __sub__(self, other):
        return self + (-other)
    
    
    def __mul__(self, other: Value) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, tag=f"{self.name}.{other.name}", _op='*', _prev=[self, other])
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out
    
    def __rmul__(self, other):
        return self * other
    
        
    def __truediv__(self, other):
        return self * other**-1
    
    
    def __pow__(self, other) -> Value:
        if not isinstance(other, (int, float)):
            raise TypeError("The power must be constant.")
        
        out = Value(self.data ** other, tag=f"{self.name}**{other}", _op='**', _prev=[self, other])
        
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward
        
        return out

    
    def exp(self):
        n = self.data
        out = Value(math.exp(n), tag=f"exp({self.name})", _op="exp", _prev=[self])
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = self.backward
    
        return out
    
    
    def tanh(self):
        n = self.data
        t = (math.exp(2*n) - 1) / (math.exp(2*n) + 1)
        out = Value(t, tag=f"tanh({self.name})", _op="tanh", _prev=[self])
        
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        
        return out
    
    
    def sigmoid(self):
        n = self.data
        s = 1 / (1 + math.exp(-n))
        out = Value(s, tag=f"sig({self.name})", _op="sig",_prev=[self])
        
        def _backward():
            self.grad += out.data * (1 - out.data) * out.grad
        out._backward = _backward
        
        return Value(out, tag=f"sig({self.name})", _op="sig", _prev=[self])
    
    
    def relu(self):
        n = self.data
        out = Value(0.0, tag=f"relu({self.data})", _op="relu", _prev=[self])
        if n > 0:
            out.data = n
            self.grad = 1
        elif n < 0:
            out.data = 0.0
            self.grad = 0
        else:
            out.data = 0.0
            self.grad = 0.5
            
        def _backward():
            if n > 0:
                self.grad += 1.0 * out.grad
            elif n < 0:
                self.grad += 0.0
            else:
                self.grad += 0.5 * out.grad
        out._backward = _backward
        
        return out
    
    
    def backward(self):
        topo = []
        visited = set()
        
        def build_topo(v: Value):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
        