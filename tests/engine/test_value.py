from engine import Value
import pytest

class TestValue:
    
    @pytest.fixture
    def sample_values(self):
        x, y = 2.5, 4
        return x, y
    
    def test_addition(self, sample_values):
        t1, t2 = sample_values
        x, y = Value(t1), Value(t2)
        test_sum = x + y
        assert isinstance(test_sum, Value)
        assert test_sum.data == t1 + t2
    
    def test_multiplication(self, sample_values):
        t1, t2 = sample_values
        x, y = Value(t1), Value(t2)
        test_mul = x * y
        assert isinstance(test_mul, Value)
        assert test_mul.data == t1 * t2
        
    def test_power(self, sample_values):
        t1, t2 = sample_values
        x, y = Value(t1), Value(t2)
        test_pow = x ** y
        assert isinstance(test_pow, Value)
        assert test_pow.data == t1 ** t2
        test_pow = y ** x
        assert test_pow.data == t2 ** t1