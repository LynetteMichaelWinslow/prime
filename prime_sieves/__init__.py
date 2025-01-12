from ._sieves import (
    f1, 
    f2,  
    f23, 
    f24, 
)

__all__ = [
    "get_primes",
]

def get_primes(n: int, method: str = "f1"):
    if n < 2:
        return []
    if method == "f1":
        return f1(n)
    elif method == "f2":
        return f2(n)
    elif method == "f23":
        return f23(n)
    elif method == "f24":
        return f24(n)
    else:
        raise ValueError(f"Unsupported method: {method}")
