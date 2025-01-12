# prime

A blazing-fast prime-sieving library built with [Numba](https://numba.pydata.org/) and [NumPy](https://numpy.org/). This package provides multiple methods for generating prime numbers up to very large ranges, with an emphasis on speed and flexibility.

## Key Features

- **Multiple Sieve Implementations**  
  Includes single-threaded and parallel versions of both the classic Eratosthenes sieve and the Euler sieve.

- **Numba-Powered**  
  Uses Numba’s `njit` (No-Python mode) and parallelization (`prange`), resulting in highly optimized and efficient code.

- **Flexible API**  
  Functions can be called directly (e.g. `f1`, `f2`, `f23`, `f24`) or through the simplified `get_primes(n, method="...")` interface.

- **Highly Scalable**  
  Capable of handling large values of $begin:math:text$ n $end:math:text$. For instance, on a MacBook Air M2 (8 GiB RAM), **ignoring warm-up (JIT compilation) time**, this library can sieve all primes up to $begin:math:text$10^{10}$end:math:text$ in as little as **3.2 seconds**.

## Installation

1. Clone or download this repository.
2. Ensure you have Python 3.7+ installed.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Make sure your system/compiler supports OpenMP if you wish to leverage parallel functionality.

## Project Structure

```
prime
├── prime_sieves
│   ├── __init__.py
│   └── _sieves.py
├── README.md
└── requirements.txt
```

- **`_sieves.py`**  
  Contains the core implementations of several sieving functions (`f1`, `f2`, `f3`, …, `f24`), many of which are `@njit`-decorated for performance.

- **`__init__.py`**  
  Exposes a simplified API (`get_primes`) for fetching prime numbers up to `n`, using a specified method.

- **`README.md`**  
  The file you’re reading now.

- **`requirements.txt`**  
  Lists the Python dependencies needed to run this library.

## Usage

### 1. Simple Prime Retrieval

After installing, you can directly import the package and call the high-level function `get_primes`:

```python
from prime_sieves import get_primes

# Get all primes <= 100
primes_up_to_100 = get_primes(100, method="f1")
print(primes_up_to_100)
```

By default, `method="f1"` uses a single-threaded Eratosthenes sieve. Other options include `"f2"`, `"f23"`, and `"f24"` (parallel or Euler-based sieves).

### 2. Command-Line Interface (Example)

A minimal CLI usage is demonstrated at the bottom of `_sieves.py` via:

```python
if __name__ == "__main__":
    # Interactive mode:
    # 1) Prompts the user for N
    # 2) Asks which sieve method to use
    # 3) Times and prints out results
```

To run it directly:
```bash
python -m prime_sieves._sieves
```
Then follow the prompts.

### 3. Performance Example

On a MacBook Air M2 (8 GiB RAM), with OpenMP-enabled Numba (and ignoring initial JIT warm-up times), this library can:

- **Sieve all primes up to $begin:math:text$10^{10}$end:math:text$ in about 3.2 seconds (best-case scenario).**

Please note:
- Actual runtime can vary based on system load, Python/Numba versions, and environment specifics.
- The first run includes overhead for JIT compilation. Subsequent runs on the same process are faster because the code is already compiled.

## Available Methods

Internally, `_sieves.py` provides a variety of sieving functions, some single-threaded, some parallel:

- **`f1`**  
  Classic single-threaded Eratosthenes sieve using bit manipulation.

- **`f2`**  
  Single-threaded Euler sieve (also known as the sieve of Euler).

- **`f3`, `f4`, `f5`**  
  Internal helper functions for segmented sieving.

- **`f6`, `f7`, `f8`, `f9`**  
  Segmented versions of Eratosthenes/Euler, single-threaded and parallel.

- **`f23`, `f24`**  
  Parallel implementations (e.g. parallel segmented sieves).

and more.

## API Reference

A simplified interface is provided through:

```python
def get_primes(n: int, method: str = "f1"):
    """
    Return a NumPy array of primes up to n using one of the available methods.
    Available methods: 'f1', 'f2', 'f23', 'f24'.

    :param n: Upper bound for prime generation. Must be >= 2.
    :param method: Which algorithm to use. Defaults to 'f1'.
    :return: A NumPy array of primes up to n.
    """
    ...
```

**Parameters:**

- `n (int)`  
  Maximum integer to test for primality.
- `method (str)`  
  Selects the sieving algorithm. Supported values are `"f1"`, `"f2"`, `"f23"`, or `"f24"`.

**Returns:**

- A `numpy.ndarray` containing all prime numbers up to `n`.

## Contributing

We welcome contributions in the form of bug reports, feature requests, or pull requests. To contribute:

1. Fork this repository.
2. Create a new branch for your feature or bugfix.
3. Make your changes, and add tests if possible.
4. Submit a pull request with a clear description of your changes.

## License

This project is released under an open license (MIT or similar). Refer to the repository’s license file (if provided) for more details.

---

**Enjoy fast prime sieving!** If you find this project helpful, feel free to give it a star and contribute your improvements.
