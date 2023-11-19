import time

def fib_recursive(n):
    return n if n <= 1 else fib_recursive(n - 1) + fib_recursive(n - 2)

def fib_non_recursive(n):
    fib_values = [0, 1] + [0] * (n - 1)
    for i in range(2, n + 1):
        fib_values[i] = fib_values[i - 1] + fib_values[i - 2]
    return fib_values[n]

def test_fib(func, n):
    result, elapsed_time = 0, 0
    if func.__name__ == 'fib_recursive':
        start_time = time.time()
        result = func(n)
        elapsed_time = time.time() - start_time
    else:
        start_time = time.time()
        result = func(n)
        elapsed_time = time.time() - start_time

    print(f"Calculating the {n}th Fibonacci number using {func.__name__}:")
    print(f"Fibonacci number: {result}")
    print(f"Time taken: {elapsed_time:.6f} seconds\n")

# Test functions with sample input
n = int(input("Enter the value of n: "))
test_fib(fib_recursive, n)
test_fib(fib_non_recursive, n)
