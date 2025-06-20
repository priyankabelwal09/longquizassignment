import time
import sys
import numpy as np

# function for standard matrix multiplication.
"""
Function returns:
    A NumPy array representing the resulting matrix (n x p).
"""
def standard_matrix_multiplication(matrix_a, matrix_b):

  # Check for compatible dimensions
  if matrix_a.shape[1] != matrix_b.shape[0]:
    print("Error: Incompatible matrix dimensions for multiplication.")
    return None

  n = matrix_a.shape[0]
  m = matrix_a.shape[1]
  p = matrix_b.shape[1]

  # Initialize the result matrix with zeros
  result_matrix = np.zeros((n, p))

  # Perform the matrix multiplication
  for i in range(n):
    for j in range(p):
      for k in range(m):
        result_matrix[i, j] += matrix_a[i, k] * matrix_b[k, j]

  return result_matrix

if __name__ == "__main__":
        if len(sys.argv) != 2:
            print("Usage: python serial_benchmark.py <matrix_size>")
            sys.exit(1)

        matrix_size = int(sys.argv[1])

        # Create matrices for benchmarking
        A = np.random.rand(matrix_size, matrix_size)
        B = np.random.rand(matrix_size, matrix_size)

        # Perform serial matrix multiplication and measure time
        start_time = time.time()
        C = standard_matrix_multiplication(A, B)
        end_time = time.time()

        # Print results in a parsable format
        print(f"serial,{matrix_size},1,{end_time - start_time:.4f}")