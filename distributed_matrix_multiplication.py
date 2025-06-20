from mpi4py import MPI
import numpy as np
import sys

def distributed_matrix_multiplication(matrix_a, matrix_b, comm):
    """
    Performs distributed matrix multiplication using MPI with row-wise
    distribution of matrix A and broadcasting of matrix B.

    Args:
        matrix_a: A NumPy array representing the first matrix (n x m).
                  Only needed on the root process (rank 0).
        matrix_b: A NumPy array representing the second matrix (m x p).
                  Only needed on the root process (rank 0).
        comm: The MPI communicator.

    Returns:
        A NumPy array representing the resulting matrix (n x p) on the
        root process (rank 0), and None on other processes.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get dimensions and dtype (only needed on root)
    if rank == 0:
        # Check if matrices are provided on the root process
        if matrix_a is None or matrix_b is None:
            print("Error: Input matrices are None on root process.")
            return None

        if matrix_a.shape[1] != matrix_b.shape[0]:
            print("Error: Incompatible matrix dimensions for multiplication.")
            return None
        n, m = matrix_a.shape
        p = matrix_b.shape[1]
        dtype = matrix_a.dtype # Get the dtype from the input matrix
        # Calculate rows per process
        rows_per_process = n // size
        remainder_rows = n % size
        sendcounts = [rows_per_process * m] * size # Send count is based on elements
        for i in range(remainder_rows):
            sendcounts[i] += m
        displs = [sum(sendcounts[:i]) for i in range(size)]
    else:
        n, m, p, dtype = None, None, None, None
        sendcounts = None
        displs = None

    # Broadcast dimensions and dtype to all processes
    n = comm.bcast(n, root=0)
    m = comm.bcast(m, root=0)
    p = comm.bcast(p, root=0)
    dtype = comm.bcast(dtype, root=0)
    sendcounts = comm.bcast(sendcounts, root=0)
    displs = comm.bcast(displs, root=0)

    if n is None: # Handle the error case from root (incompatible dimensions or None input)
        if rank == 0: # Print the specific error message before returning None
             print("Distributed matrix multiplication could not proceed due to error detected on root.")
        return None

    # Broadcast matrix B to all processes
    # Ensure matrix_b is not None before broadcasting on root
    if rank == 0 and matrix_b is None:
        print("Error: matrix_b is None on root before broadcast.")
        return None
    matrix_b = comm.bcast(matrix_b, root=0)


    # Scatter matrix A (row-wise)
    local_n = sendcounts[rank] // m # Calculate local rows from send count
    local_matrix_a = np.empty((local_n, m), dtype=dtype) # Use the broadcasted dtype
    # Ensure matrix_a is not None before Scatterv on root
    if rank == 0 and matrix_a is None:
         print("Error: matrix_a is None on root before Scatterv.")
         return None
    comm.Scatterv([matrix_a, sendcounts, displs, MPI._typedict[dtype.char]], local_matrix_a, root=0) # Use MPI type corresponding to dtype


    # Local computation
    local_result = np.dot(local_matrix_a, matrix_b)

    # Gather results
    recvcounts = [count // m * p for count in sendcounts] # Recv count is based on elements of the result
    recvdispls = [sum(recvcounts[:i]) for i in range(size)]
    result_matrix = None
    if rank == 0:
        result_matrix = np.empty((n, p), dtype=local_result.dtype) # Use the dtype of the local result
    comm.Gatherv(local_result, [result_matrix, recvcounts, recvdispls, MPI._typedict[local_result.dtype.char]], root=0) # Use MPI type corresponding to local result dtype

    return result_matrix

if __name__ == "__main__":
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if len(sys.argv) != 2:
            if rank == 0:
                print("Usage: mpirun -np <num_processes> python mpi_benchmark.py <matrix_size>")
            sys.exit(1)

        matrix_size = int(sys.argv[1])
        num_processes = comm.Get_size()

        A = None
        B = None

        if rank == 0:
            # Create matrices on the root process
            A = np.random.rand(matrix_size, matrix_size)
            B = np.random.rand(matrix_size, matrix_size)

        # Perform distributed matrix multiplication and measure time
        start_time = MPI.Wtime()
        C = distributed_matrix_multiplication(A, B, comm)
        end_time = MPI.Wtime()

        if C is not None and rank == 0:
            # Print results in a parsable format on the root process
            print(f"mpi,{matrix_size},{num_processes},{end_time - start_time:.4f}")
        elif C is None and rank == 0:
            # If multiplication was not possible, exit with a non-zero status on root
            # An error message should have been printed by the function before returning None
            sys.exit(1)