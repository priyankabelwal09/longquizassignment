import subprocess
import csv

matrix_sizes = [256, 512, 1024]
process_counts = [1, 2, 4, 8]

#matrix_sizes = [256]
#process_counts = [1]
with open("results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["method", "matrix_size", "num_processes", "time_sec"])

    # Run serial benchmarks
    for N in matrix_sizes:
        output = subprocess.check_output(["python", "standard_matrix_multiplication.py", str(N)]).decode().strip()
        writer.writerow(output.split(","))

    # Run MPI benchmarks
    for N in matrix_sizes:
        for p in process_counts:
            output = subprocess.check_output(["mpiexec", "-np", str(p), "python", "distributed_matrix_multiplication.py", str(N)]).decode().strip()
            #output = subprocess.check_output(["mpiexec", "--allow-run-as-root", "--oversubscribe", "-np", str(p), "python", "distributed_matrix_multiplication.py", str(N)]).decode().strip()

            writer.writerow(output.split(","))

print("Benchmarking complete. Results saved to results.csv.")