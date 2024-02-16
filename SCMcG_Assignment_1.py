#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

### Steven Connan-McGinty - Assignment 1

# To calculate pi we will be starting with evaluating the following integral
def integrand(x):
    return 4.0 / (1.0 + x*x)

# We definite the midpoint rule for numerical integration which is dividing this into smaller subintervals for evaluation
def midpoint_rule(a, b, N):
    delta_x = (b - a) / N
    integral = 0.0
    for i in range(N):
        x_mid = a + (i + 0.5) * delta_x
        integral += integrand(x_mid)
    integral *= delta_x
    return integral

# To attempt parallel speed-up I have attempted to balance the load per worked based on remaining 
if rank == 0:
    # Leader process
    if size <= 1:
        print("Error: Not enough processes for parallel computation.")
    else:
        total_points = 1000000
        workload_per_worker = total_points // (size - 1)
        remaining_work = total_points % (size - 1)

        # Distribute initial workload
        for dest in range(1, size):
            num_points = workload_per_worker
            if remaining_work > 0:
                num_points += 1
                remaining_work -= 1
            comm.send(num_points, dest=dest)
        
        # Receive partial results and accumulate
        total_integral = 0.0
        for _ in range(1, size):
            partial_integral = comm.recv(source=MPI.ANY_SOURCE)
            total_integral += partial_integral
        
        print("Estimated value of pi:", total_integral)

else:
    # Worker processes
    while True:
        # Receive workload from leader
        num_points = comm.recv(source=0)
        if num_points == 0:
            # Terminate if no more work
            break
        
        # Perform numerical integration using the midpoint rule
        a = rank * (1.0 / size)
        b = (rank + 1) * (1.0 / size)
        partial_integral = midpoint_rule(a, b, num_points)
        
        # Send partial result back to leader
        comm.send(partial_integral, dest=0)
