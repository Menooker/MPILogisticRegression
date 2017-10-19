#!/bin/bash
#SBATCH --time=00:10:00
#SBATCH --nodes=4
#SBATCH --ntasks=4


time mpiexec -n 4 -display-allocation LR2 path gene2.csv num_param 22283 num_points 5896 iter_num 300 thread_num 2 step_size 0.005 test_partition 0.2