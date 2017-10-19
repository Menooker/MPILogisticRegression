Logistic Regression By MPI & OpenMP
- How to compile
Run script "compile.sh", you will get two executable files "LR1" and "LR2". "LR1" is a fully synchronized Logistic Regression program. "LR2" is the Logistic Regression program using SSP protocol.

- How to run
The programs reads an input "CSV" dataset specified by the argument "path", and outputs the loss of the parameter vector learnt from the dataset. To run SSP version Logistic Regression program locally, run "run.sh". To run the program in the cluster, run command "sbatch mpiscript.sh".