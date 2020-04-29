#!/bin/sh

# Lanza al sistema el ejemplo basico de ejecucion
# Tan solo devuelve el resultado, no tiene ninguna flag

#PBS -l nodes=4:ppn=32,walltime=00:10:00
#PBS -q cpa
#PBS -d .
#PBS -W x="NACCESSPOLICY:SINGLEJOB"
OMP_NUM_THREADS=1 mpiexec ./bin/main -r 13000 13000 13000 0 1
