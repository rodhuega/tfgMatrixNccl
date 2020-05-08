#!/bin/sh

# Lanza al sistema el ejemplo basico de ejecucion
# Tan solo devuelve el resultado, no tiene ninguna flag

#PBS -l nodes=4:ppn=1,walltime=00:10:00
#PBS -q cpa
#PBS -d .
#PBS -W x="NACCESSPOLICY:SINGLEJOB"
export OMP_NUM_THREADS=32
mpiexec ../../bin/main -r 20000 20000 20000 0 1