#!/bin/sh

# Lanza al sistema el ejemplo basico de ejecucion
# Tan solo devuelve el resultado, no tiene ninguna flag

#PBS -l nodes=4:ppn=4,walltime=01:00:00
#PBS -q lpp
#PBS -d .
#PBS -W x="NACCESSPOLICY:SINGLEJOB"
export OMP_NUM_THREADS=8
mpiexec ../../bin/main -r 30000 30000 30000 0 1