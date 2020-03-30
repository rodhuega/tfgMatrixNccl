#ifndef MatrixUtilitiesCuda_H
#define MatrixUtilitiesCuda_H

#include <stdio.h>

#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "nccl.h"


#define IDX2C(i,j,ld) (((j)*(ld))+(i))

class MatrixUtilitiesCuda
{
public:

static void cudaPrintMatrixCall(int rows,int columns,double* matrix);
};
#endif