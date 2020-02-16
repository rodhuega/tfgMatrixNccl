#include "MatrixUtilities.h"

using namespace std;

void MatrixUtilities::printMatrix(int rows, int columns, double *M)
{
    int i, j, matrixIndex;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            matrixIndex = matrixCalculateIndex(columns, i, j);
            cout << M[matrixIndex] << "\t";
        }
        cout << endl;
    }
}

void MatrixUtilities::printMatrixOrMessageForOneCpu(int rows, int columns, double *M, int cpuRank, int cpuRankPrint, string message)
{
    if (cpuRank == cpuRankPrint)
    {
        cout << message << endl;
        if (M != NULL)
        {
            printMatrix(rows, columns, M);
        }
    }
}

void MatrixUtilities::debugMatrixDifferentCpus(int cpuRank, int rows, int columns, double *M, string extraMessage)
{
    usleep(cpuRank * 1000);
    cout << "Parte del proceso: " << cpuRank << " " << extraMessage << endl;
    MatrixUtilities::printMatrix(rows, columns, M);
}

bool MatrixUtilities::canMultiply(int columnsA, int rowsB)
{
    return columnsA == rowsB;
}

OperationProperties MatrixUtilities::getMeshAndMatrixSize(int rowsA, int columnsA, int rowsB, int columnsB, int cpuSize)
{
    OperationProperties res;
    //Caso de matriz cuadrada y que entre perfectamente en la malla de misma division de filas y columnas
    if (rowsA == rowsB)
    {
        int meshBothDimensionsSize = sqrt(cpuSize);
        res.meshColumnSize = meshBothDimensionsSize;
        res.meshRowSize = meshBothDimensionsSize;
        res.rowsA = rowsA;
        res.columnsAorRowsB = columnsA;
        res.columnsB = columnsB;
    }
    else
    {
        //Se calculan todas las posibilidadades y se selecciona la que mas cpus use y menos 0 contenga de esas opciones
        int i,numberOfZerosA, numberOfZerosB;
        vector<OperationProperties> allOp;
        for (i = 2; i < cpuSize - 1; i++)
        {
            allOp.push_back(calculateNonEqualMesh(rowsA,rowsB,columnsB,i,cpuSize,true));
            allOp.push_back(calculateNonEqualMesh(rowsA,rowsB,columnsB,i,cpuSize,false));
        }
        res = *min_element(begin(allOp), end(allOp), [](OperationProperties op1, OperationProperties op2) {
            if(op1.cpuSize!=op2.cpuSize)
            {
                return op1.cpuSize<op2.cpuSize;
            }
            return op1.numberOf0 < op2.numberOf0;
        });
    }
    return res;
}

OperationProperties MatrixUtilities::calculateNonEqualMesh(int rowsA, int columnsAorRowsB, int columnsB, int nCpusMesh,int cpuSize,bool isMeshRow)
{
    OperationProperties res;
    if(isMeshRow)
    {
        res.meshRowSize = nCpusMesh;
        res.meshColumnSize = floor(cpuSize / nCpusMesh);
    }else
    {
        res.meshColumnSize = nCpusMesh;
        res.meshRowSize = floor(cpuSize / nCpusMesh);
    }
    res.cpuSize=res.meshRowSize+res.meshColumnSize;
    res.rowsA = ceil(rowsA / (float)res.meshRowSize) * res.meshRowSize;
    res.columnsAorRowsB = ceil(columnsAorRowsB / (float)res.meshColumnSize) * res.meshColumnSize;
    res.columnsB = ceil(columnsB / (float)res.meshColumnSize) * res.meshColumnSize;
    int numberOf0atA=(res.rowsA*res.columnsAorRowsB)-(rowsA*columnsB);
    int numberOf0atB=(res.columnsB*res.columnsAorRowsB)-(columnsAorRowsB*columnsB);
    res.numberOf0=numberOf0atA+numberOf0atB;
    return res;
}

double *MatrixUtilities::matrixMemoryAllocation(int rows, int columns)
{
    double *matrix = (double *)calloc(rows * columns, sizeof(double));
    return matrix;
}

void MatrixUtilities::matrixFree(double *matrix)
{
    free(matrix);
}

double *MatrixUtilities::matrixCustomAddition(int rows, int columns, double *A, double *B)
{
    int i, j, matrixIndex;
    double *res = matrixMemoryAllocation(rows, columns);
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            matrixIndex = matrixCalculateIndex(columns, i, j);
            res[matrixIndex] = A[matrixIndex] + B[matrixIndex];
        }
    }
    return res;
}

/**
 * @brief Calcula el indice unidimensional de la matriz
 * 
 * @param rowSize 
 * @param rowIndex 
 * @param columnIndex 
 * @return int 
 */
int MatrixUtilities::matrixCalculateIndex(int columnSize, int rowIndex, int columnIndex)
{
    return columnSize * rowIndex + columnIndex;
}

double *MatrixUtilities::matrixBlasMultiplication(int rowsA, int columnsAorRowsB, int columnsB, double *A, double *B, double *C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB, columnsAorRowsB, 1.0, A, rowsA, B, columnsAorRowsB, 1.0, C, rowsA);
    return C;
}