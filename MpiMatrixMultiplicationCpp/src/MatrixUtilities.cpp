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

double* MatrixUtilities::getMatrixWithoutZeros(int rowsReal,int columnsUsed, int columnsReal,double* matrix)
{
    int i,j;
    int nextRowPosition=0;
    // int numberOf0Colums=columnsUsed-columnsReal;
    double* res=matrixMemoryAllocation(rowsReal,columnsReal);
    for(i=0;i<rowsReal;i++)
    {
        for(j=0;j<columnsReal;j++)
        {
            res[i*columnsReal+j]=matrix[nextRowPosition+j];
        }
        nextRowPosition+=columnsUsed;
    }
    return res;
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
        //Se calculan todas las posibilidadades y se selecciona la que mas cpus use y menos 0 contenga de esas opciones, Solo se añaden elementos validos(ninguno con meshDimension 1 o 0)
        int i, j, numberOfZerosA, numberOfZerosB;
        vector<OperationProperties> allOp;
        vector<OperationProperties> sameCpuSizeOp;
        for (i = 2; i < cpuSize - 1; i++)
        {
            for (j = i; j * i <= cpuSize; j++)
            {
                OperationProperties opRow = calculateNonEqualMesh(rowsA, rowsB, columnsB, i, j, true);
                OperationProperties opColumn = calculateNonEqualMesh(rowsA, rowsB, columnsB, i, j, false);
                if (opRow.candidate)
                {
                    allOp.push_back(opRow);
                }
                if (opColumn.candidate)
                {
                    allOp.push_back(opColumn);
                }
            }
        }
        sort(begin(allOp), end(allOp), [](OperationProperties op1, OperationProperties op2) {
            if (op1.cpuSize != op2.cpuSize)
            {
                return op1.cpuSize > op2.cpuSize;
            }
            return op1.numberOf0 < op2.numberOf0;
        });
        std::cout << "allOpSize:" << allOp.size() << std::endl;
        res = allOp[0];
    }
    return res;
}

OperationProperties MatrixUtilities::calculateNonEqualMesh(int rowsA, int columnsAorRowsB, int columnsB, int nCpusMesh1, int nCpusMesh2, bool isMeshRow)
{
    OperationProperties res;
    if (isMeshRow)
    {
        res.meshRowSize = nCpusMesh1;
        res.meshColumnSize = nCpusMesh2;
    }
    else
    {
        res.meshColumnSize = nCpusMesh1;
        res.meshRowSize = nCpusMesh2;
    }
    res.cpuSize = res.meshRowSize * res.meshColumnSize;
    res.rowsA = ceil(rowsA / (float)res.meshRowSize) * res.meshRowSize;
    res.columnsAorRowsB = max(ceil(columnsAorRowsB / (float)res.meshColumnSize) * res.meshColumnSize,
                              ceil(columnsAorRowsB / (float)res.meshColumnSize) * res.meshRowSize);
    res.columnsB = ceil(columnsB / (float)res.meshColumnSize) * res.meshColumnSize;
    int numberOf0atA = (res.rowsA * res.columnsAorRowsB) - (rowsA * columnsAorRowsB);
    int numberOf0atB = (res.columnsB * res.columnsAorRowsB) - (columnsAorRowsB * columnsB);
    res.numberOf0 = numberOf0atA + numberOf0atB;
    res.blockColumnSizeA = columnsAorRowsB / res.meshColumnSize;
    res.blockRowSizeB = columnsAorRowsB / res.meshRowSize;
    res.candidate = res.meshColumnSize > 1 && res.meshRowSize > 1 && res.blockRowSizeB == res.blockColumnSizeA;
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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB, columnsAorRowsB, 1.0, A, columnsAorRowsB, B, columnsB, 1.0, C, rowsA);
    return C;
}