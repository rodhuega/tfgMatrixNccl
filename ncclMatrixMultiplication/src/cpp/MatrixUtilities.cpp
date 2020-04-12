#include "MatrixUtilities.h"

using namespace std;
template <class Toperation>
void MatrixUtilities<Toperation>::printMatrix(int rows, int columns, Toperation *M)
{
    int i, j, matrixIndex;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            matrixIndex = matrixCalculateIndex(rows,columns, i, j);
            cout << M[matrixIndex] << "\t";
        }
        cout << endl;
    }
}
template <class Toperation>
void MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(int rows, int columns, Toperation *M, int cpuRank, int cpuRankPrint, string message)
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

template <class Toperation>
vector<tuple<int, int>> MatrixUtilities<Toperation>::checkEqualityOfMatrices(Toperation *A, Toperation *B, int rows, int columns)
{
    vector<std::tuple<int, int>> res;
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            if (fabs(A[IDX2C(i,j,rows)] - B[IDX2C(i,j,rows)]) > 0.000001)
            {
                res.push_back(std::make_tuple(i, j));
            }
        }
    }
    return res;
}

template <class Toperation>
void MatrixUtilities<Toperation>::printErrorEqualityMatricesPosition(vector<std::tuple<int, int>> errors, bool printDetailed)
{
    unsigned int i;
    if (errors.size() != 0)
    {
        cout << "Las dos matrices no son iguales" << endl;
        for (i = 0; i < errors.size() && printDetailed; i++)
        {
            cout << "Fila: " << std::get<0>(errors[i]) << ", Columna: " << std::get<1>(errors[i]) << endl;
        }
    }
    else
    {
        cout << "Las dos matrices son identicas" << endl;
    }
}

template <class Toperation>
void MatrixUtilities<Toperation>::debugMatrixDifferentCpus(int cpuRank, int rows, int columns, Toperation *M, string extraMessage)
{
    usleep(cpuRank * 1000);
    cout << "Parte del proceso: " << cpuRank << " " << extraMessage << endl;
    MatrixUtilities::printMatrix(rows, columns, M);
}

template <class Toperation>
void MatrixUtilities<Toperation>::debugMatricesLocalDifferentCpus(int cpuRank, int cpuSize, int rows, int columns, std::vector<Toperation *> M, string extraMessage)
{
    unsigned int i;
    for (i = 0; i < M.size(); i++)
    {
        std::string msg = " Matriz local: " + to_string((cpuRank + (i * cpuSize)));
        MatrixUtilities::debugMatrixDifferentCpus(cpuRank, rows, columns, M[i], msg);
    }
}
template <class Toperation>
bool MatrixUtilities<Toperation>::canMultiply(int columnsA, int rowsB)
{
    return columnsA == rowsB;
}

template <class Toperation>
OperationProperties MatrixUtilities<Toperation>::getMeshAndMatrixSize(int rowsA, int columnsA, int rowsB, int columnsB, int cpuSize)
{
    OperationProperties res;

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
        if (op1.gpuSize != op2.gpuSize)
        {
            return op1.gpuSize > op2.gpuSize;
        }
        return op1.numberOf0 < op2.numberOf0;
    });
    res = allOp[0];

    return res;
}

template <class Toperation>
OperationProperties MatrixUtilities<Toperation>::calculateNonEqualMesh(int rowsA, int columnsAorRowsB, int columnsB, int nCpusMesh1, int nCpusMesh2, bool isMeshRow)
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
    res.gpuSize = res.meshRowSize * res.meshColumnSize;
    res.rowsA = ceil(rowsA / (float)res.meshRowSize) * res.meshRowSize;
    res.columnsAorRowsB = ceil(columnsAorRowsB / (float)res.meshColumnSize) * res.meshColumnSize;
    res.columnsB = ceil(columnsB / (float)res.meshColumnSize) * res.meshColumnSize;
    int numberOf0atA = (res.rowsA * res.columnsAorRowsB) - (rowsA * columnsAorRowsB);
    int numberOf0atB = (res.columnsB * res.columnsAorRowsB) - (columnsAorRowsB * columnsB);
    res.numberOf0 = numberOf0atA + numberOf0atB;
    //PUEDE QUE AQUI NECESITE UN IF DEPENDIENDO DE CUAL SEA EL GRID DOMINANTE; DE MOMENTO EL GRID DOMINANTE AHORA ES A SIEMPRE
    res.blockColumnSizeA = res.columnsAorRowsB / res.meshColumnSize;
    res.blockRowSizeB = res.blockColumnSizeA;
    res.blockRowSizeA = res.rowsA / res.meshRowSize;
    res.blockColumnSizeB = res.columnsB / res.meshColumnSize;
    res.candidate = res.meshColumnSize > 1 && res.meshRowSize > 1;
    return res;
}

template <class Toperation>
OperationProperties MatrixUtilities<Toperation>::getMeshAndMatrixSizeFromOneDistributedMatrix(int rowsA, int columnsA, int rowsB, int columnsB, int meshRowSize,int meshColumnSize,bool isAAlreadyDistributed)
{
    OperationProperties res;
    res.meshRowSize=meshRowSize;
    res.meshColumnSize=meshColumnSize;
    if(isAAlreadyDistributed)
    {
        res.rowsA=rowsA;
        res.columnsAorRowsB=columnsA;
        res.columnsB = ceil(columnsB / (float)res.meshColumnSize) * res.meshColumnSize;
        res.blockRowSizeA = res.rowsA / res.meshRowSize;
        res.blockColumnSizeA = res.columnsAorRowsB / res.meshColumnSize;
        res.blockRowSizeB = res.blockColumnSizeA;
        res.blockColumnSizeB = res.columnsB / res.meshColumnSize;
    }else
    {
        res.columnsB=columnsB;
        res.columnsAorRowsB=rowsB;
        res.rowsA=ceil(rowsA / (float)res.meshRowSize) * res.meshRowSize;
        res.blockRowSizeA = res.rowsA / res.meshRowSize;
        res.blockColumnSizeA = res.columnsAorRowsB / res.meshColumnSize;
        res.blockRowSizeB = res.blockColumnSizeA;
        res.blockColumnSizeB = res.columnsB / res.meshColumnSize;
    }
    
    return res;
}

template <class Toperation>
Toperation *MatrixUtilities<Toperation>::matrixMemoryAllocation(int rows, int columns)
{
    Toperation *matrix = (Toperation *)calloc(rows * columns, sizeof(Toperation));
    return matrix;
}

template <class Toperation>
void MatrixUtilities<Toperation>::matrixFree(Toperation *matrix)
{
    free(matrix);
}

template <class Toperation>
int MatrixUtilities<Toperation>::matrixCalculateIndex(int rowSize, int columnSize, int rowIndex, int columnIndex)
{
    return IDX2C(rowIndex,columnIndex,rowSize);
    // return columnSize * rowIndex + columnIndex;
}

template <class Toperation>
Toperation *MatrixUtilities<Toperation>::ReadOrGenerateRandomMatrix(bool isRandom, const char *fileName, int &rows, int &columns, int boundLower, int boundUpper)
{
    int i, j, matrixIndex;
    std::ifstream file;
    if (!isRandom)
    {
        file.open(fileName);
        file >> rows >> columns;
    }
    //Configuracion del generador de numeros por si se genera una matriz random
    random_device rd;
    mt19937 eng(rd());
    uniform_real_distribution<> distr(boundLower, boundUpper);
    Toperation *matrix = MatrixUtilities<Toperation>::matrixMemoryAllocation(rows, columns);
    //Bucle de generacion o lectura de la matrizs
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            matrixIndex = MatrixUtilities<Toperation>::matrixCalculateIndex(rows,columns, i, j);
            if (isRandom)
            {
                matrix[matrixIndex] = distr(eng);
            }
            else
            {
                file >> matrix[matrixIndex];
            }
        }
    }
    if (!isRandom)
    {
        file.close();
    }
    return matrix;
}

template <class Toperation>
void MatrixUtilities<Toperation>::matrixBlasMultiplication(OperationType opt,int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C)
{
    if(opt==MultDouble)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB, columnsAorRowsB, 1.0, (double*)A, rowsA, (double*)B, columnsAorRowsB, 1.0, (double*)C, rowsA);
    }else
    {
        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB, columnsAorRowsB, 1.0, (float*)A, rowsA, (float*)B, columnsAorRowsB, 1.0, (float*)C, rowsA);
    }
}

template class MatrixUtilities<double>;
template class MatrixUtilities<float>;