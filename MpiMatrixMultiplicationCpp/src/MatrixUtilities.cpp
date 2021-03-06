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
            matrixIndex = matrixCalculateIndex(columns, i, j);
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
int MatrixUtilities<Toperation>::checkEqualityOfMatrices(Toperation *A, Toperation *B, int rows, int columns)
{
    double Anorm= frobeniusNormMatrixLapack(rows,columns, A);
    double Bnorm= frobeniusNormMatrixLapack(rows,columns, B);
    bool res= fabs(Anorm-Bnorm)/Anorm;
    std::cout<<"Norma de la primera matriz: "<<Anorm<<", norma de la segunda matriz: "<<Bnorm<<std::endl;
    return res;
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
Toperation *MatrixUtilities<Toperation>::getMatrixWithoutZeros(int rowsReal, int columnsUsed, int columnsReal, Toperation *matrix)
{
    int i, j;
    int nextRowPosition = 0;
    Toperation *res = matrixMemoryAllocation(rowsReal, columnsReal);
    for (i = 0; i < rowsReal; i++)
    {
        for (j = 0; j < columnsReal; j++)
        {
            res[i * columnsReal + j] = matrix[nextRowPosition + j];
        }
        //Se consigue el indice que apunta a la siguiente fila, asi nos saltamos los 0s extendidos de esa columna
        nextRowPosition += columnsUsed;
    }
    return res;
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
        if (op1.cpuSize != op2.cpuSize)
        {
            return op1.cpuSize > op2.cpuSize;
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
    res.cpuSize = res.meshRowSize * res.meshColumnSize;
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
int MatrixUtilities<Toperation>::matrixCalculateIndex(int columnSize, int rowIndex, int columnIndex)
{
    return columnSize * rowIndex + columnIndex;
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
    #pragma omp parallel for
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            matrixIndex = MatrixUtilities<Toperation>::matrixCalculateIndex(columns, i, j);
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
void MatrixUtilities<Toperation>::matrixBlasMultiplication(int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB, columnsAorRowsB, 1.0, (double*)A, columnsAorRowsB, (double*)B, columnsB, 1.0, (double*)C, columnsB);
}

template <class Toperation>
double MatrixUtilities<Toperation>::frobeniusNormMatrixLapack(int rows, int columns, Toperation *A)
{
    double normA=0,normB=0;
    int i, j,index;
    double elementA,elementB;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            index=matrixCalculateIndex(columns,i,j);
            elementA=A[index];
            normA+=(elementA*elementA);
        }
    }
    normA=sqrt(normA);
    return normA;
}

template <class Toperation>
void MatrixUtilities<Toperation>::Multiplicacion(int rowsA, int columnsAorRowsB, int columnsB, Toperation *A, Toperation *B, Toperation *C)
{
    for (int i = 0; i < rowsA; i++)
    {
        for (int j = 0; j < columnsB; j++)
        {
            Toperation sum = 0;
            for (int k = 0; k < columnsAorRowsB; k++)
            {
                sum = sum + A[i * columnsAorRowsB + k] * B[k * columnsB + j];
            }
            C[i * columnsB + j] += sum;
        }
    }
}

template class MatrixUtilities<double>;
template class MatrixUtilities<float>;