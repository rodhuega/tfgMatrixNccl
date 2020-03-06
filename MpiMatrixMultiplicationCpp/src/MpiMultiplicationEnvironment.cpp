#include "MpiMultiplicationEnvironment.h"

template <class Toperation>
MpiMultiplicationEnvironment<Toperation>::MpiMultiplicationEnvironment(int cpuRank, int cpuRoot, int cpuSizeInitial, MPI_Datatype basicOperationType)
{
    this->cpuRank = cpuRank;
    this->cpuRoot = cpuRoot;
    this->commOperation = commOperation;
    this->basicOperationType = basicOperationType;
    this->cpuSizeInitial = cpuSizeInitial;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setCommOperation(int cpuOperationsSize)
{
    MPI_Group groupInitial;
    MPI_Group groupOperation;
    int i;
    cpuOperationSize = cpuOperationsSize;
    MPI_Bcast(&cpuOperationSize, 1, MPI_INT, cpuRoot, MPI_COMM_WORLD);
    int membersGroupOperation[cpuOperationSize];
    for (i = 0; i < cpuOperationSize; i++)
    {
        membersGroupOperation[i] = i;
    }
    MPI_Comm_group(MPI_COMM_WORLD, &groupInitial);
    MPI_Group_incl(groupInitial, cpuOperationSize, membersGroupOperation, &groupOperation);
    MPI_Comm_create(MPI_COMM_WORLD, groupOperation, &commOperation);
    thisCpuPerformOperation = cpuRank < cpuOperationSize ? true : false;
}

template <class Toperation>
MatrixMain<Toperation> *MpiMultiplicationEnvironment<Toperation>::createAndSetNewMatrixLocalDistributed(std::string id, int rowsUsed, int columnsUsed, int rowsReal, int columnsReal)
{
    MatrixMain<Toperation> *res = new MatrixMain<Toperation>(rowsReal, columnsReal);
    res->setColumnsUsed(columnsUsed);
    res->setRowsUsed(rowsUsed);
    matricesGlobalDistributed[id] = res;
    return res;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setNewMatrixLocalDistributed(std::string id, MpiMatrix<Toperation> *mpiLocalMatrix)
{
    matricesLocalDistributed[id] = mpiLocalMatrix;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setNewMatrixLocalDistributedWithDimensions(std::string id, MpiMatrix<Toperation> *mpiLocalMatrix, int rows, int columns)
{
    matricesLocalDistributed[id] = mpiLocalMatrix;
    matricesGlobalDimensions[id] = std::make_tuple(rows, columns);
}

template <class Toperation>
MpiMatrix<Toperation> *MpiMultiplicationEnvironment<Toperation>::getAMatrixLocalDistributed(std::string id)
{
    auto it = matricesLocalDistributed.find(id);

    if (it == matricesLocalDistributed.end())
    {
        throw std::invalid_argument("La matriz localdistribuida no existe");
    }
    return it->second;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setAMatrixGlobalNonDistributedFromLocalDistributed(std::string id)
{
    Toperation *matrixFinalRes = getAMatrixLocalDistributed(id)->mpiRecoverDistributedMatrixSendRecv(cpuRoot);
    auto matrixFinalResDimensions = matricesGlobalDimensions[id];
    setNewMatrixGlobalNonDistributed(id, matrixFinalRes, std::get<0>(matrixFinalResDimensions), std::get<1>(matrixFinalResDimensions));
}

template <class Toperation>
MatrixMain<Toperation> *MpiMultiplicationEnvironment<Toperation>::getAMatrixGlobal(std::string id)
{
    MatrixMain<Toperation> *res;
    try
    {
        res = getAMatrixGlobalDistributed(id);
    }
    catch (std::invalid_argument &e)
    {
        Toperation *matrixAux = getAMatrixGlobalNonDistributed(id);
        dimensions dimensionsAux = matricesGlobalDimensions[id];
        res = new MatrixMain<Toperation>(std::get<0>(dimensionsAux), std::get<1>(dimensionsAux));
        res->setIsDistributed(false);
    }
    return res;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setNewMatrixGlobalNonDistributed(std::string id, Toperation *matrixMainGlobal, int rows, int columns)
{
    matricesGlobalNonDistributed[id] = matrixMainGlobal;
    matricesGlobalDimensions[id] = std::make_tuple(rows, columns);
}

template <class Toperation>
Toperation *MpiMultiplicationEnvironment<Toperation>::getAMatrixGlobalNonDistributed(std::string id)
{
    auto it = matricesGlobalNonDistributed.find(id);
    if (it == matricesGlobalNonDistributed.end())
    {
        throw std::invalid_argument("La matriz global no distribuida no existe");
    }
    return it->second;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setNewMatrixGlobalDistributed(std::string id, MatrixMain<Toperation> *matrixMainGlobal)
{
    matricesGlobalDistributed[id] = matrixMainGlobal;
}

template <class Toperation>
MatrixMain<Toperation> *MpiMultiplicationEnvironment<Toperation>::getAMatrixGlobalDistributed(std::string id)
{
    auto it = matricesGlobalDistributed.find(id);
    if (it == matricesGlobalDistributed.end())
    {
        throw std::invalid_argument("La matriz no existe");
    }
    return it->second;
}

template <class Toperation>
bool MpiMultiplicationEnvironment<Toperation>::getIfThisCpuPerformOperation()
{
    return thisCpuPerformOperation;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::PerformCalculations(std::string idA, std::string idB, std::string idC, bool printMatrix)
{
    bool isDistributedA, isDistributedB;
    int rowsAUsed, columnsAUsed, rowsBUsed, columnsBUsed, meshRowSize, meshColumnSize;
    int rowsAReal, columnsAReal, rowsBReal, columnsBReal;
    OperationProperties op;
    MatrixMain<Toperation> *ma, *mb, *mc;
    MpiMatrix<Toperation> *mMpiLocalA, *mMpiLocalB;

    ma = getAMatrixGlobal(idA);
    mb = getAMatrixGlobal(idB);
    ///////////////FALTA COMPROBAR SI SE PUEDE REALIZAR LA OPERACION/////////////////////////////////

    //Comprobar si ya estan distirbuida
    isDistributedA = ma->getIsDistributed();
    isDistributedB = mb->getIsDistributed();

    // //Recuperacion de las matrices
    if (!isDistributedA && !isDistributedB)
    {
        Toperation *a, *b;
        //VER SI EJECUTO ESTO SOLO, actualmente lo ejecutan todos los procesadores
        op = MatrixUtilities<double>::getMeshAndMatrixSize(ma->getRowsReal(), ma->getColumnsReal(), mb->getRowsReal(), mb->getColumnsReal(), cpuSizeInitial);
        meshRowSize = op.meshRowSize;
        meshColumnSize = op.meshColumnSize;
        ma->setRowsUsed(op.rowsA);
        ma->setColumnsUsed(op.columnsAorRowsB);
        ma->setMatrix(getAMatrixGlobalNonDistributed(idA));
        rowsAUsed = ma->getRowsUsed();
        columnsAUsed = ma->getColumnsUsed();
        rowsAReal = ma->getRowsReal();
        columnsAReal = ma->getColumnsReal();

        mb->setRowsUsed(op.columnsAorRowsB);
        mb->setColumnsUsed(op.columnsB);
        mb->setMatrix(getAMatrixGlobalNonDistributed(idB));
        rowsBUsed = mb->getRowsUsed();
        columnsBUsed = mb->getColumnsUsed();
        rowsBReal = mb->getRowsReal();
        columnsBReal = mb->getColumnsReal();
        //Aqui habra que mirar mas casos y expandir este if, de momento solo hace el caso de que ninguna de las dos matrices esta distribuida
        if (cpuRank == cpuRoot)
        {
            b = mb->getMatrix();
            a = ma->getMatrix();
            if (printMatrix)
            {
                std::cout << "A-> Rows: " << ma->getRowsReal() << ", Columns: " << ma->getColumnsReal() << ", Matriz A:" << std::endl;
                MatrixUtilities<Toperation>::printMatrix(ma->getRowsReal(), mb->getColumnsReal(), a);
                std::cout << "B-> Rows: " << mb->getRowsReal() << ", Columns: " << mb->getColumnsReal() << ", Matriz B:" << std::endl;
                MatrixUtilities<Toperation>::printMatrix(mb->getRowsReal(), ma->getColumnsReal(), b);
            }
        }
        //Ver que procesos van a realizar el calculo y crear el entorno
        setCommOperation(op.cpuSize);
        if (thisCpuPerformOperation)
        {
            MPI_Comm_size(commOperation, &cpuOperationSize);
            MPI_Comm_rank(commOperation, &cpuRank);
            //Distribucion de las matrices entre los distintos procesos

            mMpiLocalA = new MpiMatrix<Toperation>(cpuOperationSize, cpuRank, meshRowSize, meshColumnSize, ma, commOperation, basicOperationType);

            mMpiLocalB = new MpiMatrix<Toperation>(cpuOperationSize, cpuRank, meshRowSize, meshColumnSize, mb, commOperation, basicOperationType);
            mMpiLocalA->mpiDistributeMatrixSendRecv(a, cpuRoot);
            mMpiLocalB->mpiDistributeMatrixSendRecv(b, cpuRoot);

            MPI_Barrier(commOperation);
            setNewMatrixLocalDistributed(idA, mMpiLocalA);
            setNewMatrixLocalDistributed(idB, mMpiLocalB);
            ma->setIsDistributed(true);
            mb->setIsDistributed(true);
        }

    } //////////////////////////////////////////POR HACER EL CASO EN EL QUE ESTEN YA DISTRIBUIDAS; AÑADIR ELSES

    //Realizacion de la multiplicacion distribuida
    if (thisCpuPerformOperation)
    {
        Toperation *matrixLocalC = mpiSumma(*mMpiLocalA, *mMpiLocalB, meshRowSize, meshColumnSize);
        //Creacion del objeto local que contiene el resultado local de la operacion y asignacion del resultado a este objeto
        MatrixMain<Toperation> *mc = createAndSetNewMatrixLocalDistributed(idC, rowsAUsed, columnsBUsed, rowsAReal, columnsBReal);
        MpiMatrix<Toperation> *mMpiLocalC = new MpiMatrix<Toperation>(cpuOperationSize, cpuRank, meshRowSize, meshColumnSize, mc, commOperation, basicOperationType);
        mMpiLocalC->setMatrixLocal(matrixLocalC);
        setNewMatrixLocalDistributedWithDimensions(idC, mMpiLocalC, rowsAUsed, columnsBUsed);
    }
}

template <class Toperation>
Toperation *MpiMultiplicationEnvironment<Toperation>::mpiSumma(MpiMatrix<Toperation> matrixLocalA, MpiMatrix<Toperation> matrixLocalB, int meshRowsSize, int meshColumnsSize)
{
    int i;
    MPI_Group groupInitial, groupRow, groupColumn;
    MPI_Comm commRow, commCol;
    int rowsA = matrixLocalA.getMatrixMain()->getRowsUsed();
    int columnsAorRowsB = matrixLocalA.getMatrixMain()->getColumnsUsed();
    int columnsB = matrixLocalB.getMatrixMain()->getColumnsUsed();
    int blockSizeA = matrixLocalA.getBlockSize();
    int blockSizeB = matrixLocalB.getBlockSize();
    int blockRowSizeA = matrixLocalA.getBlockRowSize();
    int blockColumnsSizeA = matrixLocalA.getBlockColumnSize();
    int blockColumnsSizeB = matrixLocalB.getBlockColumnSize();
    int blockRowSizeB = matrixLocalB.getBlockRowSize();
    int blockRowSize = matrixLocalA.getBlockRowSize();
    Toperation *matrixLocalC = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeA, blockColumnsSizeB);
    Toperation *matrixAuxiliarA = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeA, blockColumnsSizeA);
    Toperation *matrixAuxiliarB = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeB, blockColumnsSizeB);
    MPI_Comm_group(commOperation, &groupInitial);
    //Conseguir a que columna y fila pertenezco
    int rowColor = matrixLocalA.getRowColor();
    int columnColor = matrixLocalA.getColumnColor();
    //Creacion de los nuevos grupos comunicadores para hacer Broadcast de filas o columnas a los pertenecientes a la malla de misma fila o columna
    int colGroupIndex[meshColumnsSize];
    int rowGroupIndex[meshRowsSize];
    for (i = 0; i < meshColumnsSize; i++)
    {
        rowGroupIndex[i] = rowColor * meshColumnsSize + i;
    }
    for (i = 0; i < meshRowsSize; i++)
    {
        colGroupIndex[i] = columnColor + i * meshColumnsSize;
    }
    MPI_Group_incl(groupInitial, meshColumnsSize, rowGroupIndex, &groupRow);
    MPI_Group_incl(groupInitial, meshRowsSize, colGroupIndex, &groupColumn);
    MPI_Comm_create(commOperation, groupRow, &commRow);
    MPI_Comm_create(commOperation, groupColumn, &commCol);
    //Realizacion de las operaciones matematicas
    for (i = 0; i < meshRowsSize; i++)
    {
        // MPI_Barrier(commOperation);
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank,blockRowSize,blockRowSize,matrixLocalC,".Inicio Iteracion: "+to_string(i));
        if (columnColor == i)
        {
            memcpy(matrixAuxiliarA, matrixLocalA.getMatrixLocal(), blockSizeA * sizeof(Toperation));
        }
        if (rowColor == i)
        {
            memcpy(matrixAuxiliarB, matrixLocalB.getMatrixLocal(), blockSizeB * sizeof(Toperation));
        }
        MPI_Bcast(matrixAuxiliarA, blockSizeA, basicOperationType, i, commRow);
        MPI_Bcast(matrixAuxiliarB, blockSizeB, basicOperationType, i, commCol);
        MatrixUtilities<Toperation>::Multiplicacion(blockRowSizeA, blockRowSizeB, blockColumnsSizeB, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
    }
    // MatrixUtilities<Toperation>::matrixBlasMultiplication(blockRowSizeA, blockRowSizeB, blockColumnsSizeB, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
    // MatrixUtilities<Toperation>::debugMatrixDifferentCpus(cpuRank, blockRowSize, blockRowSize, matrixLocalC, ".Final Iteracion: " + std::to_string(i));
    //Liberacion de las matrices auxiliares que realizaban computo
    MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarA);
    MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarB);

    return matrixLocalC;
}

template class MpiMultiplicationEnvironment<double>;
template class MpiMultiplicationEnvironment<float>;
