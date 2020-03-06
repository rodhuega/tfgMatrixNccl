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
    cpuSize = cpuOperationsSize;
    MPI_Bcast(&cpuSize, 1, MPI_INT, cpuRoot, MPI_COMM_WORLD);
    int membersGroupOperation[cpuSize];
    for (i = 0; i < cpuSize; i++)
    {
        membersGroupOperation[i] = i;
    }
    MPI_Comm_group(MPI_COMM_WORLD, &groupInitial);
    MPI_Group_incl(groupInitial, cpuSize, membersGroupOperation, &groupOperation);
    MPI_Comm_create(MPI_COMM_WORLD, groupOperation, &commOperation);
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::PerformCalculations(std::string idA, std::string idB, std::string idC, bool printMatrix)
{
    bool isDistributedA, isDistributedB;
    int rowsA, columnsA, rowsB, columnsB, meshRowSize, meshColumnSize;
    OperationProperties op;
    MatrixMain<Toperation> *ma, *mb, *mc;
    MpiMatrix<Toperation> *mMpiLocalA, *mMpiLocalB;

    //Comprobar si ya estan distirbuida
    if (cpuRank == cpuRoot)
    {
        ma = getAMatrixGlobal(idA);
        mb = getAMatrixGlobal(idB);
        isDistributedA = ma->getIsDistributed();
        isDistributedB = mb->getIsDistributed();
    }
    MPI_Bcast(&isDistributedA, 1, MPI_C_BOOL, cpuRoot, MPI_COMM_WORLD);
    MPI_Bcast(&isDistributedB, 1, MPI_C_BOOL, cpuRoot, MPI_COMM_WORLD);
    // //Recuperacion de las matrices
    if (!isDistributedA && !isDistributedB)
    {
        Toperation *a, *b;
        //Aqui habra que mirar mas casos y expandir este if, de momento solo hace el caso de que ninguna de las dos matrices esta distribuida
        if (cpuRank == cpuRoot)
        {
            op = MatrixUtilities<double>::getMeshAndMatrixSize(ma->getRowsReal(), ma->getColumnsReal(), mb->getRowsReal(), mb->getColumnsReal(), cpuSizeInitial);
            meshRowSize = op.meshRowSize;
            meshColumnSize = op.meshColumnSize;
            ma->setRowsUsed(op.rowsA);
            ma->setColumnsUsed(op.columnsAorRowsB);
            ma->setMatrix(getAMatrixGlobalNonDistributed(idA));
            rowsA = ma->getRowsUsed();
            columnsA = ma->getColumnsUsed();
            a = ma->getMatrix();

            mb->setRowsUsed(op.columnsAorRowsB);
            mb->setColumnsUsed(op.columnsB);
            mb->setMatrix(getAMatrixGlobalNonDistributed(idB));
            rowsB = mb->getRowsUsed();
            columnsB = mb->getColumnsUsed();
            b = mb->getMatrix();

            if (printMatrix)
            {
                std::cout << "A-> Rows: " << rowsA << ", Columns: " << columnsA << ", Matriz A:" << std::endl;
                MatrixUtilities<Toperation>::printMatrix(rowsA, columnsA, a);
                std::cout << "B-> Rows: " << rowsB << ", Columns: " << columnsB << ", Matriz B:" << std::endl;
                MatrixUtilities<Toperation>::printMatrix(rowsB, columnsB, b);
            }
        }
        //Ver que procesos van a realizar el calculo y crear el entorno
        setCommOperation(op.cpuSize);
        MPI_Comm_size(commOperation, &cpuSize);
        MPI_Comm_rank(commOperation, &cpuRank);
        //POR AQUI FALTA UN IFFF QUE MIRE QUE DESCARTE PROCESOS
        //Broadcasting de informacion basica pero necesaria
        MPI_Bcast(&rowsA, 1, MPI_INT, cpuRoot, commOperation);
        MPI_Bcast(&columnsA, 1, MPI_INT, cpuRoot, commOperation);
        MPI_Bcast(&rowsB, 1, MPI_INT, cpuRoot, commOperation);
        MPI_Bcast(&columnsB, 1, MPI_INT, cpuRoot, commOperation);
        MPI_Bcast(&meshRowSize, 1, MPI_INT, cpuRoot, commOperation);
        MPI_Bcast(&meshColumnSize, 1, MPI_INT, cpuRoot, commOperation);
        //Distribucion de las matrices entre los distintos procesos
        mMpiLocalA = new MpiMatrix<Toperation>(cpuSize, cpuRank, meshRowSize, meshColumnSize, rowsA, columnsA, commOperation, basicOperationType);
        mMpiLocalB = new MpiMatrix<Toperation>(cpuSize, cpuRank, meshRowSize, meshColumnSize, rowsB, columnsB, commOperation, basicOperationType);
        mMpiLocalA->mpiDistributeMatrix(a, cpuRoot);
        mMpiLocalB->mpiDistributeMatrix(b, cpuRoot);
        //Asignas a un diccionario mMpiLocalA,mMpiLocalB
        setNewMatrixLocalDistributed(idA,mMpiLocalA);
        setNewMatrixLocalDistributed(idB,mMpiLocalB);

        if (cpuRank == cpuRoot)
        {
            ma->setIsDistributed(true);
            mb->setIsDistributed(true);
        }

    } //////////////////////////////////////////POR HACER EL CASO EN EL QUE ESTEN YA DISTRIBUIDAS; AÑADIR ELSES

    //Realizacion de la multiplicacion distribuicda
    MpiMatrix<Toperation> mMpiLocalC = mpiSumma(*mMpiLocalA, *mMpiLocalB, meshRowSize, meshColumnSize); //////////////////HACER QUE ESTO TAMBIEN SEA UN PUNTERO
    setNewMatrixLocalDistributed(idC,mMpiLocalC);
    set
    ///////////////////////////////A PARTIR DE AQUI MOVER A UN METODO QUE SE LLAME RECOVERLOCALCALCULATION
    Toperation *matrixFinalRes = mMpiLocalC.mpiRecoverDistributedMatrixReduce(cpuRoot);
    //Asignacion de la nueva matriz al entorno
    if (cpuRank == cpuRoot)
    {
        int rowsAReal = ma->getRowsReal();
        int columnsBUsed = mb->getColumnsUsed();
        int columnsBReal = mb->getColumnsReal();

        Toperation *matrixWithout0 = MatrixUtilities<Toperation>::getMatrixWithoutZeros(rowsAReal, columnsBUsed, columnsBReal, matrixFinalRes);
        // setNewMatrixGlobal(idC,mc);
        //Mostrar informacion en caso de que sea necesario
        if (printMatrix)
        {
            MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(rowsA, columnsB, matrixFinalRes, cpuRank, cpuRoot, "Dimensiones C: Rows: " + std::to_string(rowsA) + ", Columns: " + std::to_string(columnsB) + ", El resultado de la multiplicacion es: ");
            MatrixUtilities<Toperation>::printMatrixOrMessageForOneCpu(rowsAReal, columnsBReal, matrixWithout0, cpuRank, cpuRoot, "Dimensiones C: Rows: " + std::to_string(rowsAReal) + ", Columns: " + std::to_string(columnsBReal) + ", Sin los 0s: ");
        }
        MatrixUtilities<Toperation>::matrixFree(matrixFinalRes);
    }
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
    Toperation *matrixFinalRes = getAMatrixLocalDistributed(id)->mpiRecoverDistributedMatrixReduce(cpuRoot);
    setNewMatrixGlobalNonDistributed(id,matrixFinalRes);
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
        dimensions dimensionsAux = matricesGlobalNonDistributedDimensions[id];
        res = new MatrixMain<Toperation>(std::get<0>(dimensionsAux), std::get<1>(dimensionsAux));
        res->setIsDistributed(false);
    }
    return res;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setNewMatrixGlobalNonDistributed(std::string id, Toperation *matrixMainGlobal, int rows, int columns)
{
    matricesGlobalNonDistributed[id] = matrixMainGlobal;
    matricesGlobalNonDistributedDimensions[id] = std::make_tuple(rows, columns);
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
MpiMatrix<Toperation> MpiMultiplicationEnvironment<Toperation>::mpiSumma(MpiMatrix<Toperation> matrixLocalA, MpiMatrix<Toperation> matrixLocalB, int meshRowsSize, int meshColumnsSize)
{
    int i;
    MPI_Group groupInitial, groupRow, groupColumn;
    MPI_Comm commRow, commCol;
    int rowsA = matrixLocalA.getRowSize();
    int columnsAorRowsB = matrixLocalA.getColumnSize();
    int columnsB = matrixLocalB.getColumnSize();
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
    int rowColor = cpuRank / meshColumnsSize;
    int columnColor = cpuRank % meshColumnsSize;
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
        // MatrixUtilities::matrixBlasMultiplication(blockRowSizeA, blockRowSizeB, blockColumnsSizeB, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
        // MatrixUtilities::debugMatrixDifferentCpus(cpuRank, blockRowSize, blockRowSize, matrixLocalC, ".Final Iteracion: " + std::to_string(i));
    }
    //Liberacion de las matrices auxiliares que realizaban computo
    MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarA);
    MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarB);
    //Creacion del objeto local que contiene el resultado local de la operacion y asignacion del resultado a este objeto
    MpiMatrix<Toperation> res = MpiMatrix<Toperation>(cpuSize, cpuRank, meshRowsSize, meshColumnsSize, rowsA, columnsB, commOperation, basicOperationType);
    res.setMatrixLocal(matrixLocalC);
    return res;
}

template class MpiMultiplicationEnvironment<double>;
template class MpiMultiplicationEnvironment<float>;
