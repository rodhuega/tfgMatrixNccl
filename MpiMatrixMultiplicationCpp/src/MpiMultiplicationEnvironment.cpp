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
MpiMultiplicationEnvironment<Toperation>::~MpiMultiplicationEnvironment()
{
    for (auto itr = matricesGlobalSimplePointer.begin(); itr != matricesGlobalSimplePointer.end(); itr++)
    {
        delete (itr->second);
    }
    for (auto itr = matricesMatrixMain.begin(); itr != matricesMatrixMain.end(); itr++)
    {
        delete (itr->second);
    }
    matricesGlobalSimplePointer.clear();
    matricesMatrixMain.clear();
    matricesGlobalDimensions.clear();
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
bool MpiMultiplicationEnvironment<Toperation>::getIfThisCpuPerformOperation()
{
    return thisCpuPerformOperation;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setOrAddMatrixGlobalSimplePointer(std::string id, Toperation *matrixMainGlobal, int rows, int columns)
{
    matricesGlobalSimplePointer[id] = matrixMainGlobal;
    matricesGlobalDimensions[id] = std::make_tuple(rows, columns);
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::setOrAddMatrixMain(std::string id, MatrixMain<Toperation> *matrixMainGlobal)
{
    matricesMatrixMain[id] = matrixMainGlobal;
}

template <class Toperation>
Toperation *MpiMultiplicationEnvironment<Toperation>::getMatrixGlobalSimplePointer(std::string id)
{
    //Primero busca en el diccionario de punteros simples
    auto it = matricesGlobalSimplePointer.find(id);
    if (it == matricesGlobalSimplePointer.end())
    {
        //En caso de no encontrarlo intenta buscarlo en el diccionaro de MatrixMain
        auto auxMatrix = getMainMatrix(id, false); //En caso de que no exista tampoco tira excepcion en ese metodo
        return auxMatrix->getMatrix();
    }
    return it->second;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::recoverDistributedMatrix(std::string id)
{
    MatrixMain<Toperation> *auxMatrix = getMainMatrix(id, false);
    Toperation *matrixFinalRes = auxMatrix->getMpiMatrix()->mpiRecoverDistributedMatrixSendRecv(cpuRoot);
    auxMatrix->setMatrix(matrixFinalRes);
}

template <class Toperation>
MatrixMain<Toperation> *MpiMultiplicationEnvironment<Toperation>::getMainMatrix(std::string id, bool create)
{
    auto it = matricesMatrixMain.find(id);
    if (it == matricesMatrixMain.end())
    {
        if (!create)
        {
            throw std::invalid_argument("La matriz global existe");
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        else
        {
            Toperation *matrixAux = getMatrixGlobalSimplePointer(id);
            dimensions dimensionsAux = matricesGlobalDimensions[id];
            MatrixMain<Toperation> *res = new MatrixMain<Toperation>(std::get<0>(dimensionsAux), std::get<1>(dimensionsAux));
            setOrAddMatrixMain(id, res);
            return res;
        }
    }
    return it->second;
}

template <class Toperation>
void MpiMultiplicationEnvironment<Toperation>::PerformCalculations(std::string idA, std::string idB, std::string idC, bool printMatrix)
{
    bool isDistributedA, isDistributedB;
    int rowsAUsed, columnsAUsed, rowsBUsed, columnsBUsed, meshRowSize, meshColumnSize, blockRowSizeA, blockRowSizeB, blockColumnSizeA, blockColumnSizeB;
    int rowsAReal, columnsAReal, rowsBReal, columnsBReal;
    OperationProperties op;
    MatrixMain<Toperation> *ma, *mb, *mc;
    MpiMatrix<Toperation> *mMpiLocalA, *mMpiLocalB;

    ma = getMainMatrix(idA, true);
    mb = getMainMatrix(idB, true);
    ///////////////FALTA COMPROBAR SI SE PUEDE REALIZAR LA OPERACION/////////////////////////////////
    if(!MatrixUtilities<Toperation>::canMultiply(ma->getColumnsReal(),mb->getRowsReal()))
    {
        throw std::invalid_argument("La operacion no se puede realizar porque las columnas no coinciden con las filas. Columnas: " +std::to_string(ma->getColumnsReal())+ ", Filas: "+ std::to_string(mb->getRowsReal()));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

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
        blockRowSizeA = op.blockRowSizeA;
        blockColumnSizeA = op.blockColumnSizeA;
        blockRowSizeB = op.blockRowSizeB;
        blockColumnSizeB = op.blockColumnSizeB;

        if (cpuRank == cpuRoot) //&& printMatrix)
        {
            std::cout << "Ncpus: " << op.cpuSize << ", meshRowSize: " << meshRowSize << ", meshColumnSize: " << meshColumnSize << ", blockRowSizeA: " << blockRowSizeA << ", blockColumnSizeA: " << blockColumnSizeA << ", blockRowSizeB: " << blockRowSizeB << ", blockColumnSizeB: " << blockColumnSizeB << ", rowsA: " << op.rowsA << ", columnsAorRowsB: " << op.columnsAorRowsB << ", columnsB: " << op.columnsB << std::endl;
        }

        ma->setRowsUsed(op.rowsA);
        ma->setColumnsUsed(op.columnsAorRowsB);
        ma->setMatrix(getMatrixGlobalSimplePointer(idA));
        rowsAUsed = ma->getRowsUsed();
        columnsAUsed = ma->getColumnsUsed();
        rowsAReal = ma->getRowsReal();
        columnsAReal = ma->getColumnsReal();

        mb->setRowsUsed(op.columnsAorRowsB);
        mb->setColumnsUsed(op.columnsB);
        mb->setMatrix(getMatrixGlobalSimplePointer(idB));
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
                MatrixUtilities<Toperation>::printMatrix(ma->getRowsReal(), ma->getColumnsReal(), a);
                std::cout << "B-> Rows: " << mb->getRowsReal() << ", Columns: " << mb->getColumnsReal() << ", Matriz B:" << std::endl;
                MatrixUtilities<Toperation>::printMatrix(mb->getRowsReal(), mb->getColumnsReal(), b);
            }
        }
        //Ver que procesos van a realizar el calculo y crear el entorno
        setCommOperation(op.cpuSize);
        if (thisCpuPerformOperation)
        {
            MPI_Comm_size(commOperation, &cpuOperationSize);
            MPI_Comm_rank(commOperation, &cpuRank);
            //Distribucion de las matrices entre los distintos procesos

            mMpiLocalA = new MpiMatrix<Toperation>(cpuOperationSize, cpuRank, meshRowSize, meshColumnSize, blockRowSizeA, blockColumnSizeA, ma, commOperation, basicOperationType);
            mMpiLocalB = new MpiMatrix<Toperation>(cpuOperationSize, cpuRank, meshRowSize, meshColumnSize, blockRowSizeB, blockColumnSizeB, mb, commOperation, basicOperationType);
            mMpiLocalA->mpiDistributeMatrixSendRecv(a, cpuRoot);
            mMpiLocalB->mpiDistributeMatrixSendRecv(b, cpuRoot);

            // MatrixUtilities<Toperation>::debugMatricesLocalDifferentCpus(cpuRank,cpuOperationSize,blockRowSizeA,blockColumnSizeA,mMpiLocalA->getMatricesLocal(),"");
            // sleep(2);
            // MPI_Barrier(commOperation);

            // MatrixUtilities<Toperation>::debugMatricesLocalDifferentCpus(cpuRank,cpuOperationSize,blockRowSizeB,blockColumnSizeB,mMpiLocalB->getMatricesLocal(),"");
            // sleep(2);
            // MPI_Barrier(commOperation);

            ma->setMpiMatrix(mMpiLocalA);
            mb->setMpiMatrix(mMpiLocalB);
        }
    }
    else
    {
        //////////////////////////////////////////POR HACER EL CASO EN EL QUE ESTEN YA DISTRIBUIDAS; AÑADIR ELSES
    }

    //Realizacion de la multiplicacion distribuida
    if (thisCpuPerformOperation)
    {
        Toperation *matrixLocalC = mpiSumma(*mMpiLocalA, *mMpiLocalB, meshRowSize, meshColumnSize);
        //Creacion del objeto local que contiene el resultado local de la operacion y asignacion del resultado a este objeto
        MatrixMain<Toperation> *mc = new MatrixMain<Toperation>(rowsAReal, columnsBReal);
        mc->setRowsUsed(rowsAUsed);
        mc->setColumnsUsed(columnsBUsed);
        MpiMatrix<Toperation> *mMpiLocalC = new MpiMatrix<Toperation>(cpuOperationSize, cpuRank, meshRowSize, meshColumnSize, blockRowSizeA, blockColumnSizeB, mc, commOperation, basicOperationType);
        mMpiLocalC->setMatrixLocal(matrixLocalC);
        mc->setMpiMatrix(mMpiLocalC);
        setOrAddMatrixMain(idC, mc);
    }
}

template <class Toperation>
Toperation *MpiMultiplicationEnvironment<Toperation>::mpiSumma(MpiMatrix<Toperation> matrixLocalA, MpiMatrix<Toperation> matrixLocalB, int meshRowsSize, int meshColumnsSize)
{
    int i, cpuRootIteration;
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
    Toperation *matrixLocalC = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeA, blockColumnsSizeB);
    Toperation *matrixAuxiliarA = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeA, blockColumnsSizeA);
    Toperation *matrixAuxiliarB = MatrixUtilities<Toperation>::matrixMemoryAllocation(blockRowSizeB, blockColumnsSizeB);
    MPI_Comm_group(commOperation, &groupInitial);
    //Conseguir a que columna y fila pertenezco
    int rowColor = matrixLocalA.getRowColor();
    int columnColor = matrixLocalA.getColumnColor();
    //Creacion de los nuevos grupos comunicadores para hacer Broadcast de filas o columnas a los pertenecientes a la malla de misma fila o columna
    int colGroupIndex[meshRowsSize];
    int rowGroupIndex[meshColumnsSize];
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
    for (i = 0; i < meshColumnsSize; i++)
    {
        if (columnColor == (i % meshColumnsSize))
        {
            memcpy(matrixAuxiliarA, matrixLocalA.getMatrixLocal(i / meshColumnsSize), blockSizeA * sizeof(Toperation));
        }
        if (rowColor == (i % meshRowsSize))
        {
            memcpy(matrixAuxiliarB, matrixLocalB.getMatrixLocal(i / meshRowsSize), blockSizeB * sizeof(Toperation));
        }
        MPI_Bcast(matrixAuxiliarA, blockSizeA, basicOperationType, (i % meshColumnsSize), commRow);
        MPI_Bcast(matrixAuxiliarB, blockSizeB, basicOperationType, (i % meshRowsSize), commCol);
        // MatrixUtilities<Toperation>::Multiplicacion(blockRowSizeA, blockRowSizeB, blockColumnsSizeB, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
        MatrixUtilities<Toperation>::matrixBlasMultiplication(blockRowSizeA, blockRowSizeB, blockColumnsSizeB, matrixAuxiliarA, matrixAuxiliarB, matrixLocalC);
        // MatrixUtilities<Toperation>::debugMatrixDifferentCpus(cpuRank,blockRowSizeA,blockColumnsSizeB,matrixLocalC,("Iteracion: "+std::to_string(i)));
        // sleep(2);
        // MPI_Barrier(commOperation);
    }
    //Liberacion de las matrices auxiliares que realizaban computo
    MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarA);
    MatrixUtilities<Toperation>::matrixFree(matrixAuxiliarB);

    return matrixLocalC;
}

template class MpiMultiplicationEnvironment<double>;
template class MpiMultiplicationEnvironment<float>;
