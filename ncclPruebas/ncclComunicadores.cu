#include <stdio.h>
#include <fstream>
#include <random>
#include <unistd.h>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "nccl.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define CUDACHECK(cmd)                                         \
	do                                                         \
	{                                                          \
		cudaError_t e = cmd;                                   \
		if (e != cudaSuccess)                                  \
		{                                                      \
			printf("Failed: Cuda error %s:%d '%s'\n",          \
				   __FILE__, __LINE__, cudaGetErrorString(e)); \
			exit(EXIT_FAILURE);                                \
		}                                                      \
	} while (0)

#define NCCLCHECK(cmd)                                         \
	do                                                         \
	{                                                          \
		ncclResult_t r = cmd;                                  \
		if (r != ncclSuccess)                                  \
		{                                                      \
			printf("Failed, NCCL error %s:%d '%s'\n",          \
				   __FILE__, __LINE__, ncclGetErrorString(r)); \
			exit(EXIT_FAILURE);                                \
		}                                                      \
	} while (0)

#define cudaCalloc(A, B, C) \
    do { \
        cudaError_t __cudaCalloc_err = cudaMalloc(A, B*C); \
        if (__cudaCalloc_err == cudaSuccess) CUDACHECK(cudaMemset(*A, 0, B*C)); \
    } while (0)

using namespace std;


struct OperationProperties
{
    /**
     * @brief Tamaño de las filas de la malla
     * 
     */
    int meshRowSize;
    /**
     * @brief Tamaño de las columnas de la malla
     * 
     */
    int meshColumnSize;
    /**
     * @brief Filas operacionales que tendra la matriz A
     * 
     */
    int rowsA;
    /**
     * @brief Columnas de la Matriz A o Filas de la Matriz B operacionales
     * 
     */
    int columnsAorRowsB;
    /**
     * @brief Columnas operacionales que tendra la matriz B
     * 
     */
    int columnsB;
    /**
     * @brief Numeros de 0s que tendra la matriz operacional al extenderse
     * 
     */
    int numberOf0;
    /**
     * @brief Numero de procesadores que realizaran la operacion de multiplicacion
     * 
     */
    int cpuSize;
    /**
     * @brief Numero de filas que tendra la matriz A de forma local
     * 
     */
    int blockRowSizeA;
    /**
     * @brief Numero de columnas que tendra la matriz A de forma local
     * 
     */
    int blockColumnSizeA;
    /**
     * @brief Numero de filas que tendra la matriz B de forma local
     * 
     */
    int blockRowSizeB;
    /**
     * @brief Numero de columnas que tendra la matriz B de forma local
     * 
     */
    int blockColumnSizeB;
    /**
     * @brief Incida si las propiedades antes indicadas son aptas para el calculo de la multiplicaicon de la matriz
     * 
     */
    bool candidate;
};

struct GpuProperties
{
	int nDevicesGlobal;
	int nDevicesOperation;
	int rankGlobal;
	int rankOperation;
	int rankRow;
	int rankCol;
	int *devicesGlobal;
	int *devicesOperation;
	int *devicesRow;
	int *devicesCol;
	ncclComm_t *commGlobal;
	ncclComm_t *commOperation;
	ncclComm_t *commRow;
	ncclComm_t *commCol;
	cudaStream_t *stream;
	double *matrixDevice;
	int rowSize;
	int colSize;

	GpuProperties(int devicesTotal, int rankGlobal)
	{
		this->nDevicesGlobal = devicesTotal;
		this->rankGlobal = rankGlobal;
	}
};


__global__ void
cudaHelloWorld()
{
	printf("Hola\n");
}

int matrixCalculateIndex(int columnSize, int rowIndex, int columnIndex)
{
	return columnSize * rowIndex + columnIndex;
}

void printMatrix(int rows, int columns, double *M)
{
    int i, j, matrixIndex;
    for (i = 0; i < rows; ++i)
    {
        for (j = 0; j < columns; ++j)
        {
            matrixIndex = IDX2C(i, j,rows);
            cout << M[matrixIndex] << "\t";
        }
        cout << endl;
    }
}

double* matrixMemoryAllocation(int rows, int columns)
{
	double *matrix = (double *)calloc(rows * columns, sizeof(double));
	return matrix;
}


void matrixFree(double *matrix)
{
	free(matrix);
}




double *ReadOrGenerateRandomMatrix(bool isRandom, const char *fileName, int &rows, int &columns, int boundLower, int boundUpper)
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
	double *matrix = matrixMemoryAllocation(rows, columns);
	//Bucle de generacion o lectura de la matrizs
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < columns; j++)
		{
			matrixIndex = IDX2C(i,j,rows);
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

OperationProperties calculateNonEqualMesh(int rowsA, int columnsAorRowsB, int columnsB, int nCpusMesh1, int nCpusMesh2, bool isMeshRow)
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

OperationProperties getMeshAndMatrixSize(int rowsA, int columnsA, int rowsB, int columnsB, int cpuSize)
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



int main(int argc, char *argv[])
{
	cout << fixed;
    cout << setprecision(2);
	//Leer parametros del lanzamiento del programa
	bool printMatrixBool = false;
	vector<string> optionsCmd;
	int rowsA,columnsA,i;
	double *matrixA;
	for (i = 0; i < argc; i++)
	{
		optionsCmd.push_back(string(argv[i]));
	}
	if (std::find(optionsCmd.begin(), optionsCmd.end(), "-h") != optionsCmd.end() || optionsCmd.size() == 1)
	{
		cout << "Uso:\tLas opciones -f y -r no se pueden usar a la vez" << endl;
		cout << "\t-h\tMuestra la ayuda" << endl;
		cout << "\t-p\t(Opcional) Muestra la matriz por pantalla" << endl;
		cout << "\t-f\tLas matrices son leidas de ficheros de texto: -f f1.txt" << endl;
		cout << "\t-r\tLas matrices son generadas de forma aleatoria(m n indican el tamaño de las matrices. bl bu indican de que numero a que numero se genera la matrix .Todos son numeros enteros) -r m n" << endl;
	}
	if (std::find(optionsCmd.begin(), optionsCmd.end(), "-p") != optionsCmd.end())
	{
		printMatrixBool = true;
	}
	auto fOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-f");
	auto rOptionChecker = std::find(optionsCmd.begin(), optionsCmd.end(), "-r");
	if (fOptionChecker != optionsCmd.end() && rOptionChecker != optionsCmd.end())
	{
		cout << "Los parametros -f y -r no se pueden usar a la vez" << endl;
		return -1;
	}
	if (fOptionChecker != optionsCmd.end())
	{
		int fPosition = std::distance(optionsCmd.begin(), fOptionChecker);
		matrixA = ReadOrGenerateRandomMatrix(false, optionsCmd[fPosition + 1].c_str(), rowsA, columnsA, -1, -1);
	}

	if (rOptionChecker != optionsCmd.end())
	{
		int rPosition = std::distance(optionsCmd.begin(), rOptionChecker);
		rowsA=atoi(optionsCmd[rPosition + 1].c_str());
		columnsA=atoi(optionsCmd[rPosition + 2].c_str());
		matrixA = ReadOrGenerateRandomMatrix(true, "", rowsA, columnsA, atoi(optionsCmd[rPosition + 3].c_str()), atoi(optionsCmd[rPosition + 4].c_str()));
	}
	printMatrix(rowsA,columnsA,matrixA);
	int nDevicesGlobal;
	CUDACHECK(cudaGetDeviceCount(&nDevicesGlobal));
	OperationProperties op = getMeshAndMatrixSize(rowsA, columnsA, rowsA, columnsA, nDevicesGlobal);
	//Configuracion del comunicador que tiene a todos los dispositivos
	
	
	GpuProperties *gpusInfo = (GpuProperties *)malloc(sizeof(GpuProperties) * nDevicesGlobal);

	int devicesGlobal[nDevicesGlobal];
	for (int i = 0; i < nDevicesGlobal; i++)
	{
		CUDACHECK(cudaSetDevice(i));
		gpusInfo[i] = GpuProperties(nDevicesGlobal, i);
		gpusInfo[i].stream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
		devicesGlobal[i] = i;
		CUDACHECK(cudaMalloc ((void**)&gpusInfo[i].matrixDevice, op.blockRowSizeA*op.blockColumnSizeA*sizeof(double)));
	}

	ncclComm_t commGlobal[nDevicesGlobal];

	//initializing NCCL
	NCCLCHECK(ncclCommInitAll(commGlobal, nDevicesGlobal, devicesGlobal));

	//Liberar memoria
	for (int i = 0; i < nDevicesGlobal; ++i)
		ncclCommDestroy(commGlobal[i]);
	free(gpusInfo);
	std::cout << "Fin del programa" << std::endl;
	return 0;
}