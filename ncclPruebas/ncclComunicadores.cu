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

#ifdef CUBLAS_API_H_
// cuBLAS API errors
static const char *_cudaGetErrorEnum(cublasStatus_t error)
{
    switch (error)
    {
        case CUBLAS_STATUS_SUCCESS:
            return "CUBLAS_STATUS_SUCCESS";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "CUBLAS_STATUS_NOT_INITIALIZED";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "CUBLAS_STATUS_ALLOC_FAILED";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "CUBLAS_STATUS_INVALID_VALUE";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "CUBLAS_STATUS_ARCH_MISMATCH";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "CUBLAS_STATUS_MAPPING_ERROR";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "CUBLAS_STATUS_EXECUTION_FAILED";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "CUBLAS_STATUS_INTERNAL_ERROR";
    }

    return "<unknown>";
}
#endif


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

#define CUBLASCHECK(cmd)                                         \
do                                                         \
{                                                          \
	cublasStatus_t s = cmd;                                   \
	if (s != CUBLAS_STATUS_SUCCESS)                                  \
	{                                                      \
		printf("Failed: Cublas error %s:%d '%s'\n",          \
				__FILE__, __LINE__, _cudaGetErrorEnum(s)); \
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
	cudaStream_t *streams;
	cublasHandle_t handle;
	double *matrixDeviceA;
	double *matrixDeviceB;
	double *matrixDeviceC;
	int rowSize;
	int colSize;

	GpuProperties(int devicesTotal, int rankGlobal)
	{
		this->nDevicesGlobal = devicesTotal;
		this->rankGlobal = rankGlobal;
	}
};

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


__global__ void
cudaPrintMatrix(int rows,int columns,double* matrix)
{
	for(int i =0;i<rows;i++)
	{
		for(int j=0;j<columns;j++)
		{
			printf("%.2lf\t",matrix[IDX2C(i,j,rows)]);
		}
		printf("\n");
	}
}

void matrixDeviceAllocation(int rows, int columns, double* matrixDevice)
{
	CUDACHECK(cudaMalloc((void**)&matrixDevice, rows*columns*sizeof(double)));
    CUDACHECK(cudaMemset(matrixDevice, 0, sizeof(double)*rows*columns));
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
	if(printMatrix)
	{
		printMatrix(rowsA,columnsA,matrixA);
	}
	int nDevicesGlobal;
	CUDACHECK(cudaGetDeviceCount(&nDevicesGlobal));
	OperationProperties op = getMeshAndMatrixSize(rowsA, columnsA, rowsA, columnsA, nDevicesGlobal);
	//Configuracion del comunicador que tiene a todos los dispositivos
	
	
	GpuProperties *gpusInfo = (GpuProperties *)malloc(sizeof(GpuProperties) * nDevicesGlobal);

	int devicesGlobal[nDevicesGlobal];
	vector<cudaEvent_t> startMalloc1,stopMalloc1,startMalloc2,stopMalloc2,startMalloc3,stopMalloc3;
	vector<cudaEvent_t> startMemSet1,stopMemSet1,startMemSet2,stopMemSet2,startMemSet3,stopMemSet3;
	vector<cudaEvent_t> startMemCpy1,stopMemCpy1,startMemCpy2,stopMemCpy2;
	vector<cudaEvent_t> startStreamCreate1,stopStreamCreate1,startStreamCreate2,stopStreamCreate2,startStreamCreate3,stopStreamCreate3;
	vector<cudaEvent_t> startCublasCreate,stopCublasCreate;;

	for (i= 0; i < nDevicesGlobal; i++)
	{
		
		devicesGlobal[i] = i;
		CUDACHECK(cudaSetDevice(i));
		gpusInfo[i] = GpuProperties(nDevicesGlobal, i);
		//Creacion de los eventos
		cudaEvent_t startCC,stopCC,startS1,stopS1,startS2,stopS2,startS3,stopS3,startMa1,stopMa1,startMa2,stopMa2,startMa3,stopMa3,startMs1,stopMs1,startMs2,stopMs2,startMs3,stopMs3,startMc1,stopMc1,startMc2,stopMc2;
		startCublasCreate.push_back(startCC);stopCublasCreate.push_back(stopCC);
		startStreamCreate1.push_back(startS1);stopStreamCreate1.push_back(stopS1);
		startStreamCreate2.push_back(startS2);stopStreamCreate2.push_back(stopS2);
		startStreamCreate3.push_back(startS3);stopStreamCreate3.push_back(stopS3);
		startMalloc1.push_back(startMa1);stopMalloc1.push_back(stopMa1);
		startMalloc2.push_back(startMa2);stopMalloc2.push_back(stopMa2);
		startMalloc3.push_back(startMa3);stopMalloc3.push_back(stopMa3);
		startMemSet1.push_back(startMs1);stopMemSet1.push_back(stopMs1);
		startMemSet2.push_back(startMs2);stopMemSet2.push_back(stopMs2);
		startMemSet3.push_back(startMs3);stopMemSet3.push_back(stopMs3);
		startMemCpy1.push_back(startMc1);stopMemCpy1.push_back(stopMc1);
		startMemCpy2.push_back(startMc2);stopMemCpy2.push_back(stopMc2);
		CUDACHECK(cudaEventCreate(&startMalloc1[i]));CUDACHECK(cudaEventCreate(&stopMalloc1[i]));
		CUDACHECK(cudaEventCreate(&startMalloc2[i]));CUDACHECK(cudaEventCreate(&stopMalloc2[i]));
		CUDACHECK(cudaEventCreate(&startMalloc3[i]));CUDACHECK(cudaEventCreate(&stopMalloc3[i]));
		CUDACHECK(cudaEventCreate(&startMemSet1[i]));CUDACHECK(cudaEventCreate(&stopMemSet1[i]));
		CUDACHECK(cudaEventCreate(&startMemSet2[i]));CUDACHECK(cudaEventCreate(&stopMemSet2[i]));
		CUDACHECK(cudaEventCreate(&startMemSet3[i]));CUDACHECK(cudaEventCreate(&stopMemSet3[i]));
		CUDACHECK(cudaEventCreate(&startMemCpy1[i]));CUDACHECK(cudaEventCreate(&stopMemCpy1[i]));
		CUDACHECK(cudaEventCreate(&startMemCpy2[i]));CUDACHECK(cudaEventCreate(&stopMemCpy2[i]));
		CUDACHECK(cudaEventCreate(&startCublasCreate[i]));CUDACHECK(cudaEventCreate(&stopCublasCreate[i]));
		CUDACHECK(cudaEventCreate(&startStreamCreate1[i]));CUDACHECK(cudaEventCreate(&stopStreamCreate1[i]));
		CUDACHECK(cudaEventCreate(&startStreamCreate2[i]));CUDACHECK(cudaEventCreate(&stopStreamCreate2[i]));
		CUDACHECK(cudaEventCreate(&startStreamCreate3[i]));CUDACHECK(cudaEventCreate(&stopStreamCreate3[i]));
		//Creacion de streams
		gpusInfo[i].streams = (cudaStream_t *)malloc(sizeof(cudaStream_t*)*3);
		CUDACHECK(cudaEventRecord(startStreamCreate1[i]));
		CUDACHECK(cudaStreamCreate(&gpusInfo[i].streams[0]));
		CUDACHECK(cudaEventRecord(stopStreamCreate1[i]));
		CUDACHECK(cudaEventRecord(startStreamCreate2[i]));
		CUDACHECK(cudaStreamCreate(&gpusInfo[i].streams[1]));
		CUDACHECK(cudaEventRecord(stopStreamCreate2[i]));
		CUDACHECK(cudaEventRecord(startStreamCreate3[i]));
		CUDACHECK(cudaStreamCreate(&gpusInfo[i].streams[2]));
		CUDACHECK(cudaEventRecord(stopStreamCreate3[i]));

		//Paso de datos a las matrices
		
		// matrixDeviceAllocation(rowsA,columnsA,gpusInfo[i].matrixDeviceA);//ME FALLA Y NO SE PORQUE 'all CUDA-capable devices are busy or unavailable'
		CUDACHECK(cudaEventRecord(startMalloc1[i]));
		CUDACHECK(cudaMalloc ((void**)&gpusInfo[i].matrixDeviceA,rowsA*columnsA*sizeof(double)));
		CUDACHECK(cudaEventRecord(stopMalloc1[i]));
		CUDACHECK(cudaEventRecord(startMemSet1[i]));
		CUDACHECK(cudaMemsetAsync(gpusInfo[i].matrixDeviceA, 0, sizeof(double)*rowsA*columnsA,gpusInfo[i].streams[0]));
		CUDACHECK(cudaEventRecord(stopMemSet1[i]));
		CUDACHECK(cudaEventRecord(startMemCpy1[i]));
		CUDACHECK(cudaMemcpyAsync(gpusInfo[i].matrixDeviceA,matrixA,rowsA*columnsA*sizeof(double),cudaMemcpyHostToDevice,gpusInfo[i].streams[0]));
		CUDACHECK(cudaEventRecord(stopMemCpy1[i]));
		
		CUDACHECK(cudaEventRecord(startMalloc2[i]));
		CUDACHECK(cudaMalloc ((void**)&gpusInfo[i].matrixDeviceB, rowsA*columnsA*sizeof(double)));
		CUDACHECK(cudaEventRecord(stopMalloc2[i]));
		CUDACHECK(cudaEventRecord(startMemSet2[i]));
		CUDACHECK(cudaMemsetAsync(gpusInfo[i].matrixDeviceB, 0, sizeof(double)*rowsA*columnsA,gpusInfo[i].streams[1]));
		CUDACHECK(cudaEventRecord(stopMemSet2[i]));
		CUDACHECK(cudaEventRecord(startMemCpy2[i]));
		CUDACHECK(cudaMemcpyAsync(gpusInfo[i].matrixDeviceB,matrixA,rowsA*columnsA*sizeof(double),cudaMemcpyHostToDevice,gpusInfo[i].streams[1]));
		CUDACHECK(cudaEventRecord(stopMemCpy2[i]));

		CUDACHECK(cudaEventRecord(startMalloc3[i]));
		CUDACHECK(cudaMalloc ((void**)&gpusInfo[i].matrixDeviceC, rowsA*columnsA*sizeof(double)));
		CUDACHECK(cudaEventRecord(stopMalloc3[i]));
		CUDACHECK(cudaEventRecord(startMemSet3[i]));
		CUDACHECK(cudaMemsetAsync(gpusInfo[i].matrixDeviceC, 0, sizeof(double)*rowsA*columnsA,gpusInfo[i].streams[2]));
		CUDACHECK(cudaEventRecord(stopMemSet3[i]));
		
		CUDACHECK(cudaEventRecord(startCublasCreate[i]));
		CUBLASCHECK(cublasCreate(&gpusInfo[i].handle));
		CUDACHECK(cudaEventRecord(stopCublasCreate[i]));

	}
	ncclComm_t commGlobal[nDevicesGlobal];
	NCCLCHECK(ncclCommInitAll(commGlobal, nDevicesGlobal, devicesGlobal));
	float timeMallocTotal,timeMemsetTotal,timeMemcpyTotal,timeCublasCreateTotal,timeStreamTotal;
	for(i=0;i<nDevicesGlobal;i++)
	{
		float timeM1,timeM2,timeM3,timeS1,timeS2,timeS3,timeSet1,timeSet2,timeSet3,timeC1,timeC2,timeCC;
		cudaEventSynchronize(stopMalloc1[i]);cudaEventSynchronize(stopMalloc2[i]);cudaEventSynchronize(stopMalloc3[i]);
		cudaEventSynchronize(stopStreamCreate1[i]);cudaEventSynchronize(stopStreamCreate2[i]);cudaEventSynchronize(stopStreamCreate3[i]);
		cudaEventSynchronize(stopMemSet1[i]);cudaEventSynchronize(stopMemSet2[i]);cudaEventSynchronize(stopMemSet2[i]);
		cudaEventSynchronize(stopMemCpy1[i]);cudaEventSynchronize(stopMemCpy1[i]);
		cudaEventSynchronize(stopCublasCreate[i]);
		cudaEventElapsedTime(&timeM1, startMalloc1[i], stopMalloc1[i]);
		cudaEventElapsedTime(&timeM2, startMalloc2[i], stopMalloc2[i]);
		cudaEventElapsedTime(&timeM3, startMalloc3[i], stopMalloc3[i]);
		cudaEventElapsedTime(&timeS1, startStreamCreate1[i], stopStreamCreate1[i]);
		cudaEventElapsedTime(&timeS2, startStreamCreate2[i], stopStreamCreate2[i]);
		cudaEventElapsedTime(&timeS3, startStreamCreate3[i], stopStreamCreate3[i]);
		cudaEventElapsedTime(&timeSet1, startMemSet1[i], stopMemSet1[i]);
		cudaEventElapsedTime(&timeSet2, startMemSet2[i], stopMemSet2[i]);
		cudaEventElapsedTime(&timeSet3, startMemSet3[i], stopMemSet3[i]);
		cudaEventElapsedTime(&timeC1, startMemCpy1[i], stopMemCpy1[i]);
		cudaEventElapsedTime(&timeC2, startMemCpy2[i], stopMemCpy2[i]);
		cudaEventElapsedTime(&timeCC, startCublasCreate[i], stopCublasCreate[i]);

		timeMallocTotal+=timeM1+timeM2+timeM3;
		timeMemsetTotal+=timeSet1,timeSet2,timeSet3;
		timeStreamTotal+=timeS1+timeS2+timeS3;
		timeMemcpyTotal+=timeC1+timeC2;
		timeCublasCreateTotal+=timeCC;

		printf("Dispositivo %d\n",i);
		printf("Malloc 1: %f\n",timeM1);
		printf("Malloc 2: %f\n",timeM2);
		printf("Malloc 3: %f\n",timeM3);
		printf("MemSet 1: %f\n",timeSet1);
		printf("MemSet 2: %f\n",timeSet2);
		printf("MemSet 3: %f\n",timeSet3);
		printf("MemCpy1: %f\n",timeC1);
		printf("MemCpy2: %f\n",timeC2);
		printf("CublasCreate: %f\n",timeCC);
		printf("Stream 1: %f\n",timeS1);
		printf("Stream 2: %f\n",timeS2);
		printf("Stream 3: %f\n",timeS3);
		printf("\n");
	}
	float tiempoMedioMalloc,tiempoMedioMemSet,tiempoMedioMemCpy,tiempoMedioCublasCreate,tiempoMedioStream;
	tiempoMedioMalloc=timeMallocTotal/(stopMalloc1.size()+stopMalloc2.size()+stopMalloc3.size());
	tiempoMedioMemSet=timeMemsetTotal/(stopMemSet1.size()+stopMemSet2.size()+stopMemSet3.size());
	tiempoMedioMemCpy=timeMemcpyTotal/(stopMemCpy1.size()+stopMemCpy1.size());
	tiempoMedioCublasCreate=timeCublasCreateTotal/stopCublasCreate.size();
	tiempoMedioStream=timeStreamTotal/(stopStreamCreate1.size()+stopStreamCreate2.size()+stopStreamCreate3.size());
	printf("Tiempo total Malloc: %f, tiempo medio: %f\n",timeMallocTotal,tiempoMedioMalloc);
	printf("Tiempo total Memset: %f, tiempo medio: %f\n",timeMemsetTotal,tiempoMedioMemSet);
	printf("Tiempo total Memcpy: %f, tiempo medio: %f\n",timeMemcpyTotal,tiempoMedioMemCpy);
	printf("Tiempo total Stream: %f, tiempo medio: %f\n",timeStreamTotal,tiempoMedioStream);
	printf("Tiempo total CublasCreate: %f, tiempo medio: %f\n",timeCublasCreateTotal,tiempoMedioCublasCreate);
	//Liberar memoria
	for (int i = 0; i < nDevicesGlobal; ++i)
	{
		ncclCommDestroy(commGlobal[i]);
	}
	free(gpusInfo);
	//Destruir eventos
	std::cout << "Fin del programa" << std::endl;
	return 0;
}