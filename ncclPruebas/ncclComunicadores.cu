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

struct GpuProperties
{
	int nDevicesGlobal;
	int nDevicesOperation;
	int rankGlobal;
	int nStreams;
	int *devicesGlobal;
	ncclComm_t *commGlobal;
	cudaStream_t *streams;
	cublasHandle_t handle;
	double *matrixDeviceA;
	double *matrixDeviceB;
	double *matrixDeviceC;
	int rowSize;
	int colSize;

	GpuProperties(int devicesTotal, int rankGlobal,int nStreams)
	{
		this->nDevicesGlobal = devicesTotal;
		this->rankGlobal = rankGlobal;
		this->nStreams=nStreams;
	}

	~GpuProperties()
	{
		cudaFree(matrixDeviceA);
		cudaFree(matrixDeviceB);
		cudaFree(matrixDeviceC);
		// int i;
		//No me va eliminarlos
		// for(i=0;i<nStreams;i++)
		// {
		// 	cudaStreamDestroy(streams[i]);
		// }
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
	if(printMatrixBool)
	{
		printMatrix(rowsA,columnsA,matrixA);
	}
	int nDevicesGlobal;
	CUDACHECK(cudaGetDeviceCount(&nDevicesGlobal));
	//Configuracion del comunicador que tiene a todos los dispositivos
	
	
	GpuProperties **gpusInfo= (GpuProperties **)malloc(sizeof(GpuProperties) * nDevicesGlobal);

	int devicesGlobal[nDevicesGlobal];
	vector<cudaEvent_t> startMalloc1,stopMalloc1,startMalloc2,stopMalloc2,startMalloc3,stopMalloc3;
	vector<cudaEvent_t> startMemSet1,stopMemSet1,startMemSet2,stopMemSet2,startMemSet3,stopMemSet3;
	vector<cudaEvent_t> startMemCpy1,stopMemCpy1;
	vector<cudaEvent_t> startStreamCreate1,stopStreamCreate1,startStreamCreate2,stopStreamCreate2,startStreamCreate3,stopStreamCreate3;
	vector<cudaEvent_t> startCublasCreate,stopCublasCreate;
	vector<cudaEvent_t> startDgemm,stopDgemm;

	for (i= 0; i < nDevicesGlobal; i++)
	{
		
		devicesGlobal[i] = i;
		CUDACHECK(cudaSetDevice(i));
		gpusInfo[i] = new GpuProperties(nDevicesGlobal, i,3);
		//Creacion de los eventos
		cudaEvent_t startCC,stopCC,startS1,stopS1,startS2,stopS2,startS3,stopS3,startMa1,stopMa1,startMa2,stopMa2,startMa3,stopMa3,startMs1,stopMs1,startMs2,stopMs2,startMs3,stopMs3,startMc1,stopMc1;
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
		CUDACHECK(cudaEventCreate(&startMalloc1[i]));CUDACHECK(cudaEventCreate(&stopMalloc1[i]));
		CUDACHECK(cudaEventCreate(&startMalloc2[i]));CUDACHECK(cudaEventCreate(&stopMalloc2[i]));
		CUDACHECK(cudaEventCreate(&startMalloc3[i]));CUDACHECK(cudaEventCreate(&stopMalloc3[i]));
		CUDACHECK(cudaEventCreate(&startMemSet1[i]));CUDACHECK(cudaEventCreate(&stopMemSet1[i]));
		CUDACHECK(cudaEventCreate(&startMemSet2[i]));CUDACHECK(cudaEventCreate(&stopMemSet2[i]));
		CUDACHECK(cudaEventCreate(&startMemSet3[i]));CUDACHECK(cudaEventCreate(&stopMemSet3[i]));
		CUDACHECK(cudaEventCreate(&startMemCpy1[i]));CUDACHECK(cudaEventCreate(&stopMemCpy1[i]));
		CUDACHECK(cudaEventCreate(&startCublasCreate[i]));CUDACHECK(cudaEventCreate(&stopCublasCreate[i]));
		CUDACHECK(cudaEventCreate(&startStreamCreate1[i]));CUDACHECK(cudaEventCreate(&stopStreamCreate1[i]));
		CUDACHECK(cudaEventCreate(&startStreamCreate2[i]));CUDACHECK(cudaEventCreate(&stopStreamCreate2[i]));
		CUDACHECK(cudaEventCreate(&startStreamCreate3[i]));CUDACHECK(cudaEventCreate(&stopStreamCreate3[i]));
		//Creacion de streams
		gpusInfo[i]->streams = (cudaStream_t *)malloc(sizeof(cudaStream_t*)*3);
		CUDACHECK(cudaEventRecord(startStreamCreate1[i]));
		CUDACHECK(cudaStreamCreate(&gpusInfo[i]->streams[0]));
		CUDACHECK(cudaEventRecord(stopStreamCreate1[i]));
		CUDACHECK(cudaEventRecord(startStreamCreate2[i]));
		CUDACHECK(cudaStreamCreate(&gpusInfo[i]->streams[1]));
		CUDACHECK(cudaEventRecord(stopStreamCreate2[i]));
		CUDACHECK(cudaEventRecord(startStreamCreate3[i]));
		CUDACHECK(cudaStreamCreate(&gpusInfo[i]->streams[2]));
		CUDACHECK(cudaEventRecord(stopStreamCreate3[i]));

		//Paso de datos a las matrices
		
		CUDACHECK(cudaEventRecord(startMalloc1[i]));
		CUDACHECK(cudaMalloc ((void**)&gpusInfo[i]->matrixDeviceA,rowsA*columnsA*sizeof(double)));
		CUDACHECK(cudaEventRecord(stopMalloc1[i]));
		CUDACHECK(cudaEventRecord(startMemSet1[i],gpusInfo[i]->streams[0]));
		CUDACHECK(cudaMemsetAsync(gpusInfo[i]->matrixDeviceA, 0, sizeof(double)*rowsA*columnsA,gpusInfo[i]->streams[0]));
		CUDACHECK(cudaEventRecord(stopMemSet1[i],gpusInfo[i]->streams[0]));
		CUDACHECK(cudaEventRecord(startMemCpy1[i],gpusInfo[i]->streams[0]));
		CUDACHECK(cudaMemcpyAsync(gpusInfo[i]->matrixDeviceA,matrixA,rowsA*columnsA*sizeof(double),cudaMemcpyHostToDevice,gpusInfo[i]->streams[0]));
		CUDACHECK(cudaEventRecord(stopMemCpy1[i],gpusInfo[i]->streams[0]));
		
		CUDACHECK(cudaEventRecord(startMalloc2[i]));
		CUDACHECK(cudaMalloc ((void**)&gpusInfo[i]->matrixDeviceB, rowsA*columnsA*sizeof(double)));
		CUDACHECK(cudaEventRecord(stopMalloc2[i]));
		CUDACHECK(cudaEventRecord(startMemSet2[i],gpusInfo[i]->streams[1]));
		CUDACHECK(cudaMemsetAsync(gpusInfo[i]->matrixDeviceB, 0, sizeof(double)*rowsA*columnsA,gpusInfo[i]->streams[1]));
		CUDACHECK(cudaEventRecord(stopMemSet2[i],gpusInfo[i]->streams[1]));

		CUDACHECK(cudaEventRecord(startMalloc3[i]));
		CUDACHECK(cudaMalloc ((void**)&gpusInfo[i]->matrixDeviceC, rowsA*columnsA*sizeof(double)));
		CUDACHECK(cudaEventRecord(stopMalloc3[i]));
		CUDACHECK(cudaEventRecord(startMemSet3[i],gpusInfo[i]->streams[2]));
		CUDACHECK(cudaMemsetAsync(gpusInfo[i]->matrixDeviceC, 0, sizeof(double)*rowsA*columnsA,gpusInfo[i]->streams[2]));
		CUDACHECK(cudaEventRecord(stopMemSet3[i],gpusInfo[i]->streams[2]));
		
		CUDACHECK(cudaEventRecord(startCublasCreate[i]));
		CUBLASCHECK(cublasCreate(&gpusInfo[i]->handle));
		CUDACHECK(cudaEventRecord(stopCublasCreate[i]));
		
	}
	cudaDeviceSynchronize();
	ncclComm_t commGlobal[nDevicesGlobal];
	NCCLCHECK(ncclCommInitAll(commGlobal, nDevicesGlobal, devicesGlobal));
	float timeMallocTotal=0,timeMemsetTotal=0,timeMemcpyTotal=0,timeCublasCreateTotal=0,timeStreamTotal=0;
	for(i=0;i<nDevicesGlobal;i++)
	{
		float timeM1,timeM2,timeM3,timeS1,timeS2,timeS3,timeSet1,timeSet2,timeSet3,timeC1,timeCC;
		CUDACHECK(cudaEventSynchronize(stopMalloc1[i]));CUDACHECK(cudaEventSynchronize(stopMalloc2[i]));CUDACHECK(cudaEventSynchronize(stopMalloc3[i]));
		CUDACHECK(cudaEventSynchronize(stopStreamCreate1[i]));CUDACHECK(cudaEventSynchronize(stopStreamCreate2[i]));CUDACHECK(cudaEventSynchronize(stopStreamCreate3[i]));
		CUDACHECK(cudaEventSynchronize(stopMemSet1[i]));CUDACHECK(cudaEventSynchronize(stopMemSet2[i]));CUDACHECK(cudaEventSynchronize(stopMemSet3[i]));
		CUDACHECK(cudaEventSynchronize(stopMemCpy1[i]));CUDACHECK(cudaEventSynchronize(stopCublasCreate[i]));
		CUDACHECK(cudaEventElapsedTime(&timeM1, startMalloc1[i], stopMalloc1[i]));
		CUDACHECK(cudaEventElapsedTime(&timeM2, startMalloc2[i], stopMalloc2[i]));
		CUDACHECK(cudaEventElapsedTime(&timeM3, startMalloc3[i], stopMalloc3[i]));
		CUDACHECK(cudaEventElapsedTime(&timeS1, startStreamCreate1[i], stopStreamCreate1[i]));
		CUDACHECK(cudaEventElapsedTime(&timeS2, startStreamCreate2[i], stopStreamCreate2[i]));
		CUDACHECK(cudaEventElapsedTime(&timeS3, startStreamCreate3[i], stopStreamCreate3[i]));
		CUDACHECK(cudaEventElapsedTime(&timeSet1, startMemSet1[i], stopMemSet1[i]));
		CUDACHECK(cudaEventElapsedTime(&timeSet2, startMemSet2[i], stopMemSet2[i]));
		CUDACHECK(cudaEventElapsedTime(&timeSet3, startMemSet3[i], stopMemSet3[i]));
		CUDACHECK(cudaEventElapsedTime(&timeC1, startMemCpy1[i], stopMemCpy1[i]));
		CUDACHECK(cudaEventElapsedTime(&timeCC, startCublasCreate[i], stopCublasCreate[i]));

		timeMallocTotal+=timeM1+timeM2+timeM3;
		timeMemsetTotal+=timeSet1,timeSet2,timeSet3;
		timeStreamTotal+=timeS1+timeS2+timeS3;
		timeMemcpyTotal+=timeC1;
		timeCublasCreateTotal+=timeCC;

		printf("Dispositivo %d\n",i);
		printf("Malloc 1: %f\n",timeM1);
		printf("Malloc 2: %f\n",timeM2);
		printf("Malloc 3: %f\n",timeM3);
		printf("MemSet 1: %f\n",timeSet1);
		printf("MemSet 2: %f\n",timeSet2);
		printf("MemSet 3: %f\n",timeSet3);
		printf("MemCpy1: %f\n",timeC1);
		printf("Stream 1: %f\n",timeS1);
		printf("Stream 2: %f\n",timeS2);
		printf("Stream 3: %f\n",timeS3);
		printf("CublasCreate: %f\n",timeCC);
		printf("\n");
	}
	//Enviamos la informacion
	cudaEvent_t startBr,stopBr;
	CUDACHECK(cudaEventCreate(&startBr));
	CUDACHECK(cudaEventCreate(&stopBr));
	CUDACHECK(cudaEventRecord(startBr,gpusInfo[3]->streams[1]));
	NCCLCHECK(ncclGroupStart());
	for(i=0;i<nDevicesGlobal;i++)
	{
		NCCLCHECK(ncclBroadcast(gpusInfo[0]->matrixDeviceA,gpusInfo[i]->matrixDeviceB,rowsA*columnsA,ncclDouble,0,commGlobal[i],gpusInfo[i]->streams[1]));
	}
	NCCLCHECK(ncclGroupEnd());
	CUDACHECK(cudaEventRecord(stopBr,gpusInfo[3]->streams[1]));
	//Esperamos la recepcion
	double alfa=1;double beta=0;
	for (i = 0; i < nDevicesGlobal; ++i) 
	{
		CUDACHECK(cudaSetDevice(i));
		cudaEvent_t stgemm,spgemm;
		startDgemm.push_back(stgemm);stopDgemm.push_back(spgemm);
		CUDACHECK(cudaEventCreate(&startDgemm[i]));CUDACHECK(cudaEventCreate(&stopDgemm[i]));
		CUDACHECK(cudaStreamSynchronize(gpusInfo[i]->streams[1]));
		CUDACHECK(cudaEventRecord(startDgemm[i],gpusInfo[i]->streams[1]));
		CUBLASCHECK(cublasDgemm(gpusInfo[i]->handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, rowsA, rowsA, &alfa, gpusInfo[i]->matrixDeviceA, rowsA, gpusInfo[i]->matrixDeviceB, rowsA, &beta, gpusInfo[i]->matrixDeviceC, rowsA));
		CUDACHECK(cudaEventRecord(stopDgemm[i],gpusInfo[i]->streams[1]));
	}
	float tiempoComunicacion;
	CUDACHECK(cudaEventElapsedTime(&tiempoComunicacion, startBr, stopBr));
	double tiempoTotalMultiplicacion=0;
	float BandRateTotal=0;
	for (i = 0; i < nDevicesGlobal; ++i) 
	{
		float timeMul,BandRate;
		CUDACHECK(cudaEventSynchronize(stopDgemm[i]));
		CUDACHECK(cudaEventElapsedTime(&timeMul, startDgemm[i], stopDgemm[i]));
		tiempoTotalMultiplicacion+=timeMul;
		BandRate=(rowsA*rowsA)*8*3/timeMul/1e6;
		BandRateTotal+=BandRate;

		printf("Dispositivo %d\n",i);
		printf("Multiplicacion: %f\n",timeMul);
		printf("Bandwidth: %f\n",BandRate);
	}
	if(printMatrixBool)
	{
		cudaPrintMatrix<<<1,1,1>>>(rowsA,rowsA,gpusInfo[3]->matrixDeviceC);
	}
	float tiempoMedioMalloc,tiempoMedioMemSet,tiempoMedioMemCpy,tiempoMedioCublasCreate,tiempoMedioStream,tiempoMedioMul;
	tiempoMedioMalloc=timeMallocTotal/(stopMalloc1.size()+stopMalloc2.size()+stopMalloc3.size());
	tiempoMedioMemSet=timeMemsetTotal/(stopMemSet1.size()+stopMemSet2.size()+stopMemSet3.size());
	tiempoMedioMemCpy=timeMemcpyTotal/(stopMemCpy1.size());
	tiempoMedioCublasCreate=timeCublasCreateTotal/stopCublasCreate.size();
	tiempoMedioMul=tiempoTotalMultiplicacion/stopDgemm.size();
	tiempoMedioStream=timeStreamTotal/(stopStreamCreate1.size()+stopStreamCreate2.size()+stopStreamCreate3.size());
	printf("Tiempo total Malloc: %f, tiempo medio: %f\n",timeMallocTotal,tiempoMedioMalloc);
	printf("Tiempo total Memset: %f, tiempo medio: %f\n",timeMemsetTotal,tiempoMedioMemSet);
	printf("Tiempo total Memcpy: %f, tiempo medio: %f\n",timeMemcpyTotal,tiempoMedioMemCpy);
	printf("Tiempo total Stream: %f, tiempo medio: %f\n",timeStreamTotal,tiempoMedioStream);
	printf("Tiempo total Multiplicar: %f, tiempo medio: %f\n",tiempoTotalMultiplicacion,tiempoMedioMul);
	printf("Tiempo total CublasCreate: %f, tiempo medio: %f\n",timeCublasCreateTotal,tiempoMedioCublasCreate);
	printf("Tiempo del Broadcast %f\n",tiempoTotalMultiplicacion);
	printf("Todos los tiempos han sido medidos en milisegundos\n");
	printf("Bandwidth medio %f GB/s\n",BandRateTotal/stopDgemm.size());



	//Liberar memoria
	for (i = 0; i < nDevicesGlobal; ++i)
	{
		ncclCommDestroy(commGlobal[i]);
	}
	for(i=0;i<nDevicesGlobal;i++)
	{
		CUBLASCHECK(cublasDestroy(gpusInfo[i]->handle));
		CUDACHECK(cudaEventDestroy(startMemCpy1[i]));
		CUDACHECK(cudaEventDestroy(stopMemCpy1[i]));
		CUDACHECK(cudaEventDestroy(stopCublasCreate[i]));
		CUDACHECK(cudaEventDestroy(startCublasCreate[i]));
	}
	delete gpusInfo;
	//Destruir eventos
	std::cout << "Fin del programa" << std::endl;
	return 0;
}