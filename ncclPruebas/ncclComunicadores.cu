#include <stdio.h>
#include <fstream>
#include <random>
#include <unistd.h>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cblas.h>
#include "cuda_runtime.h"
#include <cublas_v2.h>
#include "nccl.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>

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

static double firstcall=0.0;

int ctimer(double *elapsed, double *ucpu, double *scpu ) {

  struct timeval tm;
  struct timezone tz;
  struct tms sistema;
  double usegs;

  gettimeofday(&tm, &tz);
  times(&sistema);

  usegs = tm.tv_usec+tm.tv_sec*1E6;

  if (firstcall)  {
    *elapsed = usegs - firstcall;
    firstcall = 0.0;
  } else {
    *elapsed = 0.0;
    //*ucpu = tm.tv_usec;
    //*scpu = ;
    firstcall = usegs;
  }

  *elapsed = *elapsed/1E6;
  *ucpu = (double)sistema.tms_utime/(double)CLOCKS_PER_SEC*1E4;
  *scpu = (double)sistema.tms_stime/(double)CLOCKS_PER_SEC*1E4;

  return 0;
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

bool checkEqualityOfMatrices(double *A, double *B, int rows, int columns)
{
    int i, j;
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < columns; j++)
        {
            if (fabs(A[i * columns + j] - B[i * columns + j]) > 0.000001)
            {
				return false;
			}
        }
    }
    return true;
}

void matrixBlasMultiplication(int rowsA, int columnsAorRowsB, int columnsB, double *A, double *B, double *C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rowsA, columnsB, columnsAorRowsB, 1.0, (double*)A, columnsAorRowsB, (double*)B, columnsB, 1.0, (double*)C, columnsB);
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
	if (std::find(optionsCmd.begin(), optionsCmd.end(), "-h") != optionsCmd.end() || (optionsCmd.size() != 4 && optionsCmd.size() != 5))
	{
		cout << "\t-h\tMuestra la ayuda" << endl;
		cout << "\t-p\t(Opcional) Muestra la matriz por pantalla" << endl;
		cout << "\t m tamaño de la matriz, bl bu indican de que numero a que numero se genera la matrix .Todos son numeros enteros.\nParametros de ejecucion m bl bu [-p]" << endl;
		return -1;
	}
	if (std::find(optionsCmd.begin(), optionsCmd.end(), "-p") != optionsCmd.end())
	{
		printMatrixBool = true;
	}
	rowsA=atoi(optionsCmd[1].c_str());
	columnsA=atoi(optionsCmd[1].c_str());

	matrixA = ReadOrGenerateRandomMatrix(true, "", rowsA, columnsA, atoi(optionsCmd[2].c_str()), atoi(optionsCmd[3].c_str()));
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
	vector<cudaEvent_t> startMemCpy2,stopMemCpy2;

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
		CUBLASCHECK(cublasSetStream(gpusInfo[i]->handle,gpusInfo[i]->streams[2]));
		
	}
	cudaDeviceSynchronize();
	double elapsedComm, ucpuComm, scpuComm;
	ctimer(&elapsedComm, &ucpuComm, &scpuComm);
	ncclComm_t commGlobal[nDevicesGlobal];
	NCCLCHECK(ncclCommInitAll(commGlobal, nDevicesGlobal, devicesGlobal));
	ctimer(&elapsedComm, &ucpuComm, &scpuComm);
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
	cudaSetDevice(0);
	cudaEvent_t startBr,stopBr;
	CUDACHECK(cudaEventCreate(&startBr));
	CUDACHECK(cudaEventCreate(&stopBr));
	CUDACHECK(cudaEventRecord(startBr,gpusInfo[0]->streams[1]));
	NCCLCHECK(ncclGroupStart());
	for(i=0;i<nDevicesGlobal;i++)
	{
		NCCLCHECK(ncclBroadcast(gpusInfo[0]->matrixDeviceA,gpusInfo[i]->matrixDeviceB,rowsA*columnsA,ncclDouble,0,commGlobal[i],gpusInfo[i]->streams[1]));
	}
	NCCLCHECK(ncclGroupEnd());
	CUDACHECK(cudaEventRecord(stopBr,gpusInfo[0]->streams[1]));
	//Esperamos la recepcion
	double alfa=1;double beta=0;
	int j,nMultiplications=10;
	int generalIndexMult=0;
	for (i = 0; i < nDevicesGlobal; ++i) 
	{
		CUDACHECK(cudaSetDevice(i));
		for(j=0;j<nMultiplications;j++)
		{
			cudaEvent_t stgemm,spgemm;
			startDgemm.push_back(stgemm);stopDgemm.push_back(spgemm);
			CUDACHECK(cudaEventCreate(&startDgemm[generalIndexMult]));CUDACHECK(cudaEventCreate(&stopDgemm[generalIndexMult]));
			CUDACHECK(cudaStreamSynchronize(gpusInfo[i]->streams[1]));
			CUDACHECK(cudaEventRecord(startDgemm[generalIndexMult],gpusInfo[i]->streams[2]));
			CUBLASCHECK(cublasDgemm(gpusInfo[i]->handle, CUBLAS_OP_N, CUBLAS_OP_N, rowsA, rowsA, rowsA, &alfa, gpusInfo[i]->matrixDeviceA, rowsA, gpusInfo[i]->matrixDeviceB, rowsA, &beta, gpusInfo[i]->matrixDeviceC, rowsA));
			CUDACHECK(cudaEventRecord(stopDgemm[generalIndexMult],gpusInfo[i]->streams[2]));
			generalIndexMult++;
		}
		
	}
	float tiempoComunicacion;
	CUDACHECK(cudaEventElapsedTime(&tiempoComunicacion, startBr, stopBr));
	double tiempoTotalMultiplicacion=0;
	float timeMul;
	generalIndexMult=0;
	for (i = 0; i < nDevicesGlobal; ++i) 
	{
		for(j=0;j<nMultiplications;j++)
		{
			CUDACHECK(cudaEventSynchronize(stopDgemm[generalIndexMult]));
			CUDACHECK(cudaEventElapsedTime(&timeMul, startDgemm[generalIndexMult], stopDgemm[generalIndexMult]));
			tiempoTotalMultiplicacion+=timeMul;
			generalIndexMult++;
		}
		
	}
	cudaSetDevice(0);
	if(printMatrixBool)
	{
		cudaPrintMatrix<<<1,1,1>>>(rowsA,rowsA,gpusInfo[0]->matrixDeviceC);
	}
	//Recuperacion de la matriz a la cpu
	vector<double*> recoveredCs;
	for(i=0;i<nDevicesGlobal;i++)
	{
		cudaSetDevice(i);
		cudaEvent_t startMcpy2,stopMcpy2;
		double* matrixRecovered=matrixMemoryAllocation(rowsA,columnsA);
		recoveredCs.push_back(matrixRecovered);
		startMemCpy2.push_back(startMcpy2);stopMemCpy2.push_back(stopMcpy2);
		CUDACHECK(cudaEventCreate(&startMemCpy2[i]));CUDACHECK(cudaEventCreate(&stopMemCpy2[i]));
		CUDACHECK(cudaEventRecord(startMemCpy2[i],gpusInfo[i]->streams[0]));
		CUDACHECK(cudaMemcpyAsync(recoveredCs[i],gpusInfo[i]->matrixDeviceC,rowsA*columnsA*sizeof(double),cudaMemcpyDeviceToHost,gpusInfo[i]->streams[0]));
		CUDACHECK(cudaEventRecord(stopMemCpy2[i],gpusInfo[i]->streams[0]));

	}
	double timeRecTotal;
	std::cout<<std::endl<<"Tiempo de recuperar matriz del dispositivo al host:"<<std::endl;
	for (i = 0; i < nDevicesGlobal; ++i) 
	{
		float timeRec;
		CUDACHECK(cudaEventSynchronize(stopMemCpy2[i]));
		CUDACHECK(cudaEventElapsedTime(&timeRec, startMemCpy2[i], stopMemCpy2[i]));
		timeRecTotal+=timeRec;

		printf("Dispositivo %d\n",i);
		printf("Tiempo de recuperacion: %f\n",timeRec);
	}
	std::cout<<std::endl<<"Metricas totales y medias: "<<std::endl;
	//Mostrar tiempos
	float tiempoMedioMalloc,tiempoMedioMemSet,tiempoMedioMemCpy,tiempoMedioCublasCreate,tiempoMedioStream,tiempoMedioMul,timeRecMedio;
	tiempoMedioMalloc=timeMallocTotal/(stopMalloc1.size()+stopMalloc2.size()+stopMalloc3.size());
	tiempoMedioMemSet=timeMemsetTotal/(stopMemSet1.size()+stopMemSet2.size()+stopMemSet3.size());
	tiempoMedioMemCpy=timeMemcpyTotal/(stopMemCpy1.size());
	tiempoMedioCublasCreate=timeCublasCreateTotal/stopCublasCreate.size();
	tiempoMedioMul=tiempoTotalMultiplicacion/stopDgemm.size();
	tiempoMedioStream=timeStreamTotal/(stopStreamCreate1.size()+stopStreamCreate2.size()+stopStreamCreate3.size());
	timeRecMedio=timeRecTotal/stopMemCpy2.size();
	printf("Tiempo total Malloc: %f, tiempo medio: %f\n",timeMallocTotal,tiempoMedioMalloc);
	printf("Tiempo total Memset: %f, tiempo medio: %f\n",timeMemsetTotal,tiempoMedioMemSet);
	printf("Tiempo total Memcpy: %f, tiempo medio: %f\n",timeMemcpyTotal,tiempoMedioMemCpy);
	printf("Tiempo total Memcpy rec: %f, tiempo medio: %f\n",timeRecTotal,timeRecMedio);
	printf("Tiempo total Stream: %f, tiempo medio: %f\n",timeStreamTotal,tiempoMedioStream);
	printf("Tiempo total Multiplicar: %f, tiempo medio: %f\n",tiempoTotalMultiplicacion,tiempoMedioMul);
	printf("Tiempo total CublasCreate: %f, tiempo medio: %f\n",timeCublasCreateTotal,tiempoMedioCublasCreate);
	printf("Tiempo del Broadcast %f\n",tiempoTotalMultiplicacion);

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

	//Multiplicacion en la cpu
	// double elapsed, ucpu, scpu;
	// double* matrixC=matrixMemoryAllocation(rowsA,columnsA);
	// ctimer(&elapsed, &ucpu, &scpu);
	// matrixBlasMultiplication(rowsA, rowsA, rowsA, matrixA, matrixA, matrixC);
	// ctimer(&elapsed, &ucpu, &scpu);
	// if(printMatrixBool)
	// {
	// 	printMatrix(rowsA,columnsA,matrixC);
	// }
	// printf("El tiempo de multiplicacion de la matriz en la cpu ha sido de : %f\n", elapsed*1000);
	// printf("El tiempo de creación de comunicadores ha sido: %f\n", elapsedComm*1000);
	// printf("Todos los tiempos han sido medidos en milisegundos y el Bandwidth en GB/s\n");

	// //Comparacion de las matrices
	// if(checkEqualityOfMatrices(recoveredCs[0],matrixC,rowsA,rowsA))
	// {
	// 	printf("Las matriz obtenida por gpus y cpus son iguales.\n");
	// }else
	// {
	// 	printf("Las matriz obtenida por gpus y cpus no son iguales.\n");
	// }
	std::cout << "Fin del programa" << std::endl;
	return 0;
}