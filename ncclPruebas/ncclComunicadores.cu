#include <stdio.h>
#include <fstream>
#include <random>
#include <unistd.h>
#include <vector>
#include <iomanip>
#include <algorithm>
#include "cuda_runtime.h"
#include "nccl.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

using namespace std;

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
            matrixIndex = matrixCalculateIndex(columns, i, j);
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
			matrixIndex = matrixCalculateIndex(columns, i, j);
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
	//Configuracion del comunicador que tiene a todos los dispositivos
	int nDevicesGlobal;
	max(cudaGetDeviceCount(&nDevicesGlobal), 4);
	GpuProperties *gpusInfo = (GpuProperties *)malloc(sizeof(GpuProperties) * nDevicesGlobal);

	int devicesGlobal[nDevicesGlobal];
	for (int i = 0; i < nDevicesGlobal; i++)
	{
		gpusInfo[i] = GpuProperties(nDevicesGlobal, i);
		gpusInfo[i].stream = (cudaStream_t *)malloc(sizeof(cudaStream_t));
		devicesGlobal[i] = i;
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